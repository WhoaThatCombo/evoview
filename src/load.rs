//! Load a car straight out of its game folder.
//!
//! This is the Rust port of tools/export_car.py. The Python remains the place
//! the formats were worked out and is still the reference, but the viewer no
//! longer needs a pre-built .evocar per car: point it at
//! `content/cars/<car_id>/` and it reads meshes, textures, materials, skins and
//! animations directly.
//!
//! FORMAT NOTES that cost real time to find, kept next to the code that relies
//! on them:
//!
//!  * A `.texturemips` is 64 KB pages. ONE TILE IS ONE FULL PAGE, padded in
//!    width AND height, so a mip's page count comes from the tile grid and
//!    never from its byte size. Getting that wrong corrupts only the small
//!    mips, which looks like "artefacts at certain view distances".
//!  * Format enum 26 is BC6H, not BC7. Both are 16 bytes per block so every
//!    size calculation agrees either way; it shows up only as shading noise.
//!  * Wheels are authored at the ORIGIN and placed by bone WHEEL_xx.
//!  * Body colour is a splat map with Red/Green/Blue layers, not a base colour.

use crate::asset::{Batch, Car, Level, MaterialEntry, MeshEntry, TextureEntry};
use crate::proto::{fields, matrix, text, varints, Val};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const PAGE: usize = 65536;

/// format enum -> (gpu format, bits per pixel, tile w, tile h)
fn format_of(e: u64) -> Option<(&'static str, u32, u32, u32)> {
    Some(match e {
        2 | 3 | 5 | 6 | 10 => ("rgba8", 32, 128, 128),
        9 => ("rgba16f", 64, 128, 64),
        12 | 13 => ("bc1", 4, 512, 256),
        // ⚠ 27 is BC4 (one channel), not BC1: the same 8 bytes per block, so
        // it decoded "fine" as BC1 into purple noise. 493 textures across the
        // cars' shared parts alone - every _m/_r/_s/_at/_cc data map of the
        // brake discs, calipers, logos and stitching - all single-channel.
        27 => ("bc4", 4, 512, 256),
        26 => ("bc6h", 8, 256, 256),
        33 | 34 | 35 => ("bc7", 8, 256, 256),
        // BC3 (DXT5): the packed particle flipbooks (smoke_3_10x9_packed:
        // RGB shading, A density). Same 16-byte blocks / tiles as BC7.
        17 => ("bc3", 8, 256, 256),
        // BC3 too (DXT5: BC4-style alpha + BC1 colour) - the packed tarmac
        // variation maps "_ar_packed" (rgb a light variation, a a streak
        // map) and the dirt maps. Found by --tex-audit: unknown before, so
        // those roads lost their variation. ⚠ Not BC5: read as BC5 the
        // second half (the BC1 colour block) was white noise even at mip 3,
        // and the first (the alpha streaks, mean 0.24) blackened the road.
        16 => ("bc3", 8, 256, 256),
        // RGBA16 UNORM (the cars' display images, a carbon-fibre normal, the
        // showroom gradients): RGBA16F's 8 bytes and 128x64 tiles. Kept as
        // u16 here; material.rs turns it into half floats at upload,
        // sRGB-decoded when the slot is colour (no sRGB 16-bit format)
        25 => ("rgba16u", 64, 128, 64),
        _ => return None,
    })
}

fn row_bytes(kind: &str, width: u32, bpp: u32) -> usize {
    match kind {
        "bc1" | "bc4" => (((width + 3) / 4).max(1) * 8) as usize,
        "bc7" | "bc6h" | "bc3" => (((width + 3) / 4).max(1) * 16) as usize,
        _ => (width * bpp / 8) as usize,
    }
}

// ------------------------------------------------------------------ skeleton --

#[derive(Clone)]
pub struct Bone {
    pub name: String,
    pub parent: Option<usize>,
    /// rest pose, parent-relative, row-major
    pub local: [f32; 16],
}

/// One animated bone: a full transform per frame at the take's frame rate.
#[derive(Clone)]
pub struct Track {
    pub bone: String,
    pub frames: Vec<[f32; 16]>,
}

#[derive(Clone)]
pub struct Animation {
    pub name: String,
    pub fps: f32,
    pub tracks: Vec<Track>,
    pub frames: usize,
}

pub(crate) fn mat_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    // Row-major, row-vector convention: v * A * B, matching how the engine
    // writes translation into elements 13/14/15 (the last ROW).
    let mut m = [0.0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[r * 4 + k] * b[k * 4 + c];
            }
            m[r * 4 + c] = s;
        }
    }
    m
}

pub(crate) const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// Invert an affine row-major matrix (rotation/scale + translation only).
pub(crate) fn affine_inverse(m: &[f32; 16]) -> [f32; 16] {
    let a = [
        [m[0], m[1], m[2]],
        [m[4], m[5], m[6]],
        [m[8], m[9], m[10]],
    ];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-12 {
        return IDENTITY;
    }
    let id = 1.0 / det;
    let inv = [
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * id,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * id,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * id,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * id,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * id,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * id,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * id,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * id,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * id,
        ],
    ];
    let t = [m[12], m[13], m[14]];
    let mut out = IDENTITY;
    for r in 0..3 {
        for c in 0..3 {
            out[r * 4 + c] = inv[r][c];
        }
    }
    for c in 0..3 {
        out[12 + c] = -(t[0] * inv[0][c] + t[1] * inv[1][c] + t[2] * inv[2][c]);
    }
    out
}

pub struct Skeleton {
    pub bones: Vec<Bone>,
    pub index: HashMap<String, usize>,
    /// world-space rest pose per bone
    pub bind: Vec<[f32; 16]>,
    pub inv_bind: Vec<[f32; 16]>,
    /// What drives the bones at runtime (carrig.rs): animation bindings from
    /// the actor, suspension, steering, needles. None for tracks / .evocar.
    pub rig: Option<Box<crate::carrig::Rig>>,
}

impl Skeleton {
    fn build(bones: Vec<Bone>) -> Self {
        let index: HashMap<String, usize> =
            bones.iter().enumerate().map(|(i, b)| (b.name.clone(), i)).collect();
        let mut bind = vec![IDENTITY; bones.len()];
        for i in 0..bones.len() {
            // parents always precede children in the table; walk up anyway so
            // an out-of-order table cannot silently produce a wrong pose
            let mut m = bones[i].local;
            let mut p = bones[i].parent;
            let mut guard = 0;
            while let Some(pi) = p {
                m = mat_mul(&m, &bones[pi].local);
                p = bones[pi].parent;
                guard += 1;
                if guard > 256 {
                    break;
                }
            }
            bind[i] = m;
        }
        let inv_bind = bind.iter().map(affine_inverse).collect();
        Skeleton { bones, index, bind, inv_bind, rig: None }
    }

    /// Skinning matrices for one animation at one time.
    pub fn pose(&self, anim: Option<&Animation>, time: f32) -> Vec<[f32; 16]> {
        match anim {
            Some(a) => self.pose_many(&[(a, time)]),
            None => self.pose_many(&[]),
        }
    }

    /// Skinning matrices for SEVERAL animations at once, as
    /// `anim_world * inverse(bind_world)`.
    ///
    /// Each take drives its own bones - a door, the hood, one wiper - so they
    /// compose simply by applying each in turn. That is what lets both doors
    /// and the hood be open at the same time instead of one animation at a
    /// time. Bones nothing touches keep their rest transform, which is what
    /// leaves the rest of the car still.
    pub fn pose_many(&self, playing: &[(&Animation, f32)]) -> Vec<[f32; 16]> {
        let mut local: Vec<[f32; 16]> = self.bones.iter().map(|b| b.local).collect();
        for (a, time) in playing {
            if a.frames == 0 {
                continue;
            }
            let f = ((time * a.fps).round().max(0.0) as usize).min(a.frames - 1);
            for t in &a.tracks {
                if t.frames.is_empty() {
                    continue;
                }
                if let Some(&bi) = self.index.get(&t.bone) {
                    // ⚠ Clamp per TRACK, not per animation: a merged ext/int
                    // pair can hold takes of different lengths, and running off
                    // the end of the shorter one would freeze that half of the
                    // door part-way instead of holding it open.
                    local[bi] = t.frames[f.min(t.frames.len() - 1)];
                }
            }
        }
        let mut world = vec![IDENTITY; self.bones.len()];
        for i in 0..self.bones.len() {
            // parents come first in the table: one multiply on the parent's
            // world (walking the whole chain per bone was ~1.5 ms a frame
            // with the trackside people)
            if let Some(pi) = self.bones[i].parent.filter(|&pi| pi < i) {
                world[i] = mat_mul(&local[i], &world[pi]);
                continue;
            }
            let mut m = local[i];
            let mut p = self.bones[i].parent;
            let mut guard = 0;
            while let Some(pi) = p {
                m = mat_mul(&m, &local[pi]);
                p = self.bones[pi].parent;
                guard += 1;
                if guard > 256 {
                    break;
                }
            }
            world[i] = m;
        }
        world
            .iter()
            .zip(&self.inv_bind)
            .map(|(w, ib)| mat_mul(ib, w))
            .collect()
    }
}

// ---------------------------------------------------------------------- mesh --

pub struct Lod {
    pub positions: Vec<u8>,
    pub normals: Vec<u8>,
    pub uvs: Vec<u8>,
    pub tangents: Vec<u8>,
    /// 4 bone indices per vertex, stored by the engine as f32
    pub bone_idx: Vec<u8>,
    /// 4 bone weights per vertex
    pub bone_wgt: Vec<u8>,
    pub indices: Vec<u32>,
    pub batches: Vec<Batch>,
    pub bones: Vec<Bone>,
    /// MeshLodData.cameraFacingMode
    pub facing: u32,
    /// f13: raw vertex colour, 4 f32 per vertex (colour / 255). Tracks use it
    /// for splat weights and AO.
    pub colors: Vec<u8>,
    /// f14: second UV set, 2 f32 per vertex
    pub uv1: Vec<u8>,
    /// f25: fraction of instances this LOD keeps (grass thins with distance)
    pub density: f32,
    /// f23: ditheredLodTransition, metres either side of this LOD's switch-in
    /// distance over which it crossfades with the previous one (unscaled)
    pub dlt: f32,
    /// f24: the clip a skinned prop plays in a loop (spectators, flag
    /// marshals, the sky dancer) - an .animation over this LOD's own bones
    pub anim: String,
    /// f1 castShadows (absent = false) and f22 shadowLod
    pub cast_shadows: bool,
    pub shadow_lod: i32,
    /// f10022 isShadowProxy: a level drawn only into shadow maps
    pub shadow_proxy: bool,
}

/// One instanced track prop (or one LOD level of it) as a MeshEntry.
///
/// ⚠ LOD0 and every coarser level go through here, so any per-vertex stream
/// added to track meshes (vertex colour, a second UV set) must be added HERE
/// once, from `l` - each level has its own vertex count.
/// Append `data` to the blob, 4-byte aligned; returns its [offset, length].
fn blob_add(blob: &mut Vec<u8>, data: &[u8]) -> [u64; 2] {
    let off = blob.len() as u64;
    blob.extend_from_slice(data);
    while blob.len() % 4 != 0 {
        blob.push(0);
    }
    [off, data.len() as u64]
}

/// An instanced track LOD the viewer animates: bones, a full skin and a clip.
fn skinned(l: &Lod) -> bool {
    let n = l.positions.len() / 12;
    !l.anim.is_empty() && !l.bones.is_empty() && l.bone_idx.len() == n * 16 && l.bone_wgt.len() == n * 16
        && std::env::var_os("EVOVIEW_NO_TRACK_ANIM").is_none()
}

/// EVOVIEW_PLACE_DUMP=<file>: one line per placed track mesh copy - its
/// name, the world bounding box (min x y z, max x y z) and the copy's
/// position - for the offline placement audit (tools/place_audit.py checks
/// each against the collision ground from EVOVIEW_GROUND_DUMP).
fn place_dump(name: &str, positions: &[u8], xforms: &[[f32; 16]]) {
    use std::io::Write;
    static OUT: std::sync::Mutex<Option<std::fs::File>> = std::sync::Mutex::new(None);
    let Ok(path) = std::env::var("EVOVIEW_PLACE_DUMP") else { return };
    let mut g = OUT.lock().unwrap();
    if g.is_none() {
        *g = std::fs::File::create(&path).ok();
    }
    let Some(f) = g.as_mut() else { return };
    let ps: &[f32] = bytemuck::cast_slice(positions);
    let (mut lo, mut hi) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
    for v in ps.chunks_exact(3) {
        let v = glam::Vec3::new(v[0], v[1], v[2]);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if lo.x > hi.x {
        return;
    }
    let id = [glam::Mat4::IDENTITY.to_cols_array()];
    for m in if xforms.is_empty() { &id[..] } else { xforms } {
        let m = glam::Mat4::from_cols_array(m);
        let (mut a, mut b) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
        for k in 0..8 {
            let c = glam::Vec3::new(if k & 1 == 0 { lo.x } else { hi.x }, if k & 2 == 0 { lo.y } else { hi.y }, if k & 4 == 0 { lo.z } else { hi.z });
            let w = m.transform_point3(c);
            a = a.min(w);
            b = b.max(w);
        }
        let t = m.w_axis;
        let _ = writeln!(f, "{name}	{:.2}	{:.2}	{:.2}	{:.2}	{:.2}	{:.2}	{:.2}	{:.2}	{:.2}	{}", a.x, a.y, a.z, b.x, b.y, b.z, t.x, t.y, t.z, xforms.len());
    }
}

fn instanced_entry(
    blob: &mut Vec<u8>,
    name: String,
    l: &Lod,
    xforms: Vec<[f32; 16]>,
    lod: Option<[f32; 2]>,
) -> MeshEntry {
    MeshEntry {
        name,
        vertices: (l.positions.len() / 12) as u32,
        batches: l.batches.clone(),
        positions: Some(blob_add(blob, &l.positions)),
        normals: Some(blob_add(blob, &l.normals)),
        uvs: Some(blob_add(blob, &l.uvs)),
        tangents: Some(blob_add(blob, &l.tangents)),
        // only an animated prop keeps its skin (track_anim_skin); the rest
        // draw rigid as before
        bone_idx: skinned(l).then(|| blob_add(blob, &l.bone_idx)),
        bone_wgt: skinned(l).then(|| blob_add(blob, &l.bone_wgt)),
        indices: Some(blob_add(blob, bytemuck::cast_slice(&l.indices))),
        xforms,
        facing: l.facing,
        lod,
        lod_fade: [0.0; 4],
        shadow_mesh: None,
        // the track pipeline's second vertex stream, for every LOD level
        colors: Some(blob_add(blob, &track_vertex_stream(&l.colors, (l.positions.len() / 12) as usize, 4, 1.0 / 255.0))),
        uv1: Some(blob_add(blob, &track_vertex_stream(&l.uv1, (l.positions.len() / 12) as usize, 2, 0.0))),
    }
}

/// Metres per unit of a MeshLodData switch distance, baked at load: 1.
///
/// Field 3 is plain metres: grass tufts switch at 5 / 20 / 40 and the beech_7
/// tree at 35 / 75 / 350, where 350 is its 36-vertex billboard. The game's
/// lodDistanceScale (preset x field of view) is applied per frame by the
/// culler (cull.rs lod_scale), so bands and lodOut are stored raw here.
pub fn lod_dist_scale() -> f32 {
    1.0
}

/// EVOVIEW_NO_LOD=1 keeps only LOD0 for instanced props.
pub fn lod_enabled() -> bool {
    std::env::var_os("EVOVIEW_NO_LOD").is_none()
}

pub fn read_mesh(d: &[u8]) -> Option<Lod> {
    crate::perf::time("mesh decode", || read_mesh_lods_inner(d, 1).pop().map(|l| l.1))
}

/// MeshData f4 lodOut and f11 ditheredLodOut: past lodDistanceScale x
/// lodOut the whole mesh fades out over ditheredLodOut metres (033aa890).
/// 0 = never.
pub fn read_mesh_out(d: &[u8]) -> (f32, f32) {
    let (mut out, mut dlo) = (0.0f32, 0.0f32);
    fields(d, 0, d.len(), |fnum, v| match (fnum, v) {
        (4, Val::F32(x)) => out = x.max(0.0),
        (11, Val::F32(x)) => dlo = x.max(0.0),
        _ => {}
    });
    (out, dlo)
}

/// MeshData f10 forceShadowLod
pub fn read_force_shadow_lod(d: &[u8]) -> bool {
    let mut f = false;
    fields(d, 0, d.len(), |fnum, v| if let (10, Val::Varint(n)) = (fnum, v) { f = n != 0 });
    f
}

/// EVOVIEW_LOD_FADE=0 switches LODs with a hard pop (no crossfade, no lodOut).
pub fn lod_fade_enabled() -> bool {
    std::env::var("EVOVIEW_LOD_FADE").map_or(true, |v| v != "0")
}

/// Smallest crossfade, as a fraction of the switch distance. Only the
/// vegetation sets ditheredLodTransition (Imola: 26 of 300 switches, the
/// trees and bushes); every other prop - marshals, crowds, tents, fences -
/// and every lodOut but the trees' has 0 and pops in the game too.
/// 0 (the default) is the game's behaviour. 0.1 crossfades those too but
/// costs 0.55 ms GPU at 3440x1440 on Imola (12k-vertex trees drawn twice
/// through a band at 105 m), against 0.1 ms for the game's own widths.
pub fn lod_dither_min() -> f32 {
    std::env::var("EVOVIEW_LOD_DITHER_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0)
}

/// Every LOD in the file, finest first, each with its switch distance
/// (MeshLodData field 3; absent = 0 for LOD0). Units are not confirmed:
/// values run 0.7..30 against a top-level draw distance of 100..750, so the
/// caller scales them (see lod_dist_scale).
pub fn read_mesh_lods(d: &[u8]) -> Vec<(f32, Lod)> {
    crate::perf::time("mesh decode", || read_mesh_lods_inner(d, usize::MAX))
}

fn read_mesh_lods_inner(d: &[u8], max: usize) -> Vec<(f32, Lod)> {
    let mut out: Vec<(f32, Lod)> = Vec::new();
    fields(d, 0, d.len(), |fnum, v| {
        if fnum != 5 || out.len() >= max {
            return;
        }
        let (s, e) = match v.range() {
            Some(r) => r,
            None => return,
        };
        let mut lod = Lod {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            tangents: Vec::new(),
            bone_idx: Vec::new(),
            bone_wgt: Vec::new(),
            indices: Vec::new(),
            batches: Vec::new(),
            bones: Vec::new(),
            facing: 0,
            colors: Vec::new(),
            uv1: Vec::new(),
            density: 1.0,
            dlt: 0.0,
            anim: String::new(),
            cast_shadows: false,
            shadow_lod: 0,
            shadow_proxy: false,
        };
        let mut parents: Vec<(usize, String)> = Vec::new();
        let mut dist = 0.0f32;
        fields(d, s, e, |lf, lv| {
            if lf == 3 {
                if let Val::F32(x) = lv {
                    dist = x;
                }
                return;
            }
            if lf == 23 {
                if let Val::F32(x) = lv {
                    lod.dlt = x.max(0.0);
                }
                return;
            }
            if lf == 1 || lf == 22 || lf == 10022 {
                if let Val::Varint(n) = lv {
                    match lf { 1 => lod.cast_shadows = n != 0, 22 => lod.shadow_lod = n as i32, _ => lod.shadow_proxy = n != 0 }
                }
                return;
            }
            if lf == 25 {
                if let Val::F32(x) = lv {
                    lod.density = x.clamp(0.0, 1.0);
                }
                return;
            }
            if lf == 20 {
                if let Val::Varint(n) = lv {
                    lod.facing = n as u32;
                }
                return;
            }
            let (a, b) = match lv.range() {
                Some(r) => r,
                None => return,
            };
            match lf {
                5 => lod.positions = d[a..b].to_vec(),
                6 => lod.normals = d[a..b].to_vec(),
                7 => lod.uvs = d[a..b].to_vec(),
                8 => lod.tangents = d[a..b].to_vec(),
                9 => lod.bone_wgt = d[a..b].to_vec(),
                10 => lod.bone_idx = d[a..b].to_vec(),
                11 => lod.indices = varints(&d, a, b),
                13 => lod.colors = d[a..b].to_vec(),
                14 => lod.uv1 = d[a..b].to_vec(),
                24 => lod.anim = text(&d, (a, b)),
                4 => {
                    let mut bt = Batch {
                        name: String::new(),
                        material: String::new(),
                        start: 0,
                        count: 0,
                    };
                    fields(&d, a, b, |bf, bv| match (bf, bv) {
                        (1, Val::Bytes(x, y)) => bt.name = text(&d, (x, y)),
                        (4, Val::Bytes(x, y)) => bt.material = text(&d, (x, y)),
                        (2, Val::Varint(n)) => bt.start = n as u32,
                        (3, Val::Varint(n)) => bt.count = n as u32,
                        _ => {}
                    });
                    if bt.count > 0 {
                        lod.batches.push(bt);
                    }
                }
                12 => {
                    let mut name = String::new();
                    let mut parent = String::new();
                    let mut local = IDENTITY;
                    fields(&d, a, b, |bf, bv| {
                        if let Some((x, y)) = bv.range() {
                            match bf {
                                1 => name = text(&d, (x, y)),
                                2 => parent = text(&d, (x, y)),
                                3 => local = matrix(&d, x, y),
                                _ => {}
                            }
                        }
                    });
                    if !name.is_empty() {
                        parents.push((lod.bones.len(), parent));
                        lod.bones.push(Bone { name, parent: None, local });
                    }
                }
                _ => {}
            }
        });
        // resolve parents by name once every bone is known
        let by_name: HashMap<&str, usize> = lod
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| (b.name.as_str(), i))
            .collect();
        let resolved: Vec<Option<usize>> = parents
            .iter()
            .map(|(_, p)| by_name.get(p.as_str()).copied())
            .collect();
        for (i, p) in resolved.into_iter().enumerate() {
            lod.bones[i].parent = p;
        }
        out.push((dist, lod));
    });
    out
}

// ------------------------------------------------------------------- texture --

#[derive(Clone)]
pub(crate) struct Tex {
    pub(crate) format: String,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) levels: Vec<Level>,
    pub(crate) data: Vec<u8>,
}

fn read_texture(hdr: &[u8], raw: &[u8]) -> Option<Tex> {
    read_texture_lod(hdr, raw, 0)
}

pub(crate) fn read_texture_pub(hdr: &[u8], raw: &[u8]) -> Option<Tex> {
    read_texture(hdr, raw)
}

/// Extra top mips dropped from a CAR's textures while it loads (replayscene
/// sets it for the opponents: dozens of liveries at full size filled a 12 GB
/// card, and every pass slowed ~10x once it paged).
pub static CAR_TEX_SKIP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Sentinel for `read_texture_lod`: pick the mip skip from the texture's size.
pub const AUTO_MIP: u32 = u32::MAX;

/// How many top mips to drop from a TRACK texture.
///
/// ⚠ This was a flat 1 for every track texture, which is why track renders
/// looked soft next to car renders: every map, from a 256px sign to a 4K
/// asphalt sheet, was thrown away at mip 0 and rendered at half resolution.
/// The reason was real - the Ring ships hundreds of 4K maps and keeping every
/// texel OOMs it - but the cost was being paid by every texture on every
/// track, including the ones that were never the problem.
pub fn track_mip_skip(w: u32, h: u32) -> u32 {
    if let Ok(v) = std::env::var("EVOVIEW_TEX_LOD") {
        if let Ok(n) = v.parse() {
            return n;
        }
    }
    // Only the genuinely huge maps can give up their top level, and only
    // past a budget: the first EVOVIEW_TEX_4K_BUDGET (64, ~1 GB) keep it.
    // ⚠ Every 4K map used to drop it, on every track, to keep the Ring
    // (hundreds of them) in memory - so Monza's asphalt and grass sheets,
    // ~a dozen maps and ~190 MB, rendered at half resolution for nothing.
    let big = w.max(h);
    if big >= 4096 {
        let budget = std::env::var("EVOVIEW_TEX_4K_BUDGET").ok().and_then(|v| v.parse().ok()).unwrap_or(64u32);
        let n = TRACK_4K_SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < budget { 0 } else { 1 }
    } else {
        0
    }
}

/// 4K track maps decided so far for the track being loaded (track_mip_skip).
pub static TRACK_4K_SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// `skip_mips` drops the largest levels (1 = start at half-res).
fn read_texture_lod(hdr: &[u8], raw: &[u8], skip_mips: u32) -> Option<Tex> {
    crate::perf::time("texture decode", || read_texture_lod_inner(hdr, raw, skip_mips))
}

fn read_texture_lod_inner(hdr: &[u8], raw: &[u8], skip_mips: u32) -> Option<Tex> {
    let (mut w, mut h, mut mips, mut fmt) = (0u64, 0u64, 1u64, 0u64);
    fields(hdr, 0, hdr.len(), |f, v| {
        if let Val::Varint(n) = v {
            match f {
                1 => w = n,
                2 => h = n,
                3 => mips = n,
                4 => fmt = n,
                _ => {}
            }
        }
    });
    let (kind, bpp, tw, th) = format_of(fmt)?;
    let block: u32 = if matches!(kind, "bc1" | "bc4" | "bc7" | "bc6h" | "bc3") { 4 } else { 1 };

    // `AUTO_MIP` means "decide from this texture's own size" - the caller does
    // not know it yet, because the size lives in the header parsed just above.
    let want = if skip_mips == AUTO_MIP {
        track_mip_skip(w as u32, h as u32)
    } else {
        skip_mips
    };
    let skip = if w >= 8 && h >= 8 && matches!(kind, "bc1" | "bc4" | "bc7" | "bc6h" | "bc3") {
        want.min(mips.saturating_sub(1) as u32)
    } else {
        0
    };
    // ⚠ A payload holding only the tail of the chain (Sebring's
    // tower_roughness: 2048^2 declared, 11 pages stored - the levels from
    // 1024 down): read from the start, the top level never fit and the
    // texture came out empty. Where the stored pages are exactly the chain
    // from some level down, start there.
    let pages_of = |m: u32| -> usize {
        let mw = ((w as u32) >> m).max(1);
        let mh = ((h as u32) >> m).max(1);
        mw.div_ceil(tw).max(1) as usize * mh.div_ceil(th).max(1) as usize
    };
    let mut first = 0u32;
    {
        let mut tail: usize = (0..mips as u32).map(pages_of).sum::<usize>() * PAGE;
        if raw.len() < tail {
            let mut m = 0u32;
            while m + 1 < mips as u32 && tail > raw.len() {
                tail -= pages_of(m) * PAGE;
                m += 1;
            }
            if tail == raw.len() {
                first = m;
            }
        }
    }
    let skip = skip.max(first);
    let mut out: Vec<u8> = Vec::new();
    let mut levels = Vec::new();
    let mut off = 0usize;
    for m in first..mips as u32 {
        let mw = ((w as u32) >> m).max(1);
        let mh = ((h as u32) >> m).max(1);
        // ONE TILE IS ONE FULL PAGE, padded both ways: page count comes from
        // the tile grid, never from the mip's byte size.
        let cols = mw.div_ceil(tw).max(1) as usize;
        let rows = mh.div_ceil(th).max(1) as usize;
        let pages = cols * rows;
        if off + pages * PAGE > raw.len() {
            break;
        }
        if m < skip {
            off += pages * PAGE;
            continue;
        }
        let rb = row_bytes(kind, mw, bpp);
        let need = rb * (mh.div_ceil(block).max(1)) as usize;
        let trb = row_bytes(kind, tw, bpp);
        let mut level = vec![0u8; need];
        for r in 0..rows {
            for c in 0..cols {
                let page = off + (r * cols + c) * PAGE;
                let keep_w = row_bytes(kind, tw.min(mw - c as u32 * tw), bpp);
                let keep_h =
                    (th.min(mh - r as u32 * th)).div_ceil(block).max(1) as usize;
                for y in 0..keep_h {
                    let src = page + y * trb;
                    let dst = (r * (th / block) as usize + y) * rb + c * trb;
                    if dst + keep_w > need || src + keep_w > raw.len() {
                        break;
                    }
                    level[dst..dst + keep_w].copy_from_slice(&raw[src..src + keep_w]);
                }
            }
        }
        out.extend_from_slice(&level);
        levels.push(Level { width: mw, height: mh, bytes: need as u64 });
        off += pages * PAGE;
    }
    // ⚠ Formats 5 and 6 are B8G8R8A8 (2 / 3 are R8G8B8A8): read as RGBA the
    // Lamborghini badge is blue, Imola's "EVO" banners blue and the
    // building / garage splat masks swap their red and blue layers.
    // EVOVIEW_BGRA=0: as RGBA, as before.
    if matches!(fmt, 5 | 6) && std::env::var("EVOVIEW_BGRA").map_or(true, |v| v != "0") {
        for p in out.chunks_exact_mut(4) {
            p.swap(0, 2);
        }
    }

    Some(Tex {
        format: kind.to_string(),
        width: ((w as u32) >> skip).max(1),
        height: ((h as u32) >> skip).max(1),
        levels,
        data: out,
    })
}

/// A value in 0..1 as IEEE half-float bits (rounded to nearest).
pub(crate) fn unorm_to_half(v: f32) -> u16 {
    let v = v.clamp(0.0, 1.0);
    if v < 6.103_515_6e-5 {
        // subnormal: v / 2^-24
        return (v * 16_777_216.0).round() as u16;
    }
    let b = v.to_bits();
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    let h = ((e as u32) << 10) | (m >> 13);
    // round to nearest on the dropped bits (a carry into the exponent is right)
    (h + ((m >> 12) & 1)) as u16
}

// ------------------------------------------------------------------ cubemap --

/// A decoded cubemap: six faces, each with the same mip chain.
pub struct Cube {
    pub format: String,
    pub size: u32,
    /// per-mip (edge length, bytes per face)
    pub levels: Vec<(u32, usize)>,
    /// face-major: all mips of face 0, then face 1, ...
    pub data: Vec<u8>,
}

/// The game's own environment cubemaps.
///
/// ⚠ Two things about these are NOT what the rest of this file assumes.
///
/// First, the face count lives in header field 6 (6 for a cube); everything
/// else here reads 2D textures and ignores it.
///
/// Second - and this is the one that produced garbage for a while - a cube
/// face's payload is PLAIN LINEAR row-major blocks, paged at 64 KB. It is not
/// tiled. `read_texture_lod` reassembles 256x256 tiles, and doing that to a
/// 1024-wide face interleaves the rows: the image comes out in horizontal
/// stripes with the content still faintly visible through it. Read straight
/// through instead and mip 0 is exactly 16 pages (1048576 B) of a 1024^2 BC7
/// image, which decodes to what it should be - a black room with two angled
/// softbox strips.
///
/// ⚠ Do NOT generalise that to ordinary textures. A 2048^2 car AO map decoded
/// linearly scrambles into a visible tile grid, so 2D textures really are
/// tiled. Both layouts exist; this function is only for the cube path.
///
/// Header field 12 carries the page table - sub4 per-level page offsets, sub5
/// per-level page counts (16,4,1,1,... for 11 mips, repeated per face) - so
/// levels are indexed rather than computed. Small mips share a page.
pub fn read_cubemap(hdr: &[u8], raw: &[u8]) -> Option<Cube> {
    let (mut w, mut h, mut mips, mut fmt, mut faces) = (0u64, 0u64, 1u64, 0u64, 1u64);
    fields(hdr, 0, hdr.len(), |f, v| match (f, v) {
        (1, Val::Varint(n)) => w = n,
        (2, Val::Varint(n)) => h = n,
        (3, Val::Varint(n)) => mips = n,
        (4, Val::Varint(n)) => fmt = n,
        (6, Val::Varint(n)) => faces = n,
        _ => {}
    });
    if faces != 6 || w == 0 || w != h {
        return None;
    }
    let (kind, bpp, _tw, _th) = format_of(fmt)?;
    let block: u32 = if matches!(kind, "bc1" | "bc4" | "bc7" | "bc6h" | "bc3") { 4 } else { 1 };

    // Level sizes, packed with no alignment between them.
    //
    // ⚠ Header field 12 lists a page offset and page COUNT per level, and
    // taking those literally does not work: it claims 29 pages a face (174 in
    // total, 11.4 MB) for a file that is 8.39 MB. The counts are an
    // allocation hint - every mip below 256 px claims a whole 64 KB page and
    // does not fill it. What is actually stored is each face's chain packed
    // end to end. For 1024^2 BC7 that is
    //   1048576 + 262144 + 65536 + 16384 + 4096 + 1024 + 256 + 64 + 16+16+16
    //   = 1398128 bytes, and 8388768 / 6 = 1398128 exactly.
    let mut levels = Vec::new();
    for m in 0..mips as u32 {
        let mw = ((w as u32) >> m).max(1);
        let rb = row_bytes(kind, mw, bpp);
        let rows = mw.div_ceil(block).max(1) as usize;
        levels.push((mw, rb * rows));
    }
    let face_stride: usize = levels.iter().map(|(_, n)| *n).sum();
    if face_stride == 0 || raw.len() < face_stride * 6 {
        return None;
    }

    let mut out: Vec<u8> = Vec::with_capacity(face_stride * 6);
    for face in 0..6usize {
        let base = face * face_stride;
        out.extend_from_slice(&raw[base..base + face_stride]);
    }
    Some(Cube {
        format: kind.to_string(),
        size: w as u32,
        levels,
        data: out,
    })
}

/// The game's own split-sum BRDF lookup table.
///
/// `content\ibl\specularbrdf` - 256x256, one mip, format 8, and 262144 bytes
/// is exactly 256*256*4, so it is RG16F in plain rows. The shader has an
/// analytic fit of this curve (the usual Karis polynomial); the real table is
/// the same thing without the approximation, and it costs one texture.
pub fn brdf_lut(vfs: &mut Vfs) -> Option<(u32, Vec<u8>)> {
    let hdr = vfs.read("content/ibl/specularbrdf.texture");
    if hdr.is_empty() {
        return None;
    }
    let (mut w, mut h) = (0u64, 0u64);
    fields(&hdr, 0, hdr.len(), |f, v| {
        if let Val::Varint(n) = v {
            match f {
                1 => w = n,
                2 => h = n,
                _ => {}
            }
        }
    });
    if w == 0 || w != h {
        return None;
    }
    let raw = vfs.read("content/ibl/specularbrdf.texturemips");
    let need = (w * h * 4) as usize;
    if raw.len() < need {
        return None;
    }
    Some((w as u32, raw[..need].to_vec()))
}

/// Load a cubemap by virtual path stem (no extension).
pub fn load_cubemap(vfs: &mut Vfs, stem: &str) -> Option<Cube> {
    let hdr = vfs.read(&format!("{stem}.texture"));
    if hdr.is_empty() {
        return None;
    }
    let raw = vfs.read(&format!("{stem}.texturemips"));
    if raw.is_empty() {
        return None;
    }
    read_cubemap(&hdr, &raw)
}

/// The studio the game lights showroom cars in.
///
/// Tried in the order the game ships them: the full-resolution scene first,
/// then the low one. Absent (a car folder with no package underneath) is a
/// normal state, not an error - the shader has an analytic studio for it.
pub fn studio_env(vfs: &mut Vfs) -> Option<Cube> {
    // `--noenv` falls back to the analytic studio, which is what makes the
    // two comparable in a screenshot without rebuilding.
    if std::env::args().any(|a| a == "--noenv") {
        return None;
    }
    const STEMS: [&str; 3] = [
        "content/tracks/common_assets/textures/cubemaps/studio/studio_1_sdr",
        "content/tracks/common_assets/textures/cubemaps/studio/studio_2_sdr",
        "content/tracks/common_assets/textures/cubemaps/studio/studio_1_low_sdr",
    ];
    for stem in STEMS {
        if let Some(c) = load_cubemap(vfs, stem) {
            println!("env: {} ({} px, {} mips)", stem, c.size, c.levels.len());
            return Some(c);
        }
    }
    None
}

// ------------------------------------------------------------------ material --

fn strings_under(d: &[u8], s: usize, e: usize, depth: u32, out: &mut Vec<String>) {
    fields(d, s, e, |_f, v| {
        if let Some((a, b)) = v.range() {
            let t = &d[a..b];
            if !t.is_empty() && t.iter().all(|c| (0x20..0x7f).contains(c)) {
                out.push(String::from_utf8_lossy(t).into_owned());
            } else if depth < 4 {
                strings_under(d, a, b, depth + 1, out);
            }
        }
    });
}

/// A material parameter value: `{ f1 = scalar | f2/f3/f4 = vec2/3/4 }`, each
/// vector a message with its components in fields 1..N.
///
/// ⚠ proto3 OMITS zero fields. Collecting "every float" turned (0, .169, .456)
/// into the vec2 [.169, .456] (which then fell back to grey) and a (1, 0, 0)
/// light colour into the scalar 1. Decode by field number, missing = 0.
fn param_value(d: &[u8], s: usize, e: usize) -> Option<Vec<f32>> {
    let mut out: Option<Vec<f32>> = None;
    fields(d, s, e, |f, v| {
        match (f, v) {
            (1, Val::F32(x)) => out = Some(vec![x]),
            (2..=4, Val::Bytes(a, b)) => {
                let mut c = vec![0.0f32; f as usize];
                let mut got = false;
                fields(d, a, b, |cf, cv| {
                    if let Val::F32(x) = cv {
                        if (1..=c.len() as u32).contains(&cf) {
                            c[cf as usize - 1] = x;
                            got = true;
                        }
                    }
                });
                // An EMPTY vector is every unset param of the shader (the
                // material lists them all); treating it as a real (0,0,0,0)
                // made ext_decals' unused `Basecolor` black out the decals.
                if got {
                    // A vec4 whose w was omitted keeps 3 components, so
                    // `Mat::color` still defaults alpha to 1 as it always did
                    // (w = 0 on a Basecolor is "unset", not transparent).
                    let w_set = c.len() == 4 && c[3] != 0.0;
                    if c.len() == 4 && !w_set {
                        c.truncate(3);
                    }
                    out = Some(c);
                }
            }
            _ => {}
        }
    });
    // ⚠ An EMPTY value is NOT taken as scalar 0: texture, bool and enum
    // params all arrive empty, and inventing 0 for them hid the F2004 decals.
    out
}

#[derive(Clone)]
struct Mat {
    file: String,
    shader: Option<String>,
    slots: HashMap<String, String>,
    params: HashMap<String, f32>,
    colors: HashMap<String, Vec<f32>>,
    /// MaterialData f20 baseMaterialPath: the parent this one inherits from
    base: Option<String>,
    /// f21 / f22 overrideVariables / overrideResources: the keys this
    /// material sets over its parent (value true); None = no list
    ov_vars: Option<std::collections::HashSet<String>>,
    ov_res: Option<std::collections::HashSet<String>>,
}

/// Normalise a material reference or archive path to one comparable key:
/// lowercase, forward slashes, no leading separator.
const SEP: char = '\\';

pub fn norm_mat_path(p: &str) -> String {
    p.to_lowercase().replace(SEP, "/").trim_start_matches('/').to_string()
}

fn read_material(d: &[u8], file: &str) -> Option<Mat> {
    let d = d.to_vec();
    let mut m = Mat {
        file: file.to_string(),
        shader: None,
        slots: HashMap::new(),
        params: HashMap::new(),
        colors: HashMap::new(),
        base: None,
        ov_vars: None,
        ov_res: None,
    };
    fields(&d, 0, d.len(), |f, v| {
        // top-level f9 / f10: alphaToCoverageOpacityScale / Cutoff - the
        // engine's cutout, copied into Code_opacityScale / Code_cutoff
        // (SPEC_small §1). No car material has an `a2cCutoff` variable.
        if let (9 | 10, crate::proto::Val::F32(x)) = (f, &v) {
            m.params.insert(if f == 9 { "__a2c_scale" } else { "__a2c_cutoff" }.into(), *x);
            return;
        }
        // f11: the cutout prepass's mip bias on the alpha read (Code_mipClamp;
        // the a2c prepass 0x03350b70 passes it as SampleBias's bias)
        if let (11, crate::proto::Val::F32(x)) = (f, &v) {
            m.params.insert("__a2c_mip".into(), *x);
            return;
        }
        // f6 cullMode: 0 Front (the default: back faces culled), 1 Back,
        // 2 None (two-sided), 3 Wireframe
        if let (6, crate::proto::Val::Varint(n)) = (f, &v) {
            m.params.insert("__cull".into(), *n as f32);
            return;
        }
        // f29 windable: the vegetation the engine sways in the wind
        if let (29, crate::proto::Val::Varint(n)) = (f, &v) {
            m.params.insert("__windable".into(), if *n != 0 { 1.0 } else { 0.0 });
            return;
        }
        let (s, e) = match v.range() {
            Some(r) => r,
            None => return,
        };
        match f {
            1 => m.shader = Some(text(&d, (s, e))),
            20 => {
                let b = text(&d, (s, e));
                if !b.is_empty() {
                    m.base = Some(b);
                }
            }
            // { f1 key, f2 bool }: which variables / resources override the parent
            21 | 22 => {
                let (mut key, mut on) = (String::new(), false);
                fields(&d, s, e, |of, ov| match (of, ov) {
                    (1, crate::proto::Val::Bytes(a, b)) => key = text(&d, (a, b)),
                    (2, crate::proto::Val::Varint(n)) => on = n != 0,
                    _ => {}
                });
                let set = if f == 21 { &mut m.ov_vars } else { &mut m.ov_res };
                let set = set.get_or_insert_with(Default::default);
                if on && !key.is_empty() {
                    set.insert(key);
                }
            }
            // ⚠ A parameter is { f1 = name, f2 = <message holding the value> }.
            // The value is NOT a scalar on the parameter itself - reading it
            // that way finds nothing at all, and every material silently comes
            // out with no params and no colours.
            4 => {
                let mut name = String::new();
                let mut floats: Vec<f32> = Vec::new();
                fields(&d, s, e, |pf, pv| {
                    if let Some((a, b)) = pv.range() {
                        if pf == 1 {
                            name = text(&d, (a, b));
                        } else if pf == 2 {
                            floats = param_value(&d, a, b).unwrap_or_default();
                        }
                    }
                });
                if name.is_empty() || floats.is_empty() {
                    return;
                }
                // Keep VECTORS as well as scalars: car paint carries its colour
                // in Base_Basecolor / Red_Basecolor, not in a texture.
                if floats.len() == 1 {
                    m.params.insert(name, floats[0]);
                } else {
                    floats.truncate(4);
                    m.colors.insert(name, floats);
                }
            }
            // texture slot: { f1 = slot name, f2 = <message holding the path> }
            5 => {
                let mut name = String::new();
                let mut refs: Vec<String> = Vec::new();
                fields(&d, s, e, |sf, sv| {
                    if let Some((a, b)) = sv.range() {
                        if sf == 1 {
                            name = text(&d, (a, b));
                        } else if sf == 2 {
                            strings_under(&d, a, b, 0, &mut refs);
                        }
                    }
                });
                if !name.is_empty() {
                    let n = refs.len();
                    if let Some(r) = refs.into_iter().next() {
                        m.slots.insert(name, r);
                    } else if std::env::var("EVOVIEW_SLOT_LOG").is_ok()
                        && name.contains("BaseColorMap")
                    {
                        // ⚠ Only colour slots. A material declares the FULL slot
                        // list for its shader and leaves most of them empty -
                        // logging every one gives 7000+ lines per track and
                        // reads like a parser bug when nothing is wrong.
                        eprintln!("SLOTDROP	{}	{}	refs={}", m.file, name, n);
                    }
                }
            }
            _ => {}
        }
    });
    Some(m)
}

/// `read_material`, with its base-material chain resolved.
///
/// ⚠ Track materials INHERIT (MaterialData f20 baseMaterialPath): Monza's
/// road `apshalt_normal` sets only its colour and a few scales, and every
/// texture the road shader uses - Base_BaseMap, Detail_Map, Variation_Map,
/// the detail normal - comes from common_assets/materials/asphalt_new.
/// Read alone it was a material of editor placeholders, which is why the
/// asphalt was one stretched base photo with no grain. The child's values
/// win only for the keys its override lists name (f21 / f22), or for every
/// key it sets when it has no list.
fn read_material_vfs(vfs: &mut Vfs, d: &[u8], file: &str) -> Option<Mat> {
    let m = read_material(d, file)?;
    Some(resolve_material(vfs, m, 0))
}

fn resolve_material(vfs: &mut Vfs, m: Mat, depth: u32) -> Mat {
    let Some(base) = m.base.clone() else { return m };
    if depth > 4 {
        return m;
    }
    let Some(key) = vfs.find(&base, None) else { return m };
    let pd = vfs.read(&key);
    let Some(parent) = read_material(&pd, &base_of(&key)) else { return m };
    let parent = resolve_material(vfs, parent, depth + 1);
    let mut out = parent;
    out.file = m.file.clone();
    if m.shader.is_some() {
        out.shader = m.shader.clone();
    }
    let var_ok = |k: &str| m.ov_vars.as_ref().map_or(true, |s| s.contains(k));
    let res_ok = |k: &str| m.ov_res.as_ref().map_or(true, |s| s.contains(k));
    for (k, v) in &m.params {
        // "__" keys are the child's own top-level fields, not variables:
        // they are never in its override list and always win
        if var_ok(k) || k.starts_with("__") || !out.params.contains_key(k) && !out.colors.contains_key(k) {
            out.params.insert(k.clone(), *v);
        }
    }
    for (k, v) in &m.colors {
        if var_ok(k) || !out.colors.contains_key(k) && !out.params.contains_key(k) {
            out.colors.insert(k.clone(), v.clone());
        }
    }
    for (k, v) in &m.slots {
        // an editor placeholder never overrides a real parent texture
        let placeholder = v.to_lowercase().contains("editor\textures") || v.to_lowercase().contains("editor/textures");
        if (res_ok(k) && !placeholder) || !out.slots.contains_key(k) {
            out.slots.insert(k.clone(), v.clone());
        }
    }
    // a top-level field, not a variable: the child's own (absent = 0)
    match m.params.get("__cull") {
        Some(c) => { out.params.insert("__cull".into(), *c); }
        None => { out.params.remove("__cull"); }
    }
    out.base = None;
    out
}

/// Fence cards keep coverage in a sibling `foo_alpha.texture` the material
/// never names. Without it every grass card is an opaque black quad.
fn companion_alpha_ref(m: &Mat) -> Option<String> {
    let fence = m
        .shader
        .as_deref()
        .map(|s| s.eq_ignore_ascii_case("Fence"))
        .unwrap_or(false)
        || m.params.get("blendMode").copied().unwrap_or(0.0) >= 1.0;
    if !fence {
        return None;
    }
    if m.slots.keys().any(|k| k.to_lowercase().contains("opacity")) {
        return None;
    }
    let alb = m
        .slots
        .get("txDiffuse")
        .or_else(|| m.slots.get("Base_BaseColorMap"))?;
    let low = alb.replace('\\', "/").to_lowercase();
    if let Some(stem) = low.strip_suffix(".texture") {
        Some(format!("{stem}_alpha.texture"))
    } else {
        Some(format!("{low}_alpha.texture"))
    }
}

// -------------------------------------------------------------------- design --

/// A design slot: which material it drives, and any paint options.
struct Slot {
    id: String,
    material: String,
    paints: Vec<String>,
    /// CustomizableMaterial.is_optional (field 9): an option the visual
    /// preset has to switch on (e.g. the AE86's red interior).
    optional: bool,
    /// Channels that copy another material's channel instead of taking a
    /// paint of their own: (this channel, source material, source channel).
    /// CustomizableChannel mode 3 with f20 = material, f21 = channel.
    links: Vec<(usize, String, usize)>,
    /// The channel the slot's paint options (f6.4) dress: the first channel
    /// marked customizable (mode 1 or 2). None = not stated (primary).
    paint_channel: Option<usize>,
}

fn read_design(d: &[u8]) -> Vec<Slot> {
    let d = d.to_vec();
    let mut out = Vec::new();
    fields(&d, 0, d.len(), |f, v| {
        if f != 3 {
            return;
        }
        let (s, e) = match v.range() {
            Some(r) => r,
            None => return,
        };
        let mut slot = Slot { id: String::new(), material: String::new(), paints: Vec::new(),
                              optional: false, links: Vec::new(), paint_channel: None };
        fields(&d, s, e, |sf, sv| {
            if let (9, Val::Varint(x)) = (sf, &sv) {
                slot.optional = *x != 0;
            }
            if let Some((a, b)) = sv.range() {
                match sf {
                    1 => slot.id = text(&d, (a, b)),
                    3 => slot.material = text(&d, (a, b)),
                    6 => {
                        let mut refs = Vec::new();
                        strings_under(&d, a, b, 0, &mut refs);
                        for r in refs {
                            if r.ends_with(".oemmultilayercolor") {
                                slot.paints.push(r);
                            }
                        }
                        fields(&d, a, b, |cf, cv| {
                            let (1..=3, Some((ca, cb))) = (cf, cv.range()) else { return };
                            let (mut mode, mut src, mut ch) = (0u64, String::new(), 0u64);
                            fields(&d, ca, cb, |lf, lv| match (lf, lv) {
                                (1, Val::Varint(x)) => mode = x,
                                (21, Val::Varint(x)) => ch = x,
                                (20, Val::Bytes(x, y)) => src = text(&d, (x, y)),
                                _ => {}
                            });
                            if mode == 3 && !src.is_empty() && ch < 3 {
                                slot.links.push((cf as usize - 1, src, ch as usize));
                            }
                            if (mode == 1 || mode == 2) && slot.paint_channel.is_none() {
                                slot.paint_channel = Some(cf as usize - 1);
                            }
                        });
                    }
                    _ => {}
                }
            }
        });
        if !slot.id.is_empty() {
            out.push(slot);
        }
    });
    out
}

struct Paint {
    name: String,
    rgb: [f32; 3],
    metallic: f32,
    roughness: f32,
    clearcoat: f32,
    /// OemMultilayerColorData (Customization.proto in the exe):
    /// 7 clear_coat_roughness, 8 normal_intensity, 9 ignore_material.
    /// normal_intensity is the strength of the layer's detail normal - on body
    /// paint that is flakes_NM, which is all EVO's "metallic flake" is. The
    /// body materials ship Red_NormalScale 0 and rely on the paint for it.
    clearcoat_rough: f32,
    normal_intensity: f32,
    /// Colour only: decals, stitching and leather keep their material's own
    /// roughness/metalness/clear coat.
    ignore_material: bool,
}

fn read_paint(d: &[u8]) -> Option<Paint> {
    let d = d.to_vec();
    let mut p = Paint {
        name: String::new(),
        rgb: [0.0; 3],
        metallic: 0.0,
        roughness: 0.45,
        clearcoat: 0.0,
        clearcoat_rough: 0.0,
        normal_intensity: 0.0,
        ignore_material: false,
    };
    let mut got = false;
    fields(&d, 0, d.len(), |f, v| match (f, v) {
        (2, Val::Bytes(a, b)) => p.name = text(&d, (a, b)),
        (3, Val::Bytes(a, b)) => {
            // ⚠ proto3: a zero channel is omitted (audi_paint_blue_ascari_met
            // has no f1), so index by field number, missing = 0.
            let mut c = [0.0f32; 3];
            fields(&d, a, b, |cf, cv| {
                if let (1..=3, Val::F32(x)) = (cf, cv) {
                    c[cf as usize - 1] = x;
                }
            });
            p.rgb = c;
            got = true;
        }
        (4, Val::F32(x)) => p.metallic = x,
        (5, Val::F32(x)) => p.roughness = x,
        (6, Val::F32(x)) => p.clearcoat = x,
        (7, Val::F32(x)) => p.clearcoat_rough = x,
        (8, Val::F32(x)) => p.normal_intensity = x,
        (9, Val::Varint(x)) => p.ignore_material = x != 0,
        _ => {}
    });
    if got {
        Some(p)
    } else {
        None
    }
}

// ----------------------------------------------------------------- animation --

pub fn read_animation(d: &[u8], name: &str) -> Option<Animation> {
    let d = d.to_vec();
    let mut anim = Animation {
        name: name.to_string(),
        fps: 30.0,
        tracks: Vec::new(),
        frames: 0,
    };
    fields(&d, 0, d.len(), |f, v| {
        if f != 2 {
            return;
        }
        let (s, e) = match v.range() {
            Some(r) => r,
            None => return,
        };
        fields(&d, s, e, |tf, tv| match (tf, tv) {
            (2, Val::F32(x)) if x > 0.0 => anim.fps = x,
            // one repeated field 3 per animated bone
            (3, Val::Bytes(a, b)) => {
                let mut track = Track { bone: String::new(), frames: Vec::new() };
                fields(&d, a, b, |bf, bv| {
                    if let Some((x, y)) = bv.range() {
                        match bf {
                            1 => track.bone = text(&d, (x, y)),
                            // repeated field 2: one 4x4 per frame
                            2 => track.frames.push(matrix(&d, x, y)),
                            _ => {}
                        }
                    }
                });
                if !track.bone.is_empty() && !track.frames.is_empty() {
                    anim.frames = anim.frames.max(track.frames.len());
                    anim.tracks.push(track);
                }
            }
            _ => {}
        });
    });
    if anim.tracks.is_empty() {
        None
    } else {
        Some(anim)
    }
}

// ------------------------------------------------------------------ assembly --

/// Where a file lives: on disk, or inside a .kspkg.
enum Src {
    File(PathBuf),
    /// An archive entry and which archive holds it: 0 = the car's own
    /// package, k = `extra[k - 1]` (a base layered underneath).
    /// ⚠ Entry offsets only mean something in their own archive. Reading a
    /// base entry from the mod package returned the mod's bytes at that
    /// offset whenever it lay inside the (smaller) mod file - garbage of the
    /// right length, so the "wrong size, try the next" fallback never fired
    /// (a mod car's shared particle textures decoded as nothing).
    Pkg(crate::pkg::Entry, usize),
    /// Made in memory (online: a player's own look, mp.rs)
    Bytes(std::sync::Arc<Vec<u8>>),
}

/// One lookup surface over a car, whether it is an extracted FOLDER or a car
/// inside a `.kspkg`.
///
/// ⚠ Extracting a car folder just to view it costs 200-800 MB per car. Reading
/// the archive in place means nothing is written to disk at all - so this
/// exists to keep both paths identical to everything above it.
pub struct Vfs {
    pkg: Option<crate::pkg::Package>,
    /// packages layered underneath (the base game, for a mod car)
    extra: Vec<crate::pkg::Package>,
    /// virtual path (lowercase, '/' separators) -> where to get the bytes
    entries: HashMap<String, Src>,
    by_base: HashMap<String, Vec<String>>,
    /// prefix every car-relative lookup sits under ("" for a folder)
    root: String,
    /// Textures decoded ahead of use by `prefetch_textures`, keyed by
    /// (vfs key, skip_mips). `texture` takes from here first.
    decoded: HashMap<(String, u32), Tex>,
    /// Decoded textures kept for the life of the Vfs (livery switching
    /// reloads the same car many times; see livery.rs). None = off.
    keep: Option<HashMap<(String, u32), Tex>>,
    /// With `keep`: raw bytes of every non-texture file read so far.
    kept_files: HashMap<String, Vec<u8>>,
    /// Textures the viewer already has on the GPU (`mark_resident`): their
    /// pixels are not needed again, so `texture` hands back the header only.
    resident: std::collections::HashSet<String>,
    /// With `keep`: memoised `list` / `list_rec` results (each one scans
    /// every entry - ~80k for a car read out of content.kspkg - and a load
    /// asks for the presets folder dozens of times).
    listed: std::sync::Mutex<HashMap<(bool, String, String), Vec<String>>>,
    /// every key read so far (EVOVIEW_RENDER_AUDIT: what the car ships that
    /// nothing opened)
    reads: std::sync::Mutex<std::collections::HashSet<String>>,
}

impl Vfs {
    fn insert(&mut self, key: String, src: Src) {
        let base = key.rsplit('/').next().unwrap_or("").to_string();
        self.by_base.entry(base).or_default().push(key.clone());
        self.entries.insert(key, src);
    }

    /// A car folder on disk, plus any shared asset roots.
    pub fn folder(dir: &Path, asset_roots: &[PathBuf]) -> Self {
        let mut v = Vfs {
            pkg: None,
            extra: Vec::new(),
            entries: HashMap::new(),
            by_base: HashMap::new(),
            root: String::new(),
            decoded: HashMap::new(),
            keep: None,
            kept_files: HashMap::new(),
            resident: std::collections::HashSet::new(),
            listed: Default::default(),
            reads: Default::default(),
        };
        let mut files = Vec::new();
        collect_all(dir, &mut files);
        for f in files {
            if let Ok(rel) = f.strip_prefix(dir) {
                let key = rel.to_string_lossy().replace('\\', "/").to_lowercase();
                v.insert(key, Src::File(f.clone()));
            }
        }
        for r in asset_roots {
            let mut extra = Vec::new();
            collect_all(r, &mut extra);
            for f in extra {
                if let Ok(rel) = f.strip_prefix(r) {
                    let key = format!("::shared/{}",
                        rel.to_string_lossy().replace('\\', "/").to_lowercase());
                    v.insert(key, Src::File(f.clone()));
                }
            }
        }
        v
    }

    /// Layer a second archive UNDER this one: entries already present win.
    ///
    /// ⚠ A mod .kspkg contains only its own car. Shared assets - the tyre
    /// material above all - live in the base game package, so without this a
    /// modded car renders its tyres on the fallback material.
    /// Layer only the parts of a base archive we can actually need.
    ///
    /// `content.kspkg` holds ~80k entries; inserting all of them to reach a few
    /// hundred shared files cost ~600 ms of load time. Shared assets all live
    /// under `common_assets`, and `find()` resolves by basename, so restricting
    /// the layer to those prefixes changes nothing that resolves - it just
    /// stops building an index of every track in the game to look up a tyre.
    pub fn under_prefixes(&mut self, base: crate::pkg::Package, prefixes: &[&str]) {
        let add: Vec<(String, crate::pkg::Entry)> = base
            .by_path
            .iter()
            .map(|(k, e)| (k.replace('\\', "/").to_lowercase(), *e))
            .filter(|(k, _)| prefixes.iter().any(|p| k.starts_with(p)))
            .filter(|(k, _)| !self.entries.contains_key(k))
            .collect();
        let owner = self.extra.len() + 1;
        for (k, e) in add {
            self.insert(k, Src::Pkg(e, owner));
        }
        self.extra.push(base);
    }

    pub fn under(&mut self, base: crate::pkg::Package) {
        let add: Vec<(String, crate::pkg::Entry)> = base
            .by_path
            .iter()
            .map(|(k, e)| (k.replace('\\', "/").to_lowercase(), *e))
            .filter(|(k, _)| !self.entries.contains_key(k))
            .collect();
        let owner = self.extra.len() + 1;
        for (k, e) in add {
            self.insert(k, Src::Pkg(e, owner));
        }
        self.extra.push(base);
    }

    /// One car inside an archive. Nothing is written to disk.
    pub fn package(pkg: crate::pkg::Package, car_id: &str) -> Self {
        Self::package_prefix(pkg, format!("content/cars/{}/", car_id.to_lowercase()))
    }

    /// One track folder inside an archive (`content/tracks/<id>/`).
    pub fn package_track(pkg: crate::pkg::Package, folder: &str) -> Self {
        Self::package_prefix(pkg, format!("content/tracks/{}/", folder.to_lowercase()))
    }

    fn package_prefix(pkg: crate::pkg::Package, root: String) -> Self {
        let t = std::time::Instant::now();
        let v = Self::package_prefix_inner(pkg, root);
        crate::perf::add("vfs index", t.elapsed().as_secs_f64());
        v
    }

    fn package_prefix_inner(pkg: crate::pkg::Package, root: String) -> Self {
        let mut v = Vfs {
            entries: HashMap::new(),
            by_base: HashMap::new(),
            root,
            pkg: None,
            extra: Vec::new(),
            decoded: HashMap::new(),
            keep: None,
            kept_files: HashMap::new(),
            resident: std::collections::HashSet::new(),
            listed: Default::default(),
            reads: Default::default(),
        };
        for (path, e) in &pkg.by_path {
            v.insert(path.replace('\\', "/"), Src::Pkg(*e, 0));
        }
        v.pkg = Some(pkg);
        v
    }

    pub fn read(&mut self, key: &str) -> Vec<u8> {
        if let Some(b) = self.kept_files.get(key) {
            return b.clone();
        }
        let t = std::time::Instant::now();
        let r = self.read_inner(key);
        if self.keep.is_some() && !key.ends_with(".texture") && !key.ends_with(".texturemips") {
            self.kept_files.insert(key.to_string(), r.clone());
        }
        crate::perf::add("file read", t.elapsed().as_secs_f64());
        r
    }

    fn read_inner(&self, key: &str) -> Vec<u8> {
        if let Ok(mut r) = self.reads.lock() {
            r.insert(key.to_string());
        }
        match self.entries.get(key) {
            Some(Src::File(p)) => std::fs::read(p).unwrap_or_default(),
            Some(Src::Bytes(b)) => b.to_vec(),
            Some(Src::Pkg(e, owner)) => {
                let e = *e;
                let mut got = match owner {
                    0 => self.pkg.as_ref().map(|p| p.read(e)).unwrap_or_default(),
                    k => self.extra.get(k - 1).map(|p| p.read(e)).unwrap_or_default(),
                };
                if got.len() != e.size as usize {
                    for x in self.extra.iter() {
                        let t = x.read(e);
                        if t.len() == e.size as usize {
                            got = t;
                            break;
                        }
                    }
                }
                got
            }
            None => Vec::new(),
        }
    }

    /// Every file under the car's root (lowercase keys), and whether it
    /// has been read.
    pub fn audit_files(&self) -> Vec<(String, bool)> {
        let r = self.reads.lock().map(|r| r.clone()).unwrap_or_default();
        let mut out: Vec<(String, bool)> = self.entries.keys()
            .filter(|k| k.starts_with(&self.root))
            .map(|k| (k.clone(), r.contains(k)))
            .collect();
        out.sort();
        out
    }

    /// Whether `key` has been read.
    pub fn was_read(&self, key: &str) -> bool {
        self.reads.lock().map_or(false, |r| r.contains(key))
    }

    pub fn root(&self) -> &str {
        &self.root
    }

    /// Add a file made in memory at `rel` (car-relative, e.g.
    /// "presets/x.visualcarpreset"), replacing any of that name.
    pub fn add_file(&mut self, rel: &str, bytes: Vec<u8>) {
        let key = format!("{}{}", self.root, rel.replace('\\', "/").to_lowercase());
        self.kept_files.remove(&key);
        if !self.entries.contains_key(&key) {
            self.insert(key, Src::Bytes(std::sync::Arc::new(bytes)));
        } else {
            self.entries.insert(key, Src::Bytes(std::sync::Arc::new(bytes)));
        }
        self.forget_lists();
    }

    pub fn exists(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Read and decode one `.texture` (+ its `.texturemips`), using a
    /// prefetched result when there is one.
    fn texture(&mut self, key: &str, skip_mips: u32) -> Option<Tex> {
        let id = (key.to_string(), skip_mips);
        if let Some(t) = self.keep.as_ref().and_then(|k| k.get(&id)) {
            return Some(t.clone()); // a header only once it is resident
        }
        let t = match self.decoded.remove(&id) {
            Some(t) => Some(t),
            None => {
                let hdr = self.read(key);
                let mips = self.read(&key.replace(".texture", ".texturemips"));
                read_texture_lod(&hdr, &mips, skip_mips)
            }
        };
        if let (Some(k), Some(t)) = (self.keep.as_mut(), t.as_ref()) {
            k.insert(id, t.clone());
        }
        t
    }

    /// Keep every decoded texture (and every other file read) for later
    /// loads of the same car.
    pub fn keep_textures(&mut self) {
        self.keep.get_or_insert_with(HashMap::new);
    }

    /// The viewer holds these textures (vfs keys) on the GPU, keyed the same
    /// way: drop their pixels here and serve headers from now on.
    pub fn mark_resident(&mut self, keys: &[String]) {
        let Some(keep) = self.keep.as_mut() else { return };
        for k in keys {
            if self.resident.insert(k.clone()) {
                for ((kk, _), t) in keep.iter_mut() {
                    if kk == k {
                        t.data = Vec::new();
                    }
                }
            }
        }
    }

    /// Forget prefetched textures nobody took (after the material pass).
    pub fn drop_prefetched(&mut self) {
        self.decoded = HashMap::new();
    }

    /// Read and decode `keys` on a few worker threads, ahead of the
    /// (sequential) material pass that consumes them through `texture`.
    ///
    /// Reading is positional and decoding is pure, so the order cannot change
    /// the result - only how long the load takes. Kept to a handful of
    /// threads: the load should not take the whole machine. Returns the bytes
    /// decoded, so the caller can size its blob once.
    pub fn prefetch_textures(&mut self, keys: &[String], skip_mips: u32) -> usize {
        let mut keys: Vec<&String> = keys
            .iter()
            .filter(|k| self.entries.contains_key(k.as_str()))
            .filter(|k| !self.decoded.contains_key(&((*k).clone(), skip_mips)))
            .filter(|k| !self.keep.as_ref().map_or(false, |m| m.contains_key(&((*k).clone(), skip_mips))))
            .collect();
        keys.sort();
        keys.dedup();
        if keys.is_empty() {
            return 0;
        }
        let t0 = std::time::Instant::now();
        let threads = std::thread::available_parallelism()
            .map_or(4, |n| n.get() / 2)
            .clamp(1, 6)
            .min(keys.len());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let me: &Vfs = self;
        let results: Vec<(String, Tex)> = std::thread::scope(|sc| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    sc.spawn(|| {
                        let mut out = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(k) = keys.get(i) else { break };
                            let hdr = me.read_inner(k);
                            let mips = me.read_inner(&k.replace(".texture", ".texturemips"));
                            if let Some(t) = read_texture_lod(&hdr, &mips, skip_mips) {
                                out.push(((*k).clone(), t));
                            }
                        }
                        out
                    })
                })
                .collect();
            workers.into_iter().flat_map(|w| w.join().unwrap_or_default()).collect()
        });
        let mut total = 0usize;
        for (k, t) in results {
            total += t.data.len();
            self.decoded.insert((k, skip_mips), t);
        }
        crate::perf::add("texture prefetch (wall)", t0.elapsed().as_secs_f64());
        total
    }

    /// Files that belong to THIS track. A rooted package VFS (stock circuit)
    /// is everything under `content/tracks/<id>/`. A loose folder is only
    /// on-disk files — never the --base kspkg, or every other circuit
    /// would be planted on the drift map.
    pub fn own_keys(&self, ext: &str) -> Vec<String> {
        let dot = format!(".{}", ext.to_lowercase());
        let mut out: Vec<String> = self
            .entries
            .iter()
            .filter(|(k, src)| {
                if !k.ends_with(&dot) {
                    return false;
                }
                let low = k.to_lowercase();
                if low.contains("/irradiance") || low.contains("/physics") {
                    return false;
                }
                if !self.root.is_empty() {
                    return low.starts_with(&self.root);
                }
                matches!(src, Src::File(_))
            })
            .map(|(k, _)| k.clone())
            .collect();
        out.sort();
        out
    }

    /// Files anywhere under `<car>/<subdir>` with the given extension.
    ///
    /// ⚠ Recursive on purpose. Cars with trim variants keep their materials in
    /// `materials/<variant>/`, so listing only the top level leaves nearly
    /// every batch on the fallback material - which is opaque grey, and drawn
    /// in the OPAQUE pass, so it also makes glass solid.
    pub fn list_rec(&self, subdir: &str, ext: &str) -> Vec<String> {
        self.memo_list(true, subdir, ext, || self.list_rec_inner(subdir, ext))
    }

    /// Drop the memoised listings (a file was added).
    fn forget_lists(&self) {
        self.listed.lock().unwrap().clear();
    }

    fn memo_list(&self, rec: bool, subdir: &str, ext: &str, f: impl FnOnce() -> Vec<String>) -> Vec<String> {
        if self.keep.is_none() {
            return f();
        }
        let id = (rec, subdir.to_lowercase(), ext.to_lowercase());
        if let Some(v) = self.listed.lock().unwrap().get(&id) {
            return v.clone();
        }
        let v = f();
        self.listed.lock().unwrap().insert(id, v.clone());
        v
    }

    fn list_rec_inner(&self, subdir: &str, ext: &str) -> Vec<String> {
        // An empty subdir must not become "/" or every relative key misses.
        let want = if subdir.is_empty() {
            self.root.clone()
        } else if self.root.is_empty() {
            format!("{}/", subdir.to_lowercase())
        } else {
            format!("{}{}/", self.root, subdir.to_lowercase())
        };
        let dot = format!(".{}", ext.to_lowercase());
        let mut out: Vec<String> = self
            .entries
            .keys()
            .filter(|k| k.starts_with(&want) && k.ends_with(&dot))
            .cloned()
            .collect();
        out.sort();
        out
    }

    /// Files directly under `<car>/<subdir>` with the given extension.
    pub fn list(&self, subdir: &str, ext: &str) -> Vec<String> {
        self.memo_list(false, subdir, ext, || self.list_inner(subdir, ext))
    }

    fn list_inner(&self, subdir: &str, ext: &str) -> Vec<String> {
        let want = if subdir.is_empty() {
            self.root.clone()
        } else {
            format!("{}{}/", self.root, subdir.to_lowercase())
        };
        let dot = format!(".{}", ext.to_lowercase());
        let mut out: Vec<String> = self
            .entries
            .keys()
            .filter(|k| k.starts_with(&want) && k.ends_with(&dot))
            .filter(|k| !k[want.len()..].contains('/'))
            .cloned()
            .collect();
        out.sort();
        out
    }

    /// Resolve an engine reference by file name. `near` (a material's own
    /// folder) wins, which is what lets a livery texture beat the car default
    /// of the same name.
    pub fn find(&self, reference: &str, near: Option<&str>) -> Option<String> {
        let rel = reference.replace('\\', "/").to_lowercase();
        let base = rel.rsplit('/').next()?.to_string();
        let hits = self.by_base.get(&base)?;
        if let Some(n) = near {
            if let Some(h) = hits.iter().find(|k| parent_of(k) == n) {
                return Some(h.clone());
            }
        }
        // prefer an exact tail match over a bare name collision
        if let Some(h) = hits.iter().find(|k| k.ends_with(&rel) || rel.ends_with(*k)) {
            return Some(h.clone());
        }
        if rel.len() > 12 {
            if let Some(h) = hits.iter().find(|k| k.contains(&rel) || rel.contains(*k)) {
                return Some(h.clone());
            }
        }
        hits.first().cloned()
    }
}

/// The textures `paths` (materials) bind directly, resolved the way the
/// material passes resolve them, so they can be decoded in parallel first.
/// Anything this misses (sibling / companion lookups, tyre overrides) is
/// simply decoded later, in line.
fn material_texture_keys(vfs: &mut Vfs, paths: &[String]) -> Vec<String> {
    let mut keys = Vec::new();
    for path in paths {
        let mb = vfs.read(path);
        let Some(m) = read_material_vfs(vfs, &mb, &base_of(path)) else { continue };
        let near = parent_of(path).to_string();
        for reference in m.slots.values() {
            let low = reference.replace('\\', "/").to_lowercase();
            if low.contains("default_material") || low.starts_with("editor/")
                || low.contains("/null.texture") || low.ends_with("null.texture") {
                continue;
            }
            if let Some(found) = vfs.find(reference, Some(&near)) {
                keys.push(found);
            }
        }
    }
    keys
}

/// File name of a vfs key, lowercase.
pub(crate) fn base_of(key: &str) -> String {
    key.rsplit('/').next().unwrap_or(key).to_string()
}

fn collect_all(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(root) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_all(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn parent_of(key: &str) -> &str {
    match key.rfind('/') {
        Some(i) => &key[..i],
        None => "",
    }
}

pub struct Loaded {
    pub car: Car,
    pub skeleton: Skeleton,
    pub animations: Vec<Animation>,
}

/// Read a car - from an extracted folder or straight out of a .kspkg.
/// Everything a car can need from outside its own folder lives under one of
/// these. Widen only with a measurement: each extra prefix is load time.
const SHARED_PREFIXES: &[&str] = &[
    "content/cars/common_assets/",
    "content/tracks/common_assets/",
    // the driver: its parts, and the default outfits (~1.9k entries)
    "content/characters/",
    "system/saveddrivers/",
    // particle emitters, curves and textures (exhaust flames, tyre smoke)
    "content/particles/",
];

/// The same set in the archive's own backslash form, applied while the table is
/// being read so non-shared entries never get allocated at all.
const SHARED_PKG_PREFIXES: &[&str] = &[
    r"content\cars\common_assets\",
    r"content\tracks\common_assets\",
    r"content\characters\",
    r"system\saveddrivers\",
    r"content\particles\",
];

pub fn load_car_dir(dir: &Path, asset_roots: &[PathBuf]) -> Result<Loaded, String> {
    load(&mut car_dir_vfs(dir, asset_roots)?)
}

/// The Vfs `load_car_dir` reads from (kept by livery.rs for switching).
pub fn car_dir_vfs(dir: &Path, asset_roots: &[PathBuf]) -> Result<Vfs, String> {
    if !dir.is_dir() {
        return Err(format!("{} is not a directory", dir.display()));
    }
    // Shared assets - tyres, paints, the studio environment - belong to the
    // player's own install, so we layer their content.kspkg underneath rather
    // than shipping a copy of Kunos' files. Entries in the car folder still win.
    //
    // Opening it costs ~300 ms: content.kspkg has ~80k entries and the whole
    // table has to be read to index any of them. Walking the car folder takes
    // about as long and needs no archive, so the two run together and the
    // archive is effectively free.
    let base_job = std::thread::spawn(|| {
        let path = crate::install::find_content_kspkg()?;
        match crate::pkg::Package::open_filtered(&path, SHARED_PKG_PREFIXES) {
            Ok(p) => Some((path, p)),
            Err(e) => {
                eprintln!("shared assets: {} could not be read ({e})", path.display());
                None
            }
        }
    });

    let mut vfs = Vfs::folder(dir, asset_roots);

    match base_job.join() {
        Ok(Some((path, p))) => {
            println!("shared assets: {}", path.display());
            vfs.under_prefixes(p, SHARED_PREFIXES);
        }
        _ => {
            if asset_roots.iter().all(|r| !r.exists()) {
                eprintln!("{}", crate::install::missing_note());
            }
        }
    }
    Ok(vfs)
}

/// Every car inside an archive, without loading any of them.
pub fn list_package(pkg_path: &Path) -> Result<Vec<String>, String> {
    Ok(crate::pkg::Package::open(pkg_path)?.cars())
}

/// Every track folder inside an archive that has static meshes.
pub fn list_tracks(pkg_path: &Path) -> Result<Vec<String>, String> {
    Ok(crate::pkg::Package::open(pkg_path)?.tracks())
}

/// Read one track out of an archive. Static meshes only — no vegetation kit.
/// `world` = Some(true) loads shipped Eifel/village containers; None infers
/// from the folder name (on for nurburgring).
pub fn load_track_from_package(
    pkg_path: &Path,
    folder: &str,
    base: Option<&Path>,
) -> Result<Loaded, String> {
    load_track_from_package_opts(pkg_path, folder, base, None)
}

pub fn load_track_from_package_opts(
    pkg_path: &Path,
    folder: &str,
    base: Option<&Path>,
    world: Option<bool>,
) -> Result<Loaded, String> {
    TRACK_4K_SEEN.store(0, std::sync::atomic::Ordering::Relaxed);
    let pkg = crate::pkg::Package::open(pkg_path)?;
    let want = folder.to_lowercase();
    if !pkg.tracks().iter().any(|t| t.eq_ignore_ascii_case(&want))
        && !pkg.by_path.keys().any(|p| {
            p.replace('\\', "/").to_lowercase()
                .starts_with(&format!("content/tracks/{want}/"))
        })
    {
        return Err(format!("{folder} is not in {}", pkg_path.display()));
    }
    let mut vfs = Vfs::package_track(pkg, folder);
    if let Some(b) = base {
        if b != pkg_path {
            vfs.under(crate::pkg::Package::open(b)?);
        }
    }
    load_track_opts(&mut vfs, world.unwrap_or(is_eifel_world_folder(folder)))
}

/// `track` is a folder id inside `pkg`, or a path to a loose imported track
/// folder (then `pkg`, the game's content.kspkg, backs it for shared assets).
pub fn load_track_any(pkg: &Path, track: &str, base: Option<&Path>) -> Result<Loaded, String> {
    let dir = Path::new(track);
    if dir.is_dir() {
        load_track_dir_opts(dir, Some(pkg), None)
    } else {
        // a mod track id: installed loose under Saved Games/ACE/mods
        let m = std::env::var_os("USERPROFILE")
            .map(|h| Path::new(&h).join("Saved Games").join("ACE").join("mods").join("content").join("tracks").join(track))
            .filter(|d| d.is_dir());
        match load_track_from_package_opts(pkg, track, base, None) {
            Err(e) if m.is_some() && e.contains("is not in") => {
                println!("track {track}: the mod {}", m.as_ref().unwrap().display());
                load_track_dir_opts(m.as_ref().unwrap(), Some(pkg), None)
            }
            r => r,
        }
    }
}

/// A loose imported track folder (a mod manager / ACECM install).
/// `base` is the game content.kspkg so vegetation in common_assets resolves.
pub fn load_track_dir(dir: &Path, base: Option<&Path>) -> Result<Loaded, String> {
    load_track_dir_opts(dir, base, None)
}

pub fn load_track_dir_opts(
    dir: &Path,
    base: Option<&Path>,
    world: Option<bool>,
) -> Result<Loaded, String> {
    if !dir.is_dir() {
        return Err(format!("{} is not a directory", dir.display()));
    }
    let mut vfs = Vfs::folder(dir, &[]);
    if let Some(b) = base {
        if let Ok(pkg) = crate::pkg::Package::open(b) {
            vfs.under(pkg);
        }
    }
    let infer = is_eifel_world_folder(
        dir.file_name().and_then(|s| s.to_str()).unwrap_or(""),
    );
    load_track_opts(&mut vfs, world.unwrap_or(infer))
}

struct SceneInst {
    path: String,
    pos: [f32; 3],
    euler: [f32; 3],
    scale: [f32; 3],
}

fn f3_at(buf: &[u8], i: usize) -> [f32; 3] {
    let o = i * 12;
    if o + 12 > buf.len() {
        return [0.0; 3];
    }
    [
        f32::from_le_bytes(buf[o..o + 4].try_into().unwrap()),
        f32::from_le_bytes(buf[o + 4..o + 8].try_into().unwrap()),
        f32::from_le_bytes(buf[o + 8..o + 12].try_into().unwrap()),
    ]
}

fn write_f3(buf: &mut [u8], i: usize, v: [f32; 3]) {
    let o = i * 12;
    if o + 12 > buf.len() {
        return;
    }
    buf[o..o + 4].copy_from_slice(&v[0].to_le_bytes());
    buf[o + 4..o + 8].copy_from_slice(&v[1].to_le_bytes());
    buf[o + 8..o + 12].copy_from_slice(&v[2].to_le_bytes());
}

pub(crate) fn euler_rot(e: [f32; 3]) -> [[f32; 3]; 3] {
    // Scene eulers, left-handed Y-up (sine signs flipped vs RH); positions
    // are already in the viewer's Y-up space. ⚠ The order is X, then Y, then
    // Z (v * Rx * Ry * Rz), NOT Unity's ZXY: Imola's truss frames
    // (180, 65.8, 180) and (-173, -82.7, 173) only line up with the rows
    // their instances form - and stay upright - in this order; ZXY swung the
    // start gantry's truss diagonally. Yaw-only props and (180, yaw, 180)
    // spectators come out the same either way.
    let (sx, cx) = e[0].to_radians().sin_cos();
    let (sy, cy) = e[1].to_radians().sin_cos();
    let (sz, cz) = e[2].to_radians().sin_cos();
    let rx = [[1.0, 0.0, 0.0], [0.0, cx, sx], [0.0, -sx, cx]];
    let ry = [[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]];
    let rz = [[cz, sz, 0.0], [-sz, cz, 0.0], [0.0, 0.0, 1.0]];
    mat3_mul(rx, mat3_mul(ry, rz))
}

/// A scene euler as the column-vector rotation everything placed from a
/// scene uses (instances, nodes, statics, colliders): euler_rot's transpose.
/// ⚠ euler_rot itself went in un-transposed, which mirrored every yaw: an
/// instance at (0, a, 0) turned -a and one at (180, a, 180) turned a the
/// wrong way, so Imola's pit-wall fence panels (180, 10.7, 180) sat 21 deg
/// across the wall in a zigzag. Transposed, all 20 of Imola's fence and
/// barrier rows line up with the line their instances form (< 0.1 deg).
/// Nodes used to get negated eulers to make up for it, which equals the
/// transpose only for a pure yaw.
fn scene_rot(e: [f32; 3]) -> [[f32; 3]; 3] {
    let r = euler_rot(e);
    std::array::from_fn(|i| std::array::from_fn(|j| r[j][i]))
}

fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut m = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    m
}

fn mat3_vec(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Column-major world matrix for WGSL `mat4x4(c0,c1,c2,c3) * v`.
///
/// ⚠ The first version uploaded this row-major and relied on a transpose
/// that is correct for bone tables but inverted instance yaw/pitch. Trees
/// lay on their side and grass cards became white walls and floors.
/// Every collider mesh baked to world space (trackdrive.rs builds its grid
/// from this). One triangle list for the whole track, a surface per triangle.
/// The animated props' clips and bone slots (track.rs TrackAnim).
fn build_track_anim(vfs: &mut Vfs, meshes: &[MeshEntry], list: Vec<(usize, Vec<Bone>, String)>)
        -> Option<crate::track::TrackAnim> {
    if list.is_empty() {
        return None;
    }
    let mut clips: Vec<Animation> = Vec::new();
    let mut clip_of: HashMap<String, Option<usize>> = HashMap::new();
    let mut out = crate::track::TrackAnim::default();
    for (mi, bones, path) in list {
        let ci = *clip_of.entry(path.to_lowercase()).or_insert_with(|| {
            let k = vfs.find(&path.replace('\\', "/"), None)?;
            let a = read_animation(&vfs.read(&k), &path)?;
            clips.push(a);
            Some(clips.len() - 1)
        });
        let Some(ci) = ci else { continue };
        // a few phase-shifted copies, so a stand of identical spectators
        // does not wave in lockstep
        let n = meshes[mi].xforms.len().max(1);
        let variants = n.min(4) as u32;
        let nb = bones.len() as u32;
        out.meshes.push(crate::track::AnimMesh {
            mesh: mi, skel: Skeleton::build(bones), clip: ci, base: out.slots, variants,
        });
        out.slots += nb * variants;
    }
    out.clips = clips;
    // EVOVIEW_ANIM_DUMP=<mesh name part>: the skin matrices of that mesh at
    // frame 63 (debugging against an offline skin)
    if let Ok(want) = std::env::var("EVOVIEW_ANIM_DUMP") {
        for am in &out.meshes {
            if !meshes[am.mesh].name.contains(want.as_str()) || meshes[am.mesh].name.contains('#') {
                continue;
            }
            let a = &out.clips[am.clip];
            let p = am.skel.pose_many(&[(a, 63.0 / a.fps)]);
            for (i, b) in am.skel.bones.iter().enumerate() {
                if ["flag.013", "flag_Base", "A_Head", "RightHand"].contains(&b.name.as_str()) {
                    eprintln!("ANIMDUMP {} {} {:?}", meshes[am.mesh].name, b.name,
                              p[i].iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>());
                }
            }
        }
    }
    println!("track animation: {} animated mesh level(s), {} clip(s), {} bone slot(s)",
             out.meshes.len(), out.clips.len(), out.slots);
    (!out.meshes.is_empty()).then_some(out)
}

fn build_collision(vfs: &mut Vfs, colliders: &[(SceneInst, String, bool)], wall_insts: &[(SceneInst, bool)])
        -> Option<crate::track::Collision> {
    if colliders.is_empty() {
        return None;
    }
    let t0 = std::time::Instant::now();
    let mut c = crate::track::Collision::default();
    let mut cache: HashMap<String, Option<(Vec<[f32; 3]>, Vec<u32>)>> = HashMap::new();
    // EVOVIEW_WALL_SRC="x,z": which collider / instance made the walls near a point
    let mut tri_src: Vec<(usize, usize, String)> = Vec::new();
    let wall_src: Option<(f32, f32)> = std::env::var("EVOVIEW_WALL_SRC").ok().and_then(|v| { let (a, b) = v.split_once(',')?; Some((a.trim().parse().ok()?, b.trim().parse().ok()?)) });
    let near = |x: f32, z: f32| wall_src.map_or(false, |(px, pz)| (x - px).powi(2) + (z - pz).powi(2) < 9.0);
    let mut failed: std::collections::HashSet<String> = Default::default();
    for (inst, surf, hidden) in colliders {
        let key = inst.path.to_lowercase();
        let mesh = cache.entry(key).or_insert_with(|| {
            let k = vfs.find(&inst.path, None)?;
            let bytes = vfs.read(&k);
            let lod = read_mesh(&bytes)?;
            let p: Vec<[f32; 3]> = lod.positions.chunks_exact(12)
                .map(|q| std::array::from_fn(|i| f32::from_le_bytes(q[i * 4..i * 4 + 4].try_into().unwrap())))
                .collect();
            (!p.is_empty() && lod.indices.len() >= 3).then(|| (p, lod.indices.clone()))
        });
        let Some((pos, idx)) = mesh.as_ref() else {
            if failed.insert(inst.path.to_lowercase()) {
                println!("collision: collider mesh {} did not load", inst.path);
            }
            continue;
        };
        let si = match c.surfaces.iter().position(|s| s == surf) {
            Some(i) => i,
            None => {
                c.surfaces.push(surf.clone());
                c.surfaces.len() - 1
            }
        } as u16;
        let m = inst_matrix(inst);
        let base = c.verts.len() as u32;
        for p in pos {
            c.verts.push([
                m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
                m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
                m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
            ]);
        }
        let tri0 = c.tris.len();
        for t in idx.chunks_exact(3) {
            if t.iter().all(|&i| (i as usize) < pos.len()) {
                c.tris.push([base + t[0], base + t[1], base + t[2]]);
                c.tri_surf.push(si);
            }
        }
        tri_src.push((tri0, c.tris.len(), if *hidden { format!("{} [physics only]", inst.path) } else { inst.path.clone() }));
    }
    // EVOVIEW_TRI_SRC="x,z": every collider triangle over that point (XZ),
    // its height there and the mesh it came from
    if let Some((px, pz)) = std::env::var("EVOVIEW_TRI_SRC").ok().and_then(|v| { let (a, b) = v.split_once(',')?; Some((a.trim().parse::<f32>().ok()?, b.trim().parse::<f32>().ok()?)) }) {
        for (ti, t) in c.tris.iter().enumerate() {
            let [a, b, d] = t.map(|i| c.verts[i as usize]);
            let den = (b[2] - d[2]) * (a[0] - d[0]) + (d[0] - b[0]) * (a[2] - d[2]);
            if den.abs() < 1e-9 {
                continue;
            }
            let l1 = ((b[2] - d[2]) * (px - d[0]) + (d[0] - b[0]) * (pz - d[2])) / den;
            let l2 = ((d[2] - a[2]) * (px - d[0]) + (a[0] - d[0]) * (pz - d[2])) / den;
            if l1 < 0.0 || l2 < 0.0 || l1 + l2 > 1.0 {
                continue;
            }
            let y = l1 * a[1] + l2 * b[1] + (1.0 - l1 - l2) * d[1];
            let src = tri_src.iter().find(|r| ti >= r.0 && ti < r.1).map_or("?", |r| r.2.as_str());
            println!("TRISRC ({px}, {pz}) y {y:.3} surf {:?} {src}", c.surfaces.get(c.tri_surf[ti] as usize));
        }
    }
    // walls 1: the near-vertical triangles of the static colliders, as their
    // longest XZ edge with the triangle's height range
    for (ti, t) in c.tris.iter().enumerate() {
        let p = t.map(|i| c.verts[i as usize]);
        let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
        let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
        let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if l < 1e-9 || n[1].abs() / l > 0.35 {
            continue;
        }
        let mut best = (0.0f32, 0usize, 1usize);
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let d = (p[a][0] - p[b][0]).powi(2) + (p[a][2] - p[b][2]).powi(2);
            if d > best.0 {
                best = (d, a, b);
            }
        }
        if best.0 < 0.01 {
            continue;
        }
        let ylo = p[0][1].min(p[1][1]).min(p[2][1]);
        let yhi = p[0][1].max(p[1][1]).max(p[2][1]);
        if yhi - ylo < 0.25 {
            continue;
        }
        let src = format!("collider {} ({})", tri_src.iter().find(|r| ti >= r.0 && ti < r.1).map_or("?", |r| r.2.as_str()),
            c.surfaces.get(c.tri_surf[ti] as usize).map_or("?", |s| s.as_str()));
        if yhi - ylo <= 2.0 {
            c.walls.push([p[best.1][0], p[best.1][2], p[best.2][0], p[best.2][2], ylo, yhi]);
            c.wall_src.push(src);
            c.wall_surf.push(c.tri_surf[ti]);
        } else {
            // A tall triangle may lean up to ~20 deg: one segment along its
            // longest edge for its whole height put the wall far from where
            // the face is at car height (Imola's 60 m 06WALL014 crossed the
            // racing line at Tamburello). Slice it every metre instead:
            // each slice is the face's own line at that height.
            let mut h = ylo + 0.5;
            while h < yhi {
                let mut pts: Vec<[f32; 2]> = Vec::new();
                for (a, b) in [(0usize, 1usize), (1, 2), (2, 0)] {
                    let (ya, yb) = (p[a][1], p[b][1]);
                    if (ya - h) * (yb - h) <= 0.0 && (ya - yb).abs() > 1e-6 {
                        let t = (h - ya) / (yb - ya);
                        pts.push([p[a][0] + (p[b][0] - p[a][0]) * t, p[a][2] + (p[b][2] - p[a][2]) * t]);
                    }
                }
                if pts.len() >= 2 {
                    let (q0, q1) = (pts[0], pts[1]);
                    if (q0[0] - q1[0]).powi(2) + (q0[1] - q1[1]).powi(2) > 1e-4 {
                        c.walls.push([q0[0], q0[1], q1[0], q1[1], (h - 0.5).max(ylo), (h + 0.5).min(yhi)]);
                        c.wall_src.push(src.clone());
                        c.wall_surf.push(c.tri_surf[ti]);
                    }
                }
                h += 1.0;
            }
        }
        if near(p[best.1][0], p[best.1][2]) {
            let src = tri_src.iter().find(|r| ti >= r.0 && ti < r.1).map_or("?", |r| r.2.as_str());
            println!("WALLSRC collider {src} surf {:?} y {ylo:.2}..{yhi:.2}", c.surfaces.get(c.tri_surf[ti] as usize));
        }
    }
    // walls 2: instanced solids (barriers, tyre walls, poles): the outline
    // of each instance's bounding box
    let mut boxes: HashMap<String, Option<([f32; 3], [f32; 3])>> = HashMap::new();
    let n_static_walls = c.walls.len();
    for (inst, hidden) in wall_insts {
        let bb = boxes.entry(inst.path.to_lowercase()).or_insert_with(|| {
            let k = vfs.find(&inst.path, None)?;
            let lod = read_mesh(&vfs.read(&k))?;
            // the footprint is what a car can hit: the vertices in the bottom
            // 1.2 m (a tree's trunk, not its 10 m crown - the crown's box
            // reached over Imola's track edge); the height stays the full one
            let vs: Vec<[f32; 3]> = lod.positions.chunks_exact(12)
                .map(|q| std::array::from_fn(|i| f32::from_le_bytes(q[i * 4..i * 4 + 4].try_into().unwrap())))
                .collect();
            let y0 = vs.iter().map(|v| v[1]).fold(f32::MAX, f32::min);
            // ⚠ a tree's lowest leaves and a person's arms reach into the
            // bottom 1.2 m too: Imola's beech_7 measured 4.8 x 5.9 m there
            // (12 m scaled) against a 0.4 m trunk, and the M4 hit that box
            // at 215 km/h beside Tamburello with nothing in the way
            // (EVOVIEW_COLLIDE_LOG). Plants and people: the bottom 0.3 m
            // (the trunk, the feet). EVOVIEW_TRUNK_WALLS=0: the old box.
            let lpath = inst.path.to_lowercase();
            let band = if (lpath.contains("/vegetation/") || lpath.contains("/characters/"))
                && std::env::var("EVOVIEW_TRUNK_WALLS").as_deref() != Ok("0") { 0.3 } else { 1.2 };
            let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for v in &vs {
                hi[1] = hi[1].max(v[1]);
                lo[1] = lo[1].min(v[1]);
                if v[1] <= y0 + band {
                    for i in [0, 2] {
                        lo[i] = lo[i].min(v[i]);
                        hi[i] = hi[i].max(v[i]);
                    }
                }
            }
            if std::env::var_os("EVOVIEW_WALL_BOX_LOG").is_some() {
                let (mut l3, mut h3) = ([f32::MAX; 2], [f32::MIN; 2]);
                for v in vs.iter().filter(|v| v[1] <= y0 + 0.3) {
                    l3 = [l3[0].min(v[0]), l3[1].min(v[2])];
                    h3 = [h3[0].max(v[0]), h3[1].max(v[2])];
                }
                println!("WALLBOX {} footprint {:.2} x {:.2} (bottom 0.3 m: {:.2} x {:.2}) height {:.2} verts {}", inst.path,
                         hi[0] - lo[0], hi[2] - lo[2], h3[0] - l3[0], h3[1] - l3[1], hi[1] - lo[1], vs.len());
            }
            // flat things (signs lying down, decals) are not walls
            (lo[0] < hi[0] && hi[1] - lo[1] > 0.2).then_some((lo, hi))
        });
        let Some((lo, hi)) = *bb else { continue };
        let m = inst_matrix(inst);
        let tf = |x: f32, y: f32, z: f32| [m[0] * x + m[4] * y + m[8] * z + m[12],
                                           m[1] * x + m[5] * y + m[9] * z + m[13],
                                           m[2] * x + m[6] * y + m[10] * z + m[14]];
        let cs = [tf(lo[0], lo[1], lo[2]), tf(hi[0], lo[1], lo[2]), tf(hi[0], lo[1], hi[2]), tf(lo[0], lo[1], hi[2])];
        let top = tf(0.5 * (lo[0] + hi[0]), hi[1], 0.5 * (lo[2] + hi[2]))[1];
        let ylo = cs.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
        for k in 0..4 {
            let (a, b) = (cs[k], cs[(k + 1) % 4]);
            c.walls.push([a[0], a[2], b[0], b[2], ylo, top.max(ylo + 0.3)]);
            c.wall_src.push(format!("instance {}{}", inst.path, if *hidden { " [physics only]" } else { "" }));
            c.wall_surf.push(u16::MAX);
            if near(a[0], a[2]) {
                println!("WALLSRC instance {} y {ylo:.2}..{top:.2}", inst.path);
            }
        }
    }
    println!("collision: {} triangle(s), {} surface(s) from {} collider(s); {} wall segment(s) ({} from {} solid instance(s)), {:.2}s",
             c.tris.len(), c.surfaces.len(), colliders.len(), c.walls.len(),
             c.walls.len() - n_static_walls, wall_insts.len(), t0.elapsed().as_secs_f32());
    Some(c)
}

/// A scene node's transform as a matrix (inst_matrix's convention).
pub fn trs(pos: [f32; 3], euler: [f32; 3], scale: [f32; 3]) -> glam::Mat4 {
    glam::Mat4::from_cols_array(&inst_matrix(&SceneInst { path: String::new(), pos, euler, scale }))
}

/// `m` back as (pos, euler, scale) for inst_matrix: column lengths are the
/// scale, the rest the rotation (scene_rot, euler_rot transposed).
fn untrs(m: glam::Mat4) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let mut c = [m.x_axis.truncate(), m.y_axis.truncate(), m.z_axis.truncate()];
    let mut k = [c[0].length(), c[1].length(), c[2].length()];
    if c[0].cross(c[1]).dot(c[2]) < 0.0 {
        // mirrored: keep the flip in the scale
        k[0] = -k[0];
    }
    for j in 0..3 {
        if k[j].abs() > 1e-9 {
            c[j] /= k[j];
        }
    }
    // euler_rot's r[i][j] = c[i][j] (the matrix holds its transpose)
    let r = |i: usize, j: usize| c[i][j];
    let (x, y, z);
    let sy = (-r(0, 2)).clamp(-1.0, 1.0);
    if sy.abs() < 0.99999 {
        y = sy.asin();
        x = r(1, 2).atan2(r(2, 2));
        z = r(0, 1).atan2(r(0, 0));
    } else {
        y = sy.asin();
        x = (-r(2, 1)).atan2(r(1, 1));
        z = 0.0;
    }
    (m.w_axis.truncate().into(), [x.to_degrees(), y.to_degrees(), z.to_degrees()], k)
}

/// `inst` placed by its node / container (`node`), when that is not identity.
fn place(mut inst: SceneInst, node: glam::Mat4) -> SceneInst {
    if node.abs_diff_eq(glam::Mat4::IDENTITY, 1e-6) {
        return inst;
    }
    let (p, e, s) = untrs(node * trs(inst.pos, inst.euler, inst.scale));
    inst.pos = p;
    inst.euler = e;
    inst.scale = s;
    inst
}

fn inst_matrix(inst: &SceneInst) -> [f32; 16] {
    let r = scene_rot(inst.euler);
    let s = inst.scale;
    [
        r[0][0] * s[0], r[1][0] * s[0], r[2][0] * s[0], 0.0,
        r[0][1] * s[1], r[1][1] * s[1], r[2][1] * s[1], 0.0,
        r[0][2] * s[2], r[1][2] * s[2], r[2][2] * s[2], 0.0,
        inst.pos[0], inst.pos[1], inst.pos[2], 1.0,
    ]
}

fn mesh_span(pos: &[u8]) -> f32 {
    let n = pos.len() / 12;
    if n < 2 {
        return 0.0;
    }
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    let step = (n / 48).max(1);
    for i in (0..n).step_by(step) {
        let p = f3_at(pos, i);
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let dx = hi[0] - lo[0];
    let dy = hi[1] - lo[1];
    let dz = hi[2] - lo[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn material_is_helper(reference: &str) -> bool {
    let full = reference.replace('\\', "/").to_lowercase();
    let b = full
        .rsplit('/')
        .next()
        .unwrap_or(full.as_str())
        .trim_end_matches(".material");
    full.contains("editor/default")
        || b == "default"
        || b.starts_with("physics")
        || b.contains("physics")
        || b.starts_with("dummy")
        || b.contains("invis")
        || b.starts_with("transparent")
        || b.contains("collision")
}

fn lod_is_helper(lod: &Lod) -> bool {
    !lod.batches.is_empty() && lod.batches.iter().all(|b| material_is_helper(&b.material))
}

fn skip_track_mesh(name: &str, key: &str) -> bool {
    let n = name.to_lowercase();
    n.starts_with("physics")
        || n.starts_with("dummy")
        || n.starts_with("lowres")
        || n.starts_with("shadow")
        || n.starts_with("transparent")
        || n.starts_with("collision")
        || n.starts_with("wall_non_csp")
        || n.starts_with("null")
        || n.contains("collider")
        || n.contains("collision")
        || n.contains("shadow_cache")
        || n.contains("null_wall")
        || n.contains("_collision")
        || key.contains("irradiance")
        || key.contains("/physics")
}

fn parse_vec3(d: &[u8], s: usize, e: usize) -> [f32; 3] {
    let mut v = [0.0f32; 3];
    fields(d, s, e, |f, val| {
        let x = val.f32();
        match f {
            1 => v[0] = x,
            2 => v[1] = x,
            3 => v[2] = x,
            _ => {}
        }
    });
    v
}

pub(crate) fn parse_transform(d: &[u8], s: usize, e: usize) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let mut pos = [0.0f32; 3];
    let mut rot = [0.0f32; 3];
    let mut scl = [0.0f32; 3];
    fields(d, s, e, |f, v| {
        let Some((a, b)) = v.range() else { return };
        match f {
            1 => pos = parse_vec3(d, a, b),
            2 => rot = parse_vec3(d, a, b),
            3 => scl = parse_vec3(d, a, b),
            _ => {}
        }
    });
    if scl[0] == 0.0 && scl[1] == 0.0 && scl[2] == 0.0 {
        scl = [1.0, 1.0, 1.0];
    }
    (pos, rot, scl)
}

fn xform_is_identity(pos: [f32; 3], euler: [f32; 3], scale: [f32; 3]) -> bool {
    pos.iter().all(|c| c.abs() < 0.001)
        && euler.iter().all(|c| c.abs() < 0.05)
        && scale.iter().all(|c| (c - 1.0).abs() < 0.001)
}

fn skip_logic_scene(path: &str) -> bool {
    let l = path.to_lowercase();
    l.contains("spawnpoint")
        || l.contains("timeline")
        || l.contains("camera_sequence")
        || l.contains("common_car_cam")
        || l.contains("tv1_camera")
        || l.contains("tv2_camera")
        || l.contains("tv_camera")
        || l.contains("/fmod_")
        || l.contains("\\fmod_")
        || l.contains("/dlp_")
        || l.contains("physics_20")
        || l.contains("drs_zone")
        // Driving Academy events (academy_event_NN, common atob_ / ta_ start
        // and end gates): the game loads them only for that event - left in,
        // their inflatable arch stands on Imola's racing line, collider and all.
        // EVOVIEW_ACADEMY=1 shows them.
        || ((l.contains("academy_event") || l.contains("academy_events")) && std::env::var_os("EVOVIEW_ACADEMY").is_none())
}

fn scene_basename(path: &str) -> String {
    path.replace('\\', "/")
        .rsplit('/')
        .next()
        .unwrap_or(path)
        .trim_end_matches(".scene")
        .to_lowercase()
}

/// Adenau / boulevard / industrial / open parks — the Eifel world around
/// the Ring. Official Free Roam points at a missing `eifel` folder; these
/// containers are what actually shipped.
fn is_world_container(path: &str) -> bool {
    let n = scene_basename(path);
    n.contains("boulevard")
        || n.contains("adenau")
        || n.contains("industrial")
        || n.contains("bistrot")
        || n.contains("people_")
        || n.contains("npc_")
        || n.contains("temp_people")
        || n.contains("spawns_ow")
        || n.contains("park_entrance_open")
        || n.contains("park_nordschleife_open")
        || n.contains("park_new_exit_open")
        || n.contains("cruise")
        || n.contains("tourist")
}

fn skip_freeroam_container(path: &str) -> bool {
    is_world_container(path)
}

fn layout_key(path: &str) -> Option<String> {
    scene_basename(path)
        .strip_prefix("layout_")
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
}

/// The layout the last track load chose (pick_race_layout), so the racing
/// line and spawn follow the scenery that was loaded (trackdrive.rs).
static LOADED_LAYOUT: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Online (mp.rs): the containers the server's session runs, which then ARE
/// the layout's set (lowercase '/' paths) instead of track_containers.table.
pub static SESSION_CONTAINERS: std::sync::RwLock<Vec<String>> = std::sync::RwLock::new(Vec::new());

pub fn loaded_layout() -> Option<String> {
    LOADED_LAYOUT.lock().ok().and_then(|g| g.clone())
}

fn pick_race_layout(containers: &[(String, bool)]) -> Option<String> {
    let lay = pick_race_layout_inner(containers);
    if lay.is_some() {
        if let Ok(mut g) = LOADED_LAYOUT.lock() {
            *g = lay.clone();
        }
    }
    lay
}

fn pick_race_layout_inner(containers: &[(String, bool)]) -> Option<String> {
    let mut keys: Vec<String> = containers
        .iter()
        .filter_map(|(p, _)| layout_key(p))
        .collect();
    keys.sort();
    keys.dedup();
    // --layout <id> / EVOVIEW_LAYOUT=<id> (the ACECM Drive tab sends the
    // racing-line id, layout_<id>.ideal_line): that layout's containers
    if let Ok(want) = std::env::var("EVOVIEW_LAYOUT") {
        let want = want.to_lowercase();
        if keys.iter().any(|k| *k == want) {
            return Some(want);
        }
        if !keys.is_empty() {
            println!("layout {want:?} not in this scene ({}); the default instead", keys.join(", "));
        }
    }
    for pref in ["nordschleife", "24h", "gp_strecke", "gp", "sprint"] {
        if keys.iter().any(|k| k == pref) {
            return Some(pref.to_string());
        }
    }
    keys.into_iter().next()
}

/// The game's per-layout container set: system/track_containers.table, f2
/// -> f3 -> f8 {1 mode name, 10 track name, 11 container path (repeated),
/// 14 layout name ("Nordschleife", "Gp Strecke"), 15 = 1 the layout's base
/// "No Game Mode" set} - plus its Hotstint mode's set. The base set says which variant containers the
/// layout switches on: the Nordschleife wants t13_on, nords_connected_off
/// and the closed parks; the GP-Strecke mercedes_arena_on and so on.
/// Lowercase '/' paths; empty when the table does not know the layout.
fn layout_table_containers(vfs: &mut Vfs, layout: &str) -> Vec<String> {
    let d = vfs.read("system/track_containers.table");
    let squash = |s: &str| s.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_lowercase();
    let want = squash(layout);
    let mut root = vfs.root().to_string();
    // a loose mod track's folder has no root: every track's "gp" matched (174
    // containers of other tracks, not one of its own). Its id is its root
    // scene's name (ks_brands_bobr/ks_brands_bobr.scene).
    if root.is_empty() {
        let id = vfs.own_keys("scene").into_iter().find(|k| !k.contains('/'))
            .map(|k| k.trim_end_matches(".scene").to_lowercase());
        match id {
            Some(id) => root = format!("content/tracks/{id}/"),
            None => return Vec::new(),
        }
    }
    let mut out: Vec<String> = Vec::new();
    fields(&d, 0, d.len(), |f, v| {
        let (2, Val::Bytes(a, b)) = (f, v) else { return };
        fields(&d, a, b, |f, v| {
            let (3, Val::Bytes(a, b)) = (f, v) else { return };
            fields(&d, a, b, |f, v| {
                let (8, Val::Bytes(a, b)) = (f, v) else { return };
                let (mut name, mut base, mut paths) = (String::new(), false, Vec::new());
                let mut mode = String::new();
                fields(&d, a, b, |f, v| match (f, v) {
                    (1, Val::Bytes(a, b)) => mode = text(&d, (a, b)).to_lowercase(),
                    (11, Val::Bytes(a, b)) => paths.push(text(&d, (a, b)).replace('\\', "/").to_lowercase()),
                    (14, Val::Bytes(a, b)) => name = text(&d, (a, b)),
                    (15, Val::Varint(n)) => base = n != 0,
                    _ => {}
                });
                // plus the hotlap mode's set (what a lone car drives): its
                // gates and parks closed, the hotlap spawn
                let base = base || mode.contains("hotstint");
                if base && squash(&name) == want && paths.iter().any(|p| p.starts_with(&root)) {
                    out.extend(paths);
                }
            });
        });
    });
    out.sort();
    out.dedup();
    out
}

/// A variant container (x_on / x_off, x_open / x_closed) whose sibling the
/// layout's table set picks instead.
fn variant_switched_off(path: &str, set: &[String]) -> bool {
    let p = path.to_lowercase().replace('\\', "/");
    if set.iter().any(|s| *s == p) {
        return false;
    }
    let Some(stem) = p.strip_suffix(".scene") else { return false };
    for (a, b) in [("_on", "_off"), ("_off", "_on"), ("_open", "_closed"), ("_closed", "_open")] {
        if let Some(head) = stem.strip_suffix(a) {
            if set.iter().any(|s| *s == format!("{head}{b}.scene")) {
                return true;
            }
        }
    }
    false
}

fn container_for_layout(path: &str, layout: Option<&str>) -> bool {
    let Some(key) = layout else {
        return false;
    };
    let n = scene_basename(path);
    if n.contains(key) {
        return true;
    }
    if key == "nordschleife" {
        if n.contains("nords_connected") || n.contains("tourist") {
            return false;
        }
        return n.contains("nords")
            || n.contains("bridges_nords")
            || n.contains("grail_park");
    }
    false
}

struct SceneGraph {
    /// each container's placement (parallel to `containers`)
    container_xf: Vec<glam::Mat4>,
    /// static meshes with a surface_id (drawn or physics_only): what a car
    /// drives on (trackdrive.rs)
    /// (.., surface_id, physics_only: not drawn)
    colliders: Vec<(SceneInst, String, bool)>,
    /// instanced meshes with a surface_id or a collider (barriers, tyre
    /// walls, poles): become wall outlines (trackdrive.rs)
    wall_insts: Vec<(SceneInst, bool)>,
    statics: Vec<SceneInst>,
    instances: Vec<SceneInst>,
    /// (scene_path, loaded_on_demand)
    containers: Vec<(String, bool)>,
    terrains: Vec<String>,
    physics_skip: usize,
}

/// SceneData.scene_actors (field 2) → SceneActorData.actor (50) →
/// StaticMeshData (100) / InstancedStaticMeshData (105) / Container (121) /
/// TerrainActorDataNew (134). physics_only actors are not drawn.
/// `parent`: where the container that loaded this scene put it - every
/// node in a container is local to it (TV2 cameras, academy cones), and an
/// instance set is local to its own node (Imola's +101 m vegetation and
/// prop groups sat 101 m underground, a Ferris wheel at the origin).
fn parse_scene_graph(d: &[u8], parent: glam::Mat4) -> SceneGraph {
    let mut g = SceneGraph {
        container_xf: Vec::new(),
        colliders: Vec::new(),
        wall_insts: Vec::new(),
        statics: Vec::new(),
        instances: Vec::new(),
        containers: Vec::new(),
        terrains: Vec::new(),
        physics_skip: 0,
    };
    fields(d, 0, d.len(), |f, v| {
        if f != 2 {
            return;
        }
        let Some((s, e)) = v.range() else { return };
        let mut enabled = true;
        let mut xf = ([0.0f32; 3], [0.0f32; 3], [1.0f32; 3]);
        let mut actor: Option<(usize, usize)> = None;
        fields(d, s, e, |sf, sv| match (sf, sv) {
            (2, Val::Varint(n)) => enabled = n != 0,
            (4, Val::Bytes(a, b)) => xf = parse_transform(d, a, b),
            (50, Val::Bytes(a, b)) => actor = Some((a, b)),
            _ => {}
        });
        if !enabled {
            return;
        }
        let Some((as_, ae)) = actor else { return };
        // A node turns by the same rotation as an instance (scene_rot). It
        // was negated while instances were mirrored (see scene_rot): on the
        // Nordschleife a node yawed -2.9 deg carries 4969 card trees, 240k
        // grass tufts and 445 props over kilometres, and the wrong sense
        // swung them onto other terrain - a forest floating up to 89 m in
        // the sky (EVOVIEW_TREE_CHECK). EVOVIEW_NODE_ROT=neg / none for
        // comparisons.
        let node = parent * trs(xf.0, match std::env::var("EVOVIEW_NODE_ROT").as_deref() {
            Ok("neg") => [-xf.1[0], -xf.1[1], -xf.1[2]],
            Ok("none") => [0.0; 3],
            _ => xf.1,
        }, xf.2);
        let at = |pos: [f32; 3], euler: [f32; 3], scale: [f32; 3]| -> ([f32; 3], [f32; 3], [f32; 3]) {
            if parent.abs_diff_eq(glam::Mat4::IDENTITY, 1e-6) {
                (pos, euler, scale)
            } else {
                untrs(parent * trs(pos, euler, scale))
            }
        };
        // A node's own static mesh (and its collider) turns with the node.
        // Checked on Imola: the trackside big screens face the track and the
        // pit-out board the pit lane (side-on when mirrored).
        // EVOVIEW_STATIC_ROT=neg for comparisons.
        let xf = if std::env::var("EVOVIEW_STATIC_ROT").as_deref() == Ok("neg") {
            at(xf.0, [-xf.1[0], -xf.1[1], -xf.1[2]], xf.2)
        } else {
            at(xf.0, xf.1, xf.2)
        };
        if std::env::var_os("EVOVIEW_STATIC_ROT_LOG").is_some() && xf.1.iter().any(|a| a.abs() > 0.5) {
            fields(d, as_, ae, |kf, kv| if kf == 100 {
                if let Some((hs, he)) = kv.range() {
                    fields(d, hs, he, |mf, mv| if let (1, Val::Bytes(a, b)) = (mf, mv) {
                        println!("STATROT {:?} {:?} {}", xf.0.map(|v| v.round()), xf.1.map(|v| (v * 10.0).round() / 10.0), text(d, (a, b)));
                    });
                }
            });
        }
        fields(d, as_, ae, |kf, kv| {
            let Some((hs, he)) = kv.range() else { return };
            match kf {
                100 => {
                    let mut path = String::new();
                    let mut physics_only = false;
                    let mut surface = String::new();
                    fields(d, hs, he, |mf, mv| match (mf, mv) {
                        (1, Val::Bytes(a, b)) => path = text(d, (a, b)),
                        (2, Val::Bytes(a, b)) => surface = text(d, (a, b)),
                        (3, Val::Varint(n)) => physics_only = n != 0,
                        _ => {}
                    });
                    // every static mesh is ground; untagged ones (terrain tiles,
                    // verges) get the fallback surface "" (trackdrive.rs)
                    if path.to_lowercase().ends_with(".mesh") {
                        g.colliders.push((SceneInst {
                            path: path.replace('\\', "/"),
                            pos: xf.0,
                            euler: xf.1,
                            scale: xf.2,
                        }, surface, physics_only));
                    }
                    if physics_only {
                        g.physics_skip += 1;
                        return;
                    }
                    if path.to_lowercase().ends_with(".mesh") {
                        g.statics.push(SceneInst {
                            path: path.replace('\\', "/"),
                            pos: xf.0,
                            euler: xf.1,
                            scale: xf.2,
                        });
                    }
                }
                105 => {
                    let mut path = String::new();
                    let mut physics_only = false;
                    let mut solid = false;
                    let mut pos_b = None;
                    let mut rot_b = None;
                    let mut scl_b = None;
                    fields(d, hs, he, |mf, mv| match (mf, mv) {
                        (1, Val::Bytes(a, b)) => path = text(d, (a, b)),
                        // surface_id (f3) or a collider (f10): it is solid
                        (3, Val::Bytes(a, b)) => {
                            solid |= b > a;
                            if std::env::var_os("EVOVIEW_COLLIDER_DUMP").is_some() {
                                println!("SURFID {path} {:?}", String::from_utf8_lossy(&d[a..b]));
                            }
                        }
                        (10, Val::Bytes(a, b)) => {
                            solid = true;
                            if std::env::var_os("EVOVIEW_COLLIDER_DUMP").is_some() {
                                let mut o = String::new();
                                fields(d, a, b, |g, w| o += &match w { Val::Bytes(x, y) => format!(" {g}=[{}B {:?}]", y - x, String::from_utf8_lossy(&d[x..y.min(x + 60)])), v => format!(" {g}={}", v.f32()) });
                                println!("COLLIDER {path} {o}");
                            }
                        }
                        (4, Val::Varint(n)) => physics_only = n != 0,
                        (7, Val::Bytes(a, b)) => {
                            fields(d, a, b, |tf, tv| {
                                if let Some(r) = tv.range() {
                                    match tf {
                                        2 => pos_b = Some(r),
                                        3 => rot_b = Some(r),
                                        4 => scl_b = Some(r),
                                        _ => {}
                                    }
                                }
                            });
                        }
                        _ => {}
                    });
                    // ground cover is never a wall, whatever the scene says
                    // (converted tracks flag 120k grass tufts solid)
                    // nor are people: race_scenery_gp stands a crowd of
                    // solid-flagged spectators on the GP-Strecke just after
                    // T1, and the M4 stopped dead in it at 108 km/h.
                    // EVOVIEW_CHARACTER_WALLS=1: they are walls again.
                    let soft = { let l = path.to_lowercase().replace('\\', "/"); l.contains("/vegetation/low/") || l.contains("/grass/")
                        || (l.contains("/characters/") && std::env::var_os("EVOVIEW_CHARACTER_WALLS").is_none()) };
                    if solid && !soft && path.to_lowercase().ends_with(".mesh") {
                        if let Some((ps, pe)) = pos_b {
                            for i in 0..(pe - ps) / 12 {
                                let sc = scl_b.filter(|r| r.1 > r.0).map(|r| f3_at(&d[r.0..r.1], i)).unwrap_or([1.0; 3]);
                                g.wall_insts.push((place(SceneInst {
                                    path: path.replace('\\', "/"),
                                    pos: f3_at(&d[ps..pe], i),
                                    euler: rot_b.filter(|r| r.1 > r.0).map(|r| f3_at(&d[r.0..r.1], i)).unwrap_or([0.0; 3]),
                                    scale: if sc == [0.0; 3] { [1.0; 3] } else { sc },
                                }, node), physics_only));
                            }
                        }
                    }
                    if physics_only {
                        g.physics_skip += 1;
                        return;
                    }
                    let Some((ps, pe)) = pos_b else { return };
                    let n = (pe - ps) / 12;
                    if n == 0 || !path.to_lowercase().ends_with(".mesh") {
                        return;
                    }
                    let rot = rot_b.unwrap_or((0, 0));
                    let scl = scl_b.unwrap_or((0, 0));
                    if std::env::var_os("EVOVIEW_NODE_LOG").is_some() {
                        let (s, r, tr) = node.to_scale_rotation_translation();
                        let first = f3_at(&d[ps..pe], 0);
                        println!("NODESET {path} n={n} node t=({:.1},{:.1},{:.1}) s=({:.2},{:.2},{:.2}) r={:.3?} parent_t=({:.1},{:.1},{:.1}) first=({:.1},{:.1},{:.1})",
                                 tr.x, tr.y, tr.z, s.x, s.y, s.z, r, parent.w_axis.x, parent.w_axis.y, parent.w_axis.z, first[0], first[1], first[2]);
                    }
                    let path = path.replace('\\', "/");
                    for i in 0..n {
                        let mut inst = SceneInst {
                            path: path.clone(),
                            pos: f3_at(&d[ps..pe], i),
                            euler: if rot.1 > rot.0 {
                                f3_at(&d[rot.0..rot.1], i)
                            } else {
                                [0.0; 3]
                            },
                            scale: if scl.1 > scl.0 {
                                f3_at(&d[scl.0..scl.1], i)
                            } else {
                                [1.0; 3]
                            },
                        };
                        if inst.scale[0] == 0.0 && inst.scale[1] == 0.0 && inst.scale[2] == 0.0 {
                            inst.scale = [1.0; 3];
                        }
                        if let Ok(f) = std::env::var("EVOVIEW_INST_LOG") {
                            if path.contains(f.as_str()) {
                                let (_, r, tr) = node.to_scale_rotation_translation();
                                println!("INST {path} pos={:.2?} euler={:.2?} scale={:.2?} node_t=({:.1},{:.1},{:.1}) node_r={:.3?}",
                                         inst.pos, inst.euler, inst.scale, tr.x, tr.y, tr.z, r);
                            }
                        }
                        g.instances.push(place(inst, node));
                    }
                }
                121 => {
                    let mut path = String::new();
                    let mut on_demand = false;
                    fields(d, hs, he, |mf, mv| match (mf, mv) {
                        (1, Val::Bytes(a, b)) => path = text(d, (a, b)),
                        (3, Val::Varint(n)) => on_demand = n != 0,
                        _ => {}
                    });
                    if !path.is_empty() {
                        g.containers.push((path.replace('\\', "/"), on_demand));
                        g.container_xf.push(node);
                    }
                }
                134 => {
                    fields(d, hs, he, |mf, mv| {
                        if mf == 1 {
                            if let Some(r) = mv.range() {
                                g.terrains.push(text(d, r).replace('\\', "/"));
                            }
                        }
                    });
                }
                _ => {}
            }
        });
    });
    g
}

/// TerrainDataNew.render_settings (5) → world_uv_scale (5).
fn parse_world_uv_scale(d: &[u8]) -> f32 {
    let mut scale = 0.0f32;
    fields(d, 0, d.len(), |f, v| {
        if f != 5 {
            return;
        }
        let Some((s, e)) = v.range() else { return };
        fields(d, s, e, |sf, sv| {
            if sf == 5 {
                let x = sv.f32();
                if x > 0.0 {
                    scale = x;
                }
            }
        });
    });
    scale
}

fn xform_lod(lod: &Lod, inst: &SceneInst) -> Lod {
    let r = scene_rot(inst.euler);
    let nvert = lod.positions.len() / 12;
    let mut positions = lod.positions.clone();
    let mut normals = lod.normals.clone();
    for i in 0..nvert {
        let p = f3_at(&lod.positions, i);
        let s = [
            p[0] * inst.scale[0],
            p[1] * inst.scale[1],
            p[2] * inst.scale[2],
        ];
        let w = mat3_vec(r, s);
        write_f3(
            &mut positions,
            i,
            [w[0] + inst.pos[0], w[1] + inst.pos[1], w[2] + inst.pos[2]],
        );
        if lod.normals.len() >= (i + 1) * 12 {
            write_f3(&mut normals, i, mat3_vec(r, f3_at(&lod.normals, i)));
        }
    }
    Lod {
        positions,
        normals,
        uvs: lod.uvs.clone(),
        tangents: lod.tangents.clone(),
        bone_idx: Vec::new(),
        bone_wgt: Vec::new(),
        indices: lod.indices.clone(),
        batches: lod.batches.clone(),
        bones: Vec::new(),
        facing: lod.facing,
        colors: lod.colors.clone(),
        uv1: lod.uv1.clone(),
        density: lod.density,
        dlt: lod.dlt,
        anim: String::new(),    cast_shadows: lod.cast_shadows,
        shadow_lod: lod.shadow_lod,
        shadow_proxy: lod.shadow_proxy,
    }
}

/// A per-vertex float stream of `n` components, or a constant fill when the
/// mesh does not ship one (or ships the wrong length).
fn track_vertex_stream(raw: &[u8], verts: usize, n: usize, fill: f32) -> Vec<u8> {
    if raw.len() == verts * n * 4 {
        return raw.to_vec();
    }
    let mut out = Vec::with_capacity(verts * n * 4);
    for _ in 0..verts * n {
        out.extend_from_slice(&fill.to_le_bytes());
    }
    out
}

fn is_eifel_world_folder(s: &str) -> bool {
    let l = s.to_lowercase();
    l.contains("nurburgring") || l.contains("eifel")
}

/// Static-mesh track. Same GPU-shaped `Car` as the car loader, no skeleton.
pub fn load_track(vfs: &mut Vfs) -> Result<Loaded, String> {
    let infer = is_eifel_world_folder(&vfs.root);
    load_track_opts(vfs, infer)
}

pub fn load_track_opts(vfs: &mut Vfs, include_world: bool) -> Result<Loaded, String> {
    // per-section load timings for --bench (track load, phase by phase)
    let mut lap_t = std::time::Instant::now();
    let mut lap = |name: &'static str| {
        crate::perf::add(name, lap_t.elapsed().as_secs_f64());
        lap_t = std::time::Instant::now();
    };
    let mut blob: Vec<u8> = Vec::new();
    let mut meshes: Vec<MeshEntry> = Vec::new();
    let mut textures: Vec<TextureEntry> = Vec::new();
    let mut materials: Vec<MaterialEntry> = Vec::new();
    let mut tex_index: HashMap<String, usize> = HashMap::new();

    let add = |blob: &mut Vec<u8>, data: &[u8]| -> [u64; 2] {
        let off = blob.len() as u64;
        blob.extend_from_slice(data);
        while blob.len() % 4 != 0 {
            blob.push(0);
        }
        [off, data.len() as u64]
    };

    let scene_size = |vfs: &Vfs, key: &str| -> u64 {
        match vfs.entries.get(key) {
            Some(Src::Pkg(e, _)) => e.size as u64,
            Some(Src::File(p)) => std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
            Some(Src::Bytes(b)) => b.len() as u64,
            None => 0,
        }
    };
    let scene_too_heavy = |key: &str, sz: u64| -> bool {
        let low = key.to_lowercase();
        // The root .scene is a catalog (Nürburg 54 MB). Never skip it —
        // that used to dump every mesh by file size and show the Eifel
        // freeroam village instead of the circuit.
        if !low.contains("/containers/") && !low.contains("/content/") {
            return false;
        }
        if low.contains("_base.scene") {
            return false;
        }
        let veg = low.contains("grass")
            || low.contains("flower")
            || low.contains("clover")
            || low.contains("vegetation")
            || low.contains("ground_grass")
            || low.contains("wild_");
        // EVOVIEW_LOAD_VEG=1: load the big ground-vegetation scenes too
        (veg && sz > 16_000_000 && std::env::var_os("EVOVIEW_LOAD_VEG").is_none())
            || ((low.contains("/misc_") || low.contains("do_not_use")) && sz > 4_000_000)
    };

    let mesh_count = vfs.own_keys("mesh").len();
    let huge_track = mesh_count >= 1200;
    // ⚠ This cap DROPS geometry, keeping only the largest meshes by file
    // size - so a city track kept its roads and terrain and lost thousands of
    // individual buildings. Shutoko declares 4429 static meshes and stopped at
    // exactly 2203, which is what "the viewer is missing things that are in
    // the real map" turned out to be.
    //
    // It exists to stop a pathological folder eating all memory, so it stays -
    // just high enough to fit a real city. Shutoko renders 13M triangles at
    // ~105 fps, so there is headroom.
    let max_meshes: usize = if huge_track { 16000 } else { 4000 };

    // Root .scene plus ContainerActorData.scene_path. Spawn / timeline /
    // camera sequences are logic; layout + ground_grass are real content.
    lap("track: setup");
    let all_scenes = vfs.own_keys("scene");
    let mut queue: Vec<(String, glam::Mat4)> = all_scenes
        .iter()
        .filter(|k| !k.contains("/containers/"))
        .map(|k| (k.clone(), glam::Mat4::IDENTITY))
        .collect();
    if queue.is_empty() {
        queue = all_scenes
            .iter()
            .filter(|k| !skip_logic_scene(k))
            .map(|k| (k.clone(), glam::Mat4::IDENTITY))
            .collect();
    }
    let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut statics: Vec<SceneInst> = Vec::new();
    let mut colliders: Vec<(SceneInst, String, bool)> = Vec::new();
    let mut wall_insts: Vec<(SceneInst, bool)> = Vec::new();
    // (mesh index, its bones, clip path) of every animated instanced level
    let mut anim_meshes: Vec<(usize, Vec<Bone>, String)> = Vec::new();
    let mut insts: Vec<SceneInst> = Vec::new();
    let mut terrain_paths: Vec<String> = Vec::new();
    let mut physics_skip = 0usize;
    let mut chosen_layout: Option<String> = None;
    let mut layout_set: Vec<String> = Vec::new();
    // the track's layout names (layout_<x>.scene containers)
    let mut layout_names: Vec<String> = Vec::new();
    while let Some((s, place_xf)) = queue.pop() {
        let key = vfs.find(&s, None).unwrap_or(s.clone());
        let low = key.to_lowercase();
        if !visited.insert(low.clone()) {
            continue;
        }
        // the layout's physics set (the Nordschleife's tagged tarmac lives in
        // physics_2024) loads; other physics_20xx stay out
        let layout_physics = low.contains("physics_20") && layout_set.iter().any(|p| *p == low);
        if skip_logic_scene(&low) && !layout_physics {
            continue;
        }
        let sz = scene_size(vfs, &key);
        if scene_too_heavy(&key, sz) {
            println!("skip heavy scene {key} ({:.1} MB)", sz as f64 / 1e6);
            continue;
        }
        let bytes = vfs.read(&key);
        if bytes.is_empty() {
            continue;
        }
        let g = parse_scene_graph(&bytes, place_xf);
        physics_skip += g.physics_skip;
        for (c, _) in &g.containers {
            if let Some(k) = layout_key(c) {
                if !layout_names.contains(&k) {
                    layout_names.push(k);
                }
            }
        }
        if chosen_layout.is_none() {
            chosen_layout = pick_race_layout(&g.containers);
            if let Some(ref lay) = chosen_layout {
                // EVOVIEW_LAYOUT_TABLE=0: the name heuristics alone, as before
                let session = SESSION_CONTAINERS.read().unwrap().clone();
                if !session.is_empty() {
                    layout_set = session.iter().map(|c| c.replace('\\', "/").to_lowercase()).collect();
                    println!("layout {lay}: {} container(s) from the server's session", layout_set.len());
                } else if std::env::var("EVOVIEW_LAYOUT_TABLE").map_or(true, |v| v != "0") {
                    layout_set = layout_table_containers(vfs, lay);
                    if !layout_set.is_empty() {
                        println!("layout {lay}: {} container(s) from system/track_containers.table", layout_set.len());
                    }
                }
                println!(
                    "layout: {lay}{}",
                    if include_world { " + eifel world" } else { "" }
                );
            } else if include_world {
                println!("layout: eifel world (shipped freeroam kit)");
            }
        }
        if !g.statics.is_empty() || !g.instances.is_empty() || !g.containers.is_empty() {
            println!(
                "scene {}: {} static, {} inst, {} physics_only, {} container(s)",
                key,
                g.statics.len(),
                g.instances.len(),
                g.physics_skip,
                g.containers.len()
            );
        }
        statics.extend(g.statics);
        colliders.extend(g.colliders);
        wall_insts.extend(g.wall_insts);
        insts.extend(g.instances);
        terrain_paths.extend(g.terrains);
        for ((c, on_demand), cxf) in g.containers.into_iter().zip(g.container_xf) {
            // the layout's own set (track_containers.table) first: its
            // variants win over the world's open parks and the name rules
            let in_set = layout_set.iter().any(|p| *p == c.to_lowercase());
            if skip_logic_scene(&c) && !(in_set && c.to_lowercase().contains("physics_20")) {
                continue;
            }
            let world_bit = include_world && is_world_container(&c);
            if !include_world && skip_freeroam_container(&c) && !in_set {
                continue;
            }
            // Open parks vs closed race gates: pick open when loading world.
            let n = scene_basename(&c);
            if include_world && n.contains("park") && n.contains("closed") && !in_set {
                continue;
            }
            if !layout_set.is_empty() && variant_switched_off(&c, &layout_set) {
                continue;
            }
            // another layout's own scenery: the Nordschleife race layout's
            // table leaves out nords_exit_touristenfahrten, whose wall closes
            // the race line at the Touristenfahrten exit (the M4 hit it at
            // 255 km/h); always-on containers named for another layout of
            // this track stay out unless the layout's table lists them
            if !layout_set.is_empty() && !in_set {
                let n = scene_basename(&c);
                let ours = chosen_layout.as_deref().unwrap_or("");
                // by the other layout's own words (5+ letters, not in ours):
                // nordschleife_touristenfahrten -> touristenfahrten
                let theirs = layout_names.iter().filter(|l| l.as_str() != ours).flat_map(|l| l.split('_')).find(|w| w.len() >= 5 && !ours.split('_').any(|o| o == *w) && n.contains(*w));
                if let Some(w) = theirs {
                    println!("layout {ours}: {c} left out (another layout's: {w}; not in the table)");
                    continue;
                }
            }
            if on_demand && !in_set && !world_bit && !container_for_layout(&c, chosen_layout.as_deref()) {
                continue;
            }
            queue.push((c, cxf));
        }
    }

    let mut world_uv_scale = 0.0f32;
    for t in &terrain_paths {
        if let Some(k) = vfs.find(t, None) {
            let s = parse_world_uv_scale(&vfs.read(&k));
            if s > 0.0 {
                world_uv_scale = s;
            }
        }
    }
    for ext in ["terrain", "terrainnew"] {
        for k in vfs.own_keys(ext) {
            let s = parse_world_uv_scale(&vfs.read(&k));
            if s > 0.0 {
                world_uv_scale = s;
            }
        }
    }
    if world_uv_scale > 0.0 {
        println!("terrain world_uv_scale = {world_uv_scale}");
    }

    let mut mesh_files: Vec<String> = Vec::new();
    let mut seen_mesh: std::collections::HashSet<String> = std::collections::HashSet::new();
    for inst in statics.iter().chain(insts.iter()) {
        let p = inst.path.to_lowercase();
        if seen_mesh.insert(p.clone()) {
            mesh_files.push(inst.path.clone());
        }
    }
    // Scene listed nothing drawable — last resort is every mesh in the folder.
    if statics.is_empty() && insts.is_empty() {
        mesh_files = vfs.own_keys("mesh");
        mesh_files.retain(|k| !skip_track_mesh(&base_of(k), k));
    }
    if mesh_files.len() > max_meshes {
        let mut sized: Vec<(u32, String)> = mesh_files
            .iter()
            .map(|k| {
                let resolved = vfs.find(k, None).unwrap_or_else(|| k.clone());
                let n = match vfs.entries.get(&resolved) {
                    Some(Src::Pkg(e, _)) => e.size,
                    Some(Src::File(p)) => {
                        std::fs::metadata(p).map(|m| m.len() as u32).unwrap_or(0)
                    }
                    Some(Src::Bytes(b)) => b.len() as u32,
                    None => 0,
                };
                (n, k.clone())
            })
            .collect();
        sized.sort_by(|a, b| b.0.cmp(&a.0));
        sized.truncate(max_meshes);
        mesh_files = sized.into_iter().map(|(_, k)| k).collect();
    }

    let mut static_by: HashMap<String, Vec<SceneInst>> = HashMap::new();
    for i in statics {
        static_by.entry(i.path.to_lowercase()).or_default().push(i);
    }

    for path in &mesh_files {
        if !insts.is_empty() && !static_by.contains_key(&path.to_lowercase()) {
            // instance-only paths are uploaded in the instance pass
            if static_by.is_empty() && insts.iter().any(|i| i.path.eq_ignore_ascii_case(path)) {
                continue;
            }
        }
        if !static_by.is_empty() && !static_by.contains_key(&path.to_lowercase()) {
            continue;
        }
        let Some(key) = vfs.find(path, None).or_else(|| {
            if vfs.exists(path) {
                Some(path.clone())
            } else {
                None
            }
        }) else {
            continue;
        };
        if skip_track_mesh(&base_of(&key), &key) {
            continue;
        }
        let bytes = vfs.read(&key);
        let mut lod = match read_mesh(&bytes) {
            Some(l) => l,
            None => continue,
        };
        let dbg = std::env::var("EVOVIEW_MESH_LOG").is_ok();
        if lod.positions.is_empty() || lod.indices.is_empty() {
            if dbg { println!("MESHSKIP empty {path}"); }
            continue;
        }
        lod.batches.retain(|b| !material_is_helper(&b.material));
        if lod.batches.is_empty() || lod_is_helper(&lod) {
            if dbg { println!("MESHSKIP helper {path}"); }
            continue;
        }
        if dbg { println!("MESHOK {} tris={} span={:.0} {path}",
                          lod.batches.len(), lod.indices.len() / 3,
                          mesh_span(&lod.positions)); }
        let set = static_by.get(&path.to_lowercase());
        let xforms = match set {
            Some(set)
                if !set
                    .iter()
                    .all(|i| xform_is_identity(i.pos, i.euler, i.scale)) =>
            {
                set.iter().map(inst_matrix).collect()
            }
            _ => Vec::new(),
        };
        let verts = lod.positions.len() / 12;
        // EVOVIEW_MESH_WHERE: each static mesh's world centre (finding a
        // landmark to aim a comparison camera at)
        if std::env::var_os("EVOVIEW_MESH_WHERE").is_some() {
            let ps: &[f32] = bytemuck::cast_slice(&lod.positions);
            let (mut lo, mut hi) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
            for v in ps.chunks_exact(3) {
                let v = glam::Vec3::new(v[0], v[1], v[2]);
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let m = xforms.first().map_or(glam::Mat4::IDENTITY, |m: &[f32; 16]| glam::Mat4::from_cols_array(m));
            let c = m.transform_point3((lo + hi) * 0.5);
            let mats: Vec<&str> = lod.batches.iter().map(|b| b.material.as_str()).collect();
            println!("WHERE {:.1} {:.1} {:.1} size {:.0} x{} {} {}", c.x, c.y, c.z, (hi - lo).length(), xforms.len().max(1), base_of(&key), mats.join(","));
        }
        place_dump(&key, &lod.positions, &xforms);
        meshes.push(MeshEntry {
            name: base_of(&key),
            vertices: verts as u32,
            batches: lod.batches.clone(),
            positions: Some(add(&mut blob, &lod.positions)),
            normals: Some(add(&mut blob, &lod.normals)),
            uvs: Some(add(&mut blob, &lod.uvs)),
            tangents: Some(add(&mut blob, &lod.tangents)),
            bone_idx: None,
            bone_wgt: None,
            indices: Some(add(&mut blob, bytemuck::cast_slice(&lod.indices))),
            xforms,
            facing: lod.facing,
            lod: None,
            lod_fade: [0.0; 4],
            shadow_mesh: None,
            // every track mesh gets both, so the track pipeline can always
            // bind them (white in the engine's colour/255 units, uv1 = 0)
            colors: Some(add(&mut blob, &track_vertex_stream(&lod.colors, verts, 4, 1.0 / 255.0))),
            uv1: Some(add(&mut blob, &track_vertex_stream(&lod.uv1, verts, 2, 0.0))),
        });
    }
    println!(
        "scene graph: {} static mesh(es), physics_only skipped {}",
        meshes.len(),
        physics_skip
    );
    let mut by_path: HashMap<String, Vec<SceneInst>> = HashMap::new();
    for i in insts {
        by_path.entry(i.path.to_lowercase()).or_default().push(i);
    }
    // grandstands_a + grandstands_b (and similar pairs) list the same
    // bleachers twice. Two copies at one pivot look like a twisted stand.
    for set in by_path.values_mut() {
        if set.len() < 2 {
            continue;
        }
        set.sort_by(|a, b| {
            a.pos[0].total_cmp(&b.pos[0])
                .then(a.pos[2].total_cmp(&b.pos[2]))
                .then(a.pos[1].total_cmp(&b.pos[1]))
        });
        let mut kept = Vec::with_capacity(set.len());
        for inst in set.drain(..) {
            let dup = kept.last().is_some_and(|p: &SceneInst| {
                (p.pos[0] - inst.pos[0]).abs() < 0.15
                    && (p.pos[1] - inst.pos[1]).abs() < 0.15
                    && (p.pos[2] - inst.pos[2]).abs() < 0.15
            });
            if !dup {
                kept.push(inst);
            }
        }
        *set = kept;
    }
    let mut sets: Vec<(String, Vec<SceneInst>)> = by_path.into_iter().collect();
    lap("track: scenes + static meshes");
    // Trees / props first so a 200k grass dump cannot starve them.
    sets.sort_by(|a, b| {
        let rank = |p: &str| {
            let grass = p.contains("grass")
                || p.contains("/low/")
                || p.contains("vegetation/low")
                || p.contains("wild_grass");
            grass as u8
        };
        rank(&a.0).cmp(&rank(&b.0)).then_with(|| a.0.cmp(&b.0))
    });
    let mut inst_meshes = 0usize;
    let mut inst_copies = 0usize;
    let mut inst_miss = 0usize;
    let mut inst_bad = 0usize;
    // ⚠ This used to be 48 000 with per-set caps of 24 / 1 500 / 8 000. Those
    // predate the per-instance culler (cull.rs: frustum, sub-pixel, LOD
    // bands) and kept as little as 10% of a track: the Nordschleife's
    // 340 000 forest trees came down to 101 and its 11 300 distant forest
    // cards (cf1_b, 40 vertices) to 24, which is why the far landscape looked
    // empty. The culler decides what is drawn now; this only guards memory
    // (80 bytes an instance). EVOVIEW_INST_BUDGET overrides.
    let max_inst_total: usize = std::env::var("EVOVIEW_INST_BUDGET").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(1_500_000);
    let already: std::collections::HashSet<String> = meshes
        .iter()
        .map(|m| m.name.to_lowercase())
        .collect();
    // ⚠ Do NOT stop at the budget - spend it fairly.
    //
    // This used to `break` the moment the total was reached, which spends the
    // whole allowance on whatever sorted first and leaves every remaining mesh
    // TYPE with zero copies. Eleven of the nineteen tracks hit the cap exactly,
    // so on those the tail of the list - the grass, deliberately sorted last -
    // was not thinned, it was absent.
    //
    // Every set now gets a floor of its own before anything takes seconds, so
    // a track ends up with less of everything rather than all of some things
    // and none of the rest.
    let floor_each = if sets.is_empty() {
        0
    } else {
        (max_inst_total / sets.len().max(1) / 2).max(8)
    };
    let mut dropped_sets = 0usize;
    let mut inst_lods = 0usize;
    let mut carpet: Vec<Vec<MeshEntry>> = Vec::new();
    for (path, set) in &sets {
        if skip_track_mesh(&base_of(path), path) {
            inst_bad += 1;
            continue;
        }
        // debugging aid: EVOVIEW_SKIP=substr,substr drops matching instanced
        // meshes, which is how a bad placement gets bisected to one asset
        if let Ok(f) = std::env::var("EVOVIEW_SKIP") {
            if f.split(',').any(|t| !t.is_empty() && path.contains(t)) {
                inst_bad += 1;
                continue;
            }
        }
        if already.contains(&base_of(path)) {
            inst_bad += 1;
            continue;
        }
        let Some(key) = vfs.find(path, None) else {
            inst_miss += 1;
            continue;
        };
        let bytes = vfs.read(&key);
        let mut lods = read_mesh_lods(&bytes);
        if lods.is_empty() {
            inst_bad += 1;
            continue;
        }
        // the clip is named on some LODs only (the flagger: LOD3); every
        // level with its own bones plays it
        if let Some(a) = lods.iter().map(|l| l.1.anim.clone()).find(|a| !a.is_empty()) {
            for l in lods.iter_mut() {
                if !l.1.bones.is_empty() {
                    l.1.anim = a.clone();
                }
            }
        }
        let (_, src) = lods.remove(0);
        if src.positions.is_empty() || src.indices.is_empty() {
            inst_bad += 1;
            continue;
        }
        let mut src = src;
        src.batches.retain(|b| !material_is_helper(&b.material));
        if src.batches.is_empty() || lod_is_helper(&src) {
            inst_bad += 1;
            continue;
        }
        let verts = src.positions.len() / 12;
        let span = mesh_span(&src.positions);
        // EVOVIEW_COLOR_HIST=<substr>: the vertex colour bytes of matching
        // instanced meshes (are the wind masks 0/1 or gradients?)
        if let Ok(f) = std::env::var("EVOVIEW_COLOR_HIST") {
            if path.contains(f.as_str()) && !src.colors.is_empty() {
                let mut line = format!("COLORHIST {path} verts={verts} bytes={}", src.colors.len());
                let fl: &[f32] = bytemuck::cast_slice(&src.colors);
                for c in 0..4 {
                    let mut h = std::collections::BTreeMap::<i32, u32>::new();
                    for v in fl.chunks_exact(4) {
                        // in byte units (x255), to 0.1
                        *h.entry((v[c] * 2550.0).round() as i32).or_default() += 1;
                    }
                    let mut top: Vec<_> = h.into_iter().collect();
                    top.sort_by(|a, b| b.1.cmp(&a.1));
                    line += &format!(" | ch{c} distinct {} top {:?}", top.len(), &top[..top.len().min(6)]);
                }
                println!("{line}");
            }
        }
        let grass = path.contains("grass") || path.contains("/low/");
        let terrain_tile = span > 20.0
            && (grass
                || path.contains("forest")
                || path.contains("outer")
                || path.contains("land")
                || path.contains("terrain"));
        // A "grass" mesh tens of metres across is a terrain tile, not a tuft.
        // Instancing those even a few dozen times paints the striped white
        // slabs on Spa's hillsides.
        // Vertices already in world metres: applying the scene transform
        // again parks bleachers a lap away or on their side.
        let world_baked = span > 12.0 && {
            let p = f3_at(&src.positions, 0);
            (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() > 40.0
        };
        let cap = if world_baked {
            1
        } else if terrain_tile {
            0
        } else if span > 20.0 && verts > 400 {
            // big repeated structures (grandstand blocks, bridges)
            200
        } else {
            // everything else, including the large low-poly forest cards
            // and the trees: the culler and the LOD chain thin them by
            // distance, the way the game does
            400_000
        };
        if cap == 0 {
            inst_bad += 1;
            continue;
        }
        // Ground-cover grass is the game's "grass carpet": hundreds of
        // thousands of tufts, drawn only near the camera. Taking a thinned
        // 1-in-150 sample (all the budget allowed) spread them so far apart
        // they read as specks. Keep EVERY instance aside for the track
        // renderer, which draws the ones within its fade distance.
        if grass && !world_baked && span < 6.0 && verts <= 4000 && set.len() > 2000
            && std::env::var("EVOVIEW_NO_CARPET").is_err()
        {
            let xforms: Vec<[f32; 16]> = set.iter().map(inst_matrix).collect();
            // every LOD level: MeshLodData f3 = switch distance in METRES
            // (grass: 5 / 20 / 40), f25 = fraction of tufts the level keeps
            let mut levels = vec![instanced_entry(&mut blob, base_of(path), &src, xforms,
                                                  Some([0.0, src.density]))];
            let mut last = 0.0f32;
            for (d, mut l) in lods.drain(..) {
                l.batches.retain(|b| !material_is_helper(&b.material));
                if d > last && !l.positions.is_empty() && !l.indices.is_empty() && !l.batches.is_empty() {
                    last = d;
                    let dens = l.density;
                    let mut e = instanced_entry(&mut blob, format!("{}#carpet{}", base_of(path), levels.len()),
                                                &l, Vec::new(), Some([d, dens]));
                    // its crossfade with the level before (track.rs CarpetSet)
                    if lod_fade_enabled() {
                        // the game's own widths only: grass sets them (2 m, 5 m) and a
                        // wider band draws thousands more tufts twice
                        e.lod_fade[0] = l.dlt;
                    }
                    if std::env::var("EVOVIEW_LOD_DUMP").is_ok() {
                        eprintln!("CARPET {:<60} level {} at {:.1}±{:.1} dens {:.2}", path, levels.len(), d, l.dlt, dens);
                    }
                    levels.push(e);
                }
            }
            carpet.push(levels);
            if std::env::var("EVOVIEW_INST").is_ok() {
                eprintln!("INST {:<62} verts={:<7} span={:<8.1} want={:<7} -> carpet", path, verts, span, set.len());
            }
            continue;
        }
        // Past the budget a set still gets its floor, never nothing.
        let remain = if inst_copies < max_inst_total {
            max_inst_total - inst_copies
        } else {
            dropped_sets += 1;
            floor_each
        };
        let take = cap.min(remain.max(floor_each.min(cap))).min(set.len());
        if std::env::var("EVOVIEW_INST").is_ok() {
            eprintln!(
                "INST {:<62} verts={:<7} span={:<8.1} want={:<7} cap={:<6} take={} facing={} {}{}",
                path, verts, span, set.len(), cap, take, src.facing,
                if world_baked { "world_baked " } else { "" },
                if terrain_tile { "terrain_tile" } else { "" });
        }
        if take == 0 {
            inst_bad += 1;
            continue;
        }
        let xforms: Vec<[f32; 16]> = if world_baked {
            Vec::new()
        } else {
            let step = (set.len() / take).max(1);
            set.iter().step_by(step).take(take).map(inst_matrix).collect()
        };
        inst_meshes += 1;
        inst_copies += xforms.len();
        place_dump(path, &src.positions, &xforms);
        meshes.push(instanced_entry(&mut blob, base_of(path), &src, xforms.clone(), None));
        if skinned(&src) {
            anim_meshes.push((meshes.len() - 1, src.bones.clone(), src.anim.clone()));
        }
        // Coarser LODs of an instanced prop become sibling meshes over the
        // same placements; the culler hands each instance to exactly one of
        // them by camera distance. Only levels whose distances increase are
        // kept, and only with real geometry.
        // The game crossfades each switch (MeshLodData f23
        // ditheredLodTransition, metres either side of it) and fades the
        // whole prop out past MeshData lodOut; lod_fade carries both to the
        // culler, which hands the shader a per-instance dither amount.
        if std::env::var("EVOVIEW_LOD_DUMP").map_or(false, |v| v == "2") {
            for (i, (d, l)) in lods.iter().enumerate() {
                eprintln!("RAWLOD {:<60} #{} in {:.1} v{} cast{} shlod{} proxy{} facing{} mats {:?}", path, i + 1, d, l.positions.len() / 12,
                          l.cast_shadows as u8, l.shadow_lod, l.shadow_proxy as u8, l.facing,
                          l.batches.iter().map(|b| b.material.rsplit('/').next().unwrap_or("").to_string()).collect::<Vec<_>>());
            }
        }
        if !xforms.is_empty() && lod_enabled() {
            let fade = lod_fade_enabled();
            let mut levels: Vec<(f32, Lod)> = Vec::new();
            let scale = lod_dist_scale();
            let (out, dlo) = if fade { read_mesh_out(&bytes) } else { (0.0, 0.0) };
            let out = out * scale;
            let kmin = if fade { lod_dither_min() } else { 0.0 };
            let dlo = dlo.max(out * kmin);
            let mut last = 0.0f32;
            // raw MeshLodData index of each kept level, and the levels not
            // drawn (shadow proxies: lodIn 0) that a shadowLod may name
            let raw_n = lods.len() + 1;
            let mut raw_of: Vec<usize> = Vec::new();
            let mut unkept: Vec<(usize, Lod)> = Vec::new();
            for (i, (d, mut l)) in lods.into_iter().enumerate() {
                l.batches.retain(|b| !material_is_helper(&b.material));
                let geo = !l.positions.is_empty() && !l.indices.is_empty() && !l.batches.is_empty();
                if d > last && geo {
                    last = d;
                    l.dlt = if fade { l.dlt.max(d * scale * kmin) } else { 0.0 };
                    levels.push((d * scale, l));
                    raw_of.push(i + 1);
                } else if geo {
                    unkept.push((i + 1, l));
                }
            }
            let lod0 = meshes.len() - 1;
            if std::env::var("EVOVIEW_LOD_DUMP").is_ok() {
                eprintln!("LODS {:<60} n={:<6} out={:<7.1} dlo={:<5.1} lod0 dens={:.2} sh{}/{} fsl{} | {}", path, xforms.len(), out, dlo,
                          src.density, src.cast_shadows as u8, src.shadow_lod, read_force_shadow_lod(&bytes),
                          levels.iter().map(|(d, l)| {
                              let ((a0, b0), (a1, b1)) = (bounds(&src.positions), bounds(&l.positions));
                              let off: f32 = (0..3).map(|i| ((a1[i] + b1[i]) - (a0[i] + b0[i])) * 0.5).map(|x| x * x).sum::<f32>().sqrt();
                              format!("{:.1}±{:.1} v{} dens{:.2} sh{}/{} c{:.2}", d, l.dlt, l.positions.len() / 12, l.density,
                                      l.cast_shadows as u8, l.shadow_lod, off)
                          })
                              .collect::<Vec<_>>().join(", "));
            }
            if !levels.is_empty() || out > 0.0 {
                let first = levels.first().map_or((f32::INFINITY, 0.0), |x| (x.0, x.1.dlt));
                meshes[lod0].lod = Some([0.0, first.0]);
                meshes[lod0].lod_fade = [0.0, first.1, out, dlo];
                for k in 0..levels.len() {
                    let next = levels.get(k + 1).map_or((f32::INFINITY, 0.0), |x| (x.0, x.1.dlt));
                    let (near, l) = &levels[k];
                    let mut e = instanced_entry(
                        &mut blob, format!("{}#lod{}", base_of(path), k + 1), l,
                        xforms.clone(), Some([*near, next.0]));
                    e.lod_fade = [l.dlt, next.1, out, dlo];
                    meshes.push(e);
                    if skinned(l) {
                        anim_meshes.push((meshes.len() - 1, l.bones.clone(), l.anim.clone()));
                    }
                    inst_lods += 1;
                }
                // Shadows (MeshData f10 forceShadowLod): each level casts with
                // the geometry of raw level min(shadowLod, last) - every
                // beech level with its shadow proxy, the crowd with its
                // coarsest body - or not at all without castShadows. So a
                // prop's shadow does not change shape as its LOD switches.
                // EVOVIEW_SHADOW_LOD=0: each level casts itself.
                if read_force_shadow_lod(&bytes) && std::env::var("EVOVIEW_SHADOW_LOD").map_or(true, |v| v != "0") {
                    let mut raw_mesh: Vec<(usize, usize)> = vec![(0, lod0)];
                    raw_mesh.extend(raw_of.iter().enumerate().map(|(k, &r)| (r, lod0 + 1 + k)));
                    let casts: Vec<(bool, i32)> = std::iter::once((src.cast_shadows, src.shadow_lod))
                        .chain(levels.iter().map(|(_, l)| (l.cast_shadows, l.shadow_lod))).collect();
                    for (k, &(cast, sl)) in casts.iter().enumerate() {
                        let target = if !cast {
                            None
                        } else {
                            let t = (sl.max(0) as usize).min(raw_n - 1);
                            match raw_mesh.iter().find(|x| x.0 == t) {
                                Some(&(_, m)) => Some(m),
                                None => unkept.iter().find(|x| x.0 == t).map(|(_, l)| {
                                    // a level only shadows draw: never in a view
                                    let mut e = instanced_entry(&mut blob, format!("{}#shadow{}", base_of(path), t), l,
                                                                xforms.clone(), Some([f32::INFINITY, f32::INFINITY]));
                                    e.shadow_mesh = Some(None);
                                    meshes.push(e);
                                    if skinned(l) {
                                        anim_meshes.push((meshes.len() - 1, l.bones.clone(), l.anim.clone()));
                                    }
                                    raw_mesh.push((t, meshes.len() - 1));
                                    meshes.len() - 1
                                }),
                            }
                        };
                        meshes[lod0 + k].shadow_mesh = Some(target);
                    }
                }
                // EVOVIEW_LOD_FORCE=k: every prop draws its level k (its
                // coarsest when it has fewer) at every distance, no fades:
                // a still per k compares the levels' looks (lodpop audit)
                if let Some(f) = std::env::var("EVOVIEW_LOD_FORCE").ok().and_then(|v| v.parse::<usize>().ok()) {
                    let pick = f.min(levels.len());
                    for (j, m) in meshes[lod0..].iter_mut().enumerate() {
                        if m.name.contains("#shadow") {
                            continue;
                        }
                        m.lod = Some(if j == pick { [0.0, f32::INFINITY] } else { [f32::INFINITY, f32::INFINITY] });
                        m.lod_fade = [0.0; 4];
                    }
                }
            }
        }
    }
    if inst_lods > 0 {
        println!("scene instances: {inst_lods} coarser LOD level(s) added");
    }
    if inst_meshes + inst_miss + inst_bad > 0 {
        println!(
            "scene instances: {} unique meshes, {} placed (missed {} path(s), {} unread{})",
            inst_meshes, inst_copies, inst_miss, inst_bad,
            if dropped_sets > 0 {
                format!(", {dropped_sets} set(s) past the budget on their floor")
            } else {
                String::new()
            }
        );
    }

    if meshes.is_empty() {
        return Err("track meshes did not decode".into());
    }

    lap("track: instanced meshes");
    let mut mat_paths = vfs.own_keys("material");
    mat_paths.retain(|k| !k.to_lowercase().contains("/editor/"));
    mat_paths.sort();
    mat_paths.dedup();

    let push_material = |vfs: &mut Vfs,
                             path: &str,
                             blob: &mut Vec<u8>,
                             textures: &mut Vec<TextureEntry>,
                             tex_index: &mut HashMap<String, usize>,
                             materials: &mut Vec<MaterialEntry>| {
        let mb = vfs.read(path);
        let Some(m) = read_material_vfs(vfs, &mb, &base_of(path)) else { return };
        let mut resolved: HashMap<String, usize> = HashMap::new();
        for (slot, reference) in &m.slots {
            let low = reference.replace('\\', "/").to_lowercase();
            if low.contains("default_material") || low.starts_with("editor/")
                || low.contains("/null.texture") || low.ends_with("null.texture") {
                continue;
            }
            let near = parent_of(path).to_string();
            let Some(found) = vfs.find(reference, Some(&near)) else {
                crate::strict::note("TR3", &m.file, format!("{slot}: {reference} not found"));
                continue;
            };
            let idx = if let Some(&i) = tex_index.get(&found) {
                i
            } else {
                let Some(t) = vfs.texture(&found, AUTO_MIP) else {
                    crate::strict::note("TR3", &m.file, format!("{slot}: {found} not decodable"));
                    continue;
                };
                let name = base_of(&found).replace(".texture", "");
                let data = add(blob, &t.data);
                textures.push(TextureEntry {
                    name,
                    format: t.format,
                    width: t.width,
                    height: t.height,
                    levels: t.levels,
                    data,
                });
                let i = textures.len() - 1;
                tex_index.insert(found.clone(), i);
                i
            };
            resolved.insert(slot.clone(), idx);
        }
        // DynamicTrack (Spa asph_spa*) lists only editor albedo2. Skipping
        // that left the ribbon unbound → white. Bind the shared asphalt tile.
        let shader_l = m.shader.as_deref().unwrap_or("").to_lowercase();
        let file_l = m.file.to_lowercase();
        let has_alb = resolved.keys().any(|k| {
            let k = k.to_lowercase();
            k.contains("basecolormap")
                || k.contains("txdiffuse")
                || k.contains("albedomap")
                || k == "diffusemap"
        });
        let wants_road = !has_alb
            && (file_l.contains("asph")
                || file_l.contains("tarmac")
                || (shader_l.contains("dynamictrack")
                    && !shader_l.contains("multilayer")
                    && !file_l.contains("line")));
        if wants_road {
            const CANDS: &[&str] = &[
                "content/tracks/common_assets/textures/asphalt/track_base_albedo.texture",
                "track_base_albedo.texture",
                "asphalt_fine_2_dark_albedo.texture",
            ];
            for cand in CANDS {
                let Some(found) = vfs.find(cand, None) else { continue };
                let idx = if let Some(&i) = tex_index.get(&found) {
                    i
                } else {
                    let Some(t) = vfs.texture(&found, AUTO_MIP) else { continue };
                    let name = base_of(&found).replace(".texture", "");
                    let data = add(blob, &t.data);
                    textures.push(TextureEntry {
                        name,
                        format: t.format,
                        width: t.width,
                        height: t.height,
                        levels: t.levels,
                        data,
                    });
                    let i = textures.len() - 1;
                    tex_index.insert(found.clone(), i);
                    i
                };
                resolved.insert("Base_BaseColorMap".into(), idx);
                println!("road albedo {}: {}", m.file, found);
                crate::strict::note("TR5", &m.file, base_of(&found));
                break;
            }
        }
        // ⚠ Some materials bind NO textures at all and rely on SIBLING FILES.
        // `bark_mat.material` (every vegetation folder) sits next to
        // `bark_albedo.texture`, `bark_normal.texture`,
        // `bark_roughness.texture` and names none of them - EVO resolves them
        // by filename convention. Without this, tree trunks render flat white
        // on every track that ships only the untextured copy (Brands Hatch,
        // Cota, Donington). This is the same trick `companion_alpha_ref`
        // already does for fence cutouts, generalised.
        //
        // Deliberately narrow: the candidate name has to be DERIVED FROM THIS
        // MATERIAL'S OWN NAME, so a glass material cannot adopt an unrelated
        // neighbour's albedo just by sharing a folder.
        {
            let stem = base_of(path).to_lowercase().replace(".material", "");
            let root = stem
                .strip_suffix("_mat")
                .or_else(|| stem.strip_suffix("_material"))
                .unwrap_or(&stem)
                .to_string();
            let near = parent_of(path).to_string();
            let mut adopt = |slot: &str, sfx: &[&str], resolved: &mut HashMap<String, usize>,
                             blob: &mut Vec<u8>, textures: &mut Vec<TextureEntry>,
                             tex_index: &mut HashMap<String, usize>, vfs: &mut Vfs| {
                if resolved.contains_key(slot) {
                    return;
                }
                for stemname in [root.as_str(), stem.as_str()] {
                    for s in sfx {
                        let cand = format!("{stemname}{s}.texture");
                        let Some(found) = vfs.find(&cand, Some(&near)) else { continue };
                        let idx = if let Some(&i) = tex_index.get(&found) {
                            Some(i)
                        } else {
                            vfs.texture(&found, AUTO_MIP).map(|t| {
                                let name = base_of(&found).replace(".texture", "");
                                let data = add(blob, &t.data);
                                textures.push(TextureEntry {
                                    name, format: t.format, width: t.width,
                                    height: t.height, levels: t.levels, data,
                                });
                                let i = textures.len() - 1;
                                tex_index.insert(found.clone(), i);
                                i
                            })
                        };
                        if let Some(i) = idx {
                            resolved.insert(slot.to_string(), i);
                            crate::strict::note("TR6", &m.file, format!("{slot} <- {cand}"));
                            if std::env::var("EVOVIEW_SIBLING_LOG").is_ok() {
                                eprintln!("SIBLING {} <- {}", m.file, cand);
                            }
                            return;
                        }
                    }
                }
            };
            let has_albedo = resolved.keys().any(|k| k.contains("BaseColorMap")
                || k == "txDiffuse" || k == "ksDiffuse");
            if !has_albedo {
                adopt("Base_BaseColorMap", &["_albedo", "_basecolor", "_color", "_c"],
                      &mut resolved, blob, textures, tex_index, vfs);
                adopt("Base_NormalMap", &["_normal", "_n"],
                      &mut resolved, blob, textures, tex_index, vfs);
                adopt("Base_RoughnessMap", &["_roughness", "_r"],
                      &mut resolved, blob, textures, tex_index, vfs);
            }
        }
        if let Some(aref) = companion_alpha_ref(&m) {
            crate::strict::note("TR7", &m.file, aref.clone());
            if let Some(found) = vfs.find(&aref, Some(&parent_of(path).to_string())) {
                if !resolved.contains_key("Base_OpacityMap") {
                    let idx = if let Some(&i) = tex_index.get(&found) {
                        Some(i)
                    } else {
                        vfs.texture(&found, AUTO_MIP).map(|t| {
                            let name = base_of(&found).replace(".texture", "");
                            let data = add(blob, &t.data);
                            textures.push(TextureEntry {
                                name,
                                format: t.format,
                                width: t.width,
                                height: t.height,
                                levels: t.levels,
                                data,
                            });
                            let i = textures.len() - 1;
                            tex_index.insert(found.clone(), i);
                            i
                        })
                    };
                    if let Some(i) = idx {
                        resolved.insert("Base_OpacityMap".into(), i);
                    }
                }
            }
        }
        materials.push(MaterialEntry {
            file: m.file,
            path: norm_mat_path(path),
            shader: m.shader,
            slots: resolved,
            params: m.params,
            colors: m.colors.into_iter().collect(),
        });
    };

    let keys = material_texture_keys(vfs, &mat_paths);
    let bytes = vfs.prefetch_textures(&keys, AUTO_MIP);
    blob.reserve(bytes);
    for path in &mat_paths.clone() {
        push_material(vfs, path, &mut blob, &mut textures, &mut tex_index, &mut materials);
    }

    if std::env::var("EVOVIEW_REF_LOG").is_ok() {
        let mut seen = std::collections::BTreeSet::new();
        for m in meshes.iter() {
            for b in m.batches.iter() {
                if b.material.contains('/') || b.material.contains('\\') {
                    seen.insert(format!("PATHED {}", b.material));
                } else {
                    seen.insert(format!("BARE   {}", b.material));
                }
            }
        }
        for s in seen.iter().take(25) { eprintln!("REF {s}"); }
        eprintln!("REF total distinct {}", seen.len());
    }
    let have: std::collections::HashSet<String> = materials
        .iter()
        .map(|m| m.file.to_lowercase().replace(".material", ""))
        .collect();
    let mut missing: Vec<String> = meshes
        .iter()
        .chain(carpet.iter().flatten())
        .flat_map(|m| m.batches.iter())
        .map(|b| b.material.clone())
        .filter(|r| {
            let base = r
                .rsplit(['\\', '/'])
                .next()
                .unwrap_or(r)
                .to_lowercase()
                .replace(".material", "");
            !base.is_empty() && !have.contains(&base)
        })
        .collect();
    missing.sort();
    missing.dedup();
    for r in missing {
        if let Some(key) = vfs.find(&r, None) {
            push_material(vfs, &key, &mut blob, &mut textures, &mut tex_index, &mut materials);
        }
    }

    println!(
        "track: {} mesh(es), {} texture(s), {} material(s)",
        meshes.len(),
        textures.len(),
        materials.len()
    );
    let mut env = crate::track::load_env(vfs);
    env.collision = build_collision(vfs, &colliders, &wall_insts);
    env.anim = build_track_anim(vfs, &meshes, anim_meshes).map(std::sync::Arc::new);
    env.n_materials = materials.len();
    if !carpet.is_empty() {
        println!("grass carpet: {} set(s), {} tuft(s)",
                 carpet.len(), carpet.iter().map(|m| m[0].xforms.len()).sum::<usize>());
    }
    env.carpet = carpet;
    // The pigment map (SceneGraphicsSettings.pigment) tints grass cards by
    // the ground under them; it is a plain texture, carried like any other.
    if let Some(pp) = env.pigment_path.clone() {
        if let Some(found) = vfs.find(&pp, None) {
            let hdr = vfs.read(&found);
            let mips = vfs.read(&found.replace(".texture", ".texturemips"));
            if let Some(t) = read_texture_lod(&hdr, &mips, 0) {
                let data = add(&mut blob, &t.data);
                textures.push(TextureEntry {
                    name: base_of(&found).replace(".texture", ""),
                    format: t.format,
                    width: t.width,
                    height: t.height,
                    levels: t.levels,
                    data,
                });
                env.pigment_tex = Some(textures.len() - 1);
            }
        }
    }
    // The weather map (clouds_settings/setting_1 weatherMapPath): cloud cover
    // and rain, laid over the world by clouds.rs.
    if let Some(found) = vfs.find("content/weather/weather_map6.texture", None) {
        let hdr = vfs.read(&found);
        let mips = vfs.read(&found.replace(".texture", ".texturemips"));
        if let Some(t) = read_texture_lod(&hdr, &mips, 0) {
            let data = add(&mut blob, &t.data);
            textures.push(TextureEntry {
                name: "weather_map6".into(),
                format: t.format,
                width: t.width,
                height: t.height,
                levels: t.levels,
                data,
            });
            env.weather_tex = Some(textures.len() - 1);
        }
    }
    // The cirrus layer (content/weather/clouds/cirrus_layer.texture): drawn
    // by the track sky as a high deck over the volumetric clouds.
    if let Some(found) = vfs.find("content/weather/clouds/cirrus_layer.texture", None) {
        let hdr = vfs.read(&found);
        let mips = vfs.read(&found.replace(".texture", ".texturemips"));
        if let Some(t) = read_texture_lod(&hdr, &mips, 0) {
            let data = add(&mut blob, &t.data);
            textures.push(TextureEntry {
                name: "cirrus_layer".into(),
                format: t.format,
                width: t.width,
                height: t.height,
                levels: t.levels,
                data,
            });
            env.cirrus_tex = Some(textures.len() - 1);
        }
    }
    lap("track: materials + textures");
    vfs.drop_prefetched();
    let mut car = Car::from_parts(meshes, textures, materials, blob);
    car.world_uv_scale = world_uv_scale;
    car.track_env = Some(Box::new(env));
    // ⚠ No studio here. This is the TRACK loader, and the studio cube is the
    // showroom the game paints cars in - an outdoor scene wants its own
    // reflection captures, which are a separate asset.
    Ok(Loaded {
        car,
        skeleton: Skeleton::build(Vec::new()),
        animations: Vec::new(),
    })
}

/// Read one car out of an archive. Nothing is written to disk.
pub fn load_from_package(
    pkg_path: &Path,
    car_id: &str,
    base: Option<&Path>,
) -> Result<Loaded, String> {
    load(&mut package_vfs(pkg_path, car_id, base)?)
}

/// The Vfs `load_from_package` reads from (kept by livery.rs for switching).
pub fn package_vfs(
    pkg_path: &Path,
    car_id: &str,
    base: Option<&Path>,
) -> Result<Vfs, String> {
    let pkg = crate::pkg::Package::open(pkg_path)?;
    if !pkg.cars().iter().any(|c| c.eq_ignore_ascii_case(car_id)) {
        return Err(format!("{car_id} is not in {}", pkg_path.display()));
    }
    let mut vfs = Vfs::package(pkg, car_id);
    if let Some(b) = base {
        if b != pkg_path {
            vfs.under(crate::pkg::Package::open(b)?);
        }
    }
    Ok(vfs)
}

pub fn load(vfs: &mut Vfs) -> Result<Loaded, String> {

    let mut blob: Vec<u8> = Vec::new();
    let mut meshes: Vec<MeshEntry> = Vec::new();
    let mut textures: Vec<TextureEntry> = Vec::new();
    let mut materials: Vec<MaterialEntry> = Vec::new();
    let mut tex_index: HashMap<String, usize> = HashMap::new();
    let mut unresolved: Vec<String> = Vec::new();

    let add = |blob: &mut Vec<u8>, data: &[u8]| -> [u64; 2] {
        let off = blob.len() as u64;
        blob.extend_from_slice(data);
        while blob.len() % 4 != 0 {
            blob.push(0);
        }
        [off, data.len() as u64]
    };

    // ---- meshes, and the skeleton that places the wheels ----
    let mut skeleton_bones: Vec<Bone> = Vec::new();
    let mut bone_index: HashMap<String, usize> = HashMap::new();
    let mut bone_owner: Vec<usize> = Vec::new();
    let mut attach_lods: Vec<(String, String, Lod)> = Vec::new();
    let mut wheel_pos: HashMap<String, [f32; 3]> = HashMap::new();
    let mut shared_materials: Vec<String> = Vec::new();
    let mut rim_lods: Vec<(String, Lod)> = Vec::new();
    // what was fitted where, for EVOVIEW_WHEEL_AUDIT
    let mut wheel_src: Vec<(String, String, [f32; 3])> = Vec::new();

    // What the game fits at each wheel, from the mech preset's .compatiblerim.
    // The older .rimmesh guess below is only for cars without one.
    let plan = wheel_plan(vfs);
    let (mut rim_files, mut tyre_files) = match &plan {
        Some(p) => {
            println!(
                "wheels: {} | rims {:?} | tyres {:?} | track {:?}",
                p.label,
                p.rims.iter().map(|r| r.as_deref().map(base_of)).collect::<Vec<_>>(),
                p.tyres.iter().map(|r| r.as_deref().map(base_of)).collect::<Vec<_>>(),
                p.track);
            let mut r: Vec<String> = Vec::new();
            for k in p.rims.iter().flatten() {
                if !r.contains(k) {
                    r.push(k.clone());
                }
            }
            (r, p.tyres.iter().flatten().cloned().collect())
        }
        None => rim_selection(vfs),
    };
    let planned_rims: Vec<String> = rim_files.iter().map(|k| base_of(k).to_lowercase()).collect();
    if rim_files.is_empty() || tyre_files.is_empty() {
        let refs = actor_refs(vfs);
        for r in &refs {
            let low = r.to_lowercase();
            if !low.ends_with(".mesh") {
                continue;
            }
            if low.contains("_rim_") && !low.contains("_rim_blur_") && rim_files.is_empty() {
                if let Some(k) = vfs.find(r, None) {
                    if !rim_files.contains(&k) {
                        rim_files.push(k);
                    }
                }
            } else if low.replace('\\', "/").contains("/tyres/") {
                if let Some(k) = vfs.find(r, None) {
                    if !tyre_files.contains(&k) {
                        tyre_files.push(k);
                    }
                }
            }
        }
    }
    // Parts the actor parents to a bone: the steering wheel and the number
    // plates. Keyed by file name, since the actor spells paths the package's
    // way and the plates live outside the car folder entirely.
    // Body kit and equipment the presets fit (parts.rs).
    let mech_preset = pick_mech_preset(vfs);
    let visual_preset = pick_visual_preset(vfs);
    let fs_parts = final_state(vfs).map(|f| f.parts).unwrap_or_default();
    let fitted = crate::parts::fitted_parts(vfs, mech_preset.as_deref(), visual_preset.as_deref(),
                                            &fs_parts);
    let mut stock_lods: Vec<(String, Lod)> = Vec::new();
    let mut attach: HashMap<String, String> = HashMap::new();
    let mut attach_files: Vec<String> = Vec::new();
    for (p, bone) in actor_attachments(vfs) {
        if bone.starts_with("WHEEL_") || bone.starts_with("TYRE_") {
            continue; // already placed, and rim styles are chosen elsewhere
        }
        let name = base_of(&p.replace('\\', "/"));
        attach.insert(name.to_lowercase(), bone);
        if let Some(k) = vfs.find(&p, None) {
            attach_files.push(k);
        }
    }

    let mut mesh_files = vfs.list("meshes", "mesh");
    for r in rim_files.iter().chain(attach_files.iter()) {
        if !mesh_files.contains(r) {
            mesh_files.push(r.clone());
        }
    }
    // ⚠ `_off` meshes are the NOT-FITTED state of a part - belt_off, net_off,
    // harness_off - and the actor lists them alongside the fitted version
    // rather than instead of it. Drawing both puts a slack belt through a
    // fastened one.
    // what happened to every mesh file considered (EVOVIEW_RENDER_AUDIT)
    let mut mesh_audit: Vec<(String, &'static str)> = Vec::new();
    // A mesh a .carpart names is optional equipment even when it sits in the
    // top-level meshes/ folder (the Caterham's track harness): drawn only
    // when a preset fits it (parts.rs), never wholesale with the body.
    let mut part_meshes: Vec<String> = Vec::new();
    for k in vfs.list_rec("presets", "carpart") {
        let d = vfs.read(&k);
        let mut refs = Vec::new();
        strings_under(&d, 0, d.len(), 0, &mut refs);
        for r in refs.iter().filter(|r| r.to_lowercase().ends_with(".mesh")) {
            if let Some(f) = vfs.find(r, None) {
                part_meshes.push(f);
            }
        }
    }
    mesh_files.retain(|k| {
        let opt = part_meshes.contains(k);
        if opt {
            mesh_audit.push((k.clone(), "skip: optional part (a .carpart's mesh, not fitted)"));
        }
        !opt
    });
    // a fitted part placed by parts.rs must not also load as stock geometry
    mesh_files.retain(|k| !fitted.iter().any(|p| &p.mesh == k));
    let fitted_files: Vec<String> = mesh_files.iter().map(|k| base_of(k)).collect();
    // ⚠ The actor names each base mesh's ROLE (CarActorData.BaseMeshes):
    // seatbelt / harness / safety net each come as an _on and an _off mesh,
    // and the file names do not pair up (belt_auto is harness_on next to
    // belt_off), so the name rule below missed ~65 cars and drew the slack
    // belt through the fastened one. With a driver in the seat the _on set
    // is fitted, with an empty seat the _off set.
    let roles = actor_base_meshes(vfs);
    let seated = crate::driver::setting().is_some();
    mesh_files.retain(|k| {
        let n = base_of(k);
        let keep = match roles.get(&n.to_lowercase()) {
            Some(r) if BASE_MESH_OFF.contains(r) => !seated,
            Some(r) if BASE_MESH_ON.contains(r) => seated,
            Some(_) => true,
            None => match n.strip_suffix("_off.mesh") {
                Some(stem) => !fitted_files.iter().any(|o| o == &format!("{stem}.mesh")),
                None => true,
            },
        };
        if !keep {
            mesh_audit.push((k.clone(), if seated { "skip: _off state (driver seated)" } else { "skip: _on state (empty seat)" }));
        }
        keep
    });
    // Wheels must come last: they are placed from the body's skeleton, so
    // every non-wheel mesh has to be read first.
    mesh_files.sort_by_key(|k| {
        let n = base_of(k);
        (n.contains("_rim"), n)
    });

    for key in &mesh_files.clone() {
        let name = base_of(key);
        // rim_blur is the spinning-wheel stand-in and overlays the real rim
        // ⚠ Match "blur" ANYWHERE, not "rim_blur": a car with wheel options
        // names the style in between (rim_ade_blur_static), and a blur mesh
        // that slips through is a smeared disc drawn over the real rim.
        if (name.contains("_rim") && name.contains("blur")) || name.contains("collider") {
            mesh_audit.push((key.clone(), if name.contains("collider") { "skip: collider" } else { "skip: rim blur" }));
            continue;
        }
        let bytes = vfs.read(key);
        let mut lod = match read_mesh(&bytes) {
            Some(l) => l,
            None => {
                mesh_audit.push((key.clone(), "SKIP: unreadable mesh"));
                continue;
            }
        };
        if lod.positions.is_empty() || lod.indices.is_empty() {
            mesh_audit.push((key.clone(), "SKIP: empty mesh"));
            continue;
        }
        // ⚠ Merge every mesh's skeleton into one table and rewrite that
        // mesh's bone indices to match. Taking the richest skeleton and
        // skinning everything against it left small meshes reading bones that
        // belong to something else. The merge also gathers bones that exist in
        // only one mesh - STEER_BASE lives in the interior, never the body -
        // which is what the attachment pass below needs.
        let map = merge_bones(&mut skeleton_bones, &mut bone_index, &mut bone_owner,
                              &lod.bones);
        if !lod.bone_idx.is_empty() {
            remap_bone_idx(&mut lod.bone_idx, &map);
        }
        // Parts the actor hangs off a bone are authored at the origin, and the
        // bone may come from a mesh not read yet, so place them in a second
        // pass once the whole skeleton is known.
        if let Some(bone) = attach.get(&name.to_lowercase()) {
            mesh_audit.push((key.clone(), "attachment"));
            attach_lods.push((name.clone(), bone.clone(), lod));
            continue;
        }
        // ⚠ Wheels are authored at the origin and placed by bone WHEEL_xx.
        // A rim mesh only exists per style and often only for the front pair,
        // so it is handled in a second pass below rather than here.
        if corner_of(&name).is_some() || planned_rims.contains(&name.to_lowercase()) {
            mesh_audit.push((key.clone(), "rim"));
            rim_lods.push((name.clone(), lod));
            continue;
        }
        // held back until the whole skeleton is known: a replacement part
        // hides stock geometry by bone (see parts.rs)
        mesh_audit.push((key.clone(), "stock"));
        stock_lods.push((name.clone(), lod));
    }

    // ---- wheels ----------------------------------------------------------
    // Corners come from the SKELETON, not from which rim files happen to
    // exist: a car whose .rimmesh names only the front pair still has four
    // WHEEL_xx bones, and deriving corners from the files leaves it on two
    // wheels. The same-side front rim is reused where a rear one is missing.
    let skel_now = Skeleton::build(skeleton_bones.clone());

    // ---- stock meshes, minus what the fitted parts replace -----------------
    let bone_names: Vec<String> = skel_now.bones.iter().map(|b| b.name.clone()).collect();
    let bone_parents: Vec<Option<usize>> = skel_now.bones.iter().map(|b| b.parent).collect();
    let hidden = crate::parts::hidden_bones(&fitted, &bone_names, &bone_parents);
    let mut hidden_tris: Vec<(String, usize)> = Vec::new();
    for (name, mut lod) in stock_lods {
        let n = crate::parts::hide_triangles(&mut lod.indices, &mut lod.batches, &lod.bone_idx,
                                             &lod.bone_wgt, &hidden);
        if n > 0 {
            lod.batches.retain(|b| b.count > 0);
            hidden_tris.push((name.clone(), n));
        }
        let verts = lod.positions.len() / 12;
        // unskinned stock geometry rides on the body, so it rolls and pitches
        // with it in drive mode
        if lod.bone_idx.is_empty() || lod.bone_wgt.is_empty() {
            let root = skel_now.index.get("CAR_BODY").copied().unwrap_or(0);
            let (i, w) = rigid_skin(verts, root);
            lod.bone_idx = i;
            lod.bone_wgt = w;
        }
        meshes.push(MeshEntry {
            name: name.clone(),
            vertices: verts as u32,
            batches: lod.batches.clone(),
            positions: Some(add(&mut blob, &lod.positions)),
            normals: Some(add(&mut blob, &lod.normals)),
            uvs: Some(add(&mut blob, &lod.uvs)),
            tangents: Some(add(&mut blob, &lod.tangents)),
            bone_idx: (!lod.bone_idx.is_empty()).then(|| add(&mut blob, &lod.bone_idx)),
            bone_wgt: (!lod.bone_wgt.is_empty()).then(|| add(&mut blob, &lod.bone_wgt)),
            indices: Some(add(&mut blob, bytemuck::cast_slice(&lod.indices))),
            xforms: Vec::new(),
            facing: lod.facing,
            lod: None,
            lod_fade: [0.0; 4],
            shadow_mesh: None,
            colors: None,
            uv1: None,
        });
    }
    if !fitted.is_empty() {
        println!("parts: {} fitted, {} stock triangles replaced", fitted.len(),
                 hidden_tris.iter().map(|h| h.1).sum::<usize>());
    }

    // ---- parts the actor parents to a bone --------------------------------
    // ⚠ Full transform, not just the translation used for rims. A steering
    // wheel is raked back and a number plate is raked with the bodywork;
    // translating alone lays them flat.
    for (name, bone, proto) in &attach_lods {
        let Some(&bi) = skel_now.index.get(bone.as_str()) else {
            eprintln!("attachment {name}: no bone {bone}");
            continue;
        };
        // a replacement part on this bone (a Nardi wheel on STEER_BASE)
        // takes the actor's own attachment's place
        if hidden[bi] {
            println!("attachment {name}: replaced by a fitted part on {bone}");
            continue;
        }
        let mut lod = clone_lod(proto);
        transform_lod(&mut lod, &skel_now.bind[bi]);
        let (lo, hi) = bounds(&lod.positions);
        println!(
            "attached {name} -> {bone} at ({:.2}, {:.2}, {:.2})",
            (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0);
        let verts = lod.positions.len() / 12;
        let (bidx, bwgt) = rigid_skin(verts, bi);
        meshes.push(MeshEntry {
            name: name.clone(),
            vertices: verts as u32,
            batches: lod.batches.clone(),
            positions: Some(add(&mut blob, &lod.positions)),
            normals: Some(add(&mut blob, &lod.normals)),
            uvs: Some(add(&mut blob, &lod.uvs)),
            tangents: Some(add(&mut blob, &lod.tangents)),
            bone_idx: Some(add(&mut blob, &bidx)),
            bone_wgt: Some(add(&mut blob, &bwgt)),
            indices: Some(add(&mut blob, bytemuck::cast_slice(&lod.indices))),
            xforms: Vec::new(),
            facing: lod.facing,
            lod: None,
            lod_fade: [0.0; 4],
            shadow_mesh: None,
            colors: None,
            uv1: None,
        });
    }

    // ---- parts the presets fit (body kit, brakes, cockpit equipment) -------
    // ⚠ meshes/parts/, kouki/, sports/ ... are alternative equipment and are
    // never loaded wholesale; only what the mech preset FORCES (field 10) and
    // the visual preset's part_states are drawn, each in its attachment
    // bone's frame. The stock geometry a replacement part takes the place of
    // was dropped above. Skinned rigidly to that bone so a door trim swings
    // with its door.
    let mut parts_drawn: Vec<(String, String, bool)> = Vec::new();
    for p in &fitted {
        let Some(&bi) = skel_now.index.get(p.bone.as_str()) else {
            eprintln!("part {}: no bone {}", base_of(&p.mesh), p.bone);
            continue;
        };
        let bytes = vfs.read(&p.mesh);
        let Some(mut lod) = read_mesh(&bytes) else { continue };
        if lod.positions.is_empty() || lod.indices.is_empty() {
            continue;
        }
        let m = mat_mul(&p.local, &skel_now.bind[bi]);
        transform_lod(&mut lod, &m);
        if let Some(dk) = &p.design {
            let db = vfs.read(dk);
            let slots = read_design(&db);
            for b in lod.batches.iter_mut() {
                if let Some(s) = slots.iter().find(|s| s.id.eq_ignore_ascii_case(&b.name)
                    && !s.material.is_empty()) {
                    b.material = s.material.clone();
                }
            }
        }
        for b in &lod.batches {
            if !b.material.is_empty() {
                shared_materials.push(b.material.clone());
            }
        }
        let verts = lod.positions.len() / 12;
        let mut bidx = Vec::with_capacity(verts * 16);
        let mut bwgt = Vec::with_capacity(verts * 16);
        for _ in 0..verts {
            for k in 0..4 {
                bidx.extend_from_slice(&(if k == 0 { bi as f32 } else { 0.0 }).to_le_bytes());
                bwgt.extend_from_slice(&(if k == 0 { 1.0f32 } else { 0.0 }).to_le_bytes());
            }
        }
        println!("part {} -> {} ({})", base_of(&p.mesh), p.bone,
                 if p.additive { "additive" } else { "replacement" });
        parts_drawn.push((base_of(&p.mesh), p.bone.clone(), p.additive));
        meshes.push(MeshEntry {
            name: format!("part_{}", base_of(&p.mesh)),
            vertices: verts as u32,
            batches: lod.batches.clone(),
            positions: Some(add(&mut blob, &lod.positions)),
            normals: Some(add(&mut blob, &lod.normals)),
            uvs: Some(add(&mut blob, &lod.uvs)),
            tangents: Some(add(&mut blob, &lod.tangents)),
            bone_idx: Some(add(&mut blob, &bidx)),
            bone_wgt: Some(add(&mut blob, &bwgt)),
            indices: Some(add(&mut blob, bytemuck::cast_slice(&lod.indices))),
            xforms: Vec::new(),
            facing: lod.facing,
            lod: None,
            lod_fade: [0.0; 4],
            shadow_mesh: None,
            colors: None,
            uv1: None,
        });
    }
    if std::env::var_os("EVOVIEW_PARTS_AUDIT").is_some() {
        let hid: Vec<&str> = bone_names.iter().zip(&hidden).filter(|(_, h)| **h)
            .map(|(n, _)| n.as_str()).collect();
        println!("PARTS_AUDIT {}", serde_json::json!({
            "mech": mech_preset.as_deref().map(base_of),
            "visual": visual_preset.as_deref().map(base_of),
            "parts": parts_drawn.iter().map(|(m, b, a)| serde_json::json!(
                {"mesh": m, "bone": b, "additive": a})).collect::<Vec<_>>(),
            "hidden_bones": hid,
            "hidden_tris": hidden_tris,
        }));
    }

    // Bone transforms the tyres are placed with, per corner.
    let mut tyre_xf: HashMap<String, [f32; 16]> = HashMap::new();
    for (ci, corner) in ["LF", "RF", "LR", "RR"].into_iter().enumerate() {
        let Some(&bi) = skel_now.index.get(&format!("WHEEL_{corner}")) else { continue };
        let mut w = skel_now.bind[bi];
        // .compatiblerim track offset (spacers / wheel offset), outboard
        let off = plan.as_ref().map(|p| p.track[ci / 2]).unwrap_or(0.0);
        let out = if w[12] < 0.0 { -off } else { off };
        w[12] += out;
        let t = [w[12], w[13], w[14]];
        wheel_pos.insert(corner.to_string(), t);
        let mut tx = skel_now.index.get(&format!("TYRE_{corner}"))
            .map(|&i| skel_now.bind[i]).unwrap_or(w);
        if skel_now.index.contains_key(&format!("TYRE_{corner}")) {
            tx[12] += out;
        }
        tyre_xf.insert(corner.to_string(), tx);

        let side = &corner[0..1];
        let planned = plan.as_ref().and_then(|p| p.rims[ci].as_deref())
            .map(|k| base_of(k).to_lowercase());
        let src = planned
            .and_then(|want| rim_lods.iter().find(|(n, _)| n.to_lowercase() == want))
            .or_else(|| rim_lods.iter().find(|(n, _)| corner_of(n).as_deref() == Some(corner)))
            .or_else(|| {
                rim_lods.iter().find(|(n, _)| {
                    corner_of(n).map(|c| c.starts_with(side)).unwrap_or(false)
                })
            })
            .or_else(|| rim_lods.first());
        let Some((src_name, proto)) = src else { continue };
        wheel_src.push((format!("rim_{corner}"), src_name.clone(), t));
        let mut lod = clone_lod(proto);
        // ⚠ Never re-centre. The rim is authored at the ORIGIN, and its own
        // centre is not the wheel centre - it carries the hub and brake face
        // offset inboard. Re-centring on the bounding box pushes the rim into
        // the tyre and leaves it sitting as a small disc inside the wheel.
        // The bone's full transform carries the camber (3 deg on the F1 cars).
        transform_lod(&mut lod, &w);
        let verts = lod.positions.len() / 12;
        let (bidx, bwgt) = rigid_skin(verts, bi);
        meshes.push(MeshEntry {
            name: format!("rim_{corner}"),
            vertices: verts as u32,
            batches: lod.batches.clone(),
            positions: Some(add(&mut blob, &lod.positions)),
            normals: Some(add(&mut blob, &lod.normals)),
            uvs: Some(add(&mut blob, &lod.uvs)),
            tangents: Some(add(&mut blob, &lod.tangents)),
            bone_idx: Some(add(&mut blob, &bidx)),
            bone_wgt: Some(add(&mut blob, &bwgt)),
            indices: Some(add(&mut blob, bytemuck::cast_slice(&lod.indices))),
            xforms: Vec::new(),
            facing: lod.facing,
            lod: None,
            lod_fade: [0.0; 4],
            shadow_mesh: None,
            colors: None,
            uv1: None,
        });
        // The motion-blur rim (effects): <car>_rim_blur_<corner>, placed like
        // the rim. Its materials (VehicleRim, InverseVelocity 0) stay
        // invisible until the wheel spins (shader rim_visible), so a parked
        // car draws exactly what it did before.
        // ⚠ Opt-in (EVOVIEW_RIM_BLUR=1): the blur materials take their colour
        // from a constant Red splat layer and their alpha from the per-layer
        // Red/Green/Blue_OpacityMap, which material.rs does not read yet - they
        // render as a solid silver disc.
        let blur_name = src_name.to_lowercase().replacen("_rim_", "_rim_blur_", 1);
        if blur_name != src_name.to_lowercase() && std::env::var("EVOVIEW_RIM_BLUR").map_or(false, |v| v == "1") {
            if let Some(k) = vfs.find(&blur_name, None) {
                let bb = vfs.read(&k);
                if let Some(mut bl) = read_mesh(&bb) {
                    transform_lod(&mut bl, &w);
                    let verts = bl.positions.len() / 12;
                    meshes.push(MeshEntry {
                        name: format!("rimblur_{corner}"),
                        vertices: verts as u32,
                        batches: bl.batches.clone(),
                        positions: Some(add(&mut blob, &bl.positions)),
                        normals: Some(add(&mut blob, &bl.normals)),
                        uvs: Some(add(&mut blob, &bl.uvs)),
                        tangents: Some(add(&mut blob, &bl.tangents)),
                        bone_idx: None,
                        bone_wgt: None,
                        indices: Some(add(&mut blob, bytemuck::cast_slice(&bl.indices))),
                        xforms: Vec::new(),
                        facing: bl.facing,
                        lod: None,
                        lod_fade: [0.0; 4],
                        shadow_mesh: None,
                        colors: None,
                        uv1: None,
                    });
                }
            }
        }
    }

    // ---- the rim design: which material each rim batch wears ----
    // ⚠ A design slot replaces a material by the BATCH NAME it is fitted to
    // (slot "EXT_RIM" -> every batch named EXT_RIM), not by material file
    // name. Rim styles keep their own materials (EXT_RIM_2, EXT_RIM_MG...),
    // so the old name-only override never reached them: the RS3's black
    // pack, the 296's two-tone and the STO's magnesium rims all stayed on
    // their style's default finish.
    if let Some(dk) = fitted_rim_design(vfs) {
        let enabled = preset_enabled_options(vfs);
        let db = vfs.read(&dk);
        let slots = read_design(&db);
        for me in meshes.iter_mut().filter(|m| m.name.starts_with("rim_")) {
            for b in me.batches.iter_mut() {
                // an optional slot (a logo option...) only when the preset turns it on
                let Some(slot) = slots.iter().find(|s| s.id.eq_ignore_ascii_case(&b.name)
                    && !s.material.is_empty()
                    && (!s.optional || enabled.contains(&base_name_lower(&s.material))))
                else { continue };
                if norm_mat_path(&slot.material) != norm_mat_path(&b.material) {
                    b.material = slot.material.clone();
                    shared_materials.push(slot.material.clone());
                }
            }
        }
    }

    // ---- tyres: shared geometry, placed on the same wheel bones ----
    let chosen_tyres = if let Some(p) = plan.as_ref().filter(|p| p.tyres.iter().any(|t| t.is_some())) {
        Some((p.tyres[0].clone().or_else(|| p.tyres[1].clone()),
              p.tyres[1].clone().or_else(|| p.tyres[0].clone())))
    } else if tyre_files.is_empty() {
        tyre_meshes(vfs)
    } else {
        Some((
            tyre_files.first().cloned(),
            tyre_files.get(1).or_else(|| tyre_files.first()).cloned(),
        ))
    };
    if let Some((front, rear)) = chosen_tyres {
        for (corner, pos) in wheel_pos.iter() {
            let src = if corner.ends_with('R') { &rear } else { &front };
            let Some(src) = src.clone() else { continue };
            let tb = vfs.read(&src);
            let Some(mut lod) = read_mesh(&tb) else { continue };
            wheel_src.push((format!("tyre_{corner}"), base_of(&src), *pos));
            // ⚠ Turn, don't mirror. One tyre mesh serves all four corners and
            // only its +X sidewall carries the branding (the -X half of the
            // texture is the plain inner wall). On the right-hand side that
            // face has to point outward, so the tyre is turned 180 degrees
            // about the vertical axis. A mirror in X gets the face out too but
            // flips handedness: the sidewall lettering read backwards.
            if pos[0] < 0.0 {
                turn_about_y(&mut lod);
            }
            match tyre_xf.get(corner) {
                Some(m) => transform_lod(&mut lod, m),
                None => place_tyre(&mut lod, *pos, false),
            }
            for b in &lod.batches {
                if !b.material.is_empty() {
                    shared_materials.push(b.material.clone());
                }
            }
            let verts = lod.positions.len() / 12;
            let tb = skel_now.index.get(&format!("TYRE_{corner}"))
                .or_else(|| skel_now.index.get(&format!("WHEEL_{corner}")));
            let skin = tb.map(|&b| rigid_skin(verts, b));
            meshes.push(MeshEntry {
                name: format!("tyre_{corner}"),
                vertices: verts as u32,
                batches: lod.batches.clone(),
                positions: Some(add(&mut blob, &lod.positions)),
                normals: Some(add(&mut blob, &lod.normals)),
                uvs: Some(add(&mut blob, &lod.uvs)),
                tangents: Some(add(&mut blob, &lod.tangents)),
                bone_idx: skin.as_ref().map(|s| add(&mut blob, &s.0)),
                bone_wgt: skin.as_ref().map(|s| add(&mut blob, &s.1)),
                indices: Some(add(&mut blob, bytemuck::cast_slice(&lod.indices))),
                xforms: Vec::new(),
                facing: lod.facing,
                lod: None,
                lod_fade: [0.0; 4],
                shadow_mesh: None,
                colors: None,
                uv1: None,
            });
        }
    }

    // ---- the driver, seated and posed (driver.rs) ----
    // Its rig joins the car skeleton (bones `drv:*`) so carrig.rs can steer,
    // pedal and shift it; past MAX_BONES it stays baked and static.
    let (driver_parts, rig) = crate::driver::load(vfs);
    let rig_off = skeleton_bones.len();
    let rig_fits = !rig.is_empty() && rig_off + rig.len() <= crate::render::MAX_BONES;
    if !rig.is_empty() && !rig_fits {
        println!("driver: rig of {} bones does not fit beside {} car bones; driver stays static",
                 rig.len(), rig_off);
    }
    if rig_fits {
        for b in &rig {
            skeleton_bones.push(Bone {
                name: b.name.clone(),
                parent: b.parent.map(|p| p + rig_off),
                local: b.local,
            });
        }
    }
    for (name, mut lod) in driver_parts {
        if rig_fits {
            for c in lod.bone_idx.chunks_exact_mut(4) {
                let v = f32::from_le_bytes(c.try_into().unwrap()) + rig_off as f32;
                c.copy_from_slice(&v.to_le_bytes());
            }
        } else {
            lod.bone_idx.clear();
            lod.bone_wgt.clear();
        }
        for b in &lod.batches {
            if !b.material.is_empty() {
                shared_materials.push(b.material.clone());
            }
        }
        let verts = lod.positions.len() / 12;
        meshes.push(MeshEntry {
            name,
            vertices: verts as u32,
            batches: lod.batches.clone(),
            positions: Some(add(&mut blob, &lod.positions)),
            normals: Some(add(&mut blob, &lod.normals)),
            uvs: Some(add(&mut blob, &lod.uvs)),
            tangents: Some(add(&mut blob, &lod.tangents)),
            bone_idx: (!lod.bone_idx.is_empty()).then(|| add(&mut blob, &lod.bone_idx)),
            bone_wgt: (!lod.bone_wgt.is_empty()).then(|| add(&mut blob, &lod.bone_wgt)),
            indices: Some(add(&mut blob, bytemuck::cast_slice(&lod.indices))),
            xforms: Vec::new(),
            facing: lod.facing,
            lod: None,
            lod_fade: [0.0; 4],
            shadow_mesh: None,
            colors: None,
            uv1: None,
        });
    }

    // ---- materials ----
    let mut mat_paths = vfs.list_rec("materials", "material");
    // livery materials replace the car's own, so they must come last
    let overrides = design_overrides(vfs);
    let mut override_slot: HashMap<String, String> = HashMap::new();
    for (slot_id, p) in &overrides {
        override_slot.insert(p.clone(), slot_id.clone());
        mat_paths.push(p.clone());
    }
    // A liveried race car takes its colour from a replacement material, not a
    // paint file, so "no paint" is only a problem when there is no livery
    // either.
    println!("liveries: {}", overrides.len());
    for r in &shared_materials.clone() {
        if let Some(p) = vfs.find(r, None) {
            if !mat_paths.contains(&p) {
                mat_paths.push(p);
            }
        }
    }
    let tyre_maps = tyre_textures(vfs);

    // The seated driver's maps load one mip down: they are 2-4K authored for
    // the game's cockpit camera, and full size they cost ~150 MB of VRAM for
    // a figure that fills a few hundred pixels of any view this has.
    let lod = CAR_TEX_SKIP.load(std::sync::atomic::Ordering::Relaxed);
    let drv_skip = |p: &str| -> u32 {
        lod + if crate::material::is_driver(&norm_mat_path(p)) { 1 } else { 0 }
    };
    let (drv_paths, car_paths): (Vec<String>, Vec<String>) =
        mat_paths.iter().cloned().partition(|p| drv_skip(p) == lod + 1);
    let keys = material_texture_keys(vfs, &car_paths);
    let mut bytes = vfs.prefetch_textures(&keys, lod);
    let keys = material_texture_keys(vfs, &drv_paths);
    bytes += vfs.prefetch_textures(&keys, lod + 1);
    blob.reserve(bytes);
    for path in &mat_paths.clone() {
        let skip = drv_skip(path);
        let mb = vfs.read(path);
        let Some(mut m) = read_material_vfs(vfs, &mb, &base_of(path)) else { continue };
        if let Some(slot) = override_slot.get(path) {
            m.file = format!("{slot}.material");
        }
        if m.shader.as_deref().map(|s| s.eq_ignore_ascii_case("VehicleTyre")).unwrap_or(false) {
            for (k, v) in &tyre_maps {
                m.slots.insert(k.clone(), v.clone());
            }
        }
        let mut resolved: HashMap<String, usize> = HashMap::new();
        for (slot, reference) in &m.slots.clone() {
            // ⚠ Never bind the EDITOR's placeholder textures. Materials whose
            // real map is supplied at runtime - mirrors, instrument clusters,
            // displays - still list an `editor/textures/default_material/...`
            // reference in a legacy slot like txDiffuse. Binding it paints a
            // grey CHECKERBOARD onto every one of those surfaces. With no
            // texture bound they fall back to their base colour, which is what
            // the material intends.
            let low = reference.replace('\\', "/").to_lowercase();
            // ⚠ Except an unauthored car part on a generic Uber/UberBiplanar
            // shader whose only colour map IS the editor's albedo2 (a flat
            // 50% grey, not the checker). The game samples it, so the part is
            // mid grey; skipping it left base colour 1.0 and the Dallara EXP's
            // front grille (ext_grid) rendered as a glowing white rectangle.
            let shader_l = m.shader.as_deref().unwrap_or("").to_lowercase();
            let flat_grey = slot == "Base_BaseColorMap"
                && low.ends_with("default_material/albedo2.texture")
                && (shader_l == "uberbiplanarmaterial" || shader_l == "ubermaterial");
            if !flat_grey && (low.contains("default_material") || low.starts_with("editor/")
                || low.contains("/null.texture") || low.ends_with("null.texture")) {
                continue;
            }
            let near = parent_of(path).to_string();
            let Some(found) = vfs.find(reference, Some(&near)) else {
                // a slot that does not resolve leaves the material sampling a
                // flat white texture, which reads as "untextured"
                unresolved.push(format!("{}:{}", base_of(path), slot));
                continue;
            };
            let idx = if let Some(&i) = tex_index.get(&found) {
                i
            } else {
                let Some(t) = vfs.texture(&found, skip) else { continue };
                if std::env::var_os("EVOVIEW_TEX_LOG").is_some() {
                    println!("car texture {:5}x{:<5} fmt {:2} {} {}", t.width, t.height, t.format, slot, found);
                }
                let name = base_of(&found).replace(".texture", "");
                let data = add(&mut blob, &t.data);
                textures.push(TextureEntry {
                    name,
                    format: t.format,
                    width: t.width,
                    height: t.height,
                    levels: t.levels,
                    data,
                });
                let i = textures.len() - 1;
                tex_index.insert(found.clone(), i);
                i
            };
            resolved.insert(slot.clone(), idx);
        }
        materials.push(MaterialEntry {
            file: m.file,
            path: norm_mat_path(path),
            shader: m.shader,
            slots: resolved,
            params: m.params,
            colors: m.colors.into_iter().collect(),
        });
    }
    // ⚠ A livery REPLACES a car material by NAME (it is renamed to the slot
    // id above). Lookup resolves by full path first, so the original's path
    // still won and every livery was silently dropped - the F2004 came out
    // plain red. Unregister the path of any material a livery overrides, so
    // its batches fall through to the name, which the livery owns.
    let overridden: std::collections::HashSet<String> = override_slot
        .values()
        .map(|s| s.to_lowercase())
        .collect();
    for me in materials.iter_mut() {
        let key = me.file.trim_end_matches(".material").to_lowercase();
        if overridden.contains(&key) && !override_slot.contains_key(&me.path)
            && !override_slot.keys().any(|k| norm_mat_path(k) == me.path)
        {
            me.path.clear();
        }
    }

    // ⚠ Some batches reference materials that live OUTSIDE the car - the
    // mirrors are in common_assets/materials. Resolve whatever is still
    // missing rather than leaving it on the fallback.
    let have: std::collections::HashSet<String> = materials
        .iter()
        .map(|m| m.file.to_lowercase().replace(".material", ""))
        .collect();
    let mut missing: Vec<String> = meshes
        .iter()
        .flat_map(|m| m.batches.iter())
        .map(|b| b.material.clone())
        .filter(|r| {
            let base = r
                .rsplit(['\\', '/'])
                .next()
                .unwrap_or(r)
                .to_lowercase()
                .replace(".material", "");
            !base.is_empty() && !have.contains(&base)
        })
        .collect();
    missing.sort();
    missing.dedup();
    for r in missing {
        let Some(key) = vfs.find(&r, None) else { continue };
        let mb = vfs.read(&key);
        let Some(m) = read_material_vfs(vfs, &mb, &base_of(&key)) else { continue };
        let mut resolved: HashMap<String, usize> = HashMap::new();
        for (slot, reference) in &m.slots {
            let near = parent_of(&key).to_string();
            let Some(found) = vfs.find(reference, Some(&near)) else { continue };
            let idx = if let Some(&i) = tex_index.get(&found) {
                i
            } else {
                let Some(t) = vfs.texture(&found, CAR_TEX_SKIP.load(std::sync::atomic::Ordering::Relaxed)) else { continue };
                let name = base_of(&found).replace(".texture", "");
                let data = add(&mut blob, &t.data);
                textures.push(TextureEntry {
                    name, format: t.format, width: t.width, height: t.height,
                    levels: t.levels, data,
                });
                let i = textures.len() - 1;
                tex_index.insert(found.clone(), i);
                i
            };
            resolved.insert(slot.clone(), idx);
        }
        materials.push(MaterialEntry {
            file: m.file, path: norm_mat_path(&key), shader: m.shader,
            slots: resolved,
            params: m.params, colors: m.colors.into_iter().collect(),
        });
    }

    if !unresolved.is_empty() {
        println!("unresolved textures: {} slot(s)", unresolved.len());
        let mut u = unresolved.clone();
        u.truncate(14);
        println!("  {}", u.join(", "));
    }
    apply_paints(vfs, &mut materials);
    add_contact_decals(vfs, &wheel_pos, &mut meshes, &mut textures, &mut materials, &mut blob);
    // race numbers, projected onto the bodywork (numbers.rs)
    crate::numbers::add(vfs, &skel_now.index, &skel_now.bind, &mut meshes, &mut textures, &mut materials, &mut blob);
    // the shared windshield rain maps (rain.rs)
    let mut rain_tex = [None; 3];
    for (i, n) in ["rain_drop_mask", "rain_drop_alpha", "rain_drop_nm"].iter().enumerate() {
        let Some(found) = vfs.find(&format!("content/cars/common_assets/rain/{n}.texture"), None) else { continue };
        let hdr = vfs.read(&found);
        let mips = vfs.read(&found.replace(".texture", ".texturemips"));
        let Some(t) = read_texture(&hdr, &mips) else { continue };
        // ⚠ the mask and alpha are format 27: single-channel BC4, which reads
        // as BC1 noise (the contact-shadow trap, contact.rs)
        let format = if t.format == "bc1" && i < 2 { "bc4".to_string() } else { t.format };
        let data = add(&mut blob, &t.data);
        textures.push(TextureEntry { name: n.to_string(), format, width: t.width, height: t.height, levels: t.levels, data });
        rain_tex[i] = Some(textures.len() - 1);
    }
    // the side windows' streak maps (Rain_RS*), decoded and packed
    let mut streak_tex = |n: &str| -> Option<Tex> {
        let found = vfs.find(&format!("content/cars/common_assets/rain/{n}.texture"), None)?;
        let hdr = vfs.read(&found);
        let mips = vfs.read(&found.replace(".texture", ".texturemips"));
        read_texture(&hdr, &mips)
    };
    let flows = ["vertical", "diagonal", "horizontal"];
    let masks: Vec<Tex> = flows.iter().filter_map(|f| streak_tex(&format!("side_rain_flow{f}_mask"))).collect();
    let nms: Vec<Tex> = flows.iter().filter_map(|f| streak_tex(&format!("side_rain_flow{f}_nm"))).collect();
    let rain_streaks = crate::rain::pack_streaks(&masks, &nms).map(std::sync::Arc::new);

    // ---- animations ----
    let mut anim_keys = vfs.list_rec("animations", "animation");
    if anim_keys.is_empty() {
        let refs = actor_refs(vfs);
        for r in &refs {
            if r.to_lowercase().ends_with(".animation") {
                if let Some(k) = vfs.find(r, None) {
                    if !anim_keys.contains(&k) {
                        anim_keys.push(k);
                    }
                }
            }
        }
        anim_keys.sort();
    }
    let animations: Vec<Animation> = anim_keys
        .iter()
        .filter_map(|k| {
            let b = vfs.read(k);
            read_animation(&b, base_of(k).trim_end_matches(".animation"))
        })
        .collect();
    // ⚠ A door is TWO animations. `<car>_ext_door_l` swings DOOR_L on the body
    // and `<car>_int_door_l` swings DOOR_L_1 on the interior - different files,
    // different bone names, one physical door. Playing either alone opens the
    // outer skin and leaves the trim panel hanging in the doorway. Fold each
    // ext/int pair into one animation so a single key moves the whole door.
    let mut merged: Vec<Animation> = Vec::new();
    let mut by_part: HashMap<String, usize> = HashMap::new();
    for a in animations {
        let part = a.name.replace("_ext_", "_").replace("_int_", "_");
        match by_part.get(&part) {
            Some(&i) => {
                let m = &mut merged[i];
                m.frames = m.frames.max(a.frames);
                m.tracks.extend(a.tracks);
            }
            None => {
                by_part.insert(part.clone(), merged.len());
                merged.push(Animation { name: part, ..a });
            }
        }
    }
    let mut animations = merged;
    animations.sort_by(|a, b| a.name.cmp(&b.name));

    if std::env::var_os("EVOVIEW_WHEEL_AUDIT").is_some() {
        wheel_audit(&wheel_src, &meshes, &materials, &textures, &tyre_maps);
    }
    let mut skeleton = Skeleton::build(skeleton_bones);
    let rig = crate::carrig::Rig::build(vfs, &skeleton, &meshes, &blob);
    skeleton.rig = Some(Box::new(rig));
    // lights, LEDs, displays, particles (fxdata.rs) - adds its textures
    let mut fx = crate::fxdata::load(vfs, &wheel_pos, &mut blob, &mut textures, &mut tex_index);
    if let Some(f) = fx.as_mut() {
        f.unresolved = unresolved.clone();
    }
    vfs.drop_prefetched();
    let mut car = Car::from_parts(meshes, textures, materials, blob);
    car.fx = fx;
    car.tex_src = vec![String::new(); car.textures.len()];
    for (k, &i) in &tex_index {
        if let Some(s) = car.tex_src.get_mut(i) {
            *s = k.clone();
        }
    }
    car.env = studio_env(vfs);
    car.brdf = brdf_lut(vfs);
    car.rain_tex = rain_tex;
    car.rain_streaks = rain_streaks;
    if let Ok(mode) = std::env::var("EVOVIEW_RENDER_AUDIT") {
        render_audit(vfs, &car, &mesh_audit, &unresolved);
        if mode == "exit" {
            std::process::exit(0);
        }
    }
    Ok(Loaded {
        car,
        skeleton,
        animations,
    })
}

/// One JSON line (`RENDER_AUDIT {...}`): everything the car ships or its
/// actor names that could render, and what the load did with it - files
/// never opened (by folder and kind), what became of every mesh file, batch
/// materials that resolve to nothing, texture slots that did not resolve,
/// and which shaders the materials use.
fn render_audit(vfs: &Vfs, car: &Car, mesh_audit: &[(String, &'static str)], unresolved: &[String]) {
    let root = vfs.root().to_string();
    let rel = |k: &str| k.strip_prefix(root.as_str()).unwrap_or(k).to_string();
    let mut unread: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    let mut counts: std::collections::BTreeMap<String, [u32; 2]> = Default::default();
    for (k, read) in vfs.audit_files() {
        let r = rel(&k);
        let ext = r.rsplit('.').next().unwrap_or("").to_string();
        // the folder (first two levels) and the kind
        let dir: Vec<&str> = r.split('/').collect();
        let folder = dir[..dir.len() - 1].iter().take(2).cloned().collect::<Vec<_>>().join("/");
        let key = format!("{}/*.{}", if folder.is_empty() { "." } else { &folder }, ext);
        let c = counts.entry(key.clone()).or_default();
        c[0] += 1;
        if read {
            c[1] += 1;
        } else if !r.ends_with(".texturemips") {
            unread.entry(key).or_default().push(r);
        }
    }
    // batches whose material resolves to nothing (Materials::lookup: full
    // path, then file name)
    let norm = |p: &str| p.replace('\\', "/").to_lowercase();
    let by_path: std::collections::HashSet<String> = car.materials.iter().map(|m| m.path.clone()).collect();
    let by_name: std::collections::HashSet<String> = car.materials.iter()
        .map(|m| base_name_lower(&m.file)).chain(car.materials.iter().map(|m| base_name_lower(&m.path))).collect();
    let mut no_mat: Vec<String> = Vec::new();
    let mut tris_no_mat = 0u64;
    for m in &car.meshes {
        for b in &m.batches {
            let ok = !b.material.is_empty()
                && (by_path.contains(&norm(&b.material)) || by_name.contains(&base_name_lower(&b.material)));
            if !ok {
                tris_no_mat += b.count as u64 / 3;
                no_mat.push(format!("{}:{} -> '{}'", m.name, b.name, b.material));
            }
        }
    }
    let mut shaders: std::collections::BTreeMap<String, u32> = Default::default();
    for m in &car.materials {
        *shaders.entry(m.shader.clone().unwrap_or_default()).or_default() += 1;
    }
    let meshes: Vec<_> = mesh_audit.iter().map(|(k, d)| serde_json::json!([rel(k), d])).collect();
    // what the actor names that was not read, or lives outside the car folder
    let mut external: Vec<serde_json::Value> = Vec::new();
    if let Some(a) = vfs.list("", "actor").into_iter().next() {
        let d = vfs.read_inner(&a);
        let mut refs = Vec::new();
        strings_under(&d, 0, d.len(), 0, &mut refs);
        refs.sort();
        refs.dedup();
        for r in refs {
            let low = norm(&r);
            let Some(ext) = low.rsplit_once('.').map(|x| x.1.to_string()) else { continue };
            if ext.is_empty() || ext.len() > 24 || low.contains(' ') {
                continue;
            }
            let k = vfs.find(&r, None);
            let read = k.as_deref().map_or(false, |k| vfs.was_read(k));
            if !read || k.as_deref().map_or(true, |k| !k.starts_with(root.as_str())) {
                external.push(serde_json::json!({"ref": r, "found": k.is_some(), "read": read}));
            }
        }
    }
    println!("RENDER_AUDIT {}", serde_json::json!({
        "root": root,
        "files": counts.iter().map(|(k, c)| (k.clone(), serde_json::json!(c))).collect::<serde_json::Map<_, _>>(),
        "unread": unread,
        "meshes": meshes,
        "batches_without_material": no_mat,
        "tris_without_material": tris_no_mat,
        "unresolved_slots": unresolved,
        "shaders": shaders,
        "actor_refs": external,
        "drawn_meshes": car.meshes.iter().map(|m| m.name.clone()).collect::<Vec<_>>(),
    }));
}

/// One JSON line (`WHEEL_AUDIT {...}`) saying what was fitted at each wheel
/// and which material every wheel batch really resolves to, the way
/// `Materials::lookup` will: by full path, then by file name.
fn wheel_audit(
    src: &[(String, String, [f32; 3])],
    meshes: &[MeshEntry],
    materials: &[MaterialEntry],
    textures: &[TextureEntry],
    tyre_maps: &HashMap<String, String>,
) {
    use serde_json::json;
    let resolve = |r: &str| -> Option<&MaterialEntry> {
        let np = norm_mat_path(r);
        materials.iter().rev().find(|m| !m.path.is_empty() && m.path == np).or_else(|| {
            let b = base_name_lower(r).trim_end_matches(".material").to_string();
            materials.iter().rev().find(|m| m.file.to_lowercase().trim_end_matches(".material") == b)
        })
    };
    let mat_json = |m: &MaterialEntry| {
        let mut slots: Vec<(String, String)> = m.slots.iter()
            .map(|(k, &v)| (k.clone(), textures.get(v).map(|t| t.name.clone()).unwrap_or_default()))
            .collect();
        slots.sort();
        let col = |k: &str| m.colors.get(k).cloned();
        json!({"file": m.file, "path": m.path, "shader": m.shader,
               "slots": slots, "base": col("Base_Basecolor"), "red": col("Red_Basecolor"),
               "metal": m.params.get("Base_Metalness").or(m.params.get("Red_Metalness"))})
    };
    let mut wheels = Vec::new();
    for (name, from, pos) in src {
        let Some(me) = meshes.iter().find(|m| &m.name == name) else { continue };
        let batches: Vec<_> = me.batches.iter().map(|b| json!({
            "name": b.name, "ref": base_name_lower(&b.material),
            "mat": resolve(&b.material).map(|m| mat_json(m)),
        })).collect();
        wheels.push(json!({"mesh": name, "from": from, "pos": pos, "batches": batches}));
    }
    let calipers: Vec<_> = materials.iter()
        .filter(|m| { let f = m.file.to_lowercase(); f.contains("calip") || f.contains("brake") || f.contains("disc") })
        .map(|m| mat_json(m)).collect();
    let parts: Vec<&str> = meshes.iter().filter(|m| m.name.starts_with("part_"))
        .map(|m| m.name.as_str()).collect();
    println!("WHEEL_AUDIT {}", json!({"wheels": wheels, "calipers": calipers,
                                       "tyre_maps": tyre_maps, "parts": parts}));
}

/// Which rim meshes and tyre this car actually fits, from its `.rimmesh`.
///
/// ⚠ Not every car keeps its rims in `meshes/`. Cars with wheel options put
/// each set in `meshes/rims/<option>/` and name the chosen one in a
/// `.rimmesh` under `skins/`, which also names the `.tyremesh`. Reading the
/// top level only leaves those cars with no wheels at all; loading every
/// subfolder stacks five rim sets on top of each other.
///
/// `meshes/parts/` is never loaded wholesale: those are alternative fitments
/// (seats, steering wheels, roll cage). parts.rs draws the ones the presets fit.
/// The car's visual preset - i.e. which TRIM it wears.
///
/// ⚠ Never just take the first one. A car ships one visual preset per trim
/// (the AE86 has four), so `.next()` renders every trim identically - the
/// exact bug that made a whole fleet of "different" trims look the same.
/// A player's own look (online, mp.rs): their CarFinalStateData {10 body {2
/// design "presets/x.visualcarpreset", 3 multilayer_states {1 key, 2 state}}}.
/// The state has the visual preset's field-8 layout (1 material, 2-4
/// channels {10 oem colour}, 5 optional), so the design's own preset with
/// its field-8 entries swapped for the player's is their look as a preset.
pub(crate) fn netcar_visual(vfs: &mut Vfs, fs: &[u8]) -> Option<Vec<u8>> {
    let mut design = String::new();
    let mut states: Vec<(usize, usize)> = Vec::new();
    fields(fs, 0, fs.len(), |f, v| {
        let (10, Some((a, b))) = (f, v.range()) else { return };
        fields(fs, a, b, |f, v| match (f, v.range()) {
            (2, Some(r)) => design = text(fs, r),
            (3, Some((a, b))) => fields(fs, a, b, |f, v| {
                if let (2, Some(r)) = (f, v.range()) {
                    states.push(r);
                }
            }),
            _ => {}
        });
    });
    let key = vfs.find(&design, None)?;
    let base = vfs.read(&key);
    let mut out = Vec::with_capacity(base.len() + fs.len());
    // the preset's fields but its field-8 entries, verbatim
    let mut i = 0usize;
    let mut keep: Vec<(usize, usize)> = Vec::new();
    while i < base.len() {
        let start = i;
        let Some((tag, ni)) = read_varint(&base, i) else { break };
        i = ni;
        let (f, wt) = (tag >> 3, tag & 7);
        let end = match wt {
            0 => match read_varint(&base, i) { Some((_, n)) => n, None => break },
            1 => i + 8,
            2 => match read_varint(&base, i) { Some((l, n)) => n + l as usize, None => break },
            5 => i + 4,
            _ => break,
        };
        if end > base.len() {
            break;
        }
        if f != 8 {
            keep.push((start, end));
        }
        i = end;
    }
    for (a, b) in keep {
        out.extend_from_slice(&base[a..b]);
    }
    for (a, b) in states {
        let st = custom_paints(vfs, &fs[a..b]);
        put_len(&mut out, 8, &st);
    }
    Some(out)
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn put_len(out: &mut Vec<u8>, field: u32, body: &[u8]) {
    put_varint(out, ((field as u64) << 3) | 2);
    put_varint(out, body.len() as u64);
    out.extend_from_slice(body);
}

/// The raw fields of `d`: (field, wire type, whole field's byte range, the
/// payload's range).
fn raw_fields(d: &[u8]) -> Vec<(u32, u64, (usize, usize), (usize, usize))> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < d.len() {
        let start = i;
        let Some((tag, ni)) = read_varint(d, i) else { break };
        i = ni;
        let (f, wt) = ((tag >> 3) as u32, tag & 7);
        let (ps, end) = match wt {
            0 => match read_varint(d, i) { Some((_, n)) => (i, n), None => break },
            1 => (i, i + 8),
            2 => match read_varint(d, i) { Some((l, n)) => (n, n + l as usize), None => break },
            5 => (i, i + 4),
            _ => break,
        };
        if end > d.len() {
            break;
        }
        out.push((f, wt, (start, end), (ps, end)));
        i = end;
    }
    out
}

/// A player's material state (1 material, 2-4 channels {1 colour {x y z}, 2
/// metalness, 3 roughness, 4 clear coat, 5 its roughness, 6 normal
/// intensity, 10 oem colour path}): a channel painted in a CUSTOM colour
/// (a colour, no oem path) gets a made-up .oemmultilayercolor {3 colour, 4
/// metallic, 5 roughness, 6 clear coat, 7 its roughness, 8 normal} added
/// to the car and named in its field 10 - the paint code only reads oem files.
/// An empty colour (unused channels carry one) is left alone.
fn custom_paints(vfs: &mut Vfs, st: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(st.len() + 64);
    for (f, wt, whole, pay) in raw_fields(st) {
        if !(2..=4).contains(&f) || wt != 2 {
            out.extend_from_slice(&st[whole.0..whole.1]);
            continue;
        }
        let ch = &st[pay.0..pay.1];
        let cf = raw_fields(ch);
        let has_oem = cf.iter().any(|c| c.0 == 10 && c.3.1 > c.3.0);
        let colour = cf.iter().find(|c| c.0 == 1 && c.1 == 2).map(|c| &ch[c.3.0..c.3.1]).filter(|c| !c.is_empty());
        let Some(colour) = colour.filter(|_| !has_oem) else {
            out.extend_from_slice(&st[whole.0..whole.1]);
            continue;
        };
        let mut oem = Vec::new();
        put_len(&mut oem, 3, colour);
        for (from, to) in [(2u32, 4u32), (3, 5), (4, 6), (5, 7), (6, 8)] {
            if let Some(c) = cf.iter().find(|c| c.0 == from && c.1 == 5) {
                put_varint(&mut oem, ((to as u64) << 3) | 5);
                oem.extend_from_slice(&ch[c.3.0..c.3.1]);
            }
        }
        let mut h: u64 = 0xcbf29ce484222325;
        for b in &oem {
            h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
        }
        let name = format!("mp_c_{h:016x}.oemmultilayercolor");
        vfs.add_file(&format!("presets/{name}"), oem);
        let mut nch = ch.to_vec();
        put_len(&mut nch, 10, name.as_bytes());
        put_len(&mut out, f, &nch);
    }
    out
}

fn read_varint(d: &[u8], mut i: usize) -> Option<(u64, usize)> {
    let mut r = 0u64;
    let mut s = 0u32;
    loop {
        let b = *d.get(i)?;
        i += 1;
        r |= ((b & 0x7F) as u64) << s;
        if b & 0x80 == 0 {
            return Some((r, i));
        }
        s += 7;
        if s > 63 {
            return None;
        }
    }
}

/// ACECM names the trim it wants in EVOVIEW_VISUAL (e.g. preset_ae86_visual_2);
/// honour that, and fall back to the first only when nothing matches.
pub(crate) fn pick_visual_preset(vfs: &mut Vfs) -> Option<String> {
    let all = vfs.list("presets", "visualcarpreset");
    let wanted = std::env::var("EVOVIEW_VISUAL")
        .unwrap_or_default()
        .to_lowercase();
    // Exact match on the file stem: `contains` let "visual_1" pick visual_10.
    let wanted = wanted.trim().trim_end_matches(".visualcarpreset").to_string();
    if !wanted.is_empty() {
        if let Some(hit) = all.iter().find(|k| {
            k.replace('\\', "/")
                .rsplit('/')
                .next()
                .unwrap_or(k)
                .to_lowercase()
                .trim_end_matches(".visualcarpreset")
                == wanted
        }) {
            return Some(hit.clone());
        }
        println!("EVOVIEW_VISUAL {wanted}: no such visual preset, using the first");
    }
    // A chosen mech preset (EVOVIEW_MECH, or a mod's json) wears its own
    // default trim: the first of its visual_preset_paths (field 8), which is
    // the pairing the game's thumbnails and final states use.
    let mech = std::env::var("EVOVIEW_MECH").unwrap_or_default().to_lowercase();
    let mech = mech.trim().trim_end_matches(".mechanicalcarpreset");
    if !mech.is_empty() {
        let mk = vfs.list("presets", "mechanicalcarpreset").into_iter()
            .find(|k| base_name_lower(k).trim_end_matches(".mechanicalcarpreset") == mech);
        if let Some(mk) = mk {
            let d = vfs.read(&mk);
            let mut first = None;
            fields(&d, 0, d.len(), |f, v| {
                if let (8, Some(r)) = (f, v.range()) {
                    if first.is_none() {
                        let b = base_name_lower(&text(&d, r));
                        first = all.iter().find(|k| base_name_lower(k) == b).cloned();
                    }
                }
            });
            if first.is_some() {
                return first;
            }
        }
    }
    all.into_iter().next()
}

/// The paints the car's visual preset actually selects.
///
/// ⚠ A design lists every colour a slot COULD wear; the preset says which one
/// it does. Taking the design's first option instead gives whatever the artist
/// happened to list first - usually white or silver - so most of the fleet
/// came out white despite being "painted".
pub(crate) fn preset_paints(vfs: &mut Vfs) -> Vec<String> {
    let Some(preset) = pick_visual_preset(vfs) else {
        return Vec::new();
    };
    let d = vfs.read(&preset);
    let mut refs = Vec::new();
    strings_under(&d, 0, d.len(), 0, &mut refs);
    refs.iter()
        .filter(|r| r.to_lowercase().ends_with(".oemmultilayercolor"))
        .map(|r| {
            r.replace('\\', "/")
                .rsplit('/')
                .next()
                .unwrap_or(r)
                .to_lowercase()
        })
        .collect()
}

/// The design files the car wears BY DEFAULT: its body design and its rim
/// design, as named by the first visual preset.
///
/// ⚠ A car can ship several designs (the RS6 has Nardo Grey, Nogaro Blue and
/// more). Applying them all in turn means the last one silently wins, so the
/// colour is arbitrary and the rim paint can target a rim variant the car is
/// not actually wearing - leaving the fitted rims unpainted. Pick one.
fn default_designs(vfs: &mut Vfs) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    // ⚠ The preset lists a rim design for EVERY rim style (field 6); only the
    // one the fitted rims accept is worn. Painting them all let the last
    // style's colour win on a shared material.
    let fitted_rim = fitted_rim_design(vfs);
    if let Some(r) = &fitted_rim {
        out.push(r.clone());
    }
    if let Some(preset) = pick_visual_preset(vfs) {
        let d = vfs.read(&preset);
        let mut refs = Vec::new();
        strings_under(&d, 0, d.len(), 0, &mut refs);
        let mut rim_designs: Vec<String> = Vec::new();
        fields(&d, 0, d.len(), |f, v| {
            if let (6, Some(r)) = (f, v.range()) {
                rim_designs.push(text(&d, r));
            }
        });
        if fitted_rim.is_some() {
            refs.retain(|r| !rim_designs.contains(r));
        }
        for r in &refs {
            let low = r.to_lowercase();
            if low.ends_with(".cardesign") {
                // a .cardesign points at the .design that holds the slots
                if let Some(k) = vfs.find(r, None) {
                    let cd = vfs.read(&k);
                    let mut inner = Vec::new();
                    strings_under(&cd, 0, cd.len(), 0, &mut inner);
                    for i in inner {
                        if i.to_lowercase().ends_with(".design") {
                            if let Some(dk) = vfs.find(&i, None) {
                                if !out.contains(&dk) {
                                    out.push(dk);
                                }
                            }
                        }
                    }
                }
            } else if low.ends_with(".design") {
                if let Some(k) = vfs.find(r, None) {
                    if !out.contains(&k) {
                        out.push(k);
                    }
                }
            }
        }
    }
    if out.is_empty() {
        out = vfs.list("skins", "design").into_iter().take(1).collect();
    }
    out
}

/// Everything the car's `.actor` points at.
///
/// ⚠ Some cars are VARIANTS: the Golf R ships four body meshes and nothing
/// else, and takes its rims, tyres and animations from the GTI's folder. Their
/// own `meshes/`, `skins/` and `animations/` are empty, so without reading the
/// actor they load with no wheels and no animations at all.
/// CarActorData.BaseMeshes roles worn with a driver in the seat (seatbelt_on,
/// safety_net_on, harness_on) and with the seat empty (seatbelt_off,
/// harness_off, safety_net_off).
const BASE_MESH_ON: [u32; 3] = [12, 13, 23];
const BASE_MESH_OFF: [u32; 3] = [22, 24, 25];

/// The actor's base meshes by role: lowercase file name -> BaseMeshes field
/// (ActorData f103 car -> f12 base_meshes -> role {f1 mesh_path}).
fn actor_base_meshes(vfs: &mut Vfs) -> HashMap<String, u32> {
    let mut out = HashMap::new();
    let Some(a) = vfs.list("", "actor").into_iter().next() else { return out };
    let d = vfs.read(&a);
    fields(&d, 0, d.len(), |f, v| {
        let (103, Some((a, b))) = (f, v.range()) else { return };
        fields(&d, a, b, |f, v| {
            let (12, Some((a, b))) = (f, v.range()) else { return };
            fields(&d, a, b, |role, v| {
                // f1 body_mesh is a plain path, every other role a BaseMesh
                let (true, Some((a, b))) = (role != 1, v.range()) else { return };
                fields(&d, a, b, |f, v| {
                    if let (1, Some(r)) = (f, v.range()) {
                        out.insert(base_name_lower(&text(&d, r)), role);
                    }
                });
            });
        });
    });
    out
}

fn actor_refs(vfs: &mut Vfs) -> Vec<String> {
    let Some(a) = vfs.list("", "actor").into_iter().next() else { return Vec::new() };
    let d = vfs.read(&a);
    let mut refs = Vec::new();
    strings_under(&d, 0, d.len(), 0, &mut refs);
    refs
}

/// Which bone the car's `.actor` parents each mesh to.
///
/// The actor is the game's own answer to "what is fitted, and where does it
/// go": every entry pairs a mesh path with an optional bone name. Wheels
/// (`WHEEL_xx`) and tyres (`TYRE_xx`) were already placed by matching names,
/// but 30 cars also hang a high-detail steering wheel off `STEER_BASE` and 43
/// hang a number plate off `PLATE_R`/`PLATE_F`. Those parts are authored at
/// the ORIGIN, so without the actor they load as ordinary body meshes and are
/// drawn on the ground under the car.
///
/// Read structurally, not by field number: find any submessage whose field 1
/// is a `.mesh` path and whose field 2 is a bone name.
fn actor_attachments(vfs: &mut Vfs) -> Vec<(String, String)> {
    let Some(a) = vfs.list("", "actor").into_iter().next() else { return Vec::new() };
    let d = vfs.read(&a);
    let mut out = Vec::new();
    scan_attachments(&d, 0, d.len(), 0, &mut out);
    out
}

fn scan_attachments(
    d: &[u8],
    s: usize,
    e: usize,
    depth: u32,
    out: &mut Vec<(String, String)>,
) {
    if depth > 8 {
        return;
    }
    let mut slot: [Option<String>; 2] = [None, None];
    fields(d, s, e, |f, v| {
        let Some((a, b)) = v.range() else { return };
        let t = &d[a..b];
        let printable = !t.is_empty() && t.iter().all(|c| (0x20..0x7f).contains(c));
        if printable {
            if f == 1 || f == 2 {
                slot[f as usize - 1] = Some(String::from_utf8_lossy(t).into_owned());
            }
        } else {
            scan_attachments(d, a, b, depth + 1, out);
        }
    });
    if let (Some(p), Some(bone)) = (&slot[0], &slot[1]) {
        if p.to_lowercase().ends_with(".mesh") && !bone.is_empty() {
            out.push((p.clone(), bone.clone()));
        }
    }
}

pub(crate) fn base_name_lower(p: &str) -> String {
    p.replace('\\', "/").rsplit('/').next().unwrap_or_default().to_lowercase()
}

/// The game's resolved state for the active mech + visual preset pair: a
/// `presets/<car>_<mech>_<visual>.carfinalstate` (CarFinalStateData,
/// Customization.proto in the exe).
///   10 body  {1 mech preset path, 2 visual preset path, 3 material states}
///   20 rims  {1 .compatiblerim, 2 rim .design}
///   30 parts [{1 .compatiblepart, 2 design}]
/// The fleet ships 271 of them (not one per pair), and every car a
/// third-party tool builds ships one per visual preset. ⚠ For those cars it
/// is the ONLY place that fits anything: their mech preset names a stock
/// compatiblerim and no parts, the actor lists the body kit as an
/// aftermarket option (CarActorData.Customization f21 - availability, not a
/// fitment) and the final state fits it, plus a swapped wheel set.
#[derive(Default, Debug, Clone)]
pub(crate) struct FinalState {
    pub rims: Option<String>,
    pub rim_design: Option<String>,
    pub parts: Vec<String>,
}

/// The raw `.carfinalstate` of the active mech + visual pair.
pub(crate) fn final_state_bytes(vfs: &mut Vfs) -> Option<Vec<u8>> {
    let mech = base_name_lower(&pick_mech_preset(vfs)?);
    let vis = base_name_lower(&pick_visual_preset(vfs)?);
    for k in vfs.list("presets", "carfinalstate") {
        let d = vfs.read(&k);
        let (mut bm, mut bv) = (String::new(), String::new());
        fields(&d, 0, d.len(), |f, v| {
            if let (10, Some((a, b))) = (f, v.range()) {
                fields(&d, a, b, |sf, sv| match (sf, sv.range()) {
                    (1, Some(r)) => bm = base_name_lower(&text(&d, r)),
                    (2, Some(r)) => bv = base_name_lower(&text(&d, r)),
                    _ => {}
                });
            }
        });
        if bm == mech && bv == vis {
            return Some(d);
        }
    }
    None
}

pub(crate) fn final_state(vfs: &mut Vfs) -> Option<FinalState> {
    let mech = base_name_lower(&pick_mech_preset(vfs)?);
    let vis = base_name_lower(&pick_visual_preset(vfs)?);
    for k in vfs.list("presets", "carfinalstate") {
        let d = vfs.read(&k);
        let (mut bm, mut bv) = (String::new(), String::new());
        let mut fs = FinalState::default();
        fields(&d, 0, d.len(), |f, v| {
            let Some((a, b)) = v.range() else { return };
            match f {
                10 => fields(&d, a, b, |sf, sv| match (sf, sv.range()) {
                    (1, Some(r)) => bm = base_name_lower(&text(&d, r)),
                    (2, Some(r)) => bv = base_name_lower(&text(&d, r)),
                    _ => {}
                }),
                20 => fields(&d, a, b, |sf, sv| match (sf, sv.range()) {
                    (1, Some(r)) => fs.rims = Some(text(&d, r)),
                    (2, Some(r)) => fs.rim_design = Some(text(&d, r)),
                    _ => {}
                }),
                30 => fields(&d, a, b, |sf, sv| {
                    if let (1, Some(r)) = (sf, sv.range()) {
                        fs.parts.push(text(&d, r));
                    }
                }),
                _ => {}
            }
        });
        if bm == mech && bv == vis {
            return Some(fs);
        }
    }
    None
}

/// The first rim design the visual preset fits (field 6), as a lowercase
/// file name, e.g. `rimdesign_ae86_intra.design`.
fn preset_rim_design(vfs: &mut Vfs) -> Option<String> {
    let preset = pick_visual_preset(vfs)?;
    let d = vfs.read(&preset);
    let mut first = None;
    fields(&d, 0, d.len(), |f, v| {
        if f == 6 && first.is_none() {
            if let Some(r) = v.range() {
                first = Some(base_name_lower(&text(&d, r)));
            }
        }
    });
    first
}

/// The car's MECHANICAL preset, which is what fits the wheels.
///
/// A thumbnail is named `<mech preset>-<visual preset>`, and it is the mech
/// preset that names the `.compatiblerim`. EVOVIEW_MECH picks one by file
/// stem; otherwise take the first mech preset that offers the chosen visual
/// preset (its field 8 lists them), else the first one.
pub(crate) fn pick_mech_preset(vfs: &mut Vfs) -> Option<String> {
    let all = vfs.list("presets", "mechanicalcarpreset");
    let stem = |k: &str| base_name_lower(k).trim_end_matches(".mechanicalcarpreset").to_string();
    let wanted = std::env::var("EVOVIEW_MECH").unwrap_or_default().to_lowercase();
    let wanted = wanted.trim().trim_end_matches(".mechanicalcarpreset").to_string();
    if !wanted.is_empty() {
        if let Some(hit) = all.iter().find(|k| stem(k) == wanted) {
            return Some(hit.clone());
        }
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| println!("EVOVIEW_MECH {wanted}: no such mechanical preset"));
    }
    if let Some(vis) = pick_visual_preset(vfs).map(|v| base_name_lower(&v)) {
        for k in &all {
            let d = vfs.read(k);
            let mut hit = false;
            fields(&d, 0, d.len(), |f, v| {
                if let (8, Some(r)) = (f, v.range()) {
                    hit |= base_name_lower(&text(&d, r)) == vis;
                }
            });
            if hit {
                return Some(k.clone());
            }
        }
    }
    all.into_iter().next()
}

/// What the game fits at each wheel, read the way the game reads it.
///
///   mechanical preset f5 -> `.compatiblerim`
///     f1 front `.rimmesh`  f3 front `.tyremesh`  f2 front track offset (m)
///     f4 rear  `.rimmesh`  f6 rear  `.tyremesh`  f5 rear  track offset (m)
///   `.rimmesh` f5 left mesh, f6 right mesh, f10 the rim designs it takes
///   `.tyremesh` f7 the tyre mesh
///
/// ⚠ Front and rear are SEPARATE `.rimmesh` files on every staggered car (the
/// GTAm runs 8.5J/265 front and 10.5J/285 rear). Taking one `.rimmesh` - the
/// old "only the first one" rule - put the rear rim and tyre on all four
/// corners, or the front on all four, depending on how the sizes sorted.
/// And a car with rim options keeps every style's meshes side by side, so
/// matching rim files by corner name picked whichever style sorted first.
struct WheelPlan {
    /// vfs keys for LF, RF, LR, RR
    rims: [Option<String>; 4],
    /// tyre `.mesh` keys, front and rear
    tyres: [Option<String>; 2],
    /// lateral offset per axle, + = outboard
    track: [f32; 2],
    /// rim design files (lowercase base names) the fitted `.rimmesh`es accept
    designs: Vec<String>,
    /// "mech preset -> compatiblerim", for the log
    label: String,
}

fn wheel_plan(vfs: &mut Vfs) -> Option<WheelPlan> {
    let mech = pick_mech_preset(vfs)?;
    let md = vfs.read(&mech);
    let mut crim_ref = None;
    fields(&md, 0, md.len(), |f, v| {
        if let (5, Some(r)) = (f, v.range()) {
            crim_ref.get_or_insert_with(|| text(&md, r));
        }
    });
    // The final state's wheel set wins over the mech preset's: a swapped
    // set (the AZ-1 conversion's Supra MkIV rims) is only named there.
    let fs_rim = final_state(vfs).and_then(|f| f.rims).and_then(|r| vfs.find(&r, None));
    let crim = match fs_rim {
        Some(k) => k,
        None => vfs.find(&crim_ref?, None)?,
    };
    let cd = vfs.read(&crim);
    let mut refs: [Option<String>; 7] = Default::default();
    let mut track = [0.0f32; 2];
    fields(&cd, 0, cd.len(), |f, v| match (f, v) {
        (2, Val::F32(x)) => track[0] = x,
        (5, Val::F32(x)) => track[1] = x,
        (1 | 3 | 4 | 6, v) => {
            if let Some(r) = v.range() {
                refs[f as usize] = Some(text(&cd, r));
            }
        }
        _ => {}
    });
    let mut plan = WheelPlan { rims: Default::default(), tyres: Default::default(), track,
                               designs: Vec::new(), label: format!("{} -> {}", base_of(&mech), base_of(&crim)) };
    for (axle, (rf, tf)) in [(1usize, 3usize), (4, 6)].into_iter().enumerate() {
        if let Some(rk) = refs[rf].as_ref().and_then(|r| vfs.find(r, None)) {
            let d = vfs.read(&rk);
            fields(&d, 0, d.len(), |f, v| {
                let Some(r) = v.range() else { return };
                match f {
                    5 | 6 => plan.rims[axle * 2 + (f as usize - 5)] = Some(text(&d, r)),
                    10 => plan.designs.push(base_name_lower(&text(&d, r))),
                    _ => {}
                }
            });
        }
        if let Some(tk) = refs[tf].as_ref().and_then(|r| vfs.find(r, None)) {
            let d = vfs.read(&tk);
            fields(&d, 0, d.len(), |f, v| {
                if let (7, Some(r)) = (f, v.range()) {
                    plan.tyres[axle].get_or_insert_with(|| text(&d, r));
                }
            });
        }
    }
    for r in plan.rims.iter_mut().chain(plan.tyres.iter_mut()) {
        *r = r.take().and_then(|p| vfs.find(&p, None));
    }
    plan.rims.iter().any(|r| r.is_some()).then_some(plan)
}

/// The rim design the car wears: the visual preset lists one per rim STYLE
/// (field 6), and the right one is the first the fitted `.rimmesh` accepts.
/// ⚠ Not simply the first: the RS3's presets list Rim 1 Silver before Rim 2
/// Anthracite, but its mech preset fits Rim 2.
fn fitted_rim_design(vfs: &mut Vfs) -> Option<String> {
    let accepted = wheel_plan(vfs)?.designs;
    // the final state names the design outright (a swapped wheel set's own
    // design is in no visual preset)
    if let Some(d) = final_state(vfs).and_then(|f| f.rim_design)
        .filter(|d| accepted.is_empty() || accepted.contains(&base_name_lower(d)))
        .and_then(|d| vfs.find(&d, None))
    {
        return Some(d);
    }
    let preset = pick_visual_preset(vfs)?;
    let d = vfs.read(&preset);
    let mut out = None;
    fields(&d, 0, d.len(), |f, v| {
        if let (6, Some(r)) = (f, v.range()) {
            let p = text(&d, r);
            if out.is_none() && accepted.contains(&base_name_lower(&p)) {
                out = Some(p);
            }
        }
    });
    out.and_then(|p| vfs.find(&p, None))
}

/// Tyre compound textures: the visual preset's own (field 17, then 7), else the set
/// that dresses the mech preset's default compound, else the actor's (field
/// 18). ⚠ Not "the first road compound": the fleet uses 20 different sets
/// (Pirelli P Zero, DHF slicks, vintage, eco...).
///
/// `Some(None)`: the compound has no branded set (the F40 LM's and M2 CS
/// Racing's compound 10, the 101 "street" tyres) - the game shows a plain
/// tyre, i.e. the tyre material's own default maps, not some other brand.
fn fitted_tyre_textures(vfs: &mut Vfs) -> Option<Option<String>> {
    // A livery can dress its own tyres: field 17 lists them (slick first,
    // then wet) - the 992 Cup's Fanatec car runs yellow Dunlops. Field 7 is
    // only taken when there is no list.
    let (mut listed, mut single) = (None, None);
    if let Some(p) = pick_visual_preset(vfs) {
        let d = vfs.read(&p);
        fields(&d, 0, d.len(), |f, v| {
            let Some(r) = v.range() else { return };
            let t = text(&d, r);
            if !t.to_lowercase().ends_with(".tyretextures") {
                return;
            }
            match f {
                17 => { listed.get_or_insert(t); }
                7 => { single.get_or_insert(t); }
                _ => {}
            }
        });
    }
    let pick = listed.or(single);
    if let Some(p) = pick {
        return Some(vfs.find(&p, None));
    }
    let actor = actor_refs(vfs).into_iter()
        .find(|r| r.to_lowercase().ends_with(".tyretextures"))
        .and_then(|p| vfs.find(&p, None));
    // ⚠ The tyre BRAND follows the compound the mech preset fits by default
    // (field 7; proto3 omits 0). Each .tyretextures lists the compound ids it
    // dresses (field 4, packed): 20 = Pirelli DHF slick (yellow letters),
    // 55 = Bridgestone F1, 301 = Pirelli P Zero... The actor's tyre textures
    // are only the editor default - every GT3 and GT2 car's actor says
    // "Pirelli P Zero" road tyres, and the F2004's says Michelin.
    if let Some(mech) = pick_mech_preset(vfs) {
        let md = vfs.read(&mech);
        let mut compound = 0u64;
        fields(&md, 0, md.len(), |f, v| {
            if let (7, Val::Varint(x)) = (f, v) {
                compound = x;
            }
        });
        let mut all: Vec<String> = vfs.entries.keys()
            .filter(|k| k.ends_with(".tyretextures")).cloned().collect();
        all.sort();
        let mut hits = Vec::new();
        for k in all {
            let d = vfs.read(&k);
            let mut ids = Vec::new();
            fields(&d, 0, d.len(), |f, v| match (f, v) {
                (4, Val::Varint(x)) => ids.push(x),
                (4, Val::Bytes(a, b)) => ids.extend(varints(&d, a, b).into_iter().map(u64::from)),
                _ => {}
            });
            if ids.contains(&compound) {
                hits.push(k);
            }
        }
        // two sets can share a compound (the 2025 and 2026 F1 Pirellis):
        // the actor's own choice breaks the tie
        if let Some(a) = actor.as_ref().filter(|a| hits.contains(a)) {
            return Some(Some(a.clone()));
        }
        return Some(hits.into_iter().next());
    }
    actor.map(Some)
}

fn rim_selection(vfs: &mut Vfs) -> (Vec<String>, Vec<String>) {
    let mut rims = Vec::new();
    let mut tyres = Vec::new();
    // ⚠ Only the FIRST .rimmesh. A car with wheel options ships one per
    // style, and taking them all fits five sets of wheels at once.
    //
    // ⚠ And not the first ALPHABETICALLY: the visual preset names the fitted
    // rim design (VisualCarPresetData.rim_design_paths, field 6), and each
    // .rimmesh names its design. The AE86 Trueno wears the silver "Intra"
    // rims, but rim_ae86_black sorts first, so it got the gold Black Limited.
    let all_rims = vfs.list("skins", "rimmesh");
    let wanted = preset_rim_design(vfs);
    let pick = wanted
        .and_then(|w| {
            all_rims.iter().find(|f| {
                let d = vfs.read(f);
                let mut refs = Vec::new();
                strings_under(&d, 0, d.len(), 0, &mut refs);
                refs.iter().any(|r| base_name_lower(r) == w)
            }).cloned()
        })
        .or_else(|| all_rims.first().cloned());
    for f in pick.into_iter() {
        let d = vfs.read(&f);
        let mut refs = Vec::new();
        strings_under(&d, 0, d.len(), 0, &mut refs);
        for r in refs {
            let low = r.to_lowercase();
            if low.contains("blur") {
                continue;               // spinning-wheel stand-in
            }
            if low.ends_with(".mesh") {
                if let Some(p) = vfs.find(&r, None) {
                    if !rims.contains(&p) {
                        rims.push(p);
                    }
                }
            } else if low.ends_with(".tyremesh") {
                // a .tyremesh is a descriptor; the geometry is the .mesh it names
                if let Some(tm) = vfs.find(&r, None) {
                    {
                        let td = vfs.read(&tm);
                        let mut tr = Vec::new();
                        strings_under(&td, 0, td.len(), 0, &mut tr);
                        for t in tr {
                            if t.to_lowercase().ends_with(".mesh") {
                                if let Some(p) = vfs.find(&t, None) {
                                    if !tyres.contains(&p) {
                                        tyres.push(p);
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    (rims, tyres)
}

/// Fold one mesh's private skeleton into the car-wide one, by NAME.
///
/// ⚠ Bone indices in a mesh are LOCAL to that mesh's own bone table. The body
/// has ~85 bones, the interior 11, the steering wheel 6 - and index 3 means a
/// different bone in each. Skinning every mesh against one chosen skeleton
/// therefore transformed the small meshes by whatever bone happened to sit at
/// that index, which is what threw steering wheels onto the floor.
///
/// Returns local index -> car-wide index. Where two meshes share a bone name,
/// the definition from the RICHEST skeleton wins: a belt mesh carrying a stub
/// CAR_BODY must not overwrite the body's real one, and meshes are read in
/// name order, so "first wins" would let it.
fn merge_bones(
    global: &mut Vec<Bone>,
    index: &mut HashMap<String, usize>,
    owner: &mut Vec<usize>,
    local: &[Bone],
) -> Vec<u32> {
    let mut map = vec![0u32; local.len()];
    for (i, b) in local.iter().enumerate() {
        // parents always precede children, so a parent is already merged
        let parent = b
            .parent
            .and_then(|p| local.get(p))
            .and_then(|pb| index.get(&pb.name).copied());
        if let Some(&g) = index.get(&b.name) {
            if local.len() > owner[g] {
                global[g].local = b.local;
                global[g].parent = parent;
                owner[g] = local.len();
            }
            map[i] = g as u32;
            continue;
        }
        let g = global.len();
        global.push(Bone { name: b.name.clone(), parent, local: b.local });
        index.insert(b.name.clone(), g);
        owner.push(local.len());
        map[i] = g as u32;
    }
    map
}

/// Rewrite a mesh's local bone indices to car-wide ones. Indices are stored by
/// the engine as f32, four per vertex.
fn remap_bone_idx(idx: &mut [u8], map: &[u32]) {
    for c in idx.chunks_exact_mut(4) {
        let v = f32::from_le_bytes(c.try_into().unwrap());
        let g = map.get(v as usize).copied().unwrap_or(0);
        c.copy_from_slice(&(g as f32).to_le_bytes());
    }
}

/// Bake a bone's world transform into a mesh, for parts the engine parents at
/// runtime rather than skins. Row-major, row-vector: v' = v * m.
fn transform_lod(lod: &mut Lod, m: &[f32; 16]) {
    let xf = |v: [f32; 3], w: f32| -> [f32; 3] {
        let mut o = [0.0f32; 3];
        for c in 0..3 {
            o[c] = v[0] * m[c] + v[1] * m[4 + c] + v[2] * m[8 + c] + w * m[12 + c];
        }
        o
    };
    let dir = |buf: &mut Vec<u8>, stride: usize| {
        for i in 0..buf.len() / stride {
            let o = i * stride;
            let v = [
                f32::from_le_bytes(buf[o..o + 4].try_into().unwrap()),
                f32::from_le_bytes(buf[o + 4..o + 8].try_into().unwrap()),
                f32::from_le_bytes(buf[o + 8..o + 12].try_into().unwrap()),
            ];
            let mut n = xf(v, 0.0);
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-8 {
                for a in 0..3 {
                    n[a] /= len;
                }
            }
            for a in 0..3 {
                buf[o + a * 4..o + a * 4 + 4].copy_from_slice(&n[a].to_le_bytes());
            }
        }
    };
    for i in 0..lod.positions.len() / 12 {
        let o = i * 12;
        let v = [
            f32::from_le_bytes(lod.positions[o..o + 4].try_into().unwrap()),
            f32::from_le_bytes(lod.positions[o + 4..o + 8].try_into().unwrap()),
            f32::from_le_bytes(lod.positions[o + 8..o + 12].try_into().unwrap()),
        ];
        let p = xf(v, 1.0);
        for a in 0..3 {
            lod.positions[o + a * 4..o + a * 4 + 4].copy_from_slice(&p[a].to_le_bytes());
        }
    }
    let mut normals = std::mem::take(&mut lod.normals);
    dir(&mut normals, 12);
    lod.normals = normals;
    let mut tangents = std::mem::take(&mut lod.tangents);
    dir(&mut tangents, 16); // 4th float is the bitangent sign, leave it
    lod.tangents = tangents;
}

/// Skin every vertex fully to one bone: (indices, weights) as the engine's
/// f32 x4 streams. Parts baked into car space at their bone's rest transform
/// (rims, tyres, the steering wheel, plates) then follow that bone when the
/// rig moves it, and stay exactly where they were while it does not.
pub(crate) fn rigid_skin(verts: usize, bone: usize) -> (Vec<u8>, Vec<u8>) {
    let mut bidx = Vec::with_capacity(verts * 16);
    let mut bwgt = Vec::with_capacity(verts * 16);
    for _ in 0..verts {
        for k in 0..4 {
            bidx.extend_from_slice(&(if k == 0 { bone as f32 } else { 0.0 }).to_le_bytes());
            bwgt.extend_from_slice(&(if k == 0 { 1.0f32 } else { 0.0 }).to_le_bytes());
        }
    }
    (bidx, bwgt)
}

fn clone_lod(l: &Lod) -> Lod {
    Lod {
        positions: l.positions.clone(),
        normals: l.normals.clone(),
        uvs: l.uvs.clone(),
        tangents: l.tangents.clone(),
        bone_idx: Vec::new(),
        bone_wgt: Vec::new(),
        indices: l.indices.clone(),
        batches: l.batches.clone(),
        bones: Vec::new(),
        facing: l.facing,
        colors: l.colors.clone(),
        uv1: l.uv1.clone(),
        density: l.density,
        dlt: l.dlt,
        anim: String::new(),    cast_shadows: l.cast_shadows,
        shadow_lod: l.shadow_lod,
        shadow_proxy: l.shadow_proxy,
    }
}

fn bounds(pos: &[u8]) -> ([f32; 3], [f32; 3]) {
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for i in 0..pos.len() / 12 {
        for a in 0..3 {
            let o = i * 12 + a * 4;
            let v = f32::from_le_bytes(pos[o..o + 4].try_into().unwrap());
            lo[a] = lo[a].min(v);
            hi[a] = hi[a].max(v);
        }
    }
    (lo, hi)
}

/// Which wheel a rim mesh belongs to.
///
/// ⚠ The corner suffix is not always straight after "rim": a car with wheel
/// options names the style in between, e.g. `..._rim_black_lf.mesh`. Matching
/// only `_rim_<corner>` misses those, and the rim then loads as an ordinary
/// body mesh and is drawn at the ORIGIN inside the car while the wheels stay
/// bare.
fn corner_of(name: &str) -> Option<String> {
    let n = name.strip_suffix(".mesh")?;
    if !n.contains("_rim") || n.contains("blur") {
        return None;
    }
    for c in ["lf", "rf", "lr", "rr"] {
        if n.ends_with(&format!("_{c}")) {
            return Some(c.to_uppercase());
        }
    }
    None
}

/// Rotate a mesh 180 degrees about +Y in place: (x, y, z) -> (-x, y, -z) for
/// positions, normals and tangents. Determinant +1, so the winding stays.
fn turn_about_y(lod: &mut Lod) {
    let flip = |buf: &mut Vec<u8>, stride: usize| {
        for i in 0..buf.len() / stride {
            for a in [0usize, 2] {
                let o = i * stride + a * 4;
                let v = -f32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
                buf[o..o + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    };
    flip(&mut lod.positions, 12);
    flip(&mut lod.normals, 12);
    flip(&mut lod.tangents, 16);
}

/// Put a tyre on a wheel. ⚠ Mirroring flips winding: without reversing the
/// indices the mirrored tyres are back-face culled and the car looks like it
/// has two wheels.
fn place_tyre(lod: &mut Lod, centre: [f32; 3], mirror: bool) {
    let n = lod.positions.len() / 12;
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for i in 0..n {
        for a in 0..3 {
            let o = i * 12 + a * 4;
            let v = f32::from_le_bytes(lod.positions[o..o + 4].try_into().unwrap());
            lo[a] = lo[a].min(v);
            hi[a] = hi[a].max(v);
        }
    }
    let src = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
    let sx = if mirror { -1.0 } else { 1.0 };
    for i in 0..n {
        for a in 0..3 {
            let o = i * 12 + a * 4;
            let mut v = f32::from_le_bytes(lod.positions[o..o + 4].try_into().unwrap()) - src[a];
            if a == 0 {
                v *= sx;
            }
            lod.positions[o..o + 4].copy_from_slice(&(v + centre[a]).to_le_bytes());
        }
    }
    if mirror {
        for i in 0..lod.normals.len() / 12 {
            let o = i * 12;
            let v = -f32::from_le_bytes(lod.normals[o..o + 4].try_into().unwrap());
            lod.normals[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
        for i in 0..lod.tangents.len() / 16 {
            let o = i * 16;
            let x = -f32::from_le_bytes(lod.tangents[o..o + 4].try_into().unwrap());
            lod.tangents[o..o + 4].copy_from_slice(&x.to_le_bytes());
            let w = -f32::from_le_bytes(lod.tangents[o + 12..o + 16].try_into().unwrap());
            lod.tangents[o + 12..o + 16].copy_from_slice(&w.to_le_bytes());
        }
        for t in 0..lod.indices.len() / 3 {
            lod.indices.swap(t * 3, t * 3 + 2);
        }
    }
}

/// Front and rear tyre mesh for this car, from its own .compatibletyres.
fn tyre_meshes(vfs: &mut Vfs) -> Option<(Option<String>, Option<String>)> {
    let mut sizes: Vec<String> = Vec::new();
    for p in vfs.list("presets", "compatibletyres") {
        let d = vfs.read(&p);
        let s = String::from_utf8_lossy(&d);
        for part in s.split(".tyre") {
            // ⚠ sizes are not all road-shaped: 235_45_17 but also 355_660_13
            let tail: String = part.chars().rev().take(14).collect::<String>().chars().rev().collect();
            if let Some(sz) = size_in(&tail) {
                if !sizes.contains(&sz) {
                    sizes.push(sz);
                }
            }
        }
    }
    if sizes.is_empty() {
        return None;
    }
    let front = sizes[0].clone();
    let rear = sizes.get(1).cloned().unwrap_or_else(|| front.clone());
    Some((
        vfs.find(&format!("tyre_{front}.mesh"), None),
        vfs.find(&format!("tyre_{rear}.mesh"), None),
    ))
}

/// The `<width>_<profile>_<rim>` size inside a tyre file name.
///
/// ⚠ Take the FIRST match scanning left to right, and never start mid-number:
/// keeping the last match instead turns 355_660_13 into 55_660_13, which then
/// matches no mesh at all and the car silently loses its tyres.
fn size_in(s: &str) -> Option<String> {
    let b: Vec<char> = s.chars().collect();
    for i in 0..b.len() {
        if i > 0 && b[i - 1].is_ascii_digit() {
            continue;
        }
        let rest: String = b[i..].iter().collect();
        let parts: Vec<&str> = rest.split('_').collect();
        if parts.len() >= 3 {
            let (a, c, d) = (parts[0], parts[1], parts[2]);
            if (2..=3).contains(&a.len())
                && a.chars().all(|x| x.is_ascii_digit())
                && (2..=3).contains(&c.len())
                && c.chars().all(|x| x.is_ascii_digit())
                && d.len() == 2
                && d.chars().all(|x| x.is_ascii_digit())
            {
                return Some(format!("{a}_{c}_{d}"));
            }
        }
    }
    None
}

/// Which compound's textures to fit, from the car's own physics tyre paths.
/// ⚠ Never just take the first .tyretextures found: once shared assets hold
/// more than one set, "first" is alphabetical and a road car wears F1 slicks.
fn tyre_textures(vfs: &mut Vfs) -> HashMap<String, String> {
    match fitted_tyre_textures(vfs) {
        Some(Some(p)) => return read_tyretextures(vfs, &p),
        Some(None) => return HashMap::new(),
        None => {}
    }
    let mut family = "road_".to_string();
    for p in vfs.list("presets", "compatibletyres") {
        {
            let d = vfs.read(&p);
            let s = String::from_utf8_lossy(&d).to_lowercase();
            if let Some(i) = s.find("tyres\\") {
                let rest = &s[i + 6..];
                let fam: String = rest.chars().take_while(|c| *c != '\\').collect();
                family = if fam.starts_with("f1") {
                    "f1".into()
                } else if fam.starts_with("racing") || fam.contains("slick") {
                    "racing".into()
                } else {
                    "road_".into()
                };
                break;
            }
        }
    }
    let mut hits: Vec<String> = vfs
        .entries
        .keys()
        .filter(|k| k.ends_with(".tyretextures"))
        .cloned()
        .collect();
    hits.sort();
    let pick = hits
        .iter()
        .find(|k| base_of(k).contains(&family))
        .or_else(|| hits.first())
        .cloned();
    match pick {
        Some(p) => read_tyretextures(vfs, &p),
        None => HashMap::new(),
    }
}

/// `.tyretextures`: f5 colour, f6 normal, f7 AO (f8 a second normal, f9-12
/// the motion-blurred set) -> the tyre material's slots.
fn read_tyretextures(vfs: &mut Vfs, p: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let d = vfs.read(p);
    fields(&d, 0, d.len(), |f, v| {
        if let Some((a, b)) = v.range() {
            let slot = match f {
                5 => "Base_BaseColorMap",
                6 => "Base_NormalMap",
                7 => "Base_AmbientOcclusionMap",
                _ => return,
            };
            out.insert(slot.to_string(), text(&d, (a, b)));
        }
    });
    out
}

fn collect_ext(root: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(root) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_ext(&p, ext, out);
        } else if p.extension().map(|x| x == ext).unwrap_or(false) {
            out.push(p);
        }
    }
}

/// Livery materials a design substitutes for the car's own.
fn design_overrides(vfs: &mut Vfs) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let enabled = preset_enabled_options(vfs);
    // the fitted rim design is applied to the rim batches by name instead
    let fitted_rim = fitted_rim_design(vfs);
    for f in default_designs(vfs) {
        if Some(&f) == fitted_rim.as_ref() {
            continue;
        }
        let db = vfs.read(&f);
        for slot in read_design(&db) {
            let m = slot.material.replace('\\', "/").to_lowercase();
            if !m.contains("/skins/") {
                continue; // the car's own material, not an override
            }
            // ⚠ An optional slot is only fitted when the preset enables it.
            // Applying every one put the AE86 Trueno's optional red interior
            // into a car the game shows with the standard dark-blue one.
            if slot.optional && !enabled.contains(&base_name_lower(&m)) {
                continue;
            }
            if let Some(p) = vfs.find(&m, None) {
                out.push((slot.id.clone(), p));
            }
        }
    }
    out
}

/// Apply every painted design slot. ⚠ The paint belongs to the RED splat
/// layer, not the base: base is only the fallback for wherever the splat map
/// is black, so painting it colours the trim and carbon too.
/// The game's contact shadows (see contact.rs): a body decal and one per
/// tyre, stamped flat on the ground the tyres stand on.
///
/// ⚠ The ground is the bottom of the placed TYRES, not y = 0 and not the
/// whole car's bounds: the floor quad sits at the scene's lowest point, which
/// is the tyre contact patch, and a decal 1 cm off it either z-fights or
/// floats visibly when the camera is low.
fn add_contact_decals(
    vfs: &mut Vfs,
    wheel_pos: &HashMap<String, [f32; 3]>,
    meshes: &mut Vec<MeshEntry>,
    textures: &mut Vec<TextureEntry>,
    materials: &mut Vec<MaterialEntry>,
    blob: &mut Vec<u8>,
) {
    if std::env::var("EVOVIEW_NO_CONTACT").is_ok() {
        return;
    }
    let Some(a) = vfs.list("", "actor").into_iter().next() else { return };
    let decals = crate::contact::read_actor(&vfs.read(&a));
    if decals.body.is_none() && decals.front.is_none() && decals.rear.is_none() {
        return;
    }
    let mut add = |blob: &mut Vec<u8>, data: &[u8]| -> [u64; 2] {
        let off = blob.len() as u64;
        blob.extend_from_slice(data);
        while blob.len() % 4 != 0 {
            blob.push(0);
        }
        [off, data.len() as u64]
    };
    // Lowest tyre vertex = the ground.
    let mut ground = f32::MAX;
    for m in meshes.iter().filter(|m| m.name.starts_with("tyre_")) {
        if let Some([o, n]) = m.positions {
            let p: &[f32] = bytemuck::cast_slice(&blob[o as usize..(o + n) as usize]);
            for v in p.chunks_exact(3) {
                ground = ground.min(v[1]);
            }
        }
    }
    if ground == f32::MAX {
        ground = 0.0;
    }
    // A few mm up so the decal wins the depth test against the floor without
    // a depth bias; the blended pass does not write depth.
    let y = ground + 0.004;
    let mut push = |name: &str, d: &crate::contact::Decal, centres: &[(f32, f32)],
                    meshes: &mut Vec<MeshEntry>, textures: &mut Vec<TextureEntry>,
                    materials: &mut Vec<MaterialEntry>, blob: &mut Vec<u8>| {
        let Some(found) = vfs.find(&d.texture, None) else {
            eprintln!("contact shadow {name}: {} not found", d.texture);
            return;
        };
        let hdr = vfs.read(&found);
        let mips = vfs.read(&found.replace(".texture", ".texturemips"));
        let Some(t) = read_texture(&hdr, &mips) else { return };
        // ⚠ Format 27 is 4 bpp and was read as BC1, but the shadow masks are
        // single-channel BC4: decoded as BC1 they come out as purple noise.
        let fmt = if t.format == "bc1" && std::env::var("EVOVIEW_CONTACT_BC1").is_err() {
            "bc4".to_string()
        } else {
            t.format.clone()
        };
        let data = add(blob, &t.data);
        textures.push(TextureEntry {
            name: base_of(&found).replace(".texture", ""),
            format: fmt, width: t.width, height: t.height, levels: t.levels, data,
        });
        let ti = textures.len() - 1;
        let file = format!("evoview_contact_{name}.material");
        let mut slots = HashMap::new();
        slots.insert("Base_BaseColorMap".to_string(), ti);
        let mut params = HashMap::new();
        params.insert("blendMode".to_string(), 1.0);
        params.insert("Opacity".to_string(), d.opacity.clamp(0.0, 1.0));
        materials.push(MaterialEntry {
            file: file.clone(),
            path: format!("evoview/contact/{file}"),
            shader: Some(crate::contact::SHADER.to_string()),
            slots, params, colors: HashMap::new(),
        });
        for (i, &(cx, cz)) in centres.iter().enumerate() {
            let q = crate::contact::quad(cx, y, cz, d.width, d.length);
            meshes.push(MeshEntry {
                name: format!("contact_{name}_{i}"),
                vertices: 4,
                batches: vec![Batch { name: String::new(), material: file.clone(),
                                      start: 0, count: 6 }],
                positions: Some(add(blob, &q.positions)),
                normals: Some(add(blob, &q.normals)),
                uvs: Some(add(blob, &q.uvs)),
                tangents: Some(add(blob, &q.tangents)),
                bone_idx: None,
                bone_wgt: None,
                indices: Some(add(blob, bytemuck::cast_slice(&q.indices))),
                xforms: Vec::new(),
                facing: 0,
                lod: None,
                lod_fade: [0.0; 4],
                shadow_mesh: None,
                colors: None,
                uv1: None,
            });
        }
        println!("contact shadow {name}: {:.2} x {:.2} m, {} decal(s), {}",
                 d.width, d.length, centres.len(), base_of(&found));
    };
    if let Some(d) = &decals.body {
        push("body", d, &[(0.0, d.z_offset)], meshes, textures, materials, blob);
    }
    let axle = |front: bool| -> Vec<(f32, f32)> {
        let mut v: Vec<(f32, f32)> = wheel_pos
            .iter()
            .filter(|(c, _)| c.ends_with('F') == front)
            .map(|(_, p)| (p[0], p[2] + 0.0))
            .collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    };
    if let Some(d) = &decals.front {
        push("front", d, &axle(true), meshes, textures, materials, blob);
    }
    if let Some(d) = &decals.rear {
        push("rear", d, &axle(false), meshes, textures, materials, blob);
    }
}

/// The body colours the car's design offers (its EXT_SKIN slot's paint
/// options, lowercase file names) - what EVOVIEW_PAINT picks between.
pub(crate) fn body_paints(vfs: &mut Vfs) -> Vec<String> {
    let mut out = Vec::new();
    for f in default_designs(vfs) {
        let db = vfs.read(&f);
        for slot in read_design(&db) {
            if !slot.id.starts_with("EXT_SKIN") {
                continue;
            }
            for p in &slot.paints {
                let b = base_name_lower(p);
                if !out.contains(&b) {
                    out.push(b);
                }
            }
        }
    }
    out
}

fn apply_paints(vfs: &mut Vfs, materials: &mut [MaterialEntry]) {
    let wanted = std::env::var("EVOVIEW_PAINT").unwrap_or_default().to_lowercase();
    let chosen = preset_paints(vfs);
    // materials the preset paints channel by channel (apply_preset_paints)
    let by_preset: std::collections::HashSet<String> = preset_material_paints(vfs)
        .into_iter().map(|(m, _)| m).collect();
    let mut painted = 0usize;
    for f in default_designs(vfs) {
        let db = vfs.read(&f);
        for slot in read_design(&db) {
            if slot.paints.is_empty() {
                continue;
            }
            // ⚠ A material the preset lists is painted per channel below. This
            // name-matching pass always writes the PRIMARY layer, so where the
            // preset paints only another channel it put that colour on the
            // whole part: the STO's blue centre-lock (channel 3) turned its
            // black rims solid blue.
            let target_m = base_name_lower(&slot.material).trim_end_matches(".material").to_string();
            let hand_picked = !wanted.is_empty() && slot.id.starts_with("EXT_SKIN");
            if by_preset.contains(&target_m) && !hand_picked {
                continue;
            }
            // what the preset selects wins; the design's first option is only
            // a fallback
            let mut pick = slot
                .paints
                .iter()
                .find(|p| {
                    let b = p.replace('\\', "/").rsplit('/').next().unwrap_or(p).to_lowercase();
                    chosen.contains(&b)
                })
                .cloned()
                .unwrap_or_else(|| slot.paints[0].clone());
            // ⚠ starts_with, not ==. A trim can paint a DIFFERENT body
            // material - the AE86 "Kouki" uses EXT_SKIN_KOUKI_1 - so an exact
            // match silently skipped the override on exactly the trims that
            // needed it, and the car came out in its showroom colour.
            if !wanted.is_empty() && slot.id.starts_with("EXT_SKIN") {
                if let Some(m) = slot
                    .paints
                    .iter()
                    .find(|p| p.to_lowercase().contains(&wanted))
                {
                    pick = m.clone();
                }
            }
            let Some(pf) = vfs.find(&pick, None) else { continue };
            let pb = vfs.read(&pf);
            let Some(paint) = read_paint(&pb) else { continue };
            let target = slot
                .material
                .replace('\\', "/")
                .rsplit('/')
                .next()
                .unwrap_or_default()
                .to_lowercase()
                .replace(".material", "");
            for m in materials.iter_mut() {
                if m.file.to_lowercase().replace(".material", "") != target {
                    continue;
                }
                // ⚠ The options can belong to a SECONDARY channel - the
                // Camaro's and STO's lists are centre-lock colours - and
                // painting them onto the primary layer made silver rims of
                // black ones.
                if let Some(ch @ 1..=2) = slot.paint_channel {
                    let layer = if ch == 1 { "Green" } else { "Blue" };
                    apply_paint_layer(m, layer, &paint);
                    painted += 1;
                    println!("  paint {:9} {:32} -> {} ({layer})", slot.id, paint.name, target);
                    continue;
                }
                let layer = if m.slots.contains_key("SpatMap") { "Red" } else { "Base" };
                m.colors.insert(
                    format!("{layer}_Basecolor"),
                    vec![paint.rgb[0], paint.rgb[1], paint.rgb[2], 1.0],
                );
                if layer != "Base" {
                    m.params.insert(format!("{layer}OverrideReplace_BaseColor"), 1.0);
                    m.params.insert(format!("{layer}OverrideMultiply_BaseColor"), 0.0);
                }
                if paint.ignore_material {
                    painted += 1;
                    continue;
                }
                m.params.insert(format!("{layer}_ClearCoatRoughness"), paint.clearcoat_rough);
                m.params.insert(format!("{layer}_NormalScale"), paint.normal_intensity);
                m.params.insert(format!("{layer}_Metalness"), paint.metallic);
                m.params.insert(format!("{layer}_Roughness"), paint.roughness);
                let cc = if paint.clearcoat > 0.0 { paint.clearcoat } else { 1.0 };
                m.params.insert(format!("{layer}_ClearCoat"), cc);
                m.params.insert("ClearCoat".into(), cc);
                painted += 1;
                println!(
                    "  paint {:9} {:32} rgb=[{:.3}, {:.3}, {:.3}] -> {}",
                    slot.id, paint.name, paint.rgb[0], paint.rgb[1], paint.rgb[2], target
                );
            }
        }
    }
    painted += apply_preset_paints(vfs, materials, &wanted);
    painted += apply_paint_links(vfs, materials);
    println!("paints: {painted}");
}

/// The visual preset's own per-material paint choice.
///
/// VisualCarPresetData.multilayer_material_states (field 8) is a list of
/// { 1 material_replacement_path, 2/3/4 primary/secondary/tertiary channel
/// state { 10 oem_color_path } } (Customization.proto in the exe). The
/// primary/secondary/tertiary channels drive the Red/Green/Blue layers.
///
/// ⚠ This is the ONLY place that says which colour goes on which material.
/// The design-slot pass above matches paint NAMES against the preset, and a
/// slot that offers both colours the preset uses (Caterham: red body, black
/// option panels) got whichever the design listed first - a black Caterham.
fn preset_material_paints(vfs: &mut Vfs) -> Vec<(String, [Option<String>; 3])> {
    let Some(preset) = pick_visual_preset(vfs) else { return Vec::new() };
    let d = vfs.read(&preset);
    let mut out = Vec::new();
    fields(&d, 0, d.len(), |f, v| {
        let (8, Some((s, e))) = (f, v.range()) else { return };
        let mut mat = String::new();
        let mut ch: [Option<String>; 3] = [None, None, None];
        fields(&d, s, e, |cf, cv| {
            let Some((a, b)) = cv.range() else { return };
            match cf {
                1 => mat = text(&d, (a, b)),
                2..=4 => fields(&d, a, b, |sf, sv| {
                    if let (10, Some(r)) = (sf, sv.range()) {
                        let path = text(&d, r);
                        if !path.is_empty() {
                            ch[cf as usize - 2] = Some(path);
                        }
                    }
                }),
                _ => {}
            }
        });
        if !mat.is_empty() {
            let base = mat.replace('\\', "/").rsplit('/').next().unwrap_or_default()
                .to_lowercase().replace(".material", "");
            out.push((base, ch));
        }
    });
    out
}

/// Material files the visual preset switches on as options:
/// multilayer_material_states (field 8) with is_optional_enabled (field 5).
fn preset_enabled_options(vfs: &mut Vfs) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    let Some(preset) = pick_visual_preset(vfs) else { return out };
    let d = vfs.read(&preset);
    fields(&d, 0, d.len(), |f, v| {
        let (8, Some((s, e))) = (f, v.range()) else { return };
        let mut mat = String::new();
        let mut on = false;
        fields(&d, s, e, |cf, cv| match (cf, cv) {
            (1, Val::Bytes(a, b)) => mat = text(&d, (a, b)),
            (5, Val::Varint(x)) => on = x != 0,
            _ => {}
        });
        if on && !mat.is_empty() {
            out.insert(base_name_lower(&mat));
        }
    });
    out
}

/// Design channels that are LINKED to another material's channel (mode 3).
///
/// ⚠ This is what makes a "Solid" rim solid: the M2's 930M Solid design links
/// the rim's second channel (the machined faces) to its first, so both wear
/// the preset's Black Gloss; the Bicolour design leaves it unlinked and the
/// faces keep their machined finish. Without the link every solid rim
/// rendered two-tone. Runs after the preset paints, copying what the source
/// channel actually received.
fn apply_paint_links(vfs: &mut Vfs, materials: &mut [MaterialEntry]) -> usize {
    let chosen: HashMap<String, [Option<String>; 3]> = preset_material_paints(vfs)
        .into_iter().collect();
    let mut n = 0;
    for f in default_designs(vfs) {
        let db = vfs.read(&f);
        for slot in read_design(&db) {
            let target = base_name_lower(&slot.material).trim_end_matches(".material").to_string();
            for (ch, src, src_ch) in &slot.links {
                let src_key = base_name_lower(src).trim_end_matches(".material").to_string();
                let Some(path) = chosen.get(&src_key).and_then(|c| c[*src_ch].clone()) else { continue };
                let Some(pf) = vfs.find(&path, None) else { continue };
                let pb = vfs.read(&pf);
                let Some(paint) = read_paint(&pb) else { continue };
                for m in materials.iter_mut() {
                    let by_path = base_name_lower(&m.path).trim_end_matches(".material").to_string();
                    let by_file = m.file.to_lowercase().trim_end_matches(".material").to_string();
                    if by_path != target && by_file != target {
                        continue;
                    }
                    let layer = match ch {
                        0 if m.slots.contains_key("SpatMap")
                            || m.params.get("Splatmap_UseConstant").copied().unwrap_or(0.0) > 0.5 => "Red",
                        0 => "Base",
                        1 => "Green",
                        _ => "Blue",
                    };
                    apply_paint_layer(m, layer, &paint);
                    n += 1;
                    println!("  linked paint {:6} {:32} -> {} (from {} ch{})",
                             layer, paint.name, target, src_key, src_ch);
                }
            }
        }
    }
    n
}

/// Write one paint into a material's layer.
fn apply_paint_layer(m: &mut MaterialEntry, layer: &str, paint: &Paint) {
    m.colors.insert(format!("{layer}_Basecolor"),
                    vec![paint.rgb[0], paint.rgb[1], paint.rgb[2], 1.0]);
    // A customization colour IS the layer's colour, whatever blend mode the
    // material was authored with (many ship neither replace nor multiply).
    if layer != "Base" {
        m.params.insert(format!("{layer}OverrideReplace_BaseColor"), 1.0);
        m.params.insert(format!("{layer}OverrideMultiply_BaseColor"), 0.0);
    }
    if paint.ignore_material {
        return;
    }
    m.params.insert(format!("{layer}_ClearCoatRoughness"), paint.clearcoat_rough);
    m.params.insert(format!("{layer}_NormalScale"), paint.normal_intensity);
    m.params.insert(format!("{layer}_Metalness"), paint.metallic);
    m.params.insert(format!("{layer}_Roughness"), paint.roughness);
    let cc = if paint.clearcoat > 0.0 { paint.clearcoat } else { 1.0 };
    m.params.insert(format!("{layer}_ClearCoat"), cc);
    if layer == "Red" || layer == "Base" {
        m.params.insert("ClearCoat".into(), cc);
    }
}

fn apply_preset_paints(vfs: &mut Vfs, materials: &mut [MaterialEntry], wanted: &str) -> usize {
    // Only materials a design slot offers paints for are customizable. A
    // livery material the preset also lists (Huracan ST EVO2
    // ext_skin_presentation -> Verde STO) keeps its printed livery.
    let mut paintable: std::collections::HashSet<String> = std::collections::HashSet::new();
    for f in default_designs(vfs) {
        let db = vfs.read(&f);
        for slot in read_design(&db) {
            if !slot.paints.is_empty() {
                paintable.insert(slot.material.replace('\\', "/").rsplit('/').next()
                    .unwrap_or_default().to_lowercase().replace(".material", ""));
            }
        }
    }
    let mut n = 0;
    for (target, ch) in preset_material_paints(vfs) {
        if !paintable.contains(&target) {
            continue;
        }
        for (i, path) in ch.iter().enumerate() {
            let Some(path) = path else { continue };
            // EVOVIEW_PAINT picks the body colour by hand; leave it alone.
            if i == 0 && !wanted.is_empty() && target.starts_with("ext_skin") {
                continue;
            }
            let Some(pf) = vfs.find(path, None) else { continue };
            let pb = vfs.read(&pf);
            let Some(paint) = read_paint(&pb) else { continue };
            for m in materials.iter_mut() {
                // Livery overrides are renamed to their slot id, so match the
                // source path as well as the file name.
                let by_path = m.path.replace('\\', "/").rsplit('/').next().unwrap_or_default()
                    .to_lowercase().replace(".material", "");
                let by_file = m.file.to_lowercase().replace(".material", "");
                if by_path != target && by_file != target {
                    continue;
                }
                let layer = match i {
                    0 if m.slots.contains_key("SpatMap")
                        || m.params.get("Splatmap_UseConstant").copied().unwrap_or(0.0) > 0.5 => "Red",
                    0 => "Base",
                    1 => "Green",
                    _ => "Blue",
                };
                apply_paint_layer(m, layer, &paint);
                n += 1;
                println!("  preset paint {:6} {:32} rgb=[{:.3}, {:.3}, {:.3}] -> {}",
                         layer, paint.name, paint.rgb[0], paint.rgb[1], paint.rgb[2], target);
            }
        }
    }
    n
}

// ------------------------------------------------------------- fx helpers --
// Narrow doors for fxdata.rs / dash.rs, which read the effect data (lights,
// LEDs, particles, displays) after the car itself is loaded.

/// Decode one texture (an engine reference) into the car's tables, deduped
/// through `tex_index` so livery reloads and TexCache see a stable source.
pub(crate) fn push_fx_texture(
    vfs: &mut Vfs,
    reference: &str,
    blob: &mut Vec<u8>,
    textures: &mut Vec<TextureEntry>,
    tex_index: &mut HashMap<String, usize>,
) -> Option<usize> {
    let found = vfs.find(reference, None)?;
    if let Some(&i) = tex_index.get(&found) {
        return Some(i);
    }
    let t = vfs.texture(&found, 0)?;
    let off = blob.len() as u64;
    blob.extend_from_slice(&t.data);
    while blob.len() % 4 != 0 {
        blob.push(0);
    }
    textures.push(TextureEntry {
        name: base_of(&found).replace(".texture", ""),
        format: t.format, width: t.width, height: t.height, levels: t.levels,
        data: [off, t.data.len() as u64],
    });
    let i = textures.len() - 1;
    tex_index.insert(found, i);
    Some(i)
}

/// A material's (shader, texture slot -> reference, scalar params, vectors).
pub(crate) fn fx_material(d: &[u8], file: &str)
    -> Option<(Option<String>, HashMap<String, String>, HashMap<String, f32>, HashMap<String, Vec<f32>>)> {
    let m = read_material(d, file)?;
    Some((m.shader, m.slots, m.params, m.colors))
}

#[cfg(test)]
mod node_xf_tests {
    use super::{trs, untrs};

    #[test]
    fn untrs_round_trips() {
        for e in [[0.0, -13.05, 0.0], [160.3, -77.9, -161.5], [5.6, -72.2, -3.8], [180.0, 26.4, 180.0], [30.0, 90.0, 10.0], [-40.0, 20.0, 75.0]] {
            for s in [[1.0, 1.0, 1.0], [1.26, 1.26, 1.26], [2.0, 0.5, 1.5]] {
                let m = trs([86.1, 17.3, -352.5], e, s);
                let (p, e2, s2) = untrs(m);
                let m2 = trs(p, e2, s2);
                assert!(m.abs_diff_eq(m2, 1e-3), "{e:?} {s:?} -> {e2:?} {s2:?}");
            }
        }
    }
}

// ------------------------------------------------------------- tex audit --

/// `--tex-audit <content.kspkg>`: every `.texture` in the package through the
/// same decode the renderer uses, and every way one can come out wrong:
/// header / payload missing, unknown format, no levels, truncated or short mip
/// chains (distant shimmer), all-zero payloads, non-finite floats. One line per
/// problem texture (TEXAUDIT kind key detail) and a summary. The last line,
/// `TEXAUDIT PROBLEMS n`, counts only what would render wrong: editor / dev UI
/// content is left out, and so are the formats with loaders of their own (30
/// irradiance volumes, 8 / 11 the IBL tables) and the short chains that
/// mipfill.rs completes at upload. Rerun after a game update.
pub fn tex_audit(pkg_path: &Path) {
    let Ok(pkg) = crate::pkg::Package::open_filtered(pkg_path, &["content\\"]) else {
        eprintln!("tex-audit: cannot open {}", pkg_path.display());
        return;
    };
    let mut keys: Vec<String> = pkg.by_path.keys().filter(|k| k.ends_with(".texture")).cloned().collect();
    keys.sort();
    let mut counts: std::collections::BTreeMap<&'static str, usize> = Default::default();
    let mut fmts: std::collections::BTreeMap<u64, usize> = Default::default();
    let mut bump = |c: &mut std::collections::BTreeMap<&'static str, usize>, k: &'static str| *c.entry(k).or_default() += 1;
    let mut problems = 0usize;
    for key in &keys {
        let rendered = !key.starts_with(r"content\editor\") && !key.starts_with(r"content\devui\");
        let mut problem = |what: &str, detail: String| {
            println!("TEXAUDIT {what} {key} {detail}");
            if rendered {
                problems += 1;
            }
        };
        let hdr = pkg.read(pkg.by_path[key]);
        let mk = key.replace(".texture", ".texturemips");
        let raw = pkg.by_path.get(&mk).map(|e| pkg.read(*e)).unwrap_or_default();
        let (mut w, mut h, mut mips, mut fmt, mut faces) = (0u64, 0u64, 1u64, 0u64, 1u64);
        fields(&hdr, 0, hdr.len(), |f, v| match (f, v) {
            (1, Val::Varint(n)) => w = n,
            (2, Val::Varint(n)) => h = n,
            (3, Val::Varint(n)) => mips = n,
            (4, Val::Varint(n)) => fmt = n,
            (6, Val::Varint(n)) => faces = n,
            _ => {}
        });
        *fmts.entry(fmt).or_default() += 1;
        if faces == 6 {
            bump(&mut counts, "cubemap (separate path)");
            continue;
        }
        if hdr.is_empty() || w == 0 || h == 0 {
            problem("bad-header", format!("{w}x{h}"));
            bump(&mut counts, "bad header");
            continue;
        }
        if raw.is_empty() {
            problem("no-payload", format!("{w}x{h} fmt {fmt}"));
            bump(&mut counts, "no payload");
            continue;
        }
        let Some((kind, _, _, _)) = format_of(fmt) else {
            if matches!(fmt, 30 | 8 | 11) {
                bump(&mut counts, "own loader (irradiance volumes / IBL tables)");
            } else {
                problem("unknown-format", format!("{w}x{h} fmt {fmt}"));
                bump(&mut counts, "unknown format");
            }
            continue;
        };
        let Some(t) = read_texture_lod_inner(&hdr, &raw, 0) else {
            problem("decode-failed", format!("{w}x{h} {kind}"));
            bump(&mut counts, "decode failed");
            continue;
        };
        if t.levels.is_empty() {
            problem("no-levels", format!("{w}x{h} {kind} payload {} B", raw.len()));
            bump(&mut counts, "no levels");
            continue;
        }
        // (a payload holding just the tail of its chain decodes from the
        // first level it has: smaller, not wrong)
        let first = mips as usize - t.levels.len().min(mips as usize);
        let tail = first > 0 && t.width == ((w as u32) >> first).max(1);
        if (t.levels.len() as u64) < mips && !tail {
            problem("truncated", format!("{w}x{h} {kind} {}/{} levels, payload {} B", t.levels.len(), mips, raw.len()));
            bump(&mut counts, "truncated chain");
        } else if tail {
            bump(&mut counts, "tail-only payload (decoded from its first level)");
        }
        let last = t.levels.last().unwrap();
        if last.width.max(last.height) > 16 && w.max(h) > 64 {
            println!("TEXAUDIT short-chain {key} {w}x{h} {kind} smallest {}x{}", last.width, last.height);
            bump(&mut counts, "short chain (completed at upload)");
        }
        let l0 = t.levels[0].bytes as usize;
        if t.data[..l0.min(t.data.len())].iter().all(|&b| b == 0) {
            println!("TEXAUDIT all-zero {key} {w}x{h} {kind}");
            bump(&mut counts, "all-zero level 0");
        }
        if kind == "rgba16f" {
            let bad = t.data.chunks_exact(2).filter(|c| {
                let v = u16::from_le_bytes([c[0], c[1]]);
                (v & 0x7c00) == 0x7c00
            }).count();
            if bad > 0 {
                problem("non-finite", format!("{w}x{h} {bad} half(s)"));
                bump(&mut counts, "non-finite floats");
            }
        }
        bump(&mut counts, "checked");
    }
    println!("TEXAUDIT SUMMARY {} textures", keys.len());
    for (k, n) in &counts {
        println!("TEXAUDIT   {k}: {n}");
    }
    for (f, n) in &fmts {
        println!("TEXAUDIT   format {f} ({}): {n}", format_of(*f).map_or("UNKNOWN", |x| x.0));
    }
    println!("TEXAUDIT PROBLEMS {problems}");
}
