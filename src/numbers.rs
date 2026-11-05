//! Race-number decals: the plate background and the car's number, projected
//! onto the bodywork the way the actor places them.
//!
//! CarActorData (Scene.proto):
//!   f51 number_decal_types [ {2 category, 3 size, 4 digits_texture,
//!       5 background_texture, 7 font_color {x y z w}, 8 font_width,
//!       9 font_height, 10 font_position, 11 font_spacing_2_digits,
//!       16 font_spacing_3_digits, 17 max_digits_allowed, 15 decal_edge_factor} ]
//!   f52 number_decal_positions {1 front_a 2 front_b 3 door_l 4 door_r 5 roof
//!       6 rear_a 7 rear_b}, each REPEATED NumberDecal {1 enabled, 2 type_id,
//!       3 parent_bone, 4 offset, 5 rotation (deg), 6 scale}
//! The visual preset picks the number and, per position, which of that
//! position's decals is fitted (VisualCarPresetData f13 car_number, f15
//! number_decal_indexes {1..7 = the positions above}: 1-based, 0 = none);
//! the final state carries the same (CarFinalStateData f74, f51).
//!
//! Layout (fitted to the shipped plates): a decal is a 1024 x 1024 canvas
//! over its box's x / z; digits are `font_width` x `font_height` with their
//! top at `font_position`, centred, `font_spacing_N_digits` apart. The
//! digit sheets are 8 x 2 cells (0-7, 8-9), white, tinted by font_color.
//! Projection is along the box's local Y (its smallest extent).
//!
//! GUESS (strict ND1): uv orientation on the box and the 1024 canvas.

use crate::asset::{Batch, MaterialEntry, MeshEntry, TextureEntry};
use crate::load::{base_name_lower, Vfs};
use crate::proto::{fields, text, Val};
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct DecalType {
    pub digits: String,
    pub background: String,
    pub color: [f32; 4],
    pub font_w: f32,
    pub font_h: f32,
    pub font_pos: f32,
    pub spacing2: f32,
    pub spacing3: f32,
    pub max_digits: usize,
    /// f6 the plate's glow map, f13 digit glow strength, f14 its colour
    pub emissive_bg: String,
    pub font_emissive: f32,
    pub emissive_color: [f32; 4],
    /// f15: alpha x (1 - |depth|^edge) across the box (the game's decal
    /// shader: Log(Saturate(|z|)) * edge, alpha * (1 - Exp(..)))
    pub edge: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Decal {
    pub enabled: bool,
    pub type_id: usize,
    pub bone: String,
    pub offset: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: [f32; 3],
    /// f7: this placement glows
    pub emissive: bool,
}

const POSITIONS: [&str; 7] = ["front_a", "front_b", "door_l", "door_r", "roof", "rear_a", "rear_b"];

fn vec3(d: &[u8], a: usize, b: usize) -> [f32; 3] {
    let mut v = [0.0; 3];
    fields(d, a, b, |f, x| {
        if (1..=3).contains(&f) {
            v[f as usize - 1] = x.f32();
        }
    });
    v
}

fn int(v: Val) -> f32 {
    // int32 fields: negative values arrive as 64-bit two's complement varints
    match v {
        Val::Varint(n) => n as i64 as i32 as f32,
        x => x.f32(),
    }
}

/// (types, positions: [list per position]) from the actor.
pub fn read_actor(d: &[u8]) -> (Vec<DecalType>, Vec<Vec<Decal>>) {
    let mut types = Vec::new();
    let mut pos: Vec<Vec<Decal>> = vec![Vec::new(); 7];
    fields(d, 0, d.len(), |f, v| {
        let (103, Some((a, b))) = (f, v.range()) else { return };
        fields(d, a, b, |f, v| {
            let Some((a, b)) = v.range() else { return };
            match f {
                51 => {
                    let mut t = DecalType { color: [1.0; 4], max_digits: 3, emissive_color: [1.0; 4], edge: 10.0, ..Default::default() };
                    fields(d, a, b, |g, w| match (g, w) {
                        (4, Val::Bytes(x, y)) => t.digits = text(d, (x, y)),
                        (5, Val::Bytes(x, y)) => t.background = text(d, (x, y)),
                        (6, Val::Bytes(x, y)) => t.emissive_bg = text(d, (x, y)),
                        (13, Val::F32(x)) => t.font_emissive = x,
                        (15, Val::F32(x)) if x > 0.0 => t.edge = x,
                        (14, Val::Bytes(x, y)) => {
                            fields(d, x, y, |h, z| {
                                if (1..=4).contains(&h) {
                                    t.emissive_color[h as usize - 1] = z.f32();
                                }
                            });
                        }
                        (7, Val::Bytes(x, y)) => {
                            fields(d, x, y, |h, z| {
                                if (1..=4).contains(&h) {
                                    t.color[h as usize - 1] = z.f32();
                                }
                            });
                        }
                        (8, w) => t.font_w = int(w),
                        (9, w) => t.font_h = int(w),
                        (10, w) => t.font_pos = int(w),
                        (11, w) => t.spacing2 = int(w),
                        (16, w) => t.spacing3 = int(w),
                        (17, w) => t.max_digits = int(w).max(1.0) as usize,
                        _ => {}
                    });
                    types.push(t);
                }
                52 => fields(d, a, b, |slot, w| {
                    let (Some((x, y)), true) = (w.range(), (1..=7).contains(&slot)) else { return };
                    let mut n = Decal { scale: [1.0; 3], ..Default::default() };
                    fields(d, x, y, |g, z| match (g, z) {
                        (1, Val::Varint(e)) => n.enabled = e != 0,
                        (2, z) => n.type_id = int(z).max(0.0) as usize,
                        (3, Val::Bytes(p, q)) => n.bone = text(d, (p, q)),
                        (4, Val::Bytes(p, q)) => n.offset = vec3(d, p, q),
                        (5, Val::Bytes(p, q)) => n.rotation = vec3(d, p, q),
                        (6, Val::Bytes(p, q)) => n.scale = vec3(d, p, q),
                        (7, Val::Varint(e)) => n.emissive = e != 0,
                        _ => {}
                    });
                    pos[slot as usize - 1].push(n);
                }),
                _ => {}
            }
        });
    });
    (types, pos)
}

/// The fitted number and per-position decal indexes: EVOVIEW_CAR_NUMBER
/// overrides the number.
pub fn selection(vfs: &mut Vfs) -> (u32, [usize; 7]) {
    let mut number = 0u32;
    let mut idx = [0usize; 7];
    let mut got = false;
    let mut read = |d: &[u8], num_f: u32, idx_f: u32, number: &mut u32, idx: &mut [usize; 7]| {
        let mut any = false;
        fields(d, 0, d.len(), |f, v| {
            if f == num_f {
                *number = int(v).max(0.0) as u32;
                any = true;
            }
            if let (true, Some((a, b))) = (f == idx_f, v.range()) {
                any = true;
                fields(d, a, b, |g, w| {
                    if (1..=7).contains(&g) {
                        idx[g as usize - 1] = int(w).max(0.0) as usize;
                    }
                });
            }
        });
        any
    };
    if let Some(p) = crate::load::pick_visual_preset(vfs) {
        let d = vfs.read(&p);
        got = read(&d, 13, 15, &mut number, &mut idx);
    }
    if !got {
        if let Some(fs) = crate::load::final_state_bytes(vfs) {
            read(&fs, 74, 51, &mut number, &mut idx);
        }
    }
    if let Some(n) = crate::envc::var("EVOVIEW_CAR_NUMBER").ok().and_then(|v| v.parse().ok()) {
        number = n;
    }
    (number, idx)
}

/// One vertex of a decal mesh: car-space position / normal / tangent, the
/// decal-plane point (local x, z), and the skinning of its source triangle.
#[derive(Clone, Copy)]
struct V {
    p: Vec3,
    n: Vec3,
    t: Vec4,
    q: Vec2,
    /// coverage from the depth fade (soft edges)
    a: f32,
}

fn lerp(a: &V, b: &V, s: f32) -> V {
    V { p: a.p.lerp(b.p, s), n: a.n.lerp(b.n, s), t: a.t.lerp(b.t, s), q: a.q.lerp(b.q, s),
        a: a.a + (b.a - a.a) * s }
}

/// Sutherland-Hodgman against an axis-aligned rectangle in the decal plane.
fn clip(mut poly: Vec<V>, lo: Vec2, hi: Vec2) -> Vec<V> {
    for (axis, bound, keep_le) in [(0, hi.x, true), (0, lo.x, false), (1, hi.y, true), (1, lo.y, false)] {
        if poly.is_empty() {
            break;
        }
        let inside = |v: &V| if keep_le { v.q[axis] <= bound } else { v.q[axis] >= bound };
        let mut out = Vec::with_capacity(poly.len() + 2);
        for i in 0..poly.len() {
            let (a, b) = (&poly[i], &poly[(i + 1) % poly.len()]);
            let (ia, ib) = (inside(a), inside(b));
            if ia {
                out.push(*a);
            }
            if ia != ib {
                let s = (bound - a.q[axis]) / (b.q[axis] - a.q[axis]);
                out.push(lerp(a, b, s));
            }
        }
        poly = out;
    }
    poly
}

/// A mesh being built: one layer of one decal.
#[derive(Default)]
struct Out {
    p: Vec<f32>,
    n: Vec<f32>,
    uv: Vec<f32>,
    t: Vec<f32>,
    bi: Vec<f32>,
    bw: Vec<f32>,
    idx: Vec<u32>,
}

impl Out {
    fn poly(&mut self, poly: &[V], lift: f32, skin: ([f32; 4], [f32; 4]), uv: &dyn Fn(Vec2) -> Vec2) {
        if poly.len() < 3 {
            return;
        }
        let base = (self.p.len() / 3) as u32;
        for v in poly {
            let n = v.n.normalize_or_zero();
            let p = v.p + n * lift;
            self.p.extend_from_slice(&[p.x, p.y, p.z]);
            self.n.extend_from_slice(&[n.x, n.y, n.z]);
            let u = uv(v.q);
            self.uv.extend_from_slice(&[u.x, u.y]);
            // the fade rides in |tangent.w| (the sign stays the handedness):
            // a decal has no normal map, and the vertex format no spare slot
            let w = if v.t.w < 0.0 { -1.0 } else { 1.0 } * v.a.clamp(0.002, 1.0);
            self.t.extend_from_slice(&[v.t.x, v.t.y, v.t.z, w]);
            self.bi.extend_from_slice(&skin.0);
            self.bw.extend_from_slice(&skin.1);
        }
        for k in 1..poly.len() as u32 - 1 {
            self.idx.extend_from_slice(&[base, base + k, base + k + 1]);
        }
    }
}

/// Whether a mesh / batch can carry a number: the painted exterior only.
fn paintable(mesh: &str, material: &str) -> bool {
    let m = mesh.to_lowercase();
    let t = material.to_lowercase();
    let skip_mesh = ["driver", "tyre", "rim", "contact", "interior", "belt", "net", "steer", "cable", "number_"];
    let skip_mat = ["glass", "window", "windshield", "light", "int_", "led", "mirror", "rubber", "carbon_raw"];
    !skip_mesh.iter().any(|k| m.contains(k)) && !skip_mat.iter().any(|k| t.contains(k))
}

/// Project the fitted number decals onto the car. `bind`: each bone's
/// row-major row-vector bind matrix (Skeleton::bind), by `bone_index`.
#[allow(clippy::too_many_arguments)]
pub fn add(
    vfs: &mut Vfs,
    bone_index: &HashMap<String, usize>,
    bind: &[[f32; 16]],
    meshes: &mut Vec<MeshEntry>,
    textures: &mut Vec<TextureEntry>,
    materials: &mut Vec<MaterialEntry>,
    blob: &mut Vec<u8>,
) {
    if crate::envc::var_os("EVOVIEW_NO_NUMBERS").is_some() {
        return;
    }
    let Some(a) = vfs.list("", "actor").into_iter().next() else { return };
    let (types, positions) = read_actor(&vfs.read(&a));
    if types.is_empty() {
        return;
    }
    let (number, idx) = selection(vfs);
    let add_b = |blob: &mut Vec<u8>, data: &[u8]| -> [u64; 2] {
        let off = blob.len() as u64;
        blob.extend_from_slice(data);
        while blob.len() % 4 != 0 {
            blob.push(0);
        }
        [off, data.len() as u64]
    };
    // the source geometry, read once: (mesh index, positions, normals,
    // tangents, bone idx, bone wgt, indices, batches)
    let f32s = |blob: &Vec<u8>, r: Option<[u64; 2]>| -> Vec<f32> {
        r.map(|[o, n]| bytemuck::cast_slice::<u8, f32>(&blob[o as usize..(o + n) as usize]).to_vec()).unwrap_or_default()
    };
    let src: Vec<(String, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<Batch>)> = meshes.iter()
        .filter(|m| m.xforms.is_empty())
        .map(|m| (m.name.clone(), f32s(blob, m.positions), f32s(blob, m.normals), f32s(blob, m.tangents),
                  f32s(blob, m.bone_idx), f32s(blob, m.bone_wgt),
                  m.indices.map(|[o, n]| bytemuck::cast_slice::<u8, u32>(&blob[o as usize..(o + n) as usize]).to_vec()).unwrap_or_default(),
                  m.batches.clone()))
        .collect();
    let mut tex_cache: HashMap<String, Option<usize>> = HashMap::new();
    let mut texture = |vfs: &mut Vfs, path: &str, textures: &mut Vec<TextureEntry>, blob: &mut Vec<u8>| -> Option<usize> {
        if let Some(t) = tex_cache.get(path) {
            return *t;
        }
        let r = (|| {
            let found = vfs.find(path, None)?;
            let hdr = vfs.read(&found);
            let mips = vfs.read(&found.replace(".texture", ".texturemips"));
            let t = crate::load::read_texture_pub(&hdr, &mips)?;
            let data = add_b(blob, &t.data);
            textures.push(TextureEntry {
                name: base_name_lower(&found).replace(".texture", ""),
                format: t.format, width: t.width, height: t.height, levels: t.levels, data,
            });
            Some(textures.len() - 1)
        })();
        tex_cache.insert(path.to_string(), r);
        r
    };
    let digits: Vec<u32> = number.to_string().chars().filter_map(|c| c.to_digit(10)).collect();
    let mut made = 0;
    for (pi, list) in positions.iter().enumerate() {
        let k = idx[pi];
        let Some(dec) = (k > 0).then(|| list.get(k - 1)).flatten().filter(|d| d.enabled) else { continue };
        let Some(ty) = dec.type_id.checked_sub(1).and_then(|t| types.get(t)) else { continue };
        // EVOVIEW_NUMBER_GLOW=<strength>: light every number (no shipped car
        // sets f7 / f13 - this is how the glowing path is checked)
        let forced = crate::envc::var("EVOVIEW_NUMBER_GLOW").ok().and_then(|v| v.parse::<f32>().ok());
        let ty_glow;
        let (ty, dec) = match forced {
            Some(k) => {
                let mut t = ty.clone();
                t.font_emissive = k;
                ty_glow = (t, Decal { emissive: true, ..dec.clone() });
                (&ty_glow.0, &ty_glow.1)
            }
            None => (ty, dec),
        };
        let Some(&bi) = bone_index.get(&dec.bone) else {
            eprintln!("number decal {}: no bone {}", POSITIONS[pi], dec.bone);
            continue;
        };
        // decal frame on the bone: rotation X first, then Y, then Z (the
        // door boxes, rotation (0, +-91, -90 / -270), only project outward
        // that way; the parts' euler order sends them along the car)
        let r = dec.rotation;
        let local = Mat4::from_translation(Vec3::from(dec.offset))
            * Mat4::from_euler(glam::EulerRot::ZYX, r[2].to_radians(), r[1].to_radians(), r[0].to_radians());
        let to_car = Mat4::from_cols_array(&bind[bi]) * local;
        let to_dec = to_car.inverse();
        let half = Vec3::from(dec.scale).abs() * 0.5;
        let axis = to_car.transform_vector3(Vec3::Y).normalize_or_zero();
        // the triangles inside the box, with which way each faces the
        // projector; the side with more area is the outer skin
        let mut tris: Vec<(Vec<V>, f32, ([f32; 4], [f32; 4]))> = Vec::new();
        let mut area = [0.0f32; 2];
        for (name, p, n, t, bidx, bwgt, ind, batches) in &src {
            for b in batches {
                if !paintable(name, &b.material) {
                    continue;
                }
                let end = ((b.start + b.count) as usize).min(ind.len());
                for tri in ind[b.start as usize..end].chunks_exact(3) {
                    let at = |i: u32| Vec3::new(p[i as usize * 3], p[i as usize * 3 + 1], p[i as usize * 3 + 2]);
                    let (a, bb, c) = (at(tri[0]), at(tri[1]), at(tri[2]));
                    let la = [to_dec.transform_point3(a), to_dec.transform_point3(bb), to_dec.transform_point3(c)];
                    let lo = la[0].min(la[1]).min(la[2]);
                    let hi = la[0].max(la[1]).max(la[2]);
                    if hi.x < -half.x || lo.x > half.x || hi.z < -half.z || lo.z > half.z || hi.y < -half.y || lo.y > half.y {
                        continue;
                    }
                    let fn_ = (bb - a).cross(c - a);
                    let ar = fn_.length() * 0.5;
                    if ar < 1e-9 {
                        continue;
                    }
                    let facing = fn_.normalize().dot(axis);
                    if facing.abs() < 0.35 {
                        continue;
                    }
                    area[(facing > 0.0) as usize] += ar;
                    let vs: Vec<V> = (0..3).map(|j| {
                        let i = tri[j] as usize;
                        let nv = if n.len() > i * 3 + 2 { Vec3::new(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]) } else { fn_.normalize() };
                        let tv = if t.len() > i * 4 + 3 { Vec4::new(t[i * 4], t[i * 4 + 1], t[i * 4 + 2], t[i * 4 + 3]) } else { Vec4::new(1.0, 0.0, 0.0, 1.0) };
                        let depth = (la[j].y / half.y.max(1e-4)).abs().min(1.0);
                        V { p: [a, bb, c][j], n: nv, t: tv, q: Vec2::new(la[j].x, la[j].z),
                            a: 1.0 - depth.powf(ty.edge) }
                    }).collect();
                    let i0 = tri[0] as usize;
                    let skin = if bidx.len() > i0 * 4 + 3 && bwgt.len() > i0 * 4 + 3 {
                        ([bidx[i0 * 4], bidx[i0 * 4 + 1], bidx[i0 * 4 + 2], bidx[i0 * 4 + 3]],
                         [bwgt[i0 * 4], bwgt[i0 * 4 + 1], bwgt[i0 * 4 + 2], bwgt[i0 * 4 + 3]])
                    } else {
                        ([bi as f32, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
                    };
                    tris.push((vs, facing, skin));
                }
            }
        }
        let outer = area[1] >= area[0];
        if crate::envc::var_os("EVOVIEW_NUMBER_LOG").is_some() {
            println!("number decal {} [{}]: bone {} centre {:.3?} axis {:.2?} half {:.3?} tris {} area +{:.4} -{:.4}",
                     POSITIONS[pi], k, dec.bone, to_car.transform_point3(Vec3::ZERO), axis, half, tris.len(), area[1], area[0]);
        }
        tris.retain(|(_, f, _)| (*f > 0.0) == outer);
        if tris.is_empty() {
            eprintln!("number decal {}: nothing to land on", POSITIONS[pi]);
            continue;
        }
        // canvas (0..1024 across x, down z) <-> decal plane
        // (local +x = the canvas's right, local +z = its down)
        let to_uv = |q: Vec2| Vec2::new(q.x / (2.0 * half.x) + 0.5, q.y / (2.0 * half.z) + 0.5);
        let from_canvas = |cx: f32, cy: f32| Vec2::new((cx / 1024.0 - 0.5) * 2.0 * half.x, (cy / 1024.0 - 0.5) * 2.0 * half.z);
        let lo = Vec2::new(-half.x, -half.z);
        let hi = Vec2::new(half.x, half.z);
        // (layer, mesh, tint, colour map, glow (map, colour x strength))
        let mut layers: Vec<(String, Out, [f32; 4], usize, Option<(usize, [f32; 4])>)> = Vec::new();
        if let Some(bg) = (!ty.background.is_empty()).then(|| texture(vfs, &ty.background, textures, blob)).flatten() {
            let mut o = Out::default();
            for (vs, _, skin) in &tris {
                o.poly(&clip(vs.clone(), lo, hi), 0.0012, *skin, &|q| to_uv(q));
            }
            let glow = dec.emissive.then(|| (!ty.emissive_bg.is_empty())
                .then(|| texture(vfs, &ty.emissive_bg, textures, blob)).flatten()).flatten()
                .map(|t| (t, [1.0, 1.0, 1.0, 1.0]));
            layers.push(("bg".into(), o, [1.0; 4], bg, glow));
        }
        let shown: Vec<u32> = digits.iter().rev().take(ty.max_digits.max(1)).rev().copied().collect();
        if let (false, Some(dt)) = (shown.is_empty(), (!ty.digits.is_empty()).then(|| texture(vfs, &ty.digits, textures, blob)).flatten()) {
            let n = shown.len() as f32;
            let sp = match shown.len() { 2 => ty.spacing2, 3.. => ty.spacing3, _ => 0.0 };
            let (fw, fh) = (ty.font_w.max(1.0), ty.font_h.max(1.0));
            let total = n * fw + (n - 1.0) * sp;
            let x0 = (1024.0 - total) * 0.5;
            let mut o = Out::default();
            for (i, &dgt) in shown.iter().enumerate() {
                let cx0 = x0 + i as f32 * (fw + sp);
                let (a, b) = (from_canvas(cx0, ty.font_pos + fh), from_canvas(cx0 + fw, ty.font_pos));
                let (dlo, dhi) = (a.min(b).max(lo), a.max(b).min(hi));
                let (col, row) = ((dgt % 8) as f32, (dgt / 8) as f32);
                for (vs, _, skin) in &tris {
                    let poly = clip(vs.clone(), dlo, dhi);
                    o.poly(&poly, 0.0018, *skin, &|q| {
                        let uv = to_uv(q);
                        let (cx, cy) = (uv.x * 1024.0, uv.y * 1024.0);
                        let uu = ((cx - cx0) / fw).clamp(0.0, 1.0);
                        let vv = ((cy - ty.font_pos) / fh).clamp(0.0, 1.0);
                        Vec2::new((col + uu) / 8.0, (row + vv) / 2.0)
                    });
                }
            }
            let e = ty.emissive_color;
            let glow = (dec.emissive && ty.font_emissive > 0.0)
                .then(|| (dt, [e[0], e[1], e[2], ty.font_emissive]));
            layers.push(("digits".into(), o, ty.color, dt, glow));
        }
        for (layer, o, color, ti, glow) in layers {
            if o.idx.is_empty() {
                continue;
            }
            let file = format!("evoview_number_{}_{layer}.material", POSITIONS[pi]);
            let mut slots = HashMap::new();
            slots.insert("Base_BaseColorMap".to_string(), ti);
            let mut params = HashMap::new();
            params.insert("blendMode".to_string(), 1.0);
            params.insert("Roughness".to_string(), 0.35);
            let mut colors = HashMap::new();
            colors.insert("Basecolor".to_string(), color.to_vec());
            if let Some((gi, c)) = glow {
                slots.insert("Base_EmissiveMap".to_string(), gi);
                colors.insert("Base_EmissiveColor".to_string(), vec![c[0], c[1], c[2], 1.0]);
                params.insert("Base_EmissiveIntensity".to_string(), c[3]);
            }
            materials.push(MaterialEntry {
                file: file.clone(), path: format!("evoview/numbers/{file}"), shader: None,
                slots, params, colors,
            });
            let verts = (o.p.len() / 3) as u32;
            let count = o.idx.len() as u32;
            meshes.push(MeshEntry {
                name: format!("number_{}_{layer}", POSITIONS[pi]),
                vertices: verts,
                batches: vec![Batch { name: String::new(), material: file, start: 0, count }],
                positions: Some(add_b(blob, bytemuck::cast_slice(&o.p))),
                normals: Some(add_b(blob, bytemuck::cast_slice(&o.n))),
                uvs: Some(add_b(blob, bytemuck::cast_slice(&o.uv))),
                tangents: Some(add_b(blob, bytemuck::cast_slice(&o.t))),
                bone_idx: Some(add_b(blob, bytemuck::cast_slice(&o.bi))),
                bone_wgt: Some(add_b(blob, bytemuck::cast_slice(&o.bw))),
                indices: Some(add_b(blob, bytemuck::cast_slice(&o.idx))),
                xforms: Vec::new(),
                facing: 0,
                lod: None,
                lod_fade: [0.0; 4],
                shadow_mesh: None,
                colors: None,
                uv1: None,
            });
        }
        made += 1;
    }
    if made > 0 {
        println!("number decals: #{number} on {made} position(s)");
    }
}
