//! Track rendering, kept apart from the car path.
//!
//! A circuit gets its own pipeline (track.wgsl) with its own bind groups:
//!   group 0  frame: camera, sun, haze, the irradiance-volume atlas
//!   group 1  per-material constants and up to twelve textures
//! Car materials (a replay puts cars on the track) keep the car pipeline.
//!
//! What is read from the game here, and nowhere else:
//!   * Irradiance Probe entities in the track's root .scene: one per 72 m
//!     column, `IrradianceVolumeActorData` = { f1 bounced_sky texture, f2
//!     groundInterval 4.5, f3/f4/f5 layer heights 1.5/5.5/13.5, f6 256 ground
//!     heights }. The texture is 64x3x16 R11G11B10: x = 16 cells x 4
//!     coefficients interleaved, y = 3 height layers, z = 16 cells.
//!   * Track Info: latitude / longitude, for the sun.
//!   * the material parameters of each track material (splat layers, tile
//!     scales, vertex AO), interpreted in `TrackMatUniform::from_entry`.

use crate::asset::{Car, MaterialEntry};
use crate::load::Vfs;
use crate::material::Blend;
use crate::proto::{fields, text, Val};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

// ------------------------------------------------------------------ data ---

/// The irradiance volumes of one track, packed into one 3D texture:
/// width 16*nx cells, height 24, depth 16*nz. Rows 0..12 are the bounced sky
/// (4 coefficients x 3 layers), rows 12..24 the same for the NIGHT
/// irradiance (IrradianceVolumeActorData f12: the track's lamps, baked).
/// RGBA16F, rgb = coefficient, a = ground height of that column.
pub struct IvAtlas {
    pub origin: [f32; 2],
    pub interval: f32,
    pub nx: u32,
    pub nz: u32,
    pub elev: [f32; 3],
    pub texels: Vec<u16>,
}

/// Everything track-wide the renderer needs, read at load time.
#[derive(Default)]
pub struct TrackEnv {
    pub iv: Option<IvAtlas>,
    pub lat: f64,
    pub lon: f64,
    pub has_geo: bool,
    /// average ground height (world y) of the irradiance columns, haze base
    pub ground_y: f32,
    /// how many of the scene's materials (from index 0) are the track's
    pub n_materials: usize,
    /// SceneGraphicsSettings: haze_minimum_altitude (f1)
    pub haze_min_alt: Option<f32>,
    /// SceneGraphicsSettings f8 haze_start_distance (metres): no haze nearer
    /// than this (Kyalami 150, Monza 350, Brands Hatch 1000, Imola 2500)
    pub haze_start: Option<f32>,
    /// f41 haze_density (0.01 Spa .. 0.02 Red Bull Ring); evoview's own
    /// density is scaled by it relative to 0.01
    pub haze_k: Option<f32>,
    /// SceneGraphicsSettings.pigment: texture path, XZ rect (min x, min z,
    /// size x, size z) from its transform
    pub pigment_path: Option<String>,
    pub pigment_rect: [f32; 4],
    /// index into Car::textures once the loader has read it
    pub pigment_tex: Option<usize>,
    /// Ground-cover grass with ALL its instances (blob ranges into the Car);
    /// drawn near the camera only, by TrackRender.
    /// per tuft mesh, its LOD levels (level 0 holds the instances;
    /// MeshEntry.lod = [switch-in metres, density])
    pub carpet: Vec<Vec<crate::asset::MeshEntry>>,
    /// animated instanced props (spectators, flag marshals): load.rs
    /// build_track_anim
    pub anim: Option<std::sync::Arc<TrackAnim>>,
    /// Track Info f5: the game's TimeZone enum (sky.rs)
    pub tz: u32,
    /// LightData actors (f101) and instanced lights (f139) of every scene
    pub lights: Vec<TrackLight>,
    /// content/weather/cloud_noise.dds mip 0: 64 BC6H slices of 512x512
    /// (the game's volumetric cloud noise), uploaded as a 2D array
    pub cloud_noise: Option<Vec<u8>>,
    /// content/weather/clouds/cirrus_layer.texture, index into Car::textures:
    /// three high-cloud patterns, one per channel (R streaky cirrus, G
    /// cirrocumulus speckle, B soft veil)
    pub cirrus_tex: Option<usize>,
    /// content/weather/weather_map6.texture (clouds_settings/setting_1's
    /// weatherMapPath): 2048 x 64, R cloud cover, B rain
    pub weather_tex: Option<usize>,
    /// what a car drives on: every static mesh with a surface_id, baked to
    /// world space (load.rs build_collision)
    pub collision: Option<Collision>,
    /// SurfaceDefinitionActorData (actor f109): grip and bumps per surface
    pub surface_defs: Vec<SurfaceDef>,
}

/// World-space collider triangles, one surface index per triangle.
#[derive(Default)]
pub struct Collision {
    pub verts: Vec<[f32; 3]>,
    pub tris: Vec<[u32; 3]>,
    pub tri_surf: Vec<u16>,
    /// surface_id strings, indexed by tri_surf
    pub surfaces: Vec<String>,
    /// walls in XZ: [x0, z0, x1, z1, y low, y high]
    pub walls: Vec<[f32; 6]>,
    /// what each wall came from (a collider's or an instance's mesh path)
    pub wall_src: Vec<String>,
    /// each wall's collider surface (index into `surfaces`), u16::MAX for
    /// an instance's box
    pub wall_surf: Vec<u16>,
}

/// LogicScene.proto SurfaceDefinitionData.
#[derive(Clone, Debug, Default)]
pub struct SurfaceDef {
    pub id: String,
    /// f2 collisionCategory
    pub category: u32,
    /// f3 gripMod (1 = full road grip)
    pub grip: f32,
    /// f10 sinHeight / f11 sinLength: the surface's bumps (m)
    pub sin_height: f32,
    pub sin_length: f32,
    /// f12 damping
    pub damping: f32,
    /// f8 isValidTrack
    pub valid: bool,
    /// f20 surface_type: 0 asphalt, 1 extra turf, 2 kerb, 3 grass, 4 sand, 5 flat kerb
    pub kind: u32,
    /// f14 vibrationGain / f15 vibrationLength (m): the kerb buzz the
    /// FFB's curbs channel plays (a sawtooth over distance travelled)
    pub vibration_gain: f32,
    pub vibration_length: f32,
    /// f18 frictionRain / f19 frictionRainGamma: the surface's grip when wet
    pub friction_rain: f32,
    pub friction_rain_gamma: f32,
}

/// The game's default cloud deck (content/weather/clouds_settings/setting_1):
/// base 1100 m, thickness 1000 m, scale 1, erosion 35.05 / 0.223 / 0.081,
/// density 0.125, coverage 0.294.
pub const CLOUD_SLICES: u32 = 64;
pub const CLOUD_DIM: u32 = 512;

/// Read the cloud noise volume's first mip out of its DDS (DX10 header,
/// BC6H_UF16, 512x512x64).
pub fn load_cloud_noise(vfs: &mut Vfs) -> Option<Vec<u8>> {
    let key = vfs.find("content/weather/cloud_noise.dds", None)?;
    let d = vfs.read(&key);
    if d.len() < 148 || &d[0..4] != b"DDS " || &d[84..88] != b"DX10" {
        return None;
    }
    let u = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
    let (h, w, depth, dxgi, dim) = (u(12), u(16), u(24), u(128), u(132));
    if (w, h, depth, dxgi, dim) != (CLOUD_DIM, CLOUD_DIM, CLOUD_SLICES, 95, 4) {
        eprintln!("cloud noise: unexpected layout {w}x{h}x{depth} dxgi {dxgi} dim {dim}");
        return None;
    }
    let n = (CLOUD_DIM as usize / 4) * (CLOUD_DIM as usize / 4) * 16 * CLOUD_SLICES as usize;
    (d.len() >= 148 + n).then(|| d[148..148 + n].to_vec())
}

/// One of the track's own lamps (LogicScene.proto LightData).
#[derive(Clone, Copy, Debug)]
pub struct TrackLight {
    pub pos: [f32; 3],
    /// the way a spot points (local -Y, see load_lights)
    pub dir: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
    /// inner / outer cone half-angles, degrees (spot only)
    pub inner: f32,
    pub outer: f32,
    /// f21: 0 = inverse-square, else UE's (1 - (d/r)^2)^exp
    pub falloff: f32,
    pub spot: bool,
    /// f17 disable_auto_switch_off: lit by day as well
    pub always_on: bool,
}

fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00;
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7bff;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - e + 13);
        return sign | m as u16;
    }
    sign | ((e as u16) << 10) | ((man >> 13) as u16)
}

fn r11g11b10(u: u32) -> [f32; 3] {
    fn f(bits: u32, mb: u32) -> f32 {
        let m = bits & ((1 << mb) - 1);
        let e = bits >> mb;
        if e == 0 {
            m as f32 / (1u32 << mb) as f32 * 2f32.powi(-14)
        } else if e == 31 {
            65000.0
        } else {
            (1.0 + m as f32 / (1u32 << mb) as f32) * 2f32.powi(e as i32 - 15)
        }
    }
    [f(u & 0x7ff, 6), f((u >> 11) & 0x7ff, 6), f(u >> 22, 5)]
}

struct Probe {
    pos: [f32; 3],
    tex: String,
    /// f12 night_irradiance_texture (empty when the track has none)
    night: String,
    interval: f32,
    elev: [f32; 3],
    ground: Vec<f32>,
}

fn vec3_of(d: &[u8], s: usize, e: usize) -> [f32; 3] {
    let mut p = [0.0f32; 3];
    fields(d, s, e, |f, v| {
        if (1..=3).contains(&f) {
            p[(f - 1) as usize] = v.f32();
        }
    });
    p
}

/// Read the track-wide environment from the root scene.
/// Track props that play a clip (MeshLodData f24): spectators, flag
/// marshals, the sky dancer. Each animated mesh level owns `variants` bone
/// sets in one storage buffer, played phase-shifted; an instance picks one
/// by its index (instance row float 17 = slot + 1, see `skin_rows`).
#[derive(Default)]
pub struct TrackAnim {
    pub clips: Vec<crate::load::Animation>,
    pub meshes: Vec<AnimMesh>,
    /// bone matrices in the buffer
    pub slots: u32,
}

pub struct AnimMesh {
    /// index into the scene's meshes
    pub mesh: usize,
    pub skel: crate::load::Skeleton,
    pub clip: usize,
    pub base: u32,
    pub variants: u32,
}

impl TrackAnim {
    /// Each copy's clip time at `t`: the copies are spread over the loop
    /// (plus a little per mesh). EVOVIEW_TRACK_ANIM_SYNC: no phase shift
    /// (debugging a frame).
    fn times(&self, t: f32) -> impl Iterator<Item = (usize, u32, f32)> + '_ {
        let sync = std::env::var_os("EVOVIEW_TRACK_ANIM_SYNC").is_some();
        self.meshes.iter().enumerate().flat_map(move |(mi, m)| {
            let a = &self.clips[m.clip];
            let len = (a.frames as f32 / a.fps.max(1.0)).max(0.05);
            (0..m.variants).map(move |v| {
                let ph = if sync { 0.0 } else { (v as f32 / m.variants as f32 + m.mesh as f32 * 0.137) * len };
                (mi, v, (t + ph) % len)
            })
        })
    }

    /// `pose` into `out`, re-posing only the copies whose clip frame
    /// (pose_many rounds to frames) is not the one in `keys`. The copies are
    /// phase-shifted, so one or another steps nearly every frame, but each
    /// only at its clip's ~30 fps. False: nothing changed.
    pub fn pose_into(&self, t: f32, keys: &mut Vec<u32>, out: &mut Vec<[f32; 16]>) -> bool {
        let n = self.slots.max(1) as usize;
        if out.len() != n {
            *out = vec![crate::load::IDENTITY; n];
            keys.clear();
        }
        let mut changed = false;
        for (k, (mi, v, ct)) in self.times(t).enumerate() {
            let m = &self.meshes[mi];
            let a = &self.clips[m.clip];
            let f = (ct * a.fps).round().max(0.0) as u32;
            if keys.get(k) == Some(&f) {
                continue;
            }
            if k >= keys.len() {
                keys.resize(k + 1, u32::MAX);
            }
            keys[k] = f;
            let nb = m.skel.bones.len();
            let p = m.skel.pose_many(&[(a, ct)]);
            let o = (m.base as usize) + v as usize * nb;
            out[o..o + nb].copy_from_slice(&p[..nb]);
            changed = true;
        }
        changed
    }

    /// Every bone set at time `t` (seconds), in buffer order.
    pub fn pose(&self, t: f32) -> Vec<[f32; 16]> {
        let mut out = vec![crate::load::IDENTITY; self.slots.max(1) as usize];
        for (mi, v, ct) in self.times(t) {
            let m = &self.meshes[mi];
            let a = &self.clips[m.clip];
            let nb = m.skel.bones.len();
            {
                let p = m.skel.pose_many(&[(a, ct)]);
                let o = (m.base as usize) + v as usize * nb;
                out[o..o + nb].copy_from_slice(&p[..nb]);
            }
        }
        out
    }
}

/// Write each instance's bone slot (row float 17) into packed 20-float
/// instance rows of scene mesh `src`. Rows of other meshes are left as they are.
pub fn skin_rows(car: &Car, src: usize, rows: &mut [f32]) {
    let Some(a) = car.track_env.as_ref().and_then(|e| e.anim.as_ref()) else { return };
    let Some(m) = a.meshes.iter().find(|m| m.mesh == src) else { return };
    let nb = m.skel.bones.len() as u32;
    for (k, r) in rows.chunks_exact_mut(20).enumerate() {
        r[17] = (m.base + (k as u32 % m.variants) * nb + 1) as f32;
    }
}

/// Write each instance's mesh height bounds (rows floats 18 / 19: local
/// min / max y) for the wind's height falloff.
pub fn wind_rows(car: &Car, src: usize, rows: &mut [f32]) {
    let Some(m) = car.meshes.get(src) else { return };
    let pos = car.floats(m.positions);
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for p in pos.chunks_exact(3) {
        lo = lo.min(p[1]);
        hi = hi.max(p[1]);
    }
    if lo > hi {
        return;
    }
    for r in rows.chunks_exact_mut(20) {
        r[18] = lo;
        r[19] = hi;
    }
}

/// Whether scene mesh `src` is one the track animates.
pub fn is_animated(car: &Car, src: usize) -> bool {
    car.track_env.as_ref().and_then(|e| e.anim.as_ref()).map_or(false, |a| a.meshes.iter().any(|m| m.mesh == src))
}

pub fn load_env(vfs: &mut Vfs) -> TrackEnv {
    let mut env = TrackEnv::default();
    let t0 = std::time::Instant::now();
    let roots: Vec<String> = vfs
        .own_keys("scene")
        .into_iter()
        .filter(|k| !k.to_lowercase().contains("/containers/"))
        .collect();
    let mut probes: Vec<Probe> = Vec::new();
    for key in roots {
        let d = vfs.read(&key);
        fields(&d, 0, d.len(), |f, v| {
            let Some((s, e)) = v.range() else { return };
            if f == 3 {
                // SceneGraphicsSettings
                fields(&d, s, e, |gf, gv| match (gf, gv) {
                    (1, v) => env.haze_min_alt = Some(v.f32()),
                    (8, v) => env.haze_start = Some(v.f32()),
                    (41, v) => env.haze_k = Some(v.f32()),
                    (4, Val::Bytes(a, b)) => {
                        let mut pos = [0.0f32; 3];
                        let mut scale = [1.0f32; 3];
                        fields(&d, a, b, |pf, pv| match (pf, pv) {
                            (1, Val::Bytes(x, y)) => fields(&d, x, y, |tf, tv| {
                                if let Some((i, j)) = tv.range() {
                                    match tf {
                                        1 => pos = vec3_of(&d, i, j),
                                        3 => scale = vec3_of(&d, i, j),
                                        _ => {}
                                    }
                                }
                            }),
                            (2, Val::Bytes(x, y)) => env.pigment_path = Some(text(&d, (x, y))),
                            _ => {}
                        });
                        // the transform scales a unit [-1, 1] quad: scale is
                        // the HALF size (Road Atlanta: +-548 x +-1096 m)
                        env.pigment_rect = [pos[0] - scale[0], pos[2] - scale[2],
                                            (scale[0] * 2.0).max(1.0), (scale[2] * 2.0).max(1.0)];
                    }
                    _ => {}
                });
                return;
            }
            if f != 2 {
                return;
            }
            let mut kind = String::new();
            let mut pos = [0.0f32; 3];
            let mut body: Option<(usize, usize)> = None;
            fields(&d, s, e, |ef, ev| {
                let Some((a, b)) = ev.range() else { return };
                match ef {
                    3 => {
                        let t = text(&d, (a, b));
                        if t == "Irradiance Probe" || t == "Track Info" {
                            kind = t;
                        }
                    }
                    4 => fields(&d, a, b, |tf, tv| {
                        if let (1, Some((x, y))) = (tf, tv.range()) {
                            pos = vec3_of(&d, x, y);
                        }
                    }),
                    50 => body = Some((a, b)),
                    _ => {}
                }
            });
            let Some((a, b)) = body else { return };
            // surface definitions (actor f109): any entity carrying them
            fields(&d, a, b, |af, av| {
                let (109, Some((x, y))) = (af, av.range()) else { return };
                fields(&d, x, y, |sf, sv| {
                    let (1, Some((i, j))) = (sf, sv.range()) else { return };
                    let mut s = SurfaceDef { grip: 1.0, ..Default::default() };
                    fields(&d, i, j, |f, v| match (f, v) {
                        (1, Val::Bytes(p, q)) => s.id = text(&d, (p, q)),
                        (2, Val::Varint(n)) => s.category = n as u32,
                        (3, v) => s.grip = v.f32(),
                        (8, Val::Varint(n)) => s.valid = n != 0,
                        (10, v) => s.sin_height = v.f32(),
                        (11, v) => s.sin_length = v.f32(),
                        (12, v) => s.damping = v.f32(),
                        (14, v) => s.vibration_gain = v.f32(),
                        (15, v) => s.vibration_length = v.f32(),
                        (20, Val::Varint(n)) => s.kind = n as u32,
                        (18, v) => s.friction_rain = v.f32(),
                        (19, v) => s.friction_rain_gamma = v.f32(),
                        _ => {}
                    });
                    if !s.id.is_empty() {
                        env.surface_defs.push(s);
                    }
                });
            });
            if kind == "Irradiance Probe" {
                fields(&d, a, b, |af, av| {
                    let (120, Some((x, y))) = (af, av.range()) else { return };
                    let mut p = Probe {
                        pos,
                        tex: String::new(),
                        night: String::new(),
                        interval: 4.5,
                        elev: [1.5, 5.5, 13.5],
                        ground: Vec::new(),
                    };
                    fields(&d, x, y, |pf, pv| match (pf, pv) {
                        (1, Val::Bytes(i, j)) => p.tex = text(&d, (i, j)),
                        (12, Val::Bytes(i, j)) => p.night = text(&d, (i, j)),
                        (2, v) => p.interval = v.f32(),
                        (3, v) => p.elev[0] = v.f32(),
                        (4, v) => p.elev[1] = v.f32(),
                        (5, v) => p.elev[2] = v.f32(),
                        (6, Val::Bytes(i, j)) => {
                            p.ground = d[i..j]
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                                .collect();
                        }
                        _ => {}
                    });
                    if !p.tex.is_empty() && p.interval > 0.0 {
                        probes.push(p);
                    }
                });
            } else if kind == "Track Info" {
                // f50.f111.f1.f4 = { f1 lon f64, f2 lat f64 }
                fields(&d, a, b, |x, xv| {
                    let (111, Some((i, j))) = (x, xv.range()) else { return };
                    fields(&d, i, j, |y, yv| {
                        let (1, Some((k, l))) = (y, yv.range()) else { return };
                        fields(&d, k, l, |z, zv| {
                            if let (5, Val::Varint(t)) = (z, zv) {
                                env.tz = t as u32;
                                return;
                            }
                            let (4, Some((m, n))) = (z, zv.range()) else { return };
                            fields(&d, m, n, |w, wv| match w {
                                1 => env.lon = wv.f32() as f64,
                                2 => {
                                    env.lat = wv.f32() as f64;
                                    env.has_geo = true;
                                }
                                _ => {}
                            });
                        });
                    });
                });
            }
        });
    }
    if !probes.is_empty() {
        env.iv = build_atlas(vfs, &probes);
        if let Some(iv) = &env.iv {
            let mut sum = 0.0f64;
            let mut n = 0usize;
            for p in &probes {
                for g in &p.ground {
                    if g.is_finite() {
                        sum += *g as f64;
                        n += 1;
                    }
                }
            }
            env.ground_y = if n > 0 { (sum / n as f64) as f32 } else { 0.0 };
            println!(
                "track env: {} irradiance probe(s), atlas {}x12x{} at ({:.0},{:.0}), {:.2}s",
                probes.len(), iv.nx * 16, iv.nz * 16, iv.origin[0], iv.origin[1],
                t0.elapsed().as_secs_f32());
        }
    }
    if env.has_geo {
        println!("track env: lat {:.3} lon {:.3}, time zone {}", env.lat, env.lon,
                 crate::sky::TZ_NAMES.get(env.tz as usize).unwrap_or(&"?"));
        crate::sky::place(env.lat, env.lon, env.tz);
    }
    env.lights = load_lights(vfs);
    env.cloud_noise = load_cloud_noise(vfs);
    if env.cloud_noise.is_some() {
        println!("track env: cloud noise 512x512x64 BC6H");
    }
    if let Some(p) = &env.pigment_path {
        println!("track env: pigment {} over {:?}", p, env.pigment_rect);
    }
    env
}

/// Every LightData (actor f101) and InstancedLightData (f139) in the
/// track's scenes, containers included.
///
/// ⚠ Guesses (strict ids TL1..TL3): a spot shines along its local -Y (367 of
/// 490 spots at four circuits point down that way, 9 up; +Z would point 167
/// of them at the sky); container scenes are taken as placed at the world
/// origin; instanced positions are world positions, as the mesh loader
/// reads them.
pub fn load_lights(vfs: &mut Vfs) -> Vec<TrackLight> {
    let mut out = Vec::new();
    let keys = vfs.own_keys("scene");
    let parse_light = |d: &[u8], a: usize, b: usize| -> TrackLight {
        let mut l = TrackLight {
            pos: [0.0; 3], dir: [0.0, -1.0, 0.0], color: [1.0; 3], intensity: 1.0, radius: 10.0,
            inner: 0.0, outer: 45.0, falloff: 0.0, spot: false, always_on: false,
        };
        fields(d, a, b, |f, v| match (f, v) {
            (1, Val::Varint(t)) => l.spot = t == 1,
            (2, Val::Bytes(x, y)) => {
                let c = vec3_of(d, x, y);
                l.color = c;
            }
            (3, v) => l.intensity = v.f32(),
            (4, v) => l.radius = v.f32(),
            (5, v) => l.inner = v.f32(),
            (6, v) => l.outer = v.f32(),
            (21, v) => l.falloff = v.f32(),
            (17, Val::Varint(t)) => l.always_on = t != 0,
            _ => {}
        });
        l
    };
    // where each container scene is placed (ContainerActorData f121 under a
    // node): the lamps inside are local to it. One level, like cams.rs.
    let norm = |p: &str| p.to_lowercase().replace('\\', "/");
    let mut placed: std::collections::HashMap<String, glam::Mat4> = Default::default();
    for key in &keys {
        let d = vfs.read(key);
        fields(&d, 0, d.len(), |f, v| {
            let (2, Some((s, e))) = (f, v.range()) else { return };
            let mut xf = None;
            let mut path = None;
            fields(&d, s, e, |g, x| match (g, x.range()) {
                (4, Some((ta, tb))) => xf = Some(crate::load::parse_transform(&d, ta, tb)),
                (50, Some((ia, ib))) => fields(&d, ia, ib, |h, y| {
                    if let (121, Some((ca, cb))) = (h, y.range()) {
                        fields(&d, ca, cb, |m, z| if let (1, Some(r)) = (m, z.range()) { path = Some(text(&d, r)) });
                    }
                }),
                _ => {}
            });
            if let (Some(xf), Some(path)) = (xf, path) {
                placed.insert(norm(&path), crate::load::trs(xf.0, xf.1, xf.2));
            }
        });
    }
    for key in keys {
        let d = vfs.read(&key);
        let cont = placed.get(&norm(&key)).copied().unwrap_or(glam::Mat4::IDENTITY);
        if std::env::var_os("EVOVIEW_LIGHT_LOG").is_some() && cont != glam::Mat4::IDENTITY {
            println!("LIGHTCONT {key} at {:?}", cont.w_axis.truncate());
        }
        fields(&d, 0, d.len(), |f, v| {
            let (2, Some((s, e))) = (f, v.range()) else { return };
            let mut pos = [0.0f32; 3];
            let mut rot = [0.0f32; 3];
            let mut scl = [1.0f32; 3];
            let mut body = None;
            fields(&d, s, e, |ef, ev| {
                let Some((a, b)) = ev.range() else { return };
                match ef {
                    4 => fields(&d, a, b, |tf, tv| {
                        if let Some((x, y)) = tv.range() {
                            match tf {
                                1 => pos = vec3_of(&d, x, y),
                                2 => rot = vec3_of(&d, x, y),
                                3 => scl = vec3_of(&d, x, y),
                                _ => {}
                            }
                        }
                    }),
                    50 => body = Some((a, b)),
                    _ => {}
                }
            });
            let Some((a, b)) = body else { return };
            // the node's transform: a lamp actor's own place, and the frame
            // an instanced set's positions are in (like instanced meshes,
            // load.rs). ⚠ Container scenes' own placement is not applied
            // here yet.
            let s0 = if scl == [0.0; 3] { [1.0; 3] } else { scl };
            let node = cont * crate::load::trs(pos, rot, s0);
            let down_m = |m: glam::Mat4| -> [f32; 3] {
                let v = -m.y_axis.truncate().normalize_or_zero();
                v.into()
            };
            fields(&d, a, b, |af, av| {
                let Some((x, y)) = av.range() else { return };
                if af == 101 {
                    let mut l = parse_light(&d, x, y);
                    l.pos = node.w_axis.truncate().into();
                    l.dir = down_m(node);
                    out.push(l);
                } else if af == 139 {
                    // { f1 max_distance, f2 fade, f3 LightData, f4 InstanceSetData }
                    let mut proto = None;
                    let (mut pb, mut rb) = (None, None);
                    fields(&d, x, y, |lf, lv| {
                        let Some((i, j)) = lv.range() else { return };
                        match lf {
                            3 => proto = Some(parse_light(&d, i, j)),
                            4 => fields(&d, i, j, |sf, sv| {
                                if let Some(r) = sv.range() {
                                    match sf {
                                        2 => pb = Some(r),
                                        3 => rb = Some(r),
                                        _ => {}
                                    }
                                }
                            }),
                            _ => {}
                        }
                    });
                    let (Some(p), Some((ps, pe))) = (proto, pb) else { return };
                    let f3 = |s: usize, i: usize| -> [f32; 3] {
                        std::array::from_fn(|k| f32::from_le_bytes(d[s + i * 12 + k * 4..s + i * 12 + k * 4 + 4].try_into().unwrap()))
                    };
                    if std::env::var_os("EVOVIEW_LIGHT_LOG").is_some() {
                        println!("LIGHTSET {} lamps node pos {pos:?} rot {rot:?}", (pe - ps) / 12);
                    }
                    for i in 0..(pe - ps) / 12 {
                        let mut l = p;
                        let e = match rb {
                            Some((rs, re)) if rs + (i + 1) * 12 <= re => f3(rs, i),
                            _ => [0.0; 3],
                        };
                        let m = node * crate::load::trs(f3(ps, i), e, [1.0; 3]);
                        l.pos = m.w_axis.truncate().into();
                        l.dir = down_m(m);
                        out.push(l);
                    }
                }
            });
        });
    }
    if !out.is_empty() {
        let spots = out.iter().filter(|l| l.spot).count();
        println!("track env: {} lamp(s) ({} spot, {} point)", out.len(), spots, out.len() - spots);
        crate::strict::note("TL1", "track lamps", "spot lights shine along local -Y");
        crate::strict::note("TL2", "track lamps", "intensity units unknown; scaled by EVOVIEW_LAMP_K");
    }
    out
}

fn build_atlas(vfs: &mut Vfs, probes: &[Probe]) -> Option<IvAtlas> {
    let interval = probes[0].interval;
    let cell = interval * 16.0;
    let minx = probes.iter().map(|p| p.pos[0]).fold(f32::MAX, f32::min);
    let minz = probes.iter().map(|p| p.pos[2]).fold(f32::MAX, f32::min);
    let maxx = probes.iter().map(|p| p.pos[0]).fold(f32::MIN, f32::max);
    let maxz = probes.iter().map(|p| p.pos[2]).fold(f32::MIN, f32::max);
    let nx = ((maxx - minx) / cell).round() as u32 + 1;
    let nz = ((maxz - minz) / cell).round() as u32 + 1;
    if nx == 0 || nz == 0 || nx > 256 || nz > 256 {
        return None;
    }
    let (w, h, dpt) = ((nx * 16) as usize, 24usize, (nz * 16) as usize);
    let mut texels = vec![0u16; w * h * dpt * 4];
    let mut filled = vec![false; (nx * nz) as usize];
    // running mean, for columns no probe covers
    let mut mean = [[0.0f64; 3]; 4];
    let mut mean_g = 0.0f64;
    let mut count = 0usize;
    let mut decoded: Vec<(usize, usize, Vec<[f32; 3]>, Vec<f32>, Option<Vec<[f32; 3]>>)> = Vec::new();
    let mut n_night = 0usize;
    let mut read_iv = |vfs: &mut Vfs, tex: &str| -> Option<Vec<[f32; 3]>> {
        if tex.is_empty() {
            return None;
        }
        let found = vfs.find(&format!("{}mips", tex), None)?;
        let raw = vfs.read(&found);
        if raw.len() < 64 * 3 * 16 * 4 {
            return None;
        }
        Some(raw
            .chunks_exact(4)
            .take(64 * 3 * 16)
            .map(|c| r11g11b10(u32::from_le_bytes(c.try_into().unwrap())))
            .collect())
    };
    for p in probes {
        let ix = ((p.pos[0] - minx) / cell).round() as usize;
        let iz = ((p.pos[2] - minz) / cell).round() as usize;
        let Some(px) = read_iv(vfs, &p.tex) else { continue };
        let night = read_iv(vfs, &p.night);
        n_night += night.is_some() as usize;
        for (i, c) in px.iter().enumerate() {
            let coef = i % 4;
            for k in 0..3 {
                mean[coef][k] += c[k] as f64;
            }
        }
        let g: Vec<f32> = if p.ground.len() >= 256 { p.ground.clone() } else { vec![0.0; 256] };
        mean_g += g.iter().map(|x| *x as f64).sum::<f64>() / 256.0;
        count += 1;
        decoded.push((ix, iz, px, g, night));
    }
    if count == 0 {
        return None;
    }
    let n_px = (count * 16 * 3 * 16) as f64;
    let mean: Vec<[f32; 3]> = mean
        .iter()
        .map(|m| [(m[0] / n_px) as f32, (m[1] / n_px) as f32, (m[2] / n_px) as f32])
        .collect();
    let mean_g = (mean_g / count as f64) as f32;
    let mut put = |x: usize, y: usize, z: usize, c: [f32; 3], g: f32| {
        let o = ((z * h + y) * w + x) * 4;
        texels[o] = f32_to_f16(c[0]);
        texels[o + 1] = f32_to_f16(c[1]);
        texels[o + 2] = f32_to_f16(c[2]);
        texels[o + 3] = f32_to_f16(g);
    };
    for (ix, iz, px, g, night) in &decoded {
        filled[iz * nx as usize + ix] = true;
        for cz in 0..16 {
            for layer in 0..3 {
                for cx in 0..16 {
                    for coef in 0..4 {
                        let i = (cz * 3 + layer) * 64 + cx * 4 + coef;
                        let gh = g[cz * 16 + cx];
                        put(ix * 16 + cx, coef * 3 + layer, iz * 16 + cz, px[i], gh);
                        let nc = night.as_ref().map_or([0.0; 3], |n| n[i]);
                        put(ix * 16 + cx, 12 + coef * 3 + layer, iz * 16 + cz, nc, gh);
                    }
                }
            }
        }
    }
    if n_night > 0 {
        println!("track env: night irradiance on {n_night} of {} probe(s)", decoded.len());
    }
    for iz in 0..nz as usize {
        for ix in 0..nx as usize {
            if filled[iz * nx as usize + ix] {
                continue;
            }
            for cz in 0..16 {
                for y in 0..12 {
                    for cx in 0..16 {
                        put(ix * 16 + cx, y, iz * 16 + cz, mean[y / 3], mean_g);
                        put(ix * 16 + cx, 12 + y, iz * 16 + cz, [0.0; 3], mean_g);
                    }
                }
            }
        }
    }
    Some(IvAtlas {
        origin: [minx, minz],
        interval,
        nx,
        nz,
        elev: probes[0].elev,
        texels,
    })
}

/// Sun direction (unit vector TO the sun, world Y up) for a latitude and a
/// local solar time on a day of the year.
///
/// ⚠ World orientation: taken as +X east, -Z north. The track files carry no
/// north vector that has been found yet; override with EVOVIEW_SUN=az,el
/// (degrees, azimuth clockwise from north).
pub fn sun_direction(lat_deg: f64, day: f64, solar_hour: f64) -> (f32, f32) {
    let decl = 23.44f64.to_radians() * ((360.0 / 365.0) * (day - 81.0)).to_radians().sin();
    let lat = lat_deg.to_radians();
    let ha = (15.0 * (solar_hour - 12.0)).to_radians();
    let el = (lat.sin() * decl.sin() + lat.cos() * decl.cos() * ha.cos()).asin();
    let az = (-ha.sin()).atan2(decl.tan() * lat.cos() - lat.sin() * ha.cos());
    let az = (az.to_degrees() + 360.0) % 360.0;
    (az as f32, el.to_degrees() as f32)
}

pub fn sun_vector(az_deg: f32, el_deg: f32) -> glam::Vec3 {
    let (az, el) = (az_deg.to_radians(), el_deg.to_radians());
    // north = -Z, east = +X
    glam::Vec3::new(az.sin() * el.cos(), el.sin(), -az.cos() * el.cos()).normalize()
}

// ------------------------------------------------------------------- GPU ---

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FrameUniform {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub eye: [f32; 4],
    pub sun: [f32; 4],
    pub sun_color: [f32; 4],
    pub haze: [f32; 4],
    pub iv_origin: [f32; 4],
    pub iv_elev: [f32; 4],
    pub misc: [f32; 4],
    /// the base's sun shadow map: world -> light clip, and its parameters
    pub light_vp: [[f32; 4]; 4],
    pub shadow: [f32; 4],
    /// the lead's grade: x exposure, y use LUT, z which LUT (1 outdoor)
    pub grade: [f32; 4],
    /// pigment map rect: x min x, y min z, z 1/size x, w 1/size z (0 = none)
    pub pigment: [f32; 4],
    /// EVOVIEW_TRACK_DEBUG: 1 albedo, 2 irradiance x10, 3 AO, 4 normal, 5 shadow, 8 LOD crossfade
    pub debug: [f32; 4],
    /// time of day (sky::Sky::pack)
    pub sky: [[f32; 4]; 4],
    /// lamp grid: x min x, y min z, z 1/cell, w cells across
    pub lamp_grid: [f32; 4],
    /// x cells deep, y lamp count, z lamp strength (EVOVIEW_LAMP_K x on), w night IV on
    pub lamps: [f32; 4],
    /// the far shadow cascade (render::Uniforms.light_vp_far / shadow_far)
    pub light_vp_far: [[f32; 4]; 4],
    pub shadow_far: [f32; 4],
    /// x haze start distance (m), y cloud steps (0 = default), z per-frame
    /// cloud-march offset (golden-ratio sequence)
    pub haze2: [f32; 4],
    /// clouds: x base height above the ground (m), y thickness, z coverage
    /// (0 = none), w density
    pub clouds: [f32; 4],
    /// x noise scale, y erosion detail scale, z erosion width, w erosion
    /// hardness
    pub clouds2: [f32; 4],
    /// x time (s, for wind), y wind x, z wind z (m/s), w 1 = noise present
    pub clouds3: [f32; 4],
    /// x cirrus amount (0 = none), y cirrus height above the ground (m),
    /// z 1 = cirrus texture present, w cloud-shadow strength on the ground
    pub clouds4: [f32; 4],
    /// weather map: x present, y length m, z width m, w angle rad
    pub clouds5: [f32; 4],
    /// cloud shadow map: x min x, y min z, z span (m), w 1 = drawn
    pub cmap: [f32; 4],
    /// x ground wetness, y falling rain (rain.rs), z time
    pub wet: [f32; 4],
    /// the middle cascade and the driven car's hero map
    /// (render::Uniforms.light_vp_mid / shadow_mid / _hero)
    pub light_vp_mid: [[f32; 4]; 4],
    pub shadow_mid: [f32; 4],
    pub light_vp_hero: [[f32; 4]; 4],
    pub shadow_hero: [f32; 4],
    /// x texture LOD bias (TAA's in the main view, 0 in the others)
    pub lod: [f32; 4],
    /// physical sky (sky_pano_pass): xyz the true sun direction (the key
    /// light is the moon at night), w 1 = on
    pub atm: [f32; 4],
    /// x the cloud panorama's history blend (1 = no history), y radiance
    /// scale, z cloud march steps in the panorama, w falling rain
    pub atm2: [f32; 4],
    /// cloud light: x sun gain, y ambient gain, z extinction scale toward
    /// the sun, w powder strength
    pub atm3: [f32; 4],
    /// x the deck's extinction as seen (not as shadow) over the shadow's,
    /// y the share of TAA's mip bias see-through materials take
    pub atm4: [f32; 4],
}

/// The cloud shadow map: texels across, and the ground it spans (m). The
/// noise itself is ~16 m a texel (8 km over 512), so 16 m texels lose nothing.
pub const CMAP_DIM: u32 = 512;

/// The sky panorama (track.wgsl pano_uv): azimuth across, elevation down
/// from the zenith to 0.1 rad below the horizon, rows spaced by the square
/// root so the horizon band gets the most of them.
pub struct SkyPano {
    pub on: bool,
    /// the cloud panorama's size; the clear sky's is `sky_w` x `sky_h`
    /// (smooth, so small)
    w: u32,
    h: u32,
    sky_w: u32,
    sky_h: u32,
    sky: wgpu::Texture,
    cloud: wgpu::Texture,
    sky_hist_tex: wgpu::Texture,
    cloud_hist_tex: wgpu::Texture,
    sky_view: wgpu::TextureView,
    cloud_view: wgpu::TextureView,
    pub sky_hist: wgpu::TextureView,
    pub cloud_hist: wgpu::TextureView,
    sun: std::cell::Cell<glam::Vec3>,
    frames: std::cell::Cell<u32>,
}

pub const PANO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

impl SkyPano {
    fn new(device: &wgpu::Device) -> SkyPano {
        // EVOVIEW_SKY_MODEL=old: the gradient sky, clouds marched per pixel
        let on = std::env::var("EVOVIEW_SKY_MODEL").map_or(true, |v| v != "old");
        let (w, h, sky_w, sky_h) = if on {
            (env_f32("EVOVIEW_PANO_W", 4096.0) as u32, env_f32("EVOVIEW_PANO_H", 1024.0) as u32,
             env_f32("EVOVIEW_ATM_W", 512.0) as u32, env_f32("EVOVIEW_ATM_H", 256.0) as u32)
        } else {
            (1, 1, 1, 1)
        };
        let mk = |label: &str, w: u32, h: u32, usage: wgpu::TextureUsages| device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PANO_FORMAT,
            usage,
            view_formats: &[],
        });
        let out = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        let hist = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let sky = mk("sky pano", sky_w, sky_h, out);
        let cloud = mk("cloud pano", w, h, out);
        let sky_hist_tex = mk("sky pano (read)", sky_w, sky_h, hist);
        let cloud_hist_tex = mk("cloud pano (read)", w, h, hist);
        SkyPano {
            on,
            w,
            h,
            sky_w,
            sky_h,
            sky_view: sky.create_view(&Default::default()),
            cloud_view: cloud.create_view(&Default::default()),
            sky_hist: sky_hist_tex.create_view(&Default::default()),
            cloud_hist: cloud_hist_tex.create_view(&Default::default()),
            sky,
            cloud,
            sky_hist_tex,
            cloud_hist_tex,
            sun: std::cell::Cell::new(glam::Vec3::ZERO),
            frames: std::cell::Cell::new(0),
        }
    }
}
pub const CMAP_SPAN: f32 = 8192.0;

/// Lamps on the GPU: 3 vec4 each (pos+radius, colour*intensity+falloff,
/// dir+cos outer, then cos inner / spot / -) in one storage buffer, and a
/// uniform XZ grid of index lists so a pixel only walks the lamps that can
/// reach it.
pub const LAMP_CELL: f32 = 24.0;

pub struct LampGrid {
    pub lamps: Vec<[f32; 16]>,
    /// per cell: [start, count] into `index`
    pub cells: Vec<[u32; 2]>,
    pub index: Vec<u32>,
    pub rect: [f32; 4],
    pub nz: u32,
}

pub fn lamp_grid(lights: &[TrackLight]) -> Option<LampGrid> {
    if lights.is_empty() {
        return None;
    }
    let (mut x0, mut z0, mut x1, mut z1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for l in lights {
        x0 = x0.min(l.pos[0] - l.radius);
        z0 = z0.min(l.pos[2] - l.radius);
        x1 = x1.max(l.pos[0] + l.radius);
        z1 = z1.max(l.pos[2] + l.radius);
    }
    let nx = (((x1 - x0) / LAMP_CELL).ceil() as u32).clamp(1, 512);
    let nz = (((z1 - z0) / LAMP_CELL).ceil() as u32).clamp(1, 512);
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); (nx * nz) as usize];
    let mut lamps = Vec::with_capacity(lights.len());
    for (i, l) in lights.iter().enumerate() {
        let cell = |v: f32, o: f32, n: u32| (((v - o) / LAMP_CELL).floor() as i64).clamp(0, n as i64 - 1) as u32;
        for cz in cell(l.pos[2] - l.radius, z0, nz)..=cell(l.pos[2] + l.radius, z0, nz) {
            for cx in cell(l.pos[0] - l.radius, x0, nx)..=cell(l.pos[0] + l.radius, x0, nx) {
                lists[(cz * nx + cx) as usize].push(i as u32);
            }
        }
        let (ci, co) = (l.inner.max(0.0).min(89.0).to_radians().cos(), l.outer.max(0.1).min(89.5).to_radians().cos());
        lamps.push([
            l.pos[0], l.pos[1], l.pos[2], l.radius.max(0.1),
            l.color[0] * l.intensity, l.color[1] * l.intensity, l.color[2] * l.intensity, l.falloff,
            l.dir[0], l.dir[1], l.dir[2], co,
            ci.max(co + 1e-3), if l.spot { 1.0 } else { 0.0 }, if l.always_on { 1.0 } else { 0.0 }, 0.0,
        ]);
    }
    let mut cells = Vec::with_capacity(lists.len());
    let mut index = Vec::new();
    for l in &lists {
        cells.push([index.len() as u32, l.len() as u32]);
        index.extend_from_slice(l);
    }
    if index.is_empty() {
        index.push(0);
    }
    Some(LampGrid { lamps, cells, index, rect: [x0, z0, 1.0 / LAMP_CELL, nx as f32], nz })
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct TrackMatUniform {
    pub base: [f32; 4],
    pub surf: [f32; 4],
    pub base_uv: [f32; 4],
    pub red: [f32; 4],
    pub green: [f32; 4],
    pub blue: [f32; 4],
    pub red_uv: [f32; 4],
    pub green_uv: [f32; 4],
    pub blue_uv: [f32; 4],
    pub layer_rough: [f32; 4],
    pub splat: [f32; 4],
    pub splat_const: [f32; 4],
    pub detail: [f32; 4],
    pub vao: [f32; 4],
    pub emissive: [f32; 4],
    pub flags: [f32; 4],
    /// x pigment tint on, y light wrap, z specular scale, w unused
    pub extra: [f32; 4],
    /// vegetation wind (the game's MaterialVS, lifted): 0 weight channel,
    /// weight constant, phase channel, phase constant; 1 flutter channel,
    /// flutter constant, stiffness, intensity; 2 height intensity, frequency,
    /// height remap min, max (max <= min = no remap)
    pub wind: [[f32; 4]; 3],
    /// material AO map: x strength, y mode (0 none, 1 Uber family
    /// 1 + (map - 1) x strength, 2 Subsurface / Fence / Grass sat(map) x
    /// strength), zw unused
    pub ao: [f32; 4],
}

/// How many of the scene's materials (from index 0) the track pipeline
/// shades - 0 when it is disabled (EVOVIEW_OLD_TRACK) or there is no track.
pub fn owned_material_count(car: &Car) -> usize {
    if std::env::var("EVOVIEW_OLD_TRACK").is_ok() {
        return 0;
    }
    car.track_env.as_ref().map_or(0, |e| e.n_materials)
}

/// Material key the way Materials::lookup sees it: basename, lowercase, no
/// extension.
fn mat_key(r: &str) -> String {
    r.rsplit(['\\', '/']).next().unwrap_or(r).trim_end_matches(".material").to_lowercase()
}

/// Ground cover that the game thins out with distance: grass tufts, low
/// plants. Judged from the mesh file name (the loader keeps only that).
pub fn grass_mesh_name(name: &str) -> bool {
    let n = name.to_lowercase();
    ["grass", "clover", "flower", "crowberry", "fern", "weed", "plant", "tundra", "heather"]
        .iter()
        .any(|k| n.contains(k))
        && !n.contains("terrain")
}

/// Beyond this a grass chunk is not drawn at all (the shader has faded it
/// out by then).
pub const GRASS_MAX_DIST: f32 = 150.0;

/// The texture slots a track material binds, by role.
struct Slots {
    base: Option<usize>,
    base_n: Option<usize>,
    base_r: Option<usize>,
    splat: Option<usize>,
    red: Option<usize>,
    red_n: Option<usize>,
    green: Option<usize>,
    green_n: Option<usize>,
    blue: Option<usize>,
    blue_n: Option<usize>,
    detail: Option<usize>,
    opacity: Option<usize>,
    /// the material's AO map (Base_AmbientOcclusionMap for the Uber family,
    /// the top-level AmbientOcclusionMap for Subsurface / Fence / Grass)
    ao: Option<usize>,
    /// the base map is the loader's shared asphalt, not the material's own
    base_is_road_fallback: bool,
    /// the base map is the legacy txDiffuse (DynamicTrack roads)
    base_is_tx: bool,
    /// a DynamicTrack road with the road shader's own maps (after material
    /// inheritance): base = Base_BaseMap (packed n n a r), detail =
    /// Detail_Map (a r a o), red = Variation_Map, red_n = Detail_NormalMap
    dt: bool,
    /// the Variation_Map has a g channel (r colour, g roughness); a
    /// one-channel BC4 reads g = 0
    red_rg: bool,
}

fn v2(m: &MaterialEntry, name: &str, d: [f32; 2]) -> [f32; 2] {
    if let Some(v) = m.colors.get(name) {
        if v.len() >= 2 {
            return [v[0], v[1]];
        }
    }
    if let Some(v) = m.params.get(name) {
        return [*v, *v];
    }
    d
}

/// Whether single-sided track materials drop the face the game culls
/// (EVOVIEW_TRACK_CULL=0: everything two-sided, as before).
fn track_cull() -> bool {
    std::env::var("EVOVIEW_TRACK_CULL").map_or(true, |v| v != "0")
}

impl TrackMatUniform {
    fn from_entry(m: &MaterialEntry, s: &Slots, grass: bool) -> (Self, Blend) {
        let shader = m.shader.as_deref().unwrap_or("").to_lowercase();
        let file = m.file.to_lowercase();
        let p = |n: &str, d: f32| m.param(&[n], d);
        let mut u = TrackMatUniform::default();
        let kind = if shader.contains("multilayer") {
            2.0
        } else if shader.contains("dynamictrack") {
            1.0
        } else if shader == "fence" || shader == "subsurface" || file.contains("leaf")
            || file.contains("leaves") || file.contains("tree") || file.contains("bush")
            || file.contains("shrub") || file.contains("billboard") || grass
        {
            3.0
        } else if shader.contains("glass") {
            4.0
        } else {
            0.0
        };
        let w = |id: &'static str, d: String| crate::strict::note(id, &m.file, d);
        const KNOWN: &[&str] = &["uberbiplanarmaterial", "ubermaterial", "uberinteriormaterial", "dynamictrackmultilayer",
            "dynamictrack", "defaultmaterial", "fence", "subsurface", "grass", "interiormapping", "semaphore", "particle"];
        if !shader.is_empty() && !KNOWN.contains(&shader.as_str()) && !shader.contains("glass") {
            w("TR25", m.shader.clone().unwrap_or_default());
        }
        if kind == 2.0 {
            w("TR26", "road shaded as a generic surface".into());
        }
        if kind == 3.0 && shader != "fence" && shader != "subsurface" && !grass {
            w("TR44", "foliage by file name".into());
        }
        let blend_mode = p("blendMode", 0.0) as i32;
        if blend_mode == 3 {
            w("TR45", "blendMode 3 drawn as a cutout".into());
        }
        let blend = if kind == 4.0 || blend_mode == 1 {
            Blend::Alpha
        } else if kind == 3.0 || blend_mode >= 2 || s.opacity.is_some() {
            Blend::Cutout
        } else {
            Blend::Opaque
        };
        // A bound albedo carries the colour; Base_Basecolor then is a tint
        // (often 1). Without one it IS the colour.
        let base_mod = p("Modulation_BaseBaseColor", 0.0) > 0.5
            || p("Base_ModulateBaseColor", 0.0) > 0.5
            || p("Modulation_BaseColor", 0.0) > 0.5;
        if s.base.is_none() && !m.has(&["Base_Basecolor", "Basecolor", "ksBaseColor"]) {
            w("TR12", "no albedo, no base colour".into());
        }
        let tint = if s.base.is_some() {
            if base_mod {
                m.color(&["Base_Basecolor", "Basecolor"], [1.0, 1.0, 1.0, 1.0])
            } else {
                [1.0, 1.0, 1.0, 1.0]
            }
        } else {
            m.color(&["Base_Basecolor", "Basecolor", "ksBaseColor"], [0.6, 0.6, 0.6, 1.0])
        };
        // ⚠ Fence/foliage Base_Basecolor is a near-black subsurface tint.
        // ⚠ Only the Grass shader: props (crowd, seats, trucks) also carry
        // Pigment_Enable, but there it means a per-object colour mask.
        let pigment = p("Pigment_Enable", 0.0) > 0.5 && shader == "grass";
        let tint = if pigment {
            // Grass cards are grey; Basecolor (~2) x the pigment map gives the
            // ground's own colour.
            let b = p("Basecolor", 1.0).clamp(0.1, 4.0);
            [b, b, b, 1.0]
        } else if kind == 3.0 && s.base.is_some() {
            [1.0, 1.0, 1.0, 1.0]
        } else {
            tint
        };
        // DynamicTrack with only editor placeholders gets the shared asphalt
        // base map from the loader; the game's asphalt_dynamic material tints
        // that map by 0.25, not by the track's 0.58 grey.
        let tint = if kind == 1.0 && s.base_is_road_fallback { [0.25, 0.25, 0.25, 1.0] } else { tint };
        if kind == 1.0 && (s.base_is_road_fallback || s.base_is_tx) {
            w("TR41", "road drawn from one asphalt map".into());
        }
        // DynamicTrack: the runtime road shader builds the asphalt from
        // several packed maps we do not decode; the material's legacy
        // txDiffuse (a plain asphalt photo at UVscale) tinted by its
        // Base_Basecolor is the closest single map.
        let road_tx = kind == 1.0 && s.base_is_tx;
        let tint = if road_tx {
            let c = m.color(&["Base_Basecolor"], [0.5, 0.5, 0.5, 1.0]);
            [(c[0] * 1.6).min(1.0), (c[1] * 1.6).min(1.0), (c[2] * 1.6).min(1.0), 1.0]
        } else {
            tint
        };
        u.base = [tint[0].min(2.0), tint[1].min(2.0), tint[2].min(2.0), kind];
        if !m.has(&["Base_Roughness", "Roughness"]) {
            w("TR14", "no roughness (0.8)".into());
        }
        // ⚠ Legacy-slot materials (the real albedo is txDiffuse; the Base_*
        // maps are editor placeholders the loader drops) take the
        // unprefixed constants. Spa's background_trees has Base_Roughness 0
        // beside Roughness 1: read as 0 its tree-line walls were mirrors of
        // the sky - the blue-grey "curtains" across the hills.
        let legacy = m.slot(&["Base_BaseColorMap"]).is_none() && m.slot(&["txDiffuse"]).is_some();
        let (refl, rough) = if legacy {
            (p("Reflectance", p("Base_Reflectance", 0.5)), p("Roughness", p("Base_Roughness", 0.8)))
        } else {
            (p("Base_Reflectance", p("Reflectance", 0.5)), p("Base_Roughness", p("Roughness", 0.8)))
        };
        let (refl, rough) = (refl.clamp(0.0, 1.0), rough.clamp(0.05, 1.0));
        // the engine's cutout from the material's top-level a2c fields
        // (SPEC_small §1): 50 % coverage at cutoff + 0.5 / scale
        let (a2s, a2c) = (p("__a2c_scale", 0.0), p("__a2c_cutoff", -1.0));
        let cutoff = if blend != Blend::Cutout { 0.0 }
            else if a2c >= 0.0 && a2s > 0.0 { (a2c + 0.5 / a2s).clamp(0.05, 0.95) }
            else { p("a2cCutoff", 0.5).clamp(0.05, 0.95) };
        // The game does not threshold: its cutout prepass (0x03350b70) keeps a
        // pixel where sat(scale x (alpha - cutoff)) beats a temporally shifted
        // 4x4 Bayer, and TAA turns that into partial coverage. The shader
        // rebuilds it as sat(scale x (alpha - surf.w) + 0.5) around the 50 %
        // point above. Materials without the fields get the cars' 2.0, made
        // steep enough that alpha 0 stays at no coverage and 1 at full (a
        // leaf card thresholded at 0.05 would otherwise dither its whole
        // empty rectangle at 40 %). EVOVIEW_CUTOUT_DITHER=0: the old hard
        // threshold.
        let dith = blend == Blend::Cutout && env_f32("EVOVIEW_CUTOUT_DITHER", 1.0) > 0.5;
        let a2_scale = if !dith { 0.0 } else {
            let s = if a2c >= 0.0 && a2s > 0.0 { a2s } else { 2.0 };
            s.max(0.5 / cutoff.max(0.01)).max(0.5 / (1.0 - cutoff).max(0.01))
        };
        let nscale = p("Base_NormalScale", p("NormalScale", 1.0)).abs().clamp(0.0, 3.0);
        u.surf = [rough, refl, nscale.max(0.01), cutoff];
        // ao.z opacity scale (0 = hard threshold), ao.w the alpha read's own
        // mip bias (top-level f11)
        u.ao[2] = a2_scale;
        u.ao[3] = if dith { p("__a2c_mip", 0.0).clamp(-4.0, 4.0) } else { 0.0 };
        let buv = if road_tx {
            v2(m, "UVscale", [1.0, 1.0])
        } else {
            v2(m, "Base_UVscale", v2(m, "UVscale", [1.0, 1.0]))
        };
        u.base_uv = [buv[0], buv[1], p("Base_UVchannel", 0.0), if s.base.is_some() { 1.0 } else { 0.0 }];

        // ⚠ Modulation_<Layer>BaseColor is NOT "layer over base". Per the
        // spec (SPEC_evo_shading 2.1) it chooses whether the map MULTIPLIES the
        // layer's constant colour (1) or REPLACES it (0). Reading it as a
        // multiply onto the base map made every grass hill 4x too dark.
        let layer = |name: &str, tex: Option<usize>| -> ([f32; 4], [f32; 4], f32) {
            let on = p(&format!("{name}_Enable"), 0.0) > 0.5 && tex.is_some();
            let modulate = p(&format!("Modulation_{name}BaseColor"), 0.0) > 0.5;
            let c = if modulate {
                m.color(&[&format!("{name}_Basecolor")], [1.0, 1.0, 1.0, 1.0])
            } else {
                [1.0, 1.0, 1.0, 1.0]
            };
            let uv = v2(m, &format!("{name}_UVscale"), [1.0, 1.0]);
            let trip = p(&format!("{name}_Triplanar"), 0.0) > 0.5
                || p(&format!("{name}_WorldAlign"), 0.0) > 0.5;
            if on && p(&format!("{name}_WorldAlign"), 0.0) > 0.5 && p(&format!("{name}_Triplanar"), 0.0) < 0.5 {
                crate::strict::note("TR17", &m.file, format!("{name}_WorldAlign drawn triplanar"));
            }
            if on && !m.has(&[&format!("{name}_Roughness")]) {
                crate::strict::note("TR15", &m.file, format!("{name}_Roughness absent (0.9)"));
            }
            let ts = if trip {
                v2(m, &format!("{name}_TriplanarScale"), [0.1, 0.1])[0].abs().max(0.002)
            } else {
                0.0
            };
            let ch = p(&format!("{name}_UVchannel"), 0.0);
            (
                [c[0].min(2.5), c[1].min(2.5), c[2].min(2.5), if on { 1.0 } else { 0.0 }],
                [uv[0], uv[1], ts, ch],
                p(&format!("{name}_Roughness"), 0.9).clamp(0.05, 1.0),
            )
        };
        let (r, ruv, rr) = layer("Red", s.red);
        let (g, guv, gr) = layer("Green", s.green);
        let (b, bv, br) = layer("Blue", s.blue);
        u.red = r;
        u.green = g;
        u.blue = b;
        u.red_uv = ruv;
        u.green_uv = guv;
        u.blue_uv = bv;
        // Splatmap_MultiplyBaseColor: layers multiply what is under them
        // (kerb paint x a concrete grain) instead of replacing it.
        let modmask = 0u32;
        if p("Splatmap_MultiplyBaseColor", 0.0) > 0.5 {
            w("TR18", "Splatmap_MultiplyBaseColor ignored".into());
        }
        u.layer_rough = [rr, gr, br, modmask as f32];
        let any_layer = r[3] + g[3] + b[3] > 0.5;
        let use_const = p("Splatmap_UseConstant", 0.0) > 0.5;
        let use_vertex = p("Splatmap_UseVertexInsteadOfTexture", 0.0) > 0.5;
        let mode = if !any_layer {
            0.0
        } else if use_const {
            3.0
        } else if use_vertex {
            2.0
        } else if s.splat.is_some() {
            1.0
        } else {
            // no source of weights: Red alone, the way Splatmap_Constant=1 reads
            w("TR20", "no splat weights; Red alone".into());
            3.0
        };
        let sc = m.color(&["Splatmap_Constant"], [1.0, 0.0, 0.0, 0.0]);
        let sc = if m.colors.contains_key("Splatmap_Constant") {
            [sc[0], sc[1], sc[2]]
        } else if let Some(v) = m.params.get("Splatmap_Constant") {
            [*v, 0.0, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        let mut nmask = 0u32;
        if s.red_n.is_some() { nmask |= 1; }
        if s.green_n.is_some() { nmask |= 2; }
        if s.blue_n.is_some() { nmask |= 4; }
        let contrast = if p("Splatmap_UseHeightBlend", 0.0) > 0.5 { 1.6 } else { 1.0 };
        if contrast > 1.0 {
            w("TR19", "height blend approximated".into());
        }
        u.splat = [mode, p("Splatmap_UVchannel", 0.0), nmask as f32, contrast];
        let vscale = m.color(&["Splatmap_VertexColorScale"], [1.0, 1.0, 1.0, 1.0])[0].max(0.01);
        u.splat_const = [sc[0], sc[1], sc[2], vscale];
        // Detail = the base map re-tinting the splat result (see track.wgsl)
        let det_on = p("Detail_Enable", 0.0) > 0.5 && kind != 3.0;
        let dm = v2(m, "detailMultiplier", [0.125, 0.125]);
        // detailMultiplier reads as tiles per metre on the tracks surveyed
        // (0.125 = an 8 m tile); keep it in a sane band.
        u.detail = [if det_on { 1.0 } else { 0.0 }, p("Detail_BaseColorMultiplier", 1.0),
                    dm[0].abs().clamp(0.02, 4.0), 0.35];
        let vao_on = p("Vertex_UseVertexAmbientOcclusion", 0.0) > 0.5;
        u.vao = [
            if vao_on { 1.0 } else { 0.0 },
            p("Vertex_AmbientOcclusionIntensity", 255.0),
            p("Vertex_AmbientOcclusionOffset", 0.0),
            p("Vertex_AmbientOcclusionChannel", 3.0),
        ];
        let ec = m.color(&["Base_EmissiveColor", "EmissiveColor"], [0.0, 0.0, 0.0, 0.0]);
        let ei = p("Base_EmissiveIntensity", p("EmissiveIntensity", 0.0));
        if file.contains("emissive") && ei <= 0.0 {
            w("TR22", "glows by file name".into());
        }
        let e = if file.contains("emissive") || ei > 0.0 {
            let k = (ei.max(1.0)).min(40.0) * 0.08;
            [ec[0] * k, ec[1] * k, ec[2] * k]
        } else {
            [0.0; 3]
        };
        u.emissive = [e[0], e[1], e[2], 0.0];
        u.extra = [
            if pigment { 1.0 } else { 0.0 },
            p("LightWrap", p("Base_LightWrap", if kind == 3.0 { 0.5 } else { 0.0 })).clamp(0.0, 1.0),
            p("SpecularScale", 1.0).clamp(0.0, 2.0),
            // bit 0 Splatmap_UseMultiply, bit 1 DisableRestrictBlendToTwoLayers,
            // bit 2 doNotInvertNormalOnBackside (tree-line walls authored with
            // up-facing normals: flipped on the back face they faced the
            // ground and turned into grey curtains)
            ((p("Splatmap_UseMultiply", 0.0) > 0.5) as u32
                | (((p("Splatmap_DisableRestrictBlendToTwoLayers", 0.0) > 0.5) as u32) << 1)
                | (((p("doNotInvertNormalOnBackside", 0.0) > 0.5) as u32) << 2)
                // bit 3 MaterialData.windable (vegetation wind, track.wgsl vs_main)
                | (((p("__windable", 0.0) > 0.5 && std::env::var_os("EVOVIEW_NO_WIND").is_none()) as u32) << 3)
                // bits 4 / 5: MaterialData.cullMode Front / Back - the face
                // the game culls is discarded (track.wgsl shade). Opaque
                // materials only: our cutouts include foliage chosen by file
                // name and turned cards. EVOVIEW_TRACK_CULL=0 two-sided.
                | (((blend == Blend::Opaque && track_cull() && p("__cull", 0.0) < 0.5) as u32) << 4)
                | (((blend == Blend::Opaque && track_cull() && (p("__cull", 0.0) - 1.0).abs() < 0.5) as u32) << 5)
                // bit 6: alpha-blended (fences, nets, screens): their own
                // texture LOD bias (track.wgsl lod_bias). Not the cutouts:
                // grass and foliage cards went visibly soft at 0
                | (((blend == Blend::Alpha) as u32) << 6)) as f32,
        ];
        u.flags = [
            if s.base_n.is_some() { 1.0 } else { 0.0 },
            if s.base_r.is_some() { 1.0 } else { 0.0 },
            if s.opacity.is_some() { 1.0 } else { 0.0 },
            if grass { 1.0 } else { 0.0 },
        ];
        // proto3: an absent wind parameter is 0 (channel 0 = vertex red)
        u.wind = [
            [p("Wind_WeightChannel", 0.0), p("Wind_WeightConstant", 0.0), p("Wind_PhaseChannel", 0.0), p("Wind_PhaseConstant", 0.0)],
            [p("Wind_FlutterChannel", 0.0), p("Wind_FlutterConstant", 0.0), p("Wind_Stiffness", 0.0), p("Wind_Intensity", 0.0)],
            [p("Wind_HeightIntensity", 0.0), p("Wind_Frequency", 0.0), p("Height_RemapMin", 0.0), p("Height_RemapMax", 0.0)],
        ];
        // The Grass shader's subsurface (lift 0350dcb0): sunlight through the
        // blades, a lobe exp(Subsurface_Power (v.-l - 1)) round the sun seen
        // through them, in SubsurfaceColorMap x Subsurface_Color x Amount.
        // Carried in the blue layer (grass has no splat layers): rgb colour,
        // w on; layer_rough.w the power.
        // (also the Subsurface shader's foliage: trees light up from behind;
        // its Subsurface_ShadowFactor keeps part of it in the canopy's shade)
        if (pigment || shader == "subsurface") && p("Subsurface_Amount", 0.0) > 0.0 && s.blue.is_some() && s.splat.is_none() {
            let sc = m.color(&["Subsurface_Color"], [1.0, 1.0, 1.0, 1.0]);
            let a = p("Subsurface_Amount", 1.0);
            u.blue = [sc[0] * a, sc[1] * a, sc[2] * a, 1.0];
            u.layer_rough[2] = p("Subsurface_ShadowFactor", 0.0).clamp(0.0, 1.0);
            u.layer_rough[3] = p("Subsurface_Power", 3.0).max(0.1);
        }
        if s.dt {
            // DynamicTrack, the road shader's own recipe (splat.x = 9 picks
            // it in track.wgsl surface()):
            //   colour = Base_Basecolor x BaseMap.z x Detail.r (world XZ)
            //            -> toward x Variation.r x Variation_Basecolor
            //   normal = BaseMap.xy, plus the detail normal in world XZ
            let v2 = |n: &str, d: [f32; 2]| m.colors.get(n).filter(|v| v.len() >= 2).map_or(d, |v| [v[0], v[1]]);
            let v3 = |n: &str, d: [f32; 3]| m.colors.get(n).filter(|v| v.len() >= 3).map_or(d, |v| [v[0], v[1], v[2]]);
            let bc = m.color(&["Base_Basecolor"], [0.2, 0.2, 0.2, 1.0]);
            let buv = v2("Base_UVscale", [1.0, 1.0]);
            let vc = m.color(&["Variation_Basecolor"], [1.0, 1.0, 1.0, 1.0]);
            let vuv = v2("Variation_UVscale", [1.0, 1.0]);
            let dc = m.color(&["Detail_Basecolor"], [1.0, 1.0, 1.0, 1.0]);
            let ds = v3("Detail_BiplanarScale", [0.55, 0.55, 0.55]);
            u.splat = [9.0, 0.0, 0.0, 0.0];
            u.base = [bc[0], bc[1], bc[2], kind];
            u.base_uv = [if buv[0] != 0.0 { buv[0] } else { 1.0 }, if buv[1] != 0.0 { buv[1] } else { 1.0 }, p("Base_ModulateBaseColor", 0.0), 1.0];
            u.red = [vc[0], vc[1], vc[2], if s.red.is_some() { p("Variation_Enable", 1.0) } else { 0.0 }];
            u.red_uv = [if vuv[0] != 0.0 { vuv[0] } else { 1.0 }, if vuv[1] != 0.0 { vuv[1] } else { 1.0 },
                        p("Variation_UVchannel", 1.0), p("Variation_ModulateBaseColor", 0.0)];
            u.green = [dc[0], dc[1], dc[2], p("Detail_ModulateBaseColor", 0.0)];
            // the variation map's roughness (its g): x Variation_Roughness,
            // y Variation_ModulateRoughness, z 1 = the map has a g channel
            // w: Vertex_UseVertexVariation - the variation only where the
            // vertex colour marks it, by saturate((c + offset) x intensity):
            // 0 off, else 1 + Vertex_VariationChannel (0 r .. 3 a, the lift's
            // switch). Offset / intensity ride in ao.zw (a road has no
            // cutout, so cutout_drop never reads them)
            u.layer_rough = [p("Variation_Roughness", 1.0), p("Variation_ModulateRoughness", 0.0),
                             if s.red_rg { 1.0 } else { 0.0 },
                             if p("Vertex_UseVertexVariation", 0.0) > 0.5 { 1.0 + p("Vertex_VariationChannel", 0.0).clamp(0.0, 3.0).round() } else { 0.0 }];
            u.ao[2] = p("Vertex_VariationOffset", 0.0);
            u.ao[3] = p("Vertex_VariationIntensity", 1.0);
            u.green_uv = [ds[0] + 0.01, ds[2] + 0.01, p("Detail_NormalScale", 0.8),
                          if s.red_n.is_some() { p("Detail_HasNormalMap", 1.0) } else { 0.0 }];
            u.blue = [p("Base_Roughness", 0.7), p("Detail_Roughness", 1.0), p("Base_NormalScale", 0.75),
                      p("Detail_RoughnessStrenght", 0.5)];
            u.flags = [0.0, 0.0, if s.opacity.is_some() { 1.0 } else { 0.0 }, if grass { 1.0 } else { 0.0 }];
            crate::strict::note("TR60", &m.file, "DynamicTrack road: lifted colour recipe");
        }
        if let Ok(f) = std::env::var("EVOVIEW_MAT_PARAMS") {
            if m.file.contains(f.as_str()) {
                let mut ks: Vec<String> = m.params.iter().map(|(k, v)| format!("{k}={v}")).collect();
                ks.extend(m.colors.iter().map(|(k, v)| format!("{k}={v:?}")));
                ks.extend(m.slots.iter().map(|(k, v)| format!("{k}->#{v}")));
                ks.sort();
                eprintln!("MATP {} [{}] :: {}", m.file, m.shader.as_deref().unwrap_or("?"), ks.join(" "));
            }
        }
        // EVOVIEW_TRACKMAT_HL=a,b: those track materials (file name contains)
        // drawn flat magenta, to find which one owns a pixel
        if let Ok(hl) = std::env::var("EVOVIEW_TRACKMAT_HL") {
            if hl.split(',').any(|k| !k.is_empty() && m.file.contains(k)) {
                u.base = [1.0, 0.0, 1.0, u.base[3]];
                u.base_uv[3] = 0.0;
                u.red[3] = 0.0;
                u.green[3] = 0.0;
                u.blue[3] = 0.0;
                u.splat[0] = 0.0;
                u.detail[0] = 0.0;
            }
        }
        (u, blend)
    }
}

/// Per-draw data for one owned material.
struct TrackMat {
    group: wgpu::BindGroup,
    blend: Blend,
    /// MaterialData.windable (extra.w bit 3): drawn into the motion vectors
    wind: bool,
}

pub struct TrackRender {
    opaque: wgpu::RenderPipeline,
    cutout: wgpu::RenderPipeline,
    blended: wgpu::RenderPipeline,
    sky: wgpu::RenderPipeline,
    /// opaque, cutout, blended, sky for the other views (mirror, probe,
    /// rear camera), whose passes have no SSR targets; None without SSR
    view_pipes: Option<[wgpu::RenderPipeline; 4]>,
    frame_bg: wgpu::BindGroup,
    pub frame_buf: wgpu::Buffer,
    /// the clouds' shadow on the ground around the eye, drawn each frame
    /// (cloud_map_pass) so a track pixel reads one texel instead of
    /// marching the deck itself (12 noise reads; ~0.5 ms at 3440x1440)
    cmap_view: wgpu::TextureView,
    cmap_bg: wgpu::BindGroup,
    cmap_pipe: wgpu::RenderPipeline,
    /// the sky as a panorama around the eye (sky_pano_pass): the physical
    /// atmosphere and the cloud deck, marched once per texel instead of once
    /// per screen pixel, and averaged over frames so the march's jitter
    /// never reaches the screen
    pano: SkyPano,
    pano_bg: wgpu::BindGroup,
    pano_pipe: wgpu::RenderPipeline,
    atm_pipe: wgpu::RenderPipeline,
    /// indexed by the scene's material index (the same index Materials uses)
    mats: Vec<Option<TrackMat>>,
    pub sun_dir: glam::Vec3,
    sun_e: f32,
    exposure: f32,
    ambient: f32,
    haze_density: f32,
    /// metres from the eye before haze begins
    haze_start: f32,
    haze_base: f32,
    iv_origin: [f32; 4],
    iv_elev: [f32; 4],
    pigment: [f32; 4],
    pub lit: bool,
    carpet: Vec<CarpetSet>,
    /// furthest carpet reach, metres (the grass fade ends there)
    carpet_reach: f32,
    /// MSAA with alpha-to-coverage on the cutout pipeline
    a2c: bool,
    /// EVOVIEW_TRACK_FACING=1: camera-facing cards as authored (they spin)
    /// camera-facing cards (MeshLodData f20): 0 off, 1 aligned to the view
    /// direction (default), 2 turned toward the eye position
    facing: f32,
    /// lamp grid rect and (cells deep, lamp count)
    lamp_rect: [f32; 4],
    lamp_info: [f32; 2],
    /// Alpha-tested shadow casting for cutout materials (tree and forest
    /// cards): the car-style shadow pass has no fragment stage, so a card
    /// threw its whole rectangle as a solid shadow - the dark slabs and
    /// triangles across Spa's hillsides. One frame buffer + group per
    /// cascade, its view_proj the light matrix, with a stand-in at the
    /// shadow-map slot (the map is being written).
    shadow_cut: wgpu::RenderPipeline,
    shadow_bufs: [wgpu::Buffer; 3],
    shadow_bgs: [wgpu::BindGroup; 3],
    /// animated props and their bone buffer (binding 14)
    anim: Option<std::sync::Arc<TrackAnim>>,
    /// the clip frames last written (TrackAnim::frame_key): the clips step
    /// at their own ~30 fps, so most frames have nothing new to pose
    anim_key: std::cell::RefCell<(Vec<u32>, Vec<[f32; 16]>)>,
    bone_buf: wgpu::Buffer,
    /// motion vectors for the animated props and wind-swayed vegetation
    /// (draw_motion): last frame's bone sets, the pipelines (plain, with
    /// DLSS's RG copy), their group 2, and which scene meshes animate
    prev_bone_buf: wgpu::Buffer,
    mv_pipes: [wgpu::RenderPipeline; 2],
    mv_bgl: wgpu::BindGroupLayout,
    mv_buf: wgpu::Buffer,
    anim_src: std::collections::HashSet<usize>,
    /// last frame's clock, once a frame has been drawn since a reset
    mv_prev_t: std::cell::Cell<Option<f32>>,
    /// mirror views: their own frame uniform, built from the main one
    mirror_buf: wgpu::Buffer,
    mirror_bg: wgpu::BindGroup,
    /// the reflection probe's faces (probe.rs), likewise
    probe_buf: wgpu::Buffer,
    probe_bg: wgpu::BindGroup,
    /// the rear-view camera screen, likewise
    rear_buf: wgpu::Buffer,
    rear_bg: wgpu::BindGroup,
    last_frame: std::cell::Cell<Option<FrameUniform>>,
    /// the dry line (rain::dry_map), and whether it has been written yet
    dry_tex: wgpu::Texture,
    dry_on: std::cell::Cell<bool>,
    /// the main view's texture LOD bias (taa.rs mip_bias)
    mip_bias: f32,
}

pub const EXTRA_FLOATS: usize = 6;

/// Grass carpet: one tuft mesh with every one of its instances on the CPU,
/// bucketed into cells; only the cells near the camera are uploaded. Each
/// MeshLodData level is its own GPU mesh: near tufts get the full mesh, far
/// ones the cheap cards, and each level keeps only its `density` fraction
/// (MeshLodData f25) - the way the game thins its carpet with distance.
struct CarpetLevel {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    extra: wgpu::Buffer,
    instances: wgpu::Buffer,
    cap: u32,
    count: std::cell::Cell<u32>,
    draws: Vec<(u32, u32, usize)>,
    /// switch-in distance, metres
    near: f32,
    /// metres before `near` it starts to dither in (and after it, the
    /// level before dithers out)
    dlt: f32,
    /// fraction of tufts kept at this level
    density: f32,
}

struct CarpetSet {
    levels: Vec<CarpetLevel>,
    cells: HashMap<(i32, i32), Vec<u32>>,
    /// mean ground height of each cell, for a true 3D distance test
    cell_y: HashMap<(i32, i32), f32>,
    rows: Vec<[f32; 20]>,
    last_eye: std::cell::Cell<Option<glam::Vec3>>,
    /// the lodDistanceScale of the last upload
    last_scale: std::cell::Cell<f32>,
    /// beyond this nothing is drawn (the shader has faded it by then)
    reach: f32,
}

const CARPET_CELL: f32 = 16.0;
/// instances per level uploaded at once (nearest first past this)
const CARPET_CAP: u32 = 30_000;

impl CarpetLevel {
    fn new(device: &wgpu::Device, car: &Car, m: &crate::asset::MeshEntry,
           lookup: &dyn Fn(&str) -> usize, cap: u32) -> Option<Self> {
        let pos = car.floats(m.positions);
        let nrm = car.floats(m.normals);
        let uv = car.floats(m.uvs);
        let tan = car.floats(m.tangents);
        let col = car.floats(m.colors);
        let uv1 = car.floats(m.uv1);
        let idx = car.u32s(m.indices);
        let n = m.vertices as usize;
        if n == 0 || idx.is_empty() || pos.len() < n * 3 {
            return None;
        }
        let fpv = crate::render::FLOATS_PER_VERTEX;
        let mut v = vec![0f32; n * fpv];
        let mut e = vec![0f32; n * EXTRA_FLOATS];
        for i in 0..n {
            let o = i * fpv;
            v[o..o + 3].copy_from_slice(&pos[i * 3..i * 3 + 3]);
            if nrm.len() >= n * 3 { v[o + 3..o + 6].copy_from_slice(&nrm[i * 3..i * 3 + 3]); } else { v[o + 4] = 1.0; }
            if uv.len() >= n * 2 { v[o + 6..o + 8].copy_from_slice(&uv[i * 2..i * 2 + 2]); }
            if tan.len() >= n * 4 { v[o + 8..o + 12].copy_from_slice(&tan[i * 4..i * 4 + 4]); }
            let q = i * EXTRA_FLOATS;
            if col.len() >= n * 4 { e[q..q + 4].copy_from_slice(&col[i * 4..i * 4 + 4]); } else { e[q..q + 4].copy_from_slice(&[1.0 / 255.0; 4]); }
            if uv1.len() >= n * 2 { e[q + 4..q + 6].copy_from_slice(&uv1[i * 2..i * 2 + 2]); }
        }
        let draws = m.batches.iter()
            .filter(|b| b.count > 0 && (b.start + b.count) as usize <= idx.len())
            .map(|b| (b.start, b.count, lookup(&b.material)))
            .collect();
        let [near, density] = m.lod.unwrap_or([0.0, 1.0]);
        let dlt = m.lod_fade[0];
        Some(CarpetLevel {
            vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&m.name), contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::VERTEX,
            }),
            indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&m.name), contents: bytemuck::cast_slice(&idx), usage: wgpu::BufferUsages::INDEX,
            }),
            extra: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&m.name), contents: bytemuck::cast_slice(&e), usage: wgpu::BufferUsages::VERTEX,
            }),
            instances: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("carpet instances"),
                size: cap as u64 * 80,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            cap,
            count: std::cell::Cell::new(0),
            draws,
            near,
            dlt,
            density: density.clamp(0.0, 1.0),
        })
    }
}

impl CarpetSet {
    /// `levels[0]` carries the instances; the rest are its coarser LODs.
    fn new(device: &wgpu::Device, car: &Car, levels: &[crate::asset::MeshEntry],
           lookup: &dyn Fn(&str) -> usize) -> Option<Self> {
        let m = levels.first()?;
        let cap = CARPET_CAP.min(m.xforms.len() as u32).max(1);
        let lv: Vec<CarpetLevel> = levels.iter()
            .filter_map(|l| CarpetLevel::new(device, car, l, lookup, cap))
            .collect();
        if lv.is_empty() {
            return None;
        }
        let mut cells: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        let mut rows = Vec::with_capacity(m.xforms.len());
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for p in car.floats(m.positions).chunks_exact(3) {
            lo = lo.min(p[1]);
            hi = hi.max(p[1]);
        }
        if lo > hi {
            (lo, hi) = (0.0, 0.0);
        }
        for (k, xf) in m.xforms.iter().enumerate() {
            let key = ((xf[12] / CARPET_CELL).floor() as i32, (xf[14] / CARPET_CELL).floor() as i32);
            cells.entry(key).or_default().push(k as u32);
            let mut r = [0f32; 20];
            r[..16].copy_from_slice(xf);
            r[16] = m.facing as f32;
            // the wind's height falloff (wind_rows does it for scene meshes):
            // without bounds every tuft moved whole, its root sliding
            r[18] = lo;
            r[19] = hi;
            rows.push(r);
        }
        let cell_y: HashMap<(i32, i32), f32> = cells
            .iter()
            .map(|(k, v)| (*k, v.iter().map(|&i| m.xforms[i as usize][13]).sum::<f32>() / v.len().max(1) as f32))
            .collect();
        // the last level is drawn out to 1.75x its switch-in distance
        let last = lv.last().map_or(40.0, |l| l.near.max(20.0));
        Some(CarpetSet {
            levels: lv,
            cells,
            cell_y,
            rows,
            last_eye: std::cell::Cell::new(None),
            last_scale: std::cell::Cell::new(0.0),
            reach: last * 1.75,
        })
    }

    /// Upload the tufts within reach of `eye`, nearest cells first, each to
    /// the level its distance selects (thinned to that level's density).
    /// `s` is the lodDistanceScale (cull.rs lod_scale): the switch distances
    /// and the reach scale with it, the dither widths do not (033aa890).
    fn update(&self, queue: &wgpu::Queue, eye: glam::Vec3, max_radius: f32, s: f32) {
        // Re-sorted every EVOVIEW_CARPET_STEP metres of camera travel: at
        // the old 4 m a tuft's crossfade (2 m / 5 m wide) jumped a whole
        // band in one update, so the grass switched in visible bursts.
        static STEP: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
        let step = *STEP.get_or_init(|| env_f32("EVOVIEW_CARPET_STEP", 1.0));
        if let Some(l) = self.last_eye.get() {
            if l.distance(eye) < step && self.last_scale.get() == s {
                return;
            }
        }
        self.last_eye.set(Some(eye));
        self.last_scale.set(s);
        let radius = (self.reach * s).min(max_radius);
        let r = (radius / CARPET_CELL).ceil() as i32 + 1;
        let (cx, cz) = ((eye.x / CARPET_CELL).floor() as i32, (eye.z / CARPET_CELL).floor() as i32);
        let mut near: Vec<(f32, &Vec<u32>)> = Vec::new();
        for dz in -r..=r {
            for dx in -r..=r {
                if let Some(v) = self.cells.get(&(cx + dx, cz + dz)) {
                    let c = glam::Vec2::new((cx + dx) as f32 + 0.5, (cz + dz) as f32 + 0.5) * CARPET_CELL;
                    // 3D: from a camera high above the track the grass right
                    // under it is still too far to be drawn
                    let dy = eye.y - self.cell_y.get(&(cx + dx, cz + dz)).copied().unwrap_or(eye.y);
                    let d = (c.distance_squared(glam::Vec2::new(eye.x, eye.z)) + dy * dy).sqrt();
                    if d < radius + CARPET_CELL {
                        near.push((d, v));
                    }
                }
            }
        }
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        let nl = self.levels.len();
        let mut data: Vec<Vec<f32>> = vec![Vec::new(); nl];
        let mut n = vec![0u32; nl];
        let r2 = radius * radius;
        for (_, v) in near {
            for &k in v {
                let row = &self.rows[k as usize];
                let p = glam::Vec3::new(row[12], row[13], row[14]);
                let d2 = p.distance_squared(eye);
                if d2 > r2 {
                    continue;
                }
                let d = d2.sqrt();
                let lvl = self.levels.iter().rposition(|l| l.near * s <= d).unwrap_or(0);
                // a stable per-tuft hash, so thinning does not flicker
                let h = ((k.wrapping_mul(2654435761)) >> 8) as f32 / 16777216.0;
                // The game's crossfade (cull.rs Band): the level whose band
                // holds d draws whole; the next one dithers in over the dlt
                // metres before its switch, the previous one out over the dlt
                // after. A tuft the coarser level's density drops fades out
                // instead of vanishing.
                let mut put = |li: usize, cov: f32| {
                    let l = &self.levels[li];
                    if h >= l.density || n[li] >= l.cap || cov <= 0.0 {
                        return;
                    }
                    let at = data[li].len();
                    data[li].extend_from_slice(row);
                    data[li][at + 3] = 1.0 - (cov.min(1.0) * 63.0).floor() / 63.0;
                    n[li] += 1;
                };
                put(lvl, 1.0);
                if let Some(nx) = self.levels.get(lvl + 1) {
                    if nx.dlt > 0.0 && d >= nx.near * s - nx.dlt {
                        put(lvl + 1, (d - (nx.near * s - nx.dlt)) / nx.dlt);
                    }
                }
                let cur = &self.levels[lvl];
                if lvl > 0 && cur.dlt > 0.0 && d < cur.near * s + cur.dlt {
                    put(lvl - 1, 1.0 - (d - cur.near * s) / cur.dlt);
                }
            }
        }
        for (i, l) in self.levels.iter().enumerate() {
            if n[i] > 0 {
                queue.write_buffer(&l.instances, 0, bytemuck::cast_slice(&data[i]));
            }
            l.count.set(n[i]);
        }
    }
}

/// The grass carpet's lodDistanceScale: the props' (cull.rs lod_scale, the
/// game's preset / fov term), or 1 with EVOVIEW_CARPET_SCALE=0 (the
/// carpet's raw distances, as before).
fn carpet_scale(view_proj: glam::Mat4) -> f32 {
    if std::env::var("EVOVIEW_CARPET_SCALE").map_or(false, |v| v == "0") {
        return 1.0;
    }
    let fy = view_proj.row(1).truncate().length().max(1e-6);
    crate::cull::lod_scale(Some(2.0 * (1.0 / fy).atan()))
}

fn env_f32(name: &str, d: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

impl TrackRender {
    /// Build the track pipeline and the owned materials. Returns None when the
    /// scene has no track materials (a car on its own).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        car: &Car,
        format: wgpu::TextureFormat,
        samples: u32,
        shadow_view: &wgpu::TextureView,
        shadow_samp: &wgpu::Sampler,
        lut_view: &wgpu::TextureView,
        materials: &crate::material::Materials,
        clouds: &crate::clouds::CloudGpu,
    ) -> Option<Self> {
        if std::env::var("EVOVIEW_OLD_TRACK").is_ok() {
            return None;
        }
        // The track loader's materials come first (a replay scene merges the
        // track before any car), so ownership is just an index range.
        let n_track = owned_material_count(car);
        let owned: Vec<bool> = (0..car.materials.len()).map(|i| i < n_track).collect();
        if !owned.iter().any(|x| *x) {
            return None;
        }
        let t0 = std::time::Instant::now();
        let tex_entry = |binding: u32, dim: wgpu::TextureViewDimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: dim,
                multisampled: false,
            },
            count: None,
        };
        let storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("track frame bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex_entry(1, wgpu::TextureViewDimension::D3),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // the lead's grading LUT (grade.rs), so tracks and cars grade alike
                tex_entry(5, wgpu::TextureViewDimension::D3),
                // pigment map (ground colour for grass cards)
                tex_entry(6, wgpu::TextureViewDimension::D2),
                // the track's lamps: lamps, grid cells, cell index lists
                storage_entry(7),
                storage_entry(8),
                storage_entry(9),
                // cloud noise (64 slices) + a repeating sampler
                tex_entry(10, wgpu::TextureViewDimension::D2Array),
                // cirrus layer
                tex_entry(12, wgpu::TextureViewDimension::D2),
                // weather map
                tex_entry(13, wgpu::TextureViewDimension::D2),
                // the cloud shadow map (cloud_map_pass)
                tex_entry(15, wgpu::TextureViewDimension::D2),
                // the dry line (rain::dry_map)
                tex_entry(16, wgpu::TextureViewDimension::D2),
                // the sky panorama (sky_pano_pass): clear sky, cloud deck
                tex_entry(17, wgpu::TextureViewDimension::D2),
                tex_entry(18, wgpu::TextureViewDimension::D2),
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // animated props' bones (TrackAnim)
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let anim = car.track_env.as_ref().and_then(|e| e.anim.clone());
        let bone_buf = {
            // EVOVIEW_TRACK_ANIM_T: the time a still shot is posed at
            let t0 = std::env::var("EVOVIEW_TRACK_ANIM_T").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            let p = anim.as_ref().map_or(vec![crate::load::IDENTITY], |a| a.pose(t0));
            wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
                label: Some("track bones"),
                contents: bytemuck::cast_slice(&p),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            })
        };
        let prev_bone_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("track bones, last frame"),
            size: bone_buf.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut mat_entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ];
        for b in 2..=14 {
            mat_entries.push(tex_entry(b, wgpu::TextureViewDimension::D2));
        }
        let mat_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("track material bgl"),
            entries: &mat_entries,
        });

        // ---- irradiance atlas ----
        let env = car.track_env.as_deref();
        let (iv_view, iv_origin, iv_elev) = match env.and_then(|e| e.iv.as_ref()) {
            Some(iv) => {
                let size = wgpu::Extent3d {
                    width: iv.nx * 16,
                    height: 24,
                    depth_or_array_layers: iv.nz * 16,
                };
                let tex = device.create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("irradiance atlas"),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D3,
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    Default::default(),
                    bytemuck::cast_slice(&iv.texels),
                );
                let wm = iv.nx as f32 * 16.0 * iv.interval;
                let dm = iv.nz as f32 * 16.0 * iv.interval;
                (
                    tex.create_view(&Default::default()),
                    [iv.origin[0], iv.origin[1], 1.0 / wm, 1.0 / dm],
                    [iv.elev[0], iv.elev[1], iv.elev[2], 1.0],
                )
            }
            None => {
                let tex = device.create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("irradiance none"),
                        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D3,
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    Default::default(),
                    bytemuck::cast_slice(&[0u16; 4]),
                );
                (tex.create_view(&Default::default()), [0.0; 4], [1.5, 5.5, 13.5, 0.0])
            }
        };
        let clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("track clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let pig_view = env
            .and_then(|e| e.pigment_tex)
            .and_then(|i| crate::material::upload(device, queue, car, i, true));
        let pigment = match (env, pig_view.is_some()) {
            (Some(e), true) => [e.pigment_rect[0], e.pigment_rect[1],
                                1.0 / e.pigment_rect[2], 1.0 / e.pigment_rect[3]],
            _ => [0.0; 4],
        };
        let pig_view = pig_view.unwrap_or_else(|| crate::material::solid(device, queue, [128, 128, 128, 255]));
        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("track frame"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // ---- clouds (shared with the car pipeline, clouds.rs) ----
        let (cloud_view, cirrus_view, weather_view, cloud_samp) =
            (&clouds.noise, &clouds.cirrus, &clouds.weather, &clouds.samp);

        // ---- lamps ----
        let grid = env.and_then(|e| lamp_grid(&e.lights));
        let storage = |label: &str, bytes: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let (lamp_buf, cell_buf, index_buf, lamp_rect, lamp_info) = match &grid {
            Some(g) => (
                storage("track lamps", bytemuck::cast_slice(&g.lamps)),
                storage("track lamp cells", bytemuck::cast_slice(&g.cells)),
                storage("track lamp index", bytemuck::cast_slice(&g.index)),
                g.rect,
                [g.nz as f32, g.lamps.len() as f32],
            ),
            None => (storage("track lamps", &[0u8; 64]), storage("track lamp cells", &[0u8; 8]),
                     storage("track lamp index", &[0u8; 4]), [0.0; 4], [0.0; 2]),
        };
        let pano = SkyPano::new(device);
        let cmap_view = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud shadow map"),
            size: wgpu::Extent3d { width: CMAP_DIM, height: CMAP_DIM, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&Default::default());
        let dry_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("track dry line"),
            size: wgpu::Extent3d { width: crate::rain::DRY_DIM as u32, height: crate::rain::DRY_DIM as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dry_view = dry_tex.create_view(&Default::default());
        let frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("track frame bg"),
            layout: &frame_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&iv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&clamp) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(shadow_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(shadow_samp) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(lut_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&pig_view) },
                wgpu::BindGroupEntry { binding: 7, resource: lamp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: cell_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: index_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(cloud_view) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(cloud_samp) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(cirrus_view) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(weather_view) },
                wgpu::BindGroupEntry { binding: 14, resource: bone_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&cmap_view) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(&dry_view) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::TextureView(&pano.sky_hist) },
                wgpu::BindGroupEntry { binding: 18, resource: wgpu::BindingResource::TextureView(&pano.cloud_hist) },
            ],
        });
        // the car's mirrors (render.rs): the same frame, another camera
        let mirror_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("track mirror frame"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mirror_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("track mirror frame bg"),
            layout: &frame_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: mirror_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&iv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&clamp) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(shadow_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(shadow_samp) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(lut_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&pig_view) },
                wgpu::BindGroupEntry { binding: 7, resource: lamp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: cell_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: index_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(cloud_view) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(cloud_samp) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(cirrus_view) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(weather_view) },
                wgpu::BindGroupEntry { binding: 14, resource: bone_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&cmap_view) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(&dry_view) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::TextureView(&pano.sky_hist) },
                wgpu::BindGroupEntry { binding: 18, resource: wgpu::BindingResource::TextureView(&pano.cloud_hist) },
            ],
        });

        // the car's reflection probe (probe.rs): the same again
        let probe_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("track probe frame"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let probe_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("track probe frame bg"),
            layout: &frame_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: probe_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&iv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&clamp) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(shadow_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(shadow_samp) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(lut_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&pig_view) },
                wgpu::BindGroupEntry { binding: 7, resource: lamp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: cell_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: index_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(cloud_view) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(cloud_samp) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(cirrus_view) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(weather_view) },
                wgpu::BindGroupEntry { binding: 14, resource: bone_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&cmap_view) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(&dry_view) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::TextureView(&pano.sky_hist) },
                wgpu::BindGroupEntry { binding: 18, resource: wgpu::BindingResource::TextureView(&pano.cloud_hist) },
            ],
        });

        // the rear-view camera screen (mirrors.rs): every frame, its own camera
        let rear_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("track rear camera frame"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rear_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("track rear camera frame bg"),
            layout: &frame_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: rear_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&iv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&clamp) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(shadow_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(shadow_samp) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(lut_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&pig_view) },
                wgpu::BindGroupEntry { binding: 7, resource: lamp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: cell_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: index_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(cloud_view) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(cloud_samp) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(cirrus_view) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(weather_view) },
                wgpu::BindGroupEntry { binding: 14, resource: bone_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&cmap_view) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(&dry_view) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::TextureView(&pano.sky_hist) },
                wgpu::BindGroupEntry { binding: 18, resource: wgpu::BindingResource::TextureView(&pano.cloud_hist) },
            ],
        });

        let shadow_standin = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("track shadow stand-in"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let shadow_bufs: [wgpu::Buffer; 3] = std::array::from_fn(|_| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("track shadow frame"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let shadow_bgs: [wgpu::BindGroup; 3] = std::array::from_fn(|i| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("track shadow frame bg"),
            layout: &frame_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: shadow_bufs[i].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&iv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&clamp) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&shadow_standin) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(shadow_samp) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(lut_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&pig_view) },
                wgpu::BindGroupEntry { binding: 7, resource: lamp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: cell_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: index_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(cloud_view) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(cloud_samp) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(cirrus_view) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(weather_view) },
                wgpu::BindGroupEntry { binding: 14, resource: bone_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&cmap_view) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(&dry_view) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::TextureView(&pano.sky_hist) },
                wgpu::BindGroupEntry { binding: 18, resource: wgpu::BindingResource::TextureView(&pano.cloud_hist) },
            ],
        }));

        // ---- materials ----
        let wrap = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("track wrap"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });
        let white = crate::material::solid(device, queue, [255, 255, 255, 255]);
        let flat = crate::material::solid(device, queue, [128, 128, 255, 255]);
        let black = crate::material::solid(device, queue, [0, 0, 0, 255]);
        // ⚠ One upload per TEXTURE, not per material: the same grass tile is
        // named by a dozen terrain materials.
        // The shared texture cache Materials::build filled (material::TexCache):
        // one GPU copy per (texture, sRGB), whoever binds it.
        let mut n_get = std::collections::HashSet::new();
        let mut get = |i: Option<usize>, srgb: bool| -> Option<std::rc::Rc<wgpu::TextureView>> {
            let i = i?;
            n_get.insert((i, srgb));
            materials.textures.get(device, queue, car, i, srgb)
        };
        // which materials does a grass-ish mesh use
        let mut grass_mats: std::collections::HashSet<String> = Default::default();
        let carpet_meshes: Vec<&crate::asset::MeshEntry> =
            env.map_or(Vec::new(), |e| e.carpet.iter().flatten().collect());
        for mesh in car.meshes.iter().chain(carpet_meshes.iter().copied()) {
            if !mesh.xforms.is_empty() && grass_mesh_name(&mesh.name) {
                for b in &mesh.batches {
                    grass_mats.insert(mat_key(&b.material));
                }
            }
        }
        let mut mats: Vec<Option<TrackMat>> = Vec::with_capacity(car.materials.len());
        let mut n_splat = 0;
        for (mi, m) in car.materials.iter().enumerate() {
            if !owned[mi] {
                mats.push(None);
                continue;
            }
            let sl = |names: &[&str]| m.slot(names);
            let slots = Slots {
                base: sl(&["Base_BaseColorMap", "BaseColorMap", "txDiffuse", "DiffuseMap", "AlbedoMap"]),
                base_n: sl(&["Base_NormalMap", "NormalMap", "txNormal"]),
                base_r: sl(&["Base_RoughnessMap", "RoughnessMap", "txRoughness"]),
                splat: sl(&["SpatMap", "SplatMap"]),
                red: sl(&["Red_BaseColorMap"]),
                red_n: sl(&["Red_NormalMap"]),
                green: sl(&["Green_BaseColorMap"]),
                green_n: sl(&["Green_NormalMap"]),
                blue: sl(&["Blue_BaseColorMap"]),
                blue_n: sl(&["Blue_NormalMap"]),
                detail: sl(&["txDetail", "Detail_BaseColorMap", "DetailMap"]),
                opacity: sl(&["Base_OpacityMap", "OpacityMap", "txOpacity"]),
                ao: None,
                base_is_road_fallback: false,
                base_is_tx: false,
                red_rg: false,
                dt: false,
            };
            let mut slots = slots;
            // the road shader's own maps (lift of DynamicTrack, road/lift_036c4720)
            let is_dt = m.shader.as_deref().map_or(false, |s| s.eq_ignore_ascii_case("DynamicTrack"));
            if is_dt {
                if let (Some(bm), Some(dm)) = (m.slot(&["Base_BaseMap"]), m.slot(&["Detail_Map"])) {
                    slots.base = Some(bm);
                    slots.detail = Some(dm);
                    slots.red = m.slot(&["Variation_Map"]);
                    slots.red_rg = slots.red.and_then(|i| car.textures.get(i)).map_or(false, |t| t.format != "bc4");
                    slots.red_n = m.slot(&["Detail_NormalMap"]);
                    slots.base_n = None;
                    slots.base_r = None;
                    slots.splat = None;
                    slots.green = None;
                    slots.green_n = None;
                    slots.blue = None;
                    slots.blue_n = None;
                    slots.dt = true;
                }
            }
            // the game samples a layer map only under its Has*Map flag (and a
            // layer only when enabled); inherited materials keep the parent's
            // map bound even where the child switched it off (whiteline's
            // Red_BaseColorMap is the parent asphalt, Red_HasBaseColorMap=0)
            if !slots.dt {
                let off = |n: &str| m.params.get(n).map_or(false, |v| *v == 0.0);
                if off("Base_HasBaseColorMap") && m.slot(&["Base_BaseColorMap"]).is_some() && slots.base == m.slot(&["Base_BaseColorMap"]) {
                    slots.base = None;
                }
                if off("Base_HasNormalMap") && slots.base_n == m.slot(&["Base_NormalMap"]) {
                    slots.base_n = None;
                }
                if off("Base_HasRoughnessMap") && slots.base_r == m.slot(&["Base_RoughnessMap"]) {
                    slots.base_r = None;
                }
                for (l, c, n) in [("Red", &mut slots.red, &mut slots.red_n),
                                  ("Green", &mut slots.green, &mut slots.green_n),
                                  ("Blue", &mut slots.blue, &mut slots.blue_n)] {
                    if off(&format!("{l}_HasBaseColorMap")) || off(&format!("{l}_Enable")) {
                        *c = None;
                    }
                    if off(&format!("{l}_HasNormalMap")) || off(&format!("{l}_Enable")) {
                        *n = None;
                    }
                }
            }
            let shader_l = m.shader.as_deref().unwrap_or("").to_lowercase();
            let _ = &shader_l;
            slots.base_is_road_fallback = slots.base
                .and_then(|i| car.textures.get(i))
                .map_or(false, |t| t.name.to_lowercase().starts_with("track_base_albedo"));
            // grass-card treatment (cutout, pigment tint, distance fade) is for
            // the Grass / Subsurface card shaders only: Spa's ground
            // (grass.material, UberBiplanar) is also placed by an instanced
            // grass-named mesh, and as a card it was alpha-tested into holes
            // that showed the sky
            let grass = grass_mats.contains(&mat_key(&m.file))
                && !m.shader.as_deref().unwrap_or("").to_lowercase().starts_with("uber");
            // the Grass shader's subsurface colour map rides in the unused
            // blue layer slot (track.wgsl grass_sss)
            let mut slots = slots;
            let sss_shader = m.shader.as_deref().map_or(false, |s| s.eq_ignore_ascii_case("grass") || s.eq_ignore_ascii_case("subsurface"));
            if sss_shader && slots.splat.is_none() && slots.blue.is_none()
                && m.params.get("HasSubsurfaceColorMap").map_or(false, |v| *v > 0.5) {
                slots.blue = m.slot(&["SubsurfaceColorMap"]);
            }
            if std::env::var("EVOVIEW_MAT_PARAMS").map_or(false, |f| m.file.contains(f.as_str())) {
                let nm = |s: Option<usize>| s.and_then(|i| car.textures.get(i)).map_or("-".to_string(), |t| t.name.clone());
                eprintln!("MATSLOTS {} base={} base_n={} base_r={} splat={} red={} red_n={} green={} green_n={} blue={} blue_n={} detail={} opacity={}",
                          m.file, nm(slots.base), nm(slots.base_n), nm(slots.base_r), nm(slots.splat), nm(slots.red), nm(slots.red_n),
                          nm(slots.green), nm(slots.green_n), nm(slots.blue), nm(slots.blue_n), nm(slots.detail), nm(slots.opacity));
            }
            let (mut u, blend) = TrackMatUniform::from_entry(m, &slots, grass);
            // AO map. Uber / UberBiplanar (lift 034e5ce0): Base_AmbientOcclusion
            // x Base_AmbientOcclusionMap.r under Base_HasAmbientOcclusionMap -
            // their top-level AmbientOcclusionMap is not read (the scene's
            // ambientOcclusionMapIndex is the SSAO buffer). Subsurface (lift
            // 0414eeb0): AmbientOcclusion x sat(AmbientOcclusionMap.r) under
            // HasAmbientOcclusionMap. Fence / Grass taken to be the same
            // (strict TR63). Applied like the car Uber's AO: ambient only.
            let shader_lc = m.shader.as_deref().unwrap_or("").to_lowercase();
            let pv = |k: &str, d: f32| m.params.get(k).copied().unwrap_or(d);
            let (ao_slot, ao_mode, ao_k) = if shader_lc.starts_with("uber") {
                (if pv("Base_HasAmbientOcclusionMap", 0.0) > 0.5 { m.slot(&["Base_AmbientOcclusionMap"]) } else { None },
                 1.0, pv("Base_AmbientOcclusion", 1.0).max(0.0))
            } else if matches!(shader_lc.as_str(), "subsurface" | "fence" | "grass") {
                if shader_lc != "subsurface" {
                    crate::strict::note("TR63", &m.file, "AO map read as the Subsurface shader's");
                }
                (if pv("HasAmbientOcclusionMap", 0.0) > 0.5 { m.slot(&["AmbientOcclusionMap"]) } else { None },
                 2.0, pv("AmbientOcclusion", 1.0).max(0.0))
            } else {
                (None, 0.0, 1.0)
            };
            let ao_view = if slots.dt { None } else { get(ao_slot, false) };
            u.ao = [ao_k, if ao_view.is_some() { ao_mode } else { 0.0 }, u.ao[2], u.ao[3]];
            // textures; a missing map clears its flag so the shader never
            // samples a placeholder as if it were data
            #[allow(clippy::type_complexity)]
            let (base, base_n, base_r, opacity, splat, red, green, blue, red_n, green_n, blue_n, detail) = if slots.dt {
                // the road shader's packed maps are DATA (normal xy, albedo,
                // roughness, AO in channels): linear, never sRGB - read as
                // sRGB the albedo channels came out ~30x too dark
                let base = get(slots.base, false);
                if base.is_none() { u.base_uv[3] = 0.0; }
                let detail = get(slots.detail, false);
                let red = get(slots.red, false);
                if red.is_none() { u.red[3] = 0.0; }
                let red_n = get(slots.red_n, false);
                if red_n.is_none() { u.green_uv[3] = 0.0; }
                let opacity = get(slots.opacity, false);
                if opacity.is_none() { u.flags[2] = 0.0; }
                (base, None, None, opacity, None, red, None, None, red_n, None, None, detail)
            } else {
                let base = get(slots.base, true);
                if base.is_none() { u.base_uv[3] = 0.0; }
                let base_n = get(slots.base_n, false);
                if base_n.is_none() { u.flags[0] = 0.0; }
                let base_r = get(slots.base_r, false);
                if base_r.is_none() { u.flags[1] = 0.0; }
                let opacity = get(slots.opacity, false);
                if opacity.is_none() { u.flags[2] = 0.0; }
                let splat = if u.splat[0] == 1.0 { get(slots.splat, false) } else { None };
                if u.splat[0] == 1.0 && splat.is_none() { u.splat[0] = 3.0; }
                let red = if u.red[3] > 0.5 { get(slots.red, true) } else { None };
                if red.is_none() { u.red[3] = 0.0; }
                let green = if u.green[3] > 0.5 { get(slots.green, true) } else { None };
                if green.is_none() { u.green[3] = 0.0; }
                let blue = if u.blue[3] > 0.5 { get(slots.blue, true) } else { None };
                if blue.is_none() { u.blue[3] = 0.0; }
                let mut nmask = u.splat[2] as u32;
                let red_n = if red.is_some() { get(slots.red_n, false) } else { None };
                if red_n.is_none() { nmask &= !1; }
                let green_n = if green.is_some() { get(slots.green_n, false) } else { None };
                if green_n.is_none() { nmask &= !2; }
                let blue_n = if blue.is_some() { get(slots.blue_n, false) } else { None };
                if blue_n.is_none() { nmask &= !4; }
                u.splat[2] = nmask as f32;
                if u.red[3] + u.green[3] + u.blue[3] < 0.5 {
                    u.splat[0] = 0.0;
                }
                if u.splat[0] > 0.5 {
                    n_splat += 1;
                }
                (base, base_n, base_r, opacity, splat, red, green, blue, red_n, green_n, blue_n, None)
            };
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&m.file),
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let view = |t: &Option<std::rc::Rc<wgpu::TextureView>>, d: &wgpu::TextureView| -> *const wgpu::TextureView {
                match t {
                    Some(r) => &**r as *const _,
                    None => d as *const _,
                }
            };
            // Raw pointers only to dodge borrowck on the Rc clones above; every
            // view outlives this call (they live in `cache` or are locals).
            let views: [*const wgpu::TextureView; 13] = [
                view(&base, &white), view(&base_n, &flat), view(&base_r, &white),
                view(&splat, &black), view(&red, &white), view(&red_n, &flat),
                view(&green, &white), view(&green_n, &flat), view(&blue, &white),
                view(&blue_n, &flat), view(&detail, &white), view(&opacity, &white),
                view(&ao_view, &white),
            ];
            let mut entries = vec![
                wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&wrap) },
            ];
            for (k, v) in views.iter().enumerate() {
                // SAFETY: see above.
                let v: &wgpu::TextureView = unsafe { &**v };
                entries.push(wgpu::BindGroupEntry {
                    binding: 2 + k as u32,
                    resource: wgpu::BindingResource::TextureView(v),
                });
            }
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&m.file),
                layout: &mat_bgl,
                entries: &entries,
            });
            if std::env::var("EVOVIEW_TRACKMAT_LOG").is_ok() {
                eprintln!(
                    "TRACKMAT {:<40} kind={} splat={} R={} G={} B={} base={} grass={} cut={}/{}/{}/{} blend={} shader={}",
                    m.file, u.base[3], u.splat[0], u.red[3], u.green[3], u.blue[3],
                    u.base_uv[3], grass, u.surf[3], u.ao[2],
                    m.params.get("__a2c_cutoff").copied().unwrap_or(-1.0), u.ao[3],
                    match blend { Blend::Opaque => "opaque", Blend::Cutout => "cutout", Blend::Alpha => "alpha", _ => "glass" },
                    m.shader.as_deref().unwrap_or("?"));
            }
            mats.push(Some(TrackMat { group, blend, wind: (u.extra[3] as u32 & 8) != 0 }));
        }
        let uploaded = n_get.len();

        // ---- pipelines ----
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("track shader"),
            // EVOVIEW_TRACK_SHADER=<path> loads the WGSL from disk, for
            // tuning without a rebuild (like EVOVIEW_SHADER for cars)
            source: wgpu::ShaderSource::Wgsl(
                {
                    let src = std::env::var("EVOVIEW_TRACK_SHADER")
                        .ok()
                        .and_then(|p| std::fs::read_to_string(p).ok())
                        .map(|t| crate::clouds::track_source(&t))
                        .unwrap_or_else(|| crate::clouds::track_source(include_str!("track.wgsl")));
                    // TAA resolves the track: its texture LOD bias (taa.rs),
                    // per view from the frame uniform (`lod`)
                    crate::taa::with_mip_bias_expr(&src, "lod_bias()")
                        .replace("MV_DEPTH", if samples > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" })
                        .into()
                }),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("track layout"),
            bind_group_layouts: &[&frame_bgl, &mat_bgl],
            push_constant_ranges: &[],
        });
        let vfloats = crate::render::FLOATS_PER_VERTEX as u64 * 4;
        // `ssr`: the main pass's SSR targets (ssr.rs) after the colour, Some(true)
        // where the material writes them; None for the other views' passes
        let make = |label: &str, blend: Option<wgpu::BlendState>, depth_write: bool, a2c: bool, ssr: Option<bool>| {
            let mut targets = vec![Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })];
            // blended: covers the reflection change behind by its alpha
            let cover = ssr == Some(false) && blend.is_some();
            match ssr {
                Some(_) if cover => targets.extend(crate::ssr::behind_targets(crate::ssr::Behind::Cover)),
                Some(w) => targets.extend(crate::ssr::extra_targets(w)),
                None => {}
            }
            let entry = match ssr {
                _ if !crate::ssr::active() => "fs_main",
                Some(true) => "fs_main_ssr",
                Some(false) if cover => "fs_main_cover",
                _ => "fs_main",
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    compilation_options: Default::default(),
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: vfloats,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![
                                0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4,
                                4 => Float32x4, 5 => Float32x4],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 80,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &wgpu::vertex_attr_array![
                                6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Float32x4,
                                10 => Float32x4],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: (EXTRA_FLOATS * 4) as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![11 => Float32x4, 12 => Float32x2],
                        },
                    ],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: entry,
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: depth_write,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: samples,
                    alpha_to_coverage_enabled: a2c && samples > 1,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
        };
        let opaque = make("track opaque", None, true, false, Some(true));
        // Alpha-to-coverage on cutouts under MSAA: leaf and grass edges get
        // sub-pixel coverage instead of a hard, shimmering discard.
        let cutout = make("track cutout", None, true, true, Some(true));
        let blended = make("track blended", Some(wgpu::BlendState::ALPHA_BLENDING), false, false, Some(false));
        let shadow_cut = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("track shadow cutout"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: vfloats,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4,
                                4 => Float32x4, 5 => Float32x4],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 80,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Float32x4,
                            10 => Float32x4],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: (EXTRA_FLOATS * 4) as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![11 => Float32x4, 12 => Float32x2],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_shadow",
                compilation_options: Default::default(),
                targets: &[],
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            // same bias as the car-style shadow pipeline (render.rs)
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        // ---- motion vectors (draw_motion) ----
        let mv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("track motion bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: samples > 1 },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let mv_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("track motion layout"),
            bind_group_layouts: &[&frame_bgl, &mat_bgl, &mv_bgl],
            push_constant_ranges: &[],
        });
        let make_mv = |rg: bool| {
            let t = |f| Some(wgpu::ColorTargetState { format: f, blend: None, write_mask: wgpu::ColorWrites::ALL });
            let targets = [t(crate::motion::FORMAT), t(crate::motion::RG_FORMAT)];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("track motion vectors"),
                layout: Some(&mv_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_motion",
                    compilation_options: Default::default(),
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: vfloats,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![
                                0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4,
                                4 => Float32x4, 5 => Float32x4],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 80,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &wgpu::vertex_attr_array![
                                6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Float32x4,
                                10 => Float32x4],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: (EXTRA_FLOATS * 4) as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![11 => Float32x4, 12 => Float32x2],
                        },
                    ],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: if rg { "fs_motion_rg" } else { "fs_motion" },
                    compilation_options: Default::default(),
                    targets: &targets[..if rg { 2 } else { 1 }],
                }),
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };
        let mv_pipes = [make_mv(false), make_mv(true)];
        let mv_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("track motion"),
            size: 160,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let anim_src = anim.as_ref().map(|a| a.meshes.iter().map(|m| m.mesh).collect()).unwrap_or_default();
        let sky_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky layout"),
            bind_group_layouts: &[&frame_bgl],
            push_constant_ranges: &[],
        });
        let make_sky = |ssr: bool| device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("track sky"),
            layout: Some(&sky_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_sky",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_sky",
                compilation_options: Default::default(),
                targets: &{
                    let mut t = vec![Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })];
                    if ssr {
                        t.extend(crate::ssr::extra_targets(false));
                    }
                    t
                },
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
            multiview: None,
            cache: None,
        });
        let sky = make_sky(true);
        let view_pipes = crate::ssr::active().then(|| [
            make("track opaque (views)", None, true, false, None),
            make("track cutout (views)", None, true, true, None),
            make("track blended (views)", Some(wgpu::BlendState::ALPHA_BLENDING), false, false, None),
            make_sky(false),
        ]);

        // the cloud shadow map: only the frame and the cloud textures (the
        // map itself is not bound while it is drawn)
        let cmap_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud map bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex_entry(10, wgpu::TextureViewDimension::D2Array),
                tex_entry(13, wgpu::TextureViewDimension::D2),
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let cmap_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud map bg"),
            layout: &cmap_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(cloud_view) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(cloud_samp) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(weather_view) },
            ],
        });
        let cmap_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cloud map layout"),
            bind_group_layouts: &[&cmap_bgl],
            push_constant_ranges: &[],
        });
        let cmap_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cloud shadow map"),
            layout: Some(&cmap_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_sky",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_cloud_map",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        // the sky panorama: the frame, the cloud textures and last frame's
        // panorama (its history), into two targets
        let pano_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky pano bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex_entry(10, wgpu::TextureViewDimension::D2Array),
                tex_entry(13, wgpu::TextureViewDimension::D2),
                tex_entry(17, wgpu::TextureViewDimension::D2),
                tex_entry(18, wgpu::TextureViewDimension::D2),
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pano_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky pano bg"),
            layout: &pano_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(cloud_view) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(cloud_samp) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(weather_view) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::TextureView(&pano.sky_hist) },
                wgpu::BindGroupEntry { binding: 18, resource: wgpu::BindingResource::TextureView(&pano.cloud_hist) },
            ],
        });
        let pano_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky pano layout"),
            bind_group_layouts: &[&pano_bgl],
            push_constant_ranges: &[],
        });
        let pano_target = Some(wgpu::ColorTargetState { format: PANO_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL });
        let pano_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky pano"),
            layout: Some(&pano_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_sky",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_sky_pano",
                compilation_options: Default::default(),
                targets: &[pano_target.clone()],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let atm_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky atmosphere"),
            layout: Some(&pano_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_sky",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_sky_atm",
                compilation_options: Default::default(),
                targets: &[pano_target],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        // ---- sun ----
        // ⚠ The base renders its shadow map from render::SUN; the lighting
        // has to use the very same direction or every surface self-shadows.
        // The solar position from the track's latitude is only used when
        // EVOVIEW_SUN_GEO is set (and then shadows are wrong until the shadow
        // pass takes its direction from here).
        let (az, el) = match std::env::var("EVOVIEW_SUN").ok().and_then(|v| {
            let p: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            (p.len() == 2).then(|| (p[0], p[1]))
        }) {
            Some(ae) => ae,
            None if std::env::var("EVOVIEW_SUN_GEO").is_ok() => {
                let lat = env.filter(|e| e.has_geo).map(|e| e.lat).unwrap_or(45.0);
                sun_direction(lat, env_f32("EVOVIEW_SUN_DAY", 160.0) as f64,
                              env_f32("EVOVIEW_SUN_HOUR", 15.5) as f64)
            }
            None => {
                let d = crate::render::SUN.normalize();
                let el = d.y.asin().to_degrees();
                let az = (d.x.atan2(-d.z).to_degrees() + 360.0) % 360.0;
                (az, el)
            }
        };
        let sun_dir = sun_vector(az, el.max(3.0));
        println!(
            "track render: {} material(s) owned, {} splat, {} texture(s) uploaded, sun az {:.0} el {:.0}, {:.2}s",
            mats.iter().filter(|m| m.is_some()).count(), n_splat, uploaded, az, el,
            t0.elapsed().as_secs_f32());
        if crate::sky::active() {
            let k = crate::sky::key_dir();
            println!("  time of day: {}; key light az {:.0} el {:.0}", crate::sky::now().describe(),
                     (k.x.atan2(-k.z).to_degrees() + 360.0) % 360.0, k.y.asin().to_degrees());
        }
        let carpet: Vec<CarpetSet> = env
            .map(|e| e.carpet.iter()
                .filter_map(|lv| CarpetSet::new(device, car, lv, &|r| materials.lookup(r)))
                .collect())
            .unwrap_or_default();
        Some(TrackRender {
            opaque,
            cutout,
            blended,
            sky,
            view_pipes,
            frame_bg,
            cmap_view,
            cmap_bg,
            cmap_pipe,
            pano,
            pano_bg,
            pano_pipe,
            atm_pipe,
            frame_buf,
            mats,
            sun_dir,
            sun_e: env_f32("EVOVIEW_SUN_E", 6.0),
            exposure: env_f32("EVOVIEW_TRACK_EXPOSURE", 1.0),
            // 9 (was 2.75): sunlit / shaded ground is ~3.4:1 in the game's own
            // track previews (Monza tarmac, Spa grass); 2.75 gave ~8.9:1 and
            // shade went near black
            // 9 matched one preview's sun / shade ratio but flattened every
            // other view (shade near sunlit, contrast 2.7 vs the game's 6);
            // 3.5 with the vivid grade lands Monza / Laguna on the previews'
            // contrast and saturation
            ambient: env_f32("EVOVIEW_TRACK_AMBIENT", 3.5),
            // ⚠ Was 0.0007/m from the camera outwards, which washed every
            // hill past ~1.5 km into milk. The game holds haze off until the
            // track's haze_start_distance and sets a per-track density; the
            // unit of that density is not known, so evoview keeps its own
            // scale and follows the track's ratio to Spa's 0.01 (strict TR20).
            haze_density: env_f32("EVOVIEW_HAZE", 0.0004)
                * env.and_then(|e| e.haze_k).map_or(1.0, |k| (k / 0.01).clamp(0.25, 4.0)),
            haze_start: env_f32("EVOVIEW_HAZE_START",
                                env.and_then(|e| e.haze_start).unwrap_or(500.0)),
            haze_base: env.map(|e| e.ground_y).unwrap_or(0.0),
            iv_origin,
            iv_elev,
            pigment,
            lit: true,
            carpet_reach: carpet.iter().map(|c| c.reach).fold(0.0f32, f32::max).max(40.0),
            carpet,
            a2c: samples > 1,
            facing: std::env::var("EVOVIEW_TRACK_FACING").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            lamp_rect,
            lamp_info,
            shadow_cut,
            shadow_bufs,
            shadow_bgs,
            anim,
            anim_key: Default::default(),
            bone_buf,
            prev_bone_buf,
            mv_pipes,
            mv_bgl,
            mv_buf,
            anim_src,
            mv_prev_t: std::cell::Cell::new(None),
            mirror_buf,
            mirror_bg,
            probe_buf,
            probe_bg,
            rear_buf,
            rear_bg,
            last_frame: std::cell::Cell::new(None),
            dry_tex,
            dry_on: std::cell::Cell::new(false),
            mip_bias: crate::taa::mip_bias(),
        })
    }

    /// Advance the animated props to `t` seconds.
    pub fn animate(&self, queue: &wgpu::Queue, t: f32) {
        if let Some(a) = self.anim.as_ref() {
            let mut c = self.anim_key.borrow_mut();
            let (keys, out) = &mut *c;
            if a.pose_into(t, keys, out) {
                queue.write_buffer(&self.bone_buf, 0, bytemuck::cast_slice(out));
            }
        }
    }

    /// Motion vectors for what moves on the track: the animated props (from
    /// this and last frame's bone sets) and the vegetation in the wind
    /// (from this and last frame's clock), drawn over motion.rs's target
    /// after its own pass. Without them a waving flag reprojected as static
    /// scenery and TAA kept a faint ghost of it. `prev_vp`: the camera
    /// motion.rs reprojected through; `reset`: no last frame (a cut).
    /// EVOVIEW_MOTION_TRACK=0 off; EVOVIEW_MOTION_WIND=<m>: vegetation within
    /// that range of the eye gets them too. Off (0) by default: every tree's
    /// vertices still run, +0.6 ms at 3440x1440 even at 30 m, for sway well
    /// under a pixel a frame; the props alone cost ~0.05 ms.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_motion(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        target: (&wgpu::TextureView, Option<&wgpu::TextureView>),
        depth: &wgpu::TextureView,
        rsize: (u32, u32),
        vp: glam::Mat4,
        prev_vp: glam::Mat4,
        reset: bool,
        meshes: &[crate::render::GpuMesh],
        count: &dyn Fn(usize, u32) -> u32,
    ) {
        static ON: std::sync::OnceLock<(bool, f32)> = std::sync::OnceLock::new();
        let (on, wind_m) = *ON.get_or_init(|| (std::env::var("EVOVIEW_MOTION_TRACK").map_or(true, |v| v != "0"),
                                               env_f32("EVOVIEW_MOTION_WIND", 0.0)));
        let wind = wind_m > 0.0;
        if !on {
            return;
        }
        let Some(f) = self.last_frame.get() else { return };
        let t = f.clouds3[0];
        if reset {
            self.mv_prev_t.set(None);
        }
        let prev_t = match self.mv_prev_t.get() {
            Some(p) => p,
            None => {
                // no last frame: it is this one (zero motion)
                enc.copy_buffer_to_buffer(&self.bone_buf, 0, &self.prev_bone_buf, 0, self.bone_buf.size());
                t
            }
        };
        let mut u = [0.0f32; 40];
        u[..16].copy_from_slice(&vp.to_cols_array());
        u[16..32].copy_from_slice(&prev_vp.to_cols_array());
        u[32..36].copy_from_slice(&[rsize.0 as f32, rsize.1 as f32, 1.0 / rsize.0.max(1) as f32, 1.0 / rsize.1.max(1) as f32]);
        u[36] = prev_t;
        u[37] = wind_m;
        queue.write_buffer(&self.mv_buf, 0, bytemuck::cast_slice(&u));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("track motion bg"),
            layout: &self.mv_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.mv_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: self.prev_bone_buf.as_entire_binding() },
            ],
        });
        let att = |v| Some(wgpu::RenderPassColorAttachment {
            view: v,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
        });
        let atts = [att(target.0), target.1.and_then(att)];
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("track motion vectors"),
                color_attachments: &atts[..if target.1.is_some() { 2 } else { 1 }],
                ..Default::default()
            });
            pass.set_pipeline(&self.mv_pipes[target.1.is_some() as usize]);
            pass.set_bind_group(0, &self.frame_bg, &[]);
            pass.set_bind_group(2, &bg, &[]);
            let mut last_group = usize::MAX;
            for (gi, m) in meshes.iter().enumerate() {
                let Some(extra) = m.extra.as_ref() else { continue };
                let anim = self.anim_src.contains(&m.src);
                let n = count(gi, m.instance_count);
                if n == 0 {
                    continue;
                }
                let mut bound = false;
                for &(start, cnt, mi) in &m.draws {
                    let Some(tm) = self.mats.get(mi).and_then(|m| m.as_ref()) else { continue };
                    if !matches!(tm.blend, Blend::Opaque | Blend::Cutout) || !(anim || (wind && tm.wind)) {
                        continue;
                    }
                    if !bound {
                        pass.set_vertex_buffer(0, m.vertices.slice(..));
                        pass.set_vertex_buffer(1, m.instances.slice(..));
                        pass.set_vertex_buffer(2, extra.slice(..));
                        pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                        bound = true;
                    }
                    if last_group != mi {
                        pass.set_bind_group(1, &tm.group, &[]);
                        last_group = mi;
                    }
                    pass.draw_indexed(start..start + cnt, 0, 0..n);
                }
            }
        }
        enc.copy_buffer_to_buffer(&self.bone_buf, 0, &self.prev_bone_buf, 0, self.bone_buf.size());
        self.mv_prev_t.set(Some(t));
    }

    pub fn owns(&self, mi: usize) -> bool {
        self.mats.get(mi).map_or(false, |m| m.is_some())
    }

    /// Draw the cutout materials into shadow `layer` with their alpha test.
    /// `inst` gives each mesh's (instance buffer, count) for that cascade.
    pub fn draw_shadow_cutouts<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        layer: usize,
        meshes: &'a [crate::render::GpuMesh],
        inst: &dyn Fn(usize) -> (&'a wgpu::Buffer, u32),
    ) {
        let mut set = false;
        for (gi, m) in meshes.iter().enumerate() {
            let Some(extra) = m.extra.as_ref() else { continue };
            if !m.draws.iter().any(|d| self.blend(d.2) == Some(Blend::Cutout)) {
                continue;
            }
            let (buf, n) = inst(gi);
            if n == 0 {
                continue;
            }
            if !set {
                pass.set_pipeline(&self.shadow_cut);
                pass.set_bind_group(0, &self.shadow_bgs[layer.min(2)], &[]);
                set = true;
            }
            pass.set_vertex_buffer(0, m.vertices.slice(..));
            pass.set_vertex_buffer(1, buf.slice(..));
            pass.set_vertex_buffer(2, extra.slice(..));
            pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
            for &(start, cnt, mi) in &m.draws {
                if self.blend(mi) != Some(Blend::Cutout) {
                    continue;
                }
                let Some(tm) = self.mats[mi].as_ref() else { continue };
                pass.set_bind_group(1, &tm.group, &[]);
                pass.draw_indexed(start..start + cnt, 0, 0..n);
            }
        }
    }

    /// Does the track draw material `mi` into the shadow map itself?
    pub fn casts_cutout(&self, mi: usize) -> bool {
        self.blend(mi) == Some(Blend::Cutout)
    }

    pub fn blend(&self, mi: usize) -> Option<Blend> {
        self.mats.get(mi).and_then(|m| m.as_ref()).map(|m| m.blend)
    }

    /// Write this frame's constants.
    /// `cam` is the base's camera uniform of this frame: its shadow matrix
    /// and parameters are reused so track and car shadows agree.
    pub fn update(&self, queue: &wgpu::Queue, view_proj: glam::Mat4, eye: glam::Vec3, lit: bool,
                  cam: &crate::render::Uniforms) {
        // Time of day (sky.rs). Until one is chosen the block is the fixed
        // calibrated sun at strength 1, so nothing below changes.
        let sky = crate::sky::block();
        let timed = crate::sky::active();
        let sun_dir = if timed { crate::sky::key_dir() } else { self.sun_dir };
        let sun_col = [1.0f32 * sky[1][0], 0.94 * sky[1][1], 0.84 * sky[1][2]];
        let sky_k = sky[1][3];
        // lamps light up with the night (curve night_gsaa_vehicle_itensity)
        let night = if timed { sky[2][3] } else { 0.0 };
        // 0.05 (a unit guess) blew Monza / Bathurst at night out to solid
        // white; 0.0004 gives a floodlit straight with the stands readable
        let lamp_k = if timed { env_f32("EVOVIEW_LAMP_K", 0.0004) } else { 0.0 };
        if let Some(m) = crate::rain::dry_map(eye.x, eye.z) {
            let n = crate::rain::DRY_DIM as u32;
            queue.write_texture(
                wgpu::ImageCopyTexture { texture: &self.dry_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &m,
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(n), rows_per_image: Some(n) },
                wgpu::Extent3d { width: n, height: n, depth_or_array_layers: 1 },
            );
            self.dry_on.set(true);
        }
        let u = FrameUniform {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            eye: [eye.x, eye.y, eye.z, 1.0],
            sun: [sun_dir.x, sun_dir.y, sun_dir.z, self.sun_e * sky[0][3]],
            sun_color: [sun_col[0], sun_col[1], sun_col[2], self.exposure],
            haze: [self.haze_density * (1.0 + 3.0 * crate::rain::rain()), env_f32("EVOVIEW_HAZE_H", 180.0), self.haze_base,
                   env_f32("EVOVIEW_SKY", 1.0) * sky_k],
            iv_origin: self.iv_origin,
            iv_elev: self.iv_elev,
            misc: [self.ambient * sky_k, if lit && self.lit { 1.0 } else { 0.0 },
                   env_f32("EVOVIEW_GRASS_FADE0", self.carpet_reach * carpet_scale(view_proj) * 0.7),
                   env_f32("EVOVIEW_GRASS_FADE1", self.carpet_reach * carpet_scale(view_proj))],
            light_vp: cam.light_vp,
            shadow: cam.shadow,
            grade: cam.grade,
            pigment: self.pigment,
            // w: emissive scale. The viewer's emissives are set by eye for the
            // daylight exposure; under the ~1000x night exposure they would
            // blow out, so they are divided back by most of it (strict SKY3).
            debug: [env_f32("EVOVIEW_TRACK_DEBUG", 0.0), if self.a2c { 1.0 } else { 0.0 },
                    self.facing,
                    if timed { crate::sky::exposure().powf(-env_f32("EVOVIEW_EMISSIVE_ADAPT", 0.8)) } else { 1.0 }],
            sky,
            lamp_grid: self.lamp_rect,
            // night IV: x its curve (10) x EVOVIEW_NIGHT_IV (unit guess)
            lamps: [self.lamp_info[0], self.lamp_info[1], lamp_k,
                    night * sky[3][3] * env_f32("EVOVIEW_NIGHT_IV", 0.01)],
            light_vp_far: cam.light_vp_far,
            shadow_far: cam.shadow_far,
            // y: EVOVIEW_CLOUD_STEPS (cloud raymarch steps, 0 = 40), + 1000 x
            // EVOVIEW_CLOUD_NOISE (1 = the old golden-ratio jitter, 2 = white
            // noise per frame: the default under DLSS)
            // w: frame number mod 4096 (LOD dither shift, cloud noise)
            // z: a golden-ratio step per frame for the cloud march's start
            // offset. It was time x 7.31, only ~0.05 a frame at 140 fps, so
            // TAA's short history never saw the full range and the dither
            // stayed on screen as a checkerboard over every cloud edge.
            haze2: {
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                [self.haze_start, env_f32("EVOVIEW_CLOUD_STEPS", 0.0) + 1000.0 * env_f32("EVOVIEW_CLOUD_NOISE",
                     if crate::dlss::wanted() { 2.0 } else { 0.0 }),
                 (n as f64 * 0.618_033_988_75).fract() as f32, (n % 4096) as f32]
            },
            // EVOVIEW_CLOUDS = coverage 0..1 (0 = clear sky)
            clouds: cam.clouds[0],
            clouds2: cam.clouds[1],
            clouds3: cam.clouds[2],
            clouds4: cam.clouds[3],
            clouds5: cam.clouds[4],
            // snapped to whole texels, so the map does not crawl as the eye moves
            cmap: {
                let t = CMAP_SPAN / CMAP_DIM as f32;
                let h = CMAP_SPAN * 0.5;
                let on = std::env::var("EVOVIEW_CLOUD_MAP").map_or(true, |v| v != "0");
                [(eye.x / t).round() * t - h, (eye.z / t).round() * t - h, CMAP_SPAN, if on { 1.0 } else { 0.0 }]
            },
            // w: the dry-line map is live
            wet: [crate::rain::wetness(), crate::rain::rain(), crate::clouds::time(), if self.dry_on.get() { 1.0 } else { 0.0 }],
            light_vp_mid: cam.light_vp_mid,
            shadow_mid: cam.shadow_mid,
            light_vp_hero: cam.light_vp_hero,
            shadow_hero: cam.shadow_hero,
            lod: [self.mip_bias, 0.0, 0.0, 0.0],
            atm: {
                let d = if timed { crate::sky::now().sun } else { self.sun_dir };
                [d.x, d.y, d.z, if self.pano.on { 1.0 } else { 0.0 }]
            },
            atm2: {
                // full replacement on the first frame and after a time-of-day
                // jump; otherwise a running average that integrates the
                // march's per-frame jitter into smooth clouds
                // (the clock runs, so the sky's generation moves every frame:
                // only a jump of the sun of over half a degree counts)
                let d = if timed { crate::sky::now().sun } else { self.sun_dir };
                let last = self.pano.sun.replace(d);
                let fresh = last.dot(d) < 0.99996 || self.pano.frames.get() == 0;
                self.pano.frames.set(self.pano.frames.get() + 1);
                [if fresh { 1.0 } else { env_f32("EVOVIEW_PANO_BLEND", 0.3) },
                 env_f32("EVOVIEW_ATM_K", 12.0) * env_f32("EVOVIEW_SKY", 1.0),
                 env_f32("EVOVIEW_PANO_STEPS", 24.0), crate::rain::rain()]
            },
            atm3: [env_f32("EVOVIEW_CLOUD_SUN", 2.5), env_f32("EVOVIEW_CLOUD_AMB", if self.pano.on { 0.6 } else { 1.0 }),
                   env_f32("EVOVIEW_CLOUD_SUN_EXT", 1.0), env_f32("EVOVIEW_CLOUD_POWDER", 0.5)],
            // ⚠ The 0.16 extinction scale is a guess (strict SKY4); seen
            // through, it gave thin, fuzzy, uniformly white puffs. Real
            // cumulus is optically thick (crisp edges, grey bases): 8x as
            // seen. The ground shadow keeps the old scale.
            // y: the share of TAA's mip bias see-through materials take (lod_bias)
            atm4: [if self.pano.on { env_f32("EVOVIEW_CLOUD_VIEW_DENSITY", 8.0) } else { 1.0 },
                   env_f32("EVOVIEW_SEETHROUGH_MIP_K", 0.0), 0.0, 0.0],
        };
        queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&u));
        self.last_frame.set(Some(u));
        for (i, vp) in [cam.light_vp, cam.light_vp_far, cam.light_vp_mid].into_iter().enumerate() {
            let mut s = u;
            s.view_proj = vp;
            // no grass fade / LOD dithering by the light's "distance"
            s.misc[2] = 1.0e9;
            s.misc[3] = 1.0e9;
            queue.write_buffer(&self.shadow_bufs[i], 0, bytemuck::bytes_of(&s));
        }
        let radius = env_f32("EVOVIEW_GRASS_FADE1", 1.0e9);
        let s = carpet_scale(view_proj);
        for c in &self.carpet {
            c.update(queue, eye, radius, s);
        }
    }

    /// Every batch of this mesh is drawn by the track pipeline.
    pub fn owns_mesh(&self, m: &crate::render::GpuMesh) -> bool {
        m.extra.is_some() && m.draws.iter().all(|d| self.owns(d.2))
    }

    /// Draw every owned batch of `phase`, one pipeline switch for the lot.
    /// `count(mesh, default)` is the culler's visible instance count (the
    /// culler compacts visible instances into the mesh's instance buffer).
    pub fn draw_phase<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        meshes: &'a [crate::render::GpuMesh],
        phase: Blend,
        count: &dyn Fn(usize, u32) -> u32,
    ) {
        self.draw_phase_view(pass, meshes, phase, count, &|_| None, 0);
    }

    /// `inst`: another view's culled instance buffer per GPU mesh (None =
    /// the mesh's own); `view`: 0 the main camera, 1 the mirror camera,
    /// 2 the reflection probe's, 3 the rear-view camera's (the extra views
    /// draw no grass carpet).
    pub fn draw_phase_view<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        meshes: &'a [crate::render::GpuMesh],
        phase: Blend,
        count: &dyn Fn(usize, u32) -> u32,
        inst: &dyn Fn(usize) -> Option<&'a wgpu::Buffer>,
        view: usize,
    ) {
        let vp = self.view_pipes.as_ref().filter(|_| view != 0);
        let pipe = match phase {
            Blend::Opaque => vp.map_or(&self.opaque, |p| &p[0]),
            Blend::Cutout => vp.map_or(&self.cutout, |p| &p[1]),
            Blend::Alpha => vp.map_or(&self.blended, |p| &p[2]),
            // tinted-transmission glass is a car feature; track glass is Alpha
            _ => return,
        };
        let mut set = false;
        let mut last_group = usize::MAX;
        for (gi, m) in meshes.iter().enumerate() {
            let Some(extra) = m.extra.as_ref() else { continue };
            let n = count(gi, m.instance_count);
            if n == 0 {
                continue;
            }
            let mut bound = false;
            for &(start, cnt, mi) in &m.draws {
                if self.blend(mi) != Some(phase) {
                    continue;
                }
                let Some(tm) = self.mats[mi].as_ref() else { continue };
                if !set {
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, self.view_bg(view), &[]);
                    set = true;
                }
                if !bound {
                    pass.set_vertex_buffer(0, m.vertices.slice(..));
                    pass.set_vertex_buffer(1, inst(gi).unwrap_or(&m.instances).slice(..));
                    pass.set_vertex_buffer(2, extra.slice(..));
                    pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                    bound = true;
                }
                if last_group != mi {
                    pass.set_bind_group(1, &tm.group, &[]);
                    last_group = mi;
                }
                pass.draw_indexed(start..start + cnt, 0, 0..n);
                crate::bench::count_draw(view, cnt as u64 / 3 * n as u64);
            }
        }
        for l in self.carpet.iter().flat_map(|c| c.levels.iter()).filter(|_| view == 0) {
            let n = l.count.get();
            if n == 0 {
                continue;
            }
            let mut bound = false;
            for &(start, cnt, mi) in &l.draws {
                if self.blend(mi) != Some(phase) {
                    continue;
                }
                let Some(tm) = self.mats.get(mi).and_then(|m| m.as_ref()) else { continue };
                if !set {
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, &self.frame_bg, &[]);
                    set = true;
                }
                if !bound {
                    pass.set_vertex_buffer(0, l.vertices.slice(..));
                    pass.set_vertex_buffer(1, l.instances.slice(..));
                    pass.set_vertex_buffer(2, l.extra.slice(..));
                    pass.set_index_buffer(l.indices.slice(..), wgpu::IndexFormat::Uint32);
                    bound = true;
                }
                pass.set_bind_group(1, &tm.group, &[]);
                pass.draw_indexed(start..start + cnt, 0, 0..n);
                crate::bench::count_draw(view, cnt as u64 / 3 * n as u64);
            }
        }
    }

    /// The sky, after the opaque pass: fills every pixel nothing covered.
    /// Draws the cloud shadow map for this frame's constants (after
    /// `update`, before anything samples the track's frame).
    pub fn cloud_map_pass(&self, enc: &mut wgpu::CommandEncoder) {
        let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cloud shadow map"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.cmap_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        p.set_pipeline(&self.cmap_pipe);
        p.set_bind_group(0, &self.cmap_bg, &[]);
        p.draw(0..3, 0..1);
    }

    /// The sky panorama for this frame's constants (after `update`, before
    /// anything samples the track's frame): drawn into the output pair, then
    /// copied to the pair every other pass reads, which is also the next
    /// frame's history.
    pub fn sky_pano_pass(&self, enc: &mut wgpu::CommandEncoder) {
        let pn = &self.pano;
        if !pn.on {
            return;
        }
        // the clear sky first: the clouds take their ambient light from it.
        // The cloud target keeps its texels (a quarter is redrawn a frame).
        let passes = [
            (&self.atm_pipe, &pn.sky_view, &pn.sky, &pn.sky_hist_tex, pn.sky_w, pn.sky_h, wgpu::LoadOp::Clear(wgpu::Color::BLACK)),
            (&self.pano_pipe, &pn.cloud_view, &pn.cloud, &pn.cloud_hist_tex, pn.w, pn.h, wgpu::LoadOp::Load),
        ];
        for (pipe, view, a, b, w, h, load) in passes {
            {
                let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sky pano"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                    })],
                    ..Default::default()
                });
                p.set_pipeline(pipe);
                p.set_bind_group(0, &self.pano_bg, &[]);
                p.draw(0..3, 0..1);
            }
            let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
            enc.copy_texture_to_texture(a.as_image_copy(), b.as_image_copy(), size);
        }
    }

    pub fn draw_sky<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        self.draw_sky_view(pass, 0);
    }

    fn view_bg(&self, view: usize) -> &wgpu::BindGroup {
        match view {
            1 => &self.mirror_bg,
            2 => &self.probe_bg,
            3 => &self.rear_bg,
            _ => &self.frame_bg,
        }
    }

    pub fn draw_sky_view<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, view: usize) {
        pass.set_pipeline(self.view_pipes.as_ref().filter(|_| view != 0).map_or(&self.sky, |p| &p[3]));
        pass.set_bind_group(0, self.view_bg(view), &[]);
        pass.draw(0..3, 0..1);
    }

    /// A mirror's camera on this frame's uniform (call after `update`);
    /// `rear`: the rear-view camera's own (drawn in the same frame).
    pub fn update_mirror(&self, queue: &wgpu::Queue, view_proj: glam::Mat4, eye: glam::Vec3, rear: bool) {
        let Some(mut u) = self.last_frame.get() else { return };
        u.view_proj = view_proj.to_cols_array_2d();
        u.inv_view_proj = view_proj.inverse().to_cols_array_2d();
        u.eye = [eye.x, eye.y, eye.z, 1.0];
        // no TAA there: TAA's sharpening bias only aliased
        u.lod = [0.0; 4];
        queue.write_buffer(if rear { &self.rear_buf } else { &self.mirror_buf }, 0, bytemuck::bytes_of(&u));
    }

    /// A reflection-probe face's camera on this frame's uniform.
    pub fn update_probe(&self, queue: &wgpu::Queue, view_proj: glam::Mat4, eye: glam::Vec3) {
        let Some(mut u) = self.last_frame.get() else { return };
        u.view_proj = view_proj.to_cols_array_2d();
        u.inv_view_proj = view_proj.inverse().to_cols_array_2d();
        u.eye = [eye.x, eye.y, eye.z, 1.0];
        // unexposed: the car shader reads it as light, and exposes it with
        // the car (HDR, tonemap() then only scales)
        u.grade[0] = 1.0;
        u.lod = [0.0; 4];
        queue.write_buffer(&self.probe_buf, 0, bytemuck::bytes_of(&u));
    }
}
