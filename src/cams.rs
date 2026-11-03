//! Track TV cameras (`containers/tv*_cameras*.scene`) for the driven car.
//!
//! Each camera is a scene actor (SceneActorData f50 -> f102 CameraData) at a
//! fixed world position. Its CamTVData (f15) gives the stretch of track it
//! covers - start / end segments (f1 / f2: pos, width, height, angle) - and
//! a field-of-view curve (f6, keyed by distance to the car: 30 deg close,
//! zoomed to a few degrees at 120 m, so the car keeps its size on screen).
//! The game blends between cameras with zones along the spline and a spring
//! (tvcamsettings); here the camera covering the car's stretch wins, with
//! hysteresis, and aims through a critically damped spring.

use glam::{Mat4, Vec3};

#[derive(Clone, Debug)]
pub struct TvCam {
    pub name: String,
    pub pos: Vec3,
    pub seg: Option<(Vec3, Vec3)>,
    pub fov: f32,
    pub fov_curve: Vec<(f32, f32)>,
    /// depth of field from the camera's .graphicsoverride (TV cams only)
    pub dof: Option<Dof>,
}

/// PostProcessingSettings.DepthOfField as a camera's .graphicsoverride sets
/// it (common_assets/camera_settings/dof_*.graphicsoverride). Field numbers
/// from PostProcessing.proto; unset floats are 0 as in the game.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dof {
    pub coc_depth_radius: f32,
    pub coc_samples: u32,
    pub aperture: f32,
    pub fg_blur: f32,
    pub bg_blur: f32,
    pub bg_dist: f32,
    pub fg_dist: f32,
    /// explicitFocusDistance ? focusDistanceOverride : None (auto focus)
    pub focus: Option<f32>,
    /// explicitFocusCenter ? focusCenter : None (the screen centre)
    pub center: Option<[f32; 2]>,
}

/// The DepthOfField a .graphicsoverride enables (overrideDepthOfField f9 and
/// DepthOfField f10 with enable f1), else None.
fn read_dof(d: &[u8]) -> Option<Dof> {
    use crate::proto::{fields, Val};
    let (mut over, mut en) = (false, false);
    let mut o = Dof::default();
    let (mut explicit_f, mut explicit_c) = (false, false);
    let (mut fd, mut fc) = (0.0f32, [0.5f32, 0.5]);
    fields(d, 0, d.len(), |f, v| match (f, v) {
        (9, Val::Varint(b)) => over = b != 0,
        (10, Val::Bytes(a, b)) => fields(d, a, b, |g, w| match (g, w) {
            (1, Val::Varint(x)) => en = x != 0,
            (2, w) => o.coc_depth_radius = w.f32(),
            (3, Val::Varint(x)) => o.coc_samples = x as u32,
            (4, w) => o.aperture = w.f32(),
            (5, w) => o.fg_blur = w.f32(),
            (6, w) => o.bg_blur = w.f32(),
            (7, w) => o.bg_dist = w.f32(),
            (8, w) => o.fg_dist = w.f32(),
            (9, Val::Varint(x)) => explicit_f = x != 0,
            (10, w) => fd = w.f32(),
            (11, Val::Varint(x)) => explicit_c = x != 0,
            (12, Val::Bytes(ca, cb)) => fields(d, ca, cb, |h, y| if let (1 | 2, Val::F32(q)) = (h, y) { fc[h as usize - 1] = q }),
            _ => {}
        }),
        _ => {}
    });
    o.focus = explicit_f.then_some(fd);
    o.center = explicit_c.then_some(fc);
    (over && en).then_some(o)
}

/// Every TV camera the track's camera containers hold.
pub fn load_tv(pkg: &std::path::Path, track: &str) -> Vec<TvCam> {
    use crate::proto::{fields, text, Val};
    let prefix = format!("content\\tracks\\{}\\", track.to_lowercase());
    let Ok(p) = crate::pkg::Package::open_filtered(pkg, &[&prefix, "content\\tracks\\common_assets\\camera_curves\\",
                                                          "content\\tracks\\common_assets\\camera_settings\\"]) else {
        return Vec::new();
    };
    let mut keys: Vec<&String> = p.by_path.keys()
        .filter(|k| k.starts_with(&prefix) && k.contains("\\containers\\") && k.contains("camera") && k.ends_with(".scene")
                && !k.contains("sequence") && !k.contains("vr_cameras"))
        .collect();
    keys.sort();
    // where each camera container is placed: its cameras are local to it
    // (Imola's TV2 set sat under the track before this)
    let mut placed: std::collections::HashMap<String, Mat4> = Default::default();
    for k in p.by_path.keys().filter(|k| k.starts_with(&prefix) && k.ends_with(".scene")) {
        let d = p.read(p.by_path[k]);
        fields(&d, 0, d.len(), |f, v| {
            let (Some((a, b)), 2) = (v.range(), f) else { return };
            let mut xf = None;
            let mut path = None;
            fields(&d, a, b, |g, x| match (g, x.range()) {
                (4, Some((ta, tb))) => xf = Some(crate::load::parse_transform(&d, ta, tb)),
                (50, Some((ia, ib))) => fields(&d, ia, ib, |h, y| {
                    if let (121, Some((ca, cb))) = (h, y.range()) {
                        fields(&d, ca, cb, |m, z| if let (1, Some(r)) = (m, z.range()) { path = Some(text(&d, r)) });
                    }
                }),
                _ => {}
            });
            if let (Some(xf), Some(path)) = (xf, path) {
                placed.insert(path.to_lowercase().replace('/', "\\"), crate::load::trs(xf.0, xf.1, xf.2));
            }
        });
    }
    let vec3 = |d: &[u8], a: usize, b: usize| -> Vec3 {
        let mut v = [0.0f32; 3];
        fields(d, a, b, |f, x| if let (1..=3, Val::F32(q)) = (f, x) { v[f as usize - 1] = q });
        Vec3::from(v)
    };
    let mut curves: std::collections::HashMap<String, Vec<(f32, f32)>> = Default::default();
    let mut dofs: std::collections::HashMap<String, Option<Dof>> = Default::default();
    let mut out = Vec::new();
    for k in keys {
        let place = placed.get(&k.replace('/', "\\")).copied().unwrap_or(Mat4::IDENTITY);
        let d = p.read(p.by_path[k]);
        fields(&d, 0, d.len(), |f, v| {
            let (Some((a, b)), 2) = (v.range(), f) else { return };
            let mut cam = TvCam { name: String::new(), pos: Vec3::ZERO, seg: None, fov: 30.0, fov_curve: Vec::new(), dof: None };
            let mut is_cam = false;
            let mut curve_path = String::new();
            let mut gso = String::new();
            fields(&d, a, b, |g, x| match g {
                1 => cam.name = x.range().map(|r| text(&d, r)).unwrap_or_default(),
                4 => if let Some((ta, tb)) = x.range() {
                    fields(&d, ta, tb, |h, y| if let (1, Some((pa, pb))) = (h, y.range()) { cam.pos = vec3(&d, pa, pb) });
                },
                50 => if let Some((ia, ib)) = x.range() {
                    fields(&d, ia, ib, |h, y| {
                        let (102, Some((ca, cb))) = (h, y.range()) else { return };
                        is_cam = true;
                        fields(&d, ca, cb, |m, z| match m {
                            14 => if let Some((ba, bb)) = z.range() {
                                fields(&d, ba, bb, |n, w| match (n, w) {
                                    (2, Val::F32(fv)) => cam.fov = fv,
                                    // CameraBaseData.graphicsSettingsOverride
                                    (21, Val::Bytes(ta, tb)) => gso = text(&d, (ta, tb)),
                                    _ => {}
                                });
                            },
                            15 => if let Some((va, vb)) = z.range() {
                                let (mut s0, mut s1) = (None, None);
                                fields(&d, va, vb, |n, w| match (n, w.range()) {
                                    (1 | 2, Some((sa, sb))) => {
                                        let mut pos = None;
                                        fields(&d, sa, sb, |o, q| if let (1, Some((qa, qb))) = (o, q.range()) { pos = Some(vec3(&d, qa, qb)) });
                                        if n == 1 { s0 = pos } else { s1 = pos }
                                    }
                                    (6, Some(r)) => curve_path = text(&d, r),
                                    _ => {}
                                });
                                if let (Some(a0), Some(a1)) = (s0, s1) {
                                    cam.seg = Some((a0, a1));
                                }
                            },
                            _ => {}
                        });
                    });
                },
                _ => {}
            });
            if !is_cam {
                return;
            }
            cam.pos = place.transform_point3(cam.pos);
            cam.seg = cam.seg.map(|(a, b)| (place.transform_point3(a), place.transform_point3(b)));
            if !curve_path.is_empty() {
                let key = curve_path.to_lowercase();
                let c = curves.entry(key.clone()).or_insert_with(|| {
                    p.by_path.get(&key).map(|e| read_keys(&p.read(*e))).unwrap_or_default()
                });
                cam.fov_curve = c.clone();
            }
            if !gso.is_empty() {
                let key = gso.to_lowercase().replace('/', "\\");
                cam.dof = *dofs.entry(key.clone()).or_insert_with(|| p.by_path.get(&key).and_then(|e| read_dof(&p.read(*e))));
            }
            out.push(cam);
        });
    }
    println!("track cameras: {} TV camera(s), {} with depth of field", out.len(), out.iter().filter(|c| c.dof.is_some()).count());
    out
}

/// The keys of a .curve (CurveData-like: repeated f1 { f1 x, f10 value }).
fn read_keys(d: &[u8]) -> Vec<(f32, f32)> {
    use crate::proto::{fields, Val};
    let mut out = Vec::new();
    fields(d, 0, d.len(), |f, v| {
        let (1, Some((a, b))) = (f, v.range()) else { return };
        let (mut x, mut y) = (0.0f32, 0.0f32);
        fields(d, a, b, |g, w| match (g, w) {
            (1, Val::F32(q)) => x = q,
            (10, Val::F32(q)) => y = q,
            _ => {}
        });
        out.push((x, y));
    });
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn eval(c: &[(f32, f32)], x: f32) -> Option<f32> {
    let (f, l) = (c.first()?, c.last()?);
    if x <= f.0 {
        return Some(f.1);
    }
    if x >= l.0 {
        return Some(l.1);
    }
    c.windows(2).find(|w| x <= w[1].0).map(|w| {
        let t = (x - w[0].0) / (w[1].0 - w[0].0).max(1e-6);
        w[0].1 + (w[1].1 - w[0].1) * t
    })
}

/// Distance from `p` to the stretch a camera covers (XZ), or to the camera.
fn cover_dist(c: &TvCam, p: Vec3) -> f32 {
    let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
    match c.seg {
        Some((a, b)) => {
            let (a, b, q) = (flat(a), flat(b), flat(p));
            let ab = b - a;
            let t = ((q - a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
            (a + ab * t).distance(q)
        }
        None => flat(c.pos).distance(flat(p)) * 1.5,
    }
}

/// The live TV director: which camera, where it looks, how wide.
pub struct TvDirector {
    pub cams: Vec<TvCam>,
    pub cur: Option<usize>,
    aim: Vec3,
    aim_v: Vec3,
    fov: f32,
}

impl TvDirector {
    pub fn new(cams: Vec<TvCam>) -> TvDirector {
        TvDirector { cams, cur: None, aim: Vec3::ZERO, aim_v: Vec3::ZERO, fov: 30.0 }
    }

    /// (view_proj, eye) looking at `target` (the car), `dt` seconds on.
    pub fn view(&mut self, target: Vec3, aspect: f32, dt: f32) -> Option<(Mat4, Vec3)> {
        if self.cams.is_empty() {
            return None;
        }
        // the covering camera wins; the current one keeps it within 20 %
        let best = (0..self.cams.len())
            .min_by(|&a, &b| cover_dist(&self.cams[a], target).partial_cmp(&cover_dist(&self.cams[b], target)).unwrap())?;
        let switch = match self.cur {
            None => true,
            Some(c) => c != best && cover_dist(&self.cams[best], target) < cover_dist(&self.cams[c], target) * 0.8 - 2.0,
        };
        if switch {
            self.cur = Some(best);
            self.aim = target;
            self.aim_v = Vec3::ZERO;
        }
        let c = &self.cams[self.cur?];
        // critically damped spring on the aim point (half-life ~0.12 s)
        let dt = dt.clamp(0.0, 0.1);
        let w = 9.0f32;
        let x = self.aim - target;
        let a = -w * w * x - 2.0 * w * self.aim_v;
        self.aim_v += a * dt;
        self.aim += self.aim_v * dt;
        let dist = c.pos.distance(self.aim);
        let want = eval(&c.fov_curve, dist).unwrap_or(c.fov).clamp(3.0, 70.0);
        self.fov = if switch { want } else { self.fov + (want - self.fov) * (dt * 4.0).min(1.0) };
        let proj = Mat4::perspective_infinite_rh(self.fov.to_radians(), aspect.max(0.01), 0.3);
        Some((proj * Mat4::look_at_rh(c.pos, self.aim, Vec3::Y), c.pos))
    }

    pub fn name(&self) -> &str {
        self.cur.map_or("", |c| self.cams[c].name.as_str())
    }

    /// The current camera's depth of field, if its override has one.
    pub fn dof(&self) -> Option<Dof> {
        self.cur.and_then(|c| self.cams[c].dof)
    }
}

/// (view_proj, eye) of one of the car's own cameras, the car at `body`
/// (its mesh-space matrix).
pub fn car_view(c: &crate::fxdata::CarCam, body: Mat4, aspect: f32) -> (Mat4, Vec3) {
    let r = c.rot;
    let rot = Mat4::from_rotation_y(r[1].to_radians()) * Mat4::from_rotation_x(r[0].to_radians())
        * Mat4::from_rotation_z(r[2].to_radians());
    let eye = body.transform_point3(Vec3::from(c.pos));
    let fwd = body.transform_vector3(rot.transform_vector3(-Vec3::Z)).normalize_or_zero();
    let up = body.transform_vector3(rot.transform_vector3(Vec3::Y)).normalize_or_zero();
    let near = if c.mode == 1 || c.mode == 4 { 0.03 } else { 0.1 };
    let proj = Mat4::perspective_infinite_rh(c.fov.max(10.0).to_radians(), aspect.max(0.01), near);
    (proj * Mat4::look_at_rh(eye, eye + fwd, up), eye)
}
