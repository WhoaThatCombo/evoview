//! Driving a car on the track (`--track <id> --drive-car <car>`).
//!
//! The car is a dynamic bicycle model on the track's own collision surfaces:
//!
//! GROUND  every static mesh with a surface_id (load.rs build_collision),
//!   bucketed on a 6 m XZ grid; each wheel probes straight down for the
//!   highest surface under it. The surface's SurfaceDefinitionData gripMod
//!   scales that wheel's grip, and its sinHeight / sinLength add the bumps
//!   the game gives kerbs and grass.
//! POWERTRAIN  the car's own data (carphys.rs): torque curve, power map,
//!   gear ratios, final drive, rolling radius, idle and limiter, through an
//!   automatic gearbox; 85 % driveline efficiency, rear-wheel drive.
//! LATERAL  front and rear axle forces from slip angle with a saturating
//!   (Pacejka-shaped) curve, peak mu x load, load transfer front/rear under
//!   braking and acceleration, friction circle on the driven axle. Below
//!   ~3 m/s the model blends to kinematic steering so it can park.
//! BODY  sits on the plane through the four contact points; pitch and roll
//!   from acceleration added on springs (visual only).
//! WALLS  a wheel whose ground jumps more than 0.6 m (or vanishes) stops the
//!   car with a bounce - crude, until the barriers' colliders are read.
//!
//! ⚠ Not the game's physics (an ODE-based multi-body sim with a brush tyre
//! model): a believable car on the real track surfaces, from the car's real
//! numbers. Tyre peak mu is a fixed 1.25 until the .tyre data is decoded
//! into it (strict DRV1).

use crate::replay::Pose;
use crate::replayscene::Slot;
use crate::track::{Collision, SurfaceDef};
use glam::{Mat4, Vec2, Vec3};
use std::collections::HashMap;

const CELL: f32 = 6.0;
const G: f32 = 9.81;

pub struct Ground {
    /// wall segments (Collision.walls) and their XZ grid
    walls: Vec<[f32; 6]>,
    wall_src: Vec<String>,
    wall_grid: HashMap<(i32, i32), Vec<u32>>,
    verts: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    surf: Vec<u16>,
    grid: HashMap<(i32, i32), Vec<u32>>,
    /// per collision surface index: grip multiplier, bump (height, length)
    grip: Vec<f32>,
    bump: Vec<(f32, f32)>,
    /// per surface: frictionRain, frictionRainGamma (SurfaceDefinitionData
    /// f18/f19; frictionRain <= 0 = none)
    rain: Vec<(f32, f32)>,
    /// per surface: SurfaceDefinitionData f20 surface_type (4 = sand)
    kind: Vec<u32>,
    /// per surface: vibrationGain, vibrationLength (f14/f15; the FFB kerb buzz)
    vib: Vec<(f32, f32)>,
    /// per surface: its id ("" = untagged)
    names: Vec<String>,
    /// per surface: has a surface_id (see probe)
    tagged: Vec<bool>,
    /// EVOVIEW_GROUND_ODE=1: probes through the ODE ray collider
    ode: Option<crate::groundode::OdeGround>,
}

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub y: f32,
    pub n: Vec3,
    pub surf: u16,
}

impl Ground {
    pub fn new(c: &Collision, defs: &[SurfaceDef]) -> Ground {
        let mut grid: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        let mut big = 0usize;
        // WALL-category surfaces are never ground, however they lean: a
        // tyre pressed into Laguna's slanted barriers found 'ground' 0.4 m
        // up, the spring crushed in one step and the car took off
        let wall_surf: Vec<bool> = c.surfaces.iter()
            .map(|s| defs.iter().find(|d| d.id.eq_ignore_ascii_case(s)).map_or(false, |d| d.category == 2))
            .collect();
        let mut wall_tris = 0usize;
        for (ti, t) in c.tris.iter().enumerate() {
            if std::env::var_os("EVOVIEW_WALLS_AS_GROUND").is_none() && c.tri_surf.get(ti).map_or(false, |&s| wall_surf.get(s as usize).copied().unwrap_or(false)) {
                wall_tris += 1;
                continue;
            }
            let p = t.map(|i| c.verts[i as usize]);
            // walls are not ground: skip near-vertical triangles
            let e1 = Vec3::from(p[1]) - Vec3::from(p[0]);
            let e2 = Vec3::from(p[2]) - Vec3::from(p[0]);
            let n = e1.cross(e2);
            if n.length_squared() < 1e-12 || (n.y.abs() / n.length()) < 0.35 {
                continue;
            }
            let (x0, x1) = (p[0][0].min(p[1][0]).min(p[2][0]), p[0][0].max(p[1][0]).max(p[2][0]));
            let (z0, z1) = (p[0][2].min(p[1][2]).min(p[2][2]), p[0][2].max(p[1][2]).max(p[2][2]));
            let c = |v: f32| (v / CELL).floor() as i32;
            if (c(x1) - c(x0)) * (c(z1) - c(z0)) > 4096 {
                big += 1;
                continue;
            }
            for gz in c(z0)..=c(z1) {
                for gx in c(x0)..=c(x1) {
                    grid.entry((gx, gz)).or_default().push(ti as u32);
                }
            }
        }
        let def = |id: &str| defs.iter().find(|d| d.id.eq_ignore_ascii_case(id));
        // untagged ground (no surface_id: terrain tiles, verges) grips like
        // grass (strict DRV2)
        let grip = c.surfaces.iter().map(|s| {
            if s.is_empty() {
                return 0.7;
            }
            def(s).map_or(1.0, |d| if d.grip > 0.0 { d.grip } else { 1.0 })
        }).collect();
        let rain = c.surfaces.iter().map(|s| def(s).map_or((-1.0, 1.0), |d| (d.friction_rain, d.friction_rain_gamma))).collect();
        let bump = c.surfaces.iter().map(|s| def(s).map_or((0.0, 1.0), |d| (d.sin_height, d.sin_length.max(0.05)))).collect();
        let kind = c.surfaces.iter().map(|s| def(s).map_or(0, |d| d.kind)).collect();
        let vib = c.surfaces.iter().map(|s| def(s).map_or((0.0, 0.0), |d| (d.vibration_gain, d.vibration_length))).collect();
        if std::env::var_os("EVOVIEW_SURF_LIST").is_some() {
            let grip_v: &Vec<f32> = &grip;
            for (i, s) in c.surfaces.iter().enumerate() {
                let n = c.tri_surf.iter().filter(|t| **t as usize == i).count();
                println!("SURF {i} {:?} grip {:.2} tris {n} def {:?}", s, grip_v[i], def(s).map(|d| (d.category, d.kind, d.grip, d.sin_height, d.sin_length, d.vibration_gain, d.vibration_length, d.friction_rain, d.friction_rain_gamma)));
            }
        }
        if wall_tris > 0 {
            println!("ground: {wall_tris} wall-surface triangle(s) are not ground (EVOVIEW_WALLS_AS_GROUND=1 keeps them)");
        }
        if big > 0 {
            println!("ground: {big} triangle(s) over 4096 grid cells left out");
        }
        // ⚠ Untagged static meshes (no surface_id) are what the track draws,
        // not what the game collides with: its walls are WALL-category
        // surfaces (physics-only meshes, mostly) and its solid props. The
        // untagged ones also carry helpers nobody sees - Monza's
        // irradiance_base.mesh lies 12 cm over the main straight - and
        // their faces made walls of their own. Where the track has
        // WALL-category surfaces, untagged faces are not walls.
        // EVOVIEW_UNTAGGED_WALLS=1: they are again.
        let has_walls = c.surfaces.iter().any(|s| def(s).map_or(false, |d| d.category == 2));
        let untagged_walls = !has_walls || std::env::var_os("EVOVIEW_UNTAGGED_WALLS").is_some();
        let mut dropped = 0usize;
        let mut wall_grid: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        // EVOVIEW_WALL_DUMP="x,z,r": every wall segment within r m, kept or not
        let dump: Option<(f32, f32, f32)> = std::env::var("EVOVIEW_WALL_DUMP").ok().and_then(|v| {
            let p: Vec<f32> = v.split(',').filter_map(|a| a.trim().parse().ok()).collect();
            (p.len() == 3).then(|| (p[0], p[1], p[2]))
        });
        for (wi, w) in c.walls.iter().enumerate() {
            // distance boards (Laguna's distanz_N) stand on the verge as
            // walls no taller than a tyre: cars brushing the edge of the
            // track stopped dead on them at 140 km/h
            // ... and every other track's: dist_100_1, dist50_1, mp-Dist150
            let board = c.wall_src.get(wi).map_or(false, |s| {
                let f = s.rsplit(['/', '\\']).next().unwrap_or("").to_ascii_lowercase();
                f.starts_with("dist") || f.contains("-dist") || f.contains("_dist")
            });
            // ... and every ColliderCategoryObject surface (category 3:
            // poly, cones, plastic poles and cylinders, mobile barriers,
            // distance markers - things a car knocks over): a polystyrene
            // arrow on Fuji's hairpin apex stopped the M4 dead at 65 km/h
            // every lap. EVOVIEW_OBJECT_WALLS=1: walls again.
            let object_walls = std::env::var_os("EVOVIEW_OBJECT_WALLS").is_some();
            let drop = board || c.wall_surf.get(wi).map_or(false, |&s| c.surfaces.get(s as usize).map_or(false, |n| {
                (n.is_empty() && !untagged_walls) || n == "distance_markers"
                    || (!object_walls && def(n).map_or(false, |d| d.category == 3))
            }));
            if let Some((x, z, r)) = dump {
                let m = ((w[0] + w[2]) * 0.5 - x).hypot((w[1] + w[3]) * 0.5 - z);
                if m < r {
                    println!("WALLSEG {} {:.3} {:.3} {:.3} {:.3} {:.2} {:.2} | {}", if drop { "dropped" } else { "kept" }, w[0], w[1], w[2], w[3], w[4], w[5], c.wall_src[wi]);
                }
            }
            if drop {
                dropped += 1;
                continue;
            }
            let cc = |v: f32| (v / CELL).floor() as i32;
            let (x0, x1) = (cc(w[0].min(w[2])), cc(w[0].max(w[2])));
            let (z0, z1) = (cc(w[1].min(w[3])), cc(w[1].max(w[3])));
            if (x1 - x0 + 1) * (z1 - z0 + 1) > 400 {
                continue;
            }
            for gz in z0..=z1 {
                for gx in x0..=x1 {
                    wall_grid.entry((gx, gz)).or_default().push(wi as u32);
                }
            }
        }
        if dropped > 0 {
            println!("walls: {dropped} of {} from untagged meshes left out (EVOVIEW_UNTAGGED_WALLS=1 keeps them)", c.walls.len());
        }
        // EVOVIEW_GROUND_DUMP=<file>: the drivable ground for the offline
        // placement audit (tools/place_audit.py): u32 vertex count, u32
        // triangle count, then f32 x3 vertices and u32 x3 triangles
        if let Ok(p) = std::env::var("EVOVIEW_GROUND_DUMP") {
            let mut b: Vec<u8> = Vec::new();
            b.extend_from_slice(&(c.verts.len() as u32).to_le_bytes());
            b.extend_from_slice(&(c.tris.len() as u32).to_le_bytes());
            b.extend_from_slice(bytemuck::cast_slice(&c.verts));
            b.extend_from_slice(bytemuck::cast_slice(&c.tris));
            let _ = std::fs::write(&p, b);
        }
        let tagged = if std::env::var_os("EVOVIEW_UNTAGGED_GROUND").is_some() { vec![false; c.surfaces.len()] } else { c.surfaces.iter().map(|s| !s.is_empty()).collect() };
        // the same triangles the grid keeps (no walls, nothing steep)
        let ode = crate::groundode::OdeGround::enabled().then(|| {
            let keep = |ti: usize| {
                if std::env::var_os("EVOVIEW_WALLS_AS_GROUND").is_none() && c.tri_surf.get(ti).map_or(false, |&s| wall_surf.get(s as usize).copied().unwrap_or(false)) {
                    return false;
                }
                let p = c.tris[ti].map(|i| Vec3::from(c.verts[i as usize]));
                let n = (p[1] - p[0]).cross(p[2] - p[0]);
                n.length_squared() >= 1e-12 && n.y.abs() / n.length() >= 0.35
            };
            let tg = |ti: usize| c.tri_surf.get(ti).map_or(false, |&s| tagged.get(s as usize).copied().unwrap_or(false));
            crate::groundode::OdeGround::build(&c.verts, &c.tris, keep, tg)
        });
        Ground { walls: c.walls.clone(), wall_src: c.wall_src.clone(), wall_grid, verts: c.verts.clone(), tris: c.tris.clone(),
                 surf: c.tri_surf.clone(), grid, grip, bump, rain, kind, vib, names: c.surfaces.clone(), tagged, ode }
    }

    /// Walls near `p` (XZ) whose height range overlaps [y0, y1]: each as
    /// (closest point on it in XZ, distance).
    pub fn walls_near(&self, p: Vec3, r: f32, y0: f32, y1: f32, out: &mut Vec<(Vec3, f32, u32)>) {
        out.clear();
        let cc = |v: f32| (v / CELL).floor() as i32;
        let mut seen: Vec<u32> = Vec::new();
        for gz in cc(p.z - r)..=cc(p.z + r) {
            for gx in cc(p.x - r)..=cc(p.x + r) {
                let Some(list) = self.wall_grid.get(&(gx, gz)) else { continue };
                for &wi in list {
                    if seen.contains(&wi) {
                        continue;
                    }
                    seen.push(wi);
                    let w = self.walls[wi as usize];
                    if w[5] < y0 || w[4] > y1 {
                        continue;
                    }
                    let (a, b) = (Vec3::new(w[0], 0.0, w[1]), Vec3::new(w[2], 0.0, w[3]));
                    let q = Vec3::new(p.x, 0.0, p.z);
                    let ab = b - a;
                    let t = ((q - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
                    let c = a + ab * t;
                    let d = q.distance(c);
                    if d < r {
                        out.push((c, d, wi));
                    }
                }
            }
        }
    }

    /// Whether a wall overlapping [y0, y1] crosses the XZ segment a-b.
    pub fn wall_across(&self, a: Vec3, b: Vec3, y0: f32, y1: f32) -> bool {
        let cc = |v: f32| (v / CELL).floor() as i32;
        let cross = |o: (f32, f32), p: (f32, f32), q: (f32, f32)| (p.0 - o.0) * (q.1 - o.1) - (p.1 - o.1) * (q.0 - o.0);
        let (p0, p1) = ((a.x, a.z), (b.x, b.z));
        for gz in cc(a.z.min(b.z))..=cc(a.z.max(b.z)) {
            for gx in cc(a.x.min(b.x))..=cc(a.x.max(b.x)) {
                let Some(list) = self.wall_grid.get(&(gx, gz)) else { continue };
                for &wi in list {
                    let w = self.walls[wi as usize];
                    if w[5] < y0 || w[4] > y1 {
                        continue;
                    }
                    let (q0, q1) = ((w[0], w[1]), (w[2], w[3]));
                    if cross(p0, p1, q0) * cross(p0, p1, q1) < 0.0 && cross(q0, q1, p0) * cross(q0, q1, p1) < 0.0 {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// What the nearest wall within `r` of `p` came from (for the hit log).
    pub fn wall_name_near(&self, p: Vec3, r: f32, y0: f32, y1: f32) -> Option<(&str, [f32; 6])> {
        let cc = |v: f32| (v / CELL).floor() as i32;
        let mut best: Option<(f32, usize)> = None;
        for gz in cc(p.z - r)..=cc(p.z + r) {
            for gx in cc(p.x - r)..=cc(p.x + r) {
                let Some(list) = self.wall_grid.get(&(gx, gz)) else { continue };
                for &wi in list {
                    let w = self.walls[wi as usize];
                    if w[5] < y0 || w[4] > y1 {
                        continue;
                    }
                    let (a, b) = (Vec3::new(w[0], 0.0, w[1]), Vec3::new(w[2], 0.0, w[3]));
                    let q = Vec3::new(p.x, 0.0, p.z);
                    let ab = b - a;
                    let t = ((q - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
                    let d = q.distance(a + ab * t);
                    if d < r && best.map_or(true, |(bd, _)| d < bd) {
                        best = Some((d, wi as usize));
                    }
                }
            }
        }
        best.map(|(_, i)| (self.wall_src.get(i).map_or("?", |s| s.as_str()), self.walls[i]))
    }

    /// Wall segment `i`: its source and [x0, z0, x1, z1, y0, y1].
    pub fn wall(&self, i: u32) -> (&str, [f32; 6]) {
        (self.wall_src.get(i as usize).map_or("?", |s| s.as_str()), self.walls[i as usize])
    }

    pub fn surf_name(&self, s: u16) -> &str {
        match self.names.get(s as usize).map(|s| s.as_str()) {
            Some("") => "untagged",
            Some(n) => n,
            None => "?",
        }
    }

    /// The highest surface at (x, z) no higher than `top`.
    /// ⚠ A tagged surface (one with a surface_id: what the game drives on)
    /// wins over untagged geometry up to 0.5 m above it. The untagged meshes
    /// are the drawn track - its asphalt a few mm over the physics one, and
    /// helpers like Monza's irradiance_base 12 cm over the main straight,
    /// which threw the car 13 cm into the air at 236 km/h. Untagged ground
    /// still counts where nothing tagged is near (terrain beyond the run-off).
    /// EVOVIEW_UNTAGGED_GROUND=1: the highest of either, as before.
    pub fn probe(&self, x: f32, z: f32, top: f32) -> Option<Hit> {
        if let Some(o) = self.ode.as_ref() {
            return o.probe(x, z, top).map(|(y, n, ti)| Hit { y, n, surf: self.surf[ti as usize] });
        }
        let cell = ((x / CELL).floor() as i32, (z / CELL).floor() as i32);
        let list = self.grid.get(&cell)?;
        let (mut best, mut best_t): (Option<Hit>, Option<Hit>) = (None, None);
        for &ti in list {
            let t = self.tris[ti as usize];
            let a = Vec3::from(self.verts[t[0] as usize]);
            let b = Vec3::from(self.verts[t[1] as usize]);
            let c = Vec3::from(self.verts[t[2] as usize]);
            // barycentrics in XZ
            let d = (b.z - c.z) * (a.x - c.x) + (c.x - b.x) * (a.z - c.z);
            if d.abs() < 1e-9 {
                continue;
            }
            let l1 = ((b.z - c.z) * (x - c.x) + (c.x - b.x) * (z - c.z)) / d;
            let l2 = ((c.z - a.z) * (x - c.x) + (a.x - c.x) * (z - c.z)) / d;
            let l3 = 1.0 - l1 - l2;
            if l1 < -1e-4 || l2 < -1e-4 || l3 < -1e-4 {
                continue;
            }
            let y = l1 * a.y + l2 * b.y + l3 * c.y;
            if y > top {
                continue;
            }
            let s = self.surf[ti as usize];
            let slot = if self.tagged.get(s as usize).copied().unwrap_or(false) { &mut best_t } else { &mut best };
            if slot.map_or(true, |h| y > h.y) {
                let mut n = (b - a).cross(c - a).normalize_or_zero();
                if n.y < 0.0 {
                    n = -n;
                }
                *slot = Some(Hit { y, n, surf: s });
            }
        }
        match (best_t, best) {
            (Some(t), Some(u)) if u.y > t.y + 0.5 => Some(u),
            (Some(t), _) => Some(t),
            (None, u) => u,
        }
    }

    pub fn grip(&self, s: u16) -> f32 {
        self.grip.get(s as usize).copied().unwrap_or(1.0)
    }

    /// SurfaceDefinitionData surface_type: 0 asphalt, 1 extra turf, 2 kerb,
    /// 3 grass, 4 sand, 5 flat kerb.
    pub fn kind(&self, s: u16) -> u32 {
        self.kind.get(s as usize).copied().unwrap_or(0)
    }

    /// The surface's FFB kerb buzz: (vibrationGain, vibrationLength m).
    pub fn vibration(&self, s: u16) -> (f32, f32) {
        self.vib.get(s as usize).copied().unwrap_or((0.0, 0.0))
    }

    /// The surface's grip under `wet` (0..1) water: toward its frictionRain
    /// by wet^frictionRainGamma. ⚠ The consumer of these two fields was not
    /// found (the tyre step takes the contact's grip ready-made); this blend
    /// is a guess that fits the data (kerbs 0.95 -> 0.6 at gamma 0.1:
    /// slippery as soon as they are damp; tarmac has none).
    pub fn grip_wet(&self, s: u16, wet: f32) -> f32 {
        let g = self.grip(s);
        match self.rain.get(s as usize) {
            Some(&(fr, gamma)) if fr > 0.0 && wet > 0.0 => g + (fr - g) * wet.min(1.0).powf(gamma.max(0.0)),
            _ => g,
        }
    }

    pub fn bump(&self, s: u16, x: f32, z: f32) -> f32 {
        let (h, l) = self.bump.get(s as usize).copied().unwrap_or((0.0, 1.0));
        if h <= 0.0 {
            return 0.0;
        }
        let k = std::f32::consts::TAU / l;
        h * 0.5 * ((x * k).sin() + (z * k * 1.37).sin()) * 0.5
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Controls {
    pub throttle: f32,
    pub brake: f32,
    /// -1 right .. +1 left
    pub steer: f32,
    pub handbrake: bool,
    /// a wheel: `steer` is the steering-wheel position itself (no keyboard
    /// rate limit or speed-scaled lock)
    pub analog: bool,
    /// clutch pedal 0 released .. 1 pressed; None = the car's own clutch
    /// (auto for launches) - the rigid-body car only
    pub clutch: Option<f32>,
    /// paddles, true on the frame they are pressed
    pub shift_up: bool,
    pub shift_down: bool,
    /// H-pattern lever: Some(gear), 0 neutral, -1 reverse
    pub h_gear: Option<i32>,
    /// starter / ignition, true on the frame it is pressed
    pub starter: bool,
    /// the autopilot drives: the box shifts itself (as for the game's AI),
    /// whatever the player's auto-gear assist says
    pub auto_box: bool,
    /// a real wheel's angle, steering-wheel degrees (+ left): the
    /// rigid-body car turns it through the car's ratio, clamped at its lock
    pub wheel_deg: Option<f32>,
    /// an analog handbrake 0..1 (overrides `handbrake`)
    pub handbrake_v: Option<f32>,
}

/// See TrackDrive::rig_feed.
#[derive(Clone, Copy, Debug)]
pub struct RigFeed {
    pub speed: f32,
    pub rpm: f32,
    pub gear: i32,
    pub throttle: f32,
    pub brake: f32,
    pub steer: f32,
    pub a_long: f32,
    pub a_lat: f32,
}

/// What the car's sound (carsound.rs) is fed, from the same step that
/// moves the car - the game's own audio inputs, at their names.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoundState {
    /// engine torque at the crank, N m (the car's torque curve x throttle)
    pub torque: f32,
    /// turbo boost, bar (carphys Turbo::boost at this rpm and throttle)
    pub boost: f32,
    /// slip angles, radians, front / rear
    pub slip_f: f32,
    pub slip_r: f32,
    /// driven-axle longitudinal use 0..1 (1 = the tyres are at their limit:
    /// wheelspin under power, lock-up under braking)
    pub spin: f32,
    /// surface grip under the front / rear axle (1 asphalt .. ~0.6 grass)
    pub grip_f: f32,
    pub grip_r: f32,
    /// engine on the rev limiter
    pub limiter: bool,
    /// tyre load front 0..1 of a nominal 1 g share
    pub load: f32,
}

pub struct TrackDrive {
    /// shared with the AI cars (race.rs): the collision grid is big
    pub ground: std::sync::Arc<Ground>,
    pub slot: Slot,
    phys: Option<crate::carphys::Physics>,
    /// the front tyres' aligning moment (sctm.rs), N m, both tyres - what
    /// the steering rack feels (force feedback)
    pub mz_front: f32,
    // state
    pub pos: Vec3,
    pub yaw: f32,
    pub vx: f32,
    pub vy: f32,
    r: f32,
    steer: f32,
    thr: f32,
    brk: f32,
    pub gear: i32,
    pub rpm: f32,
    spin: f32,
    pitch: f32,
    pitch_v: f32,
    roll: f32,
    roll_v: f32,
    up: Vec3,
    wheels: [Vec3; 4],
    ax: f32,
    ay: f32,
    /// smoothed longitudinal acceleration, for load transfer
    ax_lt: f32,
    // constants
    mass: f32,
    pub a: f32,
    pub b: f32,
    iz: f32,
    h: f32,
    radius: f32,
    gears: Vec<f32>,
    reverse: f32,
    final_drive: f32,
    idle: f32,
    limiter: f32,
    pub max_steer: f32,
    pub mu: f32,
    tyres: [Option<crate::carphys::TyreModel>; 2],
    /// share of the drive at the front axle: 0 RWD, 1 FWD, 0.5 FourWD,
    /// 0.4 AWD, 0.6 AWDF (the AWD splits are typical, not read)
    pub front_share: f32,
    /// traction control on (T toggles in the viewer)
    pub tc: bool,
    /// the sound inputs of the last step
    pub snd: SoundState,
    /// EVOVIEW_RIGCAR=1: the per-wheel rigid-body car (rigcar.rs) drives
    /// instead, with its step-time accumulator
    pub rig: Option<(crate::rigcar::RigCar, f32)>,
}

impl TrackDrive {
    pub fn new(ground: Ground, slot: Slot, phys: Option<crate::carphys::Physics>) -> TrackDrive {
        let rig = std::env::var_os("EVOVIEW_RIGCAR").is_some();
        Self::with_model(std::sync::Arc::new(ground), slot, phys, rig, true)
    }

    /// `rig`: the per-wheel rigid-body car (else the simple model);
    /// `player`: the player's cockpit aids and the logs (an AI car: the
    /// car's own aid defaults, quiet).
    pub fn with_model(ground: std::sync::Arc<Ground>, slot: Slot, phys: Option<crate::carphys::Physics>, rig: bool, player: bool) -> TrackDrive {
        let p = phys.as_ref();
        let mass = p.map_or(1300.0, |p| p.mass).max(400.0);
        let wb = p.map(|p| p.wheelbase).filter(|w| *w > 1.0).unwrap_or(2.6);
        let cgf = p.map_or(0.5, |p| p.cg_front).clamp(0.3, 0.7);
        let gears = p.map(|p| p.gears.clone()).filter(|g| !g.is_empty())
            .unwrap_or_else(|| vec![3.2, 2.1, 1.5, 1.2, 1.0, 0.85]);
        let radius = p.and_then(|p| p.tyre_radius).unwrap_or(slot.radius).clamp(0.2, 0.5);
        let lock = p.map_or(30.0, |p| {
            let w = p.wheel_lock_deg();
            if w > 1.0 { w } else { 30.0 }
        });
        if player && p.map_or(true, |p| p.tyres.iter().any(|t| t.is_none())) {
            crate::strict::note("DRV1", "track drive", "no .tyre model for an axle; peak mu 1.25 there");
        }
        if player {
            crate::strict::note("DRV2", "track drive", "ground without a surface_id grips at 0.7");
        }
        let mut td = TrackDrive {
            ground,
            slot,
            pos: Vec3::ZERO,
            yaw: 0.0,
            vx: 0.0,
            vy: 0.0,
            r: 0.0,
            steer: 0.0,
            thr: 0.0,
            brk: 0.0,
            gear: 1,
            rpm: 1000.0,
            spin: 0.0,
            pitch: 0.0,
            pitch_v: 0.0,
            roll: 0.0,
            roll_v: 0.0,
            up: Vec3::Y,
            wheels: [Vec3::ZERO; 4],
            ax: 0.0,
            ay: 0.0,
            ax_lt: 0.0,
            mass,
            a: wb * (1.0 - cgf),
            b: wb * cgf,
            iz: mass * wb * wb * 0.28,
            h: 0.48,
            radius,
            reverse: p.map_or(3.2, |p| if p.reverse != 0.0 { p.reverse.abs() } else { 3.2 }),
            final_drive: p.map_or(3.9, |p| if p.final_drive > 0.0 { p.final_drive } else { 3.9 }),
            idle: p.map_or(900.0, |p| p.idle_rpm.max(500.0)),
            limiter: p.map_or(7000.0, |p| p.limiter_rpm.max(3000.0)),
            gears,
            max_steer: lock.to_radians(),
            mu: 1.25,
            tyres: p.map_or([None, None], |p| p.tyres),
            front_share: match p.map_or(0, |p| p.traction) {
                1 => 1.0,
                2 => 0.5,
                3 => 0.4,
                4 => 0.6,
                _ => 0.0,
            },
            tc: std::env::var("EVOVIEW_TC").map_or(true, |v| v != "0"),
            snd: SoundState::default(),
            mz_front: 0.0,
            rig: None,
            phys,
        };
        if rig {
            if let Some(p) = td.phys.as_ref().filter(|p| p.tyres.iter().all(|t| t.and_then(|t| t.sctm).is_some())) {
                let mut rc = crate::rigcar::RigCar::new(p, td.slot.centres, td.h, td.max_steer, td.front_share, [0.0; 3]);
                if player {
                    println!("rigcar: per-wheel rigid-body physics on");
                    // the last cockpit values for this car, then test overrides
                    crate::aids::startup(&mut rc, p.user_setup.is_empty());
                }
                td.rig = Some((rc, 0.0));
            } else if player {
                println!("rigcar: needs .tyre data for both axles; staying on the simple model");
            }
        }
        td
    }

    /// The lap sim's aero (front lift, rear lift, drag per (m/s)^2) for the
    /// rigid-body car.
    pub fn set_aero(&mut self, k: [f32; 3]) {
        if let Some((rc, _)) = self.rig.as_mut() {
            rc.aero = std::env::var("EVOVIEW_RIG_AERO").ok().and_then(|v| v.parse::<f32>().ok()).map_or(k, |m| k.map(|x| x * m));
            // EVOVIEW_RIG_AERO_FRONT=0..1: diagnostic downforce balance
            if let Some(f) = std::env::var("EVOVIEW_RIG_AERO_FRONT").ok().and_then(|v| v.parse::<f32>().ok()) {
                let t = rc.aero[0] + rc.aero[1];
                rc.aero[0] = t * f;
                rc.aero[1] = t * (1.0 - f);
            }
            println!("rigcar: aero front {:.3} rear {:.3} drag {:.3} N/(m/s)^2 = {:.0} N downforce at 200 km/h", rc.aero[0], rc.aero[1], rc.aero[2], (rc.aero[0] + rc.aero[1]) * 3086.4);
        }
    }

    /// Put the car at `at` (any height: it is dropped onto the surface below
    /// a point 50 m above), facing `yaw`, at rest.
    pub fn place(&mut self, at: Vec3, yaw: f32) -> bool {
        self.place_from(at, yaw, 50.0)
    }

    /// Another player's car (mp.rs): placed where its last packet put it,
    /// carried on by its velocity for up to 0.3 s since, set on the ground.
    pub fn set_net(&mut self, s: &crate::mp::CarState, age: f32, dt: f32) {
        let at = s.pos + s.vel * age.min(0.3);
        self.pos = at;
        self.yaw = -s.ypr.x;
        let (f, l) = (self.forward(), self.left());
        self.vx = s.vel.dot(f);
        self.vy = s.vel.dot(l);
        self.rpm = s.rpm as f32;
        self.gear = s.gear as i32 - 1;
        self.thr = s.throttle;
        self.brk = s.brake;
        self.steer = -s.steer * self.max_steer;
        self.spin += self.vx / self.radius.max(0.1) * dt;
        self.settle(0.0);
        // EVOVIEW_MP_HLOG=1: how high the game's position is over the road
        // (what the sent position must match, or our car sinks in theirs)
        if std::env::var_os("EVOVIEW_MP_HLOG").is_some() && s.vel.length() < 30.0 {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) % 97 == 0 {
                println!("HLOG dy {:+.3} speed {:.1}", at.y - self.pos.y, s.vel.length());
            }
        }
        // a car in the air or above the road (a jump, a bad probe): its own height
        if (self.pos.y - at.y).abs() > 1.0 {
            self.pos.y = at.y;
        }
    }

    /// What mp.rs sends for this car (the game's conventions).
    pub fn net_state(&self) -> crate::mp::CarState {
        let f = self.rig_feed();
        // the game sends the car's centre of mass, not its ground point
        // (players' cars measured 0.35-0.55 m over the road): ours sent
        // at road level stood half sunk into the track in their game
        let cg = self.rig.as_ref().map_or(0.42, |(rc, _)| rc.cg_height().clamp(0.2, 0.8));
        crate::mp::CarState {
            pos: self.pos + Vec3::Y * cg,
            ypr: Vec3::new(-self.yaw, self.pitch, -self.roll),
            vel: self.vel(),
            rpm: self.rpm.max(0.0) as u16,
            gear: (self.gear + 1).clamp(0, 255) as u8,
            throttle: f.throttle,
            brake: f.brake,
            steer: f.steer,
            wheel: self.vx / self.radius.max(0.1),
        }
    }

    /// After `place`: rolling straight ahead at `v` m/s (a rolling start).
    pub fn roll(&mut self, v: f32) {
        self.vx = v;
        if let Some((rc, _)) = self.rig.as_mut() {
            rc.roll(v);
        }
        self.sync_rig();
    }

    /// `place`, probing from only `above` m over `at`: a pit box or garage
    /// is under a roof the 50 m probe would land on.
    pub fn place_from(&mut self, at: Vec3, yaw: f32, above: f32) -> bool {
        let Some(h) = self.ground.probe(at.x, at.z, at.y + above) else { return false };
        self.pos = Vec3::new(at.x, h.y, at.z);
        self.yaw = yaw;
        self.vx = 0.0;
        self.vy = 0.0;
        self.r = 0.0;
        self.gear = 1;
        self.settle(0.0);
        if let Some((rc, acc)) = self.rig.as_mut() {
            rc.place(self.pos, yaw);
            *acc = 0.0;
        }
        self.sync_rig();
        true
    }

    /// Copy the rigid-body car's state onto the fields the viewer reads.
    fn sync_rig(&mut self) {
        let Some((rc, acc)) = self.rig.as_mut() else { return };
        let mz_mean = rc.mz_mean();
        let (rc, acc) = (&*rc, &*acc);
        let f = rc.forward();
        let up = rc.up();
        // the physics runs in whole 3 ms steps; the frame lands between two.
        // Carry the car on by the leftover time so it does not jump 0 or
        // 15 cm (at 180 km/h) from frame to frame - a jitter that shows as a
        // double image at speed
        let ahead = rc.v * *acc;
        self.pos = rc.origin() + ahead;
        self.yaw = f.x.atan2(f.z);
        self.up = up;
        self.pitch = 0.0;
        self.roll = 0.0;
        self.pitch_v = 0.0;
        self.roll_v = 0.0;
        self.vx = rc.v.dot(f);
        self.vy = rc.v.dot(rc.left());
        self.r = rc.w.dot(up);
        self.ax_lt = 0.0;
        self.rpm = rc.rpm;
        self.gear = rc.gear;
        self.steer = rc.steer;
        self.thr = rc.thr;
        self.brk = rc.brk;
        self.mz_front = mz_mean;
        self.spin = rc.c.iter().map(|c| c.angle).sum::<f32>() * 0.25;
        for k in 0..4 {
            // the rig's hubs are wheel centres; settle()'s are too
            self.wheels[k] = rc.hub(k) + ahead;
            // the drawn tyre stands on the road: the physics squashes the
            // tyre (and a mod's mesh can be bigger than its data, RX-8 drift
            // 0.302 vs 0.29 m), the mesh does not
            let c = &rc.c[k];
            if c.on_ground && c.gap < self.slot.radius {
                self.wheels[k] += c.gn * (self.slot.radius - c.gap);
            }
        }
        let s = &mut self.snd;
        s.torque = rc.torque.max(0.0);
        s.limiter = rc.limiter_hit;
        s.slip_f = (rc.c[0].alpha + rc.c[1].alpha) * 0.5;
        s.slip_r = (rc.c[2].alpha + rc.c[3].alpha) * 0.5;
        s.spin = rc.c.iter().map(|c| (c.sr.abs() / 0.15).min(1.0)).fold(0.0, f32::max);
        s.grip_f = rc.c[0].grip.min(rc.c[1].grip);
        s.grip_r = rc.c[2].grip.min(rc.c[3].grip);
        s.load = (rc.c[0].fz + rc.c[1].fz) / (self.mass * G * 0.5).max(1.0);
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos())
    }

    /// mesh +X: the car's left
    pub fn left(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin())
    }

    fn ratio(&self) -> f32 {
        match self.gear {
            0 => 0.0,
            g if g < 0 => -self.reverse * self.final_drive,
            g => self.gears[(g as usize - 1).min(self.gears.len() - 1)] * self.final_drive,
        }
    }

    /// Typical peak lateral friction of this car's tyres at its static load.
    /// The tyres' water clearing (TyresGrooveData grooveFactor, the lower axle's)
    pub fn groove(&self) -> f32 {
        let g = |i: usize| self.tyres[i].and_then(|t| t.sctm).map_or(0.0, |s| s.groove);
        g(0).min(g(1)).clamp(0.0, 1.0)
    }

    pub fn grip_estimate(&self) -> f32 {
        let wb = self.a + self.b;
        let f = self.tyres[0].map_or(self.mu, |t| t.mu_y(self.mass * G * self.b / wb * 0.5));
        let r = self.tyres[1].map_or(self.mu, |t| t.mu_y(self.mass * G * self.a / wb * 0.5));
        f.min(r)
    }

    /// The car-sound feed (carsound.rs) for this step, without placement.
    pub fn sound_feed(&self, handbrake: bool) -> crate::carsound::Feed {
        let s = &self.snd;
        crate::carsound::Feed {
            rpm: self.rpm,
            throttle: if self.gear < 0 { self.brk } else { self.thr },
            brake: if self.gear < 0 { self.thr } else { self.brk },
            gear: self.gear,
            speed_kmh: self.vx * 3.6,
            torque: s.torque,
            load_pct: {
                let peak = self.phys.as_ref().and_then(|p| p.peak_torque()).map_or(400.0, |p| p.1.max(1.0));
                s.torque / peak * 100.0
            },
            boost: s.boost,
            limiter: s.limiter,
            slip: s.slip_f.abs().max(s.slip_r.abs()) + if handbrake { 0.1 } else { 0.0 },
            spin: s.spin,
            grip: s.grip_f.min(s.grip_r),
            load: s.load,
            wheel_radius: self.radius,
            tc_active: self.tc && s.spin > 0.95,
            abs_active: self.brk > 0.9 && s.spin > 0.95 && self.vx.abs() > 5.0,
            ..Default::default()
        }
    }

    /// The rev limiter, rpm (the rigid-body car's own when it drives).
    pub fn rpm_limit(&self) -> f32 {
        self.rig.as_ref().map_or(self.limiter, |(rc, _)| rc.limiter_rpm())
    }

    pub fn speed_kmh(&self) -> f32 {
        self.vx * 3.6
    }

    pub fn mass_kg(&self) -> f32 {
        self.mass
    }

    /// The body's footprint for car-to-car contact (race.rs): half width,
    /// front and rear overhang from the origin along +Z, m - the wheel
    /// centres plus the bodywork past them, as collide_walls has it.
    pub fn footprint(&self) -> (f32, f32, f32) {
        let half_w = self.slot.centres.iter().map(|c| c.x.abs()).fold(0.6f32, f32::max) + 0.15;
        let zf = self.slot.centres.iter().map(|c| c.z).fold(f32::MIN, f32::max) + 0.9;
        let zr = self.slot.centres.iter().map(|c| c.z).fold(f32::MAX, f32::min) - 0.8;
        (half_w, zf, zr)
    }

    /// World velocity, horizontal.
    pub fn vel(&self) -> Vec3 {
        match self.rig.as_ref() {
            Some((rc, _)) => Vec3::new(rc.v.x, 0.0, rc.v.z),
            None => self.forward() * self.vx + self.left() * self.vy,
        }
    }

    /// Tyre smoke wanted 0..1 (race.rs AI) from how fast the rear tyres
    /// slide: the rig car's contact patches, as the player's (render.rs);
    /// this model's rear slip angle past its peak, and wheelspin - drive
    /// demand past grip under power (not braking: GT cars have ABS).
    pub fn smoke_target(&self) -> f32 {
        let slide = match self.rig.as_ref() {
            Some((rc, _)) => {
                let v = rc.v.length();
                rc.c[2..].iter().filter(|c| c.on_ground)
                    .map(|c| v * (c.sr * c.sr + c.alpha.tan().powi(2)).sqrt())
                    .fold(0.0f32, f32::max)
            }
            None => {
                let v = self.vx.abs();
                let peak = self.tyres[1].map_or(8f32.to_radians(), |t| t.peak());
                let lat = v * (self.snd.slip_r.abs() - peak * 1.5).max(0.0).min(1.2).tan();
                let spin = if self.thr > 0.5 && self.brk < 0.1 { (self.snd.spin - 1.2).max(0.0) * v.max(5.0) * 2.0 } else { 0.0 };
                lat + spin
            }
        };
        ((slide - 2.0) / 5.0).clamp(0.0, 1.0)
    }

    /// Yaw rate, rad/s (+ = turning left).
    pub fn yaw_rate(&self) -> f32 {
        self.r
    }

    /// A contact with another car (race.rs): move the car by `dp`, change its
    /// velocity by `dv` and its yaw rate by `dr`, all world space.
    pub fn nudge(&mut self, dp: Vec3, dv: Vec3, dr: f32) {
        let dp = Vec3::new(dp.x, 0.0, dp.z);
        let dv = Vec3::new(dv.x, 0.0, dv.z);
        if let Some((rc, _)) = self.rig.as_mut() {
            rc.cg += dp;
            rc.v += dv;
            rc.w += Vec3::Y * dr;
            for k in 0..4 {
                rc.c[k].hub += dp;
            }
            self.sync_rig();
            return;
        }
        self.pos += dp;
        let v = self.vel() + dv;
        self.vx = v.dot(self.forward());
        self.vy = v.dot(self.left());
        self.r += dr;
        for w in self.wheels.iter_mut() {
            *w += dp;
        }
    }

    pub fn step(&mut self, c: &Controls, dt: f32) {
        if let Some((rc, acc)) = self.rig.as_mut() {
            rc.step(c, dt, &self.ground, acc);
            self.sync_rig();
            return;
        }
        let dt = dt.clamp(0.0, 1.0 / 30.0);
        // sub-step for stability at high yaw rates
        let n = 4;
        for _ in 0..n {
            self.step_inner(c, dt / n as f32);
        }
        self.settle(dt);
    }

    fn step_inner(&mut self, c: &Controls, dt: f32) {
        let approach = |v: &mut f32, t: f32, rate: f32| *v += (t - *v).clamp(-rate * dt, rate * dt);
        approach(&mut self.thr, c.throttle, 5.0);
        approach(&mut self.brk, c.brake, 6.0);
        // less lock at speed, the way a driver steers
        let speed_k = 1.0 / (1.0 + (self.vx.abs() / 25.0).powi(2));
        if c.analog {
            self.steer = c.steer * self.max_steer;
        } else {
            approach(&mut self.steer, c.steer * self.max_steer * speed_k.max(0.25), 2.5);
        }

        // reverse: brake held at a standstill selects it, throttle leaves it
        if self.vx.abs() < 0.5 && c.brake > 0.5 && c.throttle < 0.1 && self.gear > 0 {
            self.gear = -1;
        }
        if self.gear < 0 && c.throttle > 0.1 && self.vx > -0.5 {
            self.gear = 1;
        }
        let reversing = self.gear < 0;
        let (drive_in, brake_in) = if reversing { (c.brake, c.throttle) } else { (self.thr, self.brk) };

        // engine and gearbox (automatic)
        let ratio = self.ratio();
        let wheel_rpm = self.vx / self.radius * 60.0 / std::f32::consts::TAU;
        self.rpm = (wheel_rpm * ratio).abs().clamp(self.idle, self.limiter);
        if self.gear > 0 {
            if self.rpm > self.limiter * 0.93 && (self.gear as usize) < self.gears.len() {
                self.gear += 1;
            } else if self.gear > 1 {
                let down = self.gears[self.gear as usize - 2] * self.final_drive;
                // down once the lower gear is back under 80 % (upshifts at
                // 93 %): the car leaves a corner in the gear that pulls
                if (wheel_rpm * down).abs() < self.limiter * downshift_at() {
                    self.gear -= 1;
                }
            }
        }
        // pulling away the clutch slips, so the engine runs at a launch speed
        // above the wheels' (a locked clutch at idle has no torque to give)
        let launch = (self.idle * 3.0).min(self.limiter * 0.45);
        let rpm_t = if drive_in > 0.05 && self.gear.abs() == 1 { self.rpm.max(launch * drive_in.max(0.5)) } else { self.rpm };
        if rpm_t > self.rpm {
            self.rpm = rpm_t;
        }
        let torque = self.phys.as_ref().and_then(|p| p.engine_torque(rpm_t, drive_in.max(0.01)))
            .unwrap_or(380.0 * drive_in);
        let mut fx_drive = if self.rpm < self.limiter * 0.995 || reversing {
            torque * ratio.abs() / self.radius * 0.85 * if reversing { -1.0 } else { 1.0 }
        } else {
            0.0
        };
        if drive_in < 0.02 {
            fx_drive = 0.0;
        }

        // contact patches: grip under each axle
        let f = self.forward();
        let l = self.left();
        let probe_grip = |z: f32| -> f32 {
            let p = self.pos + f * z;
            self.ground.probe(p.x, p.z, p.y + 2.0).map_or(0.6, |h| self.ground.grip(h.surf))
        };
        let (gf, gr) = (probe_grip(self.a), probe_grip(-self.b));
        let wb = self.a + self.b;
        // load, with longitudinal transfer
        // load transfer from the smoothed longitudinal acceleration (not the
        // full body ax, whose vy*r term feeds a spin back into itself)
        let fz_f = (self.mass * G * self.b - self.mass * self.ax_lt * self.h) / wb;
        let fz_r = (self.mass * G * self.a + self.mass * self.ax_lt * self.h) / wb;
        let (fz_f, fz_r) = (fz_f.max(0.1 * self.mass * G), fz_r.max(0.1 * self.mass * G));
        // each axle's own tyres (.tyre TyresModelData): peak friction at
        // this load per tyre, load-sensitive; else a flat 1.25 (strict DRV1)
        let tyre = |ax: usize, fz: f32, lat: bool| -> f32 {
            match self.tyres[ax] {
                Some(t) => if lat { t.mu_y(fz * 0.5) } else { t.mu_x(fz * 0.5) },
                None => self.mu,
            }
        };
        let mu_f = tyre(0, fz_f, true) * gf;
        let mut mu_r = tyre(1, fz_r, true) * gr;
        let mux_f = tyre(0, fz_f, false) * gf;
        let mut mux_r = tyre(1, fz_r, false) * gr;
        if c.handbrake {
            mu_r *= 0.45;
            mux_r *= 0.45;
        }

        // brakes: 60 / 40, handbrake on the rear
        let v = self.vx;
        // the brakes can use what the four tyres give, longitudinally
        let brake_total = brake_in * 0.95 * (mux_f * fz_f + mux_r * fz_r);
        let mut fx_brake = -brake_total * v.signum();
        if c.handbrake {
            fx_brake -= 0.5 * self.mass * G * v.signum();
        }
        if v.abs() < 0.3 {
            fx_brake = 0.0;
        }
        // drive at the rear, limited by what the rear tyre has
        let fx_max_r = mux_r * fz_r;
        let fx_max_f = mux_f * fz_f;
        // traction control: keep each driven axle under 60 % of what its
        // tyres have, so a full keyboard throttle leaves them some side grip
        // (off: they spin up and the car power-oversteers / understeers)
        let tc = if self.tc { 0.6 } else { 1.0 };
        // the drive is split by the car's layout (DrivetrainData tractionType)
        let fs = self.front_share;
        let fx_f = (fx_drive * fs).clamp(-fx_max_f * tc, fx_max_f * tc);
        let fx_r = (fx_drive * (1.0 - fs)).clamp(-fx_max_r * tc, fx_max_r * tc);
        let drag = 0.5 * 1.2 * 0.7 * v * v.abs() + 0.013 * self.mass * G * v.signum();

        // lateral: slip angles
        let delta = self.steer;
        // ⚠ Keyed on the TOTAL speed and atan2: past 90 degrees of slide the
        // forward speed crosses zero, and "vx > 0.5" switched the tyres'
        // side force off there - the car then stopped dead in its tracks.
        let sp = (v * v + self.vy * self.vy).sqrt();
        let (af, ar) = if sp > 0.5 {
            let vxs = v.abs().max(0.1);
            ((self.vy + self.a * self.r).atan2(vxs) - delta * if v < 0.0 { -1.0 } else { 1.0 },
             (self.vy - self.b * self.r).atan2(vxs))
        } else {
            (0.0, 0.0)
        };
        // peaks at each tyre's friction_limit_angle (8 degrees without
        // one), then falls off to ~80 %
        let peak = |ax: usize| self.tyres[ax].map_or(8f32.to_radians(), |t| t.peak());
        let (bf, br) = (2.07 / peak(0), 2.07 / peak(1));
        let curve = |alpha: f32, b: f32| -> f32 { (1.4 * (b * alpha).atan()).sin() };
        let used_f = (fx_f / fx_max_f.max(1.0)).clamp(-1.0, 1.0);
        let used = (fx_r / fx_max_r.max(1.0)).clamp(-1.0, 1.0);
        // The game's tyre model (sctm.rs) where the car has its .tyre data:
        // two tyres per axle at half the axle load, the surface's grip as
        // the grip multiplier, tread at temperature. Combined slip stays
        // this model's friction circle until wheel spin is simulated (the
        // per-wheel slip ratio, physics step 4).
        let tyre_t = std::env::var("EVOVIEW_TYRE_T").ok().and_then(|v| v.parse().ok()).unwrap_or(f32::NAN);
        let sctm_axle = |ax: usize, fz: f32, alpha: f32, grip: f32| -> Option<crate::sctm::Output> {
            let m = self.tyres[ax]?.sctm?;
            let o = m.solve(&crate::sctm::Input { fz: fz * 0.5, alpha, speed: sp, grip, temp: tyre_t, wet: crate::rain::track_wetness(), ..Default::default() });
            Some(crate::sctm::Output { fy: o.fy * 2.0, mz: o.mz * 2.0, ..o })
        };
        let hb = if c.handbrake { 0.45 } else { 1.0 };
        let no_sctm = std::env::var_os("EVOVIEW_NO_SCTM").is_some();
        let (fyf, fyr) = match (sctm_axle(0, fz_f, af, gf).filter(|_| !no_sctm), sctm_axle(1, fz_r, ar, gr * hb).filter(|_| !no_sctm)) {
            (Some(of), Some(or)) => {
                // what the rack feels, as the rig car builds it: the tyre's
                // aligning moment plus its lateral force on the kingpin's
                // mechanical (caster) trail - the plain Mz alone was ~4x light
                let (arm, up) = self.caster_arm();
                self.mz_front = of.mz * up - of.fy * arm;
                (-of.fy * (1.0 - used_f * used_f).max(0.0).sqrt(), -or.fy * (1.0 - used * used).max(0.0).sqrt())
            }
            _ => {
                self.mz_front = 0.0;
                (-mu_f * fz_f * curve(af, bf) * (1.0 - used_f * used_f).max(0.0).sqrt(),
                 -mu_r * fz_r * curve(ar, br) * (1.0 - used * used).max(0.0).sqrt())
            }
        };
        {
            // demand past what the tyres give (TC clamps the force, not the
            // demand): wheelspin / lock-up for the tyre sound
            let want_r = (fx_drive * (1.0 - fs) + fx_brake * 0.4).abs() / fx_max_r.max(1.0);
            let want_f = (fx_drive * fs + fx_brake * 0.6).abs() / fx_max_f.max(1.0);
            let boost = self.phys.as_ref().map_or(0.0, |p| p.turbos.iter().map(|t| t.boost(rpm_t, drive_in)).fold(0.0, f32::max));
            self.snd = SoundState {
                torque: if drive_in < 0.02 { -0.15 * torque.abs().max(60.0) } else { torque },
                boost,
                slip_f: af,
                slip_r: ar,
                spin: want_r.max(want_f).min(2.0) * if v.abs() < 0.3 && drive_in < 0.05 { 0.0 } else { 1.0 },
                grip_f: gf,
                grip_r: gr,
                limiter: self.rpm >= self.limiter * 0.99 && drive_in > 0.5,
                load: ((fz_f + fz_r) / (self.mass * G)).clamp(0.0, 2.0) * 0.5,
            };
        }

        // front drive pushes along the steered wheels
        let fx = fx_r + fx_f * delta.cos() + fx_brake - drag;
        let lt = (fx / self.mass).clamp(-1.3 * G, 1.0 * G);
        self.ax_lt += (lt - self.ax_lt) * (dt * 8.0).min(1.0);
        let ax = (fx - fyf * delta.sin()) / self.mass + self.vy * self.r;
        let ay = (fyf * delta.cos() + fyr + fx_f * delta.sin()) / self.mass - self.vx * self.r;
        let mut rdot = (self.a * (fyf * delta.cos() + fx_f * delta.sin()) - self.b * fyr) / self.iz;
        // stability control (with TC): yaw rate past what the steering and
        // the grip can ask for is damped, the way ESC brakes one corner
        if self.tc && v > 5.0 {
            let r_ask = v * delta.tan() / wb;
            let r_grip = 0.9 * self.grip_estimate() * G / v;
            let r_ok = r_ask.clamp(-r_grip, r_grip);
            let excess = self.r - r_ok;
            if excess.abs() > 0.05 {
                rdot -= excess.signum() * (excess.abs() - 0.05) * 6.0;
            }
        }
        let nvx = self.vx + ax * dt;
        // the brakes stop the car, not reverse it
        self.vx = if self.vx != 0.0 && nvx.signum() != self.vx.signum() && fx_drive.abs() < 1.0 { 0.0 } else { nvx };
        self.vy += ay * dt;
        self.r += rdot * dt;
        // low speed: blend to rolling without slip, so it can creep and park
        // (total speed: a car sliding sideways is not "slow")
        let k = ((self.vx * self.vx + self.vy * self.vy).sqrt() / 3.0).clamp(0.0, 1.0);
        let r_kin = self.vx * delta.tan() / wb;
        self.r = self.r * k + r_kin * (1.0 - k);
        self.vy *= k.max(0.2);
        self.ax = ax;
        self.ay = ay;

        // move, unless a wheel would hit a wall
        let step = (f * self.vx + l * self.vy) * dt;
        let next = self.pos + step;
        // a wall, or the edge of the collision mesh: nothing within a 0.6 m
        // step at the next position (the probe ignores vertical triangles)
        // only the end the car is moving toward: a car stopped against a wall
        // can still back away from it
        let lead = if self.vx >= 0.0 { self.a + 0.5 } else { -self.b - 0.5 };
        let blocked = step.length() > 1e-5 && {
            let p = next + f * lead;
            self.ground.probe(p.x, p.z, self.pos.y + 0.6 + lead.abs() * 0.15).is_none()
        };
        if !blocked {
            self.collide_walls();
        }
        if blocked {
            if std::env::var_os("EVOVIEW_DRIVE_LOG").is_some() {
                eprintln!("blocked at ({:.1} {:.1} {:.1}) v {:.1}", next.x, next.y, next.z, self.vx);
            }
            self.vx *= -0.25;
            self.vy *= 0.3;
            self.r *= 0.3;
        } else {
            self.pos = next;
        }
        self.yaw += self.r * dt;
        self.spin += self.vx * dt / self.radius;
    }

    /// Barriers: the body as three circles along its length; each overlapping
    /// wall pushes it out, takes away the speed into the wall (a little
    /// bounce), drags along it, and kicks the yaw by where it was hit.
    fn collide_walls(&mut self) {
        let f = self.forward();
        let l = self.left();
        let half_w = self.slot.centres.iter().map(|c| c.x.abs()).fold(0.6f32, f32::max) + 0.12;
        let zf = self.slot.centres.iter().map(|c| c.z).fold(f32::MIN, f32::max) + 0.55;
        let zr = self.slot.centres.iter().map(|c| c.z).fold(f32::MAX, f32::min) - 0.55;
        let mut hits = Vec::new();
        let mut any = false;
        for z in [zf - half_w, 0.5 * (zf + zr), zr + half_w] {
            let c = self.pos + f * z;
            self.ground.walls_near(c, half_w, self.pos.y + 0.15, self.pos.y + 1.2, &mut hits);
            for &(q, d, _) in hits.iter() {
                let n = Vec3::new(c.x - q.x, 0.0, c.z - q.z);
                let n = if d > 1e-4 { n / d } else { -f };
                let pen = half_w - d;
                self.pos += n * pen;
                let mut v = f * self.vx + l * self.vy;
                let vn = v.dot(n);
                if vn < 0.0 {
                    // restitution 0.15, 15 % of the sliding speed lost
                    let vt = v - n * vn;
                    v = vt * 0.85 - n * vn * 0.15;
                    let arm = c - self.pos;
                    let j = -1.15 * vn * self.mass;
                    let jv = n * j;
                    self.r += (arm.z * jv.x - arm.x * jv.z) / self.iz * 0.5;
                    self.vx = v.dot(f);
                    self.vy = v.dot(l);
                    any = true;
                }
            }
        }
        if any && std::env::var_os("EVOVIEW_DRIVE_LOG").is_some() {
            eprintln!("wall at ({:.1} {:.1} {:.1}) v {:.1}", self.pos.x, self.pos.y, self.pos.z, self.vx);
        }
    }

    /// Body on the plane through the four contact points, plus spring lean.
    fn settle(&mut self, dt: f32) {
        let f = self.forward();
        let l = self.left();
        let mut hits = [Vec3::ZERO; 4];
        for k in 0..4 {
            let c = self.slot.centres[k];
            // mesh +X is left, +Z forward
            let p = self.pos + l * c.x + f * c.z;
            let h = self.ground.probe(p.x, p.z, self.pos.y + 1.5);
            let y = h.map_or(self.pos.y, |h| h.y + self.ground.bump(h.surf, p.x, p.z));
            hits[k] = Vec3::new(p.x, y, p.z);
        }
        // 0 FR, 1 FL, 2 RR, 3 RL
        let front = (hits[0] + hits[1]) * 0.5;
        let rear = (hits[2] + hits[3]) * 0.5;
        let lft = (hits[1] + hits[3]) * 0.5;
        let rgt = (hits[0] + hits[2]) * 0.5;
        let fwd = (front - rear).normalize_or_zero();
        let lat = (lft - rgt).normalize_or_zero();
        // fwd x left = up (left x fwd pointed DOWN, so every pose fell back
        // to level: the car never took the road's slope or camber).
        // EVOVIEW_FLAT_BODY=1: the old level body.
        let up = if std::env::var_os("EVOVIEW_FLAT_BODY").is_some() { Vec3::Y } else { fwd.cross(lat).normalize_or_zero() };
        let up = if up.y > 0.3 { up } else { Vec3::Y };
        self.pos.y = (front.y + rear.y) * 0.5;
        // springs: pitch from ax, roll from ay (visual)
        if dt > 0.0 {
            // ~1.7 deg of roll per g, damped at 0.65 (was 2.6 deg / g at
            // 0.45, from dvy/dt rather than the felt cornering acceleration)
            let (wp, z) = (9.0f32, 0.65f32);
            let target_p = -self.ax_lt * 0.0025;
            let target_r = (self.ay + self.vx * self.r) * 0.003;
            self.pitch_v += (wp * wp * (target_p - self.pitch) - 2.0 * z * wp * self.pitch_v) * dt;
            self.pitch += self.pitch_v * dt;
            self.roll_v += (wp * wp * (target_r - self.roll) - 2.0 * z * wp * self.roll_v) * dt;
            self.roll += self.roll_v * dt;
        }
        self.up = up;
        for k in 0..4 {
            self.wheels[k] = hits[k] + up * self.slot.radius;
        }
    }

    /// The pose the replay code turns into body and wheel matrices.
    pub fn pose(&self) -> Pose {
        let f0 = self.forward();
        let up0 = self.up;
        let fwd = (f0 - up0 * f0.dot(up0)).normalize_or_zero();
        // spring lean on top of the ground plane
        let rot = Mat4::from_axis_angle(up0.cross(fwd).normalize_or_zero(), self.pitch)
            * Mat4::from_axis_angle(fwd, -self.roll);
        let fwd = rot.transform_vector3(fwd).normalize();
        let up = rot.transform_vector3(up0).normalize();
        let right = fwd.cross(up).normalize();
        Pose { pos: self.pos, forward: fwd, right, up, wheels: self.wheels }
    }

    /// What the car's rig (carrig.rs, via drive::Drive::follow) needs from
    /// this model: the rig then turns the steering wheel, spins and steers
    /// the wheel bones, moves the needles and leans the body itself.
    /// What the force feedback (ffb.rs) is built from: the front hubs'
    /// aligning torque (N m), the road-wheel steer angle (rad), the summed
    /// front wheel speed (rad/s), km/h and the car's ffMult.
    /// The front kingpin's lever for a lateral force at the contact patch
    /// (m: mechanical trail x the axis' vertical share) and that vertical
    /// share, from the suspension geometry; (0, 1) without one.
    /// EVOVIEW_FFB_NOCASTER=1: the bare tyre Mz, as before.
    fn caster_arm(&self) -> (f32, f32) {
        if std::env::var_os("EVOVIEW_FFB_NOCASTER").is_some() {
            return (0.0, 1.0);
        }
        match self.phys.as_ref().and_then(|p| p.geo[0].kingpin()) {
            // contact at (0, -R, 0) from the hub: arm = (R + b.y) d.z - b.z d.y,
            // the rig car's (contact - axis) x F . axis for F across the car
            Some((b, d)) if d.y > 0.3 => (((self.radius + b.y) * d.z - b.z * d.y).abs(), d.y),
            _ => (0.0, 1.0),
        }
    }

    /// The car's FFB output gamma (SteerControlsData f5; 1 = linear).
    pub fn ffb_gamma(&self) -> f32 {
        self.phys.as_ref().map(|p| p.ff_gamma).filter(|g| *g > 0.0).unwrap_or(1.0)
    }

    pub fn ffb_inputs(&self) -> (f32, f32, f32, f32, f32) {
        let ff = self.phys.as_ref().map(|p| p.ff_mult).filter(|m| *m > 0.0).unwrap_or(1.0);
        (self.mz_front, self.steer, 2.0 * self.vx / self.radius, self.speed_kmh(), ff)
    }

    pub fn rig_feed(&self) -> RigFeed {
        RigFeed {
            speed: self.vx,
            rpm: self.rpm,
            gear: self.gear,
            throttle: self.thr,
            brake: self.brk,
            // ours: road-wheel radians, + = left; the rig's: -1..1, + = right
            steer: (-self.steer / self.max_steer.max(1e-3)).clamp(-1.0, 1.0),
            a_long: self.ax_lt,
            // felt lateral acceleration, + = toward the right
            a_lat: -(self.ay + self.vx * self.r),
        }
    }

    /// The body matrix without the spring lean: with a rig, the rig's own
    /// body motion pitches and rolls the hull over the wheels.
    /// rigcar: the road's up under the car - the plane through the four
    /// (world-space) hubs (they ride the road; the body pitches and rolls on its
    /// springs above them). The simple model: its tyre-contact plane (the
    /// spring lean goes on top of it).
    pub fn road_up(&self) -> Option<glam::Vec3> {
        let Some((rc, _)) = self.rig.as_ref() else { return Some(self.up) };
        let h = |k| rc.hub(k);
        // world-space hubs, 0 FR 1 FL 2 RR 3 RL; fwd x left = up
        let fwd = (h(0) + h(1)) - (h(2) + h(3));
        let lat = (h(1) + h(3)) - (h(0) + h(2));
        let w = fwd.cross(lat).normalize_or_zero();
        (w.y > 0.3).then_some(w)
    }

    pub fn body_flat(&self) -> Mat4 {
        let up = self.up;
        let f0 = self.forward();
        let fwd = (f0 - up * f0.dot(up)).normalize_or_zero();
        let right = fwd.cross(up).normalize();
        let p = Pose { pos: self.pos, forward: fwd, right, up, wheels: self.wheels };
        crate::replayscene::car_matrix(&p, self.slot.offset)
    }

    pub fn matrices(&self) -> (Mat4, [Mat4; 4]) {
        let p = self.pose();
        let mut wm = [Mat4::ZERO; 4];
        for k in 0..4 {
            wm[k] = crate::replayscene::wheel_matrix(&p, k, self.slot.centres[k], self.slot.axles[k], self.spin, self.steer);
            // a blown tyre: flattened onto the road about its hub
            if let Some(s) = self.rig.as_ref().and_then(|(rc, _)| rc.blown_squash(k)) {
                let hub = wm[k].w_axis.truncate();
                wm[k] = Mat4::from_translation(hub) * Mat4::from_scale(glam::Vec3::new(1.0, s, 1.0))
                    * Mat4::from_translation(-hub) * wm[k];
            }
        }
        (crate::replayscene::car_matrix(&p, self.slot.offset), wm)
    }
}

/// The layout's racing line (`layouts/*.ideal_line.aisplinedata`, a GP
/// layout first): every point, in order.
/// ⚠ The layout is the one the scenery was loaded with (--layout /
/// EVOVIEW_LAYOUT, else load.rs pick_race_layout): the Nurburgring loads
/// the Nordschleife, and its GP line - this used to prefer any "gp" line -
/// runs where no GP-Strecke ground was loaded, so the car fell out of the
/// world at the start.
pub fn racing_line(pkg: &std::path::Path, track: &str) -> Vec<Vec3> {
    let lay = std::env::var("EVOVIEW_LAYOUT").ok().map(|l| l.to_lowercase()).or_else(crate::load::loaded_layout);
    if let Some(l) = lay.as_deref() {
        let pts = racing_line_layout(pkg, track, Some(l));
        if !pts.is_empty() {
            return pts;
        }
    }
    racing_line_layout(pkg, track, None)
}

/// The racing line of `layout` (e.g. "nordschleife"), or the track's GP one.
pub fn racing_line_layout(pkg: &std::path::Path, track: &str, layout: Option<&str>) -> Vec<Vec3> {
    use crate::proto::fields;
    let Ok(p) = crate::pkg::Package::open_filtered(pkg, &[&format!("content\\tracks\\{}\\", track.to_lowercase())]) else {
        return Vec::new();
    };
    let mut keys: Vec<&String> = p.by_path.keys()
        .filter(|k| k.ends_with(".ideal_line.aisplinedata") && !k.contains("pit"))
        .filter(|k| layout.map_or(true, |l| k.to_lowercase().ends_with(&format!("layout_{}.ideal_line.aisplinedata", l.to_lowercase()))))
        .collect();
    keys.sort_by_key(|k| (!k.contains("gp"), k.len()));
    let Some(e) = keys.first().and_then(|k| p.by_path.get(*k)) else { return Vec::new() };
    println!("racing line: {}", keys[0]);
    let d = p.read(*e);
    // 1 { 1 { 1 { x y z } ... } ... }
    let mut pts: Vec<Vec3> = Vec::new();
    fields(&d, 0, d.len(), |f, v| {
        let (1, Some((a, b))) = (f, v.range()) else { return };
        fields(&d, a, b, |f2, v2| {
            let (1, Some((c, e))) = (f2, v2.range()) else { return };
            fields(&d, c, e, |f3, v3| {
                let (1, Some((g, h))) = (f3, v3.range()) else { return };
                let mut q = [0.0f32; 3];
                fields(&d, g, h, |f4, v4| {
                    if (1..=3).contains(&f4) {
                        q[f4 as usize - 1] = v4.f32();
                    }
                });
                pts.push(Vec3::from(q));
            });
        });
    });
    pts
}

/// Where to start: the racing line's first point, heading along it.
pub fn spawn_point(pkg: &std::path::Path, track: &str) -> Option<(Vec3, f32)> {
    let pts = racing_line(pkg, track);
    if let (Some(&a), Some(&b)) = (pts.first(), pts.get(5).or(pts.last())) {
        let d = b - a;
        return Some((a, d.x.atan2(d.z)));
    }
    mod_spawn(track)
}

/// A loose mod track's start: containers/spawnpoints_hotlap.scene (else the
/// pit lane's first) - SceneData f2 node {1 name, 4 transform {1 pos, 2 euler
/// deg}}, the drift track's at (2.9, 0.7, 141.8) yaw -59.
fn mod_spawn(track: &str) -> Option<(Vec3, f32)> {
    let dir = std::path::Path::new(track);
    let dir = if dir.is_dir() {
        dir.to_path_buf()
    } else {
        std::path::Path::new(&std::env::var_os("USERPROFILE")?).join("Saved Games").join("ACE").join("mods").join("content").join("tracks").join(track)
    };
    // per-layout files too (spawnpoints_hotlap_gp.scene): the running layout's
    // first, else the plainest name - a track with only those spawned at 0,0
    let lay = std::env::var("EVOVIEW_LAYOUT").ok().map(|l| l.to_lowercase()).filter(|l| !l.is_empty());
    let names: Vec<String> = std::fs::read_dir(dir.join("containers")).map(|rd| {
        rd.flatten().map(|e| e.file_name().to_string_lossy().to_lowercase()).collect()
    }).unwrap_or_default();
    let mut order: Vec<String> = Vec::new();
    for kind in ["spawnpoints_hotlap", "spawnpoints_pitlane", "spawnpoints_grid"] {
        let mut v: Vec<&String> = names.iter().filter(|n| n.starts_with(kind) && n.ends_with(".scene")).collect();
        v.sort_by_key(|n| (!lay.as_ref().map_or(false, |l| n.contains(l.as_str())), n.len()));
        order.extend(v.into_iter().cloned());
    }
    for f in &order {
        let Ok(d) = std::fs::read(dir.join("containers").join(f)) else { continue };
        let mut out = None;
        crate::proto::fields(&d, 0, d.len(), |fno, v| {
            let (2, Some((a, b)), None) = (fno, v.range(), out) else { return };
            crate::proto::fields(&d, a, b, |g, w| {
                let (4, Some((x, y))) = (g, w.range()) else { return };
                let mut pos = None;
                let mut yaw = 0.0f32;
                crate::proto::fields(&d, x, y, |h, t| {
                    let Some((p, q)) = t.range() else { return };
                    let mut v = [0.0f32; 3];
                    crate::proto::fields(&d, p, q, |i, z| {
                        if (1..=3).contains(&i) {
                            v[i as usize - 1] = z.f32();
                        }
                    });
                    match h {
                        1 => pos = Some(Vec3::from(v)),
                        2 => yaw = v[1].to_radians(),
                        _ => {}
                    }
                });
                if let Some(p) = pos {
                    out = Some((p, yaw));
                }
            });
        });
        if out.is_some() {
            return out;
        }
    }
    None
}

/// A driver for the racing line: pure-pursuit steering at a look-ahead that
/// grows with speed, and a speed target from the line's curvature ahead
/// (v = sqrt(0.85 mu g R)), braking when over it. Used by --drive-test.
pub struct LineDriver {
    pub line: Vec<Vec3>,
    i: usize,
    /// seconds pushing without moving, and seconds of backing out left
    stuck: f32,
    backing: f32,
    back_steer: f32,
    /// where the backing out started, and seconds of rejoining left
    back_from: Vec3,
    rejoin: f32,
    /// race.rs's per-driver knobs: share of the line's corner speeds (1 =
    /// the ideal), braking deceleration x (1 = 6.5 m/s^2 planned)
    pub pace: f32,
    pub brake_k: f32,
    /// metres off the line to the left (- right): `offset` is driven, it
    /// eases toward `offset_target` (passing, giving room)
    pub offset: f32,
    pub offset_target: f32,
    /// a speed cap, m/s (a car ahead, the rolling start); INFINITY = none
    pub vcap: f32,
    /// the line has been moved off the walls beside it (clear_walls)
    cleared: bool,
}

impl LineDriver {
    pub fn new(line: Vec<Vec3>) -> LineDriver {
        let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
        let line = match (env("EVOVIEW_KERB_ATTACK"), env("EVOVIEW_LINE_WANDER")) {
            (None, None) => line,
            (m, w) => kerb_attack(line, m.unwrap_or(0.0), w.unwrap_or(0.0)),
        };
        LineDriver { line, i: 0, stuck: 0.0, backing: 0.0, back_steer: 0.0, back_from: Vec3::ZERO, rejoin: 0.0, pace: 1.0, brake_k: 1.0, offset: 0.0, offset_target: 0.0, vcap: f32::INFINITY, cleared: false }
    }

    /// The racing line moved off solid things beside it: the game's ideal
    /// line uses every centimetre of a kerb, and a car a hand's width off it
    /// (the driver tracks within ~1 m) put its flank into whatever stands
    /// there - a bush at Watkins Glen's last kerb at 144 km/h. Each point
    /// with a wall closer sideways than half a car plus margin (`need`)
    /// moves away (at most 1.2 m), spread smoothly along the line.
    /// EVOVIEW_LINE_CLEAR=0: the line as shipped.
    /// `half_w`: the car's half width as the wall collision sees it.
    pub fn clear_walls(&mut self, ground: &Ground, half_w: f32) {
        if std::env::var("EVOVIEW_LINE_CLEAR").map_or(false, |v| v == "0") {
            return;
        }
        let n = self.line.len();
        if n < 3 {
            return;
        }
        // the car's half width and ~0.8 m of tracking error (1.6 was the
        // MX-5's; Bathurst's Esses wall 1.72 m from the line took the
        // wider M4 GT3's flank every lap)
        let need = (half_w + 0.8).max(1.6);
        let mut push = vec![0.0f32; n];
        let mut hits = Vec::new();
        for i in 0..n {
            let c = self.line[i];
            let left = self.line_left(i);
            let tan = Vec3::new(-left.z, 0.0, left.x);
            ground.walls_near(c, need + 0.5, c.y + 0.15, c.y + 1.2, &mut hits);
            let (mut l, mut r) = (0.0f32, 0.0f32);
            if std::env::var("EVOVIEW_CLEAR_DBG").ok().and_then(|v| v.parse::<usize>().ok()) == Some(i) {
                println!("CLEAR {i} at ({:.1} {:.1} {:.1}) left {left:?}: {:?}", c.x, c.y, c.z, hits.iter().map(|h| (h.0.x, h.0.z, h.1)).collect::<Vec<_>>());
            }
            for &(q, _, _) in hits.iter() {
                let d = Vec3::new(q.x - c.x, 0.0, q.z - c.z);
                if d.dot(tan).abs() > 1.5 {
                    continue;
                }
                let lat = d.dot(left);
                if lat >= 0.0 && lat < need {
                    l = l.max(need - lat);
                } else if lat < 0.0 && -lat < need {
                    r = r.max(need + lat);
                }
            }
            // walls both sides (a narrow gap): leave it
            if l > 0.0 && r > 0.0 {
                continue;
            }
            push[i] = (r - l).clamp(-1.2, 1.2);
        }
        self.apply_push(&push, "off walls beside them");
    }

    /// Sideways moves (+ = left, per line point) applied smoothly: the
    /// strongest within +-12 points, then a +-12 mean.
    fn apply_push(&mut self, push: &[f32], why: &str) {
        let n = self.line.len();
        let w = 12usize;
        let wide: Vec<f32> = (0..n).map(|i| (0..=2 * w).map(|k| push[(i + n + k - w) % n]).fold(0.0f32, |a, b| if b.abs() > a.abs() { b } else { a })).collect();
        let smooth: Vec<f32> = (0..n).map(|i| (0..=2 * w).map(|k| wide[(i + n + k - w) % n]).sum::<f32>() / (2 * w + 1) as f32).collect();
        let lefts: Vec<Vec3> = (0..n).map(|i| self.line_left(i)).collect();
        let mut moved = 0;
        let mut most = 0.0f32;
        for i in 0..n {
            if smooth[i].abs() > 0.02 {
                moved += 1;
                most = most.max(smooth[i].abs());
                self.line[i] += lefts[i] * smooth[i];
            }
        }
        if moved > 0 {
            println!("racing line: {moved} of {n} points moved {why} (up to {most:.2} m)");
        }
    }

    /// The racing line kept inside the track limits: the ideal line runs the
    /// whole exit kerb (Watkins Glen's T1: its centre 1.9 m past the limit),
    /// and a driver ~1 m off it put all four wheels over and lost the lap
    /// every time. A lap goes when every wheel is 1.2 m past a limit; the
    /// line's centre may sit 0.8 m past one. EVOVIEW_LINE_CLEAR=0: as shipped.
    pub fn keep_inside(&mut self, tt: &crate::laps::TimingTrack) {
        if std::env::var("EVOVIEW_LINE_CLEAR").map_or(false, |v| v == "0") {
            return;
        }
        let Some((l, sl, r, sr)) = tt.limits.as_ref() else { return };
        let n = self.line.len();
        if n < 3 {
            return;
        }
        let allow = 0.8f32;
        let (mut hl, mut hr) = (None, None);
        let mut push = vec![0.0f32; n];
        for i in 0..n {
            let c = self.line[i];
            let pl = l.project(c, hl);
            let pr = r.project(c, hr);
            hl = Some(pl.seg);
            hr = Some(pr.seg);
            let (el, er) = (pl.lat * sl, pr.lat * sr);
            if el > allow {
                push[i] = -(el - allow).min(2.0);
            } else if er > allow {
                push[i] = (er - allow).min(2.0);
            }
        }
        self.apply_push(&push, "inside the track limits");
    }

    /// The line's left (unit, horizontal) at point `j`.
    pub fn line_left(&self, j: usize) -> Vec3 {
        let n = self.line.len();
        let (p0, p1) = (self.line[(j + n - 1) % n], self.line[(j + 1) % n]);
        let t = Vec3::new(p1.x - p0.x, 0.0, p1.z - p0.z).normalize_or_zero();
        Vec3::new(t.z, 0.0, -t.x)
    }

    /// Held still (a race start): no stuck timer, no backing out.
    /// (seconds pushing without moving, backing left, rejoin left) for traces
    pub fn recovery(&self) -> (f32, f32, f32) {
        (self.stuck, self.backing, self.rejoin)
    }

    pub fn rest(&mut self) {
        self.stuck = 0.0;
        self.backing = 0.0;
        self.rejoin = 0.0;
    }

    /// Jump the progress to the line point nearest `p` (a placed car).
    pub fn snap(&mut self, p: Vec3) {
        self.rest();
        let mut best = (f32::MAX, 0);
        for (j, q) in self.line.iter().enumerate() {
            let d = q.distance_squared(p);
            if d < best.0 {
                best = (d, j);
            }
        }
        self.i = best.1;
    }

    /// the line point the car is at
    pub fn progress(&self) -> usize {
        self.i
    }

    /// `dt` is the time since the last call (for the stuck / back-out timers).
    pub fn controls(&mut self, td: &TrackDrive, dt: f32) -> Controls {
        if !self.cleared {
            self.cleared = true;
            let half_w = td.slot.centres.iter().map(|c| c.x.abs()).fold(0.6f32, f32::max) + 0.12;
            self.clear_walls(&td.ground, half_w);
        }
        Controls { auto_box: true, ..self.controls_line(td, dt) }
    }

    fn controls_line(&mut self, td: &TrackDrive, dt: f32) -> Controls {
        let n = self.line.len();
        if n < 8 {
            return Controls::default();
        }
        // backing out of a wall: reverse with the wheel turned the other way
        // until the nose points back along the line (within ~30 deg) or it
        // has come 4 m back; not on into whatever is behind it; 4 s at most.
        // Then rejoin gently: the old 2 s back-out ended still facing the
        // barrier, and full lock on full throttle put it straight back in
        if self.backing > 0.0 {
            self.backing -= dt;
            let j = self.i;
            let tan = Vec3::new(self.line[(j + 3) % n].x - self.line[j].x, 0.0, self.line[(j + 3) % n].z - self.line[j].z).normalize_or_zero();
            let facing = td.forward().dot(tan) > 0.87;
            let back = Vec3::new(td.pos.x - self.back_from.x, 0.0, td.pos.z - self.back_from.z).length();
            // a wall within a metre behind the tail (or either rear corner)
            let (hw, _, zr) = td.footprint();
            let (f, l) = (td.forward(), td.left());
            let y = td.pos.y;
            let wall_behind = [-hw, 0.0, hw].iter().any(|&x| {
                let a = td.pos + f * zr + l * x;
                td.ground.wall_across(a + f * 0.3, a - f * 1.0, y + 0.1, y + 1.0)
            });
            let blocked = wall_behind || (self.backing < 3.0 && td.vx.abs() < 0.3);
            if (facing && back > 1.5) || back > 4.0 || blocked || self.backing <= 0.0 {
                self.backing = 0.0;
                self.rejoin = 3.0;
            } else {
                // the rig car takes real pedals (the keyboard ramp and its
                // steering cut left a wedged car barely moving back)
                // walking pace back (~11 km/h): faster, it hit what was behind
                let gas = if td.vx < -3.0 { 0.0 } else { 0.7 };
                return Controls { throttle: 0.0, brake: gas, steer: self.back_steer, handbrake: false, analog: td.rig.is_some(), ..Default::default() };
            }
        }
        self.rejoin = (self.rejoin - dt).max(0.0);
        // nearest point, searching forward from the last one; the whole line
        // when the car has strayed (a spin, an off, a wall)
        let mut best = (f32::MAX, self.i);
        for k in 0..60 {
            let j = (self.i + k) % n;
            let d = self.line[j].distance_squared(td.pos);
            if d < best.0 {
                best = (d, j);
            }
        }
        if best.0 > 15.0 * 15.0 {
            for j in 0..n {
                let d = self.line[j].distance_squared(td.pos);
                if d < best.0 {
                    best = (d, j);
                }
            }
        }
        self.i = best.1;
        // lost: well off the line (the grass, a gravel trap 30-50 m out).
        // Head for the line itself 10 m on, slowly; the racing-line target
        // and its Stanley term only asked for full lock, and the car
        // circled or spun its wheels on the grass
        let off_line = best.0.sqrt();
        let lost = off_line > 6.0 + self.offset.abs();
        // ease onto the wanted offset (~2.5 m/s sideways)
        self.offset += (self.offset_target - self.offset).clamp(-2.5 * dt, 2.5 * dt);
        let v = td.vx.max(0.0);
        // rejoining: aim further down the line (a gentler angle back on)
        let ahead = if lost { 10.0 } else if self.rejoin > 0.0 { 15.0 + v * 0.55 } else { 6.0 + v * 0.55 };
        let mut j = self.i;
        let mut dist = 0.0;
        while dist < ahead {
            let k = (j + 1) % n;
            dist += self.line[j].distance(self.line[k]);
            j = k;
        }
        let tgt = self.line[j] + self.line_left(j) * if lost { 0.0 } else { self.offset };
        let to = tgt - td.pos;
        // lost and the way back is behind: a three-point turn (back out
        // with the wheels the other way) rather than circling on full lock
        if lost && v < 5.0 && to.dot(td.forward()) < -0.3 * to.length() {
            self.backing = 4.0;
            self.back_from = td.pos;
            self.back_steer = -to.dot(td.left()).signum();
            self.stuck = 0.0;
        }
        let local_x = to.dot(td.left());
        let local_z = to.dot(td.forward()).max(0.5);
        let steer_angle = (2.0 * (self.a_b(td)) * local_x / (local_x * local_x + local_z * local_z)).atan();
        // the rigid-body car has real tyres (it understeers): add a Stanley
        // term on the signed distance to the line at the nearest point
        let steer_angle = if td.rig.is_some() && !lost {
            let o = self.line_left(self.i) * self.offset;
            let (p0, p1) = (self.line[self.i] + o, self.line[(self.i + 2) % n] + o);
            let tan = Vec3::new(p1.x - p0.x, 0.0, p1.z - p0.z).normalize_or_zero();
            let e = Vec3::new(p0.x - td.pos.x, 0.0, p0.z - td.pos.z);
            // + = the line is to the car's left
            let lat = e.dot(td.left()) - tan.dot(td.left()) * e.dot(tan);
            // never wind the fronts past their peak: no more lock than the
            // car can use at this speed - the kinematic angle for ~1.7 g plus
            // the tyres' ~5 deg of peak slip (more only scrubs speed, and an
            // off-line car would keep adding it)
            let lim = (self.a_b(td) * 1.7 * G / (v * v).max(1.0)).atan() + 0.09;
            // yaw damping: steer off the gap between the car's yaw rate
            // and the one its pursuit arc asks for. Without it a short,
            // quick-steering car (MX-5) weaves on Imola's straight before
            // Tamburello until the rear lets go (EVOVIEW_LINE_DAMP, 0 = off; at
            // 1.5 the MX-5 and M4 lap Imola clean for 5 min, 0.5 still spun
            // the MX-5 at Rivazza)
            let kd = std::env::var("EVOVIEW_LINE_DAMP").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5f32);
            let r_des = v * 2.0 * local_x / (local_x * local_x + local_z * local_z);
            let damp = if v > 5.0 { -kd * ((td.r - r_des) * self.a_b(td) / v).atan() } else { 0.0 };
            // catch a slide: the rears past their peak (6 deg) and slipping
            // 2 deg more than the fronts - point the fronts down the road,
            // no more than 3 deg of slip into the slide (a cap off the
            // fronts' slip with the wheels straight, so it does not feed on
            // its own countersteer). The path terms countersteered only as
            // the line drifted: the slider-mode MX-5 swung through
            // Hatzenbach's downhill flick, rears 6 -> 20 deg in a second,
            // and spun into the wall every lap (EVOVIEW_LINE_CATCH=0: off)
            let cap = match td.rig.as_ref() {
                Some((rc, _)) if v > 8.0 && std::env::var("EVOVIEW_LINE_CATCH").map_or(true, |v| v != "0") => {
                    let af = 0.5 * (rc.c[0].alpha + rc.c[1].alpha);
                    let ar = 0.5 * (rc.c[2].alpha + rc.c[3].alpha);
                    (ar.abs() > 6f32.to_radians() && ar.abs() > af.abs() + 2f32.to_radians())
                        .then(|| (ar.signum(), -(af - rc.steer) + ar.signum() * 3f32.to_radians()))
                }
                _ => None,
            };
            let mut s = (steer_angle + damp + (1.2 * lat).atan2(v + 3.0)).clamp(-lim, lim);
            // and no more lock than the fronts can use: their slip held to
            // the tyre's peak (~11 deg: the MX-5's front force still grows
            // at 11 and falls past it). Into COTA's esses it wound the
            // fronts to 15-19 deg, lost force, and ran 10 m wide
            // (EVOVIEW_LINE_FPEAK, degrees; 0 = off)
            let fpeak = std::env::var("EVOVIEW_LINE_FPEAK").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(11.0).to_radians();
            if let Some((rc, _)) = td.rig.as_ref().filter(|_| v > 8.0 && fpeak > 0.0) {
                let af0 = 0.5 * (rc.c[0].alpha + rc.c[1].alpha) - rc.steer;
                s = s.clamp(-fpeak - af0, fpeak - af0);
            }
            match cap {
                Some((sg, c)) if sg > 0.0 => s.min(c),
                Some((_, c)) => s.max(c),
                None => s,
            }
        } else {
            steer_angle
        };
        // curvature over the next ~(v^2 / 2a + 30) m
        let look = 30.0 + v * v / 12.0;
        let mut vmax = f32::MAX;
        // an S: a corner turning against the one the car has just been
        // in (the strongest turn over the last ~35 m of line). The swing
        // from one to the other unloads the rear and the car arrives with
        // its yaw going the wrong way; out of COTA's ~1 g left straight
        // into the long right the MX-5 ran 7-10 m wide in every mode.
        // Take the second one slower (EVOVIEW_LINE_SBEND: speed factor,
        // 1 = off; 0.9 still weaved the MX-5 off at the Nordschleife 3.4 km)
        let turn = |j: usize| {
            let (p0, p1, p2) = (self.line[j], self.line[(j + 4) % n], self.line[(j + 8) % n]);
            let (a, b) = ((p1 - p0) * Vec3::new(1.0, 0.0, 1.0), (p2 - p1) * Vec3::new(1.0, 0.0, 1.0));
            let ang = a.normalize_or_zero().dot(b.normalize_or_zero()).clamp(-1.0, 1.0).acos();
            (a.x * b.z - a.z * b.x).signum() * ang / (a.length() + b.length()).max(0.1)
        };
        let sbend = if td.rig.is_some() { std::env::var("EVOVIEW_LINE_SBEND").ok().and_then(|v| v.parse().ok()).unwrap_or(0.85f32) } else { 1.0 };
        let behind = (0..20usize).map(|k| turn((self.i + n - k) % n)).fold(0.0f32, |m, t| if t.abs() > m.abs() { t } else { m });
        let (mut j, mut d) = (self.i, 0.0);
        while d < look {
            let (p0, p1, p2) = (self.line[j], self.line[(j + 4) % n], self.line[(j + 8) % n]);
            let (a, b) = ((p1 - p0) * Vec3::new(1.0, 0.0, 1.0), (p2 - p1) * Vec3::new(1.0, 0.0, 1.0));
            let ang = a.normalize_or_zero().dot(b.normalize_or_zero()).clamp(-1.0, 1.0).acos();
            let seg = (a.length() + b.length()).max(0.1);
            if ang > 1e-3 {
                let r = seg / ang;
                let s_f = if behind.abs() > 1.0 / 700.0 && r < 700.0 && behind * (a.x * b.z - a.z * b.x) < 0.0 { sbend } else { 1.0 };
                // allowed speed at that radius, and what braking to it from here needs
                // a real-tyre car (rigcar) loses grip to load transfer: aim lower
                // (0.5: at 0.72 it ran wide into Tamburello's gravel every
                // lap; at 0.5 it laps Imola clean - EVOVIEW_LINE_K overrides)
                // 0.5 was tuned while the explicit path pushed the tyres 6 cm
                // over the patch (~12% less load transfer); with the force at
                // the patch it clears Imola at 0.43. The ODE modes also carry
                // the hubs' sideways mass: links (the geometry's roll centres
                // and camber) laps clean at 0.38; the slider (no camber from a
                // linkage) ran wide into Villeneuve (MX-5) and out of Variante
                // Alta (M4) until 0.36. Explicit was 0.43 until its hubs rode with
                // the body (rigcar hub_ride): now it corners like the slider
                let k_rig = match std::env::var("EVOVIEW_RIG_ODE").as_deref() {
                    Ok("links") | Err(_) => 0.38,
                    _ => 0.36,
                };
                let k = if td.rig.is_some() { std::env::var("EVOVIEW_LINE_K").ok().and_then(|v| v.parse().ok()).unwrap_or(k_rig) } else { 0.85 };
                // over a crest the road falls away under the car and takes
                // load off the tyres (v^2 x the vertical curvature): Mount
                // Panorama's Skyline (7 -> 13 % down in ~40 m) at 135 km/h
                // is ~20 % lighter, and the MX-5 ran wide into the Esses.
                // v^2 = k mu g r / (1 + k mu r kv) (EVOVIEW_LINE_CREST=0: off)
                let kv = if td.rig.is_some() && std::env::var("EVOVIEW_LINE_CREST").map_or(true, |v| v != "0") {
                    let (q0, q1, q2) = (self.line[j], self.line[(j + 12) % n], self.line[(j + 24) % n]);
                    let h = |a: Vec3, b: Vec3| Vec2::new(b.x - a.x, b.z - a.z).length().max(0.5);
                    let (d1, d2) = (h(q0, q1), h(q1, q2));
                    (((q1.y - q0.y) / d1 - (q2.y - q1.y) / d2) / (0.5 * (d1 + d2))).max(0.0)
                } else {
                    0.0
                };
                let mut vr = (k * td.grip_estimate() * G * r / (1.0 + k * td.grip_estimate() * r * kv)).sqrt() * self.pace * s_f;
                // in the wet the grip falls with speed (the tyre model's
                // water term): settle on the speed that grip allows
                let w = crate::rain::track_wetness();
                let wet_f = |v: f32| (1.0 - (1.0 - td.groove()) * crate::sctm::WET_K * w * v).clamp(0.15, 1.0) * crate::rain::rubber_grip(w);
                if w > 0.0 {
                    for _ in 0..4 {
                        vr = (k * td.grip_estimate() * wet_f(vr) * G * r).sqrt() * self.pace * s_f;
                    }
                }
                let mut brake_g = if w > 0.0 { 6.5 * wet_f(v) } else { 6.5 } * self.brake_k;
                // downhill, gravity pulls against the brakes: Mount
                // Panorama's drop from Skyline into the Esses (~13 %) left
                // the MX-5 5-8 m wide at full lock into the tyre stacks
                // (rigid-body car; EVOVIEW_LINE_SLOPE=0: flat braking)
                if td.rig.is_some() && d > 5.0 && std::env::var("EVOVIEW_LINE_SLOPE").map_or(true, |v| v != "0") {
                    let drop = (self.line[self.i].y - self.line[j].y) / d;
                    brake_g = (brake_g - G * drop.max(0.0)).max(2.0);
                }
                let reach = (vr * vr + 2.0 * brake_g * d).sqrt();
                vmax = vmax.min(reach);
            }
            d += self.line[j].distance(self.line[(j + 1) % n]);
            j = (j + 1) % n;
        }
        let mut vmax = vmax.min(self.vcap);
        // off the line or off the road: slow down to get back. A car wide of
        // the line keeps the line's speed and the steering's limit, so it
        // only ran wider; one in the gravel spun its wheels up through the
        // box and sat there in 6th (EVOVIEW_LINE_RECOVER=0: off)
        let mut thr_max = if self.rejoin > 0.0 || lost { 0.5 } else { 1.0 };
        if lost {
            vmax = vmax.min(10.0);
        }
        if recover() {
            let o = self.line[self.i] + self.line_left(self.i) * self.offset;
            let err = Vec3::new(o.x - td.pos.x, 0.0, o.z - td.pos.z).length();
            if err > 2.5 {
                vmax *= (1.0 - 0.06 * (err - 2.5)).max(0.4);
            }
            if td.snd.grip_f.min(td.snd.grip_r) < 0.85 {
                vmax = vmax.min(18.0);
                thr_max = 0.5;
            }
        }
        // understeer: the fronts saturated (past ~9 deg and 3 more than
        // the rears) - lift, and brake a little past 13 deg, as a driver
        // does when the wheel goes light (EVOVIEW_LINE_LIFT=0: off)
        if std::env::var("EVOVIEW_LINE_LIFT").map_or(true, |v| v != "0") {
            if let Some((rc, _)) = td.rig.as_ref().filter(|_| v > 8.0) {
                let af = (0.5 * (rc.c[0].alpha + rc.c[1].alpha)).abs().to_degrees();
                let ar = (0.5 * (rc.c[2].alpha + rc.c[3].alpha)).abs().to_degrees();
                if af > 9.0 && af > ar + 3.0 {
                    vmax = vmax.min(v - 0.5 - 0.5 * (af - 11.0).max(0.0));
                }
            }
        }
        let over = v - vmax;
        let c = Controls {
            throttle: if over < -2.0 { thr_max } else if over < 0.0 { 0.4f32.min(thr_max) } else { 0.0 },
            brake: if over > 1.0 { (over / 6.0).clamp(0.3, 1.0) } else { 0.0 },
            steer: (steer_angle / td.max_steer.max(0.1)).clamp(-1.0, 1.0),
            handbrake: false,
            // the angle is a real road-wheel angle: the rigid-body car takes it
            // as is (the keyboard path would cut it to 25 % at speed)
            analog: td.rig.is_some(),
            ..Default::default()
        };
        let mut c = c;
        // a driver's right foot (the rigid-body car): the throttle was
        // all-or-nothing on the speed alone, and an MX-5 floored mid-way
        // through Imola's Variante Alta (2nd gear, ~1.5 g, inside front in
        // the air) lit up its rears and spun every lap. Ease off as the rear
        // slides (all of it to 5 deg of slip, none from 10) and keep some of
        // the tyres for the corner (friction circle on the lateral g asked
        // of them); EVOVIEW_LINE_FOOT=0: the old foot
        if std::env::var("EVOVIEW_LINE_FOOT").map_or(true, |v| v != "0") {
            if let Some((rc, _)) = td.rig.as_ref() {
                let ar = rc.c[2].alpha.abs().max(rc.c[3].alpha.abs()).to_degrees();
                let slide = ((10.0 - ar) / 5.0).clamp(0.0, 1.0);
                let ay = (v * td.r).abs() / (G * td.grip_estimate().max(0.5));
                let circle = (1.0 - 0.7 * ay * ay).max(0.3);
                c.throttle = c.throttle.min(slide.min(circle));
                // and the left foot: braking hard into Brands Hatch's
                // Clearways while already turning (0.9 g) locked the MX-5's
                // rears (9-16 deg of slide every lap; the third spun into
                // the wall). Trail off the brake as the corner loads the
                // tyres and as the rear steps out; never below 0.3 of it
                let slide_b = ((10.0 - ar) / 6.0).clamp(0.3, 1.0);
                let circle_b = (1.0 - ay * ay).max(0.0).sqrt().max(0.3);
                c.brake = c.brake.min(slide_b.min(circle_b).max(if c.brake > 0.0 { 0.3 } else { 0.0 }));
            }
        }
        // a spun car going backwards: stop first (it held full throttle at
        // full lock, rolling back, and wheelspun donuts in the gravel)
        // Only slow (under ~29 km/h back): braking a fast backwards slide
        // locked it straight into the barrier (fast reverse wall hits 24 ->
        // 138); there it lifts and lets the car come round
        // In reverse gear the automatic swaps the pedals (the brake drives
        // it back): there the gas pedal is the brake - pressing the brake
        // held one car at a steady 30 km/h backwards for 110 s
        if td.vx < -2.0 {
            let stop = if td.vx > -8.0 { 1.0 } else { 0.0 };
            (c.throttle, c.brake) = if td.gear < 0 { (stop, 0.0) } else { (0.0, stop) };
        }
        // slow and on lock: feed the throttle in as a driver would coming
        // out of a spin (full lock at full throttle in a rear-drive GT3 is a
        // donut); full lock -> 30 %
        // - only while recovering (rejoining, 4 m off the line or facing
        // 45 deg off it): capped in every hairpin, the field bunched up
        let j = self.i;
        let tan = Vec3::new(self.line[(j + 3) % n].x - self.line[j].x, 0.0, self.line[(j + 3) % n].z - self.line[j].z).normalize_or_zero();
        let off = Vec3::new(td.pos.x - self.line[j].x, 0.0, td.pos.z - self.line[j].z).length();
        let recovering = self.rejoin > 0.0 || off > 4.0 + self.offset.abs() || td.forward().dot(tan) < 0.7;
        // Not from a standstill: the launch clutch only bites above ~2200
        // rpm, which 30 % never reached (148 cars sat in 1st, clutch open)
        if recovering && td.vx > 2.0 && td.vx < 15.0 && c.throttle > 0.3 {
            c.throttle = c.throttle.min(1.0 - 0.7 * c.steer.abs());
        }
        // pulling away from a standstill: at least 80 %. The launch clutch
        // bites only once the revs are up (~2200 rpm), and at the rejoin's /
        // the gravel's half throttle some engines (Mustang GT3) sat at idle -
        // the car never moved, the stuck check backed it out again, and each
        // round left it further off the line (107 'full lock' stuck cars)
        // (not in a queue: a car held behind another bumped into it)
        if td.vx.abs() < 2.0 && td.gear >= 0 && c.throttle > 0.25 && c.brake == 0.0 && self.vcap > 8.0 {
            c.throttle = c.throttle.max(0.8);
        }
        // pushing but not moving (nose in a wall, beached): back out
        if c.throttle > 0.25 && td.vx.abs() < 1.0 {
            self.stuck += dt;
            if self.stuck > 1.0 {
                self.stuck = 0.0;
                self.backing = 4.0;
                self.back_from = td.pos;
                // reversing with the wheels the other way swings the nose
                // toward where it wanted to go
                self.back_steer = if c.steer.abs() > 0.05 { -c.steer.signum() } else { 0.0 };
            }
        } else {
            self.stuck = 0.0;
        }
        c
    }

    fn a_b(&self, td: &TrackDrive) -> f32 {
        td.a + td.b
    }
}

/// EVOVIEW_KERB_ATTACK=<m>: the racing line moved `m` metres toward the
/// inside of every corner (negative: the outside), in proportion to how
/// tight it is - a driver that runs over the apex kerbs and chicane
/// sausages (or the exit kerbs) at speed, for the collide.rs event log.
/// EVOVIEW_LINE_WANDER=<a>: plus a +-a m weave with a 400 m period, which
/// crosses the track edges at a shallow angle (off onto the verge and back).
fn kerb_attack(line: Vec<Vec3>, m: f32, wander: f32) -> Vec<Vec3> {
    let n = line.len();
    if n < 16 {
        return line;
    }
    let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
    // signed curvature over ~+-12 m, + = turning left (toward +left)
    let mut k = vec![0.0f32; n];
    for i in 0..n {
        let (mut a, mut b) = (i, i);
        let (mut da, mut db) = (0.0, 0.0);
        while da < 12.0 && (i + n - a) % n < n / 4 {
            let j = (a + n - 1) % n;
            da += line[a].distance(line[j]);
            a = j;
        }
        while db < 12.0 && (b + n - i) % n < n / 4 {
            let j = (b + 1) % n;
            db += line[b].distance(line[j]);
            b = j;
        }
        let (u, v) = (flat(line[i] - line[a]).normalize_or_zero(), flat(line[b] - line[i]).normalize_or_zero());
        let cross = u.z * v.x - u.x * v.z;
        k[i] = cross.atan2(u.dot(v)) / (da + db).max(1.0);
    }
    let mut s = 0.0f32;
    (0..n).map(|i| {
        s += line[i].distance(line[(i + n - 1) % n]);
        let (p0, p1) = (line[(i + n - 1) % n], line[(i + 1) % n]);
        let t = flat(p1 - p0).normalize_or_zero();
        // left of the direction of travel
        let left = Vec3::new(t.z, 0.0, -t.x);
        // full offset from a 60 m radius in
        let w = (k[i].abs() * 60.0).min(1.0);
        line[i] + left * (k[i].signum() * m * w + wander * (s * std::f32::consts::TAU / 400.0).sin())
    }).collect()
}

/// `--drive-test SECONDS`: drive a scripted run headless (full throttle,
/// a gentle weave, a stop) and print the state every half second - a check
/// of the physics without a keyboard.
pub fn test_run(td: &mut TrackDrive, secs: f32, mut driver: Option<LineDriver>, mut laps: Option<&mut crate::laps::Session>) {
    if std::env::var_os("EVOVIEW_RIG_PAD").is_some() {
        if let Some((rc, _)) = td.rig.as_mut() {
            if std::env::var_os("EVOVIEW_RIG_STEADY").is_some() {
                crate::rigcar::steady_test(rc);
                return;
            }
            if std::env::var_os("EVOVIEW_RIG_DONUT").is_some() {
                crate::rigcar::donut_test(rc);
                return;
            }
            if std::env::var_os("EVOVIEW_RIG_FFB").is_some() {
                crate::rigcar::ffb_table(rc);
                return;
            }
            if std::env::var_os("EVOVIEW_RIG_GEARTEST").is_some() {
                crate::rigcar::gear_test(rc);
            } else {
                crate::rigcar::pad_test(rc);
            }
            return;
        }
    }
    // the line inside the track limits the lap timer judges by
    if let (Some(d), Some(l)) = (driver.as_mut(), laps.as_ref()) {
        d.keep_inside(&l.track);
    }
    // EVOVIEW_PROBE_AT=x,z,dx,dz: the ground along a line from (x, z), every
    // 0.5 m for 10 m (surface, height) - what is under a wall hit
    if let Some(v) = std::env::var("EVOVIEW_PROBE_AT").ok().map(|s| s.split(',').filter_map(|t| t.trim().parse::<f32>().ok()).collect::<Vec<_>>()).filter(|v| v.len() == 4) {
        let d = Vec3::new(v[2], 0.0, v[3]).normalize_or_zero();
        for i in 0..=20 {
            let p = Vec3::new(v[0], 0.0, v[1]) + d * (i as f32 * 0.5);
            match td.ground.probe(p.x, p.z, 1e4) {
                Some(h) => println!("PROBE {:4.1} m ({:.1}, {:.1}): y {:.2} n.y {:.2} {}", i as f32 * 0.5, p.x, p.z, h.y, h.n.y, td.ground.surf_name(h.surf)),
                None => println!("PROBE {:4.1} m ({:.1}, {:.1}): nothing", i as f32 * 0.5, p.x, p.z),
            }
        }
    }
    if std::env::var_os("EVOVIEW_DRIVE_PROBE").is_some() {
        let f = td.forward();
        let l = td.left();
        for i in 0..24 {
            let row: String = (-6..=6).map(|j| {
                let p = td.pos + f * (i as f32 * 3.0) + l * (j as f32 * 1.5);
                match td.ground.probe(p.x, p.z, td.pos.y + 3.0) {
                    Some(h) if (h.y - td.pos.y).abs() < 0.6 => '.',
                    Some(h) if h.y > td.pos.y => '^',
                    Some(_) => 'v',
                    None => '#',
                }
            }).collect();
            println!("ahead {:3} m  {row}", i * 3);
        }
    }
    println!("params: mass {:.0} kg, a {:.2} b {:.2}, lock {:.1} deg, radius {:.3}, gears {:?} x {:.2}, idle {:.0} limiter {:.0}",
             td.mass, td.a, td.b, td.max_steer.to_degrees(), td.radius, td.gears, td.final_drive, td.idle, td.limiter);
    println!("drivetrain: {:.0} % of the drive at the front", td.front_share * 100.0);
    for (i, t) in td.tyres.iter().enumerate() {
        match t {
            Some(t) => println!("tyre {}: Dy0 {:.3} Dx0 {:.3} ls {:.2}/{:.2} Fz0 {:.0} N, peak {:.1} deg -> mu at rest {:.2}",
                                ["front", "rear"][i], t.dy0, t.dx0, t.ls_y, t.ls_x, t.fz0, t.peak_deg,
                                t.mu_y(td.mass * G * if i == 0 { td.b } else { td.a } / (td.a + td.b) * 0.5)),
            None => println!("tyre {}: no model (mu 1.25)", ["front", "rear"][i]),
        }
        if let Some(m) = t.and_then(|t| t.sctm) {
            let fz = td.mass * G * if i == 0 { td.b } else { td.a } / (td.a + td.b) * 0.5;
            println!("  sctm {m:?}");
            let peak = t.map_or(8f32.to_radians(), |t| t.peak());
            let mu = t.map_or(1.25, |t| t.mu_y(fz));
            for deg in [1.0f32, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0] {
                let a = deg.to_radians();
                let old = mu * fz * (1.4 * (2.07 / peak * a).atan()).sin();
                let o = m.solve(&crate::sctm::Input { fz, alpha: a, speed: 30.0, ..Default::default() });
                println!("  alpha {deg:4.0}: old {old:7.0} N  sctm {:7.0} N  mz {:6.1} trail {:.3}", o.fy, o.mz, o.trail);
            }
            for (a, sr) in [(-0.2f32, 0.06f32), (-0.2, 0.0), (0.0, 0.06), (0.2, 0.06), (-0.2, 0.17), (-2.0, 0.06)] {
                let o = m.solve(&crate::sctm::Input { fz: 4400.0, alpha: a.to_radians(), sr, speed: 10.0, ..Default::default() });
                println!("  combined a {a:+.1} sr {sr:.2}: fx {:7.0} fy {:7.0}", o.fx, o.fy);
            }
        }
    }
    // EVOVIEW_PROBE_AT="x,z[,top];...": what the ground probe finds there
    if let Ok(v) = std::env::var("EVOVIEW_PROBE_AT") {
        for p in v.split(';') {
            let v: Vec<f32> = p.split(',').filter_map(|a| a.trim().parse().ok()).collect();
            if v.len() >= 2 {
                let (x, z) = (v[0], v[1]);
                let cell = ((x / CELL).floor() as i32, (z / CELL).floor() as i32);
                let h = td.ground.probe(x, z, v.get(2).copied().unwrap_or(1e5));
                println!("PROBE ({x}, {z}): cell {cell:?} {} tris -> {:?} {}", td.ground.grid.get(&cell).map_or(0, |l| l.len()), h,
                         h.map_or("", |h| td.ground.surf_name(h.surf)));
            }
        }
    }
    let dt = 1.0 / 120.0;
    let mut t = 0.0f32;
    let mut next = 0.0f32;
    // EVOVIEW_LINE_NEAR="x,z": the racing line's points within 120 m of a spot
    if let (Some(d), Ok(v)) = (driver.as_ref(), std::env::var("EVOVIEW_LINE_NEAR")) {
        if let Some((x, z)) = v.split_once(',').and_then(|(a, b)| Some((a.parse::<f32>().ok()?, b.parse::<f32>().ok()?))) {
            for (i, p) in d.line.iter().enumerate() {
                if (p.x - x).powi(2) + (p.z - z).powi(2) < 120.0 * 120.0 {
                    println!("LINEPT {i} {:.1} {:.1} {:.1}", p.x, p.y, p.z);
                }
            }
        }
    }
    // the wheel force the game's FFB would send (ffb.rs), peak per print
    let (fset, mut ffb, mut fpeak) = (crate::ffb::Settings::from_env(), crate::ffb::Ffb::default(), 0.0f32);
    // the road effects' peaks per print: kerb, bump, slip, -, ABS and the offset
    let mut fxpk = [0.0f32; 6];
    println!("ffMult {:.2}, ffb gamma {:.2}", td.ffb_inputs().4, td.ffb_gamma());
    if let Some(p) = td.phys.as_ref() {
        println!("evophys: coilover F {:?}", p.coilover[0]);
        println!("evophys: coilover R {:?}", p.coilover[1]);
        println!("evophys: diff {:?}  brakes {:?}", p.diff, p.brakes);
        println!("evophys: gearbox {:?}", p.gearbox);
        println!("evophys: clutch {:?} (curve {} pts)", Clutch2 { inertia: p.clutch.inertia, max_torque: p.clutch.max_torque, auto: (p.clutch.auto_forced, p.clutch.auto_min_rpm, p.clutch.auto_max_rpm, p.clutch.auto_on_changes) }, p.clutch.curve.len());
    }
    let mut headcam = crate::headcam::Headless::from_env();
    // laps on the racing line: the driver's point passing back through the
    // line's start. Per lap: time, top speed, peak cornering g (1 s mean
    // of v x yaw rate), worst distance off the line
    let (mut prev_i, mut lap_t0, mut lap_n) = (usize::MAX, f32::NAN, 0);
    let (mut top, mut g_peak, mut off_max) = (0.0f32, 0.0f32, 0.0f32);
    let mut g_win: std::collections::VecDeque<f32> = Default::default();
    let mut bad = false;
    let mut probe = AidsProbe::from_env();
    while t < secs {
        let mut c = match driver.as_mut() {
            Some(d) => d.controls(td, dt),
            // EVOVIEW_DRIVE_SPIN: 4 s flat out, then handbrake + full lock
            // (a car sliding past 90 degrees must keep sliding, not stop)
            None if std::env::var_os("EVOVIEW_DRIVE_SPIN").is_some() => Controls {
                throttle: if t < 8.0 { 1.0 } else { 0.0 },
                brake: 0.0,
                steer: if (8.0..9.0).contains(&t) { 1.0 } else { 0.0 },
                handbrake: (8.0..8.6).contains(&t),
                analog: false,
                ..Default::default()
            },
            None => Controls {
                throttle: if t < secs * 0.7 { 1.0 } else { 0.0 },
                brake: if t >= secs * 0.75 { 1.0 } else { 0.0 },
                steer: (t * 0.5).sin() * 0.2,
                handbrake: false,
                analog: false,
                ..Default::default()
            },
        };
        // a scripted drive shifts for itself (EVOVIEW_AUTO_SHIFT=0: not)
        c.auto_box = std::env::var("EVOVIEW_AUTO_SHIFT").map_or(true, |v| v != "0");
        if let Some(p) = probe.as_mut() {
            p.controls(td, t, &mut c);
        }
        // EVOVIEW_DRIVE_START_XZ="x,z,run_up_m": start on the line the run-up
        // before the line point nearest (x, z) - to replay one spot of a long
        // track (the Nordschleife's out lap is ~11 min)
        if t == 0.0 {
            if let (Some(d), Some(v)) = (driver.as_mut(), std::env::var("EVOVIEW_DRIVE_START_XZ").ok()) {
                let n: Vec<f32> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                if n.len() >= 2 {
                    let len = d.line.len();
                    let near = (0..len).min_by(|&a, &b| {
                        let da = Vec2::new(d.line[a].x - n[0], d.line[a].z - n[1]).length();
                        let db = Vec2::new(d.line[b].x - n[0], d.line[b].z - n[1]).length();
                        da.total_cmp(&db)
                    }).unwrap_or(0);
                    let (mut j, mut back) = (near, 0.0);
                    while back < n.get(2).copied().unwrap_or(300.0) && j != (near + 1) % len {
                        let pj = (j + len - 1) % len;
                        back += (d.line[j] - d.line[pj]).length();
                        j = pj;
                    }
                    let p = d.line[j];
                    let q = d.line[(j + 3) % len];
                    td.place_from(p, (q.x - p.x).atan2(q.z - p.z), 3.0);
                    d.i = j;
                    println!("START at line point {j} ({:.1}, {:.1}, {:.1}), {back:.0} m before ({:.0}, {:.0})", p.x, p.y, p.z, n[0], n[1]);
                }
            }
        }
        td.step(&c, dt);
        // out of the world (Mount Panorama: a car slid off a bank under a
        // wall's height band, past the collision mesh's edge, fell for
        // kilometres and its numbers ran wild): put it back on the line,
        // like a game's reset to track
        if let Some(d) = driver.as_mut() {
            let p = d.line[d.i];
            if td.pos.y < p.y - 15.0 || !td.pos.is_finite() {
                let q = d.line[(d.i + 3) % d.line.len()];
                let yaw = (q.x - p.x).atan2(q.z - p.z);
                println!("FELL t {t:.1} at ({:.1}, {:.1}, {:.1}): back on the line at ({:.1}, {:.1}, {:.1})", td.pos.x, td.pos.y, td.pos.z, p.x, p.y, p.z);
                td.place_from(p, yaw, 3.0);
            }
        }
        if let Some(l) = laps.as_deref_mut() {
            l.step(td, false, dt);
        }
        if let Some(h) = headcam.as_mut() {
            h.step(td, dt);
        }
        if !(td.vx.is_finite() && td.vy.is_finite() && td.r.is_finite() && td.pos.is_finite()) {
            println!("LAP ⚠ non-finite state at t {t:.2}: v ({}, {}) r {} pos {:?}", td.vx, td.vy, td.r, td.pos);
            bad = true;
            break;
        }
        if let Some(d) = driver.as_ref() {
            let (i, n) = (d.progress(), d.line.len());
            top = top.max(td.speed_kmh());
            g_win.push_back(td.vx * td.r / G);
            if g_win.len() > 120 {
                g_win.pop_front();
            }
            g_peak = g_peak.max((g_win.iter().sum::<f32>() / g_win.len() as f32).abs());
            let off = d.line[i].distance(td.pos);
            off_max = off_max.max(off);
            if prev_i != usize::MAX && prev_i > n * 9 / 10 && i < n / 10 {
                if lap_t0.is_finite() {
                    lap_n += 1;
                    let lt = t - lap_t0;
                    println!("LAP {lap_n}: {}:{:06.3}  top {top:.1} km/h  peak {g_peak:.2} g  max {off_max:.1} m off the line", (lt / 60.0) as u32, lt % 60.0);
                    if let Some(ev) = td.rig.as_mut().and_then(|(rc, _)| rc.ev.as_mut()) {
                        println!("EVENTS lap {lap_n}: {}", ev.lap_line());
                    }
                } else {
                    println!("LAP 0 (out lap) ends at t {t:.2}");
                    if let Some(ev) = td.rig.as_mut().and_then(|(rc, _)| rc.ev.as_mut()) {
                        println!("EVENTS out lap: {}", ev.lap_line());
                    }
                }
                lap_t0 = t;
                (top, g_peak, off_max) = (0.0, 0.0, 0.0);
            }
            prev_i = i;
        }
        if let Some(p) = probe.as_mut() {
            p.record(td, t);
            if p.done {
                break;
            }
        }
        let (hub, st, om, kmh, ffm) = td.ffb_inputs();
        ffb.fx = td.rig.as_ref().map_or(0.0, |(rc, _)| rc.ffb_fx.offset(&fset));
        let fo = ffb.step(&fset, hub, st, om, kmh, ffm, td.ffb_gamma(), dt);
        if fo.force.abs() > fpeak.abs() {
            fpeak = fo.force;
        }
        if let Some((rc, _)) = td.rig.as_ref() {
            for k in 0..5 {
                fxpk[k] = fxpk[k].max(rc.ffb_fx.v[k].abs());
            }
            fxpk[5] = fxpk[5].max(ffb.fx.abs());
        }
        if t >= next {
            next += if std::env::var_os("EVOVIEW_DRIVE_SPIN").is_some() { 0.25 } else if std::env::var_os("EVOVIEW_FFB_LOG").is_some() { 0.1 } else { 0.5 };
            let sp = (td.vx * td.vx + td.vy * td.vy).sqrt() * 3.6;
            // how far off the racing line (m), from the driver's own line
            let off = driver.as_ref().map_or(f32::NAN, |d| {
                d.line.iter().map(|p| Vec3::new(p.x - td.pos.x, 0.0, p.z - td.pos.z).length()).fold(f32::MAX, f32::min)
            });
            println!("t {:5.1}  {:6.1} km/h (total {:5.1}, slide {:5.0} deg)  gear {:2}  {:5.0} rpm  pos ({:7.1} {:6.1} {:7.1})  yaw {:6.1}  ffb {:+.2} (Mz {:+5.0}, lat {:+.2} g)  line {:4.1} m",
                     t, td.speed_kmh(), sp, td.vy.atan2(td.vx).to_degrees(), td.gear, td.rpm,
                     td.pos.x, td.pos.y, td.pos.z, td.yaw.to_degrees(), fpeak, hub, td.vx * td.r / G, off);
            fpeak = 0.0;
            if std::env::var_os("EVOVIEW_FFB_FX_LOG").is_some() {
                println!("      road fx: kerb {:.2} bump {:.2} slip {:.2} abs {:.2} -> offset {:.2}", fxpk[0], fxpk[1], fxpk[2], fxpk[4], fxpk[5]);
            }
            fxpk = [0.0; 6];
        }
        t += dt;
    }
    if driver.is_some() {
        println!("LAPS {lap_n} complete{}; last partial {:.1} s: top {top:.1} km/h, peak {g_peak:.2} g", if bad { " (stopped: non-finite)" } else { "" }, t - lap_t0);
    }
    if let Some(ev) = td.rig.as_ref().and_then(|(rc, _)| rc.ev.as_ref()) {
        println!("EVENTS total over {t:.0} s: {}", ev.total_line());
    }
    if let Some(p) = probe.as_ref() {
        p.report(td);
    }
}

/// `EVOVIEW_AIDS_PROBE` for --drive-test: what the cockpit aids change.
/// - `brake[:v1]`: at the racing-line driver's first braking point (Imola:
///   Tamburello) the pedal goes to 1 and stays there down to v1 km/h
///   (default 120). Prints the slip-ratio trace and the stopping distance.
/// - `tc`: every corner exit (throttle back on below 160 km/h), the driven
///   wheels' peak / mean slip ratio over the next 3 s and the TC cut.
struct AidsProbe {
    brake: bool,
    v1: f32,
    /// braking: (start t, start pos, start km/h) once on
    on: Option<(f32, Vec3, f32)>,
    done: bool,
    trace_at: f32,
    /// per axle: slip-ratio sum, steps past -0.15 (locking), the worst
    sum: [f32; 2],
    n: u32,
    locked: [u32; 2],
    worst: [f32; 2],
    abs_steps: u32,
    /// tc: the exit being watched (start t, km/h, slip sum, steps, cut sum)
    exit: Option<(f32, f32, f32, u32, f32)>,
    /// (t, km/h, peak slip, mean slip, mean cut, km/h after 3 s)
    exits: Vec<(f32, f32, f32, f32, f32, f32)>,
    prev_thr: f32,
    /// tc: the standing start to 100 km/h (peak slip, slip sum, cut sum,
    /// steps, done)
    launch: (f32, f32, f32, u32, bool),
}

impl AidsProbe {
    fn from_env() -> Option<AidsProbe> {
        let v = std::env::var("EVOVIEW_AIDS_PROBE").ok()?;
        let (kind, arg) = v.split_once(':').map_or((v.as_str(), None), |(a, b)| (a, Some(b)));
        Some(AidsProbe {
            brake: kind == "brake",
            v1: arg.and_then(|a| a.parse().ok()).unwrap_or(120.0),
            on: None,
            done: false,
            trace_at: 0.0,
            sum: [0.0; 2],
            n: 0,
            locked: [0; 2],
            worst: [0.0; 2],
            abs_steps: 0,
            exit: None,
            exits: Vec::new(),
            prev_thr: 1.0,
            launch: (0.0, 0.0, 0.0, 0, false),
        })
    }

    fn controls(&mut self, td: &TrackDrive, t: f32, c: &mut Controls) {
        if !self.brake || self.done {
            return;
        }
        if self.on.is_none() && c.brake > 0.0 && t > 2.0 {
            self.on = Some((t, td.pos, td.speed_kmh()));
            println!("AIDS brake: on at t {t:.2} s, {:.1} km/h, pos ({:.1} {:.1})", td.speed_kmh(), td.pos.x, td.pos.z);
        }
        if self.on.is_some() {
            c.brake = 1.0;
            c.throttle = 0.0;
        }
    }

    fn record(&mut self, td: &TrackDrive, t: f32) {
        let Some((rc, _)) = td.rig.as_ref() else { return };
        // ours FR FL RR RL
        let sr = [rc.c[0].sr, rc.c[1].sr, rc.c[2].sr, rc.c[3].sr];
        if self.brake {
            let Some((t0, p0, v0)) = self.on.filter(|_| !self.done) else { return };
            for ax in 0..2 {
                for k in [ax * 2, ax * 2 + 1] {
                    self.sum[ax] += sr[k] * 0.5;
                    if sr[k] < -0.15 {
                        self.locked[ax] += 1;
                    }
                    self.worst[ax] = self.worst[ax].min(sr[k]);
                }
            }
            self.n += 1;
            if rc.abs_on && rc.abs.as_ref().map_or(false, |a| a.active) {
                self.abs_steps += 1;
            }
            if t >= self.trace_at {
                self.trace_at = t + 0.05;
                println!("AIDS trace t {:5.2} {:6.1} km/h  sr FL {:+.3} FR {:+.3} RL {:+.3} RR {:+.3}  abs {}",
                    t - t0, td.speed_kmh(), sr[1], sr[0], sr[3], sr[2],
                    rc.abs.as_ref().filter(|_| rc.abs_on).map_or("-".to_string(), |a| format!("{:.2} {:.2} {:.2} {:.2}", a.mult[0], a.mult[1], a.mult[2], a.mult[3])));
            }
            if td.speed_kmh() <= self.v1 || t - t0 > 8.0 {
                self.done = true;
                let n = self.n.max(1) as f32;
                let d = Vec3::new(td.pos.x - p0.x, 0.0, td.pos.z - p0.z).length();
                println!("AIDS brake result: bias {:.1} % front, ABS {}: {:.1} -> {:.1} km/h in {:.2} s, {:.1} m (mean {:.2} g); mean slip ratio front {:+.3} rear {:+.3}, worst front {:+.3} rear {:+.3}, locking (sr < -0.15) front {:.0} % rear {:.0} % of the stop; ABS cutting {:.0} %; slide {:.1} deg",
                    rc.brake_bias() * 100.0, crate::aids::value(rc, crate::aids::Knob::Abs), v0, td.speed_kmh(), t - t0, d,
                    (v0 - td.speed_kmh()) / 3.6 / (t - t0).max(1e-3) / 9.81,
                    self.sum[0] / n, self.sum[1] / n, self.worst[0], self.worst[1],
                    self.locked[0] as f32 / (2.0 * n) * 100.0, self.locked[1] as f32 / (2.0 * n) * 100.0,
                    self.abs_steps as f32 / n * 100.0, td.vy.atan2(td.vx).to_degrees());
            }
            return;
        }
        // tc: the driven wheels' worst spin once the throttle comes back on
        let share = rc.drive_share();
        let spin = (0..4).filter(|k| share[*k] > 0.0).map(|k| sr[k]).fold(f32::MIN, f32::max);
        if !self.launch.4 {
            let l = &mut self.launch;
            // from 30 km/h: below it the TC is asleep (the cars' minSpeedKmh)
            if td.speed_kmh() > 30.0 {
                l.0 = l.0.max(spin);
                l.1 += spin;
                l.2 += 1.0 - rc.tc_cut();
                l.3 += 1;
            }
            let slide = td.vy.atan2(td.vx.abs().max(0.1)).to_degrees().abs();
            // (the slide angle means nothing at walking pace)
            if td.speed_kmh() >= 100.0 || (slide > 30.0 && (td.vx * td.vx + td.vy * td.vy).sqrt() > 4.0) || t > 15.0 {
                l.4 = true;
                let n = l.3.max(1) as f32;
                println!("AIDS launch: TC {}: {} at t {t:.2} s; from 30 km/h: driven slip peak {:.3} mean {:.3}, TC cut mean {:.0} %",
                    crate::aids::value(rc, crate::aids::Knob::Tc),
                    if td.speed_kmh() >= 100.0 { "0-100 km/h".to_string() } else { format!("SPUN ({slide:.0} deg slide, {:.0} km/h)", td.speed_kmh()) },
                    l.0, l.1 / n, l.2 / n * 100.0);
            }
        }
        if self.exit.is_none() && rc.thr > 0.5 && self.prev_thr <= 0.5 && td.speed_kmh() < 160.0 && t > 2.0 {
            self.exit = Some((t, td.speed_kmh(), 0.0, 0, 0.0));
            self.worst = [0.0; 2];
        }
        self.prev_thr = rc.thr;
        if let Some((t0, v0, sum, n, cut)) = self.exit.as_mut() {
            self.worst[0] = self.worst[0].max(spin);
            *sum += spin;
            *n += 1;
            *cut += 1.0 - rc.tc_cut();
            if t - *t0 >= 3.0 {
                let nn = (*n).max(1) as f32;
                println!("AIDS exit t {:6.1} from {:5.1} km/h: driven slip peak {:.3} mean {:.3}, TC cut mean {:.0} %, {:5.1} km/h after 3 s",
                    *t0, *v0, self.worst[0], *sum / nn, *cut / nn * 100.0, td.speed_kmh());
                self.exits.push((*t0, *v0, self.worst[0], *sum / nn, *cut / nn, td.speed_kmh()));
                self.exit = None;
            }
        }
    }

    fn report(&self, td: &TrackDrive) {
        if self.brake || self.exits.is_empty() {
            return;
        }
        let n = self.exits.len() as f32;
        let mean = |f: fn(&(f32, f32, f32, f32, f32, f32)) -> f32| self.exits.iter().map(f).sum::<f32>() / n;
        let tc = td.rig.as_ref().map_or("-".to_string(), |(rc, _)| crate::aids::value(rc, crate::aids::Knob::Tc));
        println!("AIDS tc result: TC {tc}: {} exits, driven slip peak mean {:.3} (max {:.3}), slip mean {:.3}, TC cut mean {:.0} %, gain over 3 s {:.1} km/h mean",
            self.exits.len(), mean(|e| e.2), self.exits.iter().map(|e| e.2).fold(0.0, f32::max),
            mean(|e| e.3), mean(|e| e.4) * 100.0, mean(|e| e.5 - e.1));
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct Clutch2 { inertia: f32, max_torque: f32, auto: (bool, f32, f32, bool) }

/// The automatic box's downshift point: down a gear once the lower gear
/// would turn under this share of the limiter (above ~30 km/h). It used to
/// shift down only when the engine bogged, so the racing-line driver sat in
/// 5th at 82 km/h out of every corner. EVOVIEW_DOWNSHIFT=<f> sets it,
/// 0 = the old rule (simple model 0.62, rig car on bogging only).
pub fn downshift_at() -> f32 {
    use std::sync::OnceLock;
    static D: OnceLock<f32> = OnceLock::new();
    *D.get_or_init(|| match std::env::var("EVOVIEW_DOWNSHIFT").ok().and_then(|v| v.parse::<f32>().ok()) {
        Some(v) if v <= 0.0 => 0.62,
        Some(v) => v.clamp(0.3, 0.9),
        None => 0.8,
    })
}

/// EVOVIEW_DOWNSHIFT=0: the rig car's old bog-only downshift.
pub fn downshift_old() -> bool {
    std::env::var("EVOVIEW_DOWNSHIFT").map_or(false, |v| v.parse::<f32>().map_or(false, |f| f <= 0.0))
}

/// EVOVIEW_LINE_RECOVER=0: the racing-line driver keeps the line's speed
/// when it is off the line or the road (the old behaviour).
fn recover() -> bool {
    use std::sync::OnceLock;
    static R: OnceLock<bool> = OnceLock::new();
    *R.get_or_init(|| std::env::var("EVOVIEW_LINE_RECOVER").map_or(true, |v| v != "0"))
}
