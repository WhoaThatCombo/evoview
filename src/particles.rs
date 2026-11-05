//! CPU side of the car's particle effects: exhaust backfires, tyre smoke
//! and dust, simulated from the game's own emitter definitions
//! (`content/particles/**.particleemitter`, fxdata::EmitterDef).
//!
//! What is read and how it is used (ParticleEmitterData, Renderer.proto):
//! * emission: lifetime / speed / scale / roll-speed ranges (random per
//!   particle), `area_rotation` (the emission axis is local +Y rotated by
//!   it, then by the car's emitter transform), `axis_spread` (cone, rad),
//!   `area_size`, `burst_duration`, `gravity`.
//! * quality "high": `spawn_rate` (particles / s) and `capacity`.
//! * evolution over lifetime: alpha (curve), scale (end range; absolute when
//!   `over_absolute_value`), tint (curve4).
//! * evolution over strength: particles-per-second and alpha multipliers.
//!
//! Interpretation that the data does not spell out (documented guesses):
//! a curve-less "over lifetime" param lerps 1 -> random(end_range) over the
//! life; a tint curve whose G/B are zero is the material's
//! "Tint_TintRedChannelIsLifetime" channel (flipbook position), not a colour.

use crate::fxdata::{CarFx, EmitterDef, ET_BACKFIRE1, ET_CAR_DIRT, ET_CAR_GRASS, ET_CAR_SMOKE, ET_CAR_VAPOR,
                    ET_DEBRIS, ET_TYRE_DIRT, ET_TYRE_GRASS, ET_TYRE_SMOKE, ET_TYRE_SPLASH, ET_TYRE_VAPOR};

/// What one car is doing, for its effects. World-space particles: the car's
/// matrix places them, then they stay where they were laid down.
#[derive(Clone, Copy, Debug)]
pub struct CarIn {
    /// car mesh -> world
    pub xf: glam::Mat4,
    /// world velocity, m/s (debris inherits some of it)
    pub vel: glam::Vec3,
    /// tyre smoke 0..1 (the slide)
    pub smoke: f32,
    /// per wheel: car-space position, on grass 0..1, on dirt / gravel 0..1
    /// (a tyre emitter takes the nearest; all-zero positions = every wheel)
    pub wheels: [([f32; 3], f32, f32); 4],
    /// wet road 0..1 (tyre and body spray)
    pub wet: f32,
    /// body scraping the ground 0..1, and where (car space)
    pub scrape: f32,
    pub scrape_at: [f32; 3],
    pub throttle: f32,
    pub rpm: f32,
}

impl CarIn {
    pub fn at(xf: glam::Mat4, smoke: f32) -> CarIn {
        CarIn { xf, vel: glam::Vec3::ZERO, smoke, wheels: [([0.0; 3], 0.0, 0.0); 4], wet: 0.0,
                scrape: 0.0, scrape_at: [0.0; 3], throttle: 0.0, rpm: 0.0 }
    }
}
use crate::fxstate::FxState;

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub age: f32,
    pub life: f32,
    pub size0: f32,
    pub size_end: f32,
    pub alpha_end: f32,
    pub roll: f32,
    pub roll_speed: f32,
    pub strength: f32,
    pub def: usize,
    /// heavy particles (sparks, stones, debris: gravity down) bounce off
    /// this height - about the ground where they were thrown
    pub floor: f32,
    /// water mist (the tyre spray falls at 8 m/s^2): it settles on that
    /// floor and hangs there for its life instead of bouncing - bouncing
    /// cost it 30% of its life each time, so the trail died a car length
    /// behind the wheel
    pub settle: bool,
    /// the velocity it took from the car (a streak shows the rest)
    pub carried: [f32; 3],
}

/// One live instance as the GPU wants it: pos + size, roll / life / alpha,
/// tint.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Instance {
    pub pos_size: [f32; 4],
    pub misc: [f32; 4],
    pub tint: [f32; 4],
    /// xyz world velocity, w streak length in seconds of travel (0 = a
    /// round billboard): sparks are drawn as streaks along their motion,
    /// as the game draws them - a 2 cm dot is one pixel at 15 m
    pub vel: [f32; 4],
}

pub struct Sim {
    pub parts: Vec<Particle>,
    /// fractional spawn carry per (slot, def)
    carry: Vec<f32>,
    rng: u64,
    /// backfire currently playing: (car, slot index, emitter type, age)
    bursts: Vec<(usize, usize, u32, f32)>,
    last_backfire_age: f32,
    /// per car: last throttle (backfire on a lift)
    prev_throttle: Vec<f32>,
    /// the cars emitting. Empty = one car at the origin driven by FxState
    /// (showroom / driven car).
    pub cars: Vec<CarIn>,
    /// live particles per emitter def, kept by step / spawn_at (counting
    /// them per spawn was O(particles) each - with a field of AI cars
    /// smoking, thousands of spawns a frame)
    counts: Vec<usize>,
    /// EVOVIEW_SMOKE_RATE, read once a step
    more: f32,
}

impl Default for Sim {
    fn default() -> Self {
        Self { parts: Vec::new(), carry: Vec::new(), rng: 0x9E3779B97F4A7C15, bursts: Vec::new(),
               last_backfire_age: -1.0, prev_throttle: Vec::new(), cars: Vec::new(),
               counts: Vec::new(), more: 1.0 }
    }
}

fn rot_xyz(v: [f32; 3], deg: [f32; 3]) -> [f32; 3] {
    let (ax, ay, az) = (deg[0].to_radians(), deg[1].to_radians(), deg[2].to_radians());
    let v = [v[0], v[1] * ax.cos() - v[2] * ax.sin(), v[1] * ax.sin() + v[2] * ax.cos()];
    let v = [v[0] * ay.cos() + v[2] * ay.sin(), v[1], -v[0] * ay.sin() + v[2] * ay.cos()];
    [v[0] * az.cos() - v[1] * az.sin(), v[0] * az.sin() + v[1] * az.cos(), v[2]]
}

impl Sim {
    fn rand(&mut self) -> f32 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        ((self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
    fn range(&mut self, r: [f32; 2]) -> f32 {
        let t = self.rand();
        r[0] + (r[1] - r[0]) * t
    }

    /// Which backfire emitter type this car fires: the visual probabilities
    /// (`visual_backfire_type_N_probability`) pick one.
    fn pick_backfire(&mut self, fx: &CarFx) -> u32 {
        let p = fx.backfire.as_ref().map(|b| b.visual_prob).unwrap_or([1.0, 0.0, 0.0, 0.0, 0.0]);
        let total: f32 = p.iter().sum();
        if total <= 0.0 {
            return ET_BACKFIRE1;
        }
        let mut x = self.rand() * total;
        for (i, w) in p.iter().enumerate() {
            if x < *w {
                return ET_BACKFIRE1 + i as u32;
            }
            x -= w;
        }
        ET_BACKFIRE1
    }

    fn spawn(&mut self, fx: &CarFx, slot: usize, di: usize, strength: f32, xf: glam::Mat4) {
        self.spawn_at(fx, slot, di, strength, xf, None, glam::Vec3::ZERO);
    }

    /// `at`: car-space emission point instead of the slot's; `inherit`: world
    /// velocity added to the particle's own.
    fn spawn_at(&mut self, fx: &CarFx, slot: usize, di: usize, strength: f32, xf: glam::Mat4,
                at: Option<[f32; 3]>, inherit: glam::Vec3) {
        let def: &EmitterDef = &fx.emitter_defs[di];
        let s = &fx.emitters[slot];
        let cap = if def.capacity > 0 { def.capacity as usize } else { 200 };
        let more = self.more.max(1.0);
        // debris (sparks: 10 000 / s) at its own capacity per car - at x4 a
        // pack scraping together would fill the whole particle buffer
        let per_car = if def.path.to_lowercase().contains("debris") || def.path.to_lowercase().contains("spark") { 1 } else { 4 };
        if self.counts.len() <= di {
            self.counts.resize(di + 1, 0);
        }
        if self.counts[di] as f32 >= (cap * per_car * self.cars.len().max(1)) as f32 * more {
            return;
        }
        self.counts[di] += 1;
        let def = def.clone();
        // emission axis: local +Y, by area_rotation, then the slot's transform
        let spread = def.axis_spread;
        let (a, b) = (self.rand() * std::f32::consts::TAU, self.rand().sqrt() * spread);
        let local = [b.sin() * a.cos(), b.cos(), b.sin() * a.sin()];
        let dir = rot_xyz(rot_xyz(local, def.area_rotation), s.rot);
        let speed = self.range(def.speed);
        let off = [ (self.rand() - 0.5) * def.area_size, 0.0, (self.rand() - 0.5) * def.area_size ];
        let off = rot_xyz(rot_xyz(off, def.area_rotation), s.rot);
        let mut life = self.range(def.lifetime).max(0.005);
        if let Some(c) = fx.emitter_defs[di].pps_strength.first() {
            let _ = c;
        }
        let size0 = self.range(def.scale).max(0.005);
        let end = if def.scale_life.enable { self.range(def.scale_life.end_range) } else { 1.0 };
        let size_end = if def.scale_life.absolute { end } else { size0 * end };
        let alpha_end = if def.alpha_life.enable { self.range(def.alpha_life.end_range) } else { 1.0 };
        let roll_speed = self.range(def.roll_speed).to_radians();
        life = life.max(0.01);
        let roll = self.rand() * std::f32::consts::TAU;
        let sp = at.unwrap_or(s.pos);
        let pos = xf.transform_point3(glam::Vec3::new(sp[0] + off[0], sp[1] + off[1], sp[2] + off[2]));
        let vel = xf.transform_vector3(glam::Vec3::new(dir[0], dir[1], dir[2]) * speed) + inherit;
        // heavy ones land about where the tyre / body meets the ground
        let floor = if def.gravity[1] < -1.0 { pos.y - 0.06 } else { f32::MIN };
        let settle = def.material.water > 0.0;
        self.parts.push(Particle {
            pos: pos.into(),
            vel: vel.into(),
            age: 0.0, life, size0, size_end, alpha_end, roll, roll_speed, strength, def: di, floor, settle,
            carried: inherit.into(),
        });
    }

    /// Advance by `dt`: spawn what the state asks for, move and age the rest.
    pub fn step(&mut self, fx: &CarFx, st: &FxState, dt: f32) {
        let ns = fx.emitters.len();
        let nd = fx.emitter_defs.len().max(1);
        let showroom = self.cars.is_empty();
        self.more = crate::envc::var("EVOVIEW_SMOKE_RATE").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
        self.counts.clear();
        self.counts.resize(nd, 0);
        for p in &self.parts {
            if let Some(c) = self.counts.get_mut(p.def) {
                *c += 1;
            }
        }
        let cars: Vec<CarIn> = if showroom {
            let mut c = CarIn::at(glam::Mat4::IDENTITY, st.smoke);
            for w in c.wheels.iter_mut() {
                w.2 = st.dust;
            }
            c.throttle = st.throttle;
            c.rpm = st.rpm;
            vec![c]
        } else {
            self.cars.clone()
        };
        let nc = cars.len();
        if self.carry.len() != ns * nd * nc {
            self.carry = vec![0.0; ns * nd * nc];
        }
        if self.prev_throttle.len() != nc {
            self.prev_throttle = cars.iter().map(|c| c.throttle).collect();
        }
        // ---- backfire triggers ----
        let manual = st.backfire_age >= 0.0
            && (self.last_backfire_age < 0.0 || st.backfire_age < self.last_backfire_age);
        self.last_backfire_age = st.backfire_age;
        for ci in 0..nc {
            // auto: lifting off the throttle above low_rpms (the game also
            // randomises with delay / variation; one burst per lift here)
            let (thr, rpm) = (cars[ci].throttle, cars[ci].rpm);
            let lift = self.prev_throttle[ci] > 0.5 && thr < 0.1
                && fx.backfire.as_ref().map_or(false, |b| rpm >= b.low_rpms && rpm >= b.min_rpms);
            self.prev_throttle[ci] = thr;
            if lift || (manual && ci == 0) {
                let t = match st.backfire_type {
                    Some(k) => ET_BACKFIRE1 + k.clamp(1, 5) - 1,
                    None => self.pick_backfire(fx),
                };
                for (i, s) in fx.emitters.iter().enumerate() {
                    if (2..=4).contains(&s.kind) {
                        self.bursts.push((ci, i, t, 0.0));
                    }
                }
            }
        }
        let bursts = std::mem::take(&mut self.bursts);
        let mut keep = Vec::new();
        for (ci, slot, ty, age) in bursts {
            let Some(car) = cars.get(ci) else { continue };
            let s = &fx.emitters[slot];
            let Some(&(_, di)) = s.defs.iter().find(|(t, _)| *t == ty).or(s.defs.first()) else { continue };
            let def = &fx.emitter_defs[di];
            let dur = if def.burst > 0.0 { def.burst } else { 0.2 };
            if age < dur {
                let rate = if def.spawn_rate > 0.0 { def.spawn_rate } else { 200.0 };
                let k = (ci * ns + slot) * nd + di;
                self.carry[k] += rate * dt;
                while self.carry[k] >= 1.0 {
                    self.carry[k] -= 1.0;
                    self.spawn_at(fx, slot, di, 1.0, car.xf, None, car.vel * 0.8);
                }
                keep.push((ci, slot, ty, age + dt));
            }
        }
        self.bursts = keep;
        // ---- tyres and body ----
        for (ci, car) in cars.iter().enumerate() {
        let speed = car.vel.length();
        // grass / dirt: the slide throws it; rolling over it only a little.
        // ⚠ The tyre packs' stones fly at 10-20 m/s straight up (the slots
        // have no rotation): at full strength a wheel on the grass verge was
        // a fountain to the treetops.
        let kick = smoothstep(1.5, 14.0, speed) * 0.8 * car.smoke;
        let (mut grass_avg, mut dirt_avg) = (0.0f32, 0.0f32);
        for w in &car.wheels {
            grass_avg += w.1 * 0.25;
            dirt_avg += w.2 * 0.25;
        }
        let known = car.wheels.iter().any(|w| w.0 != [0.0; 3]);
        for (slot, s) in fx.emitters.iter().enumerate() {
            if s.kind != 1 && s.kind != 0 {
                continue;
            }
            let rear = s.pos[2] < 0.0;
            // the wheel this tyre emitter rides on
            let (wg, wd, wpos) = if s.kind == 1 && known {
                let mut best = (f32::MAX, 0.0, 0.0, s.pos);
                for w in &car.wheels {
                    let d = (w.0[0] - s.pos[0]).powi(2) + (w.0[2] - s.pos[2]).powi(2);
                    if d < best.0 {
                        best = (d, w.1, w.2, w.0);
                    }
                }
                (best.1, best.2, best.3)
            } else {
                (grass_avg, dirt_avg, s.pos)
            };
            // standing water under this tyre (the track shader's puddles)
            let puddle = if s.kind == 1 && car.wet > 0.0 {
                let p = car.xf.transform_point3(glam::Vec3::from(wpos));
                crate::rain::puddle(p.x, p.z) * car.wet
            } else {
                0.0
            };
            for &(ty, di) in &s.defs {
                let def_path = &fx.emitter_defs[di].path;
                let strength = match (s.kind, ty) {
                    (1, ET_TYRE_SMOKE) if rear => car.smoke,
                    (1, ET_TYRE_GRASS) => wg * kick,
                    (1, ET_TYRE_DIRT) => wd * kick,
                    (1, ET_TYRE_VAPOR) => car.wet * smoothstep(3.0, 20.0, speed),
                    (1, ET_TYRE_SPLASH) => puddle * smoothstep(1.5, 12.0, speed),
                    // the body's dust trail: the car actually off on it (about
                    // half its wheels), not one tyre brushing the verge
                    (0, ET_CAR_GRASS) => smoothstep(0.4, 0.9, grass_avg) * smoothstep(6.0, 25.0, speed),
                    (0, ET_CAR_DIRT) => smoothstep(0.4, 0.9, dirt_avg) * smoothstep(6.0, 25.0, speed),
                    (0, ET_CAR_VAPOR) => car.wet * smoothstep(8.0, 30.0, speed),
                    // scraping: the pack's sparks only (glass, carbon and the
                    // impact puff are crash debris)
                    // ⚠ The game ships sparks only in the body's Debris set
                    // (with glass, carbon, impact smoke): crash debris. Scrape
                    // sparks are evoview's own idea - off unless
                    // EVOVIEW_SCRAPE_SPARKS=1 (they looked out of place).
                    (0, ET_DEBRIS) if def_path.to_lowercase().contains("spark")
                        && crate::envc::var_os("EVOVIEW_SCRAPE_SPARKS").is_some() => car.scrape,
                    (0, ET_CAR_SMOKE) => 0.0,
                    _ => 0.0,
                };
                // EVOVIEW_FX_OFF=40,60,...: emitter types switched off (finding
                // which effect is which)
                // EVOVIEW_FX_OFF_PATH=detail,splash: emitters off by path
                if strength <= 0.0 {
                    continue;
                }
                let off_path = crate::envc::var("EVOVIEW_FX_OFF_PATH").map_or(false, |v| {
                    v.split(',').any(|w| !w.is_empty() && def_path.to_lowercase().contains(&w.to_lowercase()))
                });
                if strength <= 0.0 || fx_off(ty) || off_path {
                    continue;
                }
                let def = &fx.emitter_defs[di];
                // EVOVIEW_SMOKE_RATE: more (or less) smoke than the emitter's own
                let mut rate = if def.spawn_rate > 0.0 { def.spawn_rate } else { 60.0 } * self.more;
                if let Some(c) = def.pps_strength.first().filter(|c| !c.is_empty()) {
                    rate *= c.eval(strength).max(0.0);
                    // the pack's 10 000 sparks / s is a crash's shower; a
                    // splitter on a kerb throws a spray (a fraction, after the
                    // strength curve, which is ~flat above 0.035; 0.1)
                    if ty == ET_DEBRIS {
                        rate *= 0.1;
                    }
                } else {
                    rate *= strength;
                }
                let k = (ci * ns + slot) * nd + di;
                self.carry[k] += rate * dt;
                let (at, inherit) = if ty == ET_DEBRIS {
                    (Some(car.scrape_at), car.vel * 0.85)
                } else if s.kind == 0 {
                    (None, car.vel * 0.3)
                } else {
                    (None, glam::Vec3::ZERO)
                };
                while self.carry[k] >= 1.0 {
                    self.carry[k] -= 1.0;
                    self.spawn_at(fx, slot, di, strength, car.xf, at, inherit);
                }
            }
        }
        }
        // ---- integrate ----
        for p in &mut self.parts {
            let g = fx.emitter_defs[p.def].gravity;
            p.vel[0] += g[0] * dt;
            p.vel[1] += g[1] * dt;
            p.vel[2] += g[2] * dt;
            p.pos[0] += p.vel[0] * dt;
            p.pos[1] += p.vel[1] * dt;
            p.pos[2] += p.vel[2] * dt;
            if p.pos[1] < p.floor && p.vel[1] < 0.0 && p.settle {
                p.pos[1] = p.floor;
                p.vel[1] = 0.0;
            } else if p.pos[1] < p.floor && p.vel[1] < 0.0 {
                // a bounce, losing most of it to the road
                p.pos[1] = p.floor;
                p.vel[1] = -p.vel[1] * 0.3;
                p.vel[0] *= 0.6;
                p.vel[2] *= 0.6;
                p.age += (p.life - p.age) * 0.3;
            }
            p.roll += p.roll_speed * dt;
            p.age += dt;
        }
        self.parts.retain(|p| p.age < p.life);
    }

    /// Live particles of emitter def `di`, back to front from `eye`.
    pub fn instances(&self, fx: &CarFx, di: usize, eye: [f32; 3]) -> Vec<Instance> {
        let def = &fx.emitter_defs[di];
        // streak: ~60 ms of the spark's own flight (a camera's smear)
        let streak = if def.path.to_lowercase().contains("spark") { 0.06 } else { 0.0 };
        let mut v: Vec<(f32, Instance)> = self.parts.iter().filter(|p| p.def == di).map(|p| {
            let t = (p.age / p.life).clamp(0.0, 1.0);
            let size = match def.scale_life.at(t, 0.0, 0) {
                Some(_) if !def.scale_life.curve.iter().all(|c| c.is_empty()) => {
                    p.size0 * def.scale_life.curve[0].eval(t).max(0.0)
                }
                Some(_) => p.size0 + (p.size_end - p.size0) * t,
                None => p.size0,
            };
            let mut alpha = if def.alpha[1] > 0.0 { (def.alpha[0] + def.alpha[1]) * 0.5 } else { 1.0 };
            match def.alpha_life.curve.first().filter(|c| !c.is_empty()) {
                Some(c) if def.alpha_life.enable => alpha *= c.eval(t),
                _ if def.alpha_life.enable => alpha *= 1.0 + (p.alpha_end - 1.0) * t,
                _ => {}
            }
            if let Some(c) = def.alpha_strength.first().filter(|c| !c.is_empty()) {
                alpha *= c.eval(p.strength).clamp(0.0, 1.0);
            }
            let mut tint = [def.tint[0], def.tint[1], def.tint[2], 1.0];
            // a red-only tint is the lifetime channel too (smoke emitters
            // set tint (1, 0, 0) with Tint_TintRedChannelIsLifetime off)
            if def.material.tint_is_life || (def.tint[1] == 0.0 && def.tint[2] == 0.0) {
                tint = [1.0, 1.0, 1.0, 1.0];
            } else if def.tint_life.enable && def.tint_life.curve.len() >= 3 {
                let tc = &def.tint_life.curve;
                let colour = !(tc[1].0.iter().all(|q| q.1 == 0.0) && tc[2].0.iter().all(|q| q.1 == 0.0));
                if colour {
                    // the curves ARE the colour when the base tint is black
                    // (sparks: base (0, 0, 0.013), curves white-hot -> orange
                    // -> out; multiplied they came out a faint purple)
                    let base_dark = tint[0] + tint[1] + tint[2] < 0.1;
                    let k = if base_dark { [1.0; 3] } else { [tint[0], tint[1], tint[2]] };
                    tint = [k[0] * tc[0].eval(t), k[1] * tc[1].eval(t), k[2] * tc[2].eval(t), 1.0];
                }
            }
            let d = [p.pos[0] - eye[0], p.pos[1] - eye[1], p.pos[2] - eye[2]];
            let dist = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            (dist, Instance {
                pos_size: [p.pos[0], p.pos[1], p.pos[2], size * 0.5],
                misc: [p.roll, t, alpha.clamp(0.0, 1.0), 0.0],
                tint,
                vel: [p.vel[0] - p.carried[0], p.vel[1] - p.carried[1], p.vel[2] - p.carried[2], streak],
            })
        }).collect();
        v.sort_by(|a, b| b.0.total_cmp(&a.0));
        v.into_iter().map(|x| x.1).collect()
    }

    /// Run from rest to `t` seconds with a fixed step (headless shots).
    pub fn warm(&mut self, fx: &CarFx, st: &FxState, t: f32) {
        let dt = 1.0 / 120.0;
        let mut s = *st;
        let steps = (t / dt).ceil() as i32;
        let start_bf = st.backfire_age;
        for i in 0..steps {
            let now = i as f32 * dt;
            s.time = st.time - t + now;
            // a backfire `start_bf` seconds before the shot
            s.backfire_age = if start_bf >= 0.0 && now >= t - start_bf { now - (t - start_bf) } else { -1.0 };
            self.step(fx, &s, dt);
        }
    }
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn fx_off(ty: u32) -> bool {
    crate::envc::var("EVOVIEW_FX_OFF").map_or(false, |v| v.split(',').any(|x| x.trim().parse::<u32>().ok() == Some(ty)))
}
