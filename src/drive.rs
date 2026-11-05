//! A keyboard drive model - not a simulator - that produces everything the
//! rig needs (carrig::Inputs) and the shared CarState (carstate.rs).
//!
//! The car stays on a treadmill: the body never travels, the wheels turn at
//! road speed and the body pitches and rolls on its springs.
//!
//! LONGITUDINAL  the engine's real torque: power curve x (1 + turbo boost) x
//!   the default map's power multiplier (carphys.rs, .carengine), through the
//!   real gear ratios and final drive (CarData f23) to the driven tyre's
//!   rolling radius (.tyre), traction-limited at 1.25 g; rpm locked to wheel
//!   speed through the same gearing, clamped to the real idle / limiter. Cars
//!   without a power curve fall back to a flat curve scaled to their mass.
//! BODY  pitch and roll are damped springs (zeta 0.35). Roll stiffness is
//!   sum over axles of (spring + ARB) * track^2 / 2, pitch stiffness
//!   2 k_f a^2 + 2 k_r b^2, both from CarData's wheel rates, track, wheelbase
//!   and CG share; inertia from the body box; CG height from tyre radius -
//!   base_y. Lateral acceleration is v^2 tan(steer) / wheelbase.
//! WHEELS  travel per corner = road input - body motion over that wheel,
//!   clamped to the bump / droop the coilover data allows. Steer: wheel angle
//!   = steering-wheel angle / steer ratio, split per side by Ackermann.

use crate::carrig::{Inputs, Kind, Rig};
use crate::carstate::CarState;
use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Default, Debug)]
pub struct Controls {
    pub throttle: bool,
    pub brake: bool,
    pub left: bool,
    pub right: bool,
    pub handbrake: bool,
    /// analogue overrides for scripts (demo, --check sweeps): -1 = unset
    pub throttle_v: f32,
    pub brake_v: f32,
    pub steer_v: Option<f32>,
}

impl Controls {
    pub fn none() -> Controls {
        Controls { throttle_v: -1.0, brake_v: -1.0, ..Default::default() }
    }
}

#[derive(Clone)]
pub struct Drive {
    pub cs: CarState,
    pub time: f32,
    steer: f32,
    throttle: f32,
    brake: f32,
    clutch: f32,
    pitch: f32,
    pitch_v: f32,
    roll: f32,
    roll_v: f32,
    accel_long: f32,
    accel_lat: f32,
    pub spin: [f32; 4],
    /// per-corner road input (m, + = the road pushes the wheel up)
    pub road: [f32; 4],
    shift: Option<(bool, f32)>,
    shift_len: f32,
    pub open: Vec<f32>,
    pub open_target: Vec<bool>,
    pub wiper: Option<f32>,
    pub wipers_on: bool,
    pub drs: bool,
    pub paddles: bool,
    // constants from the physics data
    mass: f32,
    radius: f32,
    k_roll: f32,
    k_pitch: f32,
    i_roll: f32,
    i_pitch: f32,
    h_cg: f32,
    cg_z: f32,
    torque: f32,
    /// the car's physics, for its real torque curve
    phys: Option<crate::carphys::Physics>,
    gears: Vec<f32>,
    reverse: f32,
    final_drive: f32,
    idle: f32,
    limiter: f32,
    lock_deg: f32,
    ratio: f32,
    wheelbase: f32,
    track: [f32; 2],
    travel: [(f32, f32); 4],
    corner_x: [f32; 4],
    corner_z: [f32; 4],
}

impl Drive {
    pub fn new(rig: &Rig) -> Drive {
        let p = rig.phys.as_ref();
        let mass = p.map_or(1200.0, |p| p.mass);
        let kf = p.map_or(60000.0, |p| p.coil[0].spring);
        let kr = p.map_or(60000.0, |p| p.coil[1].spring);
        let arbf = p.map_or(20000.0, |p| p.coil[0].arb);
        let arbr = p.map_or(15000.0, |p| p.coil[1].arb);
        let track = p.map_or([1.55, 1.55], |p| p.track);
        let track = [if track[0] > 0.8 { track[0] } else { 1.55 }, if track[1] > 0.8 { track[1] } else { 1.55 }];
        let wb = p.map(|p| p.wheelbase).filter(|w| *w > 1.0).unwrap_or(2.6);
        let cgf = p.map_or(0.5, |p| p.cg_front);
        let a = wb * (1.0 - cgf);
        let b = wb * cgf;
        let bx = p.map_or(Vec3::new(1.8, 1.3, 4.3), |p| p.body_box);
        // the gearing maths wants the physics' rolling radius, not the mesh
        let r = p.and_then(|p| p.tyre_radius).unwrap_or(rig.tyre_radius);
        let base_y = p.map_or(-0.2, |p| (p.base_y[0] + p.base_y[1]) * 0.5);
        let h_cg = (r - base_y).clamp(0.2, 0.8);
        // corner positions relative to the CG
        let mut corner_x = [0.0; 4];
        let mut corner_z = [0.0; 4];
        let mut travel = [(0.06, 0.06); 4];
        let mut zf = 0.0;
        let mut zr = 0.0;
        for c in &rig.corners {
            corner_x[c.idx] = c.centre.x;
            corner_z[c.idx] = c.centre.z;
            travel[c.idx] = c.travel;
            if c.front { zf = c.centre.z; } else { zr = c.centre.z; }
        }
        // the CG sits a and b from the axles
        let cg_z = if zf != 0.0 || zr != 0.0 { zf - (zf - zr) * (1.0 - cgf) } else { 0.0 };
        for z in corner_z.iter_mut() {
            *z -= cg_z;
        }
        if std::env::var_os("EVOVIEW_ATTITUDE_LOG").is_some() {
            eprintln!("rig springs: kf {kf:.0} kr {kr:.0} arb {arbf:.0}/{arbr:.0} N/m, track {:.2}/{:.2}, mass {mass:.0}, h_cg {h_cg:.2}, box {:?}",
                      track[0], track[1], bx);
        }
        let gears = p.map_or(vec![3.2, 2.1, 1.5, 1.2, 1.0, 0.85], |p| p.gears.clone());
        let paddles = rig.slot_of(&Kind::PaddleUp).is_some() || rig.slot_of(&Kind::DrvPaddleUp).is_some();
        let has_lever = rig.slot_of(&Kind::ShiftUp).is_some() || rig.slot_of(&Kind::DrvApproach).is_some();
        let mut d = Drive {
            cs: CarState::default(),
            time: 0.0,
            steer: 0.0,
            throttle: 0.0,
            brake: 0.0,
            clutch: 0.0,
            pitch: 0.0,
            pitch_v: 0.0,
            roll: 0.0,
            roll_v: 0.0,
            accel_long: 0.0,
            accel_lat: 0.0,
            spin: [0.0; 4],
            road: [0.0; 4],
            shift: None,
            shift_len: if paddles && !has_lever { 0.35 } else { rig.blend.h_in + rig.blend.h_play + rig.blend.h_out },
            open: vec![0.0; rig.slots.len()],
            open_target: vec![false; rig.slots.len()],
            wiper: None,
            wipers_on: false,
            drs: false,
            paddles: paddles && !has_lever,
            mass,
            radius: r,
            // a spring pair gives k t^2 / 2 of roll stiffness, an anti-roll
            // bar its full rate x t^2 (it was halved too: ~40 % too much roll)
            k_roll: (kf * 0.5 + arbf) * track[0] * track[0] + (kr * 0.5 + arbr) * track[1] * track[1],
            k_pitch: 2.0 * kf * a * a + 2.0 * kr * b * b,
            i_roll: mass * (bx.x * bx.x + bx.y * bx.y) / 12.0 * 0.7,
            i_pitch: mass * (bx.z * bx.z + bx.y * bx.y) / 12.0 * 0.7,
            h_cg,
            cg_z,
            torque: 220.0 * mass / 1300.0,
            phys: p.cloned(),
            gears,
            reverse: p.map_or(3.2, |p| p.reverse),
            final_drive: p.map_or(4.0, |p| p.final_drive),
            idle: p.map_or(900.0, |p| p.idle_rpm),
            limiter: p.map_or(7000.0, |p| p.limiter_rpm),
            lock_deg: p.map_or(360.0, |p| p.steer_lock_deg),
            ratio: p.map_or(14.0, |p| p.steer_ratio.abs()),
            wheelbase: wb,
            track,
            travel,
            corner_x,
            corner_z,
        };
        d.cs.rpm = d.idle;
        d.cs.rpm_limit = d.limiter;
        d.cs.gear = 0;
        d
    }

    /// Rolling radius the gearing maths uses, metres.
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Overall ratio of forward gear `g` (1-based) including the final drive.
    pub fn gear_ratio(&self, g: i32) -> Option<f32> {
        (g >= 1).then(|| self.gears.get((g - 1) as usize).map(|r| r * self.final_drive)).flatten()
    }

    pub fn limiter(&self) -> f32 {
        self.limiter
    }

    pub fn n_gears(&self) -> i32 {
        self.gears.len() as i32
    }

    pub fn shift_up(&mut self) {
        if self.cs.gear < self.n_gears() && self.shift.is_none() {
            self.cs.gear += 1;
            self.shift = Some((true, 0.0));
        }
    }

    pub fn shift_down(&mut self) {
        if self.cs.gear > -1 && self.shift.is_none() {
            self.cs.gear -= 1;
            self.shift = Some((false, 0.0));
        }
    }

    fn total_ratio(&self) -> f32 {
        match self.cs.gear {
            g if g > 0 => self.gears[(g - 1) as usize] * self.final_drive,
            -1 => -self.reverse * self.final_drive,
            _ => 0.0,
        }
    }

    /// Toggle an animated part (door, hood, sunroof...) by slot index.
    pub fn toggle(&mut self, slot: usize) {
        if let Some(t) = self.open_target.get_mut(slot) {
            *t = !*t;
        }
    }

    /// Advance the model by `dt` seconds.
    pub fn step(&mut self, rig: &Rig, c: &Controls, dt: f32) {
        let dt = dt.clamp(0.0, 0.05);
        self.time += dt;
        let approach = |v: &mut f32, target: f32, rate: f32| {
            let d = target - *v;
            *v += d.clamp(-rate * dt, rate * dt);
        };
        // pedals and steering, smoothed like a keyboard driver would
        let thr_t = if c.throttle_v >= 0.0 { c.throttle_v } else if c.throttle { 1.0 } else { 0.0 };
        let brk_t = if c.brake_v >= 0.0 { c.brake_v } else if c.brake { 1.0 } else { 0.0 };
        let st_t = c.steer_v.unwrap_or(if c.left { -1.0 } else if c.right { 1.0 } else { 0.0 });
        approach(&mut self.throttle, thr_t, 4.0);
        approach(&mut self.brake, brk_t, 5.0);
        approach(&mut self.steer, st_t, if c.steer_v.is_some() { 50.0 } else { 1.6 });
        self.cs.handbrake = if c.handbrake { (self.cs.handbrake + dt * 6.0).min(1.0) } else { (self.cs.handbrake - dt * 6.0).max(0.0) };

        // gear change: clutch in and throttle cut for its duration
        let mut thr = self.throttle;
        let mut clutch_t: f32 = 0.0;
        if let Some((up, t)) = self.shift.as_mut() {
            *t += dt;
            if !self.paddles {
                clutch_t = 1.0;
            }
            thr = 0.0;
            let _ = up;
            if *t > self.shift_len.max(0.25) {
                self.shift = None;
            }
        }
        let v = self.cs.speed_mps;
        if self.cs.gear != 0 && v.abs() < 2.0 && !self.paddles && thr > 0.05 {
            clutch_t = clutch_t.max(1.0 - v.abs() / 2.0);
        }
        if self.cs.gear == 0 || v.abs() < 0.3 && thr < 0.05 && self.cs.gear != 0 && !self.paddles {
            clutch_t = clutch_t.max(if self.cs.gear == 0 { 0.0 } else { 1.0 });
        }
        approach(&mut self.clutch, clutch_t, 8.0);

        // longitudinal
        let ratio = self.total_ratio();
        let r = self.radius;
        let wheel_rpm = v / r * 60.0 / std::f32::consts::TAU;
        let rpm_drive = (wheel_rpm * ratio).abs();
        let torque = self.phys.as_ref()
            .and_then(|p| p.engine_torque(self.cs.rpm, thr.max(0.01)).map(|t| t / thr.max(0.01)))
            .unwrap_or_else(|| self.torque
                * (0.75 + 0.25 * (std::f32::consts::PI * (self.cs.rpm / self.limiter).min(1.0)).sin()));
        let mut f = 0.0;
        if ratio != 0.0 && self.cs.rpm < self.limiter * 0.995 {
            // 85 % driveline efficiency; a slipping clutch passes ~70 %
            f += thr * torque * ratio / r * 0.85 * (1.0 - self.clutch * 0.3);
            // traction: past ~1.25 g the tyres spin instead of pushing
            let grip = 1.25 * 9.81 * self.mass;
            f = f.clamp(-grip, grip);
        }
        let brake_f = self.brake.max(self.cs.handbrake * 0.5) * 1.05 * 9.81 * self.mass;
        let drag = 0.5 * 1.2 * 0.75 * v * v.abs() + 0.012 * self.mass * 9.81 * v.signum();
        let engine_brake = if ratio != 0.0 && thr < 0.05 { rpm_drive / self.limiter * 900.0 * v.signum() } else { 0.0 };
        let mut a = (f - drag - engine_brake) / self.mass;
        if v.abs() > 0.05 || f.abs() > brake_f {
            a -= brake_f / self.mass * v.signum().max(if v == 0.0 { 0.0 } else { -1.0 });
        }
        let mut nv = v + a * dt;
        if v != 0.0 && nv.signum() != v.signum() && brake_f > 0.0 {
            nv = 0.0;
        }
        if v.abs() < 0.05 && f.abs() < 1.0 {
            nv = 0.0;
        }
        self.accel_long = (nv - v) / dt.max(1e-4);
        self.cs.speed_mps = nv;

        // engine speed: locked to the wheels through the clutch, free when not
        let free = self.idle + thr * (self.limiter - self.idle) * 0.85;
        let locked = rpm_drive.max(self.idle);
        let target = if ratio == 0.0 { free } else { locked * (1.0 - self.clutch) + free.max(locked) * self.clutch };
        let target = target.clamp(self.idle * 0.95, self.limiter);
        let rate = if ratio == 0.0 || self.clutch > 0.5 { 9000.0 } else { 60000.0 };
        approach(&mut self.cs.rpm, target, rate);
        // the limiter bounces
        if self.cs.rpm >= self.limiter * 0.99 && thr > 0.5 {
            self.cs.rpm = self.limiter * (0.97 + 0.03 * ((self.time * 30.0).sin() * 0.5 + 0.5));
        }

        // steering
        let wheel_deg = self.steer * self.lock_deg;
        let avg = (wheel_deg / self.ratio).to_radians();
        // the tyres give up near 1.2 g: past that the car would slide, not lean
        self.accel_lat = (nv * nv * avg.tan() / self.wheelbase).clamp(-1.2 * 9.81, 1.2 * 9.81);
        self.cs.speed_mps = nv;

        self.settle(rig, dt, true);
        if self.brake > 0.2 {
            self.drs = false;
        }

        self.cs.throttle = self.throttle;
        self.cs.brake = self.brake;
        self.cs.clutch = self.clutch;
        self.cs.steer = self.steer;
        self.cs.abs_active = self.brake > 0.95 && nv.abs() > 5.0;
        self.cs.tc_active = thr > 0.95 && self.cs.gear == 1 && nv.abs() < 12.0;
        self.cs.indicator = if self.steer > 0.4 { 1 } else if self.steer < -0.4 { -1 } else { 0 };
    }


    /// Follow an outside physics model (the track drive, trackdrive.rs)
    /// instead of this file's own: pedals, steering, engine and the felt
    /// accelerations come in; the springs, spin and open / close ramps are
    /// this model's as usual.
    pub fn follow(&mut self, rig: &Rig, f: &crate::trackdrive::RigFeed, handbrake: bool, dt: f32) {
        let dt = dt.clamp(1e-4, 0.05);
        self.time += dt;
        if f.gear != self.cs.gear && self.cs.gear != 0 && f.gear > 0 {
            self.shift = Some((f.gear > self.cs.gear, 0.0));
        }
        if let Some((_, t)) = self.shift.as_mut() {
            *t += dt;
            if *t > self.shift_len.max(0.25) {
                self.shift = None;
            }
        }
        self.throttle = f.throttle;
        self.brake = f.brake;
        // the keyboard steers in steps; the rig's wheel follows smoothly
        let d = f.steer - self.steer;
        self.steer += d.clamp(-6.0 * dt, 6.0 * dt);
        self.cs.speed_mps = f.speed;
        self.cs.rpm = f.rpm;
        self.cs.gear = f.gear;
        self.cs.throttle = f.throttle;
        self.cs.brake = f.brake;
        self.cs.steer = self.steer;
        self.cs.handbrake = if handbrake { 1.0 } else { 0.0 };
        self.cs.abs_active = f.brake > 0.95 && f.speed.abs() > 5.0;
        self.accel_long = f.a_long.clamp(-30.0, 30.0);
        self.accel_lat = f.a_lat.clamp(-1.6 * 9.81, 1.6 * 9.81);
        self.settle(rig, dt, true);
    }

    /// Body pitch / roll on its springs, wheel spin and the animated parts'
    /// open / close ramps - shared by the keyboard model and the replay feed.
    fn settle(&mut self, rig: &Rig, dt: f32, spin: bool) {
        let approach = |v: &mut f32, target: f32, rate: f32| {
            let d = target - *v;
            *v += d.clamp(-rate * dt, rate * dt);
        };
        // body pitch / roll
        // the body rolls about its roll axis (~0.11 m up), not the ground
        let m_roll = -self.mass * self.accel_lat * (self.h_cg - 0.11).max(0.15);
        let m_pitch = self.mass * self.accel_long.clamp(-15.0, 15.0) * self.h_cg;
        // sub-step: stiff race springs ring at 3-4 Hz
        let n = 4;
        let sub = 1.0 / n as f32;
        for _ in 0..n {
            let (mut r, mut rv) = (self.roll, self.roll_v);
            let mr = m_roll;
            let k = self.k_roll.max(1000.0);
            let i = self.i_roll;
            // dampers: ~0.65 of critical (0.35 wallowed like worn shocks)
            let c = 2.0 * 0.65 * (k * i).sqrt();
            rv += (mr - k * r - c * rv) / i * dt * sub;
            r += rv * dt * sub;
            self.roll = r;
            self.roll_v = rv;
            let (mut p, mut pv) = (self.pitch, self.pitch_v);
            let k = self.k_pitch.max(1000.0);
            let i = self.i_pitch;
            // dampers: ~0.65 of critical (0.35 wallowed like worn shocks)
            let c = 2.0 * 0.65 * (k * i).sqrt();
            pv += (m_pitch - k * p - c * pv) / i * dt * sub;
            p += pv * dt * sub;
            self.pitch = p;
            self.pitch_v = pv;
        }

        // wheels roll at road speed; a locked wheel stops
        let nv = self.cs.speed_mps;
        let w = nv / self.radius;
        for k in 0..4 {
            if !spin {
                break;
            }
            let locked = (self.brake > 0.97 && k < 2 && nv.abs() > 3.0) || (self.cs.handbrake > 0.5 && k >= 2);
            if !locked {
                self.spin[k] = (self.spin[k] + w * dt) % (std::f32::consts::TAU * 64.0);
            }
        }

        // animated parts ramp toward open / closed over their clip length
        for (i, s) in rig.slots.iter().enumerate() {
            let len = s.len_s().max(0.3);
            let tgt = if s.kind == Kind::Lights { self.cs.lights >= 2 } else { self.open_target[i] };
            approach(&mut self.open[i], if tgt { 1.0 } else { 0.0 }, 1.0 / len);
        }
        if self.wipers_on {
            self.wiper = Some(self.wiper.unwrap_or(0.0) + dt);
        } else if let Some(t) = self.wiper {
            // finish the sweep, then park
            let len = rig.slot_of(&Kind::Wiper).map_or(1.0, |i| rig.slots[i].len_s().max(0.1));
            let nt = t + dt;
            self.wiper = if (nt / len).floor() > (t / len).floor() { None } else { Some(nt) };
        }
    }

    /// (pitch, roll) in degrees, + = nose up / left side up.
    pub fn attitude_deg(&self) -> (f32, f32) {
        (self.pitch.to_degrees(), self.roll.to_degrees())
    }

    /// Body pitch / roll as a matrix about the CG, car space.
    pub fn body(&self) -> Mat4 {
        let c = Vec3::new(0.0, self.h_cg, self.cg_z);
        Mat4::from_translation(c)
            * Mat4::from_rotation_z(self.roll)
            * Mat4::from_rotation_x(-self.pitch)
            * Mat4::from_translation(-c)
    }

    /// The rig's inputs for the current state.
    pub fn inputs(&self, rig: &Rig) -> Inputs {
        let mut inp = Inputs::rest(rig.slots.len());
        inp.cs = self.cs;
        inp.time = self.time;
        inp.wheel_deg = self.steer * self.lock_deg;
        // Ackermann: the inside wheel turns tighter
        let avg = (inp.wheel_deg / self.ratio).to_radians();
        let (inner, outer) = if avg.abs() > 1e-4 {
            let rad = self.wheelbase / avg.abs().tan();
            let t = self.track[0] * 0.5;
            ((self.wheelbase / (rad - t).max(0.1)).atan(), (self.wheelbase / (rad + t)).atan())
        } else {
            (0.0, 0.0)
        };
        // corner 0 LF, 1 RF: a right turn (avg > 0) has the right wheel inside
        let s = avg.signum();
        inp.road = if avg > 0.0 { [outer * s, inner * s, 0.0, 0.0] } else { [inner * s, outer * s, 0.0, 0.0] };
        for k in 0..4 {
            // body height over this wheel: roll lifts +X when positive, pitch lifts +Z
            let y = self.roll * self.corner_x[k] + self.pitch * self.corner_z[k];
            let (droop, bump) = self.travel[k];
            inp.travel[k] = (self.road[k] - y).clamp(-droop, bump);
        }
        inp.spin = self.spin;
        inp.body = self.body();
        inp.open = self.open.clone();
        inp.wiper = self.wiper;
        inp.shift = self.shift;
        inp.paddles = self.paddles;
        inp.drs = self.drs;
        inp.g = Vec3::new(self.accel_lat, self.accel_long, 1.0 * 9.81) / 9.81;
        inp.water_c = 60.0 + 30.0 * (self.time / 60.0).min(1.0);
        inp.oil_c = 60.0 + 40.0 * (self.time / 90.0).min(1.0);
        inp
    }
}

// ------------------------------------------------------------------ demo --

/// The scripted demo: one loop of DEMO_LEN seconds that touches every input.
pub const DEMO_LEN: f32 = 32.0;

/// Phases of the demo loop, seconds. The launch is long enough for a GT3's
/// tall first gear: every stock car pulls through 1st to the limiter (and
/// usually 2nd and 3rd) with time to spare, so shift lights and lift-off
/// backfires get exercised (--check's demo_limiter holds the line).
pub mod demo {
    pub const LIGHTS: (f32, f32) = (0.5, 23.5);
    pub const WIPERS: (f32, f32) = (1.0, 5.0);
    pub const LAUNCH: (f32, f32) = (1.5, 9.0);
    pub const SLALOM: (f32, f32) = (9.0, 12.0);
    pub const STOP: (f32, f32) = (12.0, 17.0);
    pub const LOCK: (f32, f32) = (17.0, 21.5);
    pub const KERBS: [f32; 4] = [21.8, 22.4, 23.0, 23.6];
    pub const DOORS: (f32, f32) = (22.0, 24.3);
    pub const DRS_PULL: (f32, f32) = (24.5, 28.5);
    pub const DRS_STOP: (f32, f32) = (28.5, 31.0);
    /// key moments for stills: accelerating with wipers, doors open, DRS
    pub const SHOTS: [f32; 3] = [4.0, 23.3, 26.5];
}

fn within(t: f32, r: (f32, f32)) -> bool {
    t >= r.0 && t < r.1
}

/// Controls and one-shot events at time `t` of the demo loop.
pub fn demo_controls(t: f32) -> Controls {
    let t = t % DEMO_LEN;
    let mut c = Controls::none();
    if within(t, demo::LAUNCH) {
        c.throttle_v = 1.0;
    } else if within(t, demo::SLALOM) {
        c.throttle_v = 0.35;
        c.steer_v = Some(((t - demo::SLALOM.0) * std::f32::consts::PI).sin() * 0.35);
    } else if within(t, demo::STOP) {
        c.brake_v = 1.0;
    } else if within(t, demo::LOCK) {
        c.brake_v = 0.3;
        let span = demo::LOCK.1 - demo::LOCK.0;
        c.steer_v = Some(((t - demo::LOCK.0) / span * std::f32::consts::TAU).sin());
    } else if within(t, demo::DRS_PULL) {
        c.throttle_v = 0.9;
    } else if within(t, demo::DRS_STOP) {
        c.brake_v = 0.6;
    }
    c
}

/// Advance a Drive along the demo script from `from` to `to` seconds.
pub fn demo_run(d: &mut Drive, rig: &Rig, from: f32, to: f32) {
    let dt: f32 = 1.0 / 60.0;
    let mut t = from;
    while t < to - 1e-6 {
        let step = dt.min(to - t);
        let lt = t % DEMO_LEN;
        // events
        if lt < 0.02 {
            d.cs.gear = 1;
            d.cs.lights = 0;
        }
        d.cs.lights = if within(lt, demo::LIGHTS) { 2 } else { 0 };
        d.wipers_on = within(lt, demo::WIPERS);
        // shift at the limiter (after a moment on it), so the shift lights
        // reach their last stage and a lift-off backfire can fire
        let pulling = within(lt, demo::LAUNCH) || within(lt, demo::DRS_PULL);
        if pulling && d.cs.rpm > d.limiter * 0.985 && d.cs.gear < d.n_gears() {
            d.shift_up();
        }
        if (within(lt, demo::STOP) || within(lt, demo::DRS_STOP)) && d.cs.gear > 1 && d.cs.rpm < d.limiter * 0.45 {
            d.shift_down();
        }
        // kerbs: front-left, front-right, rear-left, rear-right
        let bump = |x: f32, at: f32| -> f32 {
            let u = (x - at) / 0.25;
            if (0.0..1.0).contains(&u) { (u * std::f32::consts::PI).sin() * 0.05 } else { 0.0 }
        };
        let k = demo::KERBS;
        d.road = [bump(lt, k[0]), bump(lt, k[1]), bump(lt, k[2]), bump(lt, k[3])];
        // doors, hood, tailgate and toggles open and shut together
        let open = within(lt, demo::DOORS);
        for i in rig.toggle_slots() {
            d.open_target[i] = open;
        }
        d.drs = within(lt, (demo::DRS_PULL.0 + 0.5, demo::DRS_PULL.1));
        let c = demo_controls(lt);
        d.step(rig, &c, step);
        t += step;
    }
}

// ---------------------------------------------------------------- replay --

/// Drives the rig from one car of a recorded .rpy instead of the keyboard:
/// rpm, steering-wheel angle, pedals, gear and per-wheel angular velocity come
/// from the base frames, suspension heave and twist from the world wheel
/// positions in the matrix frames. Pitch and roll on the springs are NOT in
/// the recording (the body basis is built from the wheels), so they still come
/// from the spring model, fed the replay's real accelerations.
pub struct ReplayFeed {
    pub rep: crate::replay::Replay,
    pub car: usize,
    st: ReplayRig,
    /// matrix-frame time per telemetry second (see `ReplayRig::new`)
    pub warp: f32,
}

/// One recorded car's rig state (rest heights, wheel spin, last speed),
/// without the replay itself: a replay's every car shares one `Replay`.
pub struct ReplayRig {
    pub car: usize,
    /// mean wheel and CoM heights in the car's own (wheel-plane) frame
    rest_wheel_y: [f32; 4],
    rest_com_y: f32,
    last_speed: Option<f32>,
    spin: [f32; 4],
    pub warp: f32,
}

fn local_heights(p: &crate::replay::Pose) -> ([f32; 4], f32) {
    // replay wheel order is FR, FL, RR, RL; ours is LF, RF, LR, RR
    let up = p.up.normalize_or_zero();
    let mid = (p.wheels[0] + p.wheels[1] + p.wheels[2] + p.wheels[3]) * 0.25;
    let h = |w: Vec3| (w - mid).dot(up);
    ([h(p.wheels[1]), h(p.wheels[0]), h(p.wheels[3]), h(p.wheels[2])], (p.pos - mid).dot(up))
}

impl ReplayFeed {
    pub fn new(rep: crate::replay::Replay, car: usize) -> ReplayFeed {
        let st = ReplayRig::new(&rep, car, true);
        let warp = st.warp;
        ReplayFeed { rep, car, st, warp }
    }

    pub fn length(&self) -> f32 {
        self.rep.cars.get(self.car).map_or(0.0, |c| c.n_matrix as f32 / self.rep.rate() / self.warp)
    }

    /// Set `d` to the replay at time `t` (seconds), advancing its spring
    /// model by `dt`.
    pub fn apply(&mut self, d: &mut Drive, rig: &Rig, t: f32, dt: f32) {
        self.st.apply(&self.rep, d, rig, t, dt);
    }
}

impl ReplayRig {
    /// `find_warp`: line the matrix frames' time up with the base frames'
    /// (below); off for a whole grid, whose cars are placed at the matrix
    /// frames' own time
    pub fn new(rep: &crate::replay::Replay, car: usize, find_warp: bool) -> ReplayRig {
        let n = rep.cars.get(car).map_or(0, |c| c.n_matrix as usize);
        let mut w = [0.0f32; 4];
        let mut c = 0.0f32;
        let mut k = 0.0f32;
        for f in (0..n).step_by(5) {
            let p = rep.pose(car, f);
            let (h, com) = local_heights(&p);
            for i in 0..4 {
                w[i] += h[i];
            }
            c += com;
            k += 1.0;
        }
        if k > 0.0 {
            for v in w.iter_mut() {
                *v /= k;
            }
            c /= k;
        }
        if !find_warp {
            return ReplayRig { car, rest_wheel_y: w, rest_com_y: c, last_speed: None, spin: [0.0; 4], warp: 1.0 };
        }
        // ⚠ The matrix frames do not always tick at the 60 ms replay.rs
        // assumes: on some sessions their speed curve runs ~20% fast and
        // early against the base frames' own timestamps (the Supra
        // Nurburgring time trial: 254 km/h peak at 50 s from positions, 210
        // at 58 s from the wheels). Find the time warp that best lines the
        // position-derived speed up with the wheel speed - scale-free
        // correlation, so the tyre radius does not matter.
        let span_tel = n as f32 / rep.rate();
        let wheel: Vec<f32> = (0..(span_tel * 2.0) as usize)
            .map(|i| rep.telemetry(car, i as f32 * 0.5).map_or(0.0, |t| t.wheel.iter().map(|w| w.abs()).sum::<f32>()))
            .collect();
        let corr = |k: f32| -> f32 {
            let m: Vec<f32> = (0..wheel.len()).map(|i| rep.speed(car, i as f32 * 0.5 * k)).collect();
            let n = m.len().max(1) as f32;
            let (ma, wa) = (m.iter().sum::<f32>() / n, wheel.iter().sum::<f32>() / n);
            let (mut c, mut vm, mut vw) = (0.0, 0.0, 0.0);
            for (a, b) in m.iter().zip(&wheel) {
                c += (a - ma) * (b - wa);
                vm += (a - ma) * (a - ma);
                vw += (b - wa) * (b - wa);
            }
            c / (vm * vw).sqrt().max(1e-6)
        };
        let mut best = (corr(1.0), 1.0f32);
        let mut k = 0.5f32;
        while k <= 1.5 {
            let c = corr(k);
            if c > best.0 + 1e-4 {
                best = (c, k);
            }
            k += 0.01;
        }
        let warp = if (best.1 - 1.0).abs() > 0.02 { best.1 } else { 1.0 };
        println!("replay feed: matrix time warp {warp:.2} (speed correlation {:.2})", best.0);
        ReplayRig { car, rest_wheel_y: w, rest_com_y: c, last_speed: None, spin: [0.0; 4], warp }
    }

    /// Set `d` to car `self.car` of `rep` at time `t` (seconds), advancing
    /// its spring model by `dt`.
    pub fn apply(&mut self, rep: &crate::replay::Replay, d: &mut Drive, rig: &Rig, t: f32, dt: f32) {
        let dt = dt.clamp(1e-4, 0.05);
        d.time = t;
        let speed = rep.speed(self.car, t * self.warp) * self.warp;
        let accel = self.last_speed.map_or(0.0, |s| (speed - s) / dt);
        self.last_speed = Some(speed);
        if let Some(tel) = rep.telemetry(self.car, t) {
            d.cs.rpm = tel.rpm;
            d.cs.throttle = tel.throttle;
            d.cs.brake = tel.brake;
            // byte 31: 0 R, 1 N, 2.. = 1st..
            d.cs.gear = tel.gear as i32 - 1;
            d.steer = (tel.steer_deg / d.lock_deg.max(1.0)).clamp(-1.0, 1.0);
            d.cs.steer = d.steer;
            d.accel_lat = tel.lat_g * 9.81;
            // recorded order FR, FL, RR, RL -> ours LF, RF, LR, RR
            let om = [tel.wheel[1], tel.wheel[0], tel.wheel[3], tel.wheel[2]];
            for k in 0..4 {
                self.spin[k] = (self.spin[k] + om[k] * dt) % (std::f32::consts::TAU * 64.0);
            }
            d.spin = self.spin;
        }
        d.cs.speed_mps = speed;
        d.accel_long = accel.clamp(-30.0, 30.0);
        if let Some(p) = rep.sample(self.car, t * self.warp) {
            let (h, com) = local_heights(&p);
            // the body sinking toward the wheel plane is bump on every corner;
            // a wheel standing proud of the plane is bump on that corner
            let heave = com - self.rest_com_y;
            for k in 0..4 {
                d.road[k] = (h[k] - self.rest_wheel_y[k]) - heave;
            }
        }
        d.settle(rig, dt, false);
    }
}
