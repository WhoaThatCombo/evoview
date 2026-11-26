//! Steering for a gamepad stick (and nothing else): the stick does not set a
//! steering angle, it asks for a share of the front tyres' grip.
//!
//! A thumb stick has ~1 cm of travel and no force coming back, so "stick x
//! full lock" is undrivable at speed: a few millimetres are several times
//! the tyres' peak slip angle. Here full stick puts the front tyres at
//! `target` (0.95) of their peak slip angle - the angle is measured from
//! where the front axle is actually going, so the lock shrinks with speed by
//! itself and is right for every car, load and surface. The peak comes
//! straight from the tyre model (sctm.rs `peak_tan`); nothing is learned.
//!
//! The design follows adam10603's Advanced Gamepad Assist for Assetto Corsa
//! (the constants named "AGA" below are its defaults), re-derived in
//! evoview's axes: steering, yaw and lateral velocity are all + LEFT.
//!
//! Runs at the physics rate inside RigCar::controls, so it does not depend
//! on the frame rate and a scripted stick trace gives the same car as a
//! real one (`EVOVIEW_RIG_PAD=1 EVOVIEW_RIG_STICK=1 --drive-test 1`).
//!
//! Tunables (read once per car):
//!   EVOVIEW_PAD_ASSIST   1     0 = raw: stick x full lock
//!   EVOVIEW_PAD_TARGET   0.95  front slip at full stick, x the peak slip angle
//!   EVOVIEW_PAD_FILTER   0.15  stick smoothing time constant, s (0 = none)
//!   EVOVIEW_PAD_DYNAMIC  0.575 how much of the target slip the limit may give
//!                              up when the front is already sliding into the
//!                              turn (AGA "dynamic limit reduction" 5)

/// What the assist reads from the car each physics step. Body axes: x left,
/// z forward; every angle and rate + left.
#[derive(Clone, Copy, Debug)]
pub struct Kin {
    /// CG velocity along the car and to its left, m/s
    pub vx: f32,
    pub vy: f32,
    /// yaw rate, rad/s (+ = turning left)
    pub r: f32,
    /// CG to the front / rear axle, m
    pub a: f32,
    pub b: f32,
    /// the road wheels' lock, rad
    pub lock: f32,
    /// the front tyres' peak slip angle now, rad (load-weighted)
    pub peak: f32,
    /// brake pedal 0..1
    pub brake: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub on: bool,
    pub target: f32,
    pub filter: f32,
    pub dynamic: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params { on: true, target: 0.95, filter: 0.15, dynamic: 0.575 }
    }
}

fn env(k: &str, d: f32) -> f32 {
    crate::envc::var(k).ok().and_then(|v| v.parse::<f32>().ok()).filter(|v| v.is_finite()).unwrap_or(d)
}

impl Params {
    pub fn from_env() -> Params {
        let d = Params::default();
        Params {
            on: crate::envc::var("EVOVIEW_PAD_ASSIST").map_or(true, |v| v != "0"),
            target: env("EVOVIEW_PAD_TARGET", d.target).clamp(0.3, 2.0),
            filter: env("EVOVIEW_PAD_FILTER", d.filter).clamp(0.0, 1.0),
            dynamic: env("EVOVIEW_PAD_DYNAMIC", d.dynamic).clamp(0.0, 1.0),
        }
    }
}

/// The limit's own smoothing, s (AGA rate 11: slower oscillates on turn-in,
/// faster lets tyre-load noise into the steering)
const LIMIT_TAU: f32 = 0.09;
/// The assist fades in over this front-axle speed, m/s, for a 2.5 m
/// wheelbase (below it: the full lock, for parking and hairpins)
const FADE: (f32, f32) = (2.0, 6.0);

/// An exponential approach with a small constant-speed floor so it arrives.
pub fn toward(x: f32, target: f32, dt: f32, tau: f32) -> f32 {
    if tau <= 1e-4 {
        return target;
    }
    let d = target - x;
    let step = d.abs() * (1.0 - (-dt / tau).exp()) + 0.03 * dt / tau;
    x + d.signum() * step.min(d.abs())
}

fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

#[derive(Clone, Copy, Debug)]
pub struct Assist {
    pub p: Params,
    /// the filtered stick, -1..1
    stick: f32,
    /// the smoothed steering limit, rad
    limit: f32,
    /// the side the stick was last on (+1 left)
    dir: f32,
    /// false until the first step after the pad took the car
    live: bool,
    /// the last step's working, for logs and tests: (limit rad, fade 0..1)
    pub dbg: (f32, f32),
}

impl Default for Assist {
    fn default() -> Self {
        Assist::new(Params::default())
    }
}

impl Assist {
    pub fn new(p: Params) -> Assist {
        Assist { p, stick: 0.0, limit: 0.0, dir: 1.0, live: false, dbg: (0.0, 0.0) }
    }

    /// Something else steers (keyboard, wheel, AI): start afresh next time.
    pub fn rest(&mut self) {
        self.live = false;
    }

    /// The steering limit for a stick on side `dir` (+1 left), rad, before
    /// smoothing: the angle that puts the fronts at the target slip.
    fn slip_limit(&self, k: &Kin, dir: f32) -> f32 {
        let alpha_t = k.peak * self.p.target;
        // where the front axle is going, against the car's heading
        let beta_f = (k.vy + k.r * k.a).atan2(k.vx.max(0.5));
        // going the stick's way already (a corner, a slide being caught):
        // that much more lock for the same slip. Going the other way (turn
        // in against a slide, a flick): less - but never below a share of
        // the target, or the stick would go dead
        let give = alpha_t * self.p.dynamic * (1.0 - 0.1 * k.brake.clamp(0.0, 1.0));
        (alpha_t + (beta_f * dir).max(-give)).clamp(0.0, k.lock)
    }

    /// One physics step: the stick (-1 right .. +1 left, after its deadzone
    /// and curve) to the road wheels' angle, rad (+ left).
    pub fn step(&mut self, stick: f32, k: &Kin, dt: f32) -> f32 {
        let stick = if stick.is_finite() { stick.clamp(-1.0, 1.0) } else { 0.0 };
        if !self.p.on {
            return stick * k.lock;
        }
        if !self.live {
            self.live = true;
            self.stick = stick;
            self.limit = k.lock;
        }
        // one NaN from the car must not latch
        if !(self.stick.is_finite() && self.limit.is_finite()) {
            self.stick = stick;
            self.limit = k.lock;
        }
        self.stick = toward(self.stick, stick, dt, self.p.filter);
        let s = self.stick;
        if s.abs() > 1e-3 {
            self.dir = s.signum();
        }
        // in from 2 to 6 m/s at the front axle (longer cars later); out
        // again as the car goes sideways (past ~70 deg) and backwards
        let scale = ((k.a + k.b) / 2.5).clamp(0.6, 1.6);
        let v_f = k.vx.hypot(k.vy + k.r * k.a);
        let fade = smoothstep((v_f - FADE.0 * scale) / ((FADE.1 - FADE.0) * scale)) * smoothstep(k.vx / v_f.max(0.1) / 0.3);
        let raw = self.slip_limit(k, self.dir);
        self.limit = toward(self.limit, raw, dt, LIMIT_TAU);
        let limit = k.lock + (self.limit - k.lock) * fade;
        self.dbg = (limit, fade);
        let out = s * limit;
        if out.is_finite() { out.clamp(-k.lock, k.lock) } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 0.003;

    fn kin(vx: f32) -> Kin {
        Kin { vx, vy: 0.0, r: 0.0, a: 1.3, b: 1.4, lock: 30f32.to_radians(), peak: 8f32.to_radians(), brake: 0.0 }
    }

    fn settle(a: &mut Assist, stick: f32, k: &Kin, secs: f32) -> f32 {
        let mut out = 0.0;
        for _ in 0..(secs / DT) as usize {
            out = a.step(stick, k, DT);
        }
        out
    }

    #[test]
    fn full_stick_is_the_target_slip_not_the_lock() {
        let mut a = Assist::default();
        let k = kin(30.0);
        // straight ahead at speed: 0.95 x 8 deg, to the left for + stick
        let out = settle(&mut a, 1.0, &k, 2.0);
        assert!((out.to_degrees() - 7.6).abs() < 0.02, "{}", out.to_degrees());
        let mut a = Assist::default();
        let out = settle(&mut a, -1.0, &k, 2.0);
        assert!((out.to_degrees() + 7.6).abs() < 0.02, "{}", out.to_degrees());
        // half stick: half of it
        let mut a = Assist::default();
        let out = settle(&mut a, 0.5, &k, 2.0);
        assert!((out.to_degrees() - 3.8).abs() < 0.02);
        // the target share is the knob
        let mut a = Assist::new(Params { target: 0.5, ..Default::default() });
        assert!((settle(&mut a, 1.0, &k, 2.0).to_degrees() - 4.0).abs() < 0.02);
    }

    #[test]
    fn the_limit_follows_the_front_axles_path() {
        // cornering left: the front axle moves 3 deg left of the heading
        // (yaw), so the same slip needs 3 deg more lock
        let mut k = kin(30.0);
        k.r = 3f32.to_radians().tan() * k.vx / k.a;
        let mut a = Assist::default();
        let out = settle(&mut a, 1.0, &k, 2.0);
        assert!((out.to_degrees() - 10.6).abs() < 0.03, "{}", out.to_degrees());
        // the front sliding right while the stick asks left (2 deg): the
        // slip is already there, 2 deg less lock
        let mut k = kin(30.0);
        k.vy = -(2f32.to_radians().tan()) * k.vx;
        let mut a = Assist::default();
        let out = settle(&mut a, 1.0, &k, 2.0);
        assert!((out.to_degrees() - 5.6).abs() < 0.03, "{}", out.to_degrees());
        // ... but never less than (1 - dynamic) of the target: 7.6 x 0.425
        k.vy = -(20f32.to_radians().tan()) * k.vx;
        let mut a = Assist::default();
        let out = settle(&mut a, 1.0, &k, 2.0);
        assert!((out.to_degrees() - 7.6 * 0.425).abs() < 0.03, "{}", out.to_degrees());
        // mirrored: the same numbers to the right
        k.vy = -k.vy;
        let mut a = Assist::default();
        let out = settle(&mut a, -1.0, &k, 2.0);
        assert!((out.to_degrees() + 7.6 * 0.425).abs() < 0.03, "{}", out.to_degrees());
    }

    #[test]
    fn full_lock_when_slow_or_reversing() {
        let lock = 30.0;
        // parking speed, standing still, rolling backwards
        for vx in [0.0, 1.5, -8.0] {
            let mut a = Assist::default();
            let out = settle(&mut a, 1.0, &kin(vx), 2.0);
            assert!((out.to_degrees() - lock).abs() < 1e-3, "vx {vx}: {}", out.to_degrees());
        }
        // inside the fade: between the two, monotonic
        let mut last = lock + 1.0;
        for i in 0..=20 {
            let vx = 2.0 + i as f32 * 0.25;
            let mut a = Assist::default();
            let out = settle(&mut a, 1.0, &kin(vx), 2.0).to_degrees();
            assert!(out <= last + 1e-3 && out >= 7.6 - 0.05, "vx {vx}: {out} after {last}");
            last = out;
        }
        assert!((last - 7.6).abs() < 0.05, "{last}");
    }

    #[test]
    fn the_stick_is_filtered() {
        let k = kin(30.0);
        let mut a = Assist::default();
        a.step(0.0, &k, DT);
        // let the limit settle with the stick centred
        settle(&mut a, 0.0, &k, 1.0);
        // a step to full stick: ~63 % (and the floor's bit) after one time
        // constant, no overshoot, there within a second
        let mut hi = 0.0f32;
        let mut at_tau = 0.0;
        for i in 0..(1.0 / DT) as usize {
            let out = a.step(1.0, &k, DT).to_degrees();
            hi = hi.max(out);
            if i == (0.15 / DT) as usize {
                at_tau = out;
            }
        }
        assert!(at_tau > 7.6 * 0.60 && at_tau < 7.6 * 0.72, "{at_tau}");
        assert!((hi - 7.6).abs() < 0.02, "{hi}");
        // no filter: at once
        let mut a = Assist::new(Params { filter: 0.0, ..Default::default() });
        settle(&mut a, 0.0, &k, 1.0);
        assert!((a.step(1.0, &k, DT).to_degrees() - 7.6).abs() < 0.02);
    }

    #[test]
    fn never_past_the_lock_and_never_nan() {
        let mut a = Assist::default();
        let mut k = kin(40.0);
        k.peak = 60f32.to_radians();
        assert!(settle(&mut a, 1.0, &k, 2.0) <= k.lock + 1e-6);
        let k = Kin { vx: f32::NAN, vy: f32::NAN, r: f32::NAN, ..kin(30.0) };
        let mut a = Assist::default();
        assert!(a.step(f32::NAN, &k, DT).is_finite());
        assert!(a.step(1.0, &k, DT).is_finite());
    }

    #[test]
    fn off_is_the_raw_stick() {
        let mut a = Assist::new(Params { on: false, ..Default::default() });
        assert!((a.step(0.5, &kin(50.0), DT).to_degrees() - 15.0).abs() < 1e-4);
    }
}
