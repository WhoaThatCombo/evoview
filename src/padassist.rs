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
//!   EVOVIEW_PAD_SELF_STEER   0.37  self-steer response (0 = off)
//!   EVOVIEW_PAD_DAMPING      0.37  yaw damping in the self-steer
//!   EVOVIEW_PAD_SELF_MAX     90    most self-steer, road-wheel degrees
//!   EVOVIEW_PAD_COUNTERSTEER 0.2   how far past the slide a full opposite
//!                                  stick may steer (0 = just to it; -1 = no
//!                                  countersteer limit)
//!   EVOVIEW_PAD_SELF_KNEE    30    the slide, deg, from which the self-steer
//!                                  follows it one to one (below: softened;
//!                                  AGA: 72)
//!   EVOVIEW_PAD_SELF_FILTER  0.04  the self-steer's smoothing, s (AGA: ~0.1)
//!   EVOVIEW_PAD_REAR_GUARD   2.5   degrees of lock given up per degree the
//!                                  rear slides past its own peak (0 = off)
//!   EVOVIEW_PAD_LEAD         0.2   how far ahead, s, the slide is read (its
//!                                  angle + its rate x this; 0 = as it is)
//!
//! Self-steer is what a wheel's force feedback does for free: let go and the
//! front wheels turn to where the car is going, so it straightens itself
//! and countersteers a slide. It has all of the steering at a centred stick
//! and none at a full one. The countersteer limit stops a panicked full
//! opposite stick from throwing the car the other way: the fronts go just
//! past the direction of travel, not to the far grip peak.
//!
//! The rear guard is evoview's own addition. "Fronts at their peak" is only
//! safe for a car that understeers at the limit; one whose rear lets go
//! first (the MX-5 NA here) spins slowly under a held full stick, where
//! the raw stick merely scrubbed its front tyres. So the lock is taken back
//! as the rear axle slides past its own peak slip angle - through zero and
//! into opposite lock if it must: a held full stick then means "the most
//! this car will turn", a steady small slide instead of a spin.
//!
//! All three read the slide a little ahead of time (its angle plus its rate
//! x EVOVIEW_PAD_LEAD): the stick filter and the limit's smoothing cost
//! ~0.2 s, and a slide that is collapsing must be let go of before it is
//! gone or the car swings through to the other side.
//!
//! Two of AGA's self-steer numbers are different here, both from the
//! scripted slides (rigcar.rs stick_test): its 0.1 s smoothing reached the
//! lock two tenths too late to hold a 25 deg slide the same car catches
//! with 0.04 s, and its curve (the full slide angle only at 72 deg) left
//! the fronts short of the slide - at their grip peak, still feeding the
//! spin - so from 30 deg on the wheels follow the slide one to one. A slide
//! wider than the car's lock cannot be caught by steering at all.

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
    /// the rear tyres'
    pub peak_r: f32,
    /// brake pedal 0..1
    pub brake: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub on: bool,
    pub target: f32,
    pub filter: f32,
    pub dynamic: f32,
    pub self_steer: f32,
    pub damping: f32,
    /// degrees
    pub self_max: f32,
    pub counter: f32,
    pub rear_guard: f32,
    /// degrees
    pub self_knee: f32,
    /// seconds
    pub lead: f32,
    /// seconds
    pub self_filter: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params { on: true, target: 0.95, filter: 0.15, dynamic: 0.575, self_steer: 0.37, damping: 0.37, self_max: 90.0, counter: 0.2, rear_guard: 2.5, self_knee: 30.0, lead: 0.2, self_filter: 0.04 }
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
            self_steer: env("EVOVIEW_PAD_SELF_STEER", d.self_steer).clamp(0.0, 1.0),
            damping: env("EVOVIEW_PAD_DAMPING", d.damping).clamp(0.0, 2.0),
            self_max: env("EVOVIEW_PAD_SELF_MAX", d.self_max).clamp(0.0, 90.0),
            counter: env("EVOVIEW_PAD_COUNTERSTEER", d.counter).clamp(-1.0, 1.0),
            rear_guard: env("EVOVIEW_PAD_REAR_GUARD", d.rear_guard).clamp(0.0, 5.0),
            self_knee: env("EVOVIEW_PAD_SELF_KNEE", d.self_knee).clamp(1.0, 90.0),
            lead: env("EVOVIEW_PAD_LEAD", d.lead).clamp(0.0, 1.0),
            self_filter: env("EVOVIEW_PAD_SELF_FILTER", d.self_filter).clamp(0.0, 1.0),
        }
    }
}

/// The limit's own smoothing, s (AGA rate 11: slower oscillates on turn-in,
/// faster lets tyre-load noise into the steering)
const LIMIT_TAU: f32 = 0.09;
/// The countersteer limit comes in over this much rear slide, rad (AGA
/// 4.5 to 10 deg), in 1 / 12 s and out three times slower, and only at
/// speed (10 to 20 m/s)
const COUNTER_SLIDE: (f32, f32) = (0.0785, 0.1745);
const COUNTER_TAU: f32 = 0.083;
const COUNTER_SPEED: (f32, f32) = (10.0, 20.0);
/// The slide rate's smoothing, s
const RATE_TAU: f32 = 0.03;
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
    /// the smoothed self-steer, as a share of the lock
    own: f32,
    /// 0..1: the stick is steering against a slide
    counter: f32,
    /// the rear axle's slide a step ago, rad, and its smoothed rate, rad/s
    slide_prev: f32,
    slide_rate: f32,
    /// false until the first step after the pad took the car
    live: bool,
    /// the last step's working, for logs and tests: (limit rad, fade 0..1,
    /// self-steer rad, countersteer 0..1)
    pub dbg: (f32, f32, f32, f32),
}

impl Default for Assist {
    fn default() -> Self {
        Assist::new(Params::default())
    }
}

impl Assist {
    pub fn new(p: Params) -> Assist {
        Assist { p, stick: 0.0, limit: 0.0, dir: 1.0, own: 0.0, counter: 0.0, slide_prev: 0.0, slide_rate: 0.0, live: false, dbg: (0.0, 0.0, 0.0, 0.0) }
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

    /// Where the front wheels would point by themselves, as a share of the
    /// lock: toward the rear axle's direction of travel - softened near the
    /// centre (AGA's curve), one to one from `self_knee` on - less a
    /// yaw-rate term that stops the car wobbling as it re-centres.
    fn self_steer(&self, k: &Kin, slide: f32) -> f32 {
        if self.p.self_steer <= 0.0 {
            return 0.0;
        }
        let lock = k.lock.to_degrees().max(1.0);
        let e = 1.0 + (1.0 - (10.0 * (0.9 * self.p.self_steer + 0.1)).log10());
        let knee = self.p.self_knee.max(1.0);
        let d = slide.to_degrees();
        let x = d / knee;
        let cap = self.p.self_max / lock;
        let own = (if x.abs() < 1.0 { x.signum() * x.abs().powf(e) * knee } else { d } / lock).clamp(-cap, cap);
        // + yaw is a left turn and + steer is left: the damping steers against it
        let damp = -k.r * self.p.damping * 0.15 * (30.0 / lock);
        (own + damp).clamp(-1.0, 1.0)
    }

    /// One physics step: the stick (-1 right .. +1 left, after its deadzone
    /// and curve) to the road wheels' angle, rad (+ left).
    pub fn step(&mut self, stick: f32, k: &Kin, dt: f32) -> f32 {
        let stick = if stick.is_finite() { stick.clamp(-1.0, 1.0) } else { 0.0 };
        if !self.p.on {
            return stick * k.lock;
        }
        // where the rear axle is going against the heading (+ left): the slide
        let beta_r = (k.vy - k.r * k.b).atan2(k.vx.max(0.5));
        // the pad just took the car - or one NaN from it, which must not latch
        let sane = self.stick.is_finite() && self.limit.is_finite() && self.own.is_finite() && self.counter.is_finite()
            && self.slide_prev.is_finite() && self.slide_rate.is_finite();
        if !self.live || !sane {
            self.live = true;
            self.stick = stick;
            self.limit = k.lock;
            self.own = 0.0;
            self.counter = 0.0;
            self.slide_prev = beta_r;
            self.slide_rate = 0.0;
        }
        // the slide a little ahead: its angle and where its rate takes it
        let rate = ((beta_r - self.slide_prev) / dt.max(1e-4)).clamp(-4.0, 4.0);
        self.slide_prev = beta_r;
        self.slide_rate += (rate - self.slide_rate) * (1.0 - (-dt / RATE_TAU).exp());
        let slide = (beta_r + self.p.lead * self.slide_rate).clamp(-1.5, 1.5);
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
        // the front's limit, smoothed (the peak moves with the tyres' load)
        self.limit = toward(self.limit, self.slip_limit(k, self.dir), dt, LIMIT_TAU);
        let mut limit = self.limit;
        // the stick points the way the rear is sliding: it is catching it.
        // Then the limit is the slide's own angle and a little more, not the
        // grip peak on the far side of it
        let against = s * slide > 0.0 && self.p.counter >= 0.0;
        let want = if against { ((slide.abs() - COUNTER_SLIDE.0) / (COUNTER_SLIDE.1 - COUNTER_SLIDE.0)).clamp(0.0, 1.0) } else { 0.0 };
        self.counter = toward(self.counter, want, dt, if want > self.counter { COUNTER_TAU } else { COUNTER_TAU * 3.0 });
        let at_speed = ((v_f - COUNTER_SPEED.0) / (COUNTER_SPEED.1 - COUNTER_SPEED.0)).clamp(0.0, 1.0);
        let cs = self.counter * at_speed;
        if cs > 0.0 {
            let catch = (k.peak * self.p.target * (self.p.counter.max(0.0) * 0.7 + 0.1) + slide.abs()).min(k.lock);
            limit += (catch - limit) * cs;
        }
        // the rear sliding outward under this stick, past its own peak: the
        // car is oversteering - less front, down to opposite lock
        let excess = (-slide * self.dir - k.peak_r).max(0.0);
        let limit = (limit - self.p.rear_guard * excess).clamp(-k.lock, k.lock);
        let limit = k.lock + (limit - k.lock) * fade;
        self.own = toward(self.own, self.self_steer(k, slide), dt, self.p.self_filter);
        let own = self.own * k.lock * (1.0 - s.abs()) * fade;
        self.dbg = (limit, fade, own, cs);
        let out = s * limit + own;
        if out.is_finite() { out.clamp(-k.lock, k.lock) } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 0.003;

    /// the limit and filter alone (the tests above the self-steer's)
    fn limit_only() -> Assist {
        Assist::new(Params { self_steer: 0.0, counter: -1.0, rear_guard: 0.0, ..Default::default() })
    }

    fn kin(vx: f32) -> Kin {
        Kin { vx, vy: 0.0, r: 0.0, a: 1.3, b: 1.4, lock: 30f32.to_radians(), peak: 8f32.to_radians(), peak_r: 8f32.to_radians(), brake: 0.0 }
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
        let mut a = limit_only();
        let out = settle(&mut a, 1.0, &k, 2.0);
        assert!((out.to_degrees() - 5.6).abs() < 0.03, "{}", out.to_degrees());
        // ... but never less than (1 - dynamic) of the target: 7.6 x 0.425
        k.vy = -(20f32.to_radians().tan()) * k.vx;
        let mut a = limit_only();
        let out = settle(&mut a, 1.0, &k, 2.0);
        assert!((out.to_degrees() - 7.6 * 0.425).abs() < 0.03, "{}", out.to_degrees());
        // mirrored: the same numbers to the right
        k.vy = -k.vy;
        let mut a = limit_only();
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

    /// a car sliding: the body's slip `beta` deg (nose LEFT of the path for
    /// +, as in a left-hander's oversteer) at 30 m/s, yawing `r`
    fn sliding(beta: f32, r: f32) -> Kin {
        let b = beta.to_radians();
        Kin { vx: 30.0 * b.cos(), vy: -30.0 * b.sin(), r, ..kin(30.0) }
    }

    #[test]
    fn hands_off_the_fronts_turn_into_the_slide() {
        // nose left of the path: the wheels go RIGHT (-), toward the path,
        // by less than the slide near the centre and more as it grows
        let mut a = Assist::default();
        let small = settle(&mut a, 0.0, &sliding(5.0, 0.0), 2.0).to_degrees();
        let mut a = Assist::default();
        let big = settle(&mut a, 0.0, &sliding(20.0, 0.0), 2.0).to_degrees();
        assert!(small < -1.0 && small > -5.0, "{small}");
        assert!(big < -10.0 && big > -20.0 && big / small > 4.0, "{big} {small}");
        // AGA's own curve (knee 72): 5 deg -> 1.9, 20 deg -> 12.6
        let aga = Params { self_knee: 72.0, ..Default::default() };
        let mut a = Assist::new(aga);
        assert!((settle(&mut a, 0.0, &sliding(5.0, 0.0), 2.0).to_degrees() + 1.90).abs() < 0.03);
        let mut a = Assist::new(aga);
        assert!((settle(&mut a, 0.0, &sliding(20.0, 0.0), 2.0).to_degrees() + 12.56).abs() < 0.05);
        // mirrored
        let mut a = Assist::default();
        let other = settle(&mut a, 0.0, &sliding(-20.0, 0.0), 2.0).to_degrees();
        assert!((other + big).abs() < 1e-3);
        // going straight: nothing; with full stick: none of it
        let mut a = Assist::default();
        assert_eq!(settle(&mut a, 0.0, &kin(30.0), 1.0), 0.0);
        let mut a = Assist::default();
        let mut b = Assist::new(Params { self_steer: 0.0, ..Default::default() });
        let k = sliding(20.0, 0.0);
        assert!((settle(&mut a, 1.0, &k, 2.0) - settle(&mut b, 1.0, &k, 2.0)).abs() < 1e-5);
        // off below walking pace and in reverse
        let mut a = Assist::default();
        assert_eq!(settle(&mut a, 0.0, &Kin { vx: 1.0, vy: -0.3, ..kin(1.0) }, 1.0), 0.0);
        let mut a = Assist::default();
        assert_eq!(settle(&mut a, 0.0, &Kin { vx: -10.0, vy: -3.0, ..kin(1.0) }, 1.0), 0.0);
        // the response knob: more of it steers more for a small slide; 0 = none
        let mut lo = Assist::new(Params { self_steer: 0.1, damping: 0.0, ..Default::default() });
        let mut hi = Assist::new(Params { self_steer: 0.9, damping: 0.0, ..Default::default() });
        let k = sliding(5.0, 0.0);
        assert!(settle(&mut hi, 0.0, &k, 2.0).abs() > settle(&mut lo, 0.0, &k, 2.0).abs() * 1.5);
        let mut off = Assist::new(Params { self_steer: 0.0, ..Default::default() });
        assert_eq!(settle(&mut off, 0.0, &k, 2.0), 0.0);
        // the knee: from there on the wheels follow the slide one to one
        let mut a = Assist::new(Params { self_knee: 10.0, damping: 0.0, ..Default::default() });
        assert!((settle(&mut a, 0.0, &sliding(20.0, 0.0), 2.0).to_degrees() + 20.0).abs() < 0.01);
        let mut a = Assist::new(Params { self_knee: 10.0, damping: 0.0, ..Default::default() });
        let soft = settle(&mut a, 0.0, &sliding(5.0, 0.0), 2.0).to_degrees();
        assert!(soft < -1.0 && soft > -5.0, "{soft}");
    }

    #[test]
    fn the_slide_is_read_ahead() {
        // a slide of 10 deg (nose left) collapsing at 40 deg/s: hands off,
        // the wheels are back at the centre ~0.2 s before the car is
        // straight; without the lead they are still turned into it
        let run = |lead: f32| {
            let mut a = Assist::new(Params { lead, damping: 0.0, self_knee: 5.0, self_filter: 0.1, ..Default::default() });
            settle(&mut a, 0.0, &sliding(18.0, 0.0), 1.0);
            let mut out = 0.0;
            // from 18 deg down to 4
            for i in 0..(0.35 / DT) as usize {
                out = a.step(0.0, &sliding(18.0 - 40.0 * i as f32 * DT, 0.0), DT);
            }
            out.to_degrees()
        };
        let (ahead, plain) = (run(0.2), run(0.0));
        assert!(plain < -5.0, "{plain}");
        assert!(ahead > -1.5 && ahead <= 0.5, "{ahead}");
    }

    #[test]
    fn yaw_damping_steers_against_the_rotation() {
        // no slide, yawing left (+): the wheels go right (-), and the other way
        let mut k = kin(30.0);
        // rear axle moving straight: vy = r b, so only the damping acts
        k.r = 0.5;
        k.vy = k.r * k.b;
        let mut a = Assist::default();
        let l = settle(&mut a, 0.0, &k, 1.0).to_degrees();
        assert!(l < -0.5 && l > -2.0, "{l}");
        k.r = -0.5;
        k.vy = k.r * k.b;
        let mut a = Assist::default();
        assert!((settle(&mut a, 0.0, &k, 1.0).to_degrees() + l).abs() < 1e-3);
        let mut a = Assist::new(Params { damping: 0.0, ..Default::default() });
        assert_eq!(settle(&mut a, 0.0, &k, 1.0), 0.0);
    }

    #[test]
    fn full_opposite_stick_only_just_passes_the_slide() {
        // nose 20 deg left of the path, full stick RIGHT to catch it
        let k = sliding(20.0, 0.0);
        let mut a = Assist::default();
        let out = settle(&mut a, -1.0, &k, 2.0).to_degrees();
        // the slide's 20 deg + 7.6 x (0.2 x 0.7 + 0.1) = 21.8: the fronts
        // 1.8 deg past where the car is going
        assert!((out + 21.82).abs() < 0.05, "{out}");
        // without the limit: the far peak, 20 + 7.6
        let mut b = Assist::new(Params { counter: -1.0, ..Default::default() });
        let free = settle(&mut b, -1.0, &k, 2.0).to_degrees();
        assert!((free + 27.6).abs() < 0.05, "{free}");
        // steering INTO the slide (left) is the ordinary limit's business
        let mut a = Assist::new(Params { rear_guard: 0.0, ..Default::default() });
        let into = settle(&mut a, 1.0, &k, 2.0).to_degrees();
        assert!((into - 7.6 * 0.425).abs() < 0.05, "{into}");
        // a small slide (under 4.5 deg) is not a catch: the normal limit
        let k = sliding(3.0, 0.0);
        let mut a = Assist::default();
        let out = settle(&mut a, -1.0, &k, 2.0).to_degrees();
        assert!((out + 10.6).abs() < 0.05, "{out}");
        // mirrored
        let k = sliding(-20.0, 0.0);
        let mut a = Assist::default();
        assert!((settle(&mut a, 1.0, &k, 2.0).to_degrees() - 21.82).abs() < 0.05);
    }

    #[test]
    fn the_rear_guard_takes_lock_back_when_the_rear_lets_go() {
        // full stick left, the rear within its peak (6 deg of 8): untouched
        let k = sliding(6.0, 0.0);
        let mut a = Assist::default();
        let mut b = Assist::new(Params { rear_guard: 0.0, ..Default::default() });
        assert!((settle(&mut a, 1.0, &k, 2.0) - settle(&mut b, 1.0, &k, 2.0)).abs() < 1e-6);
        // 4 deg past it: 2.5 x 4 = 10 deg less than the 7.6 x 0.425 floor -
        // through zero into opposite lock, with the stick still held left
        let k = sliding(12.0, 0.0);
        let mut a = Assist::default();
        let out = settle(&mut a, 1.0, &k, 2.0).to_degrees();
        assert!((out - (7.6 * 0.425 - 10.0)).abs() < 0.05, "{out}");
        // mirrored
        let k = sliding(-12.0, 0.0);
        let mut a = Assist::default();
        assert!((settle(&mut a, -1.0, &k, 2.0).to_degrees() + out).abs() < 1e-3);
        // never past the lock
        let k = sliding(60.0, 0.0);
        let mut a = Assist::default();
        assert!((settle(&mut a, 1.0, &k, 2.0).to_degrees() + 30.0).abs() < 1e-3);
        // a catch (stick against the slide) is not its business
        let k = sliding(20.0, 0.0);
        let mut a = Assist::default();
        let mut b = Assist::new(Params { rear_guard: 0.0, ..Default::default() });
        assert!((settle(&mut a, -1.0, &k, 2.0) - settle(&mut b, -1.0, &k, 2.0)).abs() < 1e-6);
    }

    #[test]
    fn off_is_the_raw_stick() {
        let mut a = Assist::new(Params { on: false, ..Default::default() });
        assert!((a.step(0.5, &kin(50.0), DT).to_degrees() - 15.0).abs() < 1e-4);
    }
}
