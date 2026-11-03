//! Force feedback the way AC EVO builds it (20260907 exe: Car::getFFB
//! 0x142225f30, Car::updateFFB 0x14222d070, InputManager::setFF 0x1420f9160,
//! DirectInputDevice::setFF 0x14212f0b0; L:/evoview_work/q/SPEC_FFB.md).
//!
//! Per physics step the game low-passes the steer-axis torque of the two
//! front hubs, adds a steering-rate damping term scaled by front wheel speed
//! (coefficient PhysicsEngine+0x48 = the player's dynamic_damping), flips
//! and scales it by 0.9, then by the car's ffMult x 0.001 (and an optional gamma). updateFFB fades it
//! to 50 % at standstill (PhysicsEngine+0x68 = 3 km/h, +0x6c = 0.5) and adds
//! the rack stop. InputManager::setFF multiplies by the player's ffb_gain and
//! the per-car custom gain (1.0 unless changed with the in-car binding), and
//! the damper by damper_gain; ffb.lut, the tanh soft clip and min force are
//! all off unless a file or flag turns them on. The device gets
//! clamp(force x 10000, +-10000): no per-wheel N m normalisation - the
//! wheel's own driver maps 10000 to its peak torque.
//!
//! The damper effect: ((min_damper - 0.7) x clamp(km/h / 10) + 0.7) x
//! damper_gain. With min_damper 0 that is 0.7 x damper_gain parked and
//! nothing above 10 km/h.
//!
//! The road effects (the player's ForceFeedbackEnhancements): the car fills
//! five values each physics step (fn 0x14222d4f0 -> car+0x3ec, Effects
//! below) and hands them to its controls provider (vt+0x60 -> InputManager
//! 0x1420f9b00 -> the device's vt+0x30, 0x142128e70, shared by the
//! DirectInput, Fanatec and Logitech devices, so any wheel). The device
//! keeps clamp((kerb curbs + bump gforce + slip slips + abs abs) x ffb_gain,
//! +-1) as an offset that setFF adds to the steering force before the
//! soft clip / min force and the 10000 clamp. The fifth value (rpm, with
//! rumble_gain) only drives a gamepad's motors.

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// user gain (InputDeviceSettings.ffb_gain), 0..
    pub gain: f32,
    /// rack low-pass: weight of the previous value per step (the controls
    /// provider's +0xc, 0 in the shipped game), 0 = none
    pub smooth: f32,
    /// steering-rate damping coefficient: getFFB reads car+0x3b0 -> +0x48,
    /// the PhysicsEngine's 0.002 from its ctor (0x142294130) until the
    /// player's settings are applied (fn 0x1414d2080 writes
    /// InputDeviceSettings.dynamic_damping there)
    pub rate_damping: f32,
    /// EVOVIEW_FFB_GAMMA: overrides the car's output gamma (car+0x8ca0,
    /// SteerControlsData f5); None = the car's
    pub gamma: Option<f32>,
    /// InputDeviceSettings.min_damper: the damper above 10 km/h, 0..1
    pub min_damper: f32,
    /// InputDeviceSettings.damper_gain: scales the whole damper effect
    pub damper_gain: f32,
    /// the low-speed fade: full force from this speed (km/h) ...
    pub fade_kmh: f32,
    /// ... down to this share at standstill
    pub fade_floor: f32,
    /// flip the force for wheels whose positive direction is the other way
    pub invert: bool,
    /// the wheel's full rotation, degrees (InputDeviceSettings.steer_lock):
    /// the wheel steers 1:1 through the car's ratio, as in the game
    pub wheel_range: f32,
    /// the player's extra strength (Numpad +/-), a trim on top of the game's chain
    pub boost: f32,
    /// where the settings came from ("EVO" = the game's own file)
    pub source: &'static str,
    /// InputManager::setFF's optional soft clip (flag +0x14c8, knee +0x14cc
    /// = 0.75): past the knee the force bends into tanh instead of hitting
    /// the wall, so a slide's build-up still reads near full force
    /// (EVOVIEW_FFB_SOFTCLIP=1; off in the game unless switched on)
    pub soft_clip: bool,
    /// setFF's optional minimum force (+0x14c0, level +0x14c4 = 0.06): any
    /// force at least this, past a wheel's own dead band (EVOVIEW_FFB_MIN_FORCE)
    pub min_force: f32,
    /// the road effects' shares (ForceFeedbackEnhancements)
    pub enh: Enhance,
    /// EVOVIEW_FFB_LEGACY=1: the chain before the 2026-11 audit
    /// (damper_gain as the damper itself, no low-speed fade), for A/B
    pub legacy: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            gain: 1.0,
            smooth: 0.0,
            rate_damping: 0.002,
            gamma: None,
            min_damper: 0.0,
            damper_gain: 1.0,
            fade_kmh: 3.0,
            fade_floor: 0.5,
            invert: false,
            wheel_range: 900.0,
            boost: 1.0,
            source: "defaults",
            soft_clip: false,
            min_force: 0.0,
            enh: Enhance::default(),
            legacy: false,
        }
    }
}

impl Settings {
    /// The player's own EVO settings, then EVOVIEW_FFB_GAIN / _SMOOTH /
    /// _DAMP / _GAMMA / _MIN_DAMPER / _DAMPER_GAIN / _INVERT / _LEGACY.
    pub fn from_env() -> Self {
        let mut s = Settings::default();
        let legacy = std::env::var("EVOVIEW_FFB_LEGACY").map(|v| v == "1").unwrap_or(false);
        if let Some(e) = evo_device_settings() {
            if e.gain > 0.0 { s.gain = e.gain; }
            if e.steer_lock > 0.0 { s.wheel_range = e.steer_lock; }
            s.min_damper = e.min_damper.clamp(0.0, 1.0);
            s.damper_gain = e.damper_gain.max(0.0);
            s.enh = e.enh;
            s.rate_damping = e.dynamic_damping.max(0.0);
            s.source = "EVO";
        }
        if legacy {
            s.legacy = true;
            s.fade_floor = 1.0;
        }
        if let Some(v) = std::env::var("EVOVIEW_WHEEL_RANGE").ok().and_then(|v| v.parse::<f32>().ok()) { s.wheel_range = v; }
        let f = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
        if let Some(v) = f("EVOVIEW_FFB_GAIN") { s.gain = v; }
        if let Some(v) = f("EVOVIEW_FFB_SMOOTH") { s.smooth = v.clamp(0.0, 0.99); }
        if let Some(v) = f("EVOVIEW_FFB_DAMP") { s.rate_damping = v; }
        if let Some(v) = f("EVOVIEW_FFB_GAMMA") { s.gamma = Some(v.max(0.1)); }
        if let Some(v) = f("EVOVIEW_FFB_MIN_DAMPER") { s.min_damper = v.clamp(0.0, 1.0); }
        // EVOVIEW_FFB_DAMPER is what the ACECM tab's Damper slider sets
        // (it was never read: the slider did nothing)
        if let Some(v) = f("EVOVIEW_FFB_DAMPER_GAIN").or_else(|| f("EVOVIEW_FFB_DAMPER")) { s.damper_gain = v.max(0.0); }
        // the tab's Steering damping is a share of the player's own
        if let Some(v) = f("EVOVIEW_FFB_DAMP_SCALE") { s.rate_damping *= v.max(0.0); }
        s.soft_clip = std::env::var("EVOVIEW_FFB_SOFTCLIP").map_or(false, |v| v == "1");
        if let Some(v) = f("EVOVIEW_FFB_MIN_FORCE") { s.min_force = v.clamp(0.0, 0.5); }
        if let Some(v) = f("EVOVIEW_FFB_CURBS") { s.enh.curbs = v.max(0.0); }
        if let Some(v) = f("EVOVIEW_FFB_GFORCE") { s.enh.gforce = v.max(0.0); }
        if let Some(v) = f("EVOVIEW_FFB_SLIPS") { s.enh.slips = v.max(0.0); }
        if let Some(v) = f("EVOVIEW_FFB_ABS") { s.enh.abs = v.max(0.0); }
        s.boost = load_boost();
        s.invert = std::env::var("EVOVIEW_FFB_INVERT").map(|v| v == "1").unwrap_or(false);
        s
    }
}

fn boost_path() -> Option<std::path::PathBuf> {
    std::env::var_os("APPDATA").map(|a| std::path::Path::new(&a).join("evoview").join("ffb.cfg"))
}

fn load_boost() -> f32 {
    if let Some(v) = std::env::var("EVOVIEW_FFB_BOOST").ok().and_then(|v| v.parse::<f32>().ok()) {
        return v;
    }
    boost_path().and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| t.split_whitespace().nth(1).and_then(|v| v.parse::<f32>().ok()))
        .filter(|v| *v > 0.0).unwrap_or(1.0)
}

pub fn save_boost(v: f32) {
    if let Some(p) = boost_path() {
        let _ = p.parent().map(std::fs::create_dir_all);
        let _ = std::fs::write(p, format!("boost {v}\n"));
    }
}

/// The FFB fields of the game's InputDeviceSettings.
#[derive(Clone, Copy, Debug, Default)]
pub struct EvoDevice {
    pub gain: f32,
    pub steer_lock: f32,
    pub dynamic_damping: f32,
    pub min_damper: f32,
    pub damper_gain: f32,
    pub enh: Enhance,
}

/// ForceFeedbackEnhancements (InputSettings f4): {1 curbs_gain,
/// 2 gforce_gain, 3 slips_gain, 4 abs_gain, 5 rumble_gain}, 0..1. The
/// proto default (no file) is all 0: no road effects.
#[derive(Clone, Copy, Debug, Default)]
pub struct Enhance {
    pub curbs: f32,
    pub gforce: f32,
    pub slips: f32,
    pub abs: f32,
}

/// What the car's road effects are made of, each physics step.
#[derive(Clone, Copy, Debug, Default)]
pub struct FxIn {
    /// |velocity|, m/s (car+0x8f2c)
    pub speed: f32,
    /// (vibrationGain, vibrationLength m) of the surface with the largest
    /// gain under a loaded tyre (tyre+0x670 -> +0x34 / +0x38)
    pub kerb: (f32, f32),
    /// the largest normalised slip of a loaded tyre (the SCTM's nd = s ky,
    /// tyre+0x3c0; 1 = the brush curve fully sliding)
    pub nd: f32,
    /// the body's vertical acceleration in its own frame, g (from the
    /// velocity change: gravity is not in it; car+0x4b0)
    pub up_g: f32,
    /// the car has ABS and it is working this step (car+0x1978 && +0x1979)
    pub abs: bool,
}

/// The five road-effect values (car+0x3ec..0x3fc), from fn 0x14222d4f0.
#[derive(Clone, Copy, Debug, Default)]
pub struct Effects {
    /// distance travelled, m (car+0x9208) and time, s (car+0x9380)
    dist: f32,
    t: f64,
    /// kerb, bump ("gforce"), slip, (rpm,) abs
    pub v: [f32; 5],
}

/// ((x mod p) - p/2) / (p/2): a -1..1 sawtooth of period p (fn 0x14278e8f0)
fn saw(x: f32, p: f32) -> f32 {
    ((x - (x / p).trunc() * p) - p * 0.5) / (p * 0.5)
}

impl Effects {
    pub fn step(&mut self, i: &FxIn, dt: f32) {
        // the kerb and slip buzz fade in from 0.5 to 1.5 m/s
        let fade = (i.speed - 0.5).clamp(0.0, 1.0);
        self.dist += i.speed * dt;
        let (gain, len) = i.kerb;
        let kerb = if gain != 0.0 && len != 0.0 { saw(self.dist, len) * gain } else { 0.0 };
        // a ~5 Hz-per-m/s buzz, as strong as the body's vertical g (to 1 g)
        let bump = i.up_g.abs().min(1.0) * (self.dist * 30.0).sin();
        self.t += dt as f64;
        // past half the peak slip, squared up to 1.5 x peak, 19 Hz
        let mut x = (i.nd - 0.5).max(0.0);
        if x <= 1.0 {
            x *= x;
        }
        let slip = (self.t * 120.0).sin() as f32 * (x * 0.4).clamp(0.0, 1.0);
        // ABS: a +-1 square wave at ~15.5 Hz (sin(t ms / 10.2775))
        let abs = if i.abs {
            let s = ((self.t * 1000.0) as f32 / 10.2775).sin();
            (if 0.0 < s { 1.0 } else { -1.0 }) * i.speed.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.v = [kerb * fade, bump, slip * fade, 0.0, abs];
    }

    /// The offset the device adds to the force (fn 0x142128e70).
    pub fn offset(&self, s: &Settings) -> f32 {
        let e = &s.enh;
        ((self.v[0] * e.curbs + self.v[1] * e.gforce + self.v[2] * e.slips + self.v[4] * e.abs) * s.gain).clamp(-1.0, 1.0)
    }
}

/// The game's saved input settings (Saved Games/ACE/input_settings
/// .inputsettings: InputSettings f2 InputDeviceSettings {1 ffb_gain,
/// 2 steer_lock, 3 dynamic_damping, 5 min_damper, 6 damper_gain} and
/// f4 ForceFeedbackEnhancements), when the file is there.
pub fn evo_device_settings() -> Option<EvoDevice> {
    let home = std::env::var_os("USERPROFILE")?;
    let p = std::path::Path::new(&home).join("Saved Games").join("ACE").join("input_settings.inputsettings");
    let d = std::fs::read(p).ok()?;
    let mut out = None;
    crate::proto::fields(&d, 0, d.len(), |f, v| {
        let (2, Some((a, b))) = (f, v.range()) else { return };
        let mut e = EvoDevice::default();
        crate::proto::fields(&d, a, b, |g, w| match g {
            1 => e.gain = w.f32(),
            2 => e.steer_lock = w.f32(),
            3 => e.dynamic_damping = w.f32(),
            5 => e.min_damper = w.f32(),
            6 => e.damper_gain = w.f32(),
            _ => {}
        });
        out = Some(e);
    });
    let enh = std::cell::Cell::new(Enhance::default());
    crate::proto::fields(&d, 0, d.len(), |f, v| {
        let (4, Some((a, b))) = (f, v.range()) else { return };
        let mut e = Enhance::default();
        crate::proto::fields(&d, a, b, |g, w| match g {
            1 => e.curbs = w.f32(),
            2 => e.gforce = w.f32(),
            3 => e.slips = w.f32(),
            4 => e.abs = w.f32(),
            _ => {}
        });
        enh.set(e);
    });
    out.map(|o| EvoDevice { enh: enh.get(), ..o })
}

/// The rack's stop as the game builds it (fn 0x14204f880, added in
/// updateFFB): x = steer / lock (car+0x3dc, -1..1); past 0.94 a spring
/// s x 40 x (|x| - 0.94), s = min((|x| - 0.94) / 0.06, 1), plus
/// s x 1.8 x |dx/dt| while moving further out. In device units (1 = 10000),
/// so it saturates at ~96.5 % of the lock. The sign matches Ffb::step's
/// output (+ wheel_deg = left, pushed back right).
/// Past `knee` the force bends into tanh toward 1 (slope 1 at the knee):
/// what reaches 1 at about twice the knee's headroom instead of at once.
pub fn soft_clip(f: f32, knee: f32) -> f32 {
    let a = f.abs();
    if a <= knee {
        return f;
    }
    let room = 1.0 - knee;
    f.signum() * (knee + room * ((a - knee) / room).tanh())
}

pub fn soft_lock(wheel_deg: f32, stop_deg: f32, rate: f32) -> f32 {
    let x = wheel_deg / stop_deg.max(1.0);
    let over = x.abs() - 0.94;
    if over <= 0.0 {
        return 0.0;
    }
    let s = (over / 0.06).min(1.0);
    let outward = x * rate > 0.0;
    -x.signum() * s * (40.0 * over + if outward { 1.8 * rate.abs() } else { 0.0 })
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Ffb {
    rack: f32,
    steer_prev: f32,
    /// the steering rate, low-passed (getFFB with DAT_145c93be1 set)
    rate_f: f32,
    /// the last wheel position over the lock, for the stop's rate term
    pub lock_prev: f32,
    /// the road effects' offset (Effects::offset), added to the next step
    pub fx: f32,
}

/// What one step sends to the wheel.
#[derive(Clone, Copy, Debug, Default)]
pub struct Out {
    /// constant force, -1..1 (x 10000 = DirectInput magnitude)
    pub force: f32,
    /// damper effect coefficient, 0..1
    pub damper: f32,
    /// the force before the +-1 clamp (how far past the wheel's peak)
    pub raw: f32,
}

impl Ffb {
    /// `hub_torque`: steer-axis torque of both front hubs, N m.
    /// `steer`: road-wheel steer angle, rad. `omega_front`: summed front
    /// wheel angular speed, rad/s. `ff_mult`, `car_gamma`: CarData
    /// SteerControlsData f1 and f5 (gamma 1 = linear).
    pub fn step(&mut self, s: &Settings, hub_torque: f32, steer: f32, omega_front: f32, speed_kmh: f32, ff_mult: f32, car_gamma: f32, dt: f32) -> Out {
        let dt = dt.max(1e-4);
        self.rack = (self.rack - hub_torque) * s.smooth + hub_torque;
        // The game's damping term with its smoothing switch (DAT_145c93be1,
        // 1 in the shipped exe): the steering rate low-passed by 0.2 a 3 ms
        // step (here scaled to this call's dt) and the wheel-speed factor
        // capped at 60 (front wheels ~55 rad/s each, ~70 km/h): above that
        // the damping no longer grows with speed.
        let rate = (steer - self.steer_prev) / dt;
        let a = 1.0 - 0.8f32.powf(dt / 0.003);
        self.rate_f += (rate - self.rate_f) * a;
        let damp = (omega_front.abs() * 1.08 * 0.5).min(60.0) * self.rate_f * s.rate_damping;
        self.steer_prev = steer;
        let f = -(damp + self.rack) * 0.9;
        let m = ff_mult * 0.001;
        let gamma = if s.legacy { s.gamma.unwrap_or(1.0) } else { s.gamma.unwrap_or(car_gamma) };
        let out = if gamma == 1.0 { -f * m } else { -f.signum() * (m * f).abs().powf(gamma) };
        // updateFFB's low-speed fade (PhysicsEngine+0x68 / +0x6c)
        let low = if s.fade_kmh > 0.0 {
            s.fade_floor + (1.0 - s.fade_floor) * (speed_kmh / s.fade_kmh).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let fade = (speed_kmh / 10.0).clamp(0.0, 1.0);
        // InputManager::setFF: the force x gain plus the device's road
        // effects, then the optional stages on the sum
        let mut raw = (out * low * s.gain * s.boost + self.fx) * if s.invert { -1.0 } else { 1.0 };
        if s.soft_clip {
            raw = soft_clip(raw, 0.75);
        }
        if s.min_force > 0.0 && raw.abs() > 1e-3 {
            raw = raw.signum() * (s.min_force + (1.0 - s.min_force) * raw.abs());
        }
        let damper = if s.legacy {
            (s.damper_gain.min(1.0) - 0.7) * fade + 0.7
        } else {
            ((s.min_damper - 0.7) * fade + 0.7) * s.damper_gain
        };
        Out { force: raw.clamp(-1.0, 1.0), damper: damper.clamp(0.0, 1.0), raw }
    }

    pub fn reset(&mut self) {
        *self = Ffb::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_clip_bends_past_the_knee() {
        assert_eq!(soft_clip(0.5, 0.75), 0.5);
        assert!((soft_clip(0.76, 0.75) - 0.76).abs() < 1e-3);
        let hi = soft_clip(1.5, 0.75);
        assert!(hi > 0.95 && hi < 1.0, "{hi}");
        assert!((soft_clip(-1.5, 0.75) + hi).abs() < 1e-6);
    }

    #[test]
    fn road_effects() {
        assert!((saw(0.0, 2.0) + 1.0).abs() < 1e-6);
        assert!(saw(1.0, 2.0).abs() < 1e-6);
        assert!((saw(3.5, 2.0) - 0.5).abs() < 1e-6);
        let set = Settings { gain: 1.0, enh: Enhance { curbs: 0.5, gforce: 0.5, slips: 0.5, abs: 0.5 }, ..Default::default() };
        // nothing on smooth road under the peak slip
        let mut e = Effects::default();
        let i = FxIn { speed: 20.0, nd: 0.5, ..Default::default() };
        for _ in 0..100 { e.step(&i, 0.003); }
        assert_eq!(e.offset(&set), 0.0);
        // a kerb's sawtooth: vibrationGain x 0.5 at most
        let i = FxIn { speed: 20.0, kerb: (0.3, 0.5), ..Default::default() };
        let mut hi = 0.0f32;
        for _ in 0..300 { e.step(&i, 0.003); hi = hi.max(e.offset(&set).abs()); }
        assert!(hi > 0.14 && hi <= 0.15 + 1e-4, "{hi}");
        // ABS: +-0.5 square wave, ~15.5 Hz -> ~31 sign changes a second
        let mut e = Effects::default();
        let i = FxIn { speed: 20.0, abs: true, ..Default::default() };
        let (mut flips, mut last) = (0, 0.0f32);
        for _ in 0..333 {
            e.step(&i, 0.003);
            let o = e.offset(&set);
            assert!((o.abs() - 0.5).abs() < 1e-6);
            if o * last < 0.0 { flips += 1; }
            last = o;
        }
        assert!((29..=33).contains(&flips), "{flips}");
        // slip: 0.4 x (nd - 0.5)^2 at 19 Hz; nd 1.5 -> +-0.4 x 0.5
        let mut e = Effects::default();
        let i = FxIn { speed: 20.0, nd: 1.5, ..Default::default() };
        let mut hi = 0.0f32;
        for _ in 0..333 { e.step(&i, 0.003); hi = hi.max(e.offset(&set).abs()); }
        assert!((hi - 0.2).abs() < 0.01, "{hi}");
    }

    #[test]
    fn linear_in_torque_and_ffmult() {
        let s = Settings { min_damper: 0.0, damper_gain: 0.1, ..Default::default() };
        let mut a = Ffb::default();
        let o = a.step(&s, 500.0, 0.0, 0.0, 100.0, 2.0, 1.0, 0.003);
        // -( -(500) x 0.9 ) ... out = 0.9 x 500 x 0.002
        assert!((o.force - 0.9).abs() < 1e-4, "{o:?}");
        // min_damper 0: no damper above 10 km/h, 0.7 x damper_gain parked
        assert!(o.damper.abs() < 1e-6);
        let mut b = Ffb::default();
        let p = b.step(&s, 500.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.003);
        assert!((p.damper - 0.07).abs() < 1e-6);
        // half the force at standstill (the low-speed fade)
        assert!((p.force - 0.45).abs() < 1e-4, "{p:?}");
    }
}
