//! A gamepad through XInput: xinput1_4.dll (xinput9_1_0.dll before Windows
//! 8) loaded at runtime from System32, the way wheel.rs loads dinput8 - no
//! crate, no import library. DirectInput is a poor fit for a pad: the two
//! triggers share one axis and there is no rumble.
//!
//! The first connected pad drives. Polled once a frame (~4 us); while none
//! is connected one slot is asked every half second, in turn (an empty slot
//! is the slow call, ~0.2 ms). `EVOVIEW_PAD=0` turns the whole thing off.
//!
//! The mapping (also in render.rs KEYS):
//!   left stick   steer            RT / LT   throttle / brake (analog)
//!   A            handbrake        B         pit limiter
//!   X            car lights       Y         camera
//!   RB / LB      gear up / down   Start     pit menu     Back  leaderboard
//!   d-pad        TC up / down (up / down), ABS up / down (right / left);
//!                with the pit menu up it moves in the menu and A is OK
//!   left stick click  starter     right stick held  back to the pits
//!
//! Tunables: EVOVIEW_PAD_DEADZONE (0.10), EVOVIEW_PAD_GAMMA (1.4),
//! EVOVIEW_PAD_TRIGGER_DEADZONE (0.03), EVOVIEW_PAD_BRAKE_GAMMA (1.0),
//! EVOVIEW_PAD_RUMBLE (0.4, 0 = off); the steering assist's are in
//! padassist.rs. EVOVIEW_PAD_LOG=1 prints the raw state once a second.

/// LoadLibraryExW: System32 only
const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x800;

// XINPUT_GAMEPAD buttons
const DPAD_UP: u16 = 0x0001;
const DPAD_DOWN: u16 = 0x0002;
const DPAD_LEFT: u16 = 0x0004;
const DPAD_RIGHT: u16 = 0x0008;
const START: u16 = 0x0010;
const BACK: u16 = 0x0020;
const LS: u16 = 0x0040;
const RS: u16 = 0x0080;
const LB: u16 = 0x0100;
const RB: u16 = 0x0200;
const A: u16 = 0x1000;
const B: u16 = 0x2000;
const X: u16 = 0x4000;
const Y: u16 = 0x8000;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Gamepad {
    buttons: u16,
    lt: u8,
    rt: u8,
    lx: i16,
    ly: i16,
    rx: i16,
    ry: i16,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct State {
    packet: u32,
    pad: Gamepad,
}

#[repr(C)]
struct Vibration {
    left: u16,
    right: u16,
}

type GetState = unsafe extern "system" fn(u32, *mut State) -> u32;
type SetState = unsafe extern "system" fn(u32, *mut Vibration) -> u32;
type GetCaps = unsafe extern "system" fn(u32, u32, *mut Caps) -> u32;

/// XINPUT_CAPABILITIES
#[repr(C)]
#[derive(Default)]
struct Caps {
    kind: u8,
    sub_type: u8,
    flags: u16,
    pad: Gamepad,
    vib: [u16; 2],
}

/// XINPUT_DEVSUBTYPE_GAMEPAD; 0 = unknown (some third-party pads)
fn is_gamepad(sub_type: u8) -> bool {
    sub_type == 0 || sub_type == 1
}

/// The stick curve and trigger shaping.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub deadzone: f32,
    pub gamma: f32,
    pub trigger_deadzone: f32,
    pub brake_gamma: f32,
}

impl Default for Shape {
    fn default() -> Self {
        Shape { deadzone: 0.10, gamma: 1.4, trigger_deadzone: 0.03, brake_gamma: 1.0 }
    }
}

fn env(k: &str, d: f32) -> f32 {
    crate::envc::var(k).ok().and_then(|v| v.parse::<f32>().ok()).filter(|v| v.is_finite()).unwrap_or(d)
}

impl Shape {
    pub fn from_env() -> Shape {
        let d = Shape::default();
        Shape {
            deadzone: env("EVOVIEW_PAD_DEADZONE", d.deadzone).clamp(0.0, 0.9),
            gamma: env("EVOVIEW_PAD_GAMMA", d.gamma).clamp(0.2, 5.0),
            trigger_deadzone: env("EVOVIEW_PAD_TRIGGER_DEADZONE", d.trigger_deadzone).clamp(0.0, 0.9),
            brake_gamma: env("EVOVIEW_PAD_BRAKE_GAMMA", d.brake_gamma).clamp(0.2, 5.0),
        }
    }

    /// A stick axis -1..1: the deadzone cut out and the rest rescaled so the
    /// output starts at 0 (no step at its edge), then the gamma curve.
    pub fn stick(&self, x: f32) -> f32 {
        let a = deadzone(x.clamp(-1.0, 1.0), self.deadzone);
        a.signum() * a.abs().powf(self.gamma)
    }

    /// A trigger 0..1 (linear past its deadzone; `gamma` for the brake).
    pub fn trigger(&self, v: f32, gamma: f32) -> f32 {
        deadzone(v.clamp(0.0, 1.0), self.trigger_deadzone).powf(gamma)
    }
}

/// sign(x) max(0, |x| - dz) / (1 - dz)
pub fn deadzone(x: f32, dz: f32) -> f32 {
    let dz = dz.clamp(0.0, 0.99);
    x.signum() * (x.abs() - dz).max(0.0) / (1.0 - dz)
}

/// One frame of the pad, in trackdrive's conventions.
#[derive(Clone, Copy, Debug, Default)]
pub struct PadInput {
    /// -1 right .. +1 left, after the deadzone and gamma
    pub steer: f32,
    pub throttle: f32,
    pub brake: f32,
    pub handbrake: bool,
    /// true on the frame they are pressed
    pub shift_up: bool,
    pub shift_down: bool,
    pub starter: bool,
    pub camera: bool,
    pub lights: bool,
    pub board: bool,
    /// d-pad: TC, ABS steps (-1 / +1; aids::Knob::ALL order, bias unused)
    pub aids: [i32; 3],
    /// wheel::PIT_ROLES order: limiter, menu, up, down, left, right, ok, return
    pub pit: [bool; 8],
    /// something on the pad is off its rest (it takes the car over)
    pub touched: bool,
}

/// Raw state to a PadInput; `prev` = last frame's buttons, `menu` = the pit
/// menu is up (the d-pad and A are its).
fn map(g: &Gamepad, prev: u16, menu: bool, s: &Shape) -> PadInput {
    let down = |b: u16| g.buttons & b != 0;
    let hit = |b: u16| g.buttons & b != 0 && prev & b == 0;
    // i16 is -32768..32767: both ends to exactly +-1
    let lx = (g.lx as f32 / 32767.0).max(-1.0);
    let steer = -s.stick(lx);
    let throttle = s.trigger(g.rt as f32 / 255.0, 1.0);
    let brake = s.trigger(g.lt as f32 / 255.0, s.brake_gamma);
    let mut p = PadInput {
        steer,
        throttle,
        brake,
        handbrake: down(A) && !menu,
        shift_up: hit(RB),
        shift_down: hit(LB),
        starter: hit(LS),
        camera: hit(Y),
        lights: hit(X),
        board: hit(BACK),
        touched: steer != 0.0 || throttle > 0.0 || brake > 0.0 || g.buttons != 0,
        ..Default::default()
    };
    p.pit[0] = hit(B);
    p.pit[1] = hit(START);
    p.pit[7] = hit(RS);
    if menu {
        p.pit[2] = hit(DPAD_UP);
        p.pit[3] = hit(DPAD_DOWN);
        p.pit[4] = hit(DPAD_LEFT);
        p.pit[5] = hit(DPAD_RIGHT);
        p.pit[6] = hit(A);
    } else {
        p.aids[1] = hit(DPAD_UP) as i32 - hit(DPAD_DOWN) as i32;
        p.aids[2] = hit(DPAD_RIGHT) as i32 - hit(DPAD_LEFT) as i32;
    }
    p
}

/// The two motors (low-frequency left, high-frequency right), 0..1, from
/// the road effects' inputs (rigcar's FxIn, the same data the wheel's FFB
/// effects use): tyre slip on the light motor - a faint buzz from half the
/// peak slip, strong past it; kerbs, bumps and the ABS pulse on the heavy
/// one. `t`: seconds, for the ABS pulse.
pub fn rumble(fx: &crate::ffb::FxIn, strength: f32, t: f64) -> (f32, f32) {
    if strength <= 0.0 {
        return (0.0, 0.0);
    }
    // as the wheel's effects: nothing at a standstill, in from 0.5 to 1.5 m/s
    let fade = (fx.speed - 0.5).clamp(0.0, 1.0);
    let x = ((fx.nd - 0.5) / 0.5).clamp(0.0, 1.0);
    let slip = if x > 0.0 { x.powi(4) * 0.95 + 0.05 } else { 0.0 };
    let kerb = (fx.kerb.0 * 2.5).clamp(0.0, 1.0);
    // a landing or a pothole: past 0.3 g of vertical jolt
    let bump = ((fx.up_g.abs() - 0.3) * 0.8).clamp(0.0, 1.0);
    // ABS: the wheel effect's ~15.5 Hz square wave, here on / off
    let abs = if fx.abs && (t * 1000.0 / 10.2775).sin() > 0.0 { 0.7 } else { 0.0 };
    let lo = (kerb.max(bump) * fade).max(abs * fx.speed.clamp(0.0, 1.0));
    let hi = slip * fade;
    let k = strength.min(1.0);
    let ok = |v: f32| if v.is_finite() { (v * k).clamp(0.0, 1.0) } else { 0.0 };
    (ok(lo), ok(hi))
}

pub struct Pad {
    _lib: libloading::Library,
    get: GetState,
    set: Option<SetState>,
    caps: Option<GetCaps>,
    /// right stick held this long (back to the pits needs a hold)
    rs_held: f32,
    /// the connected slot, and when the slots were last scanned
    slot: Option<u32>,
    scan_in: f32,
    scan_next: u32,
    prev: u16,
    pub shape: Shape,
    /// EVOVIEW_PAD_RUMBLE, 0..1
    pub rumble: f32,
    sent: (u16, u16),
    log_in: f32,
}

impl Pad {
    /// None with EVOVIEW_PAD=0 or without XInput (then nothing is polled).
    pub fn open() -> Option<Pad> {
        if crate::envc::var("EVOVIEW_PAD").map_or(false, |v| v == "0") {
            return None;
        }
        unsafe {
            let lib: libloading::Library = ["xinput1_4.dll", "xinput9_1_0.dll"].iter().find_map(|n| {
                libloading::os::windows::Library::load_with_flags(*n, LOAD_LIBRARY_SEARCH_SYSTEM32).ok()
            })?.into();
            let get: GetState = *lib.get::<GetState>(b"XInputGetState").ok()?;
            let set: Option<SetState> = lib.get::<SetState>(b"XInputSetState").ok().map(|s| *s);
            let caps: Option<GetCaps> = lib.get::<GetCaps>(b"XInputGetCapabilities").ok().map(|s| *s);
            let mut p = Pad {
                _lib: lib,
                get,
                set,
                caps,
                rs_held: 0.0,
                slot: None,
                scan_in: 0.0,
                scan_next: 0,
                prev: 0,
                shape: Shape::from_env(),
                rumble: env("EVOVIEW_PAD_RUMBLE", 0.4).clamp(0.0, 1.0),
                sent: (0, 0),
                log_in: 0.0,
            };
            p.scan();
            Some(p)
        }
    }

    /// Look for a pad: all four slots at start, then one slot every half
    /// second in turn - asking an empty slot takes ~0.2 ms, four at once
    /// would be a hitch in that frame.
    fn scan(&mut self) {
        self.scan_in = 0.5;
        let slots = if self.scan_next == 0 { 0..4 } else { let i = self.scan_next % 4; i..i + 1 };
        self.scan_next = self.scan_next % 4 + 1;
        for i in slots {
            let mut s = State::default();
            if unsafe { (self.get)(i, &mut s) } == 0 {
                // an Xbox-mode wheel (G920, TMX...) or a flight stick is an
                // XInput device too: it is wheel.rs's, never the pad's
                if let Some(caps) = self.caps {
                    let mut c = Caps::default();
                    if unsafe { caps(i, 0, &mut c) } == 0 && !is_gamepad(c.sub_type) {
                        continue;
                    }
                }
                self.slot = Some(i);
                self.prev = s.pad.buttons;
                println!("pad: controller {} connected (deadzone {:.2}, gamma {:.2}, rumble {:.2})", i + 1, self.shape.deadzone, self.shape.gamma, self.rumble);
                return;
            }
        }
    }

    /// This frame's input; None while no pad is connected.
    pub fn read(&mut self, dt: f32, menu: bool) -> Option<PadInput> {
        let Some(slot) = self.slot else {
            self.scan_in -= dt;
            if self.scan_in <= 0.0 {
                self.scan();
            }
            return None;
        };
        let mut s = State::default();
        if unsafe { (self.get)(slot, &mut s) } != 0 {
            println!("pad: controller {} disconnected", slot + 1);
            self.slot = None;
            self.scan_in = 0.5;
            self.sent = (0, 0);
            return None;
        }
        let mut p = map(&s.pad, self.prev, menu, &self.shape);
        self.prev = s.pad.buttons;
        // back to the pits: hold the right stick in for 0.8 s (a click is
        // too easy to hit mid-corner)
        let was = self.rs_held;
        self.rs_held = if s.pad.buttons & RS != 0 { was + dt } else { 0.0 };
        p.pit[7] = was < 0.8 && self.rs_held >= 0.8;
        if crate::envc::var_os("EVOVIEW_PAD_LOG").is_some() {
            self.log_in -= dt;
            if self.log_in <= 0.0 {
                self.log_in = 1.0;
                println!("PAD lx {:+6} lt {:3} rt {:3} buttons {:#06x} -> steer {:+.3} thr {:.3} brk {:.3}", s.pad.lx, s.pad.lt, s.pad.rt, s.pad.buttons, p.steer, p.throttle, p.brake);
            }
        }
        Some(p)
    }

    /// The motors, 0..1 each (low, high); only sent when the value changes.
    pub fn vibrate(&mut self, lo: f32, hi: f32) {
        let (Some(slot), Some(set)) = (self.slot, self.set) else { return };
        // 64 levels: a slowly changing slip does not send every frame
        let q = |v: f32| ((v.clamp(0.0, 1.0) * 63.0).round() * (65535.0 / 63.0)) as u16;
        let v = (q(lo), q(hi));
        if v != self.sent {
            self.sent = v;
            unsafe {
                set(slot, &mut Vibration { left: v.0, right: v.1 });
            }
        }
    }

    /// Motors off (focus lost, the pad handed the car back, exit).
    pub fn stop(&mut self) {
        self.vibrate(0.0, 0.0);
    }
}

impl Drop for Pad {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadzone_is_rescaled_and_odd() {
        assert_eq!(deadzone(0.0, 0.1), 0.0);
        assert_eq!(deadzone(0.1, 0.1), 0.0);
        assert_eq!(deadzone(-0.05, 0.1), 0.0);
        // no step at the edge, 1 stays 1
        assert!(deadzone(0.1001, 0.1) < 1e-3);
        assert!((deadzone(1.0, 0.1) - 1.0).abs() < 1e-6);
        assert!((deadzone(0.55, 0.1) - 0.5).abs() < 1e-6);
        assert!((deadzone(-0.55, 0.1) + 0.5).abs() < 1e-6);
        // no deadzone: the identity
        assert_eq!(deadzone(0.3, 0.0), 0.3);
    }

    #[test]
    fn stick_curve() {
        let s = Shape::default();
        assert_eq!(s.stick(0.05), 0.0);
        assert!((s.stick(1.0) - 1.0).abs() < 1e-6);
        assert!((s.stick(-1.0) + 1.0).abs() < 1e-6);
        // gamma 1.4 softens the centre: half travel past the deadzone -> 0.5^1.4
        assert!((s.stick(0.55) - 0.5f32.powf(1.4)).abs() < 1e-5);
        // monotonic and odd
        let mut last = -1.0f32;
        for i in -100..=100 {
            let x = i as f32 / 100.0;
            let y = s.stick(x);
            assert!(y >= last - 1e-6, "{x}");
            assert!((y + s.stick(-x)).abs() < 1e-6);
            last = y;
        }
        // out of range input is clamped
        assert_eq!(s.stick(3.0), 1.0);
    }

    #[test]
    fn triggers() {
        let s = Shape::default();
        assert_eq!(s.trigger(0.02, 1.0), 0.0);
        assert!((s.trigger(1.0, 1.0) - 1.0).abs() < 1e-6);
        assert!(s.trigger(0.5, 1.5) < s.trigger(0.5, 1.0));
    }

    #[test]
    fn mapping() {
        let s = Shape::default();
        // stick right = steer negative (evoview: + left), full travel = 1
        let g = Gamepad { lx: 32767, rt: 255, lt: 0, buttons: RB | A, ..Default::default() };
        let p = map(&g, 0, false, &s);
        assert!((p.steer + 1.0).abs() < 1e-6 && (p.throttle - 1.0).abs() < 1e-6 && p.brake == 0.0);
        assert!(p.shift_up && !p.shift_down && p.handbrake && p.touched);
        let g = Gamepad { lx: -32768, ..Default::default() };
        assert!((map(&g, 0, false, &s).steer - 1.0).abs() < 1e-6);
        // held buttons do not repeat
        let g = Gamepad { buttons: RB, ..Default::default() };
        assert!(!map(&g, RB, false, &s).shift_up);
        // at rest: untouched (stick noise inside the deadzone)
        let g = Gamepad { lx: 2000, ..Default::default() };
        assert!(!map(&g, 0, false, &s).touched);
        // d-pad: the aids while driving, the menu's rows while it is up
        let g = Gamepad { buttons: DPAD_UP | DPAD_LEFT | A, ..Default::default() };
        let p = map(&g, 0, false, &s);
        assert_eq!(p.aids, [0, 1, -1]);
        assert!(p.pit[2..7].iter().all(|b| !b));
        let p = map(&g, 0, true, &s);
        assert_eq!(p.aids, [0, 0, 0]);
        assert!(p.pit[2] && p.pit[4] && p.pit[6] && !p.handbrake);
    }

    #[test]
    fn rumble_levels() {
        use crate::ffb::FxIn;
        // gripping on smooth road: silent
        assert_eq!(rumble(&FxIn { speed: 30.0, nd: 0.4, ..Default::default() }, 0.4, 0.0), (0.0, 0.0));
        // near the limit a faint buzz, past it the full strength, light motor only
        let (lo, near) = rumble(&FxIn { speed: 30.0, nd: 0.6, ..Default::default() }, 0.4, 0.0);
        let (_, over) = rumble(&FxIn { speed: 30.0, nd: 1.3, ..Default::default() }, 0.4, 0.0);
        assert!(lo == 0.0 && near > 0.0 && near < 0.05 && (over - 0.4).abs() < 1e-6, "{near} {over}");
        // a kerb on the heavy motor; nothing standing still
        let (lo, hi) = rumble(&FxIn { speed: 30.0, kerb: (0.3, 0.5), ..Default::default() }, 0.4, 0.0);
        assert!(lo > 0.25 && hi == 0.0);
        assert_eq!(rumble(&FxIn { speed: 0.0, nd: 2.0, kerb: (0.3, 0.5), ..Default::default() }, 0.4, 0.0), (0.0, 0.0));
        // ABS pulses; strength 0 and NaN are silent
        let on = (0..100).filter(|i| rumble(&FxIn { speed: 30.0, abs: true, ..Default::default() }, 1.0, *i as f64 * 0.005).0 > 0.0).count();
        assert!(on > 30 && on < 70, "{on}");
        assert_eq!(rumble(&FxIn { speed: 30.0, nd: 2.0, ..Default::default() }, 0.0, 0.0), (0.0, 0.0));
        assert_eq!(rumble(&FxIn { speed: f32::NAN, nd: f32::NAN, ..Default::default() }, 0.4, 0.0), (0.0, 0.0));
    }

    /// `cargo test pad_poll_cost -- --ignored --nocapture`: what a frame
    /// pays for the pad (one XInputGetState; with none connected, a scan of
    /// the four slots every 2 s). Works with or without a pad attached.
    #[test]
    #[ignore]
    fn pad_poll_cost() {
        let Some(mut p) = Pad::open() else {
            println!("pad: XInput not available or EVOVIEW_PAD=0");
            return;
        };
        let n = 2000;
        let t = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(p.read(1.0 / 120.0, false));
        }
        let per = t.elapsed().as_secs_f64() / n as f64;
        // each slot on its own: an empty one is the slow call
        let slots: Vec<String> = (0..4).map(|i| {
            let mut s = State::default();
            let t = std::time::Instant::now();
            let mut hr = 0;
            for _ in 0..200 {
                hr = unsafe { (p.get)(i, &mut s) };
            }
            format!("{} {:.1} us", if hr == 0 { "pad" } else { "empty" }, t.elapsed().as_secs_f64() / 200.0 * 1e6)
        }).collect();
        println!("pad: connected {}, read {:.2} us a frame; XInputGetState per slot: {} (one is asked every 0.5 s while no pad is connected)", p.slot.is_some(), per * 1e6, slots.join(", "));
    }
}
