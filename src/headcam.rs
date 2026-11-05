//! In-car camera motion the way AC EVO moves its drivable cameras.
//!
//! Reversed from the 2026-09-07 exe (DrivableCamera family, vftables at
//! 0x143296148.., update = slot 2):
//! - CockpitCamera::update 0x141be0230, DashCamera 0x141be0940,
//!   BonnetCamera 0x141bdfd10, ExternalDampedCamera (chase) 0x141be1010.
//! - The camera rides the car body matrix RIGIDLY; on top of it:
//!   1. head movement 0x141be19e0 (g-forces -> seat offset, cm):
//!      a = accG (car space, g); d = a - a_prev (per frame)
//!      target = 3.6 * (-2 d.x, -(a.y + d.y), -(a.z + d.z))
//!      p += clamp((gain*10 + 1)/3 * dt, 0, 1) * (target - p)
//!      p = L * tanh(p / L) * sign(p), L = (22, 7, 10) cm
//!      ⚠ that sign(p) makes the result |.|: in the game the head only ever
//!      moves +x/+y/+z (forward when braking AND when accelerating).
//!      gain = general_movement (cockpit), x dashcam_factor (dash),
//!      x bonnet_factor (bonnet); gain <= 0 skips it (offset stays put).
//!   2. horizon lock 0x141be21d0: the camera's world rotation is slerped
//!      toward its heading-only rotation (euler 0x142863bb0, slerp
//!      0x1412a89f0) by horizon_lock (cockpit), x dashcam_factor / x
//!      bonnet_factor - pitch AND roll, after everything else. That level
//!      is the WORLD's: 0.4 of every hill comes out of the view and the dash
//!      climbs the screen uphill (Josh, Oulton: "blinded by the dashboard").
//!      evoview levels to the road under the car instead (rigcar's hub
//!      plane, else the body's slow tilt): only the car's own pitch and roll
//!      on its springs are locked. EVOVIEW_CAM_HORIZON_REF=world = the game.
//!   3. look with steer 0x141be1e30 (off by default: gain 0).
//! - Classes by CameraModeType (0x141bcda40): 1 Cockpit, 2 ExternalDamped
//!   (NearChase / FarChase), 3 ExternalFixed (Bumper: rigid, 0x141be1570),
//!   4 Dash, 5 Bonnet. ExternalDamped: q = slerp(q_prev, q_car,
//!   min(chasecam_rotation_lag_speed * dt, 1)), head movement at
//!   general_movement, horizon lock on q, eye = car origin + q (pos + head).
//!   The car's other cameras are CameraOnboard (0x141bd90d0): head 0x141bd9730
//!   (accG clamped +-2, rate (onboard_movement*3 + 1)/s, L = 2/7/1 cm), no
//!   horizon lock. TV cameras are CameraTrack (cams.rs), no head.
//! - accG: the physics low-passes WORLD velocity (v_f += 0.003 (v - v_f)
//!   per 3 ms step, tau 1 s) and reports (v - v_f)/9.806 in car space
//!   (0x1424ba2f0) - a smooth, ~1 s lagged g.
//! - The numbers: System/camerasettings.camerasettings in content.kspkg
//!   (horizon_lock 0.4, general_movement 0.8, bonnet_factor 0.1, dashcam
//!   factor unset = 0, look_with_steer unset), overridden by
//!   Saved Games/ACE/camerasettings.camerasettings (Josh has none).
//!
//! evoview's own extra (not the game's): the body evoview drives is the
//! plane through the four tyre contacts plus bump noise, with no sprung
//! mass, so every triangle edge and kerb ripple went straight into the
//! view. EVOVIEW_CAM_LP_HZ low-passes the body's heave, pitch and roll
//! (lag-free on slopes) before the camera is mounted on it. Off on the
//! rigcar body (sprung already; the filter let the eye drift 9 cm).
//!
//! Knobs (EVOVIEW_CAM_*): HEAD=0 is the old rigid camera; HORIZON, MOVE,
//! DASH, BONNET, LWS_GAIN/GAMMA/SMOOTH override the game's settings;
//! G_TAU the accG low-pass, HEAD_ABS=0 a signed head (forward when braking,
//! back when accelerating), HEAD_SCALE multiplies the offset, LP_HZ /
//! LP_MAX_CM / LP_MAX_DEG the body filter (LP_HZ=0 off), LOG=path a CSV;
//! HORIZON_REF=world the game's world-level lock, SLOPE_TAU the road-up
//! filter (s); CHASE=0 the old rigid chase, CHASE_LAG / ONBOARD override
//! those settings; VLOG=path a per-frame view CSV for any camera.

use glam::{Mat3, Mat4, Quat, Vec2, Vec3};

fn env_f(k: &str) -> Option<f32> {
    crate::envc::var(k).ok().and_then(|v| v.parse::<f32>().ok())
}

/// The game's CameraSettings (only the fields the drivable cameras read).
#[derive(Clone, Copy, Debug)]
pub struct GameSettings {
    pub horizon_lock: f32,
    pub general_movement: f32,
    pub dashcam_factor: f32,
    pub bonnet_factor: f32,
    pub lws_gain: f32,
    pub lws_gamma: f32,
    pub lws_smoothing: f32,
    /// chase cameras: the body rotation they follow is slerped toward the
    /// car's by this x dt a frame (ExternalDampedCamera 0x141be1010)
    pub chase_lag: f32,
    /// the car's onboard (non-drivable) cameras' head gain (CameraOnboard)
    pub onboard_movement: f32,
}

impl Default for GameSettings {
    /// System/camerasettings.camerasettings as shipped (2026-09 build)
    fn default() -> Self {
        GameSettings {
            horizon_lock: 0.4,
            general_movement: 0.8,
            dashcam_factor: 0.0,
            bonnet_factor: 0.1,
            lws_gain: 0.0,
            lws_gamma: 0.0,
            lws_smoothing: 0.0,
            chase_lag: 4.5,
            onboard_movement: 1.2,
        }
    }
}

impl GameSettings {
    /// CameraSettings protobuf: 11 chasecam_rotation_lag_speed, 12
    /// horizon_lock, 13 general_movement, 14 dashcam_factor, 15/16/17
    /// look_with_steer gain/gamma/smoothing, 18 bonnet_factor, 19
    /// onboard_movement. Unset floats are 0, as in the game (except the
    /// chase lag: the ExternalDampedCamera ctor sets 15 when it is 0).
    fn parse(d: &[u8]) -> GameSettings {
        use crate::proto::{fields, Val};
        let mut s = GameSettings { horizon_lock: 0.0, general_movement: 0.0, dashcam_factor: 0.0,
                                   bonnet_factor: 0.0, lws_gain: 0.0, lws_gamma: 0.0, lws_smoothing: 0.0,
                                   chase_lag: 0.0, onboard_movement: 0.0 };
        fields(d, 0, d.len(), |f, v| if let Val::F32(x) = v {
            match f {
                12 => s.horizon_lock = x,
                13 => s.general_movement = x,
                14 => s.dashcam_factor = x,
                15 => s.lws_gain = x,
                16 => s.lws_gamma = x,
                17 => s.lws_smoothing = x,
                18 => s.bonnet_factor = x,
                11 => s.chase_lag = x,
                19 => s.onboard_movement = x,
                _ => {}
            }
        });
        if s.chase_lag == 0.0 {
            s.chase_lag = 15.0;
        }
        s
    }

    /// The user's saved file if there is one, else the package's system one,
    /// else the shipped numbers.
    pub fn load(pkg: Option<&std::path::Path>) -> (GameSettings, &'static str) {
        if let Some(home) = crate::envc::var_os("USERPROFILE") {
            let p = std::path::Path::new(&home).join("Saved Games/ACE/camerasettings.camerasettings");
            if let Ok(d) = std::fs::read(&p) {
                return (Self::parse(&d), "Saved Games camerasettings");
            }
        }
        if let Some(pkg) = pkg {
            if let Ok(p) = crate::pkg::Package::open_filtered(pkg, &["system\\camerasettings"]) {
                if let Some(e) = p.by_path.get("system\\camerasettings.camerasettings") {
                    return (Self::parse(&p.read(*e)), "content.kspkg System/camerasettings");
                }
            }
        }
        (GameSettings::default(), "built-in defaults")
    }
}

/// Per-camera numbers the game derives from the settings by camera kind.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mode {
    /// head movement gain (0 = none)
    pub gain: f32,
    /// horizon lock 0..1
    pub lock: f32,
    /// look with steer on (cockpit and dash only)
    pub lws: bool,
    /// the evoview body low-pass applies (inside / on the car)
    pub lp: bool,
    /// ExternalDampedCamera (NearChase / FarChase): lagged body rotation,
    /// the horizon lock on that rotation rather than on the view
    pub chase: bool,
    /// CameraOnboard (the car's non-drivable cameras): the onboard head
    /// constants (0x141bd9730), no horizon lock
    pub onboard: bool,
}

pub struct Settings {
    pub on: bool,
    pub game: GameSettings,
    pub g_tau: f32,
    pub head_abs: bool,
    pub head_scale: f32,
    pub lp_hz: f32,
    pub lp_max_m: f32,
    pub lp_max_rad: f32,
    /// what the horizon lock levels toward: false = the world horizon (the
    /// game), true = the road under the car (its slow tilt)
    pub slope_ref: bool,
    pub slope_tau: f32,
}

impl Settings {
    pub fn new(pkg: Option<&std::path::Path>) -> Settings {
        let (mut g, from) = GameSettings::load(pkg);
        if let Some(v) = env_f("EVOVIEW_CAM_HORIZON") { g.horizon_lock = v; }
        if let Some(v) = env_f("EVOVIEW_CAM_MOVE") { g.general_movement = v; }
        if let Some(v) = env_f("EVOVIEW_CAM_DASH") { g.dashcam_factor = v; }
        if let Some(v) = env_f("EVOVIEW_CAM_BONNET") { g.bonnet_factor = v; }
        if let Some(v) = env_f("EVOVIEW_CAM_LWS_GAIN") { g.lws_gain = v; }
        if let Some(v) = env_f("EVOVIEW_CAM_LWS_GAMMA") { g.lws_gamma = v; }
        if let Some(v) = env_f("EVOVIEW_CAM_LWS_SMOOTH") { g.lws_smoothing = v; }
        if let Some(v) = env_f("EVOVIEW_CAM_CHASE_LAG") { g.chase_lag = v; }
        if let Some(v) = env_f("EVOVIEW_CAM_ONBOARD") { g.onboard_movement = v; }
        let s = Settings {
            on: crate::envc::var("EVOVIEW_CAM_HEAD").map_or(true, |v| v != "0"),
            game: g,
            g_tau: env_f("EVOVIEW_CAM_G_TAU").unwrap_or(1.0).max(0.01),
            head_abs: crate::envc::var("EVOVIEW_CAM_HEAD_ABS").map_or(true, |v| v != "0"),
            head_scale: env_f("EVOVIEW_CAM_HEAD_SCALE").unwrap_or(1.0),
            lp_hz: env_f("EVOVIEW_CAM_LP_HZ").unwrap_or(3.0).max(0.0),
            lp_max_m: env_f("EVOVIEW_CAM_LP_MAX_CM").unwrap_or(4.0) * 0.01,
            lp_max_rad: env_f("EVOVIEW_CAM_LP_MAX_DEG").unwrap_or(2.0).to_radians(),
            slope_ref: crate::envc::var("EVOVIEW_CAM_HORIZON_REF").map_or(true, |v| v != "world"),
            slope_tau: env_f("EVOVIEW_CAM_SLOPE_TAU").unwrap_or(0.15).max(0.001),
        };
        if s.on {
            println!("in-car camera ({from}): horizon lock {:.2}, movement {:.2}, dash x{:.2}, bonnet x{:.2}, \
                      look-with-steer {:.2}, chase lag {:.1}/s, onboard {:.2}; body low-pass {:.1} Hz{}; horizon levels to the {}",
                     g.horizon_lock, g.general_movement, g.dashcam_factor, g.bonnet_factor, g.lws_gain,
                     g.chase_lag, g.onboard_movement,
                     s.lp_hz, if s.head_abs { "" } else { ", signed head" },
                     if s.slope_ref { "road slope (EVOVIEW_CAM_HORIZON_REF=world: the game's world horizon)" }
                     else { "world (the game)" });
        } else {
            println!("in-car camera: rigid (EVOVIEW_CAM_HEAD=0)");
        }
        s
    }

    /// The game's camera class by CameraModeType (CameraDrivableManager
    /// 0x141bcda40 builds one per drivable camera): 1 CockpitCamera, 2
    /// ExternalDampedCamera (chase), 3 ExternalFixedCamera (bumper: rigid, no
    /// head, no lock), 4 DashCamera, 5 BonnetCamera. The car's other cameras
    /// (`drivable` false) are CameraOnboard's.
    pub fn mode(&self, kind: u32, drivable: bool) -> Mode {
        let g = &self.game;
        if !self.on {
            return Mode::default();
        }
        if !drivable {
            return Mode { gain: g.onboard_movement, lp: true, onboard: true, ..Default::default() };
        }
        match kind {
            1 => Mode { gain: g.general_movement, lock: g.horizon_lock, lws: true, lp: true, ..Default::default() },
            2 if crate::envc::var("EVOVIEW_CAM_CHASE").map_or(false, |v| v == "0") => Mode::default(),
            2 => Mode { gain: g.general_movement, lock: g.horizon_lock, lp: true, chase: true, ..Default::default() },
            4 => Mode { gain: g.general_movement * g.dashcam_factor, lock: g.horizon_lock * g.dashcam_factor,
                        lws: true, lp: true, ..Default::default() },
            5 => Mode { gain: g.general_movement * g.bonnet_factor, lock: g.horizon_lock * g.bonnet_factor,
                        lp: true, ..Default::default() },
            3 => Mode { lp: true, ..Default::default() },
            _ => Mode::default(),
        }
    }
}

/// One frame's numbers, for EVOVIEW_CAM_LOG and the report.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sample {
    pub acc_g: Vec3,
    pub head_cm: Vec3,
    /// body (pitch, roll) deg, as driven / as filtered
    pub tilt_raw: Vec2,
    pub tilt_f: Vec2,
    pub heave_raw: f32,
    pub heave_f: f32,
    pub lws_deg: f32,
}

pub struct HeadCam {
    pub s: Settings,
    prev_pos: Option<Vec3>,
    v_f: Vec3,
    acc: Vec3,
    acc_prev: Vec3,
    head: Vec3,
    // body low-pass: x = (pitch, roll, height), complementary filter
    raw_prev: Vec3,
    rate1: Vec3,
    rate2: Vec3,
    filt: Vec3,
    lws: [f32; 2],
    /// the chase camera's lagged body rotation (None: not running)
    chase_q: Option<Quat>,
    /// the road's up (slope and camber), low-passed over slope_tau
    slope_up: Vec3,
    /// this frame's road up from the physics (rigcar's hub plane), if any
    pub road_up: Option<Vec3>,
    pub last: Sample,
    log: Option<std::io::BufWriter<std::fs::File>>,
    t: f32,
}

/// A body matrix's rotation (car matrices may carry a scale)
fn rot_of(m: &Mat4) -> Quat {
    m.to_scale_rotation_translation().1.normalize()
}

/// L in the game's tanh clamp, cm (car x, y, z)
const HEAD_LIMIT: Vec3 = Vec3::new(22.0, 7.0, 10.0);
/// the same for CameraOnboard (ctor 0x141bd7290: +0x40..48 = 2, 7, 1)
const ONBOARD_LIMIT: Vec3 = Vec3::new(2.0, 7.0, 1.0);

impl HeadCam {
    pub fn new(pkg: Option<&std::path::Path>) -> HeadCam {
        let log = crate::envc::var("EVOVIEW_CAM_LOG").ok().and_then(|p| std::fs::File::create(p).ok()).map(|f| {
            use std::io::Write;
            let mut w = std::io::BufWriter::new(f);
            let _ = writeln!(w, "t,x,y,z,kmh,ax_g,ay_g,az_g,head_x_cm,head_y_cm,head_z_cm,pitch_raw,roll_raw,\
                                  pitch_f,roll_f,heave_raw,heave_f,lws_deg,eye_x,eye_y,eye_z,fwd_x,fwd_y,fwd_z,\
                                  up_x,up_y,up_z,rig_fwd_x,rig_fwd_y,rig_fwd_z,rig_up_x,rig_up_y,rig_up_z,\
                                  rig_eye_x,rig_eye_y,rig_eye_z");
            w
        });
        HeadCam {
            s: Settings::new(pkg),
            prev_pos: None,
            v_f: Vec3::ZERO,
            acc: Vec3::ZERO,
            acc_prev: Vec3::ZERO,
            head: Vec3::ZERO,
            raw_prev: Vec3::ZERO,
            rate1: Vec3::ZERO,
            rate2: Vec3::ZERO,
            filt: Vec3::ZERO,
            lws: [0.0; 2],
            chase_q: None,
            slope_up: Vec3::Y,
            road_up: None,
            last: Sample::default(),
            log,
            t: 0.0,
        }
    }

    /// The driven car is rigcar's sprung rigid body: no per-wheel bump
    /// noise to hide, and the low-pass only let the eye drift 9 cm in the
    /// cockpit (Oulton: the dash bobbed up the screen). Off unless asked for.
    pub fn sprung_body(&mut self) {
        if crate::envc::var_os("EVOVIEW_CAM_LP_HZ").is_none() && self.s.lp_hz > 0.0 {
            self.s.lp_hz = 0.0;
            println!("in-car camera: rigid-body car, body low-pass off (EVOVIEW_CAM_LP_HZ=3 for it)");
        }
    }

    /// The body's (pitch, roll, height) about its yaw-only frame: pitch +
    /// nose up, roll + left side up (car +X is left, +Z forward).
    /// The third value is the residual yaw of the tilt's scaled axis, kept
    /// as is when the body is rebuilt.
    fn tilt(body: &Mat4) -> (Quat, Vec3, f32) {
        let r = rot_of(body);
        let f = r * Vec3::Z;
        let yaw = Quat::from_rotation_y(f.x.atan2(f.z));
        let q = yaw.inverse() * r;
        // the short way round (a negative w reads as a ~360 deg turn)
        let rv = if q.w < 0.0 { -q } else { q }.to_scaled_axis();
        (yaw, Vec3::new(-rv.x, rv.z, body.w_axis.y), rv.y)
    }

    /// One physics step: the car body as drawn (world), the road-wheel steer
    /// angle in degrees (+ left) and the active camera's mode.
    pub fn step(&mut self, dt: f32, body: Mat4, steer_deg: f32, mode: Mode) {
        if dt <= 0.0 {
            return;
        }
        let pos = body.w_axis.truncate();
        let (_, raw, _) = Self::tilt(&body);
        // a teleport / reset (Enter, a wall push-out is far smaller)
        let jumped = self.prev_pos.map_or(true, |p| p.distance(pos) > 8.0);
        if jumped {
            self.prev_pos = Some(pos);
            self.v_f = Vec3::ZERO;
            self.acc = Vec3::ZERO;
            self.acc_prev = Vec3::ZERO;
            self.head = Vec3::ZERO;
            self.raw_prev = raw;
            self.rate1 = Vec3::ZERO;
            self.rate2 = Vec3::ZERO;
            self.filt = raw;
            self.lws = [0.0; 2];
            self.chase_q = None;
            self.slope_up = self.road_up.unwrap_or_else(|| rot_of(&body) * Vec3::Y);
            return;
        }
        // the road under the car if the physics knows it, else the body's
        // own tilt (the simple model's body is the tyre-contact plane)
        let up = self.road_up.unwrap_or_else(|| rot_of(&body) * Vec3::Y);
        self.slope_up = (self.slope_up + (up - self.slope_up) * (1.0 - (-dt / self.s.slope_tau).exp()))
            .normalize_or(Vec3::Y);
        // body low-pass: rate through two poles at lp_hz, integrated, pulled
        // back to the raw value with a 3x slower pole (no lag on a slope),
        // never further than LP_MAX from it
        if self.s.lp_hz > 0.0 {
            let w = std::f32::consts::TAU * self.s.lp_hz;
            let a1 = 1.0 - (-w * dt).exp();
            let a2 = 1.0 - (-w / 3.0 * dt).exp();
            let rate = (raw - self.raw_prev) / dt;
            self.rate1 += (rate - self.rate1) * a1;
            self.rate2 += (self.rate1 - self.rate2) * a1;
            self.filt += self.rate2 * dt;
            self.filt += (raw - self.filt) * a2;
            let m = Vec3::new(self.s.lp_max_rad, self.s.lp_max_rad, self.s.lp_max_m);
            self.filt = self.filt.clamp(raw - m, raw + m);
        } else {
            self.filt = raw;
        }
        self.raw_prev = raw;
        self.t += dt;
        // accG as the physics reports it: (v - v_lowpassed) / g, car space.
        // The game's v is a sprung body's; ours is taken on the low-passed
        // heave (the raw one is per-wheel bump noise: +-3 g vertical).
        let pos = Vec3::new(pos.x, self.filt.z, pos.z);
        let v = (pos - self.prev_pos.unwrap()) / dt;
        self.prev_pos = Some(pos);
        let tau = self.s.g_tau;
        self.v_f += (v - self.v_f) * (1.0 - (-dt / tau).exp());
        let a_world = (v - self.v_f) / tau;
        let rot = Mat3::from_quat(rot_of(&body));
        self.acc_prev = self.acc;
        self.acc = rot.transpose() * a_world / 9.806;

        // head movement (0x141be19e0; the onboard cameras' 0x141bd9730 is
        // the same with accG clamped to +-2 g, rate (gain*3+1)/s and limits
        // 2/7/1 cm). The game takes d per rendered frame; here per 1/60 s so
        // the jerk terms do not shrink with the frame rate.
        if mode.gain > 0.0 {
            let (mut a, mut ap) = (self.acc, self.acc_prev);
            if mode.onboard {
                a = a.clamp(Vec3::splat(-2.0), Vec3::splat(2.0));
                ap = ap.clamp(Vec3::splat(-2.0), Vec3::splat(2.0));
            }
            let d = (a - ap) * ((1.0 / 60.0) / dt);
            let target = Vec3::new(-2.0 * d.x, -(a.y + d.y), -(a.z + d.z)) * 3.6;
            let rate = if mode.onboard { mode.gain * 3.0 + 1.0 } else { (mode.gain * 10.0 + 1.0) / 3.0 };
            let k = (rate * dt).clamp(0.0, 1.0);
            self.head += (target - self.head) * k;
            let lim = if mode.onboard { ONBOARD_LIMIT } else { HEAD_LIMIT };
            for i in 0..3 {
                let l = lim[i];
                let p = self.head[i];
                let t = l * (p / l).tanh();
                self.head[i] = if self.s.head_abs { t.abs() } else { t };
            }
        }

        // look with steer (0x141be1e30): road-wheel degrees
        let g = &self.s.game;
        if mode.lws && g.lws_gain > 0.0 {
            let x = (steer_deg.abs() / 10.0 + 0.2).clamp(0.0, 1.0);
            let target = x.powf(g.lws_gamma + 1.0) * 4.0 * g.lws_gain * steer_deg;
            let k1 = ((12.5 - g.lws_smoothing * 8.0) * dt).clamp(0.0, 1.0);
            self.lws[0] = (self.lws[0] + (target - self.lws[0]) * k1).clamp(-40.0, 40.0);
            let k2 = (5.0 / (g.lws_smoothing + 0.5) * dt).clamp(0.0, 1.0);
            self.lws[1] += (self.lws[0] - self.lws[1]) * k2;
        } else {
            self.lws = [0.0; 2];
        }

        // ExternalDampedCamera's lag: slerp(prev, body, min(lag*dt, 1)) a
        // frame in the game; 1 - exp(-lag*dt) here (the same at small dt, and
        // frame-rate independent). Restarts when the camera is picked
        // (0x141bdfcf0 clears the 'has previous' flag).
        if mode.chase {
            let rb = rot_of(&self.body_filtered(body, mode));
            let k = 1.0 - (-self.s.game.chase_lag.max(0.0) * dt).exp();
            self.chase_q = Some(self.chase_q.map_or(rb, |p| p.slerp(rb, k).normalize()));
        } else {
            self.chase_q = None;
        }
        self.last = Sample {
            acc_g: self.acc,
            head_cm: self.head,
            tilt_raw: Vec2::new(raw.x, raw.y) * 57.29578,
            tilt_f: Vec2::new(self.filt.x, self.filt.y) * 57.29578,
            heave_raw: raw.z,
            heave_f: self.filt.z,
            lws_deg: self.lws[1],
        };
    }

    /// The game's horizon lock (0x141be21d0): slerp toward the rotation's
    /// heading alone by `lock`. The game levels to the world horizon, so
    /// 0.4 of every hill and camber is taken out of the view and the dash
    /// climbs the screen uphill; with slope_ref the level is the road's slow
    /// tilt instead, and only the car's own pitch and roll on it are locked.
    fn level(&self, q: Quat, lock: f32) -> Quat {
        if lock <= 0.0 {
            return q;
        }
        let s = if self.s.slope_ref { Quat::from_rotation_arc(Vec3::Y, self.slope_up) } else { Quat::IDENTITY };
        let qr = s.inverse() * q;
        let f = qr * -Vec3::Z;
        let level = Quat::from_rotation_y((-f.x).atan2(-f.z));
        (s * qr.slerp(level, lock.min(1.0))).normalize()
    }

    /// The body the camera is mounted on: the drawn one with its heave,
    /// pitch and roll low-passed.
    fn body_filtered(&self, body: Mat4, mode: Mode) -> Mat4 {
        if !mode.lp || self.s.lp_hz <= 0.0 || !self.s.on {
            return body;
        }
        let (yaw, raw, spin) = Self::tilt(&body);
        let d = self.filt - raw;
        if !d.is_finite() {
            return body;
        }
        // rebuild: yaw * tilt_f, tilt as the scaled axis (pitch about +X is
        // nose DOWN in the right-handed car frame, hence -pitch)
        let tilt = Quat::from_scaled_axis(Vec3::new(-self.filt.x, spin, self.filt.y));
        let r = yaw * tilt;
        let scale = Vec3::new(body.x_axis.truncate().length(), body.y_axis.truncate().length(),
                              body.z_axis.truncate().length());
        let mut p = body.w_axis.truncate();
        p.y += d.z;
        Mat4::from_scale_rotation_translation(scale, r, p)
    }

    /// The car camera's view: like cams::car_view, plus the game's head
    /// offset, horizon lock and look with steer on the low-passed body.
    /// `old_body` is what the rigid camera rode (logged for comparison).
    pub fn view(&mut self, c: &crate::fxdata::CarCam, body: Mat4, old_body: Mat4, aspect: f32, kmh: f32)
                -> (Mat4, Vec3) {
        let mode = self.s.mode(c.mode, c.drivable);
        let r = c.rot;
        let rot = Quat::from_rotation_y(r[1].to_radians()) * Quat::from_rotation_x(r[0].to_radians())
            * Quat::from_rotation_z(r[2].to_radians());
        let rot = rot * Quat::from_rotation_y(self.lws[1].to_radians() * (mode.lws as i32 as f32));
        let b = self.body_filtered(body, mode);
        let head = if mode.gain > 0.0 { self.head * 0.01 * self.s.head_scale } else { Vec3::ZERO };
        let rb = rot_of(&b);
        let (eye, q) = match self.chase_q.filter(|_| mode.chase) {
            // ExternalDampedCamera: car position, lagged and levelled body
            // rotation, then the camera's own offset (+ head) and angles
            Some(qc) => {
                let ql = self.level(qc, mode.lock);
                let scale = Vec3::new(b.x_axis.truncate().length(), b.y_axis.truncate().length(),
                                      b.z_axis.truncate().length());
                (b.w_axis.truncate() + ql * ((Vec3::from(c.pos) + head) * scale), (ql * rot).normalize())
            }
            // the in-car ones: rigid on the body, the lock on the view
            None => (b.transform_point3(Vec3::from(c.pos) + head), self.level((rb * rot).normalize(), mode.lock)),
        };
        let fwd = q * -Vec3::Z;
        let up = q * Vec3::Y;
        let near = if c.mode == 1 || c.mode == 4 { 0.03 } else { 0.1 };
        let proj = Mat4::perspective_infinite_rh(c.fov.max(10.0).to_radians(), aspect.max(0.01), near);
        if self.log.is_some() {
            let (_, re) = crate::cams::car_view(c, old_body, aspect);
            let rr = rot_of(&old_body) * rot;
            let (rf, ru) = (rr * -Vec3::Z, rr * Vec3::Y);
            let s = self.last;
            let p = body.w_axis;
            use std::io::Write;
            let w = self.log.as_mut().unwrap();
            let _ = writeln!(w, "{:.4},{:.3},{:.3},{:.3},{:.2},{:.4},{:.4},{:.4},{:.3},{:.3},{:.3},{:.4},{:.4},{:.4},{:.4},\
                                 {:.4},{:.4},{:.3},{:.4},{:.4},{:.4},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},\
                                 {:.5},{:.5},{:.5},{:.4},{:.4},{:.4}",
                self.t, p.x, p.y, p.z, kmh, s.acc_g.x, s.acc_g.y, s.acc_g.z, s.head_cm.x, s.head_cm.y, s.head_cm.z,
                s.tilt_raw.x, s.tilt_raw.y, s.tilt_f.x, s.tilt_f.y, s.heave_raw, s.heave_f, s.lws_deg,
                eye.x, eye.y, eye.z, fwd.x, fwd.y, fwd.z, up.x, up.y, up.z, rf.x, rf.y, rf.z, ru.x, ru.y, ru.z,
                re.x, re.y, re.z);
        }
        (proj * Mat4::look_at_rh(eye, eye + fwd, up), eye)
    }
}

/// Josh's own camera setup from the game: Saved Games/ACE/
/// CarCameras.carcamerausersettings (CarCameraUserSettings, read only).
/// drivableSettings f1: fovs per DrivableCameraType (Cockpit, Dash, Hood,
/// Bumper, Chase, FarChase), f2: map car id -> CarCameraInfo {f1 positions,
/// f2 rotations (deg), per type}. EVOVIEW_CAM_USER=0 keeps the car's own.
pub fn apply_user(cams: &mut [crate::fxdata::CarCam], car_id: &str) {
    use crate::proto::{fields, text, Val};
    if crate::envc::var("EVOVIEW_CAM_USER").map_or(false, |v| v == "0") {
        return;
    }
    let Some(home) = crate::envc::var_os("USERPROFILE") else { return };
    let Ok(d) = std::fs::read(std::path::Path::new(&home).join("Saved Games/ACE/CarCameras.carcamerausersettings"))
        else { return };
    let floats = |d: &[u8], v: Val, out: &mut Vec<f32>| match v {
        Val::F32(x) => out.push(x),
        Val::Bytes(a, b) => out.extend(d[a..b].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()))),
        _ => {}
    };
    let vec3 = |d: &[u8], a: usize, b: usize| {
        let mut v = [0.0f32; 3];
        fields(d, a, b, |f, x| if let (1..=3, Val::F32(q)) = (f, x) { v[f as usize - 1] = q });
        v
    };
    let mut fovs = Vec::new();
    let (mut pos, mut rot) = (Vec::new(), Vec::new());
    fields(&d, 0, d.len(), |f, v| if let (1, Some((a, b))) = (f, v.range()) {
        fields(&d, a, b, |g, w| match (g, w.range()) {
            (1, _) => floats(&d, w, &mut fovs),
            (2, Some((ea, eb))) => {
                let mut key = String::new();
                let mut info = None;
                fields(&d, ea, eb, |h, x| match (h, x.range()) {
                    (1, Some(r)) => key = text(&d, r),
                    (2, Some(r)) => info = Some(r),
                    _ => {}
                });
                if let (true, Some((ia, ib))) = (key.eq_ignore_ascii_case(car_id), info) {
                    fields(&d, ia, ib, |h, x| match (h, x.range()) {
                        (1, Some((va, vb))) => pos.push(vec3(&d, va, vb)),
                        (2, Some((va, vb))) => rot.push(vec3(&d, va, vb)),
                        _ => {}
                    });
                }
            }
            _ => {}
        });
    });
    let mut n = 0;
    for c in cams.iter_mut().filter(|c| c.drivable) {
        let i = match c.name.to_ascii_lowercase().as_str() {
            "cockpit" => 0,
            "dash" => 1,
            "bonnet" | "hood" => 2,
            "bumper" => 3,
            "nearchase" | "chase" => 4,
            "farchase" => 5,
            _ => continue,
        };
        let before = (c.pos, c.rot, c.fov);
        if let Some(&f) = fovs.get(i).filter(|f| **f > 1.0) { c.fov = f; }
        if let (Some(&p), Some(&r)) = (pos.get(i), rot.get(i)) {
            c.pos = p;
            c.rot = r;
        }
        if before != (c.pos, c.rot, c.fov) {
            n += 1;
            if i == 0 {
                println!("  your cockpit camera (CarCameras): pos {:?} rot {:?} fov {} (car: {:?} {:?} {})",
                         c.pos, c.rot, c.fov, before.0, before.1, before.2);
            }
        }
    }
    if n > 0 {
        println!("  {n} car camera(s) from your game settings (EVOVIEW_CAM_USER=0: the car's own)");
    }
}

/// `--drive-test` with EVOVIEW_CAM_LOG: the cockpit camera stepped on the
/// headless drive (pose with its spring lean, no rig), for whole-lap logs
/// without a window. EVOVIEW_CAM_TEST_POS/ROT override the seat (x,y,z).
pub struct Headless {
    cam: HeadCam,
    c: crate::fxdata::CarCam,
}

impl Headless {
    pub fn from_env() -> Option<Headless> {
        crate::envc::var_os("EVOVIEW_CAM_LOG")?;
        let v3 = |k: &str, d: [f32; 3]| crate::envc::var(k).ok().and_then(|v| {
            let p: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (p.len() == 3).then(|| [p[0], p[1], p[2]])
        }).unwrap_or(d);
        // ks_bmw_m4_gt3's cockpit (Josh's CarCameras = the car's own)
        let c = crate::fxdata::CarCam {
            name: "Cockpit".into(),
            pos: v3("EVOVIEW_CAM_TEST_POS", [0.31, 0.97, -0.45]),
            rot: v3("EVOVIEW_CAM_TEST_ROT", [0.7, 180.0, 0.0]),
            fov: 50.0,
            mode: 1,
            drivable: true,
        };
        Some(Headless { cam: HeadCam::new(None), c })
    }

    pub fn step(&mut self, td: &crate::trackdrive::TrackDrive, dt: f32) {
        let body = td.matrices().0;
        let m = self.cam.s.mode(1, true);
        self.cam.step(dt, body, td.ffb_inputs().1.to_degrees(), m);
        self.cam.view(&self.c, body, td.body_flat(), 16.0 / 9.0, td.speed_kmh());
    }
}

/// EVOVIEW_CAM_VLOG=path: every driven frame's view, whatever the camera
/// (car, TV, viewer chase), against the car body as drawn - for comparing
/// the cameras' pitch / roll / jitter. Angles in degrees, pitch + nose up,
/// roll + left side up; rel_* = the view in the body's frame (rel_pitch
/// below its rest value = the camera dips in the car: the dash rises).
pub struct ViewLog {
    w: std::io::BufWriter<std::fs::File>,
    t: f32,
}

impl ViewLog {
    pub fn from_env() -> Option<ViewLog> {
        let f = std::fs::File::create(crate::envc::var_os("EVOVIEW_CAM_VLOG")?).ok()?;
        let mut w = std::io::BufWriter::new(f);
        use std::io::Write;
        let _ = writeln!(w, "t,cam,kmh,x,y,z,body_pitch,body_roll,view_pitch,view_roll,rel_pitch,rel_roll,rel_yaw,eye_x,eye_y,eye_z,eye_car_x,eye_car_y,eye_car_z");
        Some(ViewLog { w, t: 0.0 })
    }

    pub fn log(&mut self, dt: f32, cam: &str, kmh: f32, body: Mat4, view_proj: Mat4, eye: Vec3) {
        use std::io::Write;
        self.t += dt;
        let inv = view_proj.inverse();
        let at = |x: f32, y: f32| { let a = inv * glam::Vec4::new(x, y, 0.5, 1.0); a.truncate() / a.w };
        let fwd = (at(0.0, 0.0) - eye).normalize_or_zero();
        let upv = at(0.0, 1.0) - eye;
        let up = (upv - fwd * upv.dot(fwd)).normalize_or_zero();
        let rb = rot_of(&body);
        // the car's forward is +Z, the view's -Z; compare like with like
        let (bf, bu) = (rb * Vec3::Z, rb * Vec3::Y);
        let pitch = |f: Vec3| f.y.clamp(-1.0, 1.0).asin().to_degrees();
        // roll: the up vector's tilt across the forward direction
        let roll = |f: Vec3, u: Vec3| {
            let l = Vec3::Y.cross(f).normalize_or_zero(); // horizontal left
            (-u.dot(l)).atan2(u.dot(Vec3::Y - f * f.y).max(1e-6)).to_degrees()
        };
        // the view in the body's frame
        let lf = rb.inverse() * fwd;
        let lu = rb.inverse() * up;
        let rel_pitch = lf.y.clamp(-1.0, 1.0).asin().to_degrees();
        let rel_yaw = lf.x.atan2(lf.z).to_degrees();
        let ll = Vec3::Y.cross(lf).normalize_or_zero();
        let rel_roll = (-lu.dot(ll)).atan2(lu.y.max(1e-6)).to_degrees();
        let ec = body.inverse().transform_point3(eye);
        let p = body.w_axis;
        let _ = writeln!(self.w, "{:.4},{},{:.2},{:.3},{:.3},{:.3},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            self.t, cam, kmh, p.x, p.y, p.z, pitch(bf), roll(bf, bu), pitch(fwd), roll(fwd, up),
            rel_pitch, rel_roll, rel_yaw, eye.x, eye.y, eye.z, ec.x, ec.y, ec.z);
    }
}
