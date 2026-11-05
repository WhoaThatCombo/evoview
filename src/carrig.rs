//! Driving a car's skeleton the way the game does.
//!
//! WHAT BINDS AN ANIMATION TO AN INPUT
//!   The `.actor` (CarActorData, field 103 of the file) says it, slot by slot -
//!   the clip names mean nothing to the engine:
//!
//!   f27 car_animations     102 wiper, 103 hood, 104 tailgate, 105-108 doors
//!                          FL FR RL RR, 119 lights (pop-ups), 109-112 suspension
//!                          FL FR RL RR, 113-115 pedals accel/brake/clutch,
//!                          116/117 gear lever up/down, 121/122 paddles up/down,
//!                          120 handbrake, 118 optional animations (wing, DRS,
//!                          splitter, sunroof...), 130 skip_physics_camber
//!   f31 driver_animations  109 base pose, 101 steer, 114 steer with paddles,
//!                          102-104 pedals, 105/106 shift change, 107 manual
//!                          shift approach, 112/113 paddles, 110 handbrake,
//!                          115 idle, 111 optional
//!   f210 driver_animation_blendings  hshifter / paddle blend timings
//!   f36 movable_parts      bone wobble with speed / g (wings, nets)
//!   f103 analogic_script   .caranalogicsystem: dash needles
//!
//!   Each slot is a GenericRendererAnimation {1 animations {1 path} (several:
//!   ext + int halves of one door), 2 speed_multiplier, 13 play_type,
//!   14 lut_normalized_positions}. play_type says how the input drives it:
//!     PulseAndReverse  the input value IS the playhead (pedals, steer)
//!     PulseFull        fire and play through on an event (a shift)
//!     Looping          loop while on (wipers)
//!     SequentialSimple open / close toggle (doors, hood, pop-up lights)
//!     SequentialLut    a value mapped through a LUT onto stops (wings, DRS)
//!   An optional animation's specialisation (f31 wing) names its input:
//!   setup wing angle, speed km/h, g force, pedals/steer, engine rpm, a
//!   static wing that only DRS moves (has_drs, value_when_drs_is_active).
//!
//! WHAT IS PROCEDURAL
//!   Nothing in the content animates the wheels or the steering wheel. The
//!   engine moves bones by name: HUB_xx follows the suspension (this module
//!   uses the physics hard points, carphys.rs), WHEEL_/TYRE_/DISC_xx spin
//!   under it while SUSP_xx (calipers) does not, STEER_BASE turns by the
//!   steering-wheel angle, and every bone with a `DIR_<bone>` partner is a
//!   look-at: it re-aims at that partner, which is what makes wishbones, tie
//!   rods and dampers follow the hub. A car with suspension clips (F2004,
//!   SF-25, Lotus Emira...) drives its HUB from the clip instead, the clip
//!   frame chosen so the hub height matches the travel.

use crate::carphys::{Geometry, Physics, SuspKind};
use crate::carstate::CarState;
use crate::load::{base_of, read_animation, Animation, Skeleton, Vfs, IDENTITY};
use crate::proto::{fields, text, Val};
use glam::{Mat4, Quat, Vec3};

pub const CORNERS: [&str; 4] = ["LF", "RF", "LR", "RR"];

/// Steering-wheel degrees a driver steer clip covers from its middle frame to
/// either end (see slot_frame).
pub const DRIVER_STEER_RANGE_DEG: f32 = 360.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Play {
    PulseFull,
    PulseAndReverse,
    Looping,
    SequentialSimple,
    SequentialLut,
}

impl Play {
    fn from(v: u64) -> Play {
        match v {
            1 => Play::PulseAndReverse,
            2 => Play::Looping,
            3 => Play::SequentialSimple,
            4 => Play::SequentialLut,
            _ => Play::PulseFull,
        }
    }
}

/// What an optional animation listens to.
#[derive(Clone, Debug)]
pub enum OptDrive {
    /// plain toggle (sunroof, roof, start button)
    Toggle,
    /// LUT over speed km/h
    Speed(Vec<(f32, f32)>),
    /// LUT over g (greatest of the three axes)
    G(Vec<(f32, f32)>),
    /// LUTs over gas / brake / steer, greatest value wins
    Inputs(Vec<(f32, f32)>, Vec<(f32, f32)>, Vec<(f32, f32)>),
    Rpm(Vec<(f32, f32)>),
    /// a static wing that DRS opens to this value
    Drs(f32),
    /// set from the car setup's wing angle: shown at its middle stop
    Setup,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Wiper,
    Hood,
    Tailgate,
    Door(usize),
    Lights,
    Susp(usize),
    Accel,
    Brake,
    Clutch,
    ShiftUp,
    ShiftDown,
    PaddleUp,
    PaddleDown,
    Handbrake,
    Optional(String),
    DrvBase,
    DrvSteer,
    DrvSteerPaddles,
    DrvAccel,
    DrvBrake,
    DrvClutch,
    DrvShiftUp,
    DrvShiftDown,
    DrvApproach,
    DrvPaddleUp,
    DrvPaddleDown,
    DrvHandbrake,
    DrvIdle,
}

impl Kind {
    pub fn label(&self) -> String {
        match self {
            Kind::Door(i) => format!("door_{}", ["fl", "fr", "rl", "rr"][*i]),
            Kind::Susp(i) => format!("suspension_{}", CORNERS[*i].to_lowercase()),
            Kind::Optional(n) => format!("optional:{n}"),
            k => format!("{k:?}").to_lowercase(),
        }
    }
    /// The driver's slots (the Drv* kinds). ⚠ Was `format!("{self:?}")
    /// .starts_with("Drv")`: a string built per slot per pose, ~10% of a
    /// 53-car replay's frame.
    pub fn is_driver(&self) -> bool {
        matches!(self, Kind::DrvBase | Kind::DrvSteer | Kind::DrvSteerPaddles | Kind::DrvAccel | Kind::DrvBrake
            | Kind::DrvClutch | Kind::DrvShiftUp | Kind::DrvShiftDown | Kind::DrvApproach | Kind::DrvPaddleUp
            | Kind::DrvPaddleDown | Kind::DrvHandbrake | Kind::DrvIdle)
    }
}

/// One bound animation slot.
#[derive(Clone)]
pub struct Slot {
    pub kind: Kind,
    pub play: Play,
    pub speed: f32,
    pub lut: Vec<f32>,
    pub opt: Option<OptDrive>,
    pub paths: Vec<String>,
    /// paths the actor names that do not resolve
    pub missing: Vec<String>,
    /// tracks whose bone the skeleton does not have
    pub orphan_tracks: Vec<String>,
    /// the clips, merged into one take; tracks name skeleton bones
    pub anim: Option<Animation>,
    /// track index -> skeleton bone index
    track_bone: Vec<Option<usize>>,
    /// Suspension clips: hub height (car space) per frame, for the
    /// travel -> frame lookup, and the rest height.
    hub_y: Vec<f32>,
}

impl Slot {
    pub fn frames(&self) -> usize {
        self.anim.as_ref().map_or(0, |a| a.frames)
    }
    pub fn len_s(&self) -> f32 {
        self.anim.as_ref().map_or(0.0, |a| a.frames as f32 / a.fps.max(1.0) / self.speed.max(0.05))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gauge {
    Rpm,
    Speed,
    Fuel,
    FuelPressure,
    WaterPressure,
    OilPressure,
    Battery,
    Turbo,
    WaterTemp,
    OilTemp,
    TurboLimiter,
    Placeholder,
    Hours,
    Minutes,
    Seconds,
}

#[derive(Clone)]
pub struct Needle {
    pub gauge: Gauge,
    pub bone_name: String,
    pub bone: Option<usize>,
    pub axis: usize,
    /// (input value, degrees)
    pub lut: Vec<(f32, f32)>,
}

#[derive(Clone)]
pub struct Osc {
    speed_min: f32,
    speed_max: f32,
    freq: Vec3,
    rot: Vec3,
    trans: Vec3,
}

#[derive(Clone)]
pub struct Movable {
    pub bone_name: String,
    pub bone: Option<usize>,
    osc: Vec<Osc>,
    g_min: f32,
    g_max: f32,
    g_rot: Vec3,
    g_trans: Vec3,
}

/// One wheel corner: what moves and how.
#[derive(Clone)]
pub struct Corner {
    /// 0..3 = LF RF LR RR (index into Inputs.travel / spin / road)
    pub idx: usize,
    pub hub: Option<usize>,
    /// bones that spin with the wheel (WHEEL, TYRE, DISC) and each one's
    /// axle in its own local frame
    pub spin: Vec<(usize, Vec3)>,
    /// wheel centre at rest, car space
    pub centre: Vec3,
    /// kingpin through this point along this axis, car space
    pub kingpin: (Vec3, Vec3),
    /// travel (m, + = bump) -> (wheel-centre offset, upright rotation), car space
    pub lut: Vec<(f32, Vec3, Quat)>,
    pub travel: (f32, f32),
    /// index into Rig.slots of this corner's suspension clip
    pub clip: Option<usize>,
    /// +X side of the car
    #[allow(dead_code)]
    pub left: bool,
    pub front: bool,
}

#[derive(Clone)]
struct Aim {
    bone: usize,
    target: usize,
    /// direction to the target in the bone's own frame at rest
    dir_local: Vec3,
}

#[derive(Clone, Copy)]
pub struct Blend {
    pub h_in: f32,
    pub h_play: f32,
    pub h_out: f32,
    pub p_in: f32,
    pub p_play: f32,
    pub p_out: f32,
}

pub struct Rig {
    pub slots: Vec<Slot>,
    pub needles: Vec<Needle>,
    pub movables: Vec<Movable>,
    pub corners: Vec<Corner>,
    pub steer_bone: Option<usize>,
    /// steering column axis in STEER_BASE's local frame, pointing forward
    pub steer_axis: Vec3,
    /// the column's pivot in STEER_BASE's local frame (origin unless the
    /// wheel's own geometry says otherwise, see fit_steering)
    pub steer_pivot: Vec3,
    aims: Vec<Aim>,
    pub phys: Option<Physics>,
    /// half-range of the driver's steer clip in steering-wheel degrees, from
    /// the hands' grip slope (diagnostic; ~300 on stock clips)
    pub drv_steer_half_deg: f32,
    /// steering-wheel degrees the driver steer clip maps to from middle to
    /// end: 360 for clips that come full circle (all stock cars), else the
    /// grip slope scaled by the stock calibration
    pub drv_steer_range_deg: f32,
    pub blend: Blend,
    pub tyre_radius: f32,
    pub skip_physics_camber: bool,
    pub analog_script: Option<String>,
    /// load-time problems, for --check
    pub notes: Vec<String>,
    rest_local: Vec<[f32; 16]>,
    parent: Vec<Option<usize>>,
    /// bones in parent-before-child order
    order: Vec<usize>,
    bind: Vec<Mat4>,
    inv_bind: Vec<Mat4>,
    pub bone_names: Vec<String>,
}

/// Everything that drives the rig for one frame (drive.rs builds it).
#[derive(Clone)]
pub struct Inputs {
    pub cs: CarState,
    /// suspension travel per corner (LF RF LR RR), metres, + = bump
    pub travel: [f32; 4],
    /// wheel rotation per corner, radians (+ = rolling forward)
    pub spin: [f32; 4],
    /// steering-wheel angle, degrees (+ = right)
    pub wheel_deg: f32,
    /// road-wheel steer per corner, radians (+ = right)
    pub road: [f32; 4],
    /// body motion on top of the rest pose (pitch / roll / heave), car space
    pub body: Mat4,
    /// open 0..1 per slot index (doors, hood, lights, optional toggles)
    pub open: Vec<f32>,
    /// seconds the wipers have been running, None = parked
    pub wiper: Option<f32>,
    /// gear change in progress: (up?, seconds since it began)
    pub shift: Option<(bool, f32)>,
    pub paddles: bool,
    pub drs: bool,
    /// acceleration in g: (lateral, longitudinal, vertical)
    pub g: Vec3,
    pub time: f32,
    pub fuel_l: f32,
    pub water_c: f32,
    pub oil_c: f32,
}

impl Inputs {
    pub fn rest(n_slots: usize) -> Inputs {
        Inputs {
            cs: CarState::default(),
            travel: [0.0; 4],
            spin: [0.0; 4],
            wheel_deg: 0.0,
            road: [0.0; 4],
            body: Mat4::IDENTITY,
            open: vec![0.0; n_slots],
            wiper: None,
            shift: None,
            paddles: false,
            drs: false,
            g: Vec3::ZERO,
            time: 0.0,
            fuel_l: 30.0,
            water_c: 85.0,
            oil_c: 95.0,
        }
    }
}

fn m4(a: &[f32; 16]) -> Mat4 {
    Mat4::from_cols_array(a)
}

fn lut_eval(lut: &[(f32, f32)], x: f32) -> f32 {
    match lut.len() {
        0 => 0.0,
        1 => lut[0].1,
        _ => {
            if x <= lut[0].0 {
                return lut[0].1;
            }
            for w in lut.windows(2) {
                if x <= w[1].0 {
                    let t = (x - w[0].0) / (w[1].0 - w[0].0).max(1e-6);
                    return w[0].1 + (w[1].1 - w[0].1) * t;
                }
            }
            lut[lut.len() - 1].1
        }
    }
}

fn vec2_list(d: &[u8], s: usize, e: usize, want: u32) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    fields(d, s, e, |f, v| {
        if f == want {
            if let Some((a, b)) = v.range() {
                let mut p = (0.0, 0.0);
                fields(d, a, b, |g, w| match g {
                    1 => p.0 = w.f32(),
                    2 => p.1 = w.f32(),
                    _ => {}
                });
                out.push(p);
            }
        }
    });
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn vec3_of(d: &[u8], s: usize, e: usize) -> Vec3 {
    let mut v = Vec3::ZERO;
    fields(d, s, e, |f, x| {
        if (1..=3).contains(&f) {
            v[(f - 1) as usize] = x.f32();
        }
    });
    v
}

struct RawSlot {
    kind: Kind,
    play: Play,
    speed: f32,
    lut: Vec<f32>,
    paths: Vec<String>,
    opt: Option<OptDrive>,
}

/// A GenericRendererAnimation.
fn read_generic(d: &[u8], s: usize, e: usize, kind: Kind) -> RawSlot {
    let mut r = RawSlot { kind, play: Play::PulseFull, speed: 1.0, lut: Vec::new(), paths: Vec::new(), opt: None };
    fields(d, s, e, |f, v| match (f, v) {
        (1, Val::Bytes(a, b)) => fields(d, a, b, |g, w| {
            if let (1, Some(rg)) = (g, w.range()) {
                r.paths.push(text(d, rg));
            }
        }),
        (2, Val::F32(x)) if x > 0.0 => r.speed = x,
        (13, Val::Varint(x)) => r.play = Play::from(x),
        (14, Val::Bytes(a, b)) => {
            for c in d[a..b].chunks_exact(4) {
                r.lut.push(f32::from_le_bytes(c.try_into().unwrap()));
            }
        }
        (14, Val::F32(x)) => r.lut.push(x),
        _ => {}
    });
    r
}

/// A GenericRendererOptionalAnimation.
fn read_optional(d: &[u8], s: usize, e: usize, driver: bool) -> Option<RawSlot> {
    let mut name = String::new();
    let mut gen: Option<(usize, usize)> = None;
    let mut opt = OptDrive::Toggle;
    fields(d, s, e, |f, v| match (f, v) {
        (1, Val::Bytes(a, b)) => name = text(d, (a, b)),
        (2, Val::Bytes(a, b)) => gen = Some((a, b)),
        (31, Val::Bytes(a, b)) => {
            let mut drs_value = None;
            let mut has_drs = false;
            fields(d, a, b, |g, w| match (g, w) {
                (1, _) => opt = OptDrive::Setup,
                (2, Val::Bytes(x, y)) => opt = OptDrive::Speed(vec2_list(d, x, y, 1)),
                (3, Val::Bytes(x, y)) => {
                    let mut l = vec2_list(d, x, y, 1);
                    if l.is_empty() { l = vec2_list(d, x, y, 2); }
                    if l.is_empty() { l = vec2_list(d, x, y, 3); }
                    opt = OptDrive::G(l);
                }
                (4, Val::Bytes(x, y)) => {
                    opt = OptDrive::Inputs(vec2_list(d, x, y, 1), vec2_list(d, x, y, 2), vec2_list(d, x, y, 3))
                }
                (5, Val::Bytes(x, y)) => opt = OptDrive::Rpm(vec2_list(d, x, y, 1)),
                (7, _) => opt = OptDrive::Drs(1.0),
                (10, Val::Varint(x)) => has_drs = x != 0,
                (11, Val::F32(x)) => drs_value = Some(x),
                _ => {}
            });
            if has_drs {
                opt = OptDrive::Drs(drs_value.unwrap_or(1.0));
            }
        }
        _ => {}
    });
    let (a, b) = gen?;
    let _ = driver;
    let mut r = read_generic(d, a, b, Kind::Optional(name));
    r.opt = Some(opt);
    Some(r)
}

/// The actor's CarActorData (field 103 of the .actor file).
fn actor_data(vfs: &mut Vfs) -> Option<(Vec<u8>, usize, usize)> {
    let a = vfs.list("", "actor").into_iter().next()?;
    let d = vfs.read(&a);
    let mut r = None;
    fields(&d, 0, d.len(), |f, v| {
        if let (103, Some(rg)) = (f, v.range()) {
            r = Some(rg);
        }
    });
    let (s, e) = r?;
    Some((d, s, e))
}

impl Rig {
    pub fn build(vfs: &mut Vfs, skel: &Skeleton, meshes: &[crate::asset::MeshEntry], blob: &[u8]) -> Rig {
        let nb = skel.bones.len();
        let bind: Vec<Mat4> = skel.bind.iter().map(m4).collect();
        let inv_bind: Vec<Mat4> = skel.inv_bind.iter().map(m4).collect();
        let parent: Vec<Option<usize>> = skel.bones.iter().map(|b| b.parent).collect();
        // parent-before-child order
        let mut order = Vec::with_capacity(nb);
        let mut placed = vec![false; nb];
        let mut guard = 0;
        while order.len() < nb && guard < 300 {
            guard += 1;
            for i in 0..nb {
                if !placed[i] && parent[i].map_or(true, |p| placed[p] || p >= nb) {
                    placed[i] = true;
                    order.push(i);
                }
            }
        }
        for i in 0..nb {
            if !placed[i] {
                order.push(i);
            }
        }
        let mut rig = Rig {
            slots: Vec::new(),
            needles: Vec::new(),
            movables: Vec::new(),
            corners: Vec::new(),
            steer_bone: None,
            steer_axis: Vec3::Z,
            steer_pivot: Vec3::ZERO,
            aims: Vec::new(),
            phys: Physics::read(vfs),
            drv_steer_half_deg: 0.0,
            drv_steer_range_deg: DRIVER_STEER_RANGE_DEG,
            blend: Blend { h_in: 0.2, h_play: 0.3, h_out: 0.3, p_in: 0.08, p_play: 0.15, p_out: 0.12 },
            tyre_radius: 0.32,
            skip_physics_camber: false,
            analog_script: None,
            notes: Vec::new(),
            rest_local: skel.bones.iter().map(|b| b.local).collect(),
            parent,
            order,
            bind,
            inv_bind,
            bone_names: skel.bones.iter().map(|b| b.name.clone()).collect(),
        };
        if rig.phys.is_none() {
            rig.notes.push("no CarData (data/*.car) - suspension and steering use defaults".into());
        }
        // tyre radius from the tyre meshes
        let mut r = 0.0f32;
        for m in meshes.iter().filter(|m| m.name.starts_with("tyre_")) {
            if let Some([o, l]) = m.positions {
                let (o, l) = (o as usize, l as usize);
                if o + l <= blob.len() {
                    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                    for c in blob[o..o + l].chunks_exact(12) {
                        let y = f32::from_le_bytes(c[4..8].try_into().unwrap());
                        lo = lo.min(y);
                        hi = hi.max(y);
                    }
                    r = r.max((hi - lo) * 0.5);
                    if crate::envc::var_os("EVOVIEW_STEER_DBG").is_some() {
                        eprintln!("TYREMESH {} radius {:.3}", m.name, (hi - lo) * 0.5);
                    }
                }
            }
        }
        if (0.15..0.6).contains(&r) {
            rig.tyre_radius = r;
        }
        rig.read_actor(vfs, skel);
        rig.build_corners(skel);
        rig.build_steering(skel);
        rig.fit_steering(meshes, blob);
        rig.build_aims(skel);
        rig.measure_driver_steer(skel);
        rig
    }

    fn read_actor(&mut self, vfs: &mut Vfs, skel: &Skeleton) {
        let Some((d, s, e)) = actor_data(vfs) else {
            self.notes.push("no .actor - nothing is bound".into());
            return;
        };
        let mut raws: Vec<RawSlot> = Vec::new();
        let mut analog = None;
        let mut movable_raw: Vec<(usize, usize)> = Vec::new();
        fields(&d, s, e, |f, v| {
            let Some((a, b)) = v.range() else {
                if let (130, Val::Varint(x)) = (f, v) {
                    self.skip_physics_camber = x != 0;
                }
                return;
            };
            match f {
                27 => fields(&d, a, b, |g, w| {
                    if let (130, Val::Varint(x)) = (g, w) {
                        self.skip_physics_camber = x != 0;
                        return;
                    }
                    let Some((x, y)) = w.range() else { return };
                    let kind = match g {
                        102 => Kind::Wiper,
                        103 => Kind::Hood,
                        104 => Kind::Tailgate,
                        105 => Kind::Door(0),
                        106 => Kind::Door(1),
                        107 => Kind::Door(2),
                        108 => Kind::Door(3),
                        119 => Kind::Lights,
                        109 => Kind::Susp(0),
                        110 => Kind::Susp(1),
                        111 => Kind::Susp(2),
                        112 => Kind::Susp(3),
                        113 => Kind::Accel,
                        114 => Kind::Brake,
                        115 => Kind::Clutch,
                        116 => Kind::ShiftUp,
                        117 => Kind::ShiftDown,
                        121 => Kind::PaddleUp,
                        122 => Kind::PaddleDown,
                        120 => Kind::Handbrake,
                        118 => {
                            raws.extend(read_optional(&d, x, y, false));
                            return;
                        }
                        _ => return,
                    };
                    raws.push(read_generic(&d, x, y, kind));
                }),
                31 => fields(&d, a, b, |g, w| {
                    let Some((x, y)) = w.range() else { return };
                    let kind = match g {
                        109 => Kind::DrvBase,
                        101 => Kind::DrvSteer,
                        114 => Kind::DrvSteerPaddles,
                        102 => Kind::DrvAccel,
                        103 => Kind::DrvBrake,
                        104 => Kind::DrvClutch,
                        105 => Kind::DrvShiftUp,
                        106 => Kind::DrvShiftDown,
                        107 => Kind::DrvApproach,
                        112 => Kind::DrvPaddleUp,
                        113 => Kind::DrvPaddleDown,
                        110 => Kind::DrvHandbrake,
                        115 => Kind::DrvIdle,
                        _ => return,
                    };
                    raws.push(read_generic(&d, x, y, kind));
                }),
                210 => fields(&d, a, b, |g, w| {
                    let x = w.f32();
                    if x <= 0.0 { return; }
                    match g {
                        2 => self.blend.h_in = x,
                        3 => self.blend.h_play = x,
                        4 => self.blend.h_out = x,
                        28 => self.blend.p_in = x,
                        27 => self.blend.p_play = x,
                        29 => self.blend.p_out = x,
                        _ => {}
                    }
                }),
                36 => movable_raw.push((a, b)),
                103 => analog = Some(text(&d, (a, b))),
                _ => {}
            }
        });
        for (a, b) in movable_raw {
            self.movables.push(read_movable(&d, a, b, skel));
        }
        // resolve clips
        for r in raws {
            if r.paths.is_empty() {
                continue;
            }
            let mut anim: Option<Animation> = None;
            let mut missing = Vec::new();
            for p in &r.paths {
                let Some(k) = vfs.find(p, None) else {
                    missing.push(p.clone());
                    continue;
                };
                let bytes = vfs.read(&k);
                let Some(mut a) = read_animation(&bytes, &base_of(&k)) else {
                    missing.push(format!("{p} (unreadable)"));
                    continue;
                };
                if r.kind.is_driver() {
                    for t in a.tracks.iter_mut() {
                        t.bone = format!("{}{}", crate::driver::BONE_PREFIX, t.bone);
                    }
                }
                match anim.as_mut() {
                    Some(m) => {
                        m.frames = m.frames.max(a.frames);
                        m.tracks.extend(a.tracks);
                    }
                    None => anim = Some(a),
                }
            }
            let mut orphan = Vec::new();
            let mut track_bone = Vec::new();
            if let Some(a) = anim.as_ref() {
                for t in &a.tracks {
                    let bi = skel.index.get(&t.bone).copied();
                    if bi.is_none() {
                        orphan.push(t.bone.clone());
                    }
                    track_bone.push(bi);
                }
            }
            self.slots.push(Slot {
                kind: r.kind,
                play: r.play,
                speed: r.speed,
                lut: r.lut,
                opt: r.opt,
                paths: r.paths,
                missing,
                orphan_tracks: orphan,
                anim,
                track_bone,
                hub_y: Vec::new(),
            });
        }
        // dash needles
        if let Some(p) = analog {
            self.analog_script = Some(p.clone());
            match vfs.find(&p, None) {
                Some(k) => {
                    let ad = vfs.read(&k);
                    self.read_analog(&ad, skel);
                }
                None => self.notes.push(format!("analogic script {p} does not resolve")),
            }
        }
    }

    fn read_analog(&mut self, d: &[u8], skel: &Skeleton) {
        fields(d, 0, d.len(), |f, v| {
            let Some((a, b)) = v.range() else { return };
            let gauge = match f {
                1 => Gauge::Rpm,
                3 => Gauge::Speed,
                4 => Gauge::Fuel,
                5 => Gauge::FuelPressure,
                6 => Gauge::WaterPressure,
                7 => Gauge::OilPressure,
                16 => Gauge::Battery,
                8 => Gauge::Turbo,
                9 => Gauge::WaterTemp,
                10 => Gauge::OilTemp,
                11 => Gauge::TurboLimiter,
                12 => Gauge::Placeholder,
                13 => Gauge::Hours,
                14 => Gauge::Minutes,
                15 => Gauge::Seconds,
                _ => return,
            };
            let mut n = Needle { gauge, bone_name: String::new(), bone: None, axis: 2, lut: Vec::new() };
            fields(d, a, b, |g, w| match (g, w) {
                (1, Val::Bytes(x, y)) => n.bone_name = text(d, (x, y)),
                (2, Val::Varint(x)) => n.axis = (x as usize).min(2),
                (3, Val::Bytes(x, y)) => {
                    let mut p = (0.0, 0.0);
                    fields(d, x, y, |h, u| match h {
                        1 => p.0 = u.f32(),
                        2 => p.1 = u.f32(),
                        _ => {}
                    });
                    n.lut.push(p);
                }
                _ => {}
            });
            n.lut.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            n.bone = skel.index.get(&n.bone_name).copied();
            self.needles.push(n);
        });
    }

    fn build_corners(&mut self, skel: &Skeleton) {
        let phys = self.phys.clone();
        for (ci, c) in CORNERS.iter().enumerate() {
            let wheel = skel.index.get(&format!("WHEEL_{c}")).copied();
            let Some(wi) = wheel else {
                self.notes.push(format!("no WHEEL_{c} bone"));
                continue;
            };
            let hub = skel.index.get(&format!("HUB_{c}")).copied()
                .or_else(|| self.parent[wi].filter(|&p| self.parent[p].is_some()))
                .or(Some(wi));
            let centre = self.bind[wi].w_axis.truncate();
            let left = centre.x > 0.0;
            let front = ci < 2;
            let mut spin = Vec::new();
            if crate::envc::var_os("EVOVIEW_WHEEL_DEBUG").is_some() {
                for pre in ["HUB_", "WHEEL_", "TYRE_", "DISC_", "SUSP_"] {
                    if let Some(&b) = skel.index.get(&format!("{pre}{c}")) {
                        let par = self.parent[b].map(|p| skel.bones[p].name.clone());
                        let m = self.bind[b];
                        eprintln!("WHEELDBG {pre}{c} b{b} parent {par:?} x {:?} y {:?} z {:?} t {:?}",
                            m.x_axis.truncate(), m.y_axis.truncate(), m.z_axis.truncate(), m.w_axis.truncate());
                    }
                }
            }
            for pre in ["WHEEL_", "TYRE_", "DISC_"] {
                if let Some(&b) = skel.index.get(&format!("{pre}{c}")) {
                    if Some(b) == hub {
                        continue;
                    }
                    // the axle in this bone's frame. ⚠ The open-wheelers'
                    // WHEEL/TYRE/DISC bones carry the static camber (F2004,
                    // SF-25: 3 deg about Z) and the rim and tyre are placed
                    // with that bind, so their axle is the bone's own X; the
                    // car's X seen from that frame is 3 deg off, and spinning
                    // about it made the whole wheel wobble +-3 deg once a turn.
                    // A bone whose X is not near the car's lateral axis keeps
                    // the car's (EVOVIEW_SPIN_AXIS=car: always the car's).
                    let inv = self.bind[b].inverse();
                    let car_ax = inv.transform_vector3(Vec3::X).normalize_or_zero();
                    let bone_x = self.bind[b].x_axis.truncate().normalize_or_zero();
                    let own = crate::envc::var("EVOVIEW_SPIN_AXIS").map_or(true, |v| v != "car")
                        && bone_x.x.abs() > 0.985;
                    let ax = if own { Vec3::X * bone_x.x.signum() } else { car_ax };
                    spin.push((b, if ax == Vec3::ZERO { Vec3::X } else { ax }));
                }
            }
            let axle = phys.as_ref().map(|p| &p.geo[if front { 0 } else { 1 }]);
            let (bump, droop) = phys.as_ref()
                .map(|p| (p.coil[if front { 0 } else { 1 }].bump, p.coil[if front { 0 } else { 1 }].droop))
                .unwrap_or((0.06, 0.06));
            let to_car = |v: Vec3| Vec3::new(if left { -v.x } else { v.x }, v.y, v.z);
            let kingpin = axle.and_then(|g| g.kingpin())
                .map(|(p, a)| (centre + to_car(p), to_car(a).normalize()))
                .unwrap_or((centre - Vec3::new(if left { 0.12 } else { -0.12 }, 0.0, 0.0), Vec3::Y));
            let lut = axle.map(|g| kinematics(g, bump, droop, left)).unwrap_or_default();
            self.corners.push(Corner {
                idx: ci,
                hub,
                spin,
                centre,
                kingpin,
                lut,
                travel: (droop, bump),
                clip: None,
                left,
                front,
            });
        }
        if self.corners.len() < 4 && !self.corners.is_empty() {
            self.notes.push(format!("only {} wheel corner(s)", self.corners.len()));
        }
        // suspension clips: hub height per frame
        for si in 0..self.slots.len() {
            let Kind::Susp(c) = self.slots[si].kind else { continue };
            let Some(corner) = self.corners.iter_mut().find(|k| k.idx == c) else { continue };
            let Some(h) = corner.hub else { continue };
            let slot = &mut self.slots[si];
            let Some(a) = slot.anim.as_ref() else { continue };
            if let Some(t) = a.tracks.iter().position(|t| t.bone == skel.bones[h].name) {
                let p = self.parent[h].map(|p| self.bind[p]).unwrap_or(Mat4::IDENTITY);
                slot.hub_y = a.tracks[t].frames.iter().map(|f| (p * m4(f)).w_axis.y).collect();
                let (lo, hi) = slot.hub_y.iter().fold((f32::MAX, f32::MIN), |a, &y| (a.0.min(y), a.1.max(y)));
                if hi - lo > 0.002 {
                    corner.clip = Some(si);
                } else {
                    self.notes.push(format!("{} clip keeps HUB_{} still; geometry drives it", slot.kind.label(), CORNERS[c]));
                }
            } else {
                self.notes.push(format!("{} clip does not move HUB_{}", slot.kind.label(), CORNERS[c]));
            }
        }
    }

    fn build_steering(&mut self, skel: &Skeleton) {
        let Some(&b) = skel.index.get("STEER_BASE") else {
            self.notes.push("no STEER_BASE bone - the steering wheel cannot turn".into());
            return;
        };
        self.steer_bone = Some(b);
        // the bone-local axis that points most nearly forward is the column
        let w = self.bind[b];
        let mut best = (0.0f32, Vec3::Z);
        for (i, ax) in [w.x_axis, w.y_axis, w.z_axis].iter().enumerate() {
            let d = ax.truncate().normalize_or_zero().dot(Vec3::Z);
            if d.abs() > best.0.abs() {
                let mut l = Vec3::ZERO;
                l[i] = d.signum();
                best = (d, l);
            }
        }
        self.steer_axis = best.1;
        if crate::envc::var_os("EVOVIEW_STEER_DBG").is_some() {
            eprintln!("STEERDBG bind {:?} axis {:?}", w, self.steer_axis);
        }
    }

    /// The steering wheel's real column from its geometry: the vertices
    /// skinned to STEER_BASE lie in a disc, so the column is the disc's normal
    /// through its centre. Converted mods place the bone off the wheel (or
    /// square to the car while the column is raked) and the wheel then swings
    /// around a point beside it; stock cars agree with the bone and keep it.
    fn fit_steering(&mut self, meshes: &[crate::asset::MeshEntry], blob: &[u8]) {
        let Some(sb) = self.steer_bone else { return };
        let fl = |r: Option<[u64; 2]>| -> Vec<f32> {
            let Some([o, l]) = r else { return Vec::new() };
            let (o, l) = (o as usize, l as usize);
            if o + l > blob.len() { return Vec::new() }
            blob[o..o + l].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
        };
        let mut pts = Vec::new();
        for m in meshes {
            let (p, bi, bw) = (fl(m.positions), fl(m.bone_idx), fl(m.bone_wgt));
            let nv = m.vertices as usize;
            if p.len() < nv * 3 || bi.len() < nv * 4 || bw.len() < nv * 4 {
                continue;
            }
            for i in 0..nv {
                let k = (0..4).max_by(|&a, &b| bw[i * 4 + a].total_cmp(&bw[i * 4 + b])).unwrap();
                if bw[i * 4 + k] > 0.5 && bi[i * 4 + k] as usize == sb {
                    pts.push(Vec3::new(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]));
                }
            }
        }
        if pts.len() < 50 {
            return;
        }
        let c = pts.iter().copied().sum::<Vec3>() / pts.len() as f32;
        let mut cov = [[0.0f32; 3]; 3];
        for q in &pts {
            let d = *q - c;
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] += d[i] * d[j];
                }
            }
        }
        let tr = cov[0][0] + cov[1][1] + cov[2][2];
        // smallest eigenvector of cov = largest of (tr I - cov)
        let mul = |v: Vec3| Vec3::new(
            tr * v.x - (cov[0][0] * v.x + cov[0][1] * v.y + cov[0][2] * v.z),
            tr * v.y - (cov[1][0] * v.x + cov[1][1] * v.y + cov[1][2] * v.z),
            tr * v.z - (cov[2][0] * v.x + cov[2][1] * v.y + cov[2][2] * v.z));
        let mut n = Vec3::Z;
        for _ in 0..100 {
            n = mul(n).normalize_or_zero();
        }
        if n == Vec3::ZERO {
            return;
        }
        if n.z < 0.0 {
            n = -n;
        }
        // the rim's centre: the middle of the extremes in the disc's plane
        let u = n.cross(Vec3::Y).normalize_or_zero();
        let v = n.cross(u);
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for q in &pts {
            let d = *q - c;
            for (i, a) in [u, v].iter().enumerate() {
                lo[i] = lo[i].min(d.dot(*a));
                hi[i] = hi[i].max(d.dot(*a));
            }
        }
        let radius = 0.25 * ((hi[0] - lo[0]) + (hi[1] - lo[1]));
        if !(0.1..0.3).contains(&radius) {
            return;
        }
        let centre = c + u * 0.5 * (lo[0] + hi[0]) + v * 0.5 * (lo[1] + hi[1]);
        let w = self.bind[sb];
        let inv = w.inverse();
        let p_l = inv.transform_point3(centre);
        let a_l = inv.transform_vector3(n).normalize();
        let off = p_l.length();
        let ang = a_l.dot(self.steer_axis).clamp(-1.0, 1.0).acos().to_degrees();
        if crate::envc::var_os("EVOVIEW_STEER_DBG").is_some() {
            eprintln!("STEERDBG fit {} verts, centre {centre:?} normal {n:?} radius {radius:.3}; bone off {off:.3} m, axis {ang:.1} deg from the bone's", pts.len());
        }
        // trust the fit only near the bone: a wheel with a screen or paddles
        // on it is no flat disc (M4 GT3: the fit lands 80 deg off)
        if off < 0.05 && ang < 20.0 && (off > 0.02 || ang > 4.0) {
            self.steer_pivot = p_l;
            self.steer_axis = a_l;
            self.notes.push(format!("steering column from the wheel's geometry (bone {off:.2} m / {ang:.0} deg off)"));
        }
    }

    fn build_aims(&mut self, skel: &Skeleton) {
        for (i, b) in skel.bones.iter().enumerate() {
            let Some(target) = b.name.strip_prefix("DIR_") else { continue };
            let Some(&bone) = skel.index.get(target) else { continue };
            // never aim at your own descendant
            let mut p = self.parent[i];
            let mut cyc = false;
            while let Some(q) = p {
                if q == bone {
                    cyc = true;
                    break;
                }
                p = self.parent[q];
            }
            if cyc {
                continue;
            }
            let d = self.bind[i].w_axis.truncate() - self.bind[bone].w_axis.truncate();
            if d.length() < 1e-4 {
                continue;
            }
            let dir_local = self.bind[bone].inverse().transform_vector3(d).normalize();
            self.aims.push(Aim { bone, target: i, dir_local });
        }
        // aims in parent-first order, so a chain resolves in one pass
        let pos: Vec<usize> = {
            let mut v = vec![0; self.order.len()];
            for (k, &b) in self.order.iter().enumerate() {
                v[b] = k;
            }
            v
        };
        self.aims.sort_by_key(|a| pos[a.bone]);
    }

    /// How many steering-wheel degrees the driver's steer clip spans from its
    /// middle frame to its last, measured from the hands orbiting the column.
    fn measure_driver_steer(&mut self, skel: &Skeleton) {
        let (Some(si), Some(sb)) = (self.slots.iter().position(|s| s.kind == Kind::DrvSteer), self.steer_bone) else {
            return;
        };
        let n = self.slots[si].frames();
        if n < 3 {
            return;
        }
        let axis = (self.bind[sb].transform_vector3(self.steer_axis)).normalize();
        let centre = self.bind[sb].w_axis.truncate();
        let hand = |rig: &Rig, f: usize| -> Option<Vec3> {
            let mut inp = Inputs::rest(rig.slots.len());
            inp.time = 0.0;
            let mut local = rig.rest_local.clone();
            rig.apply_slot(&mut local, si, f as f32, 1.0);
            let w = rig.world(&local);
            let hb = skel.index.get(&format!("{}LeftHand", crate::driver::BONE_PREFIX))
                .or_else(|| skel.index.get(&format!("{}RightHand", crate::driver::BONE_PREFIX)))?;
            Some(w[*hb].w_axis.truncate())
        };
        // While the hands grip the rim they turn with it, so the slope of the
        // hand's angle about the column near the middle frame IS the clip's
        // degrees-per-frame. (Further out the hands re-grip and cross, so the
        // end frames say nothing about the range.)
        let mid = (n - 1) / 2;
        let k = (n / 12).max(2);
        let flat = |p: Vec3| {
            let r = p - centre;
            r - axis * r.dot(axis)
        };
        let (Some(a), Some(b)) = (hand(self, mid), hand(self, mid + k)) else { return };
        let (a, b) = (flat(a), flat(b));
        if a.length() < 0.05 || b.length() < 0.05 {
            return;
        }
        let ang = a.normalize().dot(b.normalize()).clamp(-1.0, 1.0).acos().to_degrees();
        let per_frame = ang / k as f32;
        self.drv_steer_half_deg = per_frame * mid as f32;
        // a full-turn clip ends where it started; anything else (converted
        // AC1 clips) gets its range from the grip slope, scaled by what the
        // stock fleet shows (full-turn clips measure ~300 deg of grip slope
        // for their true 360: x1.2)
        if let (Some(e), Some(c)) = (hand(self, n - 1), hand(self, mid)) {
            if (e - c).length() > 0.03 && self.drv_steer_half_deg > 30.0 {
                self.drv_steer_range_deg = (self.drv_steer_half_deg * 1.2).clamp(90.0, 900.0);
                self.notes.push(format!("driver steer clip is not a full turn; mapped to +-{:.0} deg",
                                        self.drv_steer_range_deg));
            }
        }
        if let Some(v) = crate::envc::var("EVOVIEW_DRV_STEER_DEG").ok().and_then(|v| v.parse().ok()) {
            self.drv_steer_range_deg = v;
        }
    }

    // ------------------------------------------------------------- posing --

    fn world(&self, local: &[[f32; 16]]) -> Vec<Mat4> {
        let mut w = vec![Mat4::IDENTITY; local.len()];
        for &i in &self.order {
            let l = m4(&local[i]);
            w[i] = match self.parent[i] {
                Some(p) if p < local.len() => w[p] * l,
                _ => l,
            };
        }
        w
    }

    /// Lay one slot's frame `f` (fractional) over `local`, with weight `wgt`.
    fn apply_slot(&self, local: &mut [[f32; 16]], si: usize, f: f32, wgt: f32) {
        let s = &self.slots[si];
        let Some(a) = s.anim.as_ref() else { return };
        for (ti, t) in a.tracks.iter().enumerate() {
            let Some(bi) = s.track_bone.get(ti).copied().flatten() else { continue };
            if t.frames.is_empty() {
                continue;
            }
            let n = t.frames.len();
            let fc = f.clamp(0.0, (n - 1) as f32);
            let i0 = fc.floor() as usize;
            let i1 = (i0 + 1).min(n - 1);
            let u = fc - i0 as f32;
            let sample = if u < 1e-4 || i0 == i1 {
                t.frames[i0]
            } else {
                blend_m(&t.frames[i0], &t.frames[i1], u)
            };
            local[bi] = if wgt >= 0.999 { sample } else { blend_m(&local[bi], &sample, wgt) };
        }
    }

    /// Slots a key or the demo opens and shuts: doors, hood, boot, and the
    /// optional parts with no input of their own (sunroof, roof...).
    pub fn toggle_slots(&self) -> Vec<usize> {
        self.slots.iter().enumerate()
            .filter(|(_, s)| matches!(s.kind, Kind::Hood | Kind::Tailgate | Kind::Door(_))
                || (matches!(s.kind, Kind::Optional(_)) && matches!(s.opt, Some(OptDrive::Toggle) | None)))
            .map(|(i, _)| i)
            .collect()
    }

    pub fn slot_of(&self, k: &Kind) -> Option<usize> {
        self.slots.iter().position(|s| &s.kind == k)
    }

    /// Skeleton bones the wiper clip animates (rain.rs).
    pub fn wiper_bones(&self) -> Vec<usize> {
        let Some(si) = self.slot_of(&Kind::Wiper) else { return Vec::new() };
        let mut v: Vec<usize> = self.slots[si].track_bone.iter().flatten().copied().collect();
        v.sort();
        v.dedup();
        v
    }

    /// Frame (fractional) a slot shows for these inputs, None = not playing.
    pub fn slot_frame(&self, si: usize, inp: &Inputs) -> Option<f32> {
        let s = &self.slots[si];
        let a = s.anim.as_ref()?;
        let n = a.frames.max(1) as f32;
        let last = n - 1.0;
        let fps = a.fps.max(1.0) * s.speed;
        let cs = &inp.cs;
        let open = inp.open.get(si).copied().unwrap_or(0.0);
        let value = |v: f32| Some(v.clamp(0.0, 1.0) * last);
        match &s.kind {
            Kind::Accel | Kind::DrvAccel => value(cs.throttle),
            Kind::Brake | Kind::DrvBrake => value(cs.brake),
            Kind::Clutch | Kind::DrvClutch => value(cs.clutch),
            Kind::Handbrake | Kind::DrvHandbrake => value(cs.handbrake),
            Kind::Wiper => inp.wiper.map(|t| (t * fps) % n),
            Kind::Hood | Kind::Tailgate | Kind::Door(_) | Kind::Lights => value(open),
            Kind::ShiftUp | Kind::ShiftDown => {
                let (up, t) = inp.shift?;
                if inp.paddles || (s.kind == Kind::ShiftUp) != up {
                    return None;
                }
                let ft = t * fps;
                (ft <= last).then_some(ft)
            }
            Kind::PaddleUp | Kind::PaddleDown => {
                let (up, t) = inp.shift?;
                if !inp.paddles || (s.kind == Kind::PaddleUp) != up {
                    return None;
                }
                // pull then release
                let ft = t * fps;
                if ft <= last { Some(ft) } else if ft <= 2.0 * last { Some(2.0 * last - ft) } else { None }
            }
            Kind::Susp(c) => {
                let corner = self.corners.iter().find(|k| k.clip == Some(si))?;
                let h = corner.hub?;
                let p = self.parent[h].map(|p| self.bind[p]).unwrap_or(Mat4::IDENTITY);
                let rest = (p * m4(&self.rest_local[h])).w_axis.y;
                Some(frame_for_height(&s.hub_y, rest + inp.travel[*c]))
            }
            Kind::Optional(_) => {
                let v = match s.opt.as_ref() {
                    Some(OptDrive::Toggle) | None => open,
                    Some(OptDrive::Speed(l)) => lut_eval(l, cs.speed_mps.abs() * 3.6),
                    Some(OptDrive::G(l)) => lut_eval(l, inp.g.abs().max_element()),
                    Some(OptDrive::Inputs(g, b, st)) => lut_eval(g, cs.throttle)
                        .max(lut_eval(b, cs.brake)).max(lut_eval(st, cs.steer.abs())),
                    Some(OptDrive::Rpm(l)) => lut_eval(l, cs.rpm),
                    Some(OptDrive::Drs(x)) => if inp.drs { *x } else { 0.0 },
                    Some(OptDrive::Setup) => 0.5,
                };
                // SequentialLut: the value picks a stop among lut positions
                let v = v.clamp(0.0, 1.0);
                let pos = if s.play == Play::SequentialLut && s.lut.len() >= 2 {
                    let x = v * (s.lut.len() - 1) as f32;
                    let i = (x.floor() as usize).min(s.lut.len() - 2);
                    s.lut[i] + (s.lut[i + 1] - s.lut[i]) * (x - i as f32)
                } else {
                    v
                };
                Some(pos.clamp(0.0, 1.0) * last)
            }
            Kind::DrvBase => Some(0.0),
            Kind::DrvSteer | Kind::DrvSteerPaddles => {
                // Every stock steer clip spans ONE FULL TURN each way: its first
                // and last frames put the hands exactly where the middle frame
                // does (measured fleet-wide, <2 mm), and the hands' grip slope
                // is ~300 deg / half-clip on every car whatever its lock
                // (220..630). So the wheel angle, not the input, is the
                // playhead: frame = mid + mid * angle / 360.
                let range = self.drv_steer_range_deg.max(1.0);
                let x = (inp.wheel_deg / range).clamp(-1.0, 1.0);
                Some((x * 0.5 + 0.5) * last)
            }
            Kind::DrvIdle => Some((inp.time * fps) % n),
            // shift and paddle layers are timed by `pose`
            _ => None,
        }
    }

    /// Skinning matrices (row-major, row-vector, as the shader reads them) for
    /// these inputs. ~250 bones of mat4 maths: a few microseconds.
    pub fn pose(&self, inp: &Inputs) -> Vec<[f32; 16]> {
        let mut local = self.rest_local.clone();
        let paddles = inp.paddles;
        // 1. car clips
        for si in 0..self.slots.len() {
            if self.slots[si].kind.is_driver() {
                continue;
            }
            if let Some(f) = self.slot_frame(si, inp) {
                self.apply_slot(&mut local, si, f, 1.0);
            }
        }
        // 2. driver: base, idle (torso and head only), steer, legs, shifting
        let driver_order = [Kind::DrvBase, Kind::DrvIdle];
        for k in driver_order.iter() {
            if let Some(si) = self.slot_of(k) {
                if let Some(f) = self.slot_frame(si, inp) {
                    if *k == Kind::DrvIdle {
                        // the idle (looking around, breathing) is a parked
                        // car's: it fades out as the car gets going, so on the
                        // move the driver faces the road
                        let v = inp.cs.speed_mps.abs() * 3.6;
                        let w = 1.0 - ((v - 1.0) / 14.0).clamp(0.0, 1.0);
                        if crate::envc::var_os("EVOVIEW_DRV_LOG").is_some() && (inp.time * 4.0).fract() < 0.05 {
                            eprintln!("DRV t {:.2} kmh {v:.1} idle w {w:.2} frame {f:.1} wheel {:.1}", inp.time, inp.wheel_deg);
                        }
                        if w > 0.0 {
                            self.apply_idle(&mut local, si, f, w);
                        }
                    } else {
                        self.apply_slot(&mut local, si, f, 1.0);
                    }
                }
            }
        }
        let steer_kind = if paddles && self.slot_of(&Kind::DrvSteerPaddles).is_some() {
            Kind::DrvSteerPaddles
        } else {
            Kind::DrvSteer
        };
        if let Some(si) = self.slot_of(&steer_kind) {
            if let Some(f) = self.slot_frame(si, inp) {
                self.apply_slot(&mut local, si, f, 1.0);
            }
        }
        for k in [Kind::DrvAccel, Kind::DrvBrake, Kind::DrvClutch, Kind::DrvHandbrake] {
            if let Some(si) = self.slot_of(&k) {
                if let Some(f) = self.slot_frame(si, inp) {
                    if k == Kind::DrvHandbrake && inp.cs.handbrake < 0.01 {
                        continue;
                    }
                    self.apply_slot(&mut local, si, f, 1.0);
                }
            }
        }
        if let Some((up, t)) = inp.shift {
            self.apply_shift(&mut local, up, t, paddles);
        }
        // 3. procedural: hubs, spin, steering wheel, needles, movables
        let mut world = self.world(&local);
        for c in self.corners.iter() {
            let ci = c.idx;
            let Some(h) = c.hub else { continue };
            let steer = inp.road.get(ci).copied().unwrap_or(0.0);
            // steer about the kingpin: + = right = front of the wheel to -X
            let (kp, ka) = c.kingpin;
            let sk = Mat4::from_translation(kp) * Mat4::from_quat(Quat::from_axis_angle(ka, -steer))
                * Mat4::from_translation(-kp);
            let hub_w = if c.clip.is_some() {
                sk * world[h]
            } else {
                let (d, q) = sample_lut(&c.lut, inp.travel[ci]);
                let mv = Mat4::from_translation(c.centre + d) * Mat4::from_quat(q)
                    * Mat4::from_translation(-c.centre);
                mv * sk * world[h]
            };
            let pw = self.parent[h].map(|p| world[p]).unwrap_or(Mat4::IDENTITY);
            local[h] = (pw.inverse() * hub_w).to_cols_array();
            for &(b, ax) in &c.spin {
                let l = m4(&local[b]) * Mat4::from_quat(Quat::from_axis_angle(ax, inp.spin[ci]));
                local[b] = l.to_cols_array();
            }
        }
        if let Some(b) = self.steer_bone {
            let p = self.steer_pivot;
            let l = m4(&local[b]) * Mat4::from_translation(p) * Mat4::from_quat(Quat::from_axis_angle(self.steer_axis, inp.wheel_deg.to_radians())) * Mat4::from_translation(-p);
            local[b] = l.to_cols_array();
        }
        for n in &self.needles {
            let Some(b) = n.bone else { continue };
            let x = self.gauge_value(n.gauge, inp);
            let deg = lut_eval(&n.lut, x);
            let mut ax = Vec3::ZERO;
            ax[n.axis] = 1.0;
            let l = m4(&local[b]) * Mat4::from_quat(Quat::from_axis_angle(ax, deg.to_radians()));
            local[b] = l.to_cols_array();
        }
        let speed_kmh = inp.cs.speed_mps.abs() * 3.6;
        for m in &self.movables {
            let Some(b) = m.bone else { continue };
            let mut rot = Vec3::ZERO;
            let mut tr = Vec3::ZERO;
            for o in &m.osc {
                let k = ((speed_kmh - o.speed_min) / (o.speed_max - o.speed_min).max(1.0)).clamp(0.0, 1.0);
                let ph = Vec3::new(
                    (inp.time * o.freq.x * std::f32::consts::TAU).sin(),
                    (inp.time * o.freq.y * std::f32::consts::TAU).sin(),
                    (inp.time * o.freq.z * std::f32::consts::TAU).sin());
                rot += o.rot * ph * k;
                tr += o.trans * ph * k;
            }
            let g = inp.g.abs().max_element();
            let k = ((g - m.g_min) / (m.g_max - m.g_min).max(0.01)).clamp(0.0, 1.0);
            rot += m.g_rot * k;
            tr += m.g_trans * k;
            let q = Quat::from_euler(glam::EulerRot::XYZ, rot.x.to_radians(), rot.y.to_radians(), rot.z.to_radians());
            let l = m4(&local[b]) * Mat4::from_translation(tr) * Mat4::from_quat(q);
            local[b] = l.to_cols_array();
        }
        world = self.world(&local);
        // 4. look-at bones (arms, tie rods, dampers) re-aim at their DIR_ partner
        if !self.aims.is_empty() {
            for a in &self.aims {
                let p = world[a.bone].w_axis.truncate();
                let cur = world[a.bone].transform_vector3(a.dir_local).normalize_or_zero();
                let want = (world[a.target].w_axis.truncate() - p).normalize_or_zero();
                if cur == Vec3::ZERO || want == Vec3::ZERO {
                    continue;
                }
                let q = Quat::from_rotation_arc(cur, want);
                let c = Mat4::from_translation(p) * Mat4::from_quat(q) * Mat4::from_translation(-p);
                let pw = self.parent[a.bone].map(|x| world[x]).unwrap_or(Mat4::IDENTITY);
                let nw = c * world[a.bone];
                local[a.bone] = (pw.inverse() * nw).to_cols_array();
                world[a.bone] = nw;
            }
            world = self.world(&local);
        }
        // 5. skin = body * world * inverse(bind)
        world
            .iter()
            .zip(&self.inv_bind)
            .map(|(w, ib)| (inp.body * *w * *ib).to_cols_array())
            .collect()
    }

    /// The idle take breathes and looks around; only its torso, neck and head
    /// tracks are used, so it never fights the steer or pedal layers.
    fn apply_idle(&self, local: &mut [[f32; 16]], si: usize, f: f32, w: f32) {
        let s = &self.slots[si];
        let Some(a) = s.anim.as_ref() else { return };
        for (ti, t) in a.tracks.iter().enumerate() {
            let Some(bi) = s.track_bone.get(ti).copied().flatten() else { continue };
            let n = t.bone.trim_start_matches(crate::driver::BONE_PREFIX);
            if !(n.starts_with("A_Abdom") || n == "A_CEST" || n == "A_Neck" || n == "A_Head") {
                continue;
            }
            if t.frames.is_empty() {
                continue;
            }
            let k = t.frames[(f as usize).min(t.frames.len() - 1)];
            if w >= 0.999 {
                local[bi] = k;
            } else {
                // a component blend toward the take: fine for the small
                // head / torso turns an idle makes
                for j in 0..16 {
                    local[bi][j] += (k[j] - local[bi][j]) * w;
                }
            }
        }
    }

    /// The driver's hand leaves the wheel for the lever (or flicks a paddle),
    /// using the actor's blend timings.
    fn apply_shift(&self, local: &mut [[f32; 16]], up: bool, t: f32, paddles: bool) {
        let b = &self.blend;
        if paddles {
            let k = if up { Kind::DrvPaddleUp } else { Kind::DrvPaddleDown };
            let Some(si) = self.slot_of(&k) else { return };
            let len = b.p_in + b.p_play + b.p_out;
            if t > len {
                return;
            }
            let w = if t < b.p_in { t / b.p_in } else if t > b.p_in + b.p_play { (len - t) / b.p_out } else { 1.0 };
            let n = self.slots[si].frames().max(1) as f32 - 1.0;
            let f = ((t - b.p_in) / b.p_play).clamp(0.0, 1.0) * n;
            self.apply_slot(local, si, f, w.clamp(0.0, 1.0));
            return;
        }
        let approach = self.slot_of(&Kind::DrvApproach);
        let change = self.slot_of(&if up { Kind::DrvShiftUp } else { Kind::DrvShiftDown });
        let len = b.h_in + b.h_play + b.h_out;
        if t > len {
            return;
        }
        if let Some(si) = approach {
            let n = self.slots[si].frames().max(1) as f32 - 1.0;
            let (f, w) = if t < b.h_in {
                (t / b.h_in * n, (t / b.h_in).min(1.0))
            } else if t < b.h_in + b.h_play {
                (n, 1.0)
            } else {
                let u = (t - b.h_in - b.h_play) / b.h_out;
                ((1.0 - u) * n, 1.0 - u)
            };
            self.apply_slot(local, si, f, w.clamp(0.0, 1.0));
        }
        if let Some(si) = change {
            if t >= b.h_in && t < b.h_in + b.h_play {
                let n = self.slots[si].frames().max(1) as f32 - 1.0;
                let f = (t - b.h_in) / b.h_play * n;
                self.apply_slot(local, si, f, 1.0);
            }
        }
    }

    pub fn gauge_value(&self, g: Gauge, inp: &Inputs) -> f32 {
        let cs = &inp.cs;
        let ign = cs.rpm > 10.0;
        match g {
            Gauge::Rpm => cs.rpm,
            Gauge::Speed => cs.speed_mps.abs() * 3.6,
            Gauge::Fuel => inp.fuel_l,
            Gauge::WaterTemp => if ign { inp.water_c } else { 20.0 },
            Gauge::OilTemp => if ign { inp.oil_c } else { 20.0 },
            Gauge::OilPressure => if ign { 1.0 + 4.0 * (cs.rpm / cs.rpm_limit.max(1.0)).min(1.0) } else { 0.0 },
            Gauge::FuelPressure => if ign { 3.0 } else { 0.0 },
            Gauge::WaterPressure => if ign { 1.2 } else { 0.0 },
            Gauge::Battery => if ign { 14.0 } else { 12.4 },
            Gauge::Turbo => if ign { (cs.throttle * cs.rpm / cs.rpm_limit.max(1.0) * 1.2 - 0.6).max(-0.8) } else { 0.0 },
            Gauge::TurboLimiter | Gauge::Placeholder => 0.0,
            Gauge::Hours => 10.0 + inp.time / 3600.0,
            Gauge::Minutes => (inp.time / 60.0) % 60.0,
            Gauge::Seconds => inp.time % 60.0,
        }
    }

    /// World (car-space, before the body transform) bone matrices for a pose:
    /// --check reads positions out of these.
    pub fn world_of(&self, skin: &[[f32; 16]], body: Mat4) -> Vec<Mat4> {
        let ib = body.inverse();
        skin.iter().zip(&self.bind).map(|(s, b)| ib * m4(s) * *b).collect()
    }

    pub fn bind_of(&self, i: usize) -> Mat4 {
        self.bind[i]
    }
}

fn blend_m(a: &[f32; 16], b: &[f32; 16], u: f32) -> [f32; 16] {
    let (sa, ra, ta) = m4(a).to_scale_rotation_translation();
    let (sb, rb, tb) = m4(b).to_scale_rotation_translation();
    Mat4::from_scale_rotation_translation(sa.lerp(sb, u), ra.slerp(rb, u), ta.lerp(tb, u)).to_cols_array()
}

fn frame_for_height(ys: &[f32], want: f32) -> f32 {
    if ys.len() < 2 {
        return 0.0;
    }
    // ys is monotonic in practice; find the bracketing pair
    let mut best = (f32::MAX, 0.0f32);
    for i in 0..ys.len() - 1 {
        let (a, b) = (ys[i], ys[i + 1]);
        let (lo, hi) = (a.min(b), a.max(b));
        if want >= lo && want <= hi && (b - a).abs() > 1e-7 {
            return i as f32 + (want - a) / (b - a);
        }
        let d = (a - want).abs();
        if d < best.0 {
            best = (d, i as f32);
        }
    }
    let dl = (ys[ys.len() - 1] - want).abs();
    if dl < best.0 { (ys.len() - 1) as f32 } else { best.1 }
}

fn sample_lut(lut: &[(f32, Vec3, Quat)], s: f32) -> (Vec3, Quat) {
    if lut.is_empty() {
        return (Vec3::new(0.0, s, 0.0), Quat::IDENTITY);
    }
    if s <= lut[0].0 {
        return (lut[0].1, lut[0].2);
    }
    for w in lut.windows(2) {
        if s <= w[1].0 {
            let t = (s - w[0].0) / (w[1].0 - w[0].0).max(1e-6);
            return (w[0].1.lerp(w[1].1, t), w[0].2.slerp(w[1].2, t));
        }
    }
    let l = lut[lut.len() - 1];
    (l.1, l.2)
}

fn rot_about(p: Vec3, a: Vec3, b: Vec3, th: f32) -> Vec3 {
    let ax = (b - a).normalize_or_zero();
    if ax == Vec3::ZERO {
        return p;
    }
    a + Quat::from_axis_angle(ax, th) * (p - a)
}

/// Travel -> (wheel-centre offset, upright rotation) for one corner, in car
/// space. The geometry is solved in the physics' left-wheel frame (x inboard)
/// and mirrored onto the car.
///
/// Double wishbone: the lower ball joint swings about the lower arm's pivots;
/// the upper arm angle is solved so the upright keeps its length; the upright
/// turns with the line between the two ball joints. Strut: the same lower arm,
/// with the upright aimed at the fixed strut top. Trailing arm: the whole hub
/// swings about the hinge. Axle / multilink: straight up and down.
pub fn kinematics(g: &Geometry, bump: f32, droop: f32, left: bool) -> Vec<(f32, Vec3, Quat)> {
    let state = |th: f32| -> Option<(Vec3, Quat)> {
        match g.kind {
            SuspKind::DoubleWishbone | SuspKind::Strut => {
                let (lf, lr, b0) = g.low?;
                let b = rot_about(b0, lf, lr, th);
                let (u0, u) = if g.kind == SuspKind::Strut {
                    let t = g.strut_top?;
                    (t, t)
                } else {
                    let (uf, ur, u0) = g.up?;
                    let l0 = (u0 - b0).length();
                    // solve the upper arm angle: |U(phi) - B| = l0
                    let f = |phi: f32| (rot_about(u0, uf, ur, phi) - b).length() - l0;
                    let (mut lo, mut hi) = (-0.9f32, 0.9f32);
                    let (flo, fhi) = (f(lo), f(hi));
                    if flo.signum() == fhi.signum() {
                        // pick the root nearest rest by scanning
                        let mut best = (f32::MAX, 0.0);
                        let mut x = -0.9;
                        while x <= 0.9 {
                            let v = f(x).abs();
                            if v < best.0 { best = (v, x); }
                            x += 0.01;
                        }
                        (u0, rot_about(u0, uf, ur, best.1))
                    } else {
                        // bracket the root nearest zero
                        let mut a = -0.9f32;
                        let mut chosen = None;
                        let mut x = -0.9f32;
                        let mut fa = f(a);
                        while x < 0.9 {
                            x += 0.02;
                            let fx = f(x);
                            if fa.signum() != fx.signum() {
                                let cand = (a, x);
                                let mid = (cand.0 + cand.1) * 0.5;
                                if chosen.map_or(true, |c: (f32, f32)| mid.abs() < ((c.0 + c.1) * 0.5).abs()) {
                                    chosen = Some(cand);
                                }
                            }
                            a = x;
                            fa = fx;
                        }
                        let (l, h) = chosen.unwrap_or((lo, hi));
                        lo = l;
                        hi = h;
                        for _ in 0..30 {
                            let m = (lo + hi) * 0.5;
                            if f(lo).signum() == f(m).signum() { lo = m; } else { hi = m; }
                        }
                        (u0, rot_about(u0, uf, ur, (lo + hi) * 0.5))
                    }
                };
                let q = Quat::from_rotation_arc((u0 - b0).normalize(), (u - b).normalize());
                let w = b + q * (-b0);
                Some((w, q))
            }
            SuspKind::TrailingArm => {
                let (a, b) = g.hinge?;
                let ax = (b - a).normalize_or_zero();
                let q = Quat::from_axis_angle(ax, th);
                Some((a + q * (-a), q))
            }
            _ => None,
        }
    };
    // Find theta for a target wheel-centre height: march out from rest in
    // the direction that moves the wheel the right way until the height is
    // passed, then bisect that step. Marching (rather than bisecting the
    // whole +-0.8 rad range) keeps the solution on the branch connected to
    // the rest pose - far from rest a wishbone can flip to its other root.
    let solve = |s: f32| -> Option<(Vec3, Quat)> {
        if s.abs() < 1e-6 {
            return state(0.0);
        }
        let y0 = state(0.0)?.0.y;
        let yp = state(0.01)?.0.y;
        let dir = if (yp - y0).signum() == s.signum() { 1.0 } else { -1.0 };
        let step = 0.005f32;
        let mut prev = 0.0f32;
        let mut prev_y = y0;
        for i in 1..200 {
            let th = dir * step * i as f32;
            let y = state(th)?.0.y;
            if (y - s).signum() != (prev_y - s).signum() || (y - s).abs() < 1e-6 {
                let (mut lo, mut hi) = (prev, th);
                for _ in 0..30 {
                    let m = (lo + hi) * 0.5;
                    let ym = state(m)?.0.y;
                    if (ym - s).signum() == (prev_y - s).signum() { lo = m; } else { hi = m; }
                }
                return state((lo + hi) * 0.5);
            }
            // moving the wrong way (past the arm's top dead centre): give up
            if (y - prev_y).abs() < 1e-7 || (y - prev_y).signum() != s.signum() {
                return None;
            }
            prev = th;
            prev_y = y;
        }
        None
    };
    let mut out = Vec::new();
    let n = 16;
    for i in 0..=n {
        let s = -droop + (bump + droop) * i as f32 / n as f32;
        let (w, q) = solve(s).unwrap_or((Vec3::new(0.0, s, 0.0), Quat::IDENTITY));
        // physics left frame (x inboard) -> car: mirror x for the left wheel
        let (d, q) = if left {
            (Vec3::new(-w.x, w.y, w.z), Quat::from_xyzw(q.x, -q.y, -q.z, q.w))
        } else {
            (w, q)
        };
        out.push((s, d, q));
    }
    out
}

fn read_movable(d: &[u8], s: usize, e: usize, skel: &Skeleton) -> Movable {
    let mut m = Movable {
        bone_name: String::new(),
        bone: None,
        osc: Vec::new(),
        g_min: 0.0,
        g_max: 1.0,
        g_rot: Vec3::ZERO,
        g_trans: Vec3::ZERO,
    };
    fields(d, s, e, |f, v| match (f, v) {
        (1, Val::Bytes(a, b)) => m.bone_name = text(d, (a, b)),
        (310, Val::Bytes(a, b)) => {
            let mut o = Osc { speed_min: 0.0, speed_max: 100.0, freq: Vec3::ZERO, rot: Vec3::ZERO, trans: Vec3::ZERO };
            fields(d, a, b, |g, w| match (g, w) {
                (301, x) => o.speed_min = x.f32(),
                (302, x) => o.speed_max = x.f32(),
                (303, Val::Bytes(x, y)) => o.freq = vec3_of(d, x, y),
                (304, Val::Bytes(x, y)) => o.rot = vec3_of(d, x, y),
                (305, Val::Bytes(x, y)) => o.trans = vec3_of(d, x, y),
                _ => {}
            });
            m.osc.push(o);
        }
        (400, x) => m.g_min = x.f32(),
        (401, x) => m.g_max = x.f32(),
        (402, Val::Bytes(a, b)) => m.g_rot = vec3_of(d, a, b),
        (404, Val::Bytes(a, b)) => m.g_trans = vec3_of(d, a, b),
        _ => {}
    });
    m.bone = skel.index.get(&m.bone_name).copied();
    let _ = IDENTITY;
    m
}

#[cfg(test)]
mod kind_tests {
    use super::Kind;

    /// is_driver must keep agreeing with the Drv* naming it replaced.
    #[test]
    fn is_driver_matches_the_drv_names() {
        let all = [
            Kind::Wiper, Kind::Hood, Kind::Tailgate, Kind::Door(0), Kind::Lights, Kind::Susp(0), Kind::Accel,
            Kind::Brake, Kind::Clutch, Kind::ShiftUp, Kind::ShiftDown, Kind::PaddleUp, Kind::PaddleDown,
            Kind::Handbrake, Kind::Optional("x".into()), Kind::DrvBase, Kind::DrvSteer, Kind::DrvSteerPaddles,
            Kind::DrvAccel, Kind::DrvBrake, Kind::DrvClutch, Kind::DrvShiftUp, Kind::DrvShiftDown,
            Kind::DrvApproach, Kind::DrvPaddleUp, Kind::DrvPaddleDown, Kind::DrvHandbrake, Kind::DrvIdle,
        ];
        for k in all {
            assert_eq!(k.is_driver(), format!("{k:?}").starts_with("Drv"), "{k:?}");
        }
    }
}
