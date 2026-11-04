//! The handful of physics numbers the viewer needs to move a car believably:
//! steering, suspension geometry and travel, springs, mass, gearing and revs.
//!
//! All of it comes from the car's own `CarData` (`data/<id>_car.car`, or
//! `data/cardata.car` on a converted mod) plus the `.carengine` it names. The
//! field numbers are the game's own (CarData.proto, embedded in the exe;
//! dumped to L:/evoview_work/anim/desc):
//!
//! ```text
//! CarData  1 general {1 totalMass, 9 body_box_sizes{x,y,z}}
//!          2 suspensions {2 wheel_base, 3 longitudinal_cg_location (front share),
//!                         50/51 base_y F/R, 52/53 track F/R,
//!                         8/9 coilover F/R {1 springRate, 3 bumpStopUp{1 Range, 5 length},
//!                                          4 bumpStopDown{1 Range, 5 length}},
//!                         18/19 arb F/R {1 stiffness}, 40/41 geometry F/R, 22/23 their paths}
//!          4 engine {5 minimum, 6 limiter}      20 .carengine path (same message)
//!          8 controls {2 steer_lock deg, 3 steerRatio (sign = rack side)}
//!         23 gearbox {2 gears{1 name, 2 ratio}, 3 final_ratio}
//! SuspensionGeometryData 2 double wishbone, 3 strut, 9 strut multilink,
//!          4 axle, 5/7 multilink, 6 trailing arm, 8 double wishbone + coilover
//! ```
//!
//! Hard points are relative to the wheel centre at rest, in metres, with
//! **x positive INBOARD**, y up, z forward - so for the left wheel (car +X)
//! a point maps to car space as (Wx - x, Wy + y, Wz + z) and for the right
//! wheel as (Wx + x, ...).

use crate::load::Vfs;
use crate::proto::{fields, text, Val};
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SuspKind {
    DoubleWishbone,
    Strut,
    TrailingArm,
    Axle,
    MultiLink,
    None,
}

/// One axle's suspension geometry, left-wheel convention (see module doc).
#[derive(Clone, Debug)]
pub struct Geometry {
    pub kind: SuspKind,
    /// lower arm pivots on the body (front, rear) and the ball joint on the hub
    pub low: Option<(Vec3, Vec3, Vec3)>,
    /// upper arm pivots + ball joint (double wishbone)
    pub up: Option<(Vec3, Vec3, Vec3)>,
    /// strut top mount on the body (strut)
    pub strut_top: Option<Vec3>,
    /// trailing arm hinge (inner, outer)
    pub hinge: Option<(Vec3, Vec3)>,
    /// steering tie rod: body end, hub end
    pub steer: Option<(Vec3, Vec3)>,
    /// coilover body mount and its mount on the arm (DoubleWishBoneCoilover
    /// f9 carCoilover / f10 bottomCoilover); None = the spring acts at the
    /// wheel
    pub coil: Option<(Vec3, Vec3)>,
    /// BasicSuspensionData (geometry f1): hub mass (kg), static camber (deg)
    pub hub_mass: f32,
    pub static_camber: f32,
}

impl Geometry {
    /// Coilover motion ratio: spring compression per metre of wheel travel,
    /// from the lower arm turning about its body pivots (fronts and rears
    /// differ, which sets the roll-stiffness balance). 1 when the spring acts
    /// at the wheel.
    pub fn motion_ratio(&self) -> f32 {
        let (Some((top, bot)), Some((pf, pr, ball))) = (self.coil, self.low) else { return 1.0 };
        let axis = (pr - pf).normalize_or_zero();
        if axis == Vec3::ZERO {
            return 1.0;
        }
        // small rotation about the arm axis: velocity of a point = axis x r
        let v_at = |p: Vec3| axis.cross(p - pf);
        let (vb, vc) = (v_at(ball), v_at(bot));
        let dz = vb.y;
        let dl = -vc.dot((top - bot).normalize_or_zero());
        if dz.abs() < 1e-6 {
            return 1.0;
        }
        let mr = (dl / dz).abs();
        if (0.2..=1.5).contains(&mr) { mr } else { 1.0 }
    }

    fn empty() -> Self {
        Geometry { kind: SuspKind::None, low: None, up: None, strut_top: None, hinge: None, steer: None, coil: None, hub_mass: 0.0, static_camber: 0.0 }
    }
    /// Kingpin (steering) axis at rest, left-wheel convention: lower ball
    /// joint -> upper ball joint / strut top. None for a rigid or unknown
    /// layout.
    /// Camber change per metre of bump, rad (+ = the top goes outboard), from
    /// the linkage: the lower ball turns about the lower arm's axis, the upper
    /// ball about the upper arm's (double wishbone) or the strut top stays
    /// put (strut); the upright is the line between them. In the game ODE
    /// solves these joints; this is their first-order rate at ride height.
    pub fn camber_gain(&self) -> f32 {
        let Some((pf, pr, lb)) = self.low else { return 0.0 };
        let ax = (pr - pf).normalize_or_zero();
        let mut vl = ax.cross(lb - pf);
        if vl.y.abs() < 1e-6 {
            return 0.0;
        }
        vl /= vl.y;
        let (d, dd) = match (self.kind, self.up, self.strut_top) {
            (SuspKind::DoubleWishbone, Some((uf, ur, ub)), _) | (SuspKind::MultiLink, Some((uf, ur, ub)), _) => {
                let au = (ur - uf).normalize_or_zero();
                let d = ub - lb;
                let tu = au.cross(ub - uf);
                let den = tu.dot(d);
                if den.abs() < 1e-9 {
                    return 0.0;
                }
                let vu = tu * (vl.dot(d) / den);
                (d, vu - vl)
            }
            (_, _, Some(top)) => (top - lb, -vl),
            _ => return 0.0,
        };
        // camber = atan2(-d.x, d.y) in the front view (x inboard)
        let n = d.x * d.x + d.y * d.y;
        if n < 1e-9 {
            return 0.0;
        }
        (-d.y * dd.x + d.x * dd.y) / n
    }

    /// Toe change per metre of bump, rad (+ = toe-in): the tie rod (or the
    /// rear's toe link) keeps its length, so as the upright rises and leans
    /// (camber_gain) the hub end of the rod turns it about the kingpin -
    /// bump steer. First-order rate at ride height, like camber_gain.
    pub fn toe_gain(&self) -> f32 {
        let (Some((pf, pr, lb)), Some((t_in, t_hub)), Some((_, k))) = (self.low, self.steer, self.kingpin()) else { return 0.0 };
        let ax = (pr - pf).normalize_or_zero();
        let mut vl = ax.cross(lb - pf);
        if vl.y.abs() < 1e-6 {
            return 0.0;
        }
        vl /= vl.y;
        // the upright: carried with the lower ball, leaning at the camber
        // rate (about +z: the top goes outboard, -x)
        let wc = Vec3::new(0.0, 0.0, self.camber_gain());
        let h = t_hub - lb;
        let vh0 = vl + wc.cross(h);
        let r = t_hub - t_in;
        let turn = k.cross(h);
        let den = turn.dot(r);
        if den.abs() < 1e-9 {
            return 0.0;
        }
        // about +y (up): +z goes to +x, inboard = toe-in
        -(vh0.dot(r)) / den * k.y.signum()
    }

    pub fn kingpin(&self) -> Option<(Vec3, Vec3)> {
        let b = self.low?.2;
        let t = self.up.map(|u| u.2).or(self.strut_top)?;
        let d = t - b;
        (d.length() > 1e-3).then(|| (b, d.normalize()))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Coilover {
    pub spring: f32,
    /// metres of compression / extension available from rest: the gap to the
    /// bump stop plus the stop's own squash
    pub bump: f32,
    pub droop: f32,
    /// the gap to the bump stop alone - free travel before it engages
    pub bump_free: f32,
    pub arb: f32,
}

#[derive(Clone, Debug)]
pub struct Physics {
    pub source: String,
    pub mass: f32,
    pub body_box: Vec3,
    pub wheelbase: f32,
    /// static front weight share 0..1
    pub cg_front: f32,
    pub track: [f32; 2],
    pub base_y: [f32; 2],
    pub coil: [Coilover; 2],
    pub geo: [Geometry; 2],
    /// steering wheel degrees from centre to full lock
    pub steer_lock_deg: f32,
    pub steer_ratio: f32,
    /// SteerControlsData.ffMult: the game scales the rack force by
    /// ffMult * 0.001 before the wheel (Car::getFFB)
    pub ff_mult: f32,
    /// SteerControlsData f5 (named steerAssist in the proto): the car init
    /// stores it at car+0x8ca0, getFFB's output gamma (0 = 1, linear)
    pub ff_gamma: f32,
    pub idle_rpm: f32,
    pub limiter_rpm: f32,
    /// the limiter exactly as the data has it, before sanitising
    pub limiter_raw: f32,
    /// full-throttle torque before boost: (rpm, N m), from the power curve
    pub torque: Vec<(f32, f32)>,
    pub turbos: Vec<Turbo>,
    /// the default engine map's power multiplier (BoP / power modes)
    pub power_mult: f32,
    /// rolling radius of the driven (rear) tyre, from its .tyre file
    pub tyre_radius: Option<f32>,
    /// the front / rear tyres' grip model (.tyre f4 TyresModelData)
    pub tyres: [Option<TyreModel>; 2],
    /// the front / rear tyres' thermal model (.tyre f5, tyretherm.rs)
    pub thermal: [Option<crate::tyretherm::Data>; 2],
    /// DrivetrainData f2 TractionType: 0 RWD, 1 FWD, 2 FourWD, 3 AWD,
    /// 4 AWDF (front-biased)
    pub traction: u32,
    /// the game's full coilover per axle (spring, helper, stops, two-way
    /// damper), diff and brakes - evophys.rs
    pub coilover: [crate::evophys::Coilover; 2],
    /// EngineData f1 inertia (kg m^2) and f3 coastCurve (rpm -> N m off throttle)
    pub engine_inertia: f32,
    /// the .carsetup applied on top (apply_stock_setup), "" for none
    pub setup_source: String,
    /// per axle from the setup's TyreAlignment (f4: 2 camber, 3 toe,
    /// 5 static_camber), degrees; None = not in the setup. Toe + = toe-in
    pub align: [Option<(f32, f32)>; 2],
    pub gearbox: Gearbox,
    pub clutch: Clutch,
    /// CarData f7 ElectronicsData f2 (None: the car has no ABS)
    pub abs: Option<crate::evophys::Abs>,
    /// CarData f7 ElectronicsData f1 (None: no TC)
    pub tc: Option<crate::evophys::Tc>,
    pub coast: Vec<(f32, f32)>,
    pub diff: crate::evophys::Diff,
    pub brakes: crate::evophys::Brakes,
    /// the .carsetuplimits (CarData f16): what the cockpit may change
    pub limits: crate::aids::Limits,
    /// where each engine number came from, for --check
    pub engine_source: String,
    /// GeneralCarData f2 screenName (the game files the player's setups
    /// under it)
    pub screen_name: String,
    /// GeneralCarData f1 totalMass, without fuel; `mass` is this plus the
    /// setup's fuel (f8 kgPerLiter, 0.75 when absent)
    pub mass_dry: f32,
    pub kg_per_l: f32,
    pub fuel_l: f32,
    /// BrakesData totalTorque before the setup's torque_multiplier
    pub brake_base: f32,
    /// per axle from the setup: cold tyre pressure psi (the tyre starts at
    /// it: fn 0x142370f30), target ride height m and wing angle deg (the
    /// aero, lapsim::aero)
    pub pressure: [Option<f32>; 2],
    pub ride_h: [Option<f32>; 2],
    pub wing: [Option<f32>; 2],
    /// DampersData f13 lut_list per axle: a LUT damper, whose setup slow
    /// values are clicks
    pub damper_lut: [Option<String>; 2],
    /// the player's setup on top of the stock one (EVOVIEW_SETUP), "" none
    pub user_setup: String,
    /// EngineData f51 engine_type = electric_motor
    pub electric: bool,
    /// forward gear ratios, 1st first
    pub gears: Vec<f32>,
    pub reverse: f32,
    pub final_drive: f32,
    /// What was missing or clamped while reading - surfaced by --check.
    pub notes: Vec<String>,
}

fn v3(d: &[u8], s: usize, e: usize) -> Vec3 {
    let mut v = Vec3::ZERO;
    fields(d, s, e, |f, x| {
        if (1..=3).contains(&f) {
            v[(f - 1) as usize] = x.f32();
        }
    });
    v
}

fn sub(d: &[u8], s: usize, e: usize, want: u32) -> Option<(usize, usize)> {
    let mut hit = None;
    fields(d, s, e, |f, v| {
        if f == want && hit.is_none() {
            hit = v.range();
        }
    });
    hit
}

fn f32_of(d: &[u8], s: usize, e: usize, want: u32) -> Option<f32> {
    let mut hit = None;
    fields(d, s, e, |f, v| {
        if f == want {
            if let Val::F32(x) = v {
                hit = Some(x);
            } else if let Val::Varint(x) = v {
                hit = Some(x as f32);
            }
        }
    });
    hit
}

fn read_geometry(d: &[u8], s: usize, e: usize) -> Geometry {
    let mut g = Geometry::empty();
    let pt = |m: (usize, usize), f: u32| sub(d, m.0, m.1, f).map(|r| v3(d, r.0, r.1));
    fields(d, s, e, |f, v| {
        let Some(m) = v.range() else { return };
        match f {
            1 => {
                g.hub_mass = f32_of(d, m.0, m.1, 3).unwrap_or(0.0);
                g.static_camber = f32_of(d, m.0, m.1, 5).unwrap_or(0.0);
            }
            // double wishbone (2) and DW + coilover (8): same first 8 fields
            2 | 8 => {
                g.kind = SuspKind::DoubleWishbone;
                if let (Some(a), Some(b), Some(c)) = (pt(m, 1), pt(m, 2), pt(m, 5)) {
                    g.up = Some((a, b, c));
                }
                if let (Some(a), Some(b), Some(c)) = (pt(m, 3), pt(m, 4), pt(m, 6)) {
                    g.low = Some((a, b, c));
                }
                if let (Some(a), Some(b)) = (pt(m, 7), pt(m, 8)) {
                    g.steer = Some((a, b));
                }
                if f == 8 {
                    if let (Some(a), Some(b)) = (pt(m, 9), pt(m, 10)) {
                        g.coil = Some((a, b));
                    }
                }
            }
            // strut (3) and strut multilink (9): 1 top, 2 strut-on-hub, 3/4 lower pivots, 5 ball
            3 | 9 => {
                g.kind = SuspKind::Strut;
                g.strut_top = pt(m, 1);
                if let (Some(a), Some(b), Some(c)) = (pt(m, 3), pt(m, 4), pt(m, 5)) {
                    g.low = Some((a, b, c));
                }
                let (sa, sb) = if f == 3 { (6, 7) } else { (9, 10) };
                if let (Some(a), Some(b)) = (pt(m, sa), pt(m, sb)) {
                    g.steer = Some((a, b));
                }
            }
            6 => {
                g.kind = SuspKind::TrailingArm;
                if let (Some(a), Some(b)) = (pt(m, 1), pt(m, 2)) {
                    g.hinge = Some((a, b));
                }
            }
            4 => g.kind = SuspKind::Axle,
            5 | 7 => g.kind = SuspKind::MultiLink,
            _ => {}
        }
    });
    g
}


/// One turbo: AC1 semantics, which EVO kept (TurboData).
#[derive(Clone, Debug)]
pub struct Turbo {
    pub max_boost: f32,
    pub wastegate: f32,
    pub rpm_ref: f32,
    pub gamma_rpm: f32,
    /// controller stages (DynamicControllerData; Add combinator): (input,
    /// LUT). The summed LUTs are the boost target. Input enum
    /// DynamicControllerStageData.Input: 18 RPMS, 2 Gas, 17 Const, ...
    pub ctrl: Vec<(u64, Vec<(f32, f32)>)>,
}

impl Turbo {
    pub fn boost(&self, rpm: f32, throttle: f32) -> f32 {
        let spool = if self.rpm_ref > 0.0 { (rpm / self.rpm_ref).max(0.0).powf(self.gamma_rpm.max(0.1)).min(1.0) } else { 1.0 };
        // ⚠ A turbo with a controller often has NO max_boost at all (296
        // GT3, KTM GT4: proto3 omits the zero) - the controller IS the boost
        // target. Without this those cars ran on 40% of their torque.
        let target = if self.ctrl.is_empty() {
            self.max_boost
        } else {
            self.ctrl.iter().map(|(input, l)| match input {
                18 => lut(l, rpm),
                2 => lut(l, throttle),
                17 => l.first().map_or(0.0, |p| p.1),
                // speed, gear, slip...: not modelled - take the LUT's peak
                _ => l.iter().fold(0.0f32, |a, p| a.max(p.1)),
            }).sum::<f32>().max(0.0)
        };
        let mut b = target * spool * throttle;
        if self.wastegate > 0.0 {
            b = b.min(self.wastegate);
        }
        b.max(0.0)
    }
}

fn lut(l: &[(f32, f32)], x: f32) -> f32 {
    match l.len() {
        0 => 0.0,
        1 => l[0].1,
        _ => {
            if x <= l[0].0 {
                return l[0].1;
            }
            for w in l.windows(2) {
                if x <= w[1].0 {
                    let t = (x - w[0].0) / (w[1].0 - w[0].0).max(1e-6);
                    return w[0].1 + (w[1].1 - w[0].1) * t;
                }
            }
            l[l.len() - 1].1
        }
    }
}

/// A `.curve` file: CurveData {1 values {1 x, 10 y}} (the y lands in field
/// 10 of the point, not 2).
pub fn read_curve(d: &[u8]) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    fields(d, 0, d.len(), |f, v| {
        if let (1, Some((a, b))) = (f, v.range()) {
            let (mut x, mut y, mut y2) = (0.0f32, None, None);
            fields(d, a, b, |g, w| match g {
                1 => x = w.f32(),
                10 => y = Some(w.f32()),
                2 => y2 = Some(w.f32()),
                _ => {}
            });
            out.push((x, y.or(y2).unwrap_or(0.0)));
        }
    });
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn read_curve_path(vfs: &mut Vfs, path: &str) -> Vec<(f32, f32)> {
    match vfs.find(path, None) {
        Some(k) => read_curve(&vfs.read(&k)),
        None => Vec::new(),
    }
}

fn read_turbo(d: &[u8], s: usize, e: usize) -> Turbo {
    let mut t = Turbo { max_boost: 0.0, wastegate: 0.0, rpm_ref: 0.0, gamma_rpm: 1.0, ctrl: Vec::new() };
    fields(d, s, e, |f, v| match f {
        1 => t.max_boost = v.f32(),
        4 => t.rpm_ref = v.f32(),
        5 => t.gamma_rpm = v.f32(),
        7 => t.wastegate = v.f32(),
        _ => {}
    });
    t
}

/// GearBoxData's shifting side (what the box does around a change).
#[derive(Clone, Debug, Default)]
pub struct Gearbox {
    /// f4 / f5 gear_up_time / gear_dn_time, s
    pub up_time: f32,
    pub dn_time: f32,
    /// f6 auto_cut_off_time (s) and f18 gas_cut_off_level (0..1)
    pub cut_time: f32,
    pub gas_cut: f32,
    /// f19 has_dual_clutch, f7 is_shifter_supported (H-pattern)
    pub dual_clutch: bool,
    pub shifter: bool,
    /// f11 gearbox_inertia
    pub inertia: f32,
    /// f8 DownshiftProtectionData {1 isActive, 3 overrev rpm, 4 lockN}
    pub protect: bool,
    pub protect_overrev: f32,
    /// f12 AutoBlipData {1 profile, 2 isElectronic}: the throttle profile
    /// (s -> gas) a downshift plays (FUN_1423fd2c0), and whether the box
    /// blips by itself; otherwise only with the player's auto-blip assist
    pub autoblip: bool,
    pub blip_electronic: bool,
    pub blip_path: String,
    pub blip: Vec<(f32, f32)>,
    /// f8 lockN: no neutral from 1st above 2 km/h; whether f8 is present
    /// (without it a non-H box still protects, at 0 rpm over the limiter)
    pub lock_n: bool,
    pub protect_msg: bool,
    /// f9 valid_shift_rpm_window (0 = 500) and f10 controls_window_gain:
    /// an H gate refuses a gear while clutch x rpm gap x (1 + (gas - 1) x
    /// gain) is at or above the window (FUN_1422e5c90)
    pub shift_window: f32,
    pub window_gain: f32,
    /// f13 AutoShifterData present (the auto-gear assist's own table)
    pub autoshifter: bool,
}

/// ClutchData {1 clutch_inertia, 2 clutch_max_torque, 3 autoclutch, 40
/// clutch_curve}. The drivetrain (fn 0x1420e58a0 / 0x1420e9cc0 /
/// 0x1420e4570) uses it as: engagement e = curve(clutch x 100) / 100
/// (default x^(1.3 + 0.7 cos(pi x)), clutch 1 = pedal up); slipping while
/// e < 1 or the two sides differ by >= 10 %; slip torque = -dw / (|dw| +
/// 4) x max_torque x e.
#[derive(Clone, Debug, Default)]
pub struct Clutch {
    pub inertia: f32,
    pub max_torque: f32,
    pub curve: Vec<(f32, f32)>,
    /// AutoClutchData {1 upshiftProfile, 2 downShiftProfile, 3 forcedOn,
    /// 4 min_rpms, 5 max_rpms, 6 useOnChanges}
    pub auto_forced: bool,
    pub auto_min_rpm: f32,
    pub auto_max_rpm: f32,
    pub auto_on_changes: bool,
    /// the clutch through a gear change, (seconds, 1 engaged .. 0 open);
    /// empty = the auto-clutch leaves the clutch alone for that direction
    pub up_profile: Vec<(f32, f32)>,
    pub down_profile: Vec<(f32, f32)>,
}

/// The player's EVO driving assists that touch the box (CarAssistsValues as
/// RealtimePhysicsEngineController::onCarAssistsChangedCommand applies them:
/// f7 auto_gear -> car+0x7990, f8 auto_blip -> +0x78e8, f10 auto_clutch ->
/// +0x7565 and f11 auto_clutch_on_start -> +0x7564, the last two on H-gate
/// cars only, on start also on any car not forcedOn). Read from the preset
/// the game has applied, Saved Games/ACE/AssistSettings/current.assistpresetsave
/// {1 RacingSettingsAssistPreset: 1 automatic_gearbox, 2 automatic_clutch,
/// 7 automatic_clutch_on_start, 8 automatic_blip}, else the CUSTOM preset.
/// RacingSetting OFF 0 / ON 1 / FREE 2; the game tests > 0.
/// EVOVIEW_AUTO_SHIFT / EVOVIEW_AUTO_BLIP = 0 / 1 and EVOVIEW_AUTO_CLUTCH =
/// 0 / 1 (both) / start override.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Assists {
    pub gear: bool,
    pub blip: bool,
    pub clutch_changes: bool,
    pub clutch_start: bool,
}

impl Assists {
    pub fn read() -> Assists {
        let mut a = Assists::default();
        if let Some(home) = std::env::var_os("USERPROFILE") {
            let dir = std::path::Path::new(&home).join("Saved Games").join("ACE").join("AssistSettings");
            let preset = |d: &[u8], s: usize, e: usize, a: &mut Assists| fields(d, s, e, |f, v| match (f, v) {
                (1, Val::Varint(x)) => a.gear = x > 0,
                (2, Val::Varint(x)) => a.clutch_changes = x > 0,
                (7, Val::Varint(x)) => a.clutch_start = x > 0,
                (8, Val::Varint(x)) => a.blip = x > 0,
                _ => {}
            });
            if let Ok(d) = std::fs::read(dir.join("current.assistpresetsave")) {
                fields(&d, 0, d.len(), |f, v| {
                    if let (1, Some((s, e))) = (f, v.range()) {
                        preset(&d, s, e, &mut a);
                    }
                });
            } else if let Ok(d) = std::fs::read(dir.join("AssistPresetListType_CUSTOM.racingsettingsassistpreset")) {
                preset(&d, 0, d.len(), &mut a);
            }
        }
        let on = |k: &str| std::env::var(k).ok().map(|v| v != "0");
        if let Some(x) = on("EVOVIEW_AUTO_SHIFT") {
            a.gear = x;
        }
        if let Some(x) = on("EVOVIEW_AUTO_BLIP") {
            a.blip = x;
        }
        match std::env::var("EVOVIEW_AUTO_CLUTCH").as_deref() {
            Ok("0") => (a.clutch_changes, a.clutch_start) = (false, false),
            Ok("1") => (a.clutch_changes, a.clutch_start) = (true, true),
            Ok("start") => (a.clutch_changes, a.clutch_start) = (false, true),
            _ => {}
        }
        a
    }
}

/// Wet weather at load: drive on the car's wet setup and wet tyres.
/// EVOVIEW_TYRES=wet / dry decides; otherwise rain (EVOVIEW_RAIN) of 0.3 or
/// more. (The rain key F5 changes the sky, not the car already loaded.)
pub fn wants_wet() -> bool {
    match std::env::var("EVOVIEW_TYRES").as_deref() {
        Ok("wet") => return true,
        Ok("dry") => return false,
        _ => {}
    }
    std::env::var("EVOVIEW_RAIN").ok().and_then(|v| v.parse::<f32>().ok()).is_some_and(|r| r >= 0.3)
}

/// The setup the car loads with: in the wet CarData f191 wet_setup (else a
/// .carsetup named wet), dry f19 stock_setup (else the first dry one).
fn chosen_setup(vfs: &mut Vfs, cd: &[u8]) -> Option<String> {
    let wet = wants_wet();
    let mut named = String::new();
    fields(cd, 0, cd.len(), |f, v| {
        if let (19, Some(r)) = (f, v.range()) {
            if !wet {
                named = text(cd, r);
            }
        }
        if let (191, Some(r)) = (f, v.range()) {
            if wet {
                named = text(cd, r);
            }
        }
    });
    let mut keys = vfs.list_rec("data", "carsetup");
    // a car without a wet setup runs its dry one in the rain
    keys.sort_by_key(|k| k.contains("wet") != wet);
    let found = (!named.is_empty()).then(|| vfs.find(&named, None)).flatten();
    found.or_else(|| {
        let stock = if wet { chosen_setup_dry(cd) } else { None };
        stock.and_then(|s| vfs.find(&s, None)).or_else(|| keys.first().cloned())
    })
}

/// CarData f19 stock_setup.
fn chosen_setup_dry(cd: &[u8]) -> Option<String> {
    let mut named = None;
    fields(cd, 0, cd.len(), |f, v| {
        if let (19, Some(r)) = (f, v.range()) {
            named = Some(text(cd, r)).filter(|s| !s.is_empty());
        }
    });
    named
}

/// The tyres a setup fits: its alignments' f7 compound indexes the car's
/// .compatibletyres physic compounds (f2: {1 compound enum, 2 front .tyre,
/// 3 rear .tyre}); the M4 GT3's 0 is slicks, 1 wets.
fn setup_tyres(vfs: &mut Vfs, setup: &[u8]) -> Option<[String; 2]> {
    let mut idx = 0usize;
    fields(setup, 0, setup.len(), |f, v| {
        if let (4, Some((a, b))) = (f, v.range()) {
            if let Some(c) = f32_of(setup, a, b, 7) {
                idx = idx.max(c.max(0.0) as usize);
            }
        }
    });
    if idx == 0 {
        return None;
    }
    for k in vfs.list("presets", "compatibletyres") {
        let d = vfs.read(&k);
        let mut list: Vec<[String; 2]> = Vec::new();
        fields(&d, 0, d.len(), |f, v| {
            let (2, Some((a, b))) = (f, v.range()) else { return };
            let mut t = [String::new(), String::new()];
            fields(&d, a, b, |g, w| match (g, w.range()) {
                (2, Some(r)) => t[0] = text(&d, r),
                (3, Some(r)) => t[1] = text(&d, r),
                _ => {}
            });
            list.push(t);
        });
        if let Some(t) = list.get(idx).filter(|t| !t[0].is_empty() && !t[1].is_empty()) {
            return Some(t.clone());
        }
    }
    None
}

/// CarSetupData (CarSetup.proto, embedded in the exe): what a .carsetup
/// holds - the car's own in data/setup and the player's in Saved
/// Games/ACE/Car Setups/<car screen name>/<track>/<name>.carsetup (the same
/// message). proto3 drops zeros: a field the file lacks reads 0 in the game.
/// Scalar groups are indexed by field number.
#[derive(Clone, Debug, Default)]
pub struct Setup {
    /// f1 mechanical_balance: 1 arbs [F, R] N/m, 2 steer_ratio, 3 brakes
    /// {1 front_bias %, 2 torque_multiplier %, 3 brake_ducts[]}
    pub arbs: Vec<f32>,
    pub steer_ratio: Option<f32>,
    pub brakes: [Option<f32>; 3],
    pub ducts: Vec<f32>,
    /// 4 differential: 1 power, 2 coast, 3 preload, 4 friction_faces,
    /// 5 power_ramp_deg, 6 coast_ramp_deg
    pub diff: [Option<f32>; 7],
    /// f2 suspensions x4 FL FR RL RR: 1 wheel_rate, 4 helper_rate, 5
    /// helper_range; bump_stop_up / _down {1 range, 2 rate}
    pub susp: Vec<[Option<f32>; 6]>,
    pub stops: Vec<[[Option<f32>; 3]; 2]>,
    /// f3 dampers x4: 1 slow_bump, 2 fast_bump, 3 slow_rebound, 4 fast_rebound
    pub damp: Vec<[Option<f32>; 5]>,
    /// f4 alignments x4: 1 pressure psi, 2 camber, 3 toe, 4 caster, 5
    /// static_camber, 6 toe_out_linear, 7 compound
    pub align: Vec<[Option<f32>; 8]>,
    /// f5 electronics: 1 tc1, 2 tc2, 3 abs, 4 engine_map, 5 telemetry laps,
    /// 6 esc, 7 turbo_boost_lv, 8-10 ers, 11 ebb
    pub elec: Option<[Option<f32>; 12]>,
    /// f6 aero: 1 collar_positions_mm x4, 2/3 front/rear target height mm,
    /// 4/5 front/rear wing angle
    pub collar_mm: Vec<f32>,
    pub aero: [Option<f32>; 6],
    /// f7 fuel_strategy {1 fuel litres}
    pub fuel: Option<f32>,
    /// f9 final_state_name: <car id>_preset_<mech>_preset_<visual>
    pub final_state: String,
    pub version: Option<u64>,
}

fn scalars<const N: usize>(d: &[u8], a: usize, b: usize) -> [Option<f32>; N] {
    let mut o = [None; N];
    fields(d, a, b, |f, v| {
        if let (true, Val::F32(x)) = ((f as usize) < N, v) {
            o[f as usize] = Some(x);
        }
    });
    o
}

/// A repeated float: packed, or one field per value.
fn floats(d: &[u8], a: usize, b: usize, want: u32) -> Vec<f32> {
    let mut out = Vec::new();
    fields(d, a, b, |f, v| match (f == want, v) {
        (true, Val::Bytes(x, y)) => out.extend(d[x..y].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
        (true, Val::F32(x)) => out.push(x),
        _ => {}
    });
    out
}

/// Whether `d` is one protobuf message end to end (the field walk stops
/// quietly at junk; this says whether it got to the end).
fn proto_ok(d: &[u8]) -> bool {
    let mut i = 0usize;
    let vint = |i: &mut usize| -> Option<u64> {
        let (mut r, mut sh) = (0u64, 0u32);
        loop {
            let b = *d.get(*i)?;
            *i += 1;
            r |= ((b & 0x7f) as u64) << sh;
            if b < 0x80 {
                return Some(r);
            }
            sh += 7;
            if sh > 63 {
                return None;
            }
        }
    };
    while i < d.len() {
        let Some(k) = vint(&mut i) else { return false };
        if k >> 3 == 0 {
            return false;
        }
        match k & 7 {
            0 => {
                if vint(&mut i).is_none() {
                    return false;
                }
            }
            1 => i += 8,
            2 => match vint(&mut i) {
                Some(n) => i += n as usize,
                None => return false,
            },
            5 => i += 4,
            _ => return false,
        }
    }
    i == d.len()
}

impl Setup {
    pub fn parse(d: &[u8]) -> Setup {
        let mut s = Setup::default();
        fields(d, 0, d.len(), |f, v| match (f, v) {
            (1, Val::Bytes(a, b)) => {
                s.arbs = floats(d, a, b, 1);
                fields(d, a, b, |g, w| match (g, w) {
                    (2, Val::F32(x)) => s.steer_ratio = Some(x),
                    (3, Val::Bytes(x, y)) => {
                        s.brakes = scalars(d, x, y);
                        s.ducts = floats(d, x, y, 3);
                    }
                    (4, Val::Bytes(x, y)) => s.diff = scalars(d, x, y),
                    _ => {}
                });
            }
            (2, Val::Bytes(a, b)) => {
                s.susp.push(scalars(d, a, b));
                let st = |n| sub(d, a, b, n).map_or([None; 3], |(x, y)| scalars(d, x, y));
                s.stops.push([st(2), st(3)]);
            }
            (3, Val::Bytes(a, b)) => s.damp.push(scalars(d, a, b)),
            (4, Val::Bytes(a, b)) => s.align.push(scalars(d, a, b)),
            (5, Val::Bytes(a, b)) => s.elec = Some(scalars(d, a, b)),
            (6, Val::Bytes(a, b)) => {
                s.collar_mm = floats(d, a, b, 1);
                s.aero = scalars(d, a, b);
            }
            (7, Val::Bytes(a, b)) => s.fuel = f32_of(d, a, b, 1),
            (9, Val::Bytes(a, b)) => s.final_state = text(d, (a, b)),
            (10, Val::Varint(x)) => s.version = Some(x),
            _ => {}
        });
        s
    }

    /// The car id the setup was saved for (final_state_name up to
    /// "_preset_"), "" when the file does not say.
    pub fn car(&self) -> &str {
        self.final_state.split("_preset_").next().unwrap_or("")
    }

    /// A player's setup is the game's whole in-memory setup written out, so
    /// a field it lacks is a zero the player chose - wherever the car's own
    /// setup sets that field (toe taken to 0, TC off, a wing at 0 deg).
    /// Fields the stock setup never sets stay absent: the car does not
    /// adjust them.
    fn zero_fill(&mut self, stock: &Setup) {
        fn fill<const N: usize>(u: &mut [Option<f32>; N], s: &[Option<f32>; N]) {
            for i in 0..N {
                if u[i].is_none() && s[i].is_some() {
                    u[i] = Some(0.0);
                }
            }
        }
        fill(&mut self.brakes, &stock.brakes);
        fill(&mut self.diff, &stock.diff);
        fill(&mut self.aero, &stock.aero);
        for (u, s) in self.susp.iter_mut().zip(&stock.susp) {
            fill(u, s);
        }
        for (u, s) in self.damp.iter_mut().zip(&stock.damp) {
            fill(u, s);
        }
        for (u, s) in self.align.iter_mut().zip(&stock.align) {
            fill(u, s);
        }
        if let (Some(u), Some(s)) = (self.elec.as_mut(), stock.elec.as_ref()) {
            fill(u, s);
        }
        if self.fuel.is_none() && stock.fuel.is_some() {
            self.fuel = Some(0.0);
        }
    }
}

/// A LUT damper's force curve for one setup (fn 0x1423f5d90): the
/// rebound click's curve below 0 m/s, the bump click's above, each run
/// out to +-6 m/s at the setup's fast slope. Clicks are 1-based into the
/// .dampercurves list (DamperCurvesList f1; the apply step takes
/// round(click) - 1).
fn damper_lut(vfs: &mut Vfs, list: &str, bump: f32, rebound: f32, fast_bump: f32, fast_rebound: f32) -> Option<&'static [(f32, f32)]> {
    let k = vfs.find(list, None)?;
    let d = vfs.read(&k);
    let mut paths = Vec::new();
    fields(&d, 0, d.len(), |f, v| {
        if let (1, Some(r)) = (f, v.range()) {
            paths.push(text(&d, r));
        }
    });
    let curve = |vfs: &mut Vfs, click: f32| -> Vec<(f32, f32)> {
        let i = click.round() as i64 - 1;
        if i < 0 {
            return Vec::new();
        }
        paths.get(i as usize).map_or(Vec::new(), |p| read_curve_path(vfs, p))
    };
    let (b, r) = (curve(vfs, bump), curve(vfs, rebound));
    if b.is_empty() || r.is_empty() {
        return None;
    }
    let mut l = Vec::new();
    if r[0].0 < 0.0 {
        l.push((-6.0, r[0].1 - (-6.0 - r[0].0) * fast_rebound));
    }
    l.extend(r.iter().filter(|p| p.0 < 0.0 && p.0 > -6.0));
    l.push((0.0, 0.0));
    l.extend(b.iter().filter(|p| p.0 > 0.0 && p.0 < 6.0));
    if let Some(&last) = b.iter().filter(|p| p.0 > 0.0).last() {
        l.push((6.0, last.1 - (6.0 - last.0) * fast_bump));
    }
    l.dedup_by(|a, b| a.0 <= b.0);
    Some(Box::leak(l.into_boxed_slice()))
}

/// Put a setup on the car, field by field. Values outside what any car
/// could run are skipped (and noted), so a setup from another car or game
/// build cannot put NaN or a zero spring into the physics.
fn apply_setup(vfs: &mut Vfs, p: &mut Physics, s: &Setup) {
    let (mut skipped, mut lut_miss) = (Vec::new(), Vec::new());
    let mut ok = |v: Option<f32>, lo: f32, hi: f32, what: &str| -> Option<f32> {
        let v = v?;
        if v.is_finite() && (lo..=hi).contains(&v) {
            Some(v)
        } else {
            skipped.push(format!("setup {what} {v} outside {lo}..{hi}: kept the car's"));
            None
        }
    };
    for ax in 0..2 {
        if let Some(v) = ok(s.arbs.get(ax).copied().filter(|v| *v > 0.0), 0.0, 2e6, "arb") {
            p.coil[ax].arb = v;
        }
    }
    // the sign is the rack side (CarData controls f3): keep the car's
    if let Some(v) = ok(s.steer_ratio.map(f32::abs).filter(|v| *v > 1.0), 4.0, 30.0, "steer ratio") {
        p.steer_ratio = v * if p.steer_ratio < 0.0 { -1.0 } else { 1.0 };
    }
    // (a fraction, as some car data stores it, read as %)
    if let Some(v) = ok(s.brakes[1].filter(|v| *v > 0.0).map(|v| if v <= 1.0 { v * 100.0 } else { v }), 20.0, 90.0, "brake bias") {
        p.brakes.front_bias = v * 0.01;
    }
    if let Some(v) = ok(s.brakes[2].filter(|v| *v > 0.0), 10.0, 300.0, "brake torque %") {
        p.brakes.total = p.brake_base * v * 0.01;
    }
    // power / coast are fractions as the data's are 0..1; a percent is scaled
    let frac = |v: f32| if v > 1.0 { v * 0.01 } else { v };
    if let Some(v) = ok(s.diff[1], 0.0, 100.0, "diff power") {
        p.diff.power = frac(v);
    }
    if let Some(v) = ok(s.diff[2], 0.0, 100.0, "diff coast") {
        p.diff.coast = frac(v);
    }
    if let Some(v) = ok(s.diff[3], 0.0, 5000.0, "diff preload") {
        p.diff.preload = v;
    }
    let lut_on = std::env::var("EVOVIEW_DAMPER_LUT").map_or(true, |v| v != "0");
    let static_camber = std::env::var("EVOVIEW_SETUP_CAMBER").is_ok_and(|v| v == "static");
    let collars = std::env::var("EVOVIEW_SETUP_COLLAR").map_or(true, |v| v != "0");
    // per axle from its left wheel: FL (0) and RL (2)
    for (ax, i) in [(0usize, 0usize), (1, 2)] {
        if let Some(u) = s.susp.get(i) {
            let c = &mut p.coilover[ax];
            if let Some(k) = ok(u[1].filter(|k| *k > 0.0), 5000.0, 1.5e6, "spring") {
                c.k = k;
            }
            if let Some(h) = ok(u[4], 0.0, 1e6, "helper rate") {
                c.helper_k = h;
            }
            if let Some(h) = ok(u[5], 0.0, 0.5, "helper range") {
                c.helper_range = h;
            }
        }
        if let Some(u) = s.damp.get(i) {
            let fb = ok(u[2].filter(|v| *v > 0.0), 0.0, 2e5, "fast bump");
            let fr = ok(u[4].filter(|v| *v > 0.0), 0.0, 2e5, "fast rebound");
            match p.damper_lut[ax].clone() {
                // a LUT car: the slow values are clicks into the curve list
                Some(list) => {
                    let dm = p.coilover[ax].damper;
                    let (fb, fr) = (fb.unwrap_or(dm.fast_bump), fr.unwrap_or(dm.fast_rebound));
                    if let (true, Some(b), Some(r)) = (lut_on, u[1], u[3]) {
                        match damper_lut(vfs, &list, b, r, fb, fr) {
                            Some(l) => p.coilover[ax].damper.lut = Some(l),
                            None => lut_miss.push(format!("setup damper clicks {b} / {r}: not in {list}")),
                        }
                    }
                    let dm = &mut p.coilover[ax].damper;
                    (dm.fast_bump, dm.fast_rebound) = (fb, fr);
                }
                None => {
                    let dm = &mut p.coilover[ax].damper;
                    let g = |f: usize| u[f].filter(|v| *v > 0.0 && v.is_finite());
                    if let Some(v) = g(1) {
                        dm.slow_bump = v;
                    }
                    if let Some(v) = fb {
                        dm.fast_bump = v;
                    }
                    if let Some(v) = g(3) {
                        dm.slow_rebound = v;
                    }
                    if let Some(v) = fr {
                        dm.fast_rebound = v;
                    }
                }
            }
        }
        if let Some(u) = s.align.get(i) {
            // camber: the setup's camber (f2) is the wheel's at ride height;
            // static_camber (f5) is what the game puts into the hub so its
            // linkage ends there (the 296 rear: -2.5 from -6.1). evoview has
            // no hard-point camber at rest, so it takes f2 itself
            // (EVOVIEW_SETUP_CAMBER=static: f5, as before)
            let cam = if static_camber { u[5].or(u[2]) } else { u[2].or(u[5]) };
            let cam = ok(cam, -12.0, 6.0, "camber").unwrap_or(p.geo[ax].static_camber);
            // toe: the angle (f3); the game drives toe_out_linear (f6), the
            // tie-rod offset that gives it
            let toe = ok(Some(u[3].unwrap_or(0.0)), -5.0, 5.0, "toe").unwrap_or(0.0);
            p.align[ax] = Some((cam, toe));
            if let Some(v) = ok(u[1], 5.0, 60.0, "tyre pressure") {
                p.pressure[ax] = Some(v);
            }
        }
        // the game: coilover collar = collar_positions_mm / 1000, the collar
        // its ride-height solve picked for the target heights
        // (EVOVIEW_SETUP_COLLAR=0: the car data's)
        if collars {
            if let Some(v) = ok(s.collar_mm.get(i).copied(), -200.0, 400.0, "collar mm") {
                p.coilover[ax].collar = v / 1000.0;
            }
        }
        if let Some(v) = ok(s.aero[2 + ax], 5.0, 400.0, "ride height mm") {
            p.ride_h[ax] = Some(v / 1000.0);
        }
        if let Some(v) = ok(s.aero[4 + ax], -20.0, 45.0, "wing angle") {
            p.wing[ax] = Some(v);
        }
    }
    if let Some(e) = s.elec {
        if let (Some(l), Some(abs)) = (ok(e[3], 0.0, 20.0, "ABS level"), p.abs.as_mut()) {
            abs.level = l as usize;
        }
        if let (Some(l), Some(tc)) = (ok(e[1], 0.0, 20.0, "TC level"), p.tc.as_mut()) {
            tc.level = l as usize;
        }
    }
    if let Some(v) = ok(s.fuel, 0.0, 300.0, "fuel") {
        p.fuel_l = v;
    }
    p.notes.extend(skipped);
    p.notes.extend(lut_miss);
}

/// The car's stock setup (CarData f19 stock_setup, else the first dry
/// .carsetup; the wet one in the rain) on top of the base data, as the game
/// loads a car. The MX-5 NA's base data has a 9.5 kN/m front bar and its
/// setup 20 kN/m; springs and dampers are the setup's too.
fn apply_stock_setup(vfs: &mut Vfs, p: &mut Physics) -> Option<Setup> {
    let cd = vfs.read(&p.source);
    let k = chosen_setup(vfs, &cd)?;
    let s = Setup::parse(&vfs.read(&k));
    apply_setup(vfs, p, &s);
    p.setup_source = k;
    Some(s)
}

/// Where the game keeps the player's setups (EVOVIEW_SETUPS_DIR overrides).
pub fn setups_dir() -> Option<std::path::PathBuf> {
    if let Some(d) = std::env::var_os("EVOVIEW_SETUPS_DIR") {
        return Some(d.into());
    }
    let home = std::env::var_os("USERPROFILE")?;
    Some(std::path::Path::new(&home).join("Saved Games").join("ACE").join("Car Setups"))
}

/// Every saved setup: (path, its path under the setups folder with '/'
/// separators, the parsed setup). The game files them as <car screen
/// name>/<track>/<name>.carsetup.
pub fn saved_setups() -> Vec<(std::path::PathBuf, String, Setup)> {
    let Some(root) = setups_dir() else { return Vec::new() };
    let mut out = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("carsetup")) {
                let rel = p.strip_prefix(&root).map_or(p.display().to_string(), |r| r.display().to_string()).replace('\\', "/");
                let s = std::fs::read(&p).map(|d| Setup::parse(&d)).unwrap_or_default();
                out.push((p, rel, s));
            }
        }
    }
    out.sort_by(|a, b| a.1.to_lowercase().cmp(&b.1.to_lowercase()));
    out
}

/// Whether a setup saved for `setup_car` fits `car` (a mod built on a
/// Kunos car keeps its donor's final state: ks_toyota_gr86_mod_x <-
/// ks_toyota_gr86).
fn car_fits(setup_car: &str, car: &str) -> bool {
    let (a, b) = (setup_car.to_lowercase(), car.to_lowercase());
    !a.is_empty() && (a == b || b.starts_with(&format!("{a}_")))
}

/// `--list-setups [car]`: the player's saved setups, one per line,
/// tab-separated: path, car folder (screen name), track, setup name, the car
/// id it was saved for (its final state). With a car id, only that car's.
pub fn list_setups(car: Option<&str>) {
    let Some(root) = setups_dir() else {
        eprintln!("no USERPROFILE: cannot find Saved Games/ACE/Car Setups");
        return;
    };
    let all = saved_setups();
    println!("# {} setup(s) in {}{}", all.len(), root.display(), car.map_or(String::new(), |c| format!(", showing {c}'s")));
    println!("# path\tcar folder\ttrack\tname\tsaved for");
    for (p, rel, s) in &all {
        if car.is_some_and(|c| !car_fits(s.car(), c)) {
            continue;
        }
        let parts: Vec<&str> = rel.split('/').collect();
        let (folder, track) = match parts.len() {
            n if n >= 3 => (parts[n - 3], parts[n - 2]),
            2 => (parts[0], ""),
            _ => ("", ""),
        };
        let name = p.file_stem().map_or(String::new(), |s| s.to_string_lossy().into_owned());
        println!("{}\t{folder}\t{track}\t{name}\t{}", p.display(), if s.car().is_empty() { "?" } else { s.car() });
    }
}

/// The car id a Physics was read for: its .car sits in content/cars/<id>/.
fn car_id_of(source: &str) -> Option<String> {
    let low = source.replace('\\', "/").to_lowercase();
    let rest = low.split("content/cars/").nth(1)?;
    rest.split('/').next().map(|s| s.to_string()).filter(|s| !s.is_empty())
}

/// EVOVIEW_SETUP=<file | name> (main sets it from --setup): the player's
/// setup to drive with. A name is looked up under the setups folder by its
/// tail ("lagua", "laguna seca/lagua", "ferrari 296 gt3/laguna seca/lagua").
/// Err says why it was not used (the car then runs its stock setup).
fn user_setup(car: Option<&str>, screen: &str) -> Option<Result<(String, Vec<u8>, Setup), String>> {
    let want = std::env::var("EVOVIEW_SETUP").ok().filter(|v| !v.trim().is_empty())?;
    let path = if std::path::Path::new(&want).is_file() {
        std::path::PathBuf::from(&want)
    } else {
        let tail = want.replace('\\', "/").to_lowercase();
        let tail = tail.trim_end_matches(".carsetup");
        let mut hits: Vec<_> = saved_setups().into_iter().filter(|(_, rel, _)| {
            let r = rel.to_lowercase();
            let r = r.trim_end_matches(".carsetup");
            r == tail || r.ends_with(&format!("/{tail}"))
        }).collect();
        if hits.len() > 1 {
            if let Some(c) = car {
                hits.retain(|h| car_fits(h.2.car(), c));
            }
        }
        match hits.len() {
            0 => return Some(Err(format!("no setup file or saved setup named {want:?} (--list-setups shows them)"))),
            1 => hits.remove(0).0,
            _ => return Some(Err(format!("{want:?} matches {} saved setups: {}", hits.len(),
                                         hits.iter().map(|h| h.1.clone()).collect::<Vec<_>>().join(", ")))),
        }
    };
    let shown = path.display().to_string();
    let d = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => return Some(Err(format!("{shown}: {e}"))),
    };
    if d.is_empty() || !proto_ok(&d) {
        return Some(Err(format!("{shown}: not a CarSetupData protobuf")));
    }
    let s = Setup::parse(&d);
    // any protobuf walks; a setup names its car or version, or has a
    // spring / tyre pressure as a plain float (the .carsetuplimits for the
    // same fields holds messages there)
    let floats_in = s.susp.iter().any(|u| u[1].is_some()) || s.align.iter().any(|u| u[1].is_some() || u[2].is_some());
    if !floats_in || (s.final_state.is_empty() && s.version.is_none()) {
        return Some(Err(format!("{shown}: not a car setup (no springs / pressures, final state or version in it)")));
    }
    // the car it was saved for: the final state's id, else the folder the
    // game filed it under (the car's screen name)
    let folder = path.parent().and_then(|t| t.parent()).and_then(|c| c.file_name()).map(|c| c.to_string_lossy().to_lowercase());
    let fits = match (s.car(), car) {
        ("", _) => folder.as_deref().map_or(true, |f| screen.is_empty() || f == screen.to_lowercase()),
        (_, None) => true,
        (sc, Some(c)) => car_fits(sc, c),
    };
    if !fits && std::env::var_os("EVOVIEW_SETUP_FORCE").is_none() {
        return Some(Err(format!("{shown} was saved for {} ({}), not {} - EVOVIEW_SETUP_FORCE=1 applies it anyway",
                                if s.car().is_empty() { "?" } else { s.car() }, folder.unwrap_or_default(),
                                car.unwrap_or(screen))));
    }
    Some(Ok((shown, d, s)))
}

/// Every setup-driven number, for the before / after table.
fn setup_rows(p: &Physics) -> Vec<(&'static str, String)> {
    let two = |a: f32, b: f32, prec: usize| format!("{a:.prec$} / {b:.prec$}");
    let c = &p.coilover;
    let opt = |v: Option<f32>, prec: usize| v.map_or("-".to_string(), |v| format!("{v:.prec$}"));
    let dmp = |ax: usize| {
        let d = &c[ax].damper;
        format!("{:.0} / {:.0} N @ 0.05 m/s, {:.0} / {:.0} @ 0.3{}", -d.force(0.05), d.force(-0.05), -d.force(0.3), d.force(-0.3),
                if d.lut.is_some() { format!(" (LUT x{:.2})", d.lut_scale) } else { String::new() })
    };
    vec![
        ("arb F / R N/m", two(p.coil[0].arb, p.coil[1].arb, 0)),
        ("steer ratio", format!("{:.2}", p.steer_ratio.abs())),
        ("brake bias % front", format!("{:.1}", p.brakes.front_bias * 100.0)),
        ("brake torque N m", format!("{:.0}", p.brakes.total)),
        ("diff power / coast", two(p.diff.power, p.diff.coast, 3)),
        ("diff preload N m", format!("{:.0}", p.diff.preload)),
        ("spring F / R N/m", two(c[0].k, c[1].k, 0)),
        ("helper F / R N/m", two(c[0].helper_k, c[1].helper_k, 0)),
        ("collar F / R mm", two(c[0].collar * 1000.0, c[1].collar * 1000.0, 2)),
        ("damper F bump / reb", dmp(0)),
        ("damper R bump / reb", dmp(1)),
        ("damper fast F / R", format!("{:.0}-{:.0} / {:.0}-{:.0} N s/m", c[0].damper.fast_bump, c[0].damper.fast_rebound, c[1].damper.fast_bump, c[1].damper.fast_rebound)),
        ("camber F / R deg", p.align.map(|a| opt(a.map(|a| a.0), 2)).join(" / ")),
        ("toe F / R deg", p.align.map(|a| opt(a.map(|a| a.1), 2)).join(" / ")),
        ("pressure F / R psi", p.pressure.map(|v| opt(v, 1)).join(" / ")),
        ("TC level", p.tc.as_ref().map_or("-".into(), |t| t.level.to_string())),
        ("ABS level", p.abs.as_ref().map_or("-".into(), |a| a.level.to_string())),
        ("wing F / R deg", p.wing.map(|v| opt(v, 1)).join(" / ")),
        ("ride height F / R mm", p.ride_h.map(|v| opt(v.map(|v| v * 1000.0), 0)).join(" / ")),
        ("fuel L", format!("{:.1}", p.fuel_l)),
        ("mass kg (with fuel)", format!("{:.1}", p.mass_dry + p.fuel_mass())),
    ]
}

/// Whether `key` is new this process: Physics is read several times for
/// one car (rig, drive, lap sim) and the setup report prints once.
fn first_time(key: String) -> bool {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = std::sync::OnceLock::new();
    SEEN.get_or_init(Default::default).lock().map_or(false, |mut s| s.insert(key))
}

/// The before / after table: the changed rows (EVOVIEW_SETUP_TABLE=1: all).
fn print_setup_change(before: &[(&'static str, String)], after: &[(&'static str, String)]) {
    let all = std::env::var_os("EVOVIEW_SETUP_TABLE").is_some();
    let mut n = 0;
    for ((name, b), (_, a)) in before.iter().zip(after) {
        if a != b || all {
            println!("setup: {}{name:22} {b:>40}  ->  {a}", if a != b { "* " } else { "  " });
            n += (a != b) as usize;
        }
    }
    println!("setup: {n} value(s) changed from the stock setup");
}

fn read_clutch(vfs: &mut Vfs, d: &[u8], s: usize, e: usize) -> Clutch {
    let mut c = Clutch { max_torque: 450.0, auto_min_rpm: 1500.0, auto_max_rpm: 2500.0, ..Default::default() };
    let mut curve = None;
    let mut profiles: (Option<String>, Option<String>) = (None, None);
    fields(d, s, e, |f, v| match (f, v) {
        (1, Val::F32(x)) => c.inertia = x,
        (2, Val::F32(x)) if x > 0.0 => c.max_torque = x,
        (40, Val::Bytes(a, b)) => curve = Some(text(d, (a, b))),
        (3, Val::Bytes(a, b)) => fields(d, a, b, |g, w| match (g, w) {
            (1, Val::Bytes(x, y)) => profiles.0 = Some(text(d, (x, y))),
            (2, Val::Bytes(x, y)) => profiles.1 = Some(text(d, (x, y))),
            (3, Val::Varint(x)) => c.auto_forced = x != 0,
            (4, Val::F32(x)) if x > 0.0 => c.auto_min_rpm = x,
            (5, Val::F32(x)) if x > 0.0 => c.auto_max_rpm = x,
            (6, Val::Varint(x)) => c.auto_on_changes = x != 0,
            _ => {}
        }),
        _ => {}
    });
    if let Some(path) = curve.filter(|p| !p.is_empty()) {
        c.curve = read_curve_path(vfs, &path);
    }
    if let Some(path) = profiles.0.filter(|p| !p.is_empty()) {
        c.up_profile = read_curve_path(vfs, &path);
    }
    if let Some(path) = profiles.1.filter(|p| !p.is_empty()) {
        c.down_profile = read_curve_path(vfs, &path);
    }
    c
}

fn read_gearbox(d: &[u8], s: usize, e: usize, p: &mut Physics) {
    let mut fwd = Vec::new();
    let gb = &mut p.gearbox;
    fields(d, s, e, |g, w| match (g, w) {
        (4, Val::F32(x)) => gb.up_time = x,
        (5, Val::F32(x)) => gb.dn_time = x,
        (6, Val::F32(x)) => gb.cut_time = x,
        (18, Val::F32(x)) => gb.gas_cut = x,
        (19, Val::Varint(x)) => gb.dual_clutch = x != 0,
        (7, Val::Varint(x)) => gb.shifter = x != 0,
        (11, Val::F32(x)) => gb.inertia = x,
        (8, Val::Bytes(a, b)) => {
            gb.protect_msg = true;
            fields(d, a, b, |h, x| match (h, x) {
                (1, Val::Varint(v)) => gb.protect = v != 0,
                (3, Val::Varint(v)) => gb.protect_overrev = v as f32,
                (4, Val::Varint(v)) => gb.lock_n = v != 0,
                _ => {}
            })
        }
        (9, Val::F32(x)) => gb.shift_window = x,
        (10, Val::F32(x)) => gb.window_gain = x,
        (13, Val::Bytes(..)) => gb.autoshifter = true,
        // AutoBlipData {1 profile, 2 isElectronic}
        (12, Val::Bytes(a, b)) => fields(d, a, b, |h, x| match (h, x) {
            (1, Val::Bytes(i, j)) if j > i => {
                gb.autoblip = true;
                gb.blip_path = text(d, (i, j));
            }
            (2, Val::Varint(v)) => gb.blip_electronic = v != 0,
            _ => {}
        }),
        _ => {}
    });
    fields(d, s, e, |g, w| match (g, w) {
        (2, Val::Bytes(a, b)) => {
            let mut name = String::new();
            let mut ratio = 0.0;
            fields(d, a, b, |h, x| match (h, x) {
                (1, Val::Bytes(i, j)) => name = text(d, (i, j)),
                (2, Val::F32(r)) => ratio = r,
                _ => {}
            });
            if name == "R" || ratio < 0.0 {
                p.reverse = ratio.abs();
            } else if name != "N" && ratio > 0.0 {
                fwd.push(ratio);
            }
        }
        (3, Val::F32(x)) => p.final_drive = x,
        _ => {}
    });
    if !fwd.is_empty() {
        p.gears = fwd;
    }
}

/// EngineData: 2 powerCurve (rpm -> N m before boost), 4 maps {1 powerMult},
/// 24 turbos (inline TurboData) / 29 turbos_to_load (.turbo paths),
/// 25 turboControllers {2 stages {1 inputVar, 2 combinator, 3 lut}}.
fn read_engine(vfs: &mut Vfs, d: &[u8], s: usize, e: usize, p: &mut Physics) {
    let mut curve = None;
    let mut coast = None;
    let mut map_mult = None;
    if let Some(x) = f32_of(d, s, e, 1) {
        p.engine_inertia = x;
    }
    let mut inline: Vec<Turbo> = Vec::new();
    let mut files: Vec<String> = Vec::new();
    let mut ctrls: Vec<Vec<(u64, String)>> = Vec::new();
    fields(d, s, e, |f, v| {
        let Some((a, b)) = v.range() else { return };
        match f {
            2 => curve = Some(text(d, (a, b))),
            3 => coast = Some(text(d, (a, b))),
            4 if map_mult.is_none() => map_mult = f32_of(d, a, b, 1),
            24 => inline.push(read_turbo(d, a, b)),
            29 => files.push(text(d, (a, b))),
            25 => {
                let mut stages = Vec::new();
                fields(d, a, b, |g, w| {
                    if let (2, Some((x, y))) = (g, w.range()) {
                        let mut input = 0u64;
                        let mut path = None;
                        fields(d, x, y, |h, u| match (h, u) {
                            (1, Val::Varint(i)) => input = i,
                            (3, Val::Bytes(i, j)) => path = Some(text(d, (i, j))),
                            _ => {}
                        });
                        if let Some(pth) = path {
                            stages.push((input, pth));
                        }
                    }
                });
                ctrls.push(stages);
            }
            _ => {}
        }
    });
    if let Some(c) = curve {
        p.torque = read_curve_path(vfs, &c);
    }
    if let Some(c) = coast {
        p.coast = read_curve_path(vfs, &c);
    }
    p.power_mult = map_mult.filter(|m| *m > 0.05 && *m < 5.0).unwrap_or(1.0);
    // the .turbo files are the same TurboData; inline copies win when present
    let mut turbos = inline;
    if turbos.is_empty() {
        for f in &files {
            if let Some(k) = vfs.find(f, None) {
                let td = vfs.read(&k);
                turbos.push(read_turbo(&td, 0, td.len()));
            }
        }
    }
    for (i, t) in turbos.iter_mut().enumerate() {
        if let Some(stages) = ctrls.get(i) {
            for (input, st) in stages {
                let l = read_curve_path(vfs, st);
                if !l.is_empty() {
                    t.ctrl.push((*input, l));
                }
            }
        }
    }
    p.turbos = turbos.into_iter().filter(|t| t.max_boost > 0.0 || !t.ctrl.is_empty()).collect();
}

impl Physics {
    /// Full-throttle-scaled engine torque at the crank, N m: power curve x
    /// (1 + boost) x the engine map's power multiplier. None when the car
    /// has no power curve.
    pub fn engine_torque(&self, rpm: f32, throttle: f32) -> Option<f32> {
        if self.torque.len() < 2 {
            return None;
        }
        let base = lut(&self.torque, rpm.max(self.torque[0].0));
        let boost: f32 = self.turbos.iter().map(|t| t.boost(rpm, throttle)).sum();
        Some(base * (1.0 + boost) * self.power_mult * throttle)
    }

    /// Peak of engine_torque at full throttle (diagnostic).
    pub fn peak_torque(&self) -> Option<(f32, f32)> {
        let mut best: Option<(f32, f32)> = None;
        let mut r = self.idle_rpm.max(500.0);
        while r <= self.limiter_rpm {
            if let Some(t) = self.engine_torque(r, 1.0) {
                if best.map_or(true, |b| t > b.1) {
                    best = Some((r, t));
                }
            }
            r += 100.0;
        }
        best
    }
}

fn read_coil(d: &[u8], s: usize, e: usize) -> Coilover {
    let mut c = Coilover::default();
    c.spring = f32_of(d, s, e, 1).unwrap_or(0.0);
    // travel to the bump stop plus the stop's own squash; the same for droop
    let stop = |f: u32| -> f32 {
        sub(d, s, e, f).map(|r| {
            f32_of(d, r.0, r.1, 1).unwrap_or(0.0).max(0.0) + f32_of(d, r.0, r.1, 5).unwrap_or(0.0).max(0.0)
        }).unwrap_or(0.0)
    };
    c.bump = stop(3);
    c.droop = stop(4);
    c.bump_free = sub(d, s, e, 3).and_then(|r| f32_of(d, r.0, r.1, 1)).unwrap_or(c.bump).max(0.0);
    c
}

impl Physics {
    pub fn read(vfs: &mut Vfs) -> Option<Physics> {
        let mut cars: Vec<String> = vfs.list("data", "car");
        cars.sort_by_key(|k| (!k.ends_with("_car.car"), !k.ends_with("cardata.car"), k.clone()));
        let key = cars.into_iter().next()?;
        let d = vfs.read(&key);
        if d.is_empty() {
            return None;
        }
        let mut p = Physics {
            source: key.clone(),
            mass: 1200.0,
            body_box: Vec3::new(1.8, 1.3, 4.3),
            wheelbase: 0.0,
            cg_front: 0.5,
            track: [0.0; 2],
            base_y: [0.0; 2],
            coil: [Coilover::default(); 2],
            geo: [Geometry::empty(), Geometry::empty()],
            steer_lock_deg: 0.0,
            ff_mult: 0.0,
            ff_gamma: 0.0,
            steer_ratio: 0.0,
            idle_rpm: 0.0,
            limiter_rpm: 0.0,
            limiter_raw: 0.0,
            torque: Vec::new(),
            turbos: Vec::new(),
            power_mult: 1.0,
            tyre_radius: None,
            tyres: [None, None],
            thermal: [None, None],
            traction: 0,
            coilover: Default::default(),
            engine_inertia: 0.0,
            setup_source: String::new(),
            align: [None, None],
            gearbox: Gearbox::default(),
            clutch: Clutch { max_torque: 450.0, auto_min_rpm: 1500.0, auto_max_rpm: 2500.0, ..Default::default() },
            abs: None,
            tc: None,
            coast: Vec::new(),
            diff: Default::default(),
            brakes: Default::default(),
            limits: Default::default(),
            engine_source: String::new(),
            screen_name: String::new(),
            mass_dry: 0.0,
            kg_per_l: 0.75,
            fuel_l: 0.0,
            brake_base: 0.0,
            pressure: [None; 2],
            ride_h: [None; 2],
            wing: [None; 2],
            damper_lut: [None, None],
            user_setup: String::new(),
            electric: false,
            gears: Vec::new(),
            reverse: 0.0,
            final_drive: 0.0,
            notes: Vec::new(),
        };
        let n = d.len();
        let mut geo_paths: [Option<String>; 2] = [None, None];
        let mut engine_path = None;
        let mut engine_inline: Option<(usize, usize)> = None;
        let mut gearbox_path = None;
        let mut clutch_inline: Option<(usize, usize)> = None;
        let mut clutch_path: Option<String> = None;
        let mut tyre_paths: [Option<String>; 2] = [None, None];
        let mut drivetrain_path: Option<String> = None;
        let mut brakes_path: Option<String> = None;
        let mut limits_path: Option<String> = None;
        let traction_of = |d: &[u8], s: usize, e: usize| -> Option<u32> {
            let mut t = None;
            fields(d, s, e, |g, w| {
                if let (2, Val::Varint(x)) = (g, w) {
                    t = Some(x as u32);
                }
            });
            t
        };
        fields(&d, 0, n, |f, v| {
            let Some((s, e)) = v.range() else { return };
            match f {
                1 => {
                    if let Some(m) = f32_of(&d, s, e, 1) { p.mass = m; }
                    if let Some(k) = f32_of(&d, s, e, 8).filter(|k| (0.3..1.5).contains(k)) { p.kg_per_l = k; }
                    if let Some(r) = sub(&d, s, e, 2) { p.screen_name = text(&d, r); }
                    if let Some(r) = sub(&d, s, e, 9) {
                        let b = v3(&d, r.0, r.1);
                        if b.min_element() > 0.1 { p.body_box = b; }
                    }
                }
                2 => fields(&d, s, e, |g, w| match (g, w) {
                    (2, Val::F32(x)) => p.wheelbase = x,
                    (3, Val::F32(x)) => p.cg_front = x,
                    (50, Val::F32(x)) => p.base_y[0] = x,
                    (51, Val::F32(x)) => p.base_y[1] = x,
                    (52, Val::F32(x)) => p.track[0] = x,
                    (53, Val::F32(x)) => p.track[1] = x,
                    (8 | 9, Val::Bytes(a, b)) => {
                        let arb = p.coil[(g - 8) as usize].arb;
                        p.coil[(g - 8) as usize] = read_coil(&d, a, b);
                        p.coilover[(g - 8) as usize] = crate::evophys::Coilover::read(&d, a, b);
                        // CoiloverData f6 damper -> DampersData f13 lut_list
                        p.damper_lut[(g - 8) as usize] = sub(&d, a, b, 6).and_then(|r| sub(&d, r.0, r.1, 13))
                            .map(|r| text(&d, r)).filter(|t| !t.is_empty());
                        p.coil[(g - 8) as usize].arb = arb;
                    }
                    (18 | 19, Val::Bytes(a, b)) => {
                        p.coil[(g - 18) as usize].arb = f32_of(&d, a, b, 1).unwrap_or(0.0);
                    }
                    (40 | 41, Val::Bytes(a, b)) => p.geo[(g - 40) as usize] = read_geometry(&d, a, b),
                    (22 | 23, Val::Bytes(a, b)) => geo_paths[(g - 22) as usize] = Some(text(&d, (a, b))),
                    _ => {}
                }),
                4 => {
                    if let Some(x) = f32_of(&d, s, e, 5) { p.idle_rpm = x; }
                    if let Some(x) = f32_of(&d, s, e, 6) { p.limiter_rpm = x; }
                    engine_inline = Some((s, e));
                }
                20 => engine_path = Some(text(&d, (s, e))),
                3 => {
                    if let Some(t) = traction_of(&d, s, e) { p.traction = t; }
                    if let Some((a, b)) = sub(&d, s, e, 39) { p.diff = crate::evophys::Diff::read(&d, a, b); }
                }
                22 => drivetrain_path = Some(text(&d, (s, e))),
                5 => p.brakes = crate::evophys::Brakes::read(&d, s, e),
                27 => brakes_path = Some(text(&d, (s, e))),
                16 => limits_path = Some(text(&d, (s, e))),
                24 => gearbox_path = Some(text(&d, (s, e))),
                10 => tyre_paths[0] = Some(text(&d, (s, e))),
                11 => tyre_paths[1] = Some(text(&d, (s, e))),
                8 => {
                    if let Some(x) = f32_of(&d, s, e, 1) { p.ff_mult = x; }
                    if let Some(x) = f32_of(&d, s, e, 2) { p.steer_lock_deg = x; }
                    if let Some(x) = f32_of(&d, s, e, 3) { p.steer_ratio = x; }
                    if let Some(x) = f32_of(&d, s, e, 5) { p.ff_gamma = x; }
                }
                23 => read_gearbox(&d, s, e, &mut p),
                25 => clutch_inline = Some((s, e)),
                26 => clutch_path = Some(text(&d, (s, e))),
                _ => {}
            }
        });
        // geometry kept in its own file when CarData does not inline it
        for a in 0..2 {
            if p.geo[a].kind == SuspKind::None {
                if let Some(k) = geo_paths[a].as_ref().and_then(|g| vfs.find(g, None)) {
                    let gd = vfs.read(&k);
                    p.geo[a] = read_geometry(&gd, 0, gd.len());
                }
            }
        }
        // The engine: the .carengine file (CarData f20) is what the game
        // loads; the inline copy (f4) is a fallback for files that lack it.
        let engine_file = engine_path.as_ref().and_then(|e| vfs.find(e, None));
        let (ed, es, ee, src) = match engine_file {
            Some(k) => {
                let ed = vfs.read(&k);
                let n = ed.len();
                (ed, 0, n, k)
            }
            None => match engine_inline {
                Some((a, b)) => (d.clone(), a, b, format!("{key} (inline f4)")),
                None => (Vec::new(), 0, 0, String::new()),
            },
        };
        if !ed.is_empty() {
            p.engine_source = src;
            if let Some(x) = f32_of(&ed, es, ee, 6) { p.limiter_rpm = x; }
            if let Some(x) = f32_of(&ed, es, ee, 5) { p.idle_rpm = x; }
            p.electric = f32_of(&ed, es, ee, 51).map_or(false, |t| t == 1.0);
            read_engine(vfs, &ed, es, ee, &mut p);
        }
        if p.gears.is_empty() {
            if let Some(k) = gearbox_path.as_ref().and_then(|g| vfs.find(g, None)) {
                let gd = vfs.read(&k);
                read_gearbox(&gd, 0, gd.len(), &mut p);
            }
        }
        if !p.gearbox.blip_path.is_empty() {
            p.gearbox.blip = read_curve_path(vfs, &p.gearbox.blip_path.clone());
        }
        // the clutch: inline, else its own file
        if let Some((a, b)) = clutch_inline {
            p.clutch = read_clutch(vfs, &d, a, b);
        } else if let Some(k) = clutch_path.as_ref().and_then(|c| vfs.find(c, None)) {
            let cd = vfs.read(&k);
            p.clutch = read_clutch(vfs, &cd, 0, cd.len());
        }
        // the drivetrain file, when the layout is not inline
        if let Some(k) = drivetrain_path.as_ref().and_then(|t| vfs.find(t, None)) {
            let dd = vfs.read(&k);
            if let Some(t) = traction_of(&dd, 0, dd.len()) {
                p.traction = t;
            }
            if let Some((a, b)) = sub(&dd, 0, dd.len(), 39) {
                p.diff = crate::evophys::Diff::read(&dd, a, b);
            }
        }
        if let Some((a, b)) = sub(&d, 0, d.len(), 7).and_then(|(a, b)| sub(&d, a, b, 1)) {
            let tc = crate::evophys::Tc::read(&d, a, b);
            if tc.levels.iter().any(|l| l[0] > 0.0) {
                p.tc = Some(tc);
            }
        }
        if let Some((a, b)) = sub(&d, 0, d.len(), 7).and_then(|(a, b)| sub(&d, a, b, 2)) {
            let abs = crate::evophys::Abs::read(&d, a, b);
            if abs.levels.iter().any(|l| l[0] > 0.0) {
                p.abs = Some(abs);
            }
        }
        if let Some(k) = brakes_path.as_ref().and_then(|t| vfs.find(t, None)) {
            let bd = vfs.read(&k);
            p.brakes = crate::evophys::Brakes::read(&bd, 0, bd.len());
        }
        if let Some(k) = limits_path.as_ref().and_then(|t| vfs.find(t, None)) {
            p.limits = crate::aids::Limits::read(&vfs.read(&k));
        }
        p.brake_base = p.brakes.total;
        p.mass_dry = p.mass;
        // the player's setup (EVOVIEW_SETUP): checked against this car now,
        // applied over the stock setup below
        let car_id = car_id_of(&key);
        let user = if std::env::var_os("EVOVIEW_NO_SETUP").is_none() {
            match user_setup(car_id.as_deref(), &p.screen_name) {
                Some(Ok(u)) => Some(u),
                Some(Err(e)) => {
                    // once per process, loud: the car runs its stock setup
                    if first_time(format!("err {e}")) {
                        eprintln!("setup: ⚠ {e} - driving the stock setup");
                    }
                    p.notes.push(format!("EVOVIEW_SETUP not used: {e}"));
                    None
                }
                None => None,
            }
        } else {
            None
        };
        // the tyres the setup fits (the wet setup's in the wet; the
        // player's compound when theirs is loaded)
        if std::env::var_os("EVOVIEW_NO_SETUP").is_none() {
            let from = match &user {
                Some((k, sd, _)) => Some((k.clone(), sd.clone())),
                None => chosen_setup(vfs, &d).map(|k| { let sd = vfs.read(&k); (k, sd) }),
            };
            if let Some((k, sd)) = from {
                if let Some(t) = setup_tyres(vfs, &sd) {
                    p.notes.push(format!("tyres from {k}: {} / {}", t[0], t[1]));
                    tyre_paths = [Some(t[0].clone()), Some(t[1].clone())];
                }
            }
        }
        // EVOVIEW_COMPOUND=<part of a name>: one of the car's compatible
        // compounds instead (the stock MX-5 NA setup fits ECO; a time
        // attack lap is on SUPERCAR)
        if let Ok(want) = std::env::var("EVOVIEW_COMPOUND") {
            let want = want.to_ascii_lowercase();
            if let Some((_, t)) = compatible_tyres(vfs, &key).into_iter().find(|(_, t)| t[0].to_ascii_lowercase().contains(&want)) {
                p.notes.push(format!("tyres from EVOVIEW_COMPOUND: {} / {}", t[0], t[1]));
                tyre_paths = [Some(t[0].clone()), Some(t[1].clone())];
            }
        }
        // each axle's tyre model: TyresCompoundData f4 TyresModelData
        for (ax, tp) in tyre_paths.iter().enumerate() {
            let Some(k) = tp.as_ref().and_then(|t| vfs.find(t, None)) else { continue };
            let td = vfs.read(&k);
            let Some((a, b)) = sub(&td, 0, td.len(), 4) else { continue };
            let g = |f: u32| f32_of(&td, a, b, f);
            let m = TyreModel {
                dy0: g(1).unwrap_or(0.0),
                dx0: g(2).unwrap_or(0.0),
                ls_y: g(3).unwrap_or(0.0),
                ls_x: g(4).unwrap_or(0.0),
                fz0: g(5).unwrap_or(0.0),
                peak_deg: g(6).unwrap_or(0.0),
                sctm: crate::sctm::Sctm::from_compound(&td),
            };
            if m.dy0 > 0.3 && m.fz0 > 100.0 {
                p.tyres[ax] = Some(m);
            }
            p.thermal[ax] = crate::tyretherm::Data::read(&td, &mut |c| read_curve_path(vfs, c));
            if std::env::var_os("EVOVIEW_TYRE_LOG").is_some() {
                if let Some(d) = &p.thermal[ax] {
                    println!("TYRE RR axle {ax}: rr {:?} pressure gain {} ideal {} psi spring gain {} curve {:?} limit-angle gain {}", d.rr, d.rr_p_gain, d.ideal_p, d.spring_gain, d.spring_curve, d.limit_gain);
                }
            }
        }
        // driven-tyre radius: the rear .tyre (f11), else the front (f10)
        for tp in [&tyre_paths[1], &tyre_paths[0]] {
            if p.tyre_radius.is_some() {
                break;
            }
            if let Some(k) = tp.as_ref().and_then(|t| vfs.find(t, None)) {
                let td = vfs.read(&k);
                if let Some((a, b)) = sub(&td, 0, td.len(), 3) {
                    if let Some(r) = f32_of(&td, a, b, 2).filter(|r| (0.15..0.6).contains(r)) {
                        p.tyre_radius = Some(r);
                    }
                }
            }
        }
        if std::env::var_os("EVOVIEW_NO_SETUP").is_none() {
            let stock = apply_stock_setup(vfs, &mut p).unwrap_or_default();
            if let Some((path, _, mut s)) = user {
                let before = setup_rows(&p);
                let n0 = p.notes.len();
                s.zero_fill(&stock);
                if s.version.map_or(true, |v| v > 1) {
                    p.notes.push(format!("setup {path}: version {:?} (this build writes 1) - applied anyway", s.version));
                }
                if s.susp.len() != 4 || s.align.len() != 4 {
                    p.notes.push(format!("setup {path}: {} suspensions / {} alignments, not 4", s.susp.len(), s.align.len()));
                }
                apply_setup(vfs, &mut p, &s);
                let after = setup_rows(&p);
                if first_time(format!("{key} {path}")) {
                    println!("setup: {path} (saved for {}, version {:?}) over the stock {}", if s.car().is_empty() { "?" } else { s.car() }, s.version, p.setup_source);
                    print_setup_change(&before, &after);
                    for n in p.notes[n0..].iter().filter(|n| n.starts_with("setup ")) {
                        println!("setup: ⚠ {n}");
                    }
                }
                p.user_setup = path;
            }
        }
        // the fuel the setup carries (EVOVIEW_FUEL_MASS=0: the dry mass, as
        // before). GUESS that totalMass excludes it, as AC1's TOTALMASS does
        if std::env::var("EVOVIEW_FUEL_MASS").map_or(true, |v| v != "0") {
            p.mass = p.mass_dry + p.fuel_mass();
        }
        p.limiter_raw = p.limiter_rpm;
        p.sanitise();
        Some(p)
    }

    /// Replace missing or absurd values with something drivable, and say so.
    fn sanitise(&mut self) {
        let mut note = |s: String| self.notes.push(s);
        if !(1.5..=4.5).contains(&self.wheelbase) {
            note(format!("wheelbase {:.3} out of range", self.wheelbase));
        }
        for (i, ax) in ["front", "rear"].iter().enumerate() {
            if !(1.0..=2.3).contains(&self.track[i]) {
                note(format!("{ax} track {:.3} out of range", self.track[i]));
            }
            let c = &mut self.coil[i];
            if !(5000.0..=1.5e6).contains(&c.spring) {
                note(format!("{ax} spring {:.0} N/m out of range", c.spring));
                c.spring = c.spring.clamp(5000.0, 1.5e6);
            }
            if !(0.005..=0.40).contains(&c.bump) {
                note(format!("{ax} bump travel {:.3} m out of range", c.bump));
                c.bump = c.bump.clamp(0.02, 0.15);
            }
            if !(0.003..=c.bump).contains(&c.bump_free) {
                c.bump_free = c.bump * 0.75;
            }
            if !(0.005..=0.40).contains(&c.droop) {
                note(format!("{ax} droop travel {:.3} m out of range", c.droop));
                c.droop = c.droop.clamp(0.02, 0.15);
            }
            if self.geo[i].kind == SuspKind::None {
                note(format!("{ax} suspension geometry missing"));
            }
        }
        if !(90.0..=1080.0).contains(&self.steer_lock_deg) {
            note(format!("steer lock {:.0} deg out of range", self.steer_lock_deg));
            self.steer_lock_deg = 360.0;
        }
        if !(4.0..=30.0).contains(&self.steer_ratio.abs()) {
            note(format!("steer ratio {:.2} out of range", self.steer_ratio));
            self.steer_ratio = 14.0;
        }
        if self.limiter_rpm > 0.0 && self.limiter_rpm < 100.0 {
            note(format!("rev limiter {:.0} rpm: looks like thousands of rpm (a unit slip)", self.limiter_rpm));
        }
        if !(1000.0..=22000.0).contains(&self.limiter_rpm) {
            note(format!("rev limiter {:.0} out of range", self.limiter_rpm));
            self.limiter_rpm = 7000.0;
        }
        if !(300.0..=3000.0).contains(&self.idle_rpm) {
            self.idle_rpm = 900.0;
        }
        if self.gears.is_empty() {
            note("no forward gears".into());
            self.gears = vec![3.2, 2.1, 1.5, 1.2, 1.0, 0.85];
        }
        if !(0.8..=12.0).contains(&self.final_drive) {
            note(format!("final drive {:.2} out of range", self.final_drive));
            self.final_drive = 4.0;
        }
        if self.reverse <= 0.0 {
            self.reverse = self.gears[0];
        }
        if !(0.2..=0.8).contains(&self.cg_front) {
            self.cg_front = 0.5;
        }
        if !(300.0..=4000.0).contains(&self.mass) {
            self.mass = 1200.0;
        }
    }

    /// kg of fuel the setup carries
    pub fn fuel_mass(&self) -> f32 {
        self.fuel_l * self.kg_per_l
    }

    /// Road-wheel angle at full lock, degrees.
    pub fn wheel_lock_deg(&self) -> f32 {
        self.steer_lock_deg / self.steer_ratio.abs()
    }
}

/// A tyre's grip, from its .tyre file (TyresData.proto TyresModelData): the
/// game's own brush-model numbers, of which a peak-friction curve uses the
/// peak friction at a reference load, the load sensitivity and the peak
/// slip angle.
#[derive(Clone, Copy, Debug)]
pub struct TyreModel {
    /// f1 Dy0 / f2 Dx0: peak lateral / longitudinal friction at Fz0
    pub dy0: f32,
    pub dx0: f32,
    /// f3 lsExpY / f4 lsExpX: load sensitivity exponents (grip ~ Fz^ls)
    pub ls_y: f32,
    pub ls_x: f32,
    /// f5 Fz0: reference load per tyre (N)
    pub fz0: f32,
    /// f6 friction_limit_angle: slip angle of peak lateral force (deg)
    pub peak_deg: f32,
    /// the whole compound as the game's tyre model runs it (sctm.rs)
    pub sctm: Option<crate::sctm::Sctm>,
}

impl TyreModel {
    /// Peak lateral friction of one tyre at load `fz` (N).
    ///
    /// The game's own (ksTyreModel, verified in the exe): at load the
    /// compound stores K = D0 Fz0 / Fz0^ls, and its friction function is
    /// mu(Fz) = K Fz^ls / Fz - i.e. D0 (Fz / Fz0)^(ls - 1). On top the
    /// runtime applies thermal grip, 1 / (1 + |p - p_ideal| k), the wear
    /// curve and 1 / (1 + 0.2 clamp(x / 100)), all ~1 for a fresh tyre at
    /// temperature.
    pub fn mu_y(&self, fz: f32) -> f32 {
        let ls = if self.ls_y > 0.3 && self.ls_y <= 1.2 { self.ls_y } else { 0.85 };
        (self.dy0 * (fz.max(50.0) / self.fz0).powf(ls - 1.0)).clamp(0.4, 3.0)
    }

    pub fn mu_x(&self, fz: f32) -> f32 {
        // the game: Dx0 == 0 means 1.2 x Dy0
        let d = if self.dx0 > 0.3 { self.dx0 } else { self.dy0 * 1.2 };
        let ls = if self.ls_x > 0.3 && self.ls_x <= 1.2 { self.ls_x } else { 0.85 };
        (d * (fz.max(50.0) / self.fz0).powf(ls - 1.0)).clamp(0.4, 3.0)
    }

    /// Peak slip angle, radians (4..15 degrees).
    pub fn peak(&self) -> f32 {
        (if self.peak_deg > 1.0 { self.peak_deg } else { 8.0 }).clamp(4.0, 15.0).to_radians()
    }
}

// ---- pit stop tyre choices (pits.rs) ----

/// One .tyre as a pit stop fits it: the grip model and the thermal data,
/// read the way Physics::read reads the car's own.
pub fn read_tyre(vfs: &mut Vfs, path: &str) -> Option<(TyreModel, Option<crate::tyretherm::Data>)> {
    let k = vfs.find(path, None)?;
    let td = vfs.read(&k);
    let (a, b) = sub(&td, 0, td.len(), 4)?;
    let g = |f: u32| f32_of(&td, a, b, f);
    let m = TyreModel {
        dy0: g(1).unwrap_or(0.0),
        dx0: g(2).unwrap_or(0.0),
        ls_y: g(3).unwrap_or(0.0),
        ls_x: g(4).unwrap_or(0.0),
        fz0: g(5).unwrap_or(0.0),
        peak_deg: g(6).unwrap_or(0.0),
        sctm: crate::sctm::Sctm::from_compound(&td),
    };
    let th = crate::tyretherm::Data::read(&td, &mut |c| read_curve_path(vfs, c));
    (m.dy0 > 0.3 && m.fz0 > 100.0).then_some((m, th))
}

/// The tyres the car may fit: presets/*.compatibletyres f2 {1 compound
/// (PhysicsTyreCompound), 2 front .tyre, 3 rear .tyre}, in order (the
/// setups' alignment f7 indexes it: the M4 GT3's 0 slicks, 1 wets). Else
/// the CarData's own f10 / f11.
pub fn compatible_tyres(vfs: &mut Vfs, car_data: &str) -> Vec<(u32, [String; 2])> {
    let mut out = Vec::new();
    for k in vfs.list("presets", "compatibletyres") {
        let d = vfs.read(&k);
        fields(&d, 0, d.len(), |f, v| {
            let (2, Some((a, b))) = (f, v.range()) else { return };
            let (mut kind, mut t) = (0u32, [String::new(), String::new()]);
            fields(&d, a, b, |g, w| match (g, w) {
                (1, Val::Varint(x)) => kind = x as u32,
                (2, Val::Bytes(x, y)) => t[0] = text(&d, (x, y)),
                (3, Val::Bytes(x, y)) => t[1] = text(&d, (x, y)),
                _ => {}
            });
            if !t[0].is_empty() && !t[1].is_empty() {
                out.push((kind, t));
            }
        });
        if !out.is_empty() {
            return out;
        }
    }
    let d = vfs.read(car_data);
    let mut t = [String::new(), String::new()];
    fields(&d, 0, d.len(), |f, v| match (f, v.range()) {
        (10, Some(r)) if t[0].is_empty() => t[0] = text(&d, r),
        (11, Some(r)) if t[1].is_empty() => t[1] = text(&d, r),
        _ => {}
    });
    if !t[0].is_empty() && !t[1].is_empty() {
        out.push((21, t));
    }
    out
}
