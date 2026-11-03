//! A per-wheel rigid-body car (EVOVIEW_RIGCAR=1), stepped the way AC EVO
//! steps its physics (L:/evoview_work/q/SPEC_PHYS.md):
//!
//! - the game's 333 Hz world step (dt 0.003), here split in two sub-steps;
//! - body: 6-DOF rigid body, gravity -9.806;
//! - each corner: a hub sliding along the body's up axis (an unsprung
//!   mass), held by the game's coilover (spring, helper, collar, bump stops,
//!   two-way damper; evophys.rs) and the anti-roll bars, on a tyre that is a
//!   radial spring + damper (TyresData rate / progressive_rate / damping);
//! - tyre forces from SCTM (sctm.rs) on each wheel's own slip angle, slip
//!   ratio and load;
//! - wheel spin from drive, brakes (evophys Brakes) and Fx, implicit in the
//!   tyre's slip stiffness so it stays stable at low speed;
//! - drivetrain: engine (power / coast curves, inertia) -> gearbox -> the
//!   game's diff (lock torque by type) between the driven wheels;
//! - aero: the lap sim's wings + body maps (front / rear downforce, drag).
//!
//! Simplifications against the game, each one a known gap: the suspension
//! moves along the body's up axis with motion ratio 1 (EVOVIEW_RIG_MR), not
//! the ODE wishbone links; tyre in-plane forces act on the body at a roll
//! centre a fixed height above the contact patch; the clutch slips only to
//! launch; ODE's constraint solver is replaced by explicit forces - unless
//! EVOVIEW_RIG_ODE=1, which integrates the chassis and hubs as ODE bodies
//! on slider joints with the game's world settings (rigode.rs);
//! EVOVIEW_RIG_ODE=links builds the double wishbones the game's way, five
//! DBall rods a corner (rigwish.rs).

use crate::carphys::Assists;
use crate::evophys::{Brakes, Coilover, Diff};
use crate::sctm::{Input, Sctm};
use crate::trackdrive::{Controls, Ground, Hit};
use glam::{Mat3, Quat, Vec2, Vec3};

const G: f32 = 9.806;
/// the game's physics step
pub const DT: f32 = 0.003;

fn env_t(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
const SUB: usize = 2;

#[derive(Clone, Copy, Debug, Default)]
pub struct Corner {
    /// spring compression from free length, m, and its rate
    pub x: f32,
    pub xv: f32,
    /// wheel spin, rad/s, and the wheel's turned angle (for rendering)
    pub omega: f32,
    pub angle: f32,
    /// tyre load, slip angle (rad), slip ratio, forces (N), aligning torque
    pub fz: f32,
    pub alpha: f32,
    pub sr: f32,
    pub fx: f32,
    pub fy: f32,
    pub mz: f32,
    /// world wheel centre
    pub hub: Vec3,
    pub on_ground: bool,
    pub grip: f32,
    /// hub to the ground along its normal (contact), and that normal
    pub gap: f32,
    pub gn: Vec3,
    /// the relaxed slip state (tan of the slip angle, slip ratio) and the
    /// last solve's normalised slip, which shortens the relaxation length
    pub relax_ta: f32,
    pub relax_sr: f32,
    pub nd: f32,
    /// the collision surface under the patch
    pub surf: u16,
}

pub struct RigCar {
    // state
    pub cg: Vec3,
    pub q: Quat,
    pub v: Vec3,
    pub w: Vec3,
    pub c: [Corner; 4],
    pub steer: f32,
    pub thr: f32,
    pub brk: f32,
    /// handbrake 0..1 (rear wheels, BrakesData handBrakeTorque)
    pub hb: f32,
    pub gear: i32,
    pub rpm: f32,
    /// engine torque this step, N m
    pub torque: f32,
    pub limiter_hit: bool,
    /// front tyres' aligning torque, N m (force feedback)
    pub mz_front: f32,
    /// the rack torque summed over the substeps since the last mz_mean()
    mz_sum: f32,
    mz_n: u32,
    // engine, clutch and box (SPEC_PHYS: clutch fn 0x1420e9cc0 / 0x1420e4570)
    /// crank speed, rad/s; false = stalled
    pub engine_w: f32,
    pub running: bool,
    crank: f32,
    stalled_for: f32,
    /// clutch engagement 0..1 and whether it is locked
    pub clutch_e: f32,
    pub clutch_locked: bool,
    /// a change in progress: target gear, seconds left, upshift; the gear
    /// it left and the change's length (a DCT blends the two ratios)
    shift: Option<(i32, f32, bool)>,
    shift_from: i32,
    shift_len: f32,
    cut_left: f32,
    /// seconds into a downshift's throttle blip (AutoBlip, car+0x78e8)
    blip_t: Option<f32>,
    /// the clutch input (car+0x3e0, 1 engaged) after the auto-clutch
    pub clutch_in: f32,
    /// the box shifts itself regardless of the assists (tests); the
    /// player's auto-gear assist (car+0x7990, EVOVIEW_AUTO_SHIFT) and the
    /// autopilot's Controls::auto_box also do
    pub auto_box: bool,
    /// this car moves the track's surface water (rain::flow): the player's
    /// only - an AI car stepping it too would run the flow N times over
    pub water_flow: bool,
    /// whether this step's box ran automatic (the keyboard's brake-to-reverse)
    auto_now: bool,
    /// first paddle / H input switches an automatic box to manual
    pub manual: bool,
    /// the last H gate seen (engage on change only)
    h_prev: Option<i32>,
    /// a gear that would not go in (no clutch, revs far off)
    pub missed: bool,
    gb: crate::carphys::Gearbox,
    cl: crate::carphys::Clutch,
    /// the game's AutoClutch (car+0x7558, FUN_1423fc890): its clutch
    /// value (1 engaged), the shift profile playing (up, seconds in), and
    /// the player's EVO assists (auto clutch on changes, on start)
    ac_value: f32,
    ac_prof: Option<(bool, f32)>,
    pedal_mapped: bool,
    pub assists: Assists,
    acc_prev: Vec3,
    t: f32,
    next_log: f32,
    // constants
    mass: f32,
    inertia: Vec3,
    /// body-frame (x left, y up, z forward) hub anchor at design ride, and
    /// the CG, relative to the mesh origin
    anchor: [Vec3; 4],
    cg_local: Vec3,
    x_static: [f32; 4],
    coil: [Coilover; 2],
    arb: [f32; 2],
    /// coilover motion ratio per axle (spring compression per m of wheel
    /// travel, carphys Geometry::motion_ratio); the corner state is wheel travel
    mr: [f32; 2],
    tyre: [Option<Sctm>; 2],
    /// tyre temperatures (tyretherm.rs): each axle's data, each wheel's
    /// state; the air and road temperatures; whether the tyre model reads
    /// the temperature (EVOVIEW_TYRE_T_MODEL=1 - see tyretherm / SPEC_SCTM:
    /// at the decoded falloff a slick under ~114 C would keep 25 % sliding)
    therm_data: [Option<crate::tyretherm::Data>; 2],
    pub therm: [Option<crate::tyretherm::Tyre>; 4],
    air_t: f32,
    road_t: f32,
    therm_model: bool,
    /// EVOVIEW_NO_RR=1: no rolling resistance on the wheels
    no_rr: bool,
    /// the tyres' relaxation lengths in use (EVOVIEW_NO_RELAX=1: instant slip)
    relax: bool,
    /// when the last hard wall hit / TC cut was printed
    last_wall_log: f32,
    /// EVOVIEW_TYRE_CONTACT=old: the tyre stands on the centre triangle's
    /// plane (kerb ridges launched the car; see tyre_envelope)
    old_contact: bool,
    /// EVOVIEW_COLLIDE_LOG: contact events (collide.rs)
    pub ev: Option<Box<crate::collide::Events>>,
    last_tc_log: f32,
    /// the FFB flat-spot thump's phase (getFFB car+0x920c) and gain
    /// (car+0x3b0 = PhysicsEngine, +0x418 = 0.05 from its ctor 0x142294130;
    /// EVOVIEW_FLATSPOT_FFB overrides)
    vib_phase: f32,
    vib_gain: f32,
    m_unsprung: [f32; 2],
    /// static camber per axle, rad (BasicSuspensionData; no camber gain
    /// with travel yet)
    camber: [f32; 2],
    /// body height (CarData body box), for the roof / sides on the ground
    body_h: f32,
    /// the body's length (CarData body_box z), for the roll-over contacts
    body_len: f32,
    /// the car's ABS (None: none, or EVOVIEW_NO_ABS)
    pub abs: Option<crate::evophys::Abs>,
    pub tc: Option<crate::evophys::Tc>,
    /// the driver's switches (keys A / T while driving)
    pub abs_on: bool,
    pub tc_on: bool,
    tc_mult: f32,
    /// the FFB's road effects (kerb, bump, slip, ABS), per physics step,
    /// and the velocity a step ago for the bump channel's vertical g
    pub ffb_fx: crate::ffb::Effects,
    fx_v: Vec3,
    ffb_fz: bool,
    /// camber change per metre of bump, rad (Geometry::camber_gain)
    camber_gain: [f32; 2],
    /// toe per axle, rad (setup TyreAlignment f3; + = toe-in)
    toe: [f32; 2],
    /// toe change per metre of bump (Geometry::toe_gain)
    toe_gain: [f32; 2],
    /// front hubs' steering axis in body axes, relative to the wheel centre
    /// (point, direction): geometry's left-wheel kingpin, x mirrored per side
    kingpin: [Option<(Vec3, Vec3)>; 2],
    brakes: Brakes,
    diff: Diff,
    front_share: f32,
    gears: Vec<f32>,
    reverse: f32,
    final_drive: f32,
    idle: f32,
    limiter: f32,
    engine_inertia: f32,
    pub aero: [f32; 3],
    pub max_steer: f32,
    /// steering-wheel degrees per road-wheel degree (setup steer_ratio)
    pub steer_ratio: f32,
    rc_height: f32,
    phys: crate::carphys::Physics,
    /// EVOVIEW_RIG_ODE=1: the bodies integrated by ODE (rigode.rs)
    ode: Option<Box<crate::rigode::OdeRig>>,
    /// EVOVIEW_RIG_ODE=links: the wishbones as ODE rods (rigwish.rs)
    links: Option<Box<crate::rigwish::Links>>,
}

fn rate_curve(c: &[(f32, f32)], x: f32) -> f32 {
    if c.is_empty() {
        return 0.0;
    }
    if x <= c[0].0 {
        return c[0].1;
    }
    for w in c.windows(2) {
        if x <= w[1].0 {
            let t = (x - w[0].0) / (w[1].0 - w[0].0).max(1e-6);
            return w[0].1 + (w[1].1 - w[0].1) * t;
        }
    }
    c[c.len() - 1].1
}

impl RigCar {
    /// `centres`: the four wheel centres in mesh space (0 FR, 1 FL, 2 RR,
    /// 3 RL; +X left, +Z forward); `cg_h`: CG height above the ground.
    /// The CG's height over the car's ground origin at rest.
    pub fn cg_height(&self) -> f32 {
        self.cg_local.y
    }

    pub fn new(p: &crate::carphys::Physics, centres: [Vec3; 4], cg_h: f32, max_steer: f32, front_share: f32, aero: [f32; 3]) -> RigCar {
        let mass = p.mass.max(400.0);
        let zf = (centres[0].z + centres[1].z) * 0.5;
        let zr = (centres[2].z + centres[3].z) * 0.5;
        let wb = (zf - zr).abs().max(1.0);
        let cgf = p.cg_front.clamp(0.3, 0.7);
        // the CG height: the suspension's base_y is the wheel centre against
        // the CG (Kunos' BASEY), so h = tyre radius - base_y (M4 0.43 m,
        // F2004 0.27); the caller's guess only without the data
        let radius = [0, 1].map(|ax| p.tyres[ax].and_then(|t| t.sctm).map_or(0.33, |t| t.radius));
        let cg_h = if p.base_y.iter().any(|b| *b != 0.0) {
            (0.5 * ((radius[0] - p.base_y[0]) + (radius[1] - p.base_y[1]))).clamp(0.15, 0.8)
        } else {
            cg_h
        };
        let cg_local = Vec3::new(0.0, cg_h, zr + wb * cgf);
        let bx = if p.body_box.min_element() > 0.1 { p.body_box } else { Vec3::new(1.9, 1.2, 4.5) };
        // a solid box is too lazy in yaw for a car with its mass between
        // the axles; 0.8 of it
        let inertia = Vec3::new(
            mass / 12.0 * (bx.y * bx.y + bx.z * bx.z) * 0.8,
            mass / 12.0 * (bx.x * bx.x + bx.z * bx.z) * 0.8,
            mass / 12.0 * (bx.x * bx.x + bx.y * bx.y) * 0.8,
        );
        let mr_env: Option<f32> = std::env::var("EVOVIEW_RIG_MR").ok().and_then(|v| v.parse().ok());
        let mr = [0, 1].map(|ax| mr_env.unwrap_or_else(|| p.geo[ax].motion_ratio()));
        let tyre = [p.tyres[0].and_then(|t| t.sctm), p.tyres[1].and_then(|t| t.sctm)];
        // tyre + the geometry's hubMass (25 kg when the data has none)
        let m_unsprung = [0, 1].map(|ax| tyre[ax].map_or(12.0, |t| t.mass) + if p.geo[ax].hub_mass > 0.0 { p.geo[ax].hub_mass } else { 25.0 });
        let camber = [0, 1].map(|ax| p.align[ax].map_or(p.geo[ax].static_camber, |a| a.0).to_radians());
        let toe = [0, 1].map(|ax| p.align[ax].map_or(0.0, |a| a.1).to_radians());
        let coil = p.coilover;
        // the data's mass is the whole car: the body carries it less the hubs
        let mass = mass - 2.0 * (m_unsprung[0] + m_unsprung[1]);
        // static compression: spring force = the corner's sprung weight
        let mut x_static = [0.0; 4];
        for k in 0..4 {
            let ax = k / 2;
            let share = if ax == 0 { cgf } else { 1.0 - cgf };
            let load = mass * share * 0.5 * G;
            let (mut lo, mut hi) = (-0.3f32, 0.5f32);
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if coil[ax].force(mid * mr[ax], 0.0).0 * mr[ax] < load {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            x_static[k] = 0.5 * (lo + hi);
        }
        if std::env::var_os("EVOVIEW_RIG_DBG").is_some() {
            eprintln!("RIGGEO front low {:?} up {:?} strut {:?} kingpin {:?} | rear low {:?}", p.geo[0].low, p.geo[0].up, p.geo[0].strut_top, p.geo[0].kingpin(), p.geo[1].low);
            eprintln!("RIGINFO toe_gain {:?} deg/10cm camber_gain {:?} deg/10cm base_y {:?} radius {:?} camber {camber:?} unsprung {m_unsprung:?} mass {mass:.0} cgf {cgf:.3} cg_h {cg_h:.2} inertia {inertia:?} arb {:?} mr {mr:?} geo {:?} x_static {x_static:?} anchors {centres:?}", [p.geo[0].toe_gain().to_degrees() * 0.1, p.geo[1].toe_gain().to_degrees() * 0.1], [p.geo[0].camber_gain().to_degrees() * 0.1, p.geo[1].camber_gain().to_degrees() * 0.1], p.base_y, [tyre[0].map(|t| t.radius), tyre[1].map(|t| t.radius)], [p.coil[0].arb, p.coil[1].arb], [(p.geo[0].kind, p.geo[0].coil.is_some(), p.geo[0].low.is_some()), (p.geo[1].kind, p.geo[1].coil.is_some(), p.geo[1].low.is_some())]);
        }
        let gears = if p.gears.is_empty() { vec![3.2, 2.1, 1.5, 1.2, 1.0, 0.85] } else { p.gears.clone() };
        RigCar {
            cg: Vec3::ZERO,
            q: Quat::IDENTITY,
            v: Vec3::ZERO,
            w: Vec3::ZERO,
            c: [Corner { grip: 1.0, ..Default::default() }; 4],
            steer: 0.0,
            thr: 0.0,
            brk: 0.0,
            hb: 0.0,
            gear: 1,
            rpm: p.idle_rpm.max(500.0),
            torque: 0.0,
            limiter_hit: false,
            mz_front: 0.0,
            mz_sum: 0.0,
            mz_n: 0,
            engine_w: p.idle_rpm.max(500.0) * std::f32::consts::TAU / 60.0,
            running: true,
            crank: 0.0,
            stalled_for: 0.0,
            clutch_e: 1.0,
            clutch_locked: false,
            shift: None,
            shift_from: 1,
            shift_len: 0.0,
            cut_left: 0.0,
            blip_t: None,
            clutch_in: 1.0,
            auto_box: false,
            water_flow: true,
            auto_now: false,
            manual: std::env::var_os("EVOVIEW_MANUAL").is_some(),
            h_prev: None,
            missed: false,
            gb: p.gearbox.clone(),
            cl: p.clutch.clone(),
            ac_value: 0.0,
            ac_prof: None,
            pedal_mapped: false,
            assists: Assists::read(),
            acc_prev: Vec3::ZERO,
            t: 0.0,
            next_log: 0.0,
            mass,
            inertia,
            anchor: centres,
            cg_local,
            x_static,
            coil,
            arb: [p.coil[0].arb, p.coil[1].arb],
            mr,
            tyre,
            therm_data: p.thermal.clone(),
            // EVOVIEW_NO_TYRE_T=1: no thermal model (grip as at temperature)
            therm: [0, 1, 2, 3].map(|k| p.thermal[k / 2].as_ref().filter(|_| std::env::var_os("EVOVIEW_NO_TYRE_T").is_none()).map(|d| {
                // the game starts them at the car's preset (car data +0x3a8)
                // or 25 C; here the tyre's blanket temperature, else the air
                let air = env_t("EVOVIEW_AIR_T", 25.0);
                let t0 = std::env::var("EVOVIEW_TYRE_T0").ok().and_then(|v| v.parse().ok())
                    .unwrap_or(if d.blanket > 0.0 { d.blanket } else { air });
                let mut t = crate::tyretherm::Tyre::new(d, t0, air);
                // the setup's pressure, else the tyre's ideal
                // (EVOVIEW_SETUP_PRESSURE=0: the ideal, as before)
                if let Some(psi) = p.pressure[k / 2].filter(|_| std::env::var("EVOVIEW_SETUP_PRESSURE").map_or(true, |v| v != "0")) {
                    t.set_pressure(psi);
                }
                // EVOVIEW_TYRE_BLOWN=k: start with wheel k (0 FR 1 FL 2 RR 3 RL) blown
                t.blown = std::env::var("EVOVIEW_TYRE_BLOWN").ok().and_then(|v| v.parse::<usize>().ok()) == Some(k);
                t
            })),
            air_t: env_t("EVOVIEW_AIR_T", 25.0),
            road_t: env_t("EVOVIEW_ROAD_T", f32::NAN),
            therm_model: std::env::var_os("EVOVIEW_TYRE_T_MODEL").is_some(),
            no_rr: std::env::var_os("EVOVIEW_NO_RR").is_some(),
            relax: std::env::var_os("EVOVIEW_NO_RELAX").is_none(),
            last_wall_log: -1.0,
            ev: crate::collide::Events::from_env(),
            old_contact: std::env::var("EVOVIEW_TYRE_CONTACT").as_deref() == Ok("old"),
            last_tc_log: -1.0,
            vib_phase: 0.0,
            vib_gain: env_t("EVOVIEW_FLATSPOT_FFB", 0.05),
            m_unsprung,
            camber,
            toe,
            toe_gain: [0, 1].map(|ax| if std::env::var_os("EVOVIEW_RIG_NOBUMPSTEER").is_some() { 0.0 } else { p.geo[ax].toe_gain().clamp(-0.5, 0.5) }),
            body_h: bx.y.clamp(0.9, 1.8),
            body_len: bx.z.clamp(3.0, 6.0),
            abs: p.abs.clone().filter(|_| std::env::var_os("EVOVIEW_NO_ABS").is_none()),
            tc: p.tc.clone().filter(|_| std::env::var_os("EVOVIEW_NO_TC").is_none()),
            abs_on: true,
            ffb_fx: Default::default(),
            fx_v: Vec3::ZERO,
            tc_on: true,
            tc_mult: 1.0,
            ffb_fz: std::env::var_os("EVOVIEW_FFB_FZ").is_some(),
            camber_gain: [0, 1].map(|ax| if std::env::var_os("EVOVIEW_RIG_NOCAMBER").is_some() { 0.0 } else { p.geo[ax].camber_gain() }),
            kingpin: {
                // geometry x is inboard: the left wheel's inboard is -x here
                let kp = p.geo[0].kingpin();
                [0usize, 1].map(|k| kp.map(|(b, d)| {
                    let m = if k == 1 { -1.0 } else { 1.0 };
                    (Vec3::new(b.x * m, b.y, b.z), Vec3::new(d.x * m, d.y, d.z))
                }))
            },
            brakes: p.brakes,
            diff: p.diff,
            front_share,
            gears,
            reverse: if p.reverse != 0.0 { p.reverse.abs() } else { 3.2 },
            final_drive: if p.final_drive > 0.0 { p.final_drive } else { 3.9 },
            idle: p.idle_rpm.max(500.0),
            limiter: p.limiter_rpm.max(3000.0),
            engine_inertia: if p.engine_inertia > 0.0 { p.engine_inertia } else { 0.15 },
            aero,
            max_steer,
            steer_ratio: if p.steer_ratio.abs() > 1.0 { p.steer_ratio.abs() } else { 15.0 },
            rc_height: std::env::var("EVOVIEW_RIG_RC").ok().and_then(|v| v.parse().ok()).unwrap_or(0.06),
            phys: p.clone(),
            ode: crate::rigode::OdeRig::enabled().then(|| Box::new(crate::rigode::OdeRig::new(mass, inertia, m_unsprung, &centres, cg_local))),
            links: crate::rigwish::Links::enabled()
                .then(|| {
                    let l = crate::rigwish::Links::new(mass, inertia, m_unsprung, [&p.geo[0], &p.geo[1]], &centres, cg_local, x_static, mr);
                    if l.is_none() {
                        println!("RIG links: not double wishbones front and rear ({:?} / {:?}): the explicit integrator", p.geo[0].kind, p.geo[1].kind);
                    }
                    l.map(Box::new)
                })
                .flatten(),
        }
    }

    fn rot(&self) -> Mat3 {
        Mat3::from_quat(self.q)
    }

    /// mesh origin in the world (what the renderer places)
    pub fn origin(&self) -> Vec3 {
        self.cg - self.rot() * self.cg_local
    }

    pub fn forward(&self) -> Vec3 {
        self.rot() * Vec3::Z
    }

    pub fn up(&self) -> Vec3 {
        self.rot() * Vec3::Y
    }

    pub fn left(&self) -> Vec3 {
        self.rot() * Vec3::X
    }

    /// Put the mesh origin at `origin` (on the ground) facing `yaw`, at rest
    /// on its static ride.
    pub fn place(&mut self, origin: Vec3, yaw: f32) {
        self.q = Quat::from_rotation_y(yaw);
        self.cg = origin + self.rot() * self.cg_local;
        self.v = Vec3::ZERO;
        self.w = Vec3::ZERO;
        self.acc_prev = Vec3::ZERO;
        self.fx_v = Vec3::ZERO;
        for k in 0..4 {
            self.c[k].x = self.x_static[k];
            self.c[k].xv = 0.0;
            self.c[k].omega = 0.0;
            self.c[k].relax_ta = 0.0;
            self.c[k].relax_sr = 0.0;
            self.c[k].nd = 0.0;
        }
        self.gear = 1;
        // a reset also ends any change in progress, engine back at idle
        self.shift = None;
        self.cut_left = 0.0;
        self.blip_t = None;
        // the auto-clutch starts open, as on the grid
        self.ac_value = 0.0;
        self.ac_prof = None;
        self.h_prev = None;
        self.missed = false;
        self.running = true;
        self.crank = 0.0;
        self.engine_w = self.idle * std::f32::consts::TAU / 60.0;
    }

    /// After `place`: rolling straight ahead at `v` m/s, wheels turning with
    /// the road, in the highest gear that keeps the engine over 40 % of the
    /// limiter, clutch in (a rolling start).
    pub fn roll(&mut self, v: f32) {
        self.v = self.rot() * Vec3::Z * v;
        for k in 0..4 {
            self.c[k].omega = v / self.radius(k).max(0.1);
        }
        let road_w = v / self.radius(2).max(0.1);
        let mut g = 1;
        for (i, r) in self.gears.iter().enumerate() {
            if road_w * r * self.final_drive * 60.0 / std::f32::consts::TAU > self.limiter * 0.4 {
                g = i as i32 + 1;
            }
        }
        self.gear = g;
        self.engine_w = (road_w * self.ratio()).max(self.idle * std::f32::consts::TAU / 60.0);
        self.ac_value = 1.0;
    }

    fn ratio(&self) -> f32 {
        match self.gear {
            0 => 0.0,
            g if g < 0 => -self.reverse * self.final_drive,
            g => self.gears[(g as usize - 1).min(self.gears.len() - 1)] * self.final_drive,
        }
    }

    /// The rack torque averaged over every substep since the last call (the
    /// game feeds FFB each 3 ms step; a frame-rate reader of the last
    /// substep alone got a noisy sample of it).
    pub fn mz_mean(&mut self) -> f32 {
        let m = if self.mz_n > 0 { self.mz_sum / self.mz_n as f32 } else { self.mz_front };
        self.mz_sum = 0.0;
        self.mz_n = 0;
        m
    }

    /// A blown wheel k's vertical squash for drawing: the hub rides at the
    /// rim radius, so the wheel mesh (rim and tyre in one) is flattened
    /// about the hub until its bottom meets the road. None when intact.
    pub fn blown_squash(&self, k: usize) -> Option<f32> {
        let d = self.therm_data[k / 2].as_ref()?;
        let th = self.therm[k].as_ref()?;
        (th.blown && d.radius > 0.0).then(|| (d.rim_radius / d.radius).clamp(0.4, 1.0))
    }

    /// The tyres' pressures, psi, in the dash's order FL FR RL RR (ours is
    /// FR FL RR RL); None without the thermal model.
    pub fn tyre_psi(&self) -> Option<[f32; 4]> {
        let p = |k: usize| Some(self.therm[k].as_ref()?.pressure(self.therm_data[k / 2].as_ref()?));
        Some([p(1)?, p(0)?, p(3)?, p(2)?])
    }

    /// TC's throttle multiplier this step (1 = no cut).
    pub fn tc_cut(&self) -> f32 {
        self.tc_mult
    }

    /// Each wheel's share of the drive (FR FL RR RL).
    pub fn drive_share(&self) -> [f32; 4] {
        self.driven()
    }

    /// Front share of the brake torque, 0..1 (the setup's front_bias).
    pub fn brake_bias(&self) -> f32 {
        self.brakes.front_bias
    }

    /// Live: the next physics step brakes with it (aids.rs).
    pub fn set_brake_bias(&mut self, f: f32) {
        self.brakes.front_bias = f.clamp(0.0, 1.0);
    }

    /// The car's setup limits (.carsetuplimits), for the cockpit aids.
    pub fn limits(&self) -> &crate::aids::Limits {
        &self.phys.limits
    }

    /// The CarData path the car was read from.
    pub fn source(&self) -> &str {
        &self.phys.source
    }

    pub fn speed(&self) -> f32 {
        self.v.dot(self.forward())
    }

    /// Advance by `dt` (any frame time): whole 3 ms steps, each in SUB parts.
    pub fn step(&mut self, ctl: &Controls, dt: f32, ground: &Ground, acc: &mut f32) {
        // the track's water runs downhill around the car (rain::flow)
        let top = self.cg.y + 3.0;
        if self.water_flow {
            crate::rain::flow(self.cg.x, self.cg.z, dt.min(0.1), |x, z| ground.probe(x, z, top).map(|h| h.y));
        }
        *acc += dt.min(0.1);
        while *acc >= DT {
            *acc -= DT;
            self.controls(ctl, DT);
            self.t += DT;
            if std::env::var_os("EVOVIEW_RIG_LOG").is_some() && self.t >= self.next_log {
                self.next_log = self.t + std::env::var("EVOVIEW_RIG_LOG").ok().and_then(|v| v.parse().ok()).unwrap_or(0.1f32);
                let c = &self.c;
                println!("RIG t {:6.2} v {:5.1} thr {:.2} brk {:.2} st {:+.3} g {} rpm {:.0} run {} ce {:.2}{} sh {:?} r {:+.2} | {}", self.t, self.speed() * 3.6, self.thr, self.brk, self.steer, self.gear, self.engine_w * 60.0 / std::f32::consts::TAU, self.running, self.clutch_e, if self.clutch_locked { "L" } else { "" }, self.shift.map(|s| s.0),
                    self.w.dot(self.up()),
                    (0..4).map(|k| format!("[fz {:5.0} a {:+5.1} sr {:+.2} fx {:+5.0} fy {:+5.0} x {:+.3} g {:.2}]", c[k].fz, c[k].alpha.to_degrees(), c[k].sr, c[k].fx, c[k].fy, c[k].x, c[k].grip)).collect::<Vec<_>>().join(" "));
            }
            // EVOVIEW_TYRE_LOG=1: the tyres' temperatures once a second
            if std::env::var_os("EVOVIEW_TYRE_LOG").is_some() && (self.t / 1.0).floor() != ((self.t - DT) / 1.0).floor() {
                let d = &self.therm_data;
                println!("TYRES t {:5.0} v {:5.1} | {}", self.t, self.speed() * 3.6, (0..4).filter_map(|k| {
                    let (dd, th) = (d[k / 2].as_ref()?, self.therm[k].as_ref()?);
                    Some(format!("[surf {:5.1} core {:5.1} prac {:5.1} p {:4.1} flat {:.3} tread {:.3} grain {:.2} grip {:.3}]", th.mean(), th.core, th.practical(dd, 0.0), th.pressure(dd), th.flat, th.tread_at(dd, 0.0), th.grain, th.grip(dd, 0.0)))
                }).collect::<Vec<_>>().join(" "));
            }
            // ABS at its own rate on the wheels' slip (game order FL FR RL
            // RR; ours FR FL RR RL)
            if let Some(abs) = self.abs.as_mut().filter(|_| self.abs_on) {
                let g = [1usize, 0, 3, 2];
                let v = self.v.length();
                abs.step(DT, v, self.brk, g.map(|k| self.c[k].sr), g.map(|k| self.c[k].alpha));
            }
            // TC on the driven wheels' worst spin and their slip angle
            let sh = self.driven();
            if let Some(tc) = self.tc.as_mut().filter(|_| self.tc_on) {
                let (mut sr, mut al) = (0.0f32, 0.0f32);
                for k in 0..4 {
                    if sh[k] > 0.0 {
                        sr = sr.max(self.c[k].sr);
                        al = al.max(self.c[k].alpha.abs());
                    }
                }
                let v = self.v.length();
                self.tc_mult = tc.step(DT, v, sr, al, self.thr);
                // say when TC takes power (it cuts harder as the rear's
                // slip angle grows: power oversteer is where it bites)
                if self.tc_mult < 0.9 && self.thr > 0.1 && self.t - self.last_tc_log > 1.0 {
                    self.last_tc_log = self.t;
                    println!("TC cutting to {:.0}% (slip ratio {:.2}, slip angle {:.1} deg) - T turns it off", self.tc_mult * 100.0, sr, al.to_degrees());
                }
            } else {
                self.tc_mult = 1.0;
            }
            let keep = (self.cg, self.q, self.c);
            for _ in 0..SUB {
                self.substep(ground, DT / SUB as f32);
            }
            if let Some(mut ev) = self.ev.take() {
                ev.step(self, ground);
                self.ev = Some(ev);
            }
            // the FFB's road effects at the game's rate (fn 0x14222d4f0):
            // the strongest kerb and the most slip under a loaded tyre
            let up_g = ((self.v - self.fx_v) / DT / 9.806).dot(self.up());
            self.fx_v = self.v;
            let mut fi = crate::ffb::FxIn {
                speed: self.v.length(),
                up_g,
                abs: self.abs_on && self.abs.as_ref().map_or(false, |a| a.active),
                ..Default::default()
            };
            for c in &self.c {
                if c.fz > 0.0 {
                    let (g, l) = ground.vibration(c.surf);
                    if fi.kerb.0 < g {
                        fi.kerb = (g, l);
                    }
                    fi.nd = fi.nd.max(c.nd);
                }
            }
            self.ffb_fx.step(&fi, DT);
            // never let one bad step poison the car: back to the last good
            // state, at rest
            if !(self.cg.is_finite() && self.v.is_finite() && self.w.is_finite() && self.q.is_finite()) {
                (self.cg, self.q, self.c) = keep;
                self.v = Vec3::ZERO;
                self.w = Vec3::ZERO;
                for c in self.c.iter_mut() {
                    c.xv = 0.0;
                    c.omega = 0.0;
                }
            }
        }
    }

    fn controls(&mut self, c: &Controls, dt: f32) {
        let idle = Controls::default();
        let c = if std::env::var_os("EVOVIEW_RIG_IDLE").is_some() { &idle } else { c };
        let approach = |v: &mut f32, t: f32, rate: f32| *v += (t - *v).clamp(-rate * dt, rate * dt);
        self.hb = c.handbrake_v.unwrap_or(if c.handbrake { 1.0 } else { 0.0 });
        if c.analog {
            // pedals are pedals: no keyboard ramp
            self.thr = c.throttle;
            self.brk = c.brake;
        } else {
            approach(&mut self.thr, c.throttle, 8.0);
            approach(&mut self.brk, c.brake, 10.0);
        }
        if let Some(wd) = c.wheel_deg {
            // the wheel turns the road wheels through the car's own ratio
            self.steer = (wd / self.steer_ratio).to_radians().clamp(-self.max_steer, self.max_steer);
        } else if c.analog {
            self.steer = c.steer * self.max_steer;
        } else {
            let vx = self.speed().abs();
            let speed_k = (1.0 / (1.0 + (vx / 25.0).powi(2))).max(0.25);
            approach(&mut self.steer, c.steer * self.max_steer * speed_k, 2.5);
        }
        self.gearbox(c, dt);
    }

    fn gear_ratio(&self, g: i32) -> f32 {
        match g {
            0 => 0.0,
            g if g < 0 => -self.reverse * self.final_drive,
            g => self.gears[(g as usize - 1).min(self.gears.len() - 1)] * self.final_drive,
        }
    }

    /// Ask for gear `to` with the paddles / sequential lever (FUN_1422e1a10
    /// up, FUN_1422e1510 down): the box sits in neutral for the change's
    /// time (gear_up_time / gear_dn_time, 100 / 150 ms when 0) - a DCT
    /// blends the ratios instead; an upshift cuts the gas for
    /// auto_cut_off_time. Refuses a downshift that would over-rev (the
    /// DownshiftProtection: every non-H box unless its data turns it off,
    /// H boxes never), neutral from 1st above 2 km/h with lockN, and
    /// reverse while rolling forward.
    fn request(&mut self, to: i32) {
        let to = to.clamp(-1, self.gears.len() as i32);
        if to == self.gear || self.shift.is_some() {
            return;
        }
        let up = to > self.gear;
        let protect = !self.gb.shifter && (!self.gb.protect_msg || self.gb.protect);
        if !up && to > 0 && protect {
            let rpm = (self.driven_omega() * self.gear_ratio(to)).abs() * 60.0 / std::f32::consts::TAU;
            if rpm > self.limiter + self.gb.protect_overrev.max(0.0) {
                return;
            }
        }
        if to < 0 && self.speed() > 2.0 {
            return;
        }
        if to == 0 && self.gear == 1 && self.gb.lock_n && self.speed().abs() * 3.6 > 2.0 {
            return;
        }
        let t = if up {
            if self.gb.up_time > 0.0 { self.gb.up_time } else { 100.0 }
        } else if self.gb.dn_time > 0.0 {
            self.gb.dn_time
        } else {
            150.0
        } * 0.001;
        self.shift = Some((to, t, up));
        self.shift_from = self.gear;
        self.shift_len = t;
        self.gear_event(up);
        if up {
            self.cut_left = self.gb.cut_time.max(0.0) * 0.001;
        } else if self.clutch_in > 0.1 {
            // AutoBlip's OnGearRequestEvent: a downshift with the clutch up
            self.blip_t = Some(0.0);
        }
    }

    /// Whether the car clutches through gear changes by itself: an H-gate
    /// car follows the player's auto-clutch assist, any other the car's own
    /// useOnChanges (the assist only overrides when is_shifter_supported,
    /// car+0x1068 - str__Creating player car controller / assists apply).
    fn changes_auto(&self, pedal: bool) -> bool {
        if self.gb.shifter { self.assists.clutch_changes || (!pedal && kb_clutch()) } else { self.cl.auto_on_changes }
    }

    /// OnGearRequestEvent (FUN_1423fc650): with the auto-clutch working
    /// changes and the clutch engaged, play the direction's profile.
    fn gear_event(&mut self, up: bool) {
        let prof = if up { &self.cl.up_profile } else { &self.cl.down_profile };
        if self.changes_auto(self.pedal_mapped) && self.ac_value > 0.01 && !prof.is_empty() {
            self.ac_prof = Some((up, 0.0));
        }
    }

    fn gearbox(&mut self, c: &Controls, dt: f32) {
        if c.shift_up || c.shift_down || c.h_gear.map_or(false, |g| g != 0) {
            self.manual = true;
        }
        if c.starter && !self.running {
            self.crank = 1.0;
        }
        // a change in progress
        if let Some((to, t, up)) = self.shift {
            let t = t - dt;
            if t <= 0.0 {
                self.gear = to;
                self.shift = None;
            } else {
                self.shift = Some((to, t, up));
            }
        }
        self.cut_left = (self.cut_left - dt).max(0.0);
        // the blip profile's x is in ms (its length tracks gear_dn_time)
        if let Some(t) = self.blip_t {
            let end = self.gb.blip.last().map_or(0.0, |p| p.0) * 0.001;
            self.blip_t = (self.shift.is_some() && t + dt <= end).then_some(t + dt);
        }
        let vx = self.speed();
        // the gear is the player's unless the auto-gear assist is on (Josh's
        // is off); the autopilot shifts for itself, as the game's AI does
        let automatic = (self.auto_box || self.assists.gear || c.auto_box) && !self.manual;
        self.auto_now = automatic;
        // an H lever only on a car that has a gate (is_shifter_supported,
        // FUN_142405a40); any other takes the paddles
        match (automatic, c.h_gear.filter(|_| self.gb.shifter)) {
            // H-pattern (FUN_1422e5c90, every step): neutral always goes in;
            // a gear goes in unless clutch engagement x the rpm gap across
            // it x (1 + (gas - 1) x controls_window_gain) reaches
            // valid_shift_rpm_window - it then stays where it was (neutral,
            // the lever having passed through it) until the clutch or the
            // revs let it in. No gear event: no clutch profile, no blip.
            (false, Some(g)) => {
                let g = g.clamp(-1, self.gears.len() as i32);
                self.h_prev = Some(g);
                if self.shift.is_some() {
                    // a paddle change in progress finishes first
                } else if g == 0 {
                    self.gear = 0;
                    self.missed = false;
                } else if g != self.gear {
                    let w_in = self.driven_omega() * self.gear_ratio(g);
                    let gap = (self.engine_w - w_in).abs() * self.clutch_e;
                    let need = gap * (1.0 + (self.thr - 1.0) * self.gb.window_gain) * 60.0 / std::f32::consts::TAU;
                    let window = if self.gb.shift_window > 0.0 { self.gb.shift_window } else { 500.0 };
                    if need < window {
                        self.gear = g;
                        self.missed = false;
                    } else {
                        self.missed = true;
                    }
                } else {
                    self.missed = false;
                }
            }
            // paddles (sequential)
            (false, None) => {
                if c.shift_up {
                    self.request(self.gear + 1);
                }
                if c.shift_down {
                    self.request(self.gear - 1);
                }
            }
            // automatic (the auto-gear assist, EVOVIEW_AUTO_SHIFT, autopilot)
            (true, _) => {
                if vx.abs() < 0.5 && c.brake > 0.5 && c.throttle < 0.1 && self.gear > 0 {
                    self.gear = -1;
                }
                if self.gear < 0 && c.throttle > 0.1 && vx > -0.5 {
                    self.gear = 1;
                }
                if self.gear == 0 && self.shift.is_none() {
                    self.gear = 1;
                }
                if self.gear > 0 && self.shift.is_none() {
                    let wheel = self.driven_omega().abs() * 60.0 / std::f32::consts::TAU;
                    // up on the wheel side's revs (a free-revving engine with
                    // the clutch open must not count); down only once the
                    // current gear bogs - a spinning launch otherwise shifts
                    // 1-2-1 forever, the clutch open through every change
                    let road = (self.speed().abs() / self.tyre[1].map_or(0.33, |t| t.radius).max(0.1)) * 60.0 / std::f32::consts::TAU;
                    let cur = wheel.max(road) * self.gear_ratio(self.gear).abs();
                    if cur > self.limiter * 0.95 && (self.gear as usize) < self.gears.len() {
                        self.request(self.gear + 1);
                    } else if self.gear > 1 {
                        let down = self.gears[self.gear as usize - 2] * self.final_drive;
                        if road * down < self.limiter * 0.6 && cur < (self.idle * 1.3).max(self.limiter * 0.25) {
                            self.request(self.gear - 1);
                        } else if self.speed().abs() > 8.0 && !crate::trackdrive::downshift_old()
                            && cur * self.gear_ratio(self.gear - 1) / self.gear_ratio(self.gear).max(1e-3) < self.limiter * crate::trackdrive::downshift_at()
                            && cur < self.limiter * 0.7 {
                            // rolling: down once the lower gear is back under
                            // 80 % of the limiter (up at 95 %), as a driver.
                            // Measured on the same revs as the upshift (the
                            // tyre's data radius read low, so 1-2-1 hunted)
                            // would - not only when the engine bogs. Keyed
                            // on the road speed (no wheelspin), above 29 km/h
                            // (a spinning launch would shift 1-2-1); and not
                            // with the wheels spinning up the revs (a
                            // downshift then snaps the rear round)
                            self.request(self.gear - 1);
                        }
                    }
                }
            }
        }
        // the clutch: the driver's pedal, or the game's auto-clutch
        // (decoded: L:/evoview_work/q/SPEC_AUTOCLUTCH.md)
        let pedal = c.clutch.map(|p| 1.0 - p.clamp(0.0, 1.0));
        self.pedal_mapped = pedal.is_some();
        let rpm = self.engine_w * 60.0 / std::f32::consts::TAU;
        let kmh = self.speed().abs() * 3.6;
        let mut auto: Option<f32> = None;
        // a shift profile, only while rolling (FUN_1423fcc70)
        if let Some((up, t)) = self.ac_prof {
            let prof = if up { &self.cl.up_profile } else { &self.cl.down_profile };
            if kmh > 5.0 {
                let v = rate_curve(prof, t).clamp(0.0, 1.0);
                let end = prof.last().map_or(0.0, |p| p.0);
                self.ac_prof = (t + dt <= end).then_some((up, t + dt));
                self.ac_value = v;
                auto = Some(v);
            } else {
                self.ac_prof = None;
            }
        }
        // launches (FUN_1423fc890): on when the car forces it, the player's
        // auto-clutch-on-start assist is on, or nothing is mapped to a clutch
        // (evoview: also with no clutch mapped at all, as the game's player
        // controller starts out - +0x13c4 - unless EVOVIEW_KB_CLUTCH=0)
        let launch = self.cl.auto_forced || self.assists.clutch_start || (pedal.is_none() && kb_clutch());
        if auto.is_none() && launch {
            let (lo, hi) = (self.cl.auto_min_rpm, self.cl.auto_max_rpm.max(self.cl.auto_min_rpm + 1.0));
            let target = match self.gear {
                // reverse and 1st: engage across the rpm window
                -1 | 1 => ((rpm - lo) / (hi - lo)).clamp(0.0, 1.0),
                // neutral at a standstill: in only with some throttle
                0 if kmh < 5.0 => {
                    let t = if self.thr > 0.2 { 1.0 } else { 0.0 };
                    self.ac_value = t;
                    t
                }
                0 => 1.0,
                // every other gear: engaged, open below the window (no stall)
                _ => if rpm < lo { 0.0 } else { 1.0 },
            };
            // ~1 / s travel (the object's rate, 1.0 at init)
            let step = dt;
            self.ac_value = if (target - self.ac_value).abs() >= step {
                self.ac_value + step * (target - self.ac_value).signum()
            } else {
                target
            };
            // open while held on the brakes at a standstill - unless the
            // automatic box is in reverse, where the brake key is the gas
            // (it held the clutch open: no car could back out of a wall)
            if self.brk > 0.2 && kmh < 5.0 && !(self.gear < 0 && self.auto_now) {
                self.ac_value = 0.0;
            }
            self.ac_value = self.ac_value.clamp(0.0, 1.0);
            auto = Some(self.ac_value);
        } else if auto.is_none() {
            self.ac_value = pedal.unwrap_or(1.0);
        }
        // the auto-clutch writes car+0x3e0 over the pedal: on a forcedOn car
        // (every DCT / GT paddle car) the pedal does nothing at all
        let x = match (auto, pedal) {
            (Some(a), Some(p)) if pedal_dips() => a.min(p),
            (Some(a), _) => a,
            (None, Some(p)) => p,
            (None, None) => 1.0,
        };
        self.clutch_in = x;
        self.clutch_e = if self.cl.curve.is_empty() {
            let x = x.clamp(0.0, 1.0);
            x.powf((std::f32::consts::PI * x).cos() * 0.7 + 1.3)
        } else {
            (rate_curve(&self.cl.curve, x * 100.0) * 0.01).clamp(0.0, 1.0)
        };
    }

    fn driven(&self) -> [f32; 4] {
        let f = self.front_share;
        [f * 0.5, f * 0.5, (1.0 - f) * 0.5, (1.0 - f) * 0.5]
    }

    fn driven_omega(&self) -> f32 {
        // the shares sum to 1: the drive-weighted mean wheel speed
        let s = self.driven();
        (0..4).map(|k| self.c[k].omega * s[k]).sum::<f32>()
    }

    fn substep(&mut self, ground: &Ground, dt: f32) {
        let r = self.rot();
        let (fw, up) = (r * Vec3::Z, r * Vec3::Y);
        // the keyboard's automatic reverses on the brake key; a driver who
        // picked R (paddles, H gate) uses the pedals as they are
        let reversing = self.gear < 0 && self.auto_now;
        let (drive_in, brake_in) = if reversing { (self.brk, self.thr) } else { (self.thr, self.brk) };

        // ---- engine, clutch, box (the game's clutch: slip torque
        // -dw / (|dw| + 4) x max_torque x e; locked once e = 1 and the two
        // sides are within 10 %)
        let tau = std::f32::consts::TAU;
        // mid-change: neutral, or a DCT's blend of the two ratios
        // (FUN_1422e7c20, has_dual_clutch: (1 - f) old + f new)
        let ratio = match self.shift {
            Some((to, left, _)) if self.gb.dual_clutch && !dct_neutral() => {
                let f = (1.0 - left / self.shift_len.max(1e-4)).clamp(0.0, 1.0);
                (1.0 - f) * self.gear_ratio(self.shift_from) + f * self.gear_ratio(to)
            }
            Some(_) => 0.0,
            None => self.ratio(),
        };
        let w_in = self.driven_omega() * ratio;
        let rpm = self.engine_w * 60.0 / tau;
        // throttle: the driver's, cut for an upshift, blipped for a downshift,
        // plus the idle hold below idle. Below idle the torque is read at
        // idle: the 296 GT3's map is 0 at half idle, and an engine bogged
        // there sat frozen at full throttle with the clutch open
        let mut throttle = drive_in.clamp(0.0, 1.0) * self.tc_mult;
        // the upshift cut (gas_cut_off_level) above 5 km/h
        let kmh = self.speed().abs() * 3.6;
        if self.cut_left > 0.0 && kmh > 5.0 {
            throttle = throttle.min(self.gb.gas_cut.max(0.0));
        }
        // the downshift blip (FUN_1423fd2c0): gas = max(gas, profile(t))
        // while the change runs, above 5 km/h, into a gear; an electronic
        // box always, any other only with the auto-blip assist
        if let (Some((to, _, false)), Some(t)) = (self.shift, self.blip_t) {
            if to > 0 && kmh >= 5.0 && (self.gb.blip_electronic || self.assists.blip) && !self.gb.blip.is_empty() {
                throttle = throttle.max(rate_curve(&self.gb.blip, t * 1000.0).clamp(0.0, 1.0));
            }
        }
        if self.running && rpm < self.idle {
            // up to full well below idle: capped at 0.35, an engine whose
            // friction outweighs 35 % at idle (Mustang GT3) sat at the stall
            // threshold, 'running' at 240 rpm, forever
            // (clutch open only: a dumped clutch still stalls)
            let cap = if self.clutch_e < 0.3 { 1.0 } else { 0.35 };
            throttle = throttle.max(((self.idle - rpm) / (0.25 * self.idle)).clamp(0.0, cap));
        }
        self.limiter_hit = rpm >= self.limiter;
        let on = if self.running { self.phys.engine_torque(rpm.max(self.idle), throttle).unwrap_or(400.0 * throttle) } else { 0.0 };
        let coast = if self.phys.coast.is_empty() {
            -40.0 * rpm / self.limiter
        } else {
            -rate_curve(&self.phys.coast, rpm).abs()
        };
        let mut te = if self.limiter_hit { coast } else { on + coast * (1.0 - throttle) };
        if !self.running {
            te = coast * 1.5 + if self.crank > 0.0 { 120.0 } else { 0.0 };
        }
        self.torque = te;
        let ie_eng = self.engine_inertia + self.cl.inertia;
        let e = self.clutch_e;
        let locked_now = ratio != 0.0 && e >= 1.0 && (w_in - self.engine_w).abs() <= 0.1 * self.engine_w.abs().max(1.0);
        let (t_gear, ie) = if locked_now {
            // rigid: the engine turns with the wheels, its inertia reflected
            self.engine_w = w_in;
            (te, ie_eng * ratio * ratio)
        } else if ratio == 0.0 || e <= 0.0 {
            self.engine_w = (self.engine_w + dt * te / ie_eng).max(0.0);
            (0.0, 0.0)
        } else {
            // slipping, implicit in the clutch's slope
            let dw = self.engine_w - w_in;
            let tc = dw / (dw.abs() + 4.0) * -self.cl.max_torque * e;
            let k = self.cl.max_torque * e * 4.0 / (dw.abs() + 4.0).powi(2);
            self.engine_w = (self.engine_w + dt * (te + tc) / (ie_eng + dt * k)).max(0.0);
            (-tc, 0.0)
        };
        self.clutch_locked = locked_now;
        // stall below ~30 % of idle; the starter brings it back, or by itself
        // after a second with the clutch in or in neutral
        let rpm = self.engine_w * 60.0 / tau;
        if self.running && rpm < self.idle * 0.3 {
            self.running = false;
            self.stalled_for = 0.0;
        }
        if !self.running {
            self.stalled_for += dt;
            if self.crank <= 0.0 && self.stalled_for > 1.0 && (self.clutch_e < 0.3 || self.gear == 0) {
                self.crank = 1.0;
            }
            if self.crank > 0.0 {
                self.crank -= dt;
                if rpm > self.idle * 0.4 || self.crank <= 0.0 {
                    self.running = true;
                    self.engine_w = self.engine_w.max(self.idle * 0.7 * tau / 60.0);
                    self.crank = 0.0;
                }
            }
        }
        self.rpm = rpm.min(self.limiter);
        let t_axle = t_gear * ratio * 0.9;

        // ---- per corner: hub, tyre contact, forces
        // with ODE the world's gravity pulls the chassis and the hubs
        // the wishbones' state: the hubs where the rods hold them, the
        // spring travel from the linkage
        let kin = self.links.as_mut().map(|l| {
            l.sync(self.cg, self.q, self.v, self.w);
            l.kin()
        });
        if let Some(kin) = kin.as_ref() {
            // EVOVIEW_RIG_LINKLOG=1: each hub's yaw / camber in the chassis
            // frame (deg) against the asked steer, every 0.25 s
            if std::env::var_os("EVOVIEW_RIG_LINKLOG").is_some() && (self.t / 0.25).floor() != ((self.t - dt) / 0.25).floor() {
                let rt = Mat3::from_quat(self.q).transpose();
                let a: Vec<String> = kin.iter().map(|kn| {
                    let (f, l) = (rt * (kn.rot * Vec3::Z), rt * (kn.rot * Vec3::X));
                    format!("yaw {:+.2} cam {:+.2}", f.x.atan2(f.z).to_degrees(), l.y.asin().to_degrees())
                }).collect();
                println!("LINK t {:.2} steer {:+.2} | {}", self.t, self.steer.to_degrees(), a.join(" | "));
            }
            for k in 0..4 {
                self.c[k].x = kin[k].travel;
                self.c[k].xv = kin[k].rate;
            }
        }
        let ode_on = self.ode.is_some() || kin.is_some();
        let mut f_body = if ode_on { Vec3::ZERO } else { Vec3::new(0.0, -self.mass * G, 0.0) };
        let mut t_body = Vec3::ZERO;
        // ODE: each hub's own force and torque (about the hub)
        let mut f_hub = [Vec3::ZERO; 4];
        let mut t_hub = [Vec3::ZERO; 4];
        let apply = |f_body: &mut Vec3, t_body: &mut Vec3, f: Vec3, at: Vec3, cg: Vec3| {
            *f_body += f;
            *t_body += (at - cg).cross(f);
        };
        // aero, on the axle lines, and drag at the CG
        let vsq = self.v.length_squared();
        let zf = (self.anchor[0].z + self.anchor[1].z) * 0.5;
        let zr = (self.anchor[2].z + self.anchor[3].z) * 0.5;
        let o = self.cg - r * self.cg_local;
        apply(&mut f_body, &mut t_body, -up * self.aero[0] * vsq, o + fw * zf, self.cg);
        apply(&mut f_body, &mut t_body, -up * self.aero[1] * vsq, o + fw * zr, self.cg);
        f_body -= self.v.normalize_or_zero() * self.aero[2] * vsq;

        let mut brakes = self.brakes.torques(brake_in, self.hb);
        if let Some(abs) = self.abs.as_ref().filter(|_| self.abs_on) {
            // ours FR FL RR RL <- the game's FL FR RL RR
            let m = abs.mult;
            let pedal = self.brakes.torques(brake_in, 0.0);
            for (k, gi) in [(0usize, 1usize), (1, 0), (2, 3), (3, 2)] {
                // the cut is on the pedal's torque, not the handbrake's
                brakes[k] -= pedal[k] * (1.0 - m[gi]);
            }
        }
        // spring side (coil + helper + bump stops, then the ARB) and the
        // damper as a linear coefficient at the current speed, solved
        // implicitly with the hub below
        let mut susp = [0.0f32; 4];
        let mut c_eff = [0.0f32; 4];
        for k in 0..4 {
            let ax = k / 2;
            let m = self.mr[ax];
            let (sp, dmp) = self.coil[ax].force(self.c[k].x * m, self.c[k].xv * m);
            let v = self.c[k].xv * m;
            let d = &self.coil[ax].damper;
            let stop_d = dmp + d.force(v);
            susp[k] = (sp + stop_d) * m;
            c_eff[k] = if v.abs() > 1e-4 { (d.force(v) / v).abs() } else { d.slow_bump.max(d.slow_rebound) } * m * m;
        }
        // ARB on the compression difference, pushing like the springs
        for ax in 0..2 {
            let (rk, lk) = (ax * 2, ax * 2 + 1);
            let d = self.c[lk].x - self.c[rk].x;
            susp[lk] += self.arb[ax] * d;
            susp[rk] -= self.arb[ax] * d;
        }

        let shares = self.driven();
        let mut t_drive = [0.0f32; 4];
        for k in 0..4 {
            t_drive[k] = t_axle * shares[k];
        }
        let mut mz_front = 0.0;
        for k in 0..4 {
            let ax = k / 2;
            let kn = kin.as_ref().map(|kk| kk[k]);
            let anchor_w = kn.map_or(o + r * self.anchor[k], |kn| kn.href);
            let rel = self.c[k].x - self.x_static[k];
            let hub = kn.map_or(anchor_w + up * rel, |kn| kn.hub);
            self.c[k].hub = hub;
            let hub_v = kn.map_or(self.v + self.w.cross(hub - self.cg) + up * self.c[k].xv, |kn| kn.v);
            let Some(mut t) = self.tyre[ax] else { continue };
            // a blown tyre rolls on its rim (the game's radius getter lerps
            // to rimRadius as the tyre's +0x458 goes to 0)
            if let (Some(d), Some(th)) = (self.therm_data[ax].as_ref(), self.therm[k].as_ref()) {
                if th.blown {
                    t.radius = d.rim_radius;
                }
            }
            // a tyre only touches ground under the car: rolled over, the
            // ground beneath the hub is above the wheel and the roof takes it
            let hit = ground.probe(hub.x, hub.z, hub.y + 0.5).filter(|h| h.n.dot(up) > 0.25);
            if std::env::var_os("EVOVIEW_RIG_DBG").is_some() {
                match hit {
                    None => eprintln!("PROBE t {:.3} k {k} none at ({:.2} {:.2} {:.2})", self.t, hub.x, hub.y, hub.z),
                    Some(h) if (hub.y - h.y) > t.radius + 0.03 => eprintln!("PROBE t {:.3} k {k} gap {:.3} hub y {:.3} ground {:.3} n {:?}", self.t, hub.y - h.y - t.radius, hub.y, h.y, h.n),
                    _ => {}
                }
            }
            let mut f_hub_u = susp[k] * -1.0 - self.m_unsprung[ax] * G * up.y;
            self.c[k].on_ground = false;
            self.c[k].fz = 0.0;
            // the tyre's envelope over what it rolls on (tyre_envelope);
            // EVOVIEW_TYRE_CONTACT=old: the centre triangle's own plane
            let env = if self.old_contact { None } else { hit.and_then(|h| tyre_envelope(ground, hub, fw, up, self.steer_of(k), t.radius, h)) };
            let hit = match env {
                Some((_, n, s)) => hit.map(|h| Hit { n, surf: s, ..h }),
                None if self.old_contact => hit,
                // nothing low enough under the tyre to stand on
                None => None,
            };
            if let Some(h) = hit {
                let n = h.n;
                let dist = match env {
                    Some((gy, n, _)) => (hub.y - gy) * n.y,
                    None => {
                        // the tyre envelops what it rolls over: the ground height is
                        // the mean of five samples along the patch (+-0.25 m), so a
                        // kerb face ramps in instead of landing as one step
                        let mut gy = 0.0;
                        let mut nh = 0.0;
                        for d in [-0.25f32, -0.12, 0.0, 0.12, 0.25] {
                            let q = hub + fw * d;
                            if let Some(hq) = ground.probe(q.x, q.z, hub.y + 0.5) {
                                // the sample's height under the hub along its plane
                                gy += hq.y + ground.bump(hq.surf, q.x, q.z) - d * fw.y;
                                nh += 1.0;
                            }
                        }
                        let gp = Vec3::new(hub.x, if nh > 0.0 { gy / nh } else { h.y }, hub.z);
                        (hub - gp).dot(n)
                    }
                };
                let defl = t.radius - dist;
                self.c[k].gap = dist;
                self.c[k].gn = n;
                self.c[k].surf = h.surf;
                if defl > 0.0 {
                    self.c[k].on_ground = true;
                    let ddot = -hub_v.dot(n);
                    // the radial rate moves with the tyre's pressure
                    let rate = match (self.therm_data[ax].as_ref(), self.therm[k].as_ref()) {
                        (Some(d), Some(th)) => th.spring_rate(d, t.rate),
                        _ => t.rate,
                    };
                    let fz = (rate * defl + t.rate_prog * defl * defl + t.damping * ddot).max(0.0);
                    // wheel heading in the ground plane
                    // toe-in turns each wheel's nose toward the car's middle:
                    // the left (+x) wheel to the right
                    let toe_k = self.toe[ax] + self.toe_gain[ax] * (self.c[k].x - self.x_static[k]);
                    let toe = if k % 2 == 1 { -toe_k } else { toe_k };
                    let steer = if ax == 0 { self.steer } else { 0.0 } + toe;
                    // with the rods the hub's own nose: steer, toe and bump
                    // steer are the linkage's
                    let heading = kn.map_or(Quat::from_axis_angle(up, steer) * fw, |kn| kn.rot * Vec3::Z);
                    let wf = (heading - n * heading.dot(n)).normalize_or_zero();
                    let wl = n.cross(wf);
                    let contact = hub - n * dist;
                    let vc = kn.map_or(self.v + self.w.cross(contact - self.cg), |kn| kn.v + kn.w.cross(contact - kn.hub));
                    let (vl, vt) = (vc.dot(wf), vc.dot(wl));
                    let re = (t.radius - defl * 0.3).max(0.1);
                    let den = vl.abs().max(2.0);
                    // Relaxation (the game's tyre step, FUN_14287d580): the
                    // slip angle's tangent and the slip ratio follow the
                    // instant slip over the rolled length L - twice the data's
                    // at small slip, the data's from the grip peak on:
                    //   s <- e^(-dt|v|/L) (p/|v| + s) - p/|v|   (|v| > 0)
                    //   s <- s - p dt / L                       (rolling stopped)
                    // with p the slip velocity, so it settles at -p/|v|.
                    // Without it (EVOVIEW_NO_RELAX=1) the slip is instant on a
                    // 2 m/s floor.
                    let relaxed = (self.relax && t.relax_x > 0.0 && t.relax_y > 0.0).then(|| {
                        let kn = self.c[k].nd.clamp(0.0, 1.0);
                        let (ly, lx) = (t.relax_y * (2.0 - kn), t.relax_x * (2.0 - kn));
                        let v = vl.abs();
                        let step = |l: f32, p: f32, prev: f32| if v > 1e-4 {
                            let q = p / v;
                            (-dt * v / l).exp() * (q + prev) - q
                        } else {
                            prev - p * dt / l
                        };
                        let ta = step(ly, vt, self.c[k].relax_ta).clamp(-57.0, 57.0);
                        let s = step(lx, -(self.c[k].omega * re - vl), self.c[k].relax_sr).clamp(-3.0, 3.0);
                        self.c[k].relax_ta = ta;
                        self.c[k].relax_sr = s;
                        // d(slip ratio) / d(omega) this step, for the implicit spin
                        let dsr = if v > 1e-4 { re * (1.0 - (-dt * v / lx).exp()) / v } else { re * dt / lx };
                        (ta.atan(), s, dsr)
                    });
                    let alpha = relaxed.map_or((-vt).atan2(den), |r| r.0);
                    // water under the patch: the tyre model's wet input, and
                    // the rain washing out the rubbered line's grip bonus
                    let wet = crate::rain::patch_wetness(contact.x, contact.z);
                    crate::rain::dry_patch(contact.x, contact.z, vl, dt);
                    let mut grip = ground.grip_wet(h.surf, wet) * crate::rain::rubber_grip(wet);
                    // sand (surface_type 4): the game's rolling resistance x10
                    let rr_k = if ground.kind(h.surf) == 4 { 10.0 } else { 1.0 };
                    // the wheel's camber to the body: static + the linkage's
                    // gain with bump; then the body's roll to the ground
                    // tilts both tops toward the low side (the tyre frame's
                    // lateral axis is the car's left: a top leaning inward
                    // is + on the right, - on the left)
                    // the rods tilt the hub itself: then only the static camber
                    let cb = self.camber[ax] + if kn.is_some() { 0.0 } else { self.camber_gain[ax] * (self.c[k].x - self.x_static[k]) };
                    let lat = kn.map_or(r * Vec3::X, |kn| kn.rot * Vec3::X);
                    let roll = lat.dot(n).clamp(-1.0, 1.0).asin();
                    let camber = if k % 2 == 1 { cb } else { -cb } - roll;
                    // the compound's thermal performance curve
                    let mut temp = f32::NAN;
                    let grain = self.therm[k].as_ref().map_or(0.0, |th| th.grain);
                    let pressure = match (self.therm_data[ax].as_ref(), self.therm[k].as_ref()) {
                        (Some(d), Some(th)) => th.pressure_ratio(d),
                        _ => 0.0,
                    };
                    if let (Some(d), Some(th)) = (self.therm_data[ax].as_ref(), self.therm[k].as_ref()) {
                        grip *= th.grip(d, camber);
                        if self.therm_model {
                            temp = th.model_temp(d);
                        }
                    }
                    // SCTM's grip input is the tyre's own chain (thermal / wear,
                    // ~1): its slip stiffness divides by gripSlipFactor x
                    // (grip - 1) + 1, which goes negative on a low-grip surface
                    // (M4: factor 5). The surface scales the forces instead.
                    let solve = |sr: f32| {
                        let mut o = t.solve(&Input { fz, alpha, sr, camber, speed: vc.length(), grip: 1.0, wet, temp, grain, pressure, ..Default::default() });
                        o.fx *= grip;
                        o.fy *= grip;
                        o.mz *= grip;
                        o
                    };
                    let sr = relaxed.map_or(((self.c[k].omega * re - vl) / den).clamp(-3.0, 3.0), |r| r.1);
                    let o0 = solve(sr);
                    if std::env::var_os("EVOVIEW_RIG_DBG").is_some() && o0.fy.abs() > 3000.0 && alpha.abs() < 0.01 {
                        eprintln!("DBG k {k} fz {fz:.0} alpha {alpha:.5} sr {sr:.4} speed {:.2} grip {grip:.2} -> fx {:.0} fy {:.0}", vc.length(), o0.fx, o0.fy);
                    }
                    // slip stiffness for the implicit spin update
                    let ds = 0.002;
                    let kx = ((solve(sr + ds).fx - o0.fx) / ds).abs();
                    self.c[k].fz = fz;
                    self.c[k].alpha = alpha;
                    self.c[k].sr = sr;
                    self.c[k].fx = o0.fx;
                    self.c[k].fy = o0.fy;
                    self.c[k].mz = o0.mz;
                    self.c[k].grip = grip;
                    self.c[k].nd = o0.nd_slip;
                    // the friction work and rolling into the tread
                    if let (Some(d), Some(th)) = (self.therm_data[ax].as_ref(), self.therm[k].as_mut()) {
                        let v_slip = Vec2::new(self.c[k].omega * re - vl, vt).length();
                        let road = if self.road_t.is_nan() { self.air_t + 8.0 * (1.0 - wet.min(1.0)) } else { self.road_t };
                        th.heat(d, fz, t.d_y(fz) * grip, v_slip, self.c[k].omega, camber, road, rr_k);
                        th.wear(d, fz, v_slip, vc.length(), alpha, o0.nd_slip, camber, dt);
                    }
                    let f = n * fz + wf * o0.fx + wl * o0.fy;
                    if ax == 0 {
                        // what the rack feels: the tyre's whole force about the
                        // hub's steering axis (caster trail, scrub, kingpin
                        // inclination) plus its aligning moment; the plain Mz
                        // without the geometry
                        let kp = match (kn, self.kingpin[k]) {
                            // the rods' steering axis, lower ball up to the upper
                            (Some(kn), _) => Some(kn.kp),
                            (None, Some((pt, dir))) => Some((hub + r * pt, (r * dir).normalize_or_zero())),
                            _ => None,
                        };
                        mz_front += match kp {
                            Some((a, kd)) => {
                                // the game's (tyre step 0x1421820b0 -> hub
                                // addForceAtPos flag 1, 0x14222cd40): only
                                // the road-plane force Fx + Fy at the contact,
                                // plus Mz about the normal - the load is not
                                // in it (EVOVIEW_FFB_FZ=1 puts it back)
                                let fs = if self.ffb_fz { f } else { wf * o0.fx + wl * o0.fy };
                                (contact - a).cross(fs).dot(kd) + o0.mz * kd.dot(n)
                            }
                            None => o0.mz,
                        };
                    }
                    // the part along the hub's slide goes to the hub; the rest
                    // straight into the body at the roll centre
                    if ode_on {
                        // all of it on the hub; the slider hands the
                        // off-axis part to the chassis
                        // (the rods: at the patch itself, no roll centre)
                        let at = if kn.is_some() { contact } else { contact + n * self.rc_height };
                        f_hub[k] += f;
                        t_hub[k] += (at - hub).cross(f);
                    } else {
                        let along = f.dot(up);
                        f_hub_u += along;
                        let rest = f - up * along;
                        apply(&mut f_body, &mut t_body, rest, contact + n * self.rc_height, self.cg);
                    }

                    // wheel spin: drive - Fx R - brake, implicit in kx
                    let iw = t.inertia + ie * shares[k] * if shares[k] > 0.0 { 1.0 } else { 0.0 };
                    let g = kx * re * relaxed.map_or(re / den, |r| r.2);
                    let mut om = self.c[k].omega + dt / iw * (t_drive[k] - o0.fx * re) / (1.0 + dt * g / iw);
                    // rolling resistance (the tyre's own, tyre+0x400; x10 in
                    // sand) stops the wheel like a brake does
                    let rr = match (self.therm_data[ax].as_ref(), self.therm[k].as_ref()) {
                        (Some(d), Some(th)) if !self.no_rr => th.rr_torque(d, fz, om) * rr_k,
                        _ => 0.0,
                    };
                    let tb = (brakes[k] + rr) * dt / iw;
                    om = if om.abs() <= tb { 0.0 } else { om - om.signum() * tb };
                    self.c[k].omega = om;
                    if let (Some(d), Some(th)) = (self.therm_data[ax].as_ref(), self.therm[k].as_mut()) {
                        th.flat_spot(d, om, sr, vl.abs(), fz, dt);
                    }
                }
            }
            if let (Some(d), Some(th)) = (self.therm_data[ax].as_ref(), self.therm[k].as_mut()) {
                let wet = if self.c[k].on_ground { crate::rain::track_wetness() } else { 0.0 };
                th.step(d, dt, self.c[k].omega, self.air_t, wet);
            }
            if !self.c[k].on_ground {
                // off the ground the patch lets go of its slip
                self.c[k].relax_ta = 0.0;
                self.c[k].relax_sr = 0.0;
                let iw = t.inertia;
                let tb = brakes[k] * dt / iw;
                let mut om = self.c[k].omega + dt / iw * t_drive[k];
                om = if om.abs() <= tb { 0.0 } else { om - om.signum() * tb };
                self.c[k].omega = om;
            }
            // hub along the body's up: its own forces minus the anchor's
            // acceleration (the frame it slides in), damper implicit
            // the hub's own velocity along the strut, against the anchor's
            // (body translation + rotation): damper implicit between them
            if let Some(kn) = kn {
                // the spring on its own line (susp holds the spring and the
                // stops; the main damper is c_eff, as on the explicit path);
                // the hub takes its share at the ball, the arm's body pivot
                // the rest
                let fs = (susp[k] + c_eff[k] * self.c[k].xv) * kn.per_wheel;
                let fh = -kn.axis * fs * kn.hub_share;
                f_hub[k] += fh;
                t_hub[k] += (kn.bottom - hub).cross(fh);
                apply(&mut f_body, &mut t_body, kn.axis * fs, kn.top, self.cg);
                if kn.hub_share < 1.0 {
                    apply(&mut f_body, &mut t_body, -kn.axis * fs * (1.0 - kn.hub_share), kn.pivot, self.cg);
                }
                self.c[k].angle += self.c[k].omega * dt;
                continue;
            }
            if ode_on {
                // spring and stops (susp) plus the main damper (c xv)
                // between hub and chassis, as the explicit path's body sees
                let fs = susp[k] + c_eff[k] * self.c[k].xv;
                f_hub[k] -= up * fs;
                apply(&mut f_body, &mut t_body, up * fs, anchor_w, self.cg);
                self.c[k].angle += self.c[k].omega * dt;
                continue;
            }
            let mu = self.m_unsprung[ax];
            let va = (self.v + self.w.cross(anchor_w - self.cg)).dot(up);
            let u = va + self.c[k].xv;
            let c = c_eff[k];
            let u_new = (u + dt / mu * (f_hub_u + c * va)) / (1.0 + dt * c / mu);
            let xv = u_new - va;
            self.c[k].xv = xv;
            self.c[k].x += xv * dt;
            self.c[k].angle += self.c[k].omega * dt;
            // the body feels spring + damper at the anchor (equal, opposite)
            apply(&mut f_body, &mut t_body, up * (susp[k] + c * xv), anchor_w, self.cg);
        }
        // Flat spots thump the rack once a wheel turn (getFFB, above
        // 7 rad/s): saw(phase, 4 pi) x max(flat FL, FR) x gain x the mean
        // front load, added after the game's -(rack) x 0.9 - so here
        // taken off the rack / 0.9. (The game's other term, its blister
        // percent x 0.003, is not modelled.)
        let w_sum = (self.c[0].omega + self.c[1].omega).abs();
        self.vib_phase = (self.vib_phase + w_sum * dt) % (4.0 * std::f32::consts::PI);
        if w_sum > 7.0 {
            let flat = self.therm[0].as_ref().map_or(0.0, |t| t.flat).max(self.therm[1].as_ref().map_or(0.0, |t| t.flat));
            if flat > 0.0 {
                let p = 2.0 * std::f32::consts::PI;
                let saw = (self.vib_phase % (2.0 * p) - p) / p;
                mz_front -= saw * flat * self.vib_gain * (self.c[0].fz + self.c[1].fz) * 0.5 / 0.9;
            }
        }
        self.mz_front = mz_front;
        self.mz_sum += mz_front;
        self.mz_n += 1;
        // the diff: equalise the driven pair within its lock
        let (lk, rk) = if self.front_share >= 1.0 { (1, 0) } else { (3, 2) };
        if let Some(tl) = self.tyre[lk / 2] {
            let iw = tl.inertia + ie * 0.5;
            let dw = self.c[lk].omega - self.c[rk].omega;
            let need = dw * iw / dt;
            let lock = self.diff.lock(t_axle, self.c[lk].fx, self.c[rk].fx);
            let td = need.clamp(-lock, lock) * 0.5;
            self.c[lk].omega -= td * dt / iw;
            self.c[rk].omega += td * dt / iw;
        }

        if std::env::var_os("EVOVIEW_RIG_NOWALLS").is_none() {
            self.walls(ground);
        }
        self.body_ground(ground);

        if let Some(mut l) = self.links.take() {
            // the rack: steer on the fronts, the setup's toe on all four
            // (toe-in turns a left wheel's nose right)
            let ang = [0usize, 1, 2, 3].map(|k| if k < 2 { self.steer } else { 0.0 } + if k % 2 == 1 { -self.toe[k / 2] } else { self.toe[k / 2] });
            l.set_wheel_angles(ang);
            let (cg, q, v, w) = l.step(f_body, t_body, &f_hub, &t_hub, dt);
            self.links = Some(l);
            self.acc_prev = (v - self.v) / dt;
            (self.cg, self.q, self.v, self.w) = (cg, q, v, w);
            return;
        }
        if let Some(mut ode) = self.ode.take() {
            let frame = crate::rigode::Frame {
                cg: self.cg,
                q: self.q,
                v: self.v,
                w: self.w,
                cg_local: self.cg_local,
                anchor: &self.anchor,
                x_static: &self.x_static,
                x: self.c.map(|c| c.x),
                xv: self.c.map(|c| c.xv),
                f_body,
                t_body,
                f_hub,
                t_hub,
            };
            let (cg, q, v, w, hubs) = ode.step(&frame, dt);
            self.ode = Some(ode);
            self.acc_prev = (v - self.v) / dt;
            (self.cg, self.q, self.v, self.w) = (cg, q, v, w);
            for k in 0..4 {
                self.c[k].x = hubs[k].x;
                self.c[k].xv = hubs[k].xv;
            }
            return;
        }
        // ---- body integration (semi-implicit Euler)
        let acc = f_body / self.mass;
        self.v += acc * dt;
        self.acc_prev = acc;
        self.cg += self.v * dt;
        let i_world = r * Mat3::from_diagonal(self.inertia) * r.transpose();
        let gyro = self.w.cross(i_world * self.w);
        self.w += i_world.inverse() * (t_body - gyro) * dt;
        let wq = Quat::from_xyzw(self.w.x, self.w.y, self.w.z, 0.0);
        self.q = (self.q + wq * self.q * (0.5 * dt)).normalize();
    }
}

/// The ground a tyre of radius `r` at `hub` stands on: (height under the
/// hub, normal, surface). ⚠ The single triangle under the hub is not it:
/// kerbs are modelled with ridges (Imola's KERB faces lean 50 deg), and the
/// hub's distance to that steep plane read as 13-14 cm of tyre squash with
/// the force thrown 50 deg sideways (EVOVIEW_COLLIDE_LOG: ~90 NJUMP / PEN
/// events a lap at Imola). The game casts per wheel against the mesh
/// (OPCODE ray colliders); here the tyre is a disc in its own plane:
/// - the ground is sampled along the wheel's heading at d = -0.95 r ..
///   0.95 r and joined into a polyline; the contact is the point of it
///   that reaches deepest into the disc (on each segment the maximum of
///   h(d) + sqrt(r^2 - d^2) has a closed form), so a step ahead ramps in
///   continuously as the wheel rolls onto it and the gaps between ridges
///   are bridged;
/// - the normal runs from that contact point to the hub - the plane's own
///   normal on a slope, tilted back against a step's edge - plus the
///   camber from two samples across the tread; limited to 30 / 20 deg, so
///   a ridge face never throws the car sideways;
/// - anything above the hub is not ground (the body's walls take it).
pub fn tyre_envelope(ground: &Ground, hub: Vec3, fw: Vec3, up: Vec3, steer: f32, r: f32, centre: Hit) -> Option<(f32, Vec3, u16)> {
    let heading = Quat::from_axis_angle(up, steer) * fw;
    let fh = Vec3::new(heading.x, 0.0, heading.z).normalize_or_zero();
    if fh == Vec3::ZERO {
        return None;
    }
    let lh = Vec3::new(fh.z, 0.0, -fh.x);
    let top = hub.y + 0.02;
    // the geometry only: the surface's sinHeight bumps (grass, sand) go on
    // the height under the hub below - in the samples they tilted the
    // normal +-11 deg at ~90 Hz across grass at speed
    let h_at = |q: Vec3| ground.probe(q.x, q.z, top).map(|h| (h.y, h.surf));
    const N: usize = 11;
    let mut pts: [Option<(f32, f32, u16)>; N] = [None; N];
    for (i, p) in pts.iter_mut().enumerate() {
        let d = r * 0.95 * (2.0 * i as f32 / (N - 1) as f32 - 1.0);
        *p = h_at(hub + fh * d).map(|(y, s)| (d, y, s));
    }
    // deepest point of the polyline into the disc: lift = h(d) + sqrt(r^2 - d^2)
    let mut best: Option<(f32, f32, f32, u16)> = None; // lift, d, h, surface
    let mut consider = |d: f32, h: f32, s: u16| {
        let lift = h + (r * r - d * d).max(0.0).sqrt();
        if best.map_or(true, |b| lift > b.0) {
            best = Some((lift, d, h, s));
        }
    };
    for i in 0..N {
        let Some((d0, h0, s0)) = pts[i] else { continue };
        consider(d0, h0, s0);
        let Some((d1, h1, s1)) = pts.get(i + 1).copied().flatten() else { continue };
        let m = (h1 - h0) / (d1 - d0);
        let d = (r * m / (1.0 + m * m).sqrt()).clamp(d0, d1);
        consider(d, h0 + m * (d - d0), if d - d0 < d1 - d { s0 } else { s1 });
    }
    let (_, d, h, surf) = best?;
    // contact -> hub, in the wheel's plane
    let (dx, dy) = (-d, hub.y - h);
    let dist = (dx * dx + dy * dy).sqrt();
    let lim = 0.58; // tan 30 deg
    let slope_f = if dy > 1e-4 { (-dx / dy).clamp(-lim, lim) } else { 0.0 };
    let w = 0.12;
    let slope_l = match (h_at(hub + lh * w), h_at(hub - lh * w)) {
        (Some(a), Some(b)) => (a.0 - b.0) / (2.0 * w),
        // one side off an edge: the centre triangle's camber
        _ => if centre.n.y > 0.5 { -centre.n.dot(lh) / centre.n.y } else { 0.0 },
    };
    // across the tread at most 20 deg: straddling a kerb's edge the two
    // samples see its whole height over 24 cm
    let nrm = (Vec3::Y - fh * slope_f - lh * slope_l.clamp(-0.36, 0.36)).normalize();
    // as a height under the hub, for the caller's (hub.y - gy) n.y
    Some((hub.y - dist / nrm.y + ground.bump(surf, hub.x, hub.z), nrm, surf))
}

impl RigCar {
    /// Barriers: the body as three circles along its length; an
    /// overlapping wall pushes it out and takes a rigid-body impulse at the
    /// contact (restitution 0.15, friction 0.3).
    /// The body itself on the ground (rolled over, on its side): points on
    /// the upper body and roof pushed out along the ground normal with an
    /// impulse and sliding friction. The lower body is left to the tyres so
    /// kerbs and bumps never catch it.
    fn body_ground(&mut self, ground: &Ground) {
        let r = self.rot();
        let (fw, up) = (r * Vec3::Z, r * Vec3::Y);
        // upright, the tyres carry the car: the body points then only find
        // surfaces over it - a pit wall's flat top 0.9 m up, under which a
        // car rubbing the wall had its corner - and lifted it onto the wall
        if up.y > 0.7 {
            return;
        }
        let lat = r * Vec3::X;
        let o = self.cg - r * self.cg_local;
        let hw = self.anchor.iter().map(|c| c.x.abs()).fold(0.6f32, f32::max) + 0.05;
        // the whole length, overhangs and wing included: from the axles
        // +-0.5 m the nose, tail and wing of a car on its roof sank
        // through the road
        let af = self.anchor.iter().map(|c| c.z).fold(f32::MIN, f32::max);
        let ar = self.anchor.iter().map(|c| c.z).fold(f32::MAX, f32::min);
        let mid = 0.5 * (af + ar);
        let half = (0.5 * self.body_len).max(0.5 * (af - ar) + 0.5);
        let (zf, zr) = (mid + half, mid - half);
        let i_world = r * Mat3::from_diagonal(self.inertia) * r.transpose();
        let i_inv = i_world.inverse();
        let h = self.body_h;
        // the deepest point's push-out, once: summed over the 25 points a
        // car beside Laguna's earth banks went up 1.4 m in one step
        let mut out = (0.0f32, Vec3::Y);
        for z in [zf, zf - 0.25 * (zf - zr), 0.5 * (zf + zr), zr + 0.25 * (zf - zr), zr] {
            for (x, y) in [(-hw, 0.5), (hw, 0.5), (-hw * 0.8, h), (hw * 0.8, h), (0.0, h)] {
                let p = o + fw * z + lat * x + up * y;
                let Some(g) = ground.probe(p.x, p.z, p.y + 1.5) else { continue };
                let pen = (g.y - p.y) * g.n.y;
                // over half a metre under a surface is not sinking into it
                // (a body sinks a few cm a step): it is a bank or barrier top
                // beside the car, and walls() deals with those
                if pen <= 0.0 || pen > 0.5 {
                    continue;
                }
                let n = g.n;
                if pen > 0.05 && std::env::var_os("EVOVIEW_RIG_DBG").is_some() {
                    eprintln!("BODYGROUND t {:.3} point ({x:+.2} {y:.2} {z:+.2}) at ({:.1} {:.2} {:.1}) ground y {:.2} n {:?} pen {pen:.2} surf {}", self.t, p.x, p.y, p.z, g.y, g.n, ground.surf_name(g.surf));
                }
                if pen > out.0 {
                    out = (pen, n);
                }
                let arm = p - self.cg;
                let vp = self.v + self.w.cross(arm);
                let vn = vp.dot(n);
                if vn >= 0.0 {
                    continue;
                }
                let rn = arm.cross(n);
                let j = -1.2 * vn / (1.0 / self.mass + rn.dot(i_inv * rn));
                let vt = vp - n * vn;
                let t = vt.normalize_or_zero();
                let rt = arm.cross(t);
                let jt = (vt.length() / (1.0 / self.mass + rt.dot(i_inv * rt))).min(0.5 * j);
                let imp = n * j - t * jt;
                self.v += imp / self.mass;
                self.w += i_inv * arm.cross(imp);
            }
        }
        self.cg += out.1 * out.0.min(0.2);
    }

    fn walls(&mut self, ground: &Ground) {
        let r = self.rot();
        let (fw, up) = (r * Vec3::Z, r * Vec3::Y);
        let o = self.cg - r * self.cg_local;
        let half_w = self.anchor.iter().map(|c| c.x.abs()).fold(0.6f32, f32::max) + 0.12;
        let zf = self.anchor.iter().map(|c| c.z).fold(f32::MIN, f32::max) + 0.55;
        let zr = self.anchor.iter().map(|c| c.z).fold(f32::MAX, f32::min) - 0.55;
        let i_world = r * Mat3::from_diagonal(self.inertia) * r.transpose();
        let i_inv = i_world.inverse();
        let mut hits = Vec::new();
        for z in [zf - half_w, 0.5 * (zf + zr), zr + half_w] {
            let c = o + fw * z + up * 0.3;
            hits.clear();
            ground.walls_near(c, half_w, o.y + 0.15, o.y + 1.2, &mut hits);
            for &(q, d, wi) in hits.iter() {
                let n = Vec3::new(c.x - q.x, 0.0, c.z - q.z);
                let n = if d > 1e-4 { n / d } else { -fw };
                if std::env::var_os("EVOVIEW_RIG_DBG").is_some() {
                    eprintln!("WALL t {:.3} z {z:.2} c ({:.1} {:.1} {:.1}) q ({:.1} {:.1} {:.1}) d {d:.3} n {n:?} v {:.1}", self.t, c.x, c.y, c.z, q.x, q.y, q.z, self.speed());
                }
                self.cg += n * (half_w - d);
                let arm = Vec3::new(q.x, c.y, q.z) - self.cg;
                let vp = self.v + self.w.cross(arm);
                let vn = vp.dot(n);
                if vn >= 0.0 {
                    continue;
                }
                if let Some(ev) = self.ev.as_mut() {
                    ev.wall(self.t, ground, wi, Vec3::new(q.x, c.y, q.z), n, vn, self.v.length() * 3.6);
                }
                // a hard hit: say where and what (finding invisible walls)
                if vn < -2.0 && self.t - self.last_wall_log > 0.5 {
                    self.last_wall_log = self.t;
                    if let Some((src, w)) = ground.wall_name_near(c, half_w, o.y + 0.15, o.y + 1.2) {
                        println!("WALL HIT t {:.1} at ({:.1}, {:.1}, {:.1}) {:.1} m/s into it, {:.0} km/h: {src} segment ({:.1},{:.1})-({:.1},{:.1}) y {:.2}..{:.2}",
                            self.t, c.x, c.y, c.z, -vn, self.speed() * 3.6, w[0], w[1], w[2], w[3], w[4], w[5]);
                    }
                }
                let rn = arm.cross(n);
                let k = 1.0 / self.mass + rn.dot(i_inv * rn);
                let j = -1.15 * vn / k;
                let vt = vp - n * vn;
                let t = vt.normalize_or_zero();
                let rt = arm.cross(t);
                let jt = (vt.length() / (1.0 / self.mass + rt.dot(i_inv * rt))).min(0.3 * j);
                let imp = n * j - t * jt;
                self.v += imp / self.mass;
                self.w += i_inv * arm.cross(imp);
            }
        }
    }

    /// corner k's road-wheel steer + toe, as the tyre step turns it
    fn steer_of(&self, k: usize) -> f32 {
        let ax = k / 2;
        let toe_k = self.toe[ax] + self.toe_gain[ax] * (self.c[k].x - self.x_static[k]);
        let toe = if k % 2 == 1 { -toe_k } else { toe_k };
        (if ax == 0 { self.steer } else { 0.0 }) + toe
    }

    pub fn t(&self) -> f32 {
        self.t
    }

    /// corner k's tyre radius (0 without a tyre model)
    pub fn radius(&self, k: usize) -> f32 {
        self.tyre[k / 2].map_or(0.0, |t| t.radius)
    }

    /// mesh-space hub of corner k at this moment (for wheel matrices)
    pub fn hub(&self, k: usize) -> Vec3 {
        self.c[k].hub
    }
}

// ---- pit stop / garage changes: pits.rs `apply` is the only caller ----
impl RigCar {
    /// The sprung (body) mass, kg.
    pub fn body_mass(&self) -> f32 {
        self.mass
    }

    /// Fuel in or out: the body gets heavier by `dm` kg, its inertia in
    /// proportion (the tank sits near the CG; its offset is not modelled).
    pub fn add_mass(&mut self, dm: f32) {
        let m = (self.mass + dm).max(100.0);
        self.inertia *= m / self.mass;
        self.mass = m;
    }

    /// Fresh tyres on axle 0 front / 1 rear: the compound's grip model and
    /// thermal data, both wheels new (full tread, no flat spot, not blown)
    /// at the blanket temperature (else the air), set to `psi` [right, left].
    pub fn fit_tyres(&mut self, axle: usize, sctm: Option<Sctm>, data: Option<crate::tyretherm::Data>, psi: [f32; 2]) {
        if sctm.is_some() {
            self.tyre[axle] = sctm;
        }
        if data.is_some() {
            self.therm_data[axle] = data;
        }
        for (i, k) in [axle * 2, axle * 2 + 1].into_iter().enumerate() {
            self.therm[k] = self.therm_data[axle].as_ref().map(|d| {
                let t0 = if d.blanket > 0.0 { d.blanket } else { self.air_t };
                let mut t = crate::tyretherm::Tyre::new(d, t0, self.air_t);
                if psi[i] > 0.0 && d.ideal_p > 0.0 {
                    t.set_pressure(psi[i]);
                }
                t
            });
            self.c[k].relax_ta = 0.0;
            self.c[k].relax_sr = 0.0;
        }
    }

    /// Tread left per wheel (ours: FR FL RR RL), mm, the three rows' mean.
    pub fn tread_mm(&self) -> Option<[f32; 4]> {
        let t = |k: usize| self.therm[k].as_ref().map(|t| t.tread.iter().sum::<f32>() / 3.0);
        Some([t(0)?, t(1)?, t(2)?, t(3)?])
    }

    /// A new tyre's tread, mm, per axle.
    pub fn tread_new(&self) -> [f32; 2] {
        [0, 1].map(|ax| self.therm_data[ax].as_ref().map_or(0.0, |d| d.tread_mm))
    }

    /// Anti-roll bar rates per axle (the setup's units, N/m).
    pub fn limiter_rpm(&self) -> f32 {
        self.limiter
    }

    pub fn arb(&self) -> [f32; 2] {
        self.arb
    }

    pub fn set_arb(&mut self, axle: usize, v: f32) {
        self.arb[axle.min(1)] = v.max(0.0);
    }
}

/// A flat 4 x 4 km asphalt pad (50 m quads), for `pad_test`.
pub fn flat_pad() -> Ground {
    let (n, s) = (80usize, 50.0f32);
    let mut c = crate::track::Collision {
        verts: Vec::new(),
        tris: Vec::new(),
        tri_surf: Vec::new(),
        surfaces: vec!["pad_asphalt".into()],
        walls: Vec::new(),
        wall_src: Vec::new(),
        wall_surf: Vec::new(),
    };
    let o = -(n as f32) * s * 0.5;
    for j in 0..=n {
        for i in 0..=n {
            c.verts.push([o + i as f32 * s, 0.0, o + j as f32 * s]);
        }
    }
    for j in 0..n {
        for i in 0..n {
            let a = (j * (n + 1) + i) as u32;
            let (b, d, e) = (a + 1, a + n as u32 + 1, a + n as u32 + 2);
            c.tris.push([a, d, b]);
            c.tris.push([b, d, e]);
            c.tri_surf.push(0);
            c.tri_surf.push(0);
        }
    }
    Ground::new(&c, &[])
}

/// `EVOVIEW_RIG_PAD=1 --drive-test N`: the car's numbers on a flat pad -
/// static loads, 0-100 / 0-200, 200-0 braking, steady-state cornering.
pub fn pad_test(rc: &mut RigCar) {
    let g = flat_pad();
    rc.auto_box = true;
    // EVOVIEW_GRIP_WET=0..1: the track's wetness for the run
    if crate::rain::track_wetness() > 0.0 {
        crate::rain::set_weather(1.0, 1.0);
        println!("PAD wet {}: tyres {:?}", crate::rain::track_wetness(), rc.phys.notes.iter().find(|n| n.starts_with("tyres")));
    }
    let mut acc = 0.0;
    let run = |rc: &mut RigCar, acc: &mut f32, c: Controls, secs: f32| {
        let n = (secs / DT) as usize;
        for _ in 0..n {
            rc.step(&c, DT, &g, acc);
        }
    };
    rc.place(Vec3::new(0.0, 0.0, -1800.0), 0.0);
    run(rc, &mut acc, Controls::default(), 2.0);
    let fz: Vec<String> = rc.c.iter().map(|c| format!("{:.0}", c.fz)).collect();
    let tot: f32 = rc.c.iter().map(|c| c.fz).sum();
    println!("PAD static: corner loads FR FL RR RL {} N (sum {:.0} = {:.0} kg), travel {:?}, body y {:.3}",
        fz.join(" "), tot, tot / G, rc.c.iter().map(|c| (c.x * 1000.0).round() / 1000.0).collect::<Vec<_>>(), rc.origin().y);
    // launch
    let full = Controls { throttle: 1.0, ..Default::default() };
    let (mut t, mut t100, mut t200) = (0.0f32, None, None);
    while t < 40.0 && t200.is_none() {
        run(rc, &mut acc, full, 0.05);
        t += 0.05;
        let kmh = rc.speed() * 3.6;
        if t100.is_none() && kmh >= 100.0 {
            t100 = Some(t);
        }
        if kmh >= 200.0 {
            t200 = Some(t);
        }
    }
    println!("PAD launch: 0-100 {:?} s, 0-200 {:?} s (gear {}, {:.0} rpm; TC {})", t100, t200, rc.gear, rc.rpm, rc.tc.as_ref().map_or("none".to_string(), |t| format!("level {} of {}, {:?}", t.level, t.levels.len(), t.levels.get(t.level))));
    // brake from here to a stop, straight
    let p0 = rc.origin();
    let v0 = rc.speed() * 3.6;
    let brake = Controls { brake: 1.0, ..Default::default() };
    let mut tb = 0.0;
    let mut peak = 0.0f32;
    while rc.speed() > 0.5 && tb < 20.0 {
        let v = rc.speed();
        run(rc, &mut acc, brake, 0.05);
        peak = peak.max((v - rc.speed()) / 0.05 / G);
        tb += 0.05;
    }
    println!("PAD braking: {:.0} km/h -> 0 in {:.1} m, {:.2} s, peak {:.2} g ({})", v0, (rc.origin() - p0).length(), tb, peak,
        rc.abs.as_ref().map_or("no ABS".to_string(), |a| format!("ABS level {} of {}", a.level, a.levels.len())));
    // skidpad: hold a speed with throttle, fixed steer
    for (kmh, steer) in [(80.0f32, 0.15f32), (80.0, 0.3), (120.0, 0.1), (160.0, 0.06)] {
        rc.place(Vec3::new(0.0, 0.0, -1800.0), 0.0);
        rc.v = rc.forward() * (kmh / 3.6);
        for c in rc.c.iter_mut() {
            c.omega = kmh / 3.6 / 0.34;
        }
        rc.gear = 3;
        let mut ay = 0.0;
        let mut beta = 0.0;
        let mut mz = 0.0;
        for i in 0..200 {
            let err = kmh / 3.6 - rc.speed();
            let thr = (err * 0.5).clamp(0.0, 1.0);
            let c = Controls { throttle: thr, brake: (-err * 0.3).clamp(0.0, 1.0), steer: steer / rc.max_steer.max(1e-3), analog: true, handbrake: false, ..Default::default() };
            run(rc, &mut acc, c, 0.05);
            if i >= 150 {
                ay += rc.speed() * rc.w.dot(rc.up()) / G / 50.0;
                beta += rc.v.dot(rc.left()).atan2(rc.speed()).to_degrees() / 50.0;
                mz += rc.mz_front / 50.0;
            }
        }
        let ffm = if rc.phys.ff_mult > 0.0 { rc.phys.ff_mult } else { 1.0 };
        println!("PAD ffb {kmh:.0} km/h steer {:.1} deg: rack torque {mz:+.0} N m -> device {:+.2} (ffMult {ffm})", steer.to_degrees(), mz * 0.9 * ffm * 0.001);
        println!("PAD skidpad {kmh:.0} km/h, road-wheel steer {:.1} deg: lateral {:.2} g, body slip {:+.1} deg, speed now {:.0} km/h, loads {:?}",
            steer.to_degrees(), ay, beta, rc.speed() * 3.6, rc.c.iter().map(|c| c.fz.round()).collect::<Vec<_>>());
    }
}

/// `EVOVIEW_RIG_PAD=1 EVOVIEW_RIG_FFB=1 --drive-test 1`: the wheel force
/// on the flat pad, both chains fed every 3 ms step as the game does - the
/// game's (ffb.rs with the player's EVO settings) and the pre-audit one
/// (EVOVIEW_FFB_LEGACY). Steady corners over speed x road-wheel angle, a
/// steering step at speed (where the rate damping acts) and parked.
pub fn ffb_table(rc: &mut RigCar) {
    let g = flat_pad();
    let mut acc = 0.0;
    let game = crate::ffb::Settings { boost: 1.0, ..crate::ffb::Settings::from_env() };
    std::env::set_var("EVOVIEW_FFB_LEGACY", "1");
    let old = crate::ffb::Settings { boost: 1.0, ..crate::ffb::Settings::from_env() };
    std::env::remove_var("EVOVIEW_FFB_LEGACY");
    let ffm = if rc.phys.ff_mult > 0.0 { rc.phys.ff_mult } else { 1.0 };
    let gam = if rc.phys.ff_gamma > 0.0 { rc.phys.ff_gamma } else { 1.0 };
    println!("FFB car gamma {gam} (SteerControlsData f5; the old chain ignored it)");
    println!("FFB ffMult {ffm} (device per rack N m {:.6}), max steer {:.1} deg, steer ratio {:.1}", 0.9 * ffm * 0.001, rc.max_steer.to_degrees(), rc.steer_ratio);
    println!("FFB game chain {game:?}");
    println!("FFB old chain  {old:?}");
    // one run: hold kmh with throttle, steer road-wheel `deg` (a ramp of
    // `ramp` s); returns per-step (t, ay g, rack N m, game Out, old Out)
    let run = |rc: &mut RigCar, acc: &mut f32, kmh: f32, deg: f32, ramp: f32, secs: f32| {
        rc.place(Vec3::new(0.0, 0.0, -1800.0), 0.0);
        rc.v = rc.forward() * (kmh / 3.6);
        for c in rc.c.iter_mut() {
            c.omega = kmh / 3.6 / 0.34;
        }
        rc.gear = if kmh < 1.0 { 1 } else { 3 };
        let (mut fg, mut fo) = (crate::ffb::Ffb::default(), crate::ffb::Ffb::default());
        let mut rows = Vec::new();
        let n = (secs / DT) as usize;
        for i in 0..n {
            let t = i as f32 * DT;
            let err = kmh / 3.6 - rc.speed();
            let st = deg.to_radians() * if ramp > 0.0 { (t / ramp).min(1.0) } else { 1.0 };
            let (thr, brk) = if kmh < 1.0 { (0.0, 1.0) } else { ((err * 0.5).clamp(0.0, 1.0), (-err * 0.3).clamp(0.0, 1.0)) };
            let c = Controls { throttle: thr, brake: brk, steer: st / rc.max_steer.max(1e-3), analog: true, handbrake: false, ..Default::default() };
            rc.step(&c, DT, &g, acc);
            let om = rc.c[0].omega + rc.c[1].omega;
            let sp = rc.speed() * 3.6;
            let og = fg.step(&game, rc.mz_front, rc.steer, om, sp, ffm, gam, DT);
            let oo = fo.step(&old, rc.mz_front, rc.steer, om, sp, ffm, 1.0, DT);
            rows.push((t, rc.speed() * rc.w.dot(rc.up()) / G, rc.mz_front, og, oo));
        }
        rows
    };
    println!("FFB steady corners (last 2 s mean; force = DirectInput / 10000, 1.0 = the wheel's peak):");
    println!("FFB  km/h  road deg  lat g   rack N m   game force  old force   game damper  old damper");
    for kmh in [60.0f32, 100.0, 160.0, 220.0] {
        for deg in [0.5f32, 1.0, 2.0, 3.0, 4.5, 6.0] {
            let rows = run(rc, &mut acc, kmh, deg, 0.3, 8.0);
            let tail = &rows[rows.len() - (2.0 / DT) as usize..];
            let k = tail.len() as f32;
            let m = |f: &dyn Fn(&(f32, f32, f32, crate::ffb::Out, crate::ffb::Out)) -> f32| tail.iter().map(f).sum::<f32>() / k;
            println!("FFB  {kmh:4.0}  {deg:8.1}  {:5.2}  {:9.0}  {:+10.3}  {:+9.3}  {:11.3}  {:10.3}{}",
                m(&|r| r.1), m(&|r| r.2), m(&|r| r.3.raw), m(&|r| r.4.raw), m(&|r| r.3.damper), m(&|r| r.4.damper),
                if m(&|r| r.3.raw).abs() > 1.0 { "  CLIPS" } else { "" });
        }
    }
    println!("FFB steering step at 160 km/h, 0 -> 3 deg road wheel in 0.1 s (where the rate damping acts):");
    println!("FFB     t   rack N m   game force  old force");
    let rows = run(rc, &mut acc, 160.0, 3.0, 0.1, 1.0);
    for r in rows.iter().step_by((0.025 / DT) as usize).take(16) {
        println!("FFB  {:5.3}  {:9.0}  {:+10.3}  {:+9.3}", r.0, r.2, r.3.raw, r.4.raw);
    }
    println!("FFB parked (brake held), steer held:");
    for deg in [2.0f32, 5.0, 10.0] {
        let rows = run(rc, &mut acc, 0.0, deg, 0.3, 3.0);
        let r = rows.last().unwrap();
        println!("FFB  parked {deg:4.1} deg: rack {:6.0} N m  game force {:+.3} damper {:.3}  old force {:+.3} damper {:.3}", r.2, r.3.raw, r.3.damper, r.4.raw, r.4.damper);
    }
}

/// `EVOVIEW_RIG_DONUT=1 --drive-test 1`: full lock and full throttle from
/// 30 km/h in 2nd on the pad, aids off - a donut, logged every 0.1 s.
pub fn donut_test(rc: &mut RigCar) {
    let g = flat_pad();
    let mut acc = 0.0;
    rc.abs_on = false;
    rc.tc_on = false;
    rc.place(Vec3::new(0.0, 0.0, -1800.0), 0.0);
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let kmh = env("EVOVIEW_DONUT_KMH", 30.0);
    rc.v = rc.forward() * (kmh / 3.6);
    for c in rc.c.iter_mut() {
        c.omega = kmh / 3.6 / 0.34;
    }
    rc.gear = env("EVOVIEW_DONUT_GEAR", 2.0) as i32;
    let secs: f32 = std::env::var("EVOVIEW_DONUT_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(8.0);
    let steer: f32 = std::env::var("EVOVIEW_DONUT_STEER").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let c = Controls { throttle: env("EVOVIEW_DONUT_THR", 1.0), steer, analog: true, ..Default::default() };
    for i in 0..(secs / 0.1) as usize {
        for _ in 0..(0.1 / DT) as usize {
            rc.step(&c, DT, &g, &mut acc);
        }
        let beta = rc.v.dot(rc.left()).atan2(rc.v.dot(rc.forward())).to_degrees();
        let yaw = rc.w.dot(rc.up()).to_degrees();
        let w = |k: usize| { let c = &rc.c[k]; format!("[om*r {:5.1} sr {:+5.2} a {:+6.1} fx {:+6.0} fy {:+6.0} fz {:5.0}]", c.omega * 0.34, c.sr, c.alpha.to_degrees(), c.fx, c.fy, c.fz) };
        println!("DONUT t {:4.1} v {:5.1} km/h beta {:+6.1} yaw {:+6.1} deg/s gear {} rpm {:5.0} | RR {} RL {}",
            (i + 1) as f32 * 0.1, rc.v.length() * 3.6, beta, yaw, rc.gear, rc.rpm, w(2), w(3));
    }
}

/// `EVOVIEW_RIG_GEARTEST=1 --drive-test 1`: the box and clutch on the pad -
/// paddle shifts, an H-pattern launch on the clutch, a stall, a missed gear.
pub fn gear_test(rc: &mut RigCar) {
    let g = flat_pad();
    let mut acc = 0.0;
    let mut run = |rc: &mut RigCar, c: Controls, secs: f32| {
        for _ in 0..(secs / DT).round() as usize {
            rc.step(&c, DT, &g, &mut acc);
        }
    };
    let state = |rc: &RigCar| format!("gear {:2} {:5.0} rpm clutch in {:.2} e {:.2}{} {:5.1} km/h{}{}", rc.gear, rc.engine_w * 60.0 / std::f32::consts::TAU,
        rc.clutch_in, rc.clutch_e, if rc.clutch_locked { " locked" } else { "" }, rc.speed() * 3.6, if rc.running { "" } else { " STALLED" },
        if rc.missed { " (gear refused)" } else { "" });
    let (kind, use_) = gearbox_kind(&rc.gb, &rc.cl, rc.assists);
    println!("GEAR {} | {kind} | {use_} | assists {:?}", rc.phys.screen_name, rc.assists);
    println!("GEAR box: up {} dn {} cut {} ms at gas {:.2}, DCT {}, H {}, window {} rpm gain {}, protect {} (+{} rpm) lockN {}, blip {} pts{} ({} s)",
        rc.gb.up_time, rc.gb.dn_time, rc.gb.cut_time, rc.gb.gas_cut, rc.gb.dual_clutch, rc.gb.shifter, rc.gb.shift_window, rc.gb.window_gain,
        !rc.gb.shifter && (!rc.gb.protect_msg || rc.gb.protect), rc.gb.protect_overrev, rc.gb.lock_n, rc.gb.blip.len(),
        if rc.gb.blip_electronic { " electronic" } else { "" }, rc.gb.blip.last().map_or(0.0, |p| p.0));
    println!("GEAR blip profile: {:?}", rc.gb.blip);
    println!("GEAR clutch: max {:.0} N m, auto forced {} {:.0}-{:.0} rpm, on changes {}, up profile {:?}, down profile {:?}",
        rc.cl.max_torque, rc.cl.auto_forced, rc.cl.auto_min_rpm, rc.cl.auto_max_rpm, rc.cl.auto_on_changes, rc.cl.up_profile, rc.cl.down_profile);
    // 1. paddles: launch and shift at 95 % of the limiter, the clutch pedal
    // not mapped / mapped and released / held to the floor / half
    let mut summary = Vec::new();
    for (label, pedal) in [("no pedal", None), ("pedal released", Some(0.0)), ("pedal HELD", Some(1.0)), ("pedal half", Some(0.5))] {
        rc.place(Vec3::new(0.0, 0.0, -1800.0), 0.0);
        rc.manual = false;
        rc.gear = 1;
        run(rc, Controls { clutch: pedal, analog: true, ..Default::default() }, 1.0);
        let rest = state(rc);
        let (mut t, mut t100, mut shifts) = (0.0f32, None, Vec::new());
        let mut log = Vec::new();
        while t < 12.0 && rc.speed() * 3.6 < 200.0 {
            // up at 95 % of the limiter on the wheels' side (not a slipping clutch's)
            let r = rc.tyre[1].map_or(0.33, |t| t.radius).max(0.1);
            let wheel_rpm = (rc.speed() / r * rc.gear_ratio(rc.gear)).abs() * 60.0 / std::f32::consts::TAU;
            let up = wheel_rpm > rc.limiter * 0.95 && rc.shift.is_none() && (rc.gear as usize) < rc.gears.len() && rc.running;
            let g0 = rc.gear;
            run(rc, Controls { throttle: 1.0, analog: true, shift_up: up, clutch: pedal, ..Default::default() }, if up { DT } else { 0.01 });
            if up && (rc.shift.is_some() || rc.gear != g0) {
                // time the change: neutral (or the DCT's blend) until the gear is in
                let mut dt_in = DT;
                let mut rpm_lo = rc.rpm;
                while rc.shift.is_some() && dt_in < 1.0 {
                    run(rc, Controls { throttle: 1.0, analog: true, clutch: pedal, ..Default::default() }, DT);
                    dt_in += DT;
                    rpm_lo = rpm_lo.min(rc.rpm);
                }
                t += dt_in;
                shifts.push(format!("{g0}->{} {:.0}ms", rc.gear, dt_in * 1000.0));
                log.push(format!("GEAR   t {t:5.2} {g0}->{}: {:.0} ms, rpm fell to {rpm_lo:.0}, {}", rc.gear, dt_in * 1000.0, state(rc)));
            }
            t += if up { DT } else { 0.01 };
            if t100.is_none() && rc.speed() * 3.6 >= 100.0 {
                t100 = Some(t);
            }
        }
        println!("GEAR paddles, {label}: at rest {rest}");
        for l in &log {
            println!("{l}");
        }
        println!("GEAR paddles, {label}: 0-100 {:?} s, {:.0} km/h at {t:.2} s, {}", t100.map(|x| (x * 100.0).round() / 100.0), rc.speed() * 3.6, state(rc));
        summary.push(format!("{label}: 0-100 {:?} s, top {:.0} km/h at {t:.1} s, stalled {}, shifts [{}]", t100.map(|x| (x * 100.0).round() / 100.0),
                             rc.speed() * 3.6, !rc.running, shifts.join(", ")));
        if pedal.is_none() {
            // downshifts under braking: the blip and the change time
            for _ in 0..3 {
                let g0 = rc.gear;
                let rpm0 = rc.rpm;
                run(rc, Controls { brake: 0.6, analog: true, shift_down: true, clutch: pedal, ..Default::default() }, DT);
                let mut dt_in = DT;
                let mut rpm_hi: f32 = rc.rpm;
                while rc.shift.is_some() && dt_in < 1.0 {
                    run(rc, Controls { brake: 0.6, analog: true, clutch: pedal, ..Default::default() }, DT);
                    dt_in += DT;
                    rpm_hi = rpm_hi.max(rc.rpm);
                }
                println!("GEAR   downshift {g0}->{}{} under braking: {:.0} ms, rpm {rpm0:.0} -> peak {rpm_hi:.0} in the change, {}", rc.gear, if rc.gear == g0 { " REFUSED (over-rev protection)" } else { "" }, dt_in * 1000.0, state(rc));
                run(rc, Controls { brake: 0.6, analog: true, clutch: pedal, ..Default::default() }, 0.3);
            }
        }
    }
    for s in &summary {
        println!("GEAR SUMMARY {s}");
    }
    if !rc.gb.shifter {
        println!("GEAR this car has no H gate (an H lever is ignored, paddles only)");
        return;
    }
    // 2. H pattern + clutch pedal
    let h = |gear: i32, clutch: f32, throttle: f32| Controls { h_gear: Some(gear), clutch: Some(clutch), throttle, analog: true, ..Default::default() };
    rc.place(Vec3::new(0.0, 0.0, -1800.0), 0.0);
    rc.manual = false;
    rc.gear = 0;
    run(rc, h(0, 0.0, 0.0), 1.0);
    println!("GEAR H: neutral idle {}", state(rc));
    run(rc, h(0, 1.0, 0.0), 0.3);
    run(rc, h(1, 1.0, 0.0), 0.3);
    println!("GEAR H: clutch in, 1st {}", state(rc));
    // ease the clutch out over 2 s at half gas
    for i in 0..20 {
        let p = 1.0 - (i as f32 + 1.0) / 20.0;
        run(rc, h(1, p, 0.5), 0.1);
    }
    run(rc, h(1, 0.0, 0.8), 2.0);
    println!("GEAR H: launched on the clutch {}", state(rc));
    run(rc, h(1, 1.0, 0.0), 0.15);
    run(rc, h(0, 1.0, 0.0), 0.1);
    run(rc, h(2, 1.0, 0.0), 0.1);
    run(rc, h(2, 0.0, 0.8), 1.0);
    println!("GEAR H: 1 -> 2 on the clutch {}", state(rc));
    run(rc, h(0, 0.0, 0.8), 0.1);
    run(rc, h(4, 0.0, 0.8), 0.2);
    println!("GEAR H: into 4th, NO clutch, on the gas: {}", state(rc));
    run(rc, h(4, 0.0, 0.0), 0.3);
    println!("GEAR H: 4th, NO clutch, off the gas (revs fall to match): {}", state(rc));
    run(rc, h(4, 1.0, 0.0), 0.2);
    run(rc, h(4, 0.0, 0.8), 0.5);
    println!("GEAR H: 4th after the clutch: {}", state(rc));
    // 3. launches without the clutch: the pedal mapped but left up
    for (label, pedal) in [("pedal mapped, left UP", Some(0.0)), ("no pedal mapped", None)] {
        rc.place(Vec3::new(0.0, 0.0, -1800.0), 0.0);
        rc.gear = 0;
        let c = |gear: i32, throttle: f32| Controls { h_gear: Some(gear), clutch: pedal, throttle, analog: true, ..Default::default() };
        run(rc, c(0, 0.0), 1.0);
        run(rc, c(1, 0.0), 0.5);
        let a = state(rc);
        run(rc, c(1, 0.5), 2.0);
        println!("GEAR H launch, {label}: into 1st {a} | 2 s at half gas {}", state(rc));
    }
    // 4. stall: stop in 1st, clutch dumped at idle
    rc.place(Vec3::new(0.0, 0.0, -1800.0), 0.0);
    rc.gear = 0;
    run(rc, h(0, 1.0, 0.0), 0.5);
    run(rc, h(1, 1.0, 0.0), 0.2);
    run(rc, h(1, 0.0, 0.0), 1.0);
    println!("GEAR H: clutch dumped at idle {}", state(rc));
    run(rc, h(0, 1.0, 0.0), 2.0);
    println!("GEAR H: neutral, clutch in, after 2 s {}", state(rc));
    // 5. reverse: stopped, clutch in, R, ease out
    rc.place(Vec3::new(0.0, 0.0, -1800.0), 0.0);
    rc.gear = 0;
    run(rc, h(0, 1.0, 0.0), 1.5);
    run(rc, h(-1, 1.0, 0.0), 0.3);
    println!("GEAR H: clutch in, R {}", state(rc));
    for i in 0..10 {
        run(rc, h(-1, 1.0 - (i as f32 + 1.0) / 10.0, 0.6), 0.12);
    }
    run(rc, h(-1, 0.0, 0.5), 2.0);
    println!("GEAR H: reversing {}", state(rc));
}

/// How a car's box and clutch behave for the player in EVO, from its data
/// and the player's assists (see request, gearbox, FUN_1422e5c90 notes): the box kind, and what the
/// clutch pedal does.
pub fn gearbox_kind(gb: &crate::carphys::Gearbox, cl: &crate::carphys::Clutch, assists: Assists) -> (&'static str, String) {
    let kind = match (gb.shifter, gb.dual_clutch) {
        (true, false) => "H-pattern",
        (true, true) => "H + DCT",
        (false, true) => "DCT paddles",
        (false, false) => "sequential",
    };
    let launch = if cl.auto_forced || assists.clutch_start {
        "auto launch, pedal ignored"
    } else {
        "pedal launch (stalls)"
    };
    let changes = if gb.shifter {
        if assists.clutch_changes { "auto changes" } else { "pedal/window changes" }
    } else if cl.auto_on_changes {
        "auto-clutch changes"
    } else {
        "clutchless changes"
    };
    (kind, format!("{launch}; {changes}"))
}

/// `--gearbox-table`: every car in the package -> box kind -> clutch use.
pub fn fleet_table(pkg: &std::path::Path, base: Option<&std::path::Path>) {
    let cars = match crate::load::list_package(pkg) {
        Ok(c) => c,
        Err(e) => return eprintln!("gearbox-table: {e}"),
    };
    let a = Assists::read();
    println!("GEARBOX assists: {a:?}");
    println!("GEARBOX {:28} {:12} {:>5} {:>5} {:>4} {:>5} {:>5} {:>4} {:>6} {:>4} {:5} {}", "car", "box", "up ms", "dn ms", "cut", "lvl", "win", "gain",
             "blip", "prot", "autoc", "clutch use (these assists)");
    let mut n = std::collections::BTreeMap::new();
    for car in cars {
        let Ok(mut vfs) = crate::load::package_vfs(pkg, &car, base) else { continue };
        let Some(p) = crate::carphys::Physics::read(&mut vfs) else {
            println!("GEARBOX {car:28} (no physics)");
            continue;
        };
        let (gb, cl) = (&p.gearbox, &p.clutch);
        let (kind, use_) = gearbox_kind(gb, cl, a);
        *n.entry(kind).or_insert(0) += 1;
        let blip = match (gb.blip.is_empty(), gb.blip_electronic) {
            (true, _) => "-".to_string(),
            (false, true) => format!("E{:.0}", gb.blip.last().map_or(0.0, |b| b.0) * 1000.0),
            (false, false) => format!("a{:.0}", gb.blip.last().map_or(0.0, |b| b.0) * 1000.0),
        };
        let prot = if gb.shifter { "-".to_string() } else if !gb.protect_msg || gb.protect { format!("{:.0}", gb.protect_overrev) } else { "off".into() };
        let autoc = format!("{}{}{}{}", if cl.auto_forced { "F" } else { "." }, if cl.auto_on_changes { "C" } else { "." },
                            if cl.up_profile.is_empty() { "." } else { "u" }, if cl.down_profile.is_empty() { "." } else { "d" });
        println!("GEARBOX {car:28} {kind:12} {:5.0} {:5.0} {:4.0} {:5.2} {:5.0} {:4.2} {blip:>6} {prot:>4} {autoc:5} {use_}",
                 gb.up_time, gb.dn_time, gb.cut_time, gb.gas_cut, gb.shift_window, gb.window_gain);
    }
    println!("GEARBOX totals: {n:?}");
}

/// EVOVIEW_KB_CLUTCH=0: no clutch mapped means no clutch (the game with its
/// auto-clutch assists off); default: the car launches itself then.
fn kb_clutch() -> bool {
    std::env::var("EVOVIEW_KB_CLUTCH").map_or(true, |v| v != "0")
}

/// EVOVIEW_PEDAL_DIPS=1: the old evoview rule, the pedal dips the
/// auto-clutch's value (min of the two).
fn pedal_dips() -> bool {
    std::env::var_os("EVOVIEW_PEDAL_DIPS").is_some()
}

/// EVOVIEW_DCT_NEUTRAL=1: a DCT drops to neutral through a change as a
/// sequential box does (the old evoview behaviour).
fn dct_neutral() -> bool {
    std::env::var_os("EVOVIEW_DCT_NEUTRAL").is_some()
}
