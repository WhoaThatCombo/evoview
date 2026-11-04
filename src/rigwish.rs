//! The rig car's double wishbones as AC EVO builds them in ODE
//! (EVOVIEW_RIG_ODE=links; SPEC_PHYS "Double wishbone in ODE"):
//!
//! - each hub one rigid body (a box of the unsprung mass, 0.2 x 0.6 x 0.6)
//!   at the wheel centre, held to the chassis by five DBall rods - the two
//!   legs of each wishbone to its ball joint, and the tie rod - with the
//!   game's joint ERP 0.6 / CFM 1e-7; no arm bodies;
//! - the rack slides each tie rod's body end sideways; toe and steer are
//!   rack travel, so bump steer and toe change come out of the linkage;
//! - the spring: at the wheel along the chassis' up axis when the data has
//!   no coilover, else between its body mount and a point on the virtual
//!   lower arm (fraction t along it, offset d off it, fixed at build).
//!
//! Points come in the CG frame (x left, y up, z forward); the data is for
//! the right-hand (-x) wheel, mirrored on x for the left (carphys keeps x
//! inboard-positive, the game negates x when the hub sits at +x).

use crate::carphys::{Geometry, SuspKind};
use crate::ode::joints::{JointKind, PARAM_CFM, PARAM_ERP};
use crate::ode::{BodyId, JointId, Mass, World};
use crate::rigode::{evo_world, CFM, ERP};
use glam::{Mat3, Quat, Vec3};

fn d3(v: Vec3) -> [f64; 3] {
    [v.x as f64, v.y as f64, v.z as f64]
}
fn f3(a: &[f64]) -> Vec3 {
    Vec3::new(a[0] as f32, a[1] as f32, a[2] as f32)
}

enum Spring {
    /// spring and damper at the wheel, along the chassis' up axis
    Wheel,
    /// coilover from `top` (body) to the virtual lower arm A..B (A on the
    /// body, B hub-local): bottom = A + t (B - A) + d perp
    Arm { top: Vec3, a: Vec3, b: Vec3, t: f32, d: f32, perp: Vec3, len: f32, rest: f32 },
}

struct Link {
    hub: BodyId,
    tie: JointId,
    /// the wheel centre at design (body frame); the tie rod's body end
    href: Vec3,
    tie_body: Vec3,
    /// rack metres per radian of wheel yaw about the kingpin, at design
    rack_per_rad: f32,
    /// ball joints, hub-local (upper, lower)
    ball_up: Vec3,
    ball_low: Vec3,
    spring: Spring,
    x_static: f32,
    mr: f32,
}

/// One corner as the tyre and spring code need it this substep.
#[derive(Clone, Copy, Debug)]
pub struct Kin {
    pub hub: Vec3,
    pub rot: Mat3,
    pub v: Vec3,
    pub w: Vec3,
    /// the wheel centre's design point on the chassis, in the world
    pub href: Vec3,
    /// rig units: wheel travel (compression) and its rate
    pub travel: f32,
    pub rate: f32,
    /// the spring's line (top on the chassis, bottom on the hub side), its
    /// unit axis bottom -> top, and the share of the bottom force the hub
    /// takes (the rest goes into the arm's body pivot `pivot`)
    pub top: Vec3,
    pub bottom: Vec3,
    pub axis: Vec3,
    pub hub_share: f32,
    pub pivot: Vec3,
    /// spring force per unit of the rig's (wheel-rate) force
    pub per_wheel: f32,
    /// steering axis: lower ball (world) and unit direction up it
    pub kp: (Vec3, Vec3),
}

pub struct Links {
    w: World,
    body: BodyId,
    link: [Link; 4],
    /// the chassis state last handed back (to see the rig move it itself)
    last: (Vec3, Quat, Vec3, Vec3),
}

impl Links {
    pub fn enabled() -> bool {
        std::env::var("EVOVIEW_RIG_ODE").map_or(false, |v| v == "links")
    }

    /// None unless both axles are double wishbones with tie rods.
    #[allow(clippy::too_many_arguments)]
    pub fn new(mass: f32, inertia: Vec3, m_unsprung: [f32; 2], geo: [&Geometry; 2], anchor: &[Vec3; 4], cg_local: Vec3, x_static: [f32; 4], mr: [f32; 2], radius: [f32; 2]) -> Option<Links> {
        for g in geo {
            if g.kind != SuspKind::DoubleWishbone || g.low.is_none() || g.up.is_none() || g.steer.is_none() {
                return None;
            }
        }
        let mut w = evo_world();
        let body = w.body_create();
        let mut m = Mass { mass: mass as f64, ..Default::default() };
        m.i[0] = inertia.x as f64;
        m.i[5] = inertia.y as f64;
        m.i[10] = inertia.z as f64;
        w.body_set_mass(body, &m);
        let link = [0usize, 1, 2, 3].map(|k| {
            let g = geo[k / 2];
            // FR FL RR RL: the left wheels (odd) mirror the data's x
            let mir = |p: Vec3| if k % 2 == 1 { Vec3::new(-p.x, p.y, p.z) } else { p };
            let href = anchor[k] - cg_local;
            let (u0, u1, ub) = g.up.unwrap();
            let (l0, l1, lb) = g.low.unwrap();
            let (t0, t1) = g.steer.unwrap();
            let pt = |p: Vec3| mir(p) + href;
            let hub = w.body_create();
            w.body_set_mass(hub, &Mass::box_total(m_unsprung[k / 2] as f64, 0.2, 0.6, 0.6));
            let h = d3(href);
            w.body_set_position(hub, h[0], h[1], h[2]);
            let rod = |w: &mut World, a: Vec3, b: Vec3| {
                let j = w.joint_create_dball(None);
                w.joint_attach(j, Some(body), Some(hub));
                let (a, b) = (d3(a), d3(b));
                w.joint_set_dball_anchor1(j, a[0], a[1], a[2]);
                w.joint_set_dball_anchor2(j, b[0], b[1], b[2]);
                w.joint_set_erp_cfm_param(j, PARAM_ERP, ERP);
                w.joint_set_erp_cfm_param(j, PARAM_CFM, CFM);
                j
            };
            // the game's order: upper rear/front legs, lower, tie rod
            rod(&mut w, pt(u1), pt(ub));
            rod(&mut w, pt(u0), pt(ub));
            rod(&mut w, pt(l1), pt(lb));
            rod(&mut w, pt(l0), pt(lb));
            let tie = rod(&mut w, pt(t0), pt(t1));
            // rack per radian: turning the hub by dθ about the kingpin K..
            // moves the hub end by dθ k x (P8 - K); the body end slides on x
            // to keep the rod's length
            let (kb, ku) = (pt(lb), pt(ub));
            let kd = (ku - kb).normalize_or_zero();
            let (p6, p8) = (pt(t0), pt(t1));
            let rod_v = p8 - p6;
            let rack_per_rad = if rod_v.x.abs() > 1e-4 { rod_v.dot(kd.cross(p8 - kb)) / rod_v.x } else { 0.0 };
            let spring = match g.coil {
                None => Spring::Wheel,
                Some((top, bot)) => {
                    let a = 0.5 * (pt(l0) + pt(l1));
                    let b = pt(lb);
                    let p9 = pt(bot);
                    let ab = b - a;
                    let len = ab.length().max(1e-4);
                    let t = (p9 - a).dot(ab) / (len * len);
                    let r = (p9 - a) - ab * t;
                    let d = r.length();
                    let (d, perp) = if d <= 1e-5 { (0.0, Vec3::ZERO) } else { (d, r / d) };
                    Spring::Arm { top: pt(top), a, b: mir(lb), t, d, perp, len, rest: (pt(top) - p9).length() }
                }
            };
            Link { hub, tie, href, tie_body: p6, rack_per_rad, ball_up: mir(ub), ball_low: mir(lb), spring, x_static: x_static[k], mr: mr[k / 2].max(1e-3) }
        });
        if std::env::var_os("EVOVIEW_RIG_LINKLOG").is_some() {
            for ax in 0..2 {
                println!("LINK geometry {}: roll centre {:.3} m over the ground, instant centre {:?}", ["front", "rear"][ax], roll_centre(geo[ax], anchor[ax * 2] - cg_local, radius[ax]).0, roll_centre(geo[ax], anchor[ax * 2] - cg_local, radius[ax]).1);
            }
        }
        Some(Links { w, body, link, last: (Vec3::ZERO, Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO) })
    }

    /// At rest at (cg, q): the hubs back where they were built, nothing
    /// moving (a reset to the track)
    pub fn reset(&mut self, cg: Vec3, q: Quat) {
        for k in 0..4 {
            let (hub, href) = (self.link[k].hub, self.link[k].href);
            self.put(hub, cg + q * href, q, Vec3::ZERO, Vec3::ZERO);
        }
        self.put(self.body, cg, q, Vec3::ZERO, Vec3::ZERO);
        self.last = (cg, q, Vec3::ZERO, Vec3::ZERO);
    }

    fn get(&self, b: BodyId) -> (Vec3, Quat, Vec3, Vec3) {
        let b = &self.w.bodies[b];
        (f3(&b.pos), Quat::from_xyzw(b.q[1] as f32, b.q[2] as f32, b.q[3] as f32, b.q[0] as f32).normalize(), f3(&b.lvel), f3(&b.avel))
    }
    fn put(&mut self, b: BodyId, p: Vec3, q: Quat, v: Vec3, wv: Vec3) {
        let (p, v, wv) = (d3(p), d3(v), d3(wv));
        self.w.body_set_position(b, p[0], p[1], p[2]);
        self.w.body_set_quaternion(b, &[q.w as f64, q.x as f64, q.y as f64, q.z as f64]);
        self.w.body_set_linear_vel(b, v[0], v[1], v[2]);
        self.w.body_set_angular_vel(b, wv[0], wv[1], wv[2]);
    }

    /// The rig's chassis state: when the rig moved it (placing the car, an
    /// impulse off a wall), carry the hubs along rigidly.
    pub fn sync(&mut self, cg: Vec3, q: Quat, v: Vec3, w: Vec3) {
        let (c0, q0, v0, w0) = self.last;
        if (cg, q, v, w) == (c0, q0, v0, w0) {
            return;
        }
        let dq = q * q0.inverse();
        for k in 0..4 {
            let (p, hq, hv, hw) = self.get(self.link[k].hub);
            let rel = dq * (p - c0);
            let rv = dq * (hv - (v0 + w0.cross(p - c0)));
            let p1 = cg + rel;
            self.put(self.link[k].hub, p1, (dq * hq).normalize(), v + w.cross(rel) + rv, w + dq * (hw - w0));
        }
        self.put(self.body, cg, q, v, w);
        self.last = (cg, q, v, w);
    }

    /// Each wheel's yaw (steer + toe), rad, as rack travel.
    pub fn set_wheel_angles(&mut self, ang: [f32; 4]) {
        for k in 0..4 {
            let l = &self.link[k];
            let at = l.tie_body + Vec3::X * (ang[k] * l.rack_per_rad);
            if let JointKind::DBall(d) = &mut self.w.joints[l.tie].kind {
                d.anchor1[0] = at.x as f64;
                d.anchor1[1] = at.y as f64;
                d.anchor1[2] = at.z as f64;
            }
        }
    }

    pub fn kin(&self) -> [Kin; 4] {
        let (cg, q, v, w) = self.get(self.body);
        let r = Mat3::from_quat(q);
        let up = r * Vec3::Y;
        let pv = |p: Vec3, c: Vec3, v: Vec3, w: Vec3| v + w.cross(p - c);
        [0usize, 1, 2, 3].map(|k| {
            let l = &self.link[k];
            let (hp, hq, hv, hw) = self.get(l.hub);
            let hr = Mat3::from_quat(hq);
            let href = cg + r * l.href;
            let kb = hp + hr * l.ball_low;
            let kd = (hr * (l.ball_up - l.ball_low)).normalize_or_zero();
            let mut kn = Kin {
                hub: hp,
                rot: hr,
                v: hv,
                w: hw,
                href,
                travel: 0.0,
                rate: 0.0,
                top: href,
                bottom: hp,
                axis: up,
                hub_share: 1.0,
                pivot: href,
                per_wheel: 1.0,
                kp: (kb, kd),
            };
            match l.spring {
                Spring::Wheel => {
                    kn.travel = l.x_static + (hp - href).dot(up);
                    kn.rate = (hv - pv(href, cg, v, w)).dot(up);
                }
                Spring::Arm { top, a, b, t, d, perp, len, rest } => {
                    let aw = cg + r * a;
                    let bw = hp + hr * b;
                    let ab = bw - aw;
                    let u = ab.normalize_or_zero();
                    let mut pw = r * perp;
                    pw -= u * pw.dot(u);
                    let pw = pw.normalize_or_zero();
                    let bot = aw + ab * t + pw * d;
                    let (va, vb) = (pv(aw, cg, v, w), pv(bw, hp, hv, hw));
                    let om = ab.cross(vb - va) / (len * len);
                    let vbot = va + om.cross(bot - aw);
                    let tw = cg + r * top;
                    let ax = (tw - bot).normalize_or_zero();
                    let vt = pv(tw, cg, v, w);
                    kn.travel = l.x_static + (rest - (tw - bot).length()) / l.mr;
                    kn.rate = (vbot - vt).dot(ax) / l.mr;
                    kn.top = tw;
                    kn.bottom = bw;
                    kn.axis = ax;
                    kn.hub_share = t;
                    kn.pivot = aw;
                    kn.per_wheel = 1.0 / l.mr;
                }
            }
            kn
        })
    }

    /// One substep with the chassis' and hubs' forces (torques about each
    /// body's centre); returns the chassis' new state.
    pub fn step(&mut self, f_body: Vec3, t_body: Vec3, f_hub: &[Vec3; 4], t_hub: &[Vec3; 4], dt: f32) -> (Vec3, Quat, Vec3, Vec3) {
        let add = |w: &mut World, b: BodyId, f: Vec3, t: Vec3| {
            let (f, t) = (d3(f), d3(t));
            w.body_add_force(b, f[0], f[1], f[2]);
            w.body_add_torque(b, t[0], t[1], t[2]);
        };
        add(&mut self.w, self.body, f_body, t_body);
        for k in 0..4 {
            add(&mut self.w, self.link[k].hub, f_hub[k], t_hub[k]);
        }
        self.w.quick_step(dt as f64);
        self.last = self.get(self.body);
        self.last
    }
}

/// Front-view roll centre of the right-hand (-x) corner at design: the arms
/// projected on x/y, their instant centre, the line from the contact patch
/// through it crossing the car's centre line. (height over the ground,
/// instant centre in the CG frame)
pub fn roll_centre(g: &Geometry, href: Vec3, radius: f32) -> (f32, glam::Vec2) {
    let (Some((u0, u1, ub)), Some((l0, l1, lb))) = (g.up, g.low) else { return (f32::NAN, glam::Vec2::NAN) };
    let p = |v: Vec3| glam::Vec2::new(v.x + href.x, v.y + href.y);
    let (ua, ubp) = (p(0.5 * (u0 + u1)), p(ub));
    let (la, lbp) = (p(0.5 * (l0 + l1)), p(lb));
    // intersect lines ua..ubp and la..lbp
    let (d1, d2) = (ubp - ua, lbp - la);
    let den = d1.x * d2.y - d1.y * d2.x;
    let ic = if den.abs() < 1e-9 { glam::Vec2::new(f32::INFINITY, f32::INFINITY) } else {
        let t = ((la.x - ua.x) * d2.y - (la.y - ua.y) * d2.x) / den;
        ua + d1 * t
    };
    let cp = glam::Vec2::new(href.x, href.y - radius);
    let s = (0.0 - cp.x) / (ic.x - cp.x);
    let y = cp.y + (ic.y - cp.y) * s;
    (y - cp.y, ic)
}
