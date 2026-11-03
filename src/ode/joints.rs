//! ODE's joints (joints/*.cpp): the ones AC EVO's physics uses (ball,
//! hinge, slider, fixed, distance ball, contact, angular motor).
//!
//! getInfo2 writes rows of 16 reals, the QuickStep Jacobian layout:
//! J1 (lin xyz, ang xyz) | rhs cfm | J2 (lin, ang) | lo hi.

use super::math::*;
use super::world::{Body, BodyId, WorldParams};

pub const ROW: usize = 16;
pub const J1: usize = 0;
pub const RHS: usize = 6;
pub const CFM: usize = 7;
pub const J2: usize = 8;
pub const LO: usize = 14;
pub const HI: usize = 15;
/// angular part of a J block
pub const JA: usize = 3;

// dParam* (one group; AMotor adds 0x100 per axis)
pub const PARAM_LO_STOP: i32 = 0;
pub const PARAM_HI_STOP: i32 = 1;
pub const PARAM_VEL: i32 = 2;
pub const PARAM_LO_VEL: i32 = 3;
pub const PARAM_HI_VEL: i32 = 4;
pub const PARAM_FMAX: i32 = 5;
pub const PARAM_FUDGE_FACTOR: i32 = 6;
pub const PARAM_BOUNCE: i32 = 7;
pub const PARAM_CFM: i32 = 8;
pub const PARAM_STOP_ERP: i32 = 9;
pub const PARAM_STOP_CFM: i32 = 10;
pub const PARAM_SUSPENSION_ERP: i32 = 11;
pub const PARAM_SUSPENSION_CFM: i32 = 12;
pub const PARAM_ERP: i32 = 13;
pub const PARAM_GROUP: i32 = 0x100;

/// dJointFeedback: the force/torque the joint applied to each body
#[derive(Clone, Copy, Debug, Default)]
pub struct Feedback {
    pub f1: V3,
    pub t1: V3,
    pub f2: V3,
    pub t2: V3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointType {
    Ball,
    Hinge,
    Slider,
    Contact,
    Fixed,
    AMotor,
    DBall,
}

/// dxJointLimitMotor
#[derive(Clone, Copy, Debug, Default)]
pub struct LimitMotor {
    pub vel: Real,
    pub fmax: Real,
    pub lostop: Real,
    pub histop: Real,
    pub fudge_factor: Real,
    pub normal_cfm: Real,
    pub stop_erp: Real,
    pub stop_cfm: Real,
    pub bounce: Real,
    pub limit: i32,
    pub limit_err: Real,
}
impl LimitMotor {
    pub fn new(w: &WorldParams) -> LimitMotor {
        LimitMotor {
            vel: 0.0,
            fmax: 0.0,
            lostop: -INFINITY,
            histop: INFINITY,
            fudge_factor: 1.0,
            normal_cfm: w.cfm,
            stop_erp: w.erp,
            stop_cfm: w.cfm,
            bounce: 0.0,
            limit: 0,
            limit_err: 0.0,
        }
    }
    pub fn set(&mut self, num: i32, value: Real) {
        match num {
            PARAM_LO_STOP => self.lostop = value,
            PARAM_HI_STOP => self.histop = value,
            PARAM_VEL => self.vel = value,
            PARAM_FMAX => {
                if value >= 0.0 {
                    self.fmax = value
                }
            }
            PARAM_FUDGE_FACTOR => {
                if (0.0..=1.0).contains(&value) {
                    self.fudge_factor = value
                }
            }
            PARAM_BOUNCE => self.bounce = value,
            PARAM_CFM => self.normal_cfm = value,
            PARAM_STOP_ERP => self.stop_erp = value,
            PARAM_STOP_CFM => self.stop_cfm = value,
            _ => {}
        }
    }
    pub fn get(&self, num: i32) -> Real {
        match num {
            PARAM_LO_STOP => self.lostop,
            PARAM_HI_STOP => self.histop,
            PARAM_VEL => self.vel,
            PARAM_FMAX => self.fmax,
            PARAM_FUDGE_FACTOR => self.fudge_factor,
            PARAM_BOUNCE => self.bounce,
            PARAM_CFM => self.normal_cfm,
            PARAM_STOP_ERP => self.stop_erp,
            PARAM_STOP_CFM => self.stop_cfm,
            _ => 0.0,
        }
    }
    pub fn test_rotational_limit(&mut self, angle: Real) -> bool {
        if angle <= self.lostop {
            self.limit = 1;
            self.limit_err = angle - self.lostop;
            true
        } else if angle >= self.histop {
            self.limit = 2;
            self.limit_err = angle - self.histop;
            true
        } else {
            self.limit = 0;
            false
        }
    }
    /// addLimot: one powered and/or limited row at `rows[0..ROW]`
    pub fn add_limot(&self, node: [Option<BodyId>; 2], bodies: &mut [Body], fps: Real, rows: &mut [Real], ax1: &[Real], rotational: bool) -> bool {
        let mut powered = self.fmax > 0.0;
        if !(powered || self.limit != 0) {
            return false;
        }
        let b0 = node[0].unwrap();
        let (u1, u2) = if rotational { (J1 + JA, J2 + JA) } else { (J1, J2) };
        rows[u1..u1 + 3].copy_from_slice(&ax1[..3]);
        if node[1].is_some() {
            rows[u2] = -ax1[0];
            rows[u2 + 1] = -ax1[1];
            rows[u2 + 2] = -ax1[2];
        }
        let mut ltd: V3 = [0.0; 4];
        if !rotational {
            if let Some(b1) = node[1] {
                let (p0, p1) = (bodies[b0].pos, bodies[b1].pos);
                let c = [0.5 * (p1[0] - p0[0]), 0.5 * (p1[1] - p0[1]), 0.5 * (p1[2] - p0[2]), 0.0];
                ltd = cross(&c, ax1);
                rows[J1 + JA..J1 + JA + 3].copy_from_slice(&ltd[..3]);
                rows[J2 + JA..J2 + JA + 3].copy_from_slice(&ltd[..3]);
            }
        }
        if self.limit != 0 && self.lostop == self.histop {
            powered = false;
        }
        if powered {
            rows[CFM] = self.normal_cfm;
            if self.limit == 0 {
                rows[RHS] = self.vel;
                rows[LO] = -self.fmax;
                rows[HI] = self.fmax;
            } else {
                let mut fm = self.fmax;
                if self.vel > 0.0 || (self.vel == 0.0 && self.limit == 2) {
                    fm = -fm;
                }
                if (self.limit == 1 && self.vel > 0.0) || (self.limit == 2 && self.vel < 0.0) {
                    fm *= self.fudge_factor;
                }
                let (f0, f1, f2) = (fm * ax1[0], fm * ax1[1], fm * ax1[2]);
                if rotational {
                    if let Some(b1) = node[1] {
                        add_torque(&mut bodies[b1], f0, f1, f2);
                    }
                    add_torque(&mut bodies[b0], -f0, -f1, -f2);
                } else {
                    if let Some(b1) = node[1] {
                        let (n0, n1, n2) = (-fm * ltd[0], -fm * ltd[1], -fm * ltd[2]);
                        add_torque(&mut bodies[b0], n0, n1, n2);
                        add_torque(&mut bodies[b1], n0, n1, n2);
                        add_force(&mut bodies[b1], f0, f1, f2);
                    }
                    add_force(&mut bodies[b0], -f0, -f1, -f2);
                }
            }
        }
        if self.limit != 0 {
            let k = fps * self.stop_erp;
            rows[RHS] = -k * self.limit_err;
            rows[CFM] = self.stop_cfm;
            if self.lostop == self.histop {
                rows[LO] = -INFINITY;
                rows[HI] = INFINITY;
            } else {
                if self.limit == 1 {
                    rows[LO] = 0.0;
                    rows[HI] = INFINITY;
                } else {
                    rows[LO] = -INFINITY;
                    rows[HI] = 0.0;
                }
                if self.bounce > 0.0 {
                    let mut vel;
                    if rotational {
                        vel = dot3(&bodies[b0].avel, ax1);
                        if let Some(b1) = node[1] {
                            vel -= dot3(&bodies[b1].avel, ax1);
                        }
                    } else {
                        vel = dot3(&bodies[b0].lvel, ax1);
                        if let Some(b1) = node[1] {
                            vel -= dot3(&bodies[b1].lvel, ax1);
                        }
                    }
                    if self.limit == 1 {
                        if vel < 0.0 {
                            let newc = -self.bounce * vel;
                            if newc > rows[RHS] {
                                rows[RHS] = newc;
                            }
                        }
                    } else if vel > 0.0 {
                        let newc = -self.bounce * vel;
                        if newc < rows[RHS] {
                            rows[RHS] = newc;
                        }
                    }
                }
            }
        }
        true
    }
}

fn add_force(b: &mut Body, x: Real, y: Real, z: Real) {
    b.facc[0] += x;
    b.facc[1] += y;
    b.facc[2] += z;
}
fn add_torque(b: &mut Body, x: Real, y: Real, z: Real) {
    b.tacc[0] += x;
    b.tacc[1] += y;
    b.tacc[2] += z;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Ball {
    pub anchor1: V3,
    pub anchor2: V3,
    pub erp: Real,
    pub cfm: Real,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Hinge {
    pub anchor1: V3,
    pub anchor2: V3,
    pub axis1: V3,
    pub axis2: V3,
    pub qrel: Quat,
    pub limot: LimitMotor,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Slider {
    pub axis1: V3,
    pub qrel: Quat,
    pub offset: V3,
    pub limot: LimitMotor,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Fixed {
    pub qrel: Quat,
    pub offset: V3,
    pub erp: Real,
    pub cfm: Real,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct DBall {
    pub anchor1: V3,
    pub anchor2: V3,
    pub target_distance: Real,
    pub erp: Real,
    pub cfm: Real,
}

// dSurfaceParameters.mode
pub const CONTACT_MU2: i32 = 0x001;
pub const CONTACT_AXIS_DEP: i32 = 0x001;
pub const CONTACT_FDIR1: i32 = 0x002;
pub const CONTACT_BOUNCE: i32 = 0x004;
pub const CONTACT_SOFT_ERP: i32 = 0x008;
pub const CONTACT_SOFT_CFM: i32 = 0x010;
pub const CONTACT_MOTION1: i32 = 0x020;
pub const CONTACT_MOTION2: i32 = 0x040;
pub const CONTACT_MOTION_N: i32 = 0x080;
pub const CONTACT_SLIP1: i32 = 0x100;
pub const CONTACT_SLIP2: i32 = 0x200;
pub const CONTACT_ROLLING: i32 = 0x400;
pub const CONTACT_APPROX1_1: i32 = 0x1000;
pub const CONTACT_APPROX1_2: i32 = 0x2000;
pub const CONTACT_APPROX1_N: i32 = 0x4000;
pub const CONTACT_APPROX1: i32 = 0x7000;

/// dSurfaceParameters
#[derive(Clone, Copy, Debug, Default)]
pub struct Surface {
    pub mode: i32,
    pub mu: Real,
    pub mu2: Real,
    pub rho: Real,
    pub rho2: Real,
    pub rho_n: Real,
    pub bounce: Real,
    pub bounce_vel: Real,
    pub soft_erp: Real,
    pub soft_cfm: Real,
    pub motion1: Real,
    pub motion2: Real,
    pub motion_n: Real,
    pub slip1: Real,
    pub slip2: Real,
}
/// dContactGeom (the geoms themselves are the collision layer's)
#[derive(Clone, Copy, Debug, Default)]
pub struct ContactGeom {
    pub pos: V3,
    pub normal: V3,
    pub depth: Real,
    /// the two geoms (collision.rs GeomId)
    pub g1: usize,
    pub g2: usize,
    pub side1: i32,
    pub side2: i32,
}
/// dContact
#[derive(Clone, Copy, Debug, Default)]
pub struct Contact {
    pub surface: Surface,
    pub geom: ContactGeom,
    pub fdir1: V3,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct ContactJoint {
    pub contact: Contact,
    pub the_m: i32,
}

pub const AMOTOR_USER: i32 = 0;
pub const AMOTOR_EULER: i32 = 1;
/// dJointBodyRelativity
pub const REL_GLOBAL: i32 = 0;
pub const REL_BODY1: i32 = 1;
pub const REL_BODY2: i32 = 2;

#[derive(Clone, Copy, Debug, Default)]
pub struct AMotor {
    pub mode: i32,
    pub num: usize,
    pub rel: [i32; 3],
    pub axis: [V3; 3],
    pub references: [V3; 2],
    pub angle: [Real; 3],
    pub limot: [LimitMotor; 4],
}

#[derive(Clone, Copy, Debug)]
pub enum JointKind {
    Ball(Ball),
    Hinge(Hinge),
    Slider(Slider),
    Fixed(Fixed),
    DBall(DBall),
    Contact(ContactJoint),
    AMotor(AMotor),
}
impl JointKind {
    pub fn type_of(&self) -> JointType {
        match self {
            JointKind::Ball(_) => JointType::Ball,
            JointKind::Hinge(_) => JointType::Hinge,
            JointKind::Slider(_) => JointType::Slider,
            JointKind::Fixed(_) => JointType::Fixed,
            JointKind::DBall(_) => JointType::DBall,
            JointKind::Contact(_) => JointType::Contact,
            JointKind::AMotor(_) => JointType::AMotor,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Joint {
    pub node: [Option<BodyId>; 2],
    pub reverse: bool,
    pub disabled: bool,
    pub in_group: bool,
    pub feedback: Option<Feedback>,
    pub lambda: [Real; 6],
    pub kind: JointKind,
    pub(crate) tag: i32,
    pub(crate) alive: bool,
}

/// Info1: rows used (m) and unbounded rows (nub)
#[derive(Clone, Copy, Debug, Default)]
pub struct Info1 {
    pub m: u32,
    pub nub: u32,
}

impl Joint {
    pub(crate) fn new(kind: JointKind, in_group: bool) -> Joint {
        Joint { node: [None, None], reverse: false, disabled: false, in_group, feedback: None, lambda: [0.0; 6], kind, tag: 0, alive: true }
    }

    /// setRelativeValues (on attach)
    pub(crate) fn set_relative_values(&mut self, bodies: &[Body]) {
        let node = self.node;
        let reverse = self.reverse;
        match &mut self.kind {
            JointKind::Ball(j) => {
                let a = ball_anchor(node, reverse, bodies, &j.anchor1, &j.anchor2);
                set_anchors(node, bodies, a[0], a[1], a[2], &mut j.anchor1, &mut j.anchor2);
            }
            JointKind::Hinge(j) => {
                let a = ball_anchor(node, reverse, bodies, &j.anchor1, &j.anchor2);
                set_anchors(node, bodies, a[0], a[1], a[2], &mut j.anchor1, &mut j.anchor2);
                let ax = get_axis(node, bodies, &j.axis1);
                set_axes(node, bodies, ax[0], ax[1], ax[2], Some(&mut j.axis1), Some(&mut j.axis2));
                j.qrel = initial_relative_rotation(node, bodies, j.qrel);
            }
            JointKind::Slider(j) => {
                j.offset = slider_offset(node, bodies, j.offset);
                j.qrel = initial_relative_rotation(node, bodies, j.qrel);
            }
            JointKind::DBall(j) => j.target_distance = dball_distance(node, bodies, j),
            _ => {}
        }
    }

    pub(crate) fn info1(&mut self, bodies: &[Body]) -> Info1 {
        let node = self.node;
        let reverse = self.reverse;
        match &mut self.kind {
            JointKind::Ball(_) => Info1 { m: 3, nub: 3 },
            JointKind::Fixed(_) => Info1 { m: 6, nub: 6 },
            JointKind::DBall(_) => Info1 { m: 1, nub: 1 },
            JointKind::Hinge(j) => {
                let mut m = if j.limot.fmax > 0.0 { 6 } else { 5 };
                if (j.limot.lostop >= -std::f64::consts::PI || j.limot.histop <= std::f64::consts::PI) && j.limot.lostop <= j.limot.histop {
                    let angle = hinge_angle(bodies, node, &j.axis1, &j.qrel);
                    if j.limot.test_rotational_limit(angle) {
                        m = 6;
                    }
                }
                Info1 { m, nub: 5 }
            }
            JointKind::Slider(j) => {
                let mut m = if j.limot.fmax > 0.0 { 6 } else { 5 };
                j.limot.limit = 0;
                if (j.limot.lostop > -INFINITY || j.limot.histop < INFINITY) && j.limot.lostop <= j.limot.histop {
                    let pos = slider_position(bodies, node, reverse, j);
                    if pos <= j.limot.lostop {
                        j.limot.limit = 1;
                        j.limot.limit_err = pos - j.limot.lostop;
                        m = 6;
                    } else if pos >= j.limot.histop {
                        j.limot.limit = 2;
                        j.limot.limit_err = pos - j.limot.histop;
                        m = 6;
                    }
                }
                Info1 { m, nub: 5 }
            }
            JointKind::Contact(c) => {
                let s = &mut c.contact.surface;
                let mut m = 1;
                let mut nub = 0;
                let roll = s.mode & CONTACT_ROLLING != 0;
                if s.mu < 0.0 {
                    s.mu = 0.0;
                }
                if s.mode & CONTACT_AXIS_DEP != 0 {
                    if s.mu2 < 0.0 {
                        s.mu2 = 0.0;
                    }
                    if s.mu > 0.0 {
                        m += 1;
                    }
                    if s.mu2 > 0.0 {
                        m += 1;
                    }
                    if s.mu == INFINITY {
                        nub += 1;
                    }
                    if s.mu2 == INFINITY {
                        nub += 1;
                    }
                    if roll {
                        if s.rho < 0.0 {
                            s.rho = 0.0;
                        } else {
                            m += 1;
                        }
                        if s.rho2 < 0.0 {
                            s.rho2 = 0.0;
                        } else {
                            m += 1;
                        }
                        if s.rho_n < 0.0 {
                            s.rho_n = 0.0;
                        } else {
                            m += 1;
                        }
                        if s.rho == INFINITY {
                            nub += 1;
                        }
                        if s.rho2 == INFINITY {
                            nub += 1;
                        }
                        if s.rho_n == INFINITY {
                            nub += 1;
                        }
                    }
                } else {
                    if s.mu > 0.0 {
                        m += 2;
                    }
                    if s.mu == INFINITY {
                        nub += 2;
                    }
                    if roll {
                        if s.rho < 0.0 {
                            s.rho = 0.0;
                        } else {
                            m += 3;
                        }
                        if s.rho == INFINITY {
                            nub += 3;
                        }
                    }
                }
                c.the_m = m;
                Info1 { m: m as u32, nub }
            }
            JointKind::AMotor(a) => {
                let mut m = 0;
                if a.mode == AMOTOR_EULER {
                    let ax = amotor_global_axes(a, node, reverse, bodies);
                    amotor_euler_angles(a, node, reverse, bodies, &ax);
                }
                for i in 0..a.num {
                    let ang = a.angle[i];
                    if a.limot[i].test_rotational_limit(ang) || a.limot[i].fmax > 0.0 {
                        m += 1;
                    }
                }
                Info1 { m, nub: 0 }
            }
        }
    }

    /// getInfo2: fill this joint's rows (`rows` starts at its first row,
    /// already cleared by the stepper); `findex` is relative to the joint.
    pub(crate) fn info2(&mut self, bodies: &mut [Body], w: &WorldParams, fps: Real, erp: Real, rows: &mut [Real], findex: &mut [i32]) {
        let node = self.node;
        let reverse = self.reverse;
        match &mut self.kind {
            JointKind::Ball(j) => {
                for r in 0..3 {
                    rows[r * ROW + CFM] = j.cfm;
                }
                set_ball(node, bodies, fps, j.erp, rows, &j.anchor1, &j.anchor2);
            }
            JointKind::Hinge(j) => {
                set_ball(node, bodies, fps, erp, rows, &j.anchor1, &j.anchor2);
                let b0 = node[0].unwrap();
                let ax1 = mul0_331(&bodies[b0].r, &j.axis1);
                let (p, q) = plane_space(&ax1);
                let has1 = node[1].is_some();
                let r = 3 * ROW;
                rows[r + J1 + JA..r + J1 + JA + 3].copy_from_slice(&p[..3]);
                if has1 {
                    put3(rows, r + J2 + JA, &neg3(&p));
                }
                let r = 4 * ROW;
                rows[r + J1 + JA..r + J1 + JA + 3].copy_from_slice(&q[..3]);
                if has1 {
                    put3(rows, r + J2 + JA, &neg3(&q));
                }
                let b = if let Some(b1) = node[1] {
                    let ax2 = mul0_331(&bodies[b1].r, &j.axis2);
                    cross(&ax1, &ax2)
                } else {
                    cross(&ax1, &j.axis2)
                };
                let k = fps * erp;
                rows[3 * ROW + RHS] = k * dot3(&b, &p);
                rows[4 * ROW + RHS] = k * dot3(&b, &q);
                let lm = j.limot;
                lm.add_limot(node, bodies, fps, &mut rows[5 * ROW..], &ax1, true);
            }
            JointKind::Slider(j) => {
                set_fixed_orientation(node, bodies, fps, erp, rows, &j.qrel);
                let b0 = node[0].unwrap();
                let pos1 = bodies[b0].pos;
                let mut ax1 = mul0_331(&bodies[b0].r, &j.axis1);
                let (p, q) = plane_space(&ax1);
                let mut c = [0.0; 4];
                if let Some(b1) = node[1] {
                    c = sub3(&bodies[b1].pos, &pos1);
                }
                let has1 = node[1].is_some();
                for (rr, v) in [(3usize, p), (4usize, q)] {
                    let r = rr * ROW;
                    put3(rows, r + J1, &v);
                    if has1 {
                        put3(rows, r + J2, &neg3(&v));
                        let tmp = cross(&c, &v);
                        let s = scaled3(&tmp, 0.5);
                        put3(rows, r + J1 + JA, &s);
                        put3(rows, r + J2 + JA, &s);
                    }
                }
                let k = fps * erp;
                if let Some(b1) = node[1] {
                    let ofs = mul0_331(&bodies[b1].r, &j.offset);
                    c = add3(&c, &ofs);
                    rows[3 * ROW + RHS] = k * dot3(&p, &c);
                    rows[4 * ROW + RHS] = k * dot3(&q, &c);
                } else {
                    let ofs = sub3(&j.offset, &pos1);
                    rows[3 * ROW + RHS] = k * dot3(&p, &ofs);
                    rows[4 * ROW + RHS] = k * dot3(&q, &ofs);
                    if reverse {
                        ax1 = neg3(&ax1);
                    }
                }
                let lm = j.limot;
                lm.add_limot(node, bodies, fps, &mut rows[5 * ROW..], &ax1, false);
            }
            JointKind::Fixed(j) => {
                set_fixed_orientation(node, bodies, fps, erp, &mut rows[3 * ROW..], &j.qrel);
                rows[J1] = 1.0;
                rows[ROW + J1 + 1] = 1.0;
                rows[2 * ROW + J1 + 2] = 1.0;
                let k = fps * j.erp;
                let b0 = node[0].unwrap();
                let ofs = mul0_331(&bodies[b0].r, &j.offset);
                if node[1].is_some() {
                    set_cross_matrix_plus(&mut rows[J1 + JA..], &ofs, ROW);
                    rows[J2] = -1.0;
                    rows[ROW + J2 + 1] = -1.0;
                    rows[2 * ROW + J2 + 2] = -1.0;
                }
                if let Some(b1) = node[1] {
                    for i in 0..3 {
                        rows[i * ROW + RHS] = k * (bodies[b1].pos[i] - bodies[b0].pos[i] + ofs[i]);
                    }
                } else {
                    for i in 0..3 {
                        rows[i * ROW + RHS] = k * (j.offset[i] - bodies[b0].pos[i]);
                    }
                }
                for r in 0..3 {
                    rows[r * ROW + CFM] = j.cfm;
                }
            }
            JointKind::DBall(j) => {
                let b0 = &bodies[node[0].unwrap()];
                let ga1 = b0.rel_point_pos(j.anchor1[0], j.anchor1[1], j.anchor1[2]);
                let ga2 = match node[1] {
                    Some(b1) => bodies[b1].rel_point_pos(j.anchor2[0], j.anchor2[1], j.anchor2[2]),
                    None => v3(j.anchor2[0], j.anchor2[1], j.anchor2[2]),
                };
                let mut q = sub3(&ga1, &ga2);
                const MIN_LENGTH: Real = 1e-12;
                if length3(&q) < MIN_LENGTH {
                    let v1 = b0.point_vel(ga1[0], ga1[1], ga1[2]);
                    let v2 = match node[1] {
                        Some(b1) => bodies[b1].point_vel(ga2[0], ga2[1], ga2[2]),
                        None => [0.0; 4],
                    };
                    q = sub3(&v1, &v2);
                    if length3(&q) < MIN_LENGTH {
                        q = v3(1.0, 0.0, 0.0);
                    }
                }
                normalize3(&mut q);
                put3(rows, J1, &q);
                let rel_a1 = b0.vector_to_world(j.anchor1[0], j.anchor1[1], j.anchor1[2]);
                let mut a1m = [0.0; 12];
                set_cross_matrix_minus(&mut a1m, &rel_a1, 4);
                put3(rows, J1 + JA, &mul1_331(&a1m, &q));
                if let Some(b1) = node[1] {
                    put3(rows, J2, &neg3(&q));
                    let rel_a2 = bodies[b1].vector_to_world(j.anchor2[0], j.anchor2[1], j.anchor2[2]);
                    let mut a2m = [0.0; 12];
                    set_cross_matrix_plus(&mut a2m, &rel_a2, 4);
                    put3(rows, J2 + JA, &mul1_331(&a2m, &q));
                }
                let k = fps * j.erp;
                rows[RHS] = k * (j.target_distance - distance3(&ga1, &ga2));
                rows[CFM] = j.cfm;
            }
            JointKind::Contact(c) => contact_info2(c, node, reverse, bodies, w, fps, erp, rows, findex),
            JointKind::AMotor(a) => {
                let ax = amotor_global_axes(a, node, reverse, bodies);
                let mut axp = ax;
                if a.mode == AMOTOR_EULER {
                    axp[2] = cross(&ax[0], &ax[1]);
                    axp[0] = cross(&ax[1], &ax[2]);
                }
                let mut r = 0;
                for i in 0..a.num {
                    let lm = a.limot[i];
                    if lm.add_limot(node, bodies, fps, &mut rows[r..], &axp[i], true) {
                        r += ROW;
                    }
                }
            }
        }
    }

    /// getSureMaxInfo
    pub fn sure_max(&self) -> u32 {
        match &self.kind {
            JointKind::Ball(_) => 3,
            JointKind::Hinge(_) | JointKind::Slider(_) | JointKind::Fixed(_) => 6,
            JointKind::DBall(_) => 1,
            JointKind::Contact(c) => {
                if c.contact.surface.mode & CONTACT_ROLLING != 0 {
                    6
                } else {
                    3
                }
            }
            JointKind::AMotor(a) => a.num as u32,
        }
    }
}

fn put3(rows: &mut [Real], at: usize, v: &[Real]) {
    rows[at] = v[0];
    rows[at + 1] = v[1];
    rows[at + 2] = v[2];
}

#[allow(clippy::too_many_arguments)]
fn contact_info2(c: &ContactJoint, node: [Option<BodyId>; 2], reverse: bool, bodies: &[Body], w: &WorldParams, fps: Real, world_erp: Real, rows: &mut [Real], findex: &mut [i32]) {
    let ct = &c.contact;
    let normal = if reverse { neg3(&ct.geom.normal) } else { [ct.geom.normal[0], ct.geom.normal[1], ct.geom.normal[2], 0.0] };
    let b0 = &bodies[node[0].unwrap()];
    let c1 = sub3(&ct.geom.pos, &b0.pos);
    let mut c2 = [0.0; 4];
    put3(rows, J1, &normal);
    put3(rows, J1 + JA, &cross(&c1, &normal));
    if let Some(b1) = node[1] {
        c2 = sub3(&ct.geom.pos, &bodies[b1].pos);
        put3(rows, J2, &neg3(&normal));
        put3(rows, J2 + JA, &cross(&normal, &c2));
    }
    let mode = ct.surface.mode;
    let erp = if mode & CONTACT_SOFT_ERP != 0 { ct.surface.soft_erp } else { world_erp };
    let k = fps * erp;
    let mut depth = ct.geom.depth - w.contactp.min_depth;
    if depth < 0.0 {
        depth = 0.0;
    }
    let motion_n = if mode & CONTACT_MOTION_N != 0 { ct.surface.motion_n } else { 0.0 };
    let pushout = k * depth + motion_n;
    let maxvel = w.contactp.max_vel;
    let mut cc = if pushout > maxvel { maxvel } else { pushout };
    if mode & CONTACT_BOUNCE != 0 {
        let mut outgoing = dot3(&rows[J1..], &b0.lvel) + dot3(&rows[J1 + JA..], &b0.avel);
        if let Some(b1) = node[1] {
            outgoing += dot3(&rows[J2..], &bodies[b1].lvel) + dot3(&rows[J2 + JA..], &bodies[b1].avel);
        }
        outgoing -= motion_n;
        if ct.surface.bounce_vel >= 0.0 && (-outgoing) > ct.surface.bounce_vel {
            let newc = -ct.surface.bounce * outgoing + motion_n;
            if newc > cc {
                cc = newc;
            }
        }
    }
    rows[RHS] = cc;
    if mode & CONTACT_SOFT_CFM != 0 {
        rows[CFM] = ct.surface.soft_cfm;
    }
    rows[LO] = 0.0;
    rows[HI] = INFINITY;
    if c.the_m > 1 {
        let (t1, t2) = if mode & CONTACT_FDIR1 != 0 {
            let t1 = [ct.fdir1[0], ct.fdir1[1], ct.fdir1[2], 0.0];
            (t1, cross(&normal, &t1))
        } else {
            plane_space(&normal)
        };
        let mut row = 1usize;
        let has1 = node[1].is_some();
        if ct.surface.mu > 0.0 {
            let r = row * ROW;
            put3(rows, r + J1, &t1);
            put3(rows, r + J1 + JA, &cross(&c1, &t1));
            if has1 {
                put3(rows, r + J2, &neg3(&t1));
                put3(rows, r + J2 + JA, &cross(&t1, &c2));
            }
            if mode & CONTACT_MOTION1 != 0 {
                rows[r + RHS] = ct.surface.motion1;
            }
            if mode & CONTACT_SLIP1 != 0 {
                rows[r + CFM] = ct.surface.slip1;
            }
            rows[r + LO] = -ct.surface.mu;
            rows[r + HI] = ct.surface.mu;
            if mode & CONTACT_APPROX1_1 != 0 {
                findex[row] = 0;
            }
            row += 1;
        }
        let mu2 = if mode & CONTACT_MU2 != 0 { ct.surface.mu2 } else { ct.surface.mu };
        if mu2 > 0.0 {
            let r = row * ROW;
            put3(rows, r + J1, &t2);
            put3(rows, r + J1 + JA, &cross(&c1, &t2));
            if has1 {
                put3(rows, r + J2, &neg3(&t2));
                put3(rows, r + J2 + JA, &cross(&t2, &c2));
            }
            if mode & CONTACT_MOTION2 != 0 {
                rows[r + RHS] = ct.surface.motion2;
            }
            if mode & CONTACT_SLIP2 != 0 {
                rows[r + CFM] = ct.surface.slip2;
            }
            rows[r + LO] = -mu2;
            rows[r + HI] = mu2;
            if mode & CONTACT_APPROX1_2 != 0 {
                findex[row] = 0;
            }
            row += 1;
        }
        if mode & CONTACT_ROLLING != 0 {
            let rho = if mode & CONTACT_AXIS_DEP != 0 {
                [ct.surface.rho, ct.surface.rho2, ct.surface.rho_n]
            } else {
                [ct.surface.rho; 3]
            };
            let ax = [t1, t2, normal];
            let approx = [mode & CONTACT_APPROX1_1 != 0, mode & CONTACT_APPROX1_2 != 0, mode & CONTACT_APPROX1_N != 0];
            for i in 0..3 {
                if rho[i] > 0.0 {
                    let r = row * ROW;
                    put3(rows, r + J1 + JA, &ax[i]);
                    if has1 {
                        put3(rows, r + J2 + JA, &neg3(&ax[i]));
                    }
                    rows[r + LO] = -rho[i];
                    rows[r + HI] = rho[i];
                    if approx[i] {
                        findex[row] = 0;
                    }
                    row += 1;
                }
            }
        }
    }
}

// ---- joint.cpp helpers

/// setBall: three rows holding anchor1 (body 1) on anchor2 (body 2 / world)
fn set_ball(node: [Option<BodyId>; 2], bodies: &[Body], fps: Real, erp: Real, rows: &mut [Real], anchor1: &V3, anchor2: &V3) {
    rows[J1] = 1.0;
    rows[ROW + J1 + 1] = 1.0;
    rows[2 * ROW + J1 + 2] = 1.0;
    let b0 = &bodies[node[0].unwrap()];
    let a1 = mul0_331(&b0.r, anchor1);
    set_cross_matrix_minus(&mut rows[J1 + JA..], &a1, ROW);
    let mut a2 = [0.0; 4];
    if let Some(b1) = node[1] {
        rows[J2] = -1.0;
        rows[ROW + J2 + 1] = -1.0;
        rows[2 * ROW + J2 + 2] = -1.0;
        a2 = mul0_331(&bodies[b1].r, anchor2);
        set_cross_matrix_plus(&mut rows[J2 + JA..], &a2, ROW);
    }
    let k = fps * erp;
    if let Some(b1) = node[1] {
        let b1 = &bodies[b1];
        for j in 0..3 {
            rows[j * ROW + RHS] = k * (a2[j] + b1.pos[j] - a1[j] - b0.pos[j]);
        }
    } else {
        for j in 0..3 {
            rows[j * ROW + RHS] = k * (anchor2[j] - a1[j] - b0.pos[j]);
        }
    }
}

/// setFixedOrientation: three rows holding the relative rotation at qrel
fn set_fixed_orientation(node: [Option<BodyId>; 2], bodies: &[Body], fps: Real, erp: Real, rows: &mut [Real], qrel: &Quat) {
    rows[J1 + JA] = 1.0;
    rows[ROW + J1 + JA + 1] = 1.0;
    rows[2 * ROW + J1 + JA + 2] = 1.0;
    if node[1].is_some() {
        rows[J2 + JA] = -1.0;
        rows[ROW + J2 + JA + 1] = -1.0;
        rows[2 * ROW + J2 + JA + 2] = -1.0;
    }
    let b0 = &bodies[node[0].unwrap()];
    let mut qerr = if let Some(b1) = node[1] {
        let qq = q_mul1(&b0.q, &bodies[b1].q);
        q_mul2(&qq, qrel)
    } else {
        q_mul3(&b0.q, qrel)
    };
    if qerr[0] < 0.0 {
        qerr[1] = -qerr[1];
        qerr[2] = -qerr[2];
        qerr[3] = -qerr[3];
    }
    let e = mul0_331(&b0.r, &qerr[1..]);
    let k_mul_2 = fps * erp * 2.0;
    rows[RHS] = k_mul_2 * e[0];
    rows[ROW + RHS] = k_mul_2 * e[1];
    rows[2 * ROW + RHS] = k_mul_2 * e[2];
}

/// setAnchors: world point -> each body's frame
pub(crate) fn set_anchors(node: [Option<BodyId>; 2], bodies: &[Body], x: Real, y: Real, z: Real, anchor1: &mut V3, anchor2: &mut V3) {
    if let Some(b0) = node[0] {
        let b0 = &bodies[b0];
        let q = v3(x - b0.pos[0], y - b0.pos[1], z - b0.pos[2]);
        *anchor1 = mul1_331(&b0.r, &q);
        if let Some(b1) = node[1] {
            let b1 = &bodies[b1];
            let q = v3(x - b1.pos[0], y - b1.pos[1], z - b1.pos[2]);
            *anchor2 = mul1_331(&b1.r, &q);
        } else {
            anchor2[0] = x;
            anchor2[1] = y;
            anchor2[2] = z;
        }
    }
    anchor1[3] = 0.0;
    anchor2[3] = 0.0;
}

/// setAxes: a world axis (normalised) -> each body's frame
pub(crate) fn set_axes(node: [Option<BodyId>; 2], bodies: &[Body], x: Real, y: Real, z: Real, axis1: Option<&mut V3>, axis2: Option<&mut V3>) {
    if let Some(b0) = node[0] {
        let mut q = v3(x, y, z);
        normalize3(&mut q);
        if let Some(a1) = axis1 {
            *a1 = mul1_331(&bodies[b0].r, &q);
            a1[3] = 0.0;
        }
        if let Some(a2) = axis2 {
            if let Some(b1) = node[1] {
                *a2 = mul1_331(&bodies[b1].r, &q);
            } else {
                a2[0] = x;
                a2[1] = y;
                a2[2] = z;
            }
            a2[3] = 0.0;
        }
    }
}
/// getAnchor
pub(crate) fn get_anchor(node: [Option<BodyId>; 2], bodies: &[Body], anchor1: &V3) -> V3 {
    match node[0] {
        Some(b0) => {
            let b0 = &bodies[b0];
            let r = mul0_331(&b0.r, anchor1);
            v3(r[0] + b0.pos[0], r[1] + b0.pos[1], r[2] + b0.pos[2])
        }
        None => [0.0; 4],
    }
}
/// getAnchor2
pub(crate) fn get_anchor2(node: [Option<BodyId>; 2], bodies: &[Body], anchor2: &V3) -> V3 {
    match node[1] {
        Some(b1) => {
            let b1 = &bodies[b1];
            let r = mul0_331(&b1.r, anchor2);
            v3(r[0] + b1.pos[0], r[1] + b1.pos[1], r[2] + b1.pos[2])
        }
        None => v3(anchor2[0], anchor2[1], anchor2[2]),
    }
}
/// getAxis
pub(crate) fn get_axis(node: [Option<BodyId>; 2], bodies: &[Body], axis1: &V3) -> V3 {
    match node[0] {
        Some(b0) => mul0_331(&bodies[b0].r, axis1),
        None => [0.0; 4],
    }
}
/// dJointGet{Ball,Hinge}Anchor
fn ball_anchor(node: [Option<BodyId>; 2], reverse: bool, bodies: &[Body], a1: &V3, a2: &V3) -> V3 {
    if reverse {
        get_anchor2(node, bodies, a2)
    } else {
        get_anchor(node, bodies, a1)
    }
}
/// computeInitialRelativeRotation (hinge, slider, fixed)
pub(crate) fn initial_relative_rotation(node: [Option<BodyId>; 2], bodies: &[Body], old: Quat) -> Quat {
    match node {
        [Some(b0), Some(b1)] => q_mul1(&bodies[b0].q, &bodies[b1].q),
        [Some(b0), None] => {
            let q = bodies[b0].q;
            [q[0], -q[1], -q[2], -q[3]]
        }
        _ => old,
    }
}
/// dxJointSlider::computeOffset
pub(crate) fn slider_offset(node: [Option<BodyId>; 2], bodies: &[Body], old: V3) -> V3 {
    match node {
        [Some(b0), Some(b1)] => {
            let (p0, p1) = (bodies[b0].pos, bodies[b1].pos);
            let c = v3(p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2]);
            mul1_331(&bodies[b1].r, &c)
        }
        [Some(b0), None] => {
            let p = bodies[b0].pos;
            [p[0], p[1], p[2], old[3]]
        }
        _ => old,
    }
}
/// dxJointDBall::updateTargetDistance
fn dball_distance(node: [Option<BodyId>; 2], bodies: &[Body], j: &DBall) -> Real {
    let p1 = match node[0] {
        Some(b) => bodies[b].rel_point_pos(j.anchor1[0], j.anchor1[1], j.anchor1[2]),
        None => j.anchor1,
    };
    let p2 = match node[1] {
        Some(b) => bodies[b].rel_point_pos(j.anchor2[0], j.anchor2[1], j.anchor2[2]),
        None => j.anchor2,
    };
    distance3(&p1, &p2)
}
/// getHingeAngleFromRelativeQuat
fn hinge_angle_from_rel(qrel: &Quat, axis: &V3) -> Real {
    let cost2 = qrel[0];
    let sint2 = (qrel[1] * qrel[1] + qrel[2] * qrel[2] + qrel[3] * qrel[3]).sqrt();
    let mut theta = if dot3(&qrel[1..], axis) >= 0.0 { 2.0 * sint2.atan2(cost2) } else { 2.0 * sint2.atan2(-cost2) };
    if theta > std::f64::consts::PI {
        theta -= 2.0 * std::f64::consts::PI;
    }
    -theta
}
/// getHingeAngle
pub(crate) fn hinge_angle(bodies: &[Body], node: [Option<BodyId>; 2], axis: &V3, q_initial: &Quat) -> Real {
    let b0 = &bodies[node[0].unwrap()];
    let qrel = if let Some(b1) = node[1] {
        let qq = q_mul1(&b0.q, &bodies[b1].q);
        q_mul2(&qq, q_initial)
    } else {
        q_mul3(&b0.q, q_initial)
    };
    hinge_angle_from_rel(&qrel, axis)
}
/// dJointGetSliderPosition
pub(crate) fn slider_position(bodies: &[Body], node: [Option<BodyId>; 2], reverse: bool, j: &Slider) -> Real {
    let b0 = &bodies[node[0].unwrap()];
    let mut ax1 = mul0_331(&b0.r, &j.axis1);
    let mut q;
    if let Some(b1) = node[1] {
        let b1 = &bodies[b1];
        q = mul0_331(&b1.r, &j.offset);
        for i in 0..3 {
            q[i] = b0.pos[i] - q[i] - b1.pos[i];
        }
    } else {
        q = v3(b0.pos[0] - j.offset[0], b0.pos[1] - j.offset[1], b0.pos[2] - j.offset[2]);
        if reverse {
            ax1 = neg3(&ax1);
        }
    }
    dot3(&ax1, &q)
}

// ---- amotor.cpp

fn amotor_first_body_axis(reverse: bool) -> usize {
    if reverse {
        2
    } else {
        0
    }
}
/// computeGlobalAxes
pub(crate) fn amotor_global_axes(a: &AMotor, node: [Option<BodyId>; 2], reverse: bool, bodies: &[Body]) -> [V3; 3] {
    let mut ax = [[0.0; 4]; 3];
    let b0 = node[0].unwrap();
    if a.mode == AMOTOR_USER {
        for i in 0..a.num {
            let mut assigned = false;
            if a.rel[i] == REL_BODY1 {
                ax[i] = mul0_331(&bodies[b0].r, &a.axis[i]);
                assigned = true;
            } else if a.rel[i] == REL_BODY2 {
                if let Some(b1) = node[1] {
                    ax[i] = mul0_331(&bodies[b1].r, &a.axis[i]);
                    assigned = true;
                }
            }
            if !assigned {
                ax[i] = a.axis[i];
            }
        }
    } else if a.mode == AMOTOR_EULER {
        let first = amotor_first_body_axis(reverse);
        ax[first] = mul0_331(&bodies[b0].r, &a.axis[first]);
        let second = 2 - first;
        if let Some(b1) = node[1] {
            ax[second] = mul0_331(&bodies[b1].r, &a.axis[second]);
        } else {
            ax[second] = a.axis[second];
        }
        ax[1] = cross(&ax[2], &ax[0]);
        normalize3(&mut ax[1]);
    }
    ax
}
/// computeEulerAngles
fn amotor_euler_angles(a: &mut AMotor, node: [Option<BodyId>; 2], reverse: bool, bodies: &[Body], ax: &[V3; 3]) {
    let b0 = node[0].unwrap();
    let mut refs = [[0.0; 4]; 2];
    refs[0] = mul0_331(&bodies[b0].r, &a.references[0]);
    refs[1] = match node[1] {
        Some(b1) => mul0_331(&bodies[b1].r, &a.references[1]),
        None => a.references[1],
    };
    let first = if reverse { 1 } else { 0 };
    let q = cross(&ax[0], &refs[first]);
    a.angle[0] = -dot3(&ax[2], &q).atan2(dot3(&ax[2], &refs[first]));
    let q = cross(&ax[0], &ax[1]);
    a.angle[1] = -dot3(&ax[2], &ax[0]).atan2(dot3(&ax[2], &q));
    let second = 1 - first;
    let q = cross(&ax[1], &ax[2]);
    a.angle[2] = -dot3(&refs[second], &ax[1]).atan2(dot3(&refs[second], &q));
}
/// setEulerReferenceVectors
pub(crate) fn amotor_set_euler_references(a: &mut AMotor, node: [Option<BodyId>; 2], reverse: bool, bodies: &[Body]) {
    let first = amotor_first_body_axis(reverse);
    let second = 2 - first;
    if let (Some(b0), Some(b1)) = (node[0], node[1]) {
        let r = mul0_331(&bodies[b0].r, &a.axis[first]);
        a.references[1] = mul1_331(&bodies[b1].r, &r);
        let r = mul0_331(&bodies[b1].r, &a.axis[second]);
        a.references[0] = mul1_331(&bodies[b0].r, &r);
    } else if let Some(b0) = node[0] {
        a.references[1] = mul0_331(&bodies[b0].r, &a.axis[first]);
        a.references[0] = mul1_331(&bodies[b0].r, &a.axis[second]);
    }
}
