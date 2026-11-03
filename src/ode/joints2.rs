//! The rest of ODE 0.15.2's joints (joints/hinge2, universal, pr, pu,
//! piston, plane2d, lmotor, transmission, dhinge, null.cpp). AC EVO's
//! physics doesn't use them; they complete the port. Same row layout as
//! joints.rs.

use super::joints::*;
use super::math::*;
use super::world::{Body, BodyId, JointId, World};

type Node = [Option<BodyId>; 2];
const PI: Real = std::f64::consts::PI;

/// the rotational-limit guard most joints use before measuring an angle
fn rot_limited(l: &LimitMotor) -> bool {
    (l.lostop >= -PI || l.histop <= PI) && l.lostop <= l.histop
}
/// the same for a linear (or piston-rotational) limit
fn lin_limited(l: &LimitMotor) -> bool {
    (l.lostop > -INFINITY || l.histop < INFINITY) && l.lostop <= l.histop
}
fn r_of(bodies: &[Body], b: BodyId) -> &[Real] {
    &bodies[b].r
}
/// getAxis2
fn get_axis2(node: Node, bodies: &[Body], axis2: &V3) -> V3 {
    match node[1] {
        Some(b1) => mul0_331(&bodies[b1].r, axis2),
        None => v3(axis2[0], axis2[1], axis2[2]),
    }
}
/// dRFrom2Axes; ODE leaves R uninitialised for parallel axes
fn rf2a(ax: Real, ay: Real, az: Real, bx: Real, by: Real, bz: Real) -> M3 {
    r_from_2_axes(ax, ay, az, bx, by, bz).unwrap_or(IDENTITY_R)
}
fn set3(dst: &mut V3, v: &[Real]) {
    dst[0] = v[0];
    dst[1] = v[1];
    dst[2] = v[2];
}

/// setBall2: setBall along `axis` and two normals, `erp1` on the axis
#[allow(clippy::too_many_arguments)]
fn set_ball2(node: Node, bodies: &[Body], fps: Real, erp: Real, rows: &mut [Real], anchor1: &V3, anchor2: &V3, axis: &V3, erp1: Real) {
    let (q1, q2) = plane_space(axis);
    put3(rows, J1, axis);
    put3(rows, ROW + J1, &q1);
    put3(rows, 2 * ROW + J1, &q2);
    let b0 = &bodies[node[0].unwrap()];
    let mut a1 = mul0_331(&b0.r, anchor1);
    put3(rows, J1 + JA, &cross(&a1, axis));
    put3(rows, ROW + J1 + JA, &cross(&a1, &q1));
    put3(rows, 2 * ROW + J1 + JA, &cross(&a1, &q2));
    a1 = add3(&a1, &b0.pos);
    let k1 = fps * erp1;
    let k = fps * erp;
    let d = if let Some(b1) = node[1] {
        put3(rows, J2, &neg3(axis));
        put3(rows, ROW + J2, &neg3(&q1));
        put3(rows, 2 * ROW + J2, &neg3(&q2));
        let mut a2 = mul0_331(&bodies[b1].r, anchor2);
        put3(rows, J2 + JA, &cross(axis, &a2));
        put3(rows, ROW + J2 + JA, &cross(&q1, &a2));
        put3(rows, 2 * ROW + J2 + JA, &cross(&q2, &a2));
        a2 = add3(&a2, &bodies[b1].pos);
        sub3(&a2, &a1)
    } else {
        sub3(anchor2, &a1)
    };
    rows[RHS] = k1 * dot3(axis, &d);
    rows[ROW + RHS] = k * dot3(&q1, &d);
    rows[2 * ROW + RHS] = k * dot3(&q2, &d);
}

/// addTwoPointLimot: a linear limit/motor row between two points
#[allow(clippy::too_many_arguments)]
fn add_two_point_limot(l: &LimitMotor, node: Node, bodies: &mut [Body], fps: Real, rows: &mut [Real], ax1: &V3, pt1: &V3, pt2: &V3) -> bool {
    let mut powered = l.fmax > 0.0;
    if !(powered || l.limit != 0) {
        return false;
    }
    let b0 = node[0].unwrap();
    put3(rows, J1, ax1);
    put3(rows, J1 + JA, &cross(pt1, ax1));
    if node[1].is_some() {
        put3(rows, J2, &neg3(ax1));
        let j2l = [rows[J2], rows[J2 + 1], rows[J2 + 2]];
        put3(rows, J2 + JA, &cross(pt2, &j2l));
    }
    if l.limit != 0 && l.lostop == l.histop {
        powered = false;
    }
    if powered {
        rows[CFM] = l.normal_cfm;
        if l.limit == 0 {
            rows[RHS] = l.vel;
            rows[LO] = -l.fmax;
            rows[HI] = l.fmax;
        } else {
            let mut fm = l.fmax;
            if l.vel > 0.0 || (l.vel == 0.0 && l.limit == 2) {
                fm = -fm;
            }
            if (l.limit == 1 && l.vel > 0.0) || (l.limit == 2 && l.vel < 0.0) {
                fm *= l.fudge_factor;
            }
            let t1 = [rows[J1 + JA], rows[J1 + JA + 1], rows[J1 + JA + 2]];
            add_force(&mut bodies[b0], -fm * ax1[0], -fm * ax1[1], -fm * ax1[2]);
            add_torque(&mut bodies[b0], -fm * t1[0], -fm * t1[1], -fm * t1[2]);
            if let Some(b1) = node[1] {
                let t2 = [rows[J2 + JA], rows[J2 + JA + 1], rows[J2 + JA + 2]];
                add_force(&mut bodies[b1], fm * ax1[0], fm * ax1[1], fm * ax1[2]);
                add_torque(&mut bodies[b1], -fm * t2[0], -fm * t2[1], -fm * t2[2]);
            }
        }
    }
    if l.limit != 0 {
        let k = fps * l.stop_erp;
        rows[RHS] = -k * l.limit_err;
        rows[CFM] = l.stop_cfm;
        if l.lostop == l.histop {
            rows[LO] = -INFINITY;
            rows[HI] = INFINITY;
        } else {
            if l.limit == 1 {
                rows[LO] = 0.0;
                rows[HI] = INFINITY;
            } else {
                rows[LO] = -INFINITY;
                rows[HI] = 0.0;
            }
            if l.bounce > 0.0 {
                let mut vel = dot3(&bodies[b0].lvel, &rows[J1..]) + dot3(&bodies[b0].avel, &rows[J1 + JA..]);
                if let Some(b1) = node[1] {
                    vel += dot3(&bodies[b1].lvel, &rows[J2..]) + dot3(&bodies[b1].avel, &rows[J2 + JA..]);
                }
                if l.limit == 1 {
                    if vel < 0.0 {
                        let newc = -l.bounce * vel;
                        if newc > rows[RHS] {
                            rows[RHS] = newc;
                        }
                    }
                } else if vel > 0.0 {
                    let newc = -l.bounce * vel;
                    if newc < rows[RHS] {
                        rows[RHS] = newc;
                    }
                }
            }
        }
    }
    true
}

// ---- hinge2

#[derive(Clone, Copy, Debug)]
pub struct Hinge2 {
    pub anchor1: V3,
    pub anchor2: V3,
    pub axis1: V3,
    pub axis2: V3,
    pub c0: Real,
    pub s0: Real,
    pub v1: V3,
    pub v2: V3,
    pub w1: V3,
    pub w2: V3,
    pub limot1: LimitMotor,
    pub limot2: LimitMotor,
    pub susp_erp: Real,
    pub susp_cfm: Real,
}
impl Hinge2 {
    fn measure_angle1(&self, node: Node, bodies: &[Body]) -> Real {
        let p = match node[1] {
            Some(b1) => mul0_331(r_of(bodies, b1), &self.axis2),
            None => self.axis2,
        };
        let q = match node[0] {
            Some(b0) => mul1_331(r_of(bodies, b0), &p),
            None => p,
        };
        let x = dot3(&self.v1, &q);
        let y = dot3(&self.v2, &q);
        -y.atan2(x)
    }
    fn measure_angle2(&self, node: Node, bodies: &[Body]) -> Real {
        let p = match node[0] {
            Some(b0) => mul0_331(r_of(bodies, b0), &self.axis1),
            None => self.axis1,
        };
        let q = match node[1] {
            Some(b1) => mul1_331(r_of(bodies, b1), &p),
            None => p,
        };
        let x = dot3(&self.w1, &q);
        let y = dot3(&self.w2, &q);
        -y.atan2(x)
    }
    /// getAxisInfo: (ax1, ax2, ax1 x ax2, sin, cos)
    fn axis_info(&self, node: Node, bodies: &[Body]) -> (V3, V3, V3, Real, Real) {
        let ax1 = mul0_331(r_of(bodies, node[0].unwrap()), &self.axis1);
        let ax2 = mul0_331(r_of(bodies, node[1].unwrap()), &self.axis2);
        let c = cross(&ax1, &ax2);
        let s = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        (ax1, ax2, c, s, dot3(&ax1, &ax2))
    }
    fn make_v1_v2(&mut self, node: Node, bodies: &[Body]) {
        if let Some(b0) = node[0] {
            let ax1 = mul0_331(r_of(bodies, b0), &self.axis1);
            let ax2 = mul0_331(r_of(bodies, node[1].unwrap()), &self.axis2);
            let k = dot3(&ax1, &ax2);
            let mut ax2 = add_scaled3(&ax2, &ax1, -k);
            if safe_normalize3(&mut ax2) {
                let v = cross(&ax1, &ax2);
                self.v1 = mul1_331(r_of(bodies, b0), &ax2);
                self.v2 = mul1_331(r_of(bodies, b0), &v);
            }
        }
    }
    fn make_w1_w2(&mut self, node: Node, bodies: &[Body]) {
        if let Some(b1) = node[1] {
            let ax1 = mul0_331(r_of(bodies, node[0].unwrap()), &self.axis1);
            let ax2 = mul0_331(r_of(bodies, b1), &self.axis2);
            let k = dot3(&ax2, &ax1);
            let mut ax1 = add_scaled3(&ax1, &ax2, -k);
            if safe_normalize3(&mut ax1) {
                let w = cross(&ax2, &ax1);
                self.w1 = mul1_331(r_of(bodies, b1), &ax1);
                self.w2 = mul1_331(r_of(bodies, b1), &w);
            }
        }
    }
    fn update_axes(&mut self, node: Node, bodies: &[Body]) {
        let (_, _, _, s, c) = self.axis_info(node, bodies);
        self.s0 = s;
        self.c0 = c;
        self.make_v1_v2(node, bodies);
        self.make_w1_w2(node, bodies);
    }
    pub(crate) fn info1(&mut self, node: Node, bodies: &[Body]) -> Info1 {
        let mut m = 4;
        self.limot1.limit = 0;
        if rot_limited(&self.limot1) {
            let a = self.measure_angle1(node, bodies);
            self.limot1.test_rotational_limit(a);
        }
        if self.limot1.limit != 0 || self.limot1.fmax > 0.0 {
            m += 1;
        }
        self.limot2.limit = 0;
        if self.limot2.fmax > 0.0 {
            m += 1;
        }
        Info1 { m, nub: 4 }
    }
    pub(crate) fn info2(&self, node: Node, bodies: &mut [Body], fps: Real, erp: Real, rows: &mut [Real]) {
        let (ax1, ax2, mut q, s, c) = self.axis_info(node, bodies);
        normalize3(&mut q);
        set_ball2(node, bodies, fps, erp, rows, &self.anchor1, &self.anchor2, &ax1, self.susp_erp);
        rows[CFM] = self.susp_cfm;
        put3(rows, 3 * ROW + J1 + JA, &q);
        if node[1].is_some() {
            put3(rows, 3 * ROW + J2 + JA, &neg3(&q));
        }
        let k = fps * erp;
        rows[3 * ROW + RHS] = k * (self.c0 * s - self.s0 * c);
        let mut r = 4 * ROW;
        if self.limot1.add_limot(node, bodies, fps, &mut rows[r..], &ax1, true) {
            r += ROW;
        }
        self.limot2.add_limot(node, bodies, fps, &mut rows[r..], &ax2, true);
    }
    fn set_relative(&mut self, node: Node, reverse: bool, bodies: &[Body]) {
        let a = if reverse { get_anchor2(node, bodies, &self.anchor2) } else { get_anchor(node, bodies, &self.anchor1) };
        set_anchors(node, bodies, a[0], a[1], a[2], &mut self.anchor1, &mut self.anchor2);
        if let Some(b0) = node[0] {
            let ax = mul0_331(r_of(bodies, b0), &self.axis1);
            set_axes(node, bodies, ax[0], ax[1], ax[2], Some(&mut self.axis1), None);
            let ax = match node[1] {
                Some(b1) => mul0_331(r_of(bodies, b1), &self.axis2),
                None => [0.0; 4],
            };
            set_axes(node, bodies, ax[0], ax[1], ax[2], None, Some(&mut self.axis2));
        }
        self.update_axes(node, bodies);
    }
}

// ---- universal

#[derive(Clone, Copy, Debug)]
pub struct Universal {
    pub anchor1: V3,
    pub anchor2: V3,
    pub axis1: V3,
    pub axis2: V3,
    pub qrel1: Quat,
    pub qrel2: Quat,
    pub limot1: LimitMotor,
    pub limot2: LimitMotor,
}
impl Universal {
    fn get_axes(&self, node: Node, bodies: &[Body]) -> (V3, V3) {
        let ax1 = mul0_331(r_of(bodies, node[0].unwrap()), &self.axis1);
        (ax1, get_axis2(node, bodies, &self.axis2))
    }
    fn get_angles(&self, node: Node, bodies: &[Body]) -> (Real, Real) {
        let Some(b0) = node[0] else { return (0.0, 0.0) };
        let (ax1, ax2) = self.get_axes(node, bodies);
        let r = rf2a(ax1[0], ax1[1], ax1[2], ax2[0], ax2[1], ax2[2]);
        let qcross = q_from_r(&r);
        let qq = q_mul1(&bodies[b0].q, &qcross);
        let qrel = q_mul2(&qq, &self.qrel1);
        let a1 = hinge_angle_from_rel(&qrel, &self.axis1);
        let mut qrel = [0.0, ax1[0] + ax2[0], ax1[1] + ax2[1], ax1[2] + ax2[2]];
        let l = 1.0 / (qrel[1] * qrel[1] + qrel[2] * qrel[2] + qrel[3] * qrel[3]).sqrt();
        qrel[1] *= l;
        qrel[2] *= l;
        qrel[3] *= l;
        let qcross2 = q_mul0(&qrel, &qcross);
        let qrel = match node[1] {
            Some(b1) => {
                let qq = q_mul1(&bodies[b1].q, &qcross2);
                q_mul2(&qq, &self.qrel2)
            }
            None => q_mul2(&qcross2, &self.qrel2),
        };
        (a1, -hinge_angle_from_rel(&qrel, &self.axis2))
    }
    fn angle1(&self, node: Node, bodies: &[Body]) -> Real {
        let Some(b0) = node[0] else { return 0.0 };
        let (ax1, ax2) = self.get_axes(node, bodies);
        let r = rf2a(ax1[0], ax1[1], ax1[2], ax2[0], ax2[1], ax2[2]);
        let qq = q_mul1(&bodies[b0].q, &q_from_r(&r));
        hinge_angle_from_rel(&q_mul2(&qq, &self.qrel1), &self.axis1)
    }
    fn angle2(&self, node: Node, bodies: &[Body]) -> Real {
        if node[0].is_none() {
            return 0.0;
        }
        let (ax1, ax2) = self.get_axes(node, bodies);
        let r = rf2a(ax2[0], ax2[1], ax2[2], ax1[0], ax1[1], ax1[2]);
        let qcross = q_from_r(&r);
        let qrel = match node[1] {
            Some(b1) => q_mul2(&q_mul1(&bodies[b1].q, &qcross), &self.qrel2),
            None => q_mul2(&qcross, &self.qrel2),
        };
        -hinge_angle_from_rel(&qrel, &self.axis2)
    }
    fn info1_limits(&mut self, node: Node, bodies: &[Body]) -> u32 {
        let (l1, l2) = (rot_limited(&self.limot1), rot_limited(&self.limot2));
        self.limot1.limit = 0;
        self.limot2.limit = 0;
        if l1 || l2 {
            let (a1, a2) = self.get_angles(node, bodies);
            if l1 {
                self.limot1.test_rotational_limit(a1);
            }
            if l2 {
                self.limot2.test_rotational_limit(a2);
            }
        }
        let mut m = 0;
        if self.limot1.limit != 0 || self.limot1.fmax > 0.0 {
            m += 1;
        }
        if self.limot2.limit != 0 || self.limot2.fmax > 0.0 {
            m += 1;
        }
        m
    }
    pub(crate) fn info1(&mut self, node: Node, bodies: &[Body]) -> Info1 {
        Info1 { m: 4 + self.info1_limits(node, bodies), nub: 4 }
    }
    pub(crate) fn info2(&self, node: Node, bodies: &mut [Body], fps: Real, erp: Real, rows: &mut [Real]) {
        set_ball(node, bodies, fps, erp, rows, &self.anchor1, &self.anchor2);
        let (ax1, ax2) = self.get_axes(node, bodies);
        let k = dot3(&ax1, &ax2);
        let t = add_scaled3(&ax2, &ax1, -k);
        let mut p = cross(&ax1, &t);
        normalize3(&mut p);
        put3(rows, 3 * ROW + J1 + JA, &p);
        if node[1].is_some() {
            put3(rows, 3 * ROW + J2 + JA, &neg3(&p));
        }
        rows[3 * ROW + RHS] = fps * erp * (-k);
        let mut r = 4 * ROW;
        if self.limot1.add_limot(node, bodies, fps, &mut rows[r..], &ax1, true) {
            r += ROW;
        }
        self.limot2.add_limot(node, bodies, fps, &mut rows[r..], &ax2, true);
    }
    /// computeInitialRelativeRotations
    fn initial_rotations(&mut self, node: Node, bodies: &[Body]) {
        if let Some(b0) = node[0] {
            let (ax1, ax2) = self.get_axes(node, bodies);
            let r = rf2a(ax1[0], ax1[1], ax1[2], ax2[0], ax2[1], ax2[2]);
            self.qrel1 = q_mul1(&bodies[b0].q, &q_from_r(&r));
            let r = rf2a(ax2[0], ax2[1], ax2[2], ax1[0], ax1[1], ax1[2]);
            let qcross = q_from_r(&r);
            self.qrel2 = match node[1] {
                Some(b1) => q_mul1(&bodies[b1].q, &qcross),
                None => qcross,
            };
        }
    }
    /// dJointGetUniversalAxis1/2 (world, reverse-aware)
    fn world_axes(&self, node: Node, reverse: bool, bodies: &[Body]) -> (V3, V3) {
        let a1 = get_axis(node, bodies, &self.axis1);
        let a2 = get_axis2(node, bodies, &self.axis2);
        if reverse {
            (a2, a1)
        } else {
            (a1, a2)
        }
    }
    /// setAxes for the reverse-aware first/second axis
    fn set_axis(&mut self, node: Node, reverse: bool, bodies: &[Body], second: bool, x: Real, y: Real, z: Real) {
        if reverse != second {
            set_axes(node, bodies, x, y, z, None, Some(&mut self.axis2));
        } else {
            set_axes(node, bodies, x, y, z, Some(&mut self.axis1), None);
        }
    }
    fn set_relative(&mut self, node: Node, reverse: bool, bodies: &[Body]) {
        let a = if reverse { get_anchor2(node, bodies, &self.anchor2) } else { get_anchor(node, bodies, &self.anchor1) };
        set_anchors(node, bodies, a[0], a[1], a[2], &mut self.anchor1, &mut self.anchor2);
        let (ax1, ax2) = self.world_axes(node, reverse, bodies);
        self.set_axis(node, reverse, bodies, false, ax1[0], ax1[1], ax1[2]);
        self.set_axis(node, reverse, bodies, true, ax2[0], ax2[1], ax2[2]);
        self.initial_rotations(node, bodies);
    }
}

// ---- PR (prismatic + rotoide)

#[derive(Clone, Copy, Debug)]
pub struct Pr {
    pub anchor2: V3,
    pub axis_r1: V3,
    pub axis_r2: V3,
    pub axis_p1: V3,
    pub qrel: Quat,
    pub offset: V3,
    pub limot_r: LimitMotor,
    pub limot_p: LimitMotor,
}
impl Pr {
    fn position(&self, node: Node, reverse: bool, bodies: &[Body]) -> Real {
        let b0 = &bodies[node[0].unwrap()];
        let mut q = mul0_331(&b0.r, &self.offset);
        if let Some(b1) = node[1] {
            let a2 = mul0_331(&bodies[b1].r, &self.anchor2);
            for i in 0..3 {
                q[i] = (b0.pos[i] + q[i]) - (bodies[b1].pos[i] + a2[i]);
            }
        } else {
            for i in 0..3 {
                q[i] = (b0.pos[i] + q[i]) - self.anchor2[i];
            }
            if reverse {
                q = neg3(&q);
            }
        }
        dot3(&mul0_331(&b0.r, &self.axis_p1), &q)
    }
    pub(crate) fn info1(&mut self, node: Node, reverse: bool, bodies: &[Body]) -> Info1 {
        let mut m = 4;
        self.limot_p.limit = 0;
        if lin_limited(&self.limot_p) {
            let p = self.position(node, reverse, bodies);
            self.limot_p.test_rotational_limit(p);
        }
        if self.limot_p.limit != 0 || self.limot_p.fmax > 0.0 {
            m += 1;
        }
        self.limot_r.limit = 0;
        if rot_limited(&self.limot_r) {
            let a = hinge_angle(bodies, node, &self.axis_r1, &self.qrel);
            self.limot_r.test_rotational_limit(a);
        }
        if self.limot_r.limit != 0 || self.limot_r.fmax > 0.0 {
            m += 1;
        }
        Info1 { m, nub: 4 }
    }
    pub(crate) fn info2(&self, node: Node, reverse: bool, bodies: &mut [Body], fps: Real, erp: Real, rows: &mut [Real]) {
        let k = fps * erp;
        let b0 = node[0].unwrap();
        let (pos1, r1) = (bodies[b0].pos, bodies[b0].r);
        let ax_p = mul0_331(&r1, &self.axis_p1);
        let mut wanchor2 = [0.0; 4];
        let dist = if let Some(b1) = node[1] {
            wanchor2 = mul0_331(&bodies[b1].r, &self.anchor2);
            let p2 = bodies[b1].pos;
            v3(wanchor2[0] + p2[0] - pos1[0], wanchor2[1] + p2[1] - pos1[1], wanchor2[2] + p2[2] - pos1[2])
        } else if reverse {
            sub3(&pos1, &self.anchor2)
        } else {
            sub3(&self.anchor2, &pos1)
        };
        let ax1 = mul0_331(&r1, &self.axis_r1);
        let q = cross(&ax1, &ax_p);
        let has1 = node[1].is_some();
        put3(rows, J1 + JA, &ax_p);
        if has1 {
            put3(rows, J2 + JA, &neg3(&ax_p));
        }
        put3(rows, ROW + J1 + JA, &q);
        if has1 {
            put3(rows, ROW + J2 + JA, &neg3(&q));
        }
        let ax2 = match node[1] {
            Some(b1) => mul0_331(&bodies[b1].r, &self.axis_r2),
            None => self.axis_r2,
        };
        let b = cross(&ax1, &ax2);
        rows[RHS] = k * dot3(&b, &ax_p);
        rows[ROW + RHS] = k * dot3(&b, &q);
        for (rr, v) in [(2usize, ax1), (3usize, q)] {
            let r = rr * ROW;
            put3(rows, r + J1, &v);
            put3(rows, r + J1 + JA, &cross(&dist, &v));
            if has1 {
                put3(rows, r + J2, &neg3(&v));
                let c = if rr == 2 { cross(&ax2, &wanchor2) } else { cross(&q, &wanchor2) };
                put3(rows, r + J2 + JA, &c);
            }
        }
        let err = sub3(&dist, &mul0_331(&r1, &self.offset));
        rows[2 * ROW + RHS] = k * dot3(&ax1, &err);
        rows[3 * ROW + RHS] = k * dot3(&q, &err);
        let mut r = 4 * ROW;
        let axp = if has1 || !reverse { ax_p } else { neg3(&ax_p) };
        if self.limot_p.add_limot(node, bodies, fps, &mut rows[r..], &axp, false) {
            r += ROW;
        }
        self.limot_r.add_limot(node, bodies, fps, &mut rows[r..], &ax1, true);
    }
    fn set_relative(&mut self, node: Node, bodies: &[Body]) {
        let a = match node[1] {
            Some(_) => get_anchor2(node, bodies, &self.anchor2),
            None => self.anchor2,
        };
        set_anchors(node, bodies, a[0], a[1], a[2], &mut self.offset, &mut self.anchor2);
        let ax = get_axis(node, bodies, &self.axis_p1);
        set_axes(node, bodies, ax[0], ax[1], ax[2], Some(&mut self.axis_p1), None);
        let ax = get_axis(node, bodies, &self.axis_r1);
        set_axes(node, bodies, ax[0], ax[1], ax[2], Some(&mut self.axis_r1), Some(&mut self.axis_r2));
        self.qrel = initial_relative_rotation(node, bodies, self.qrel);
    }
}

// ---- PU (prismatic + universal)

#[derive(Clone, Copy, Debug)]
pub struct Pu {
    pub u: Universal,
    pub axis_p1: V3,
    pub limot_p: LimitMotor,
}
impl Pu {
    fn position(&self, node: Node, reverse: bool, bodies: &[Body]) -> Real {
        let b0 = &bodies[node[0].unwrap()];
        let mut q = mul0_331(&b0.r, &self.u.anchor1);
        if let Some(b1) = node[1] {
            let a2 = mul0_331(&bodies[b1].r, &self.u.anchor2);
            for i in 0..3 {
                q[i] = (b0.pos[i] + q[i]) - (bodies[b1].pos[i] + a2[i]);
            }
        } else {
            for i in 0..3 {
                q[i] = (b0.pos[i] + q[i]) - self.u.anchor2[i];
            }
            if reverse {
                q = neg3(&q);
            }
        }
        dot3(&mul0_331(&b0.r, &self.axis_p1), &q)
    }
    pub(crate) fn info1(&mut self, node: Node, reverse: bool, bodies: &[Body]) -> Info1 {
        let mut m = 3;
        self.limot_p.limit = 0;
        if lin_limited(&self.limot_p) {
            let p = self.position(node, reverse, bodies);
            self.limot_p.test_rotational_limit(p);
        }
        if self.limot_p.limit != 0 || self.limot_p.fmax > 0.0 {
            m += 1;
        }
        m += self.u.info1_limits(node, bodies);
        Info1 { m, nub: 3 }
    }
    pub(crate) fn info2(&self, node: Node, reverse: bool, bodies: &mut [Body], fps: Real, erp: Real, rows: &mut [Real]) {
        let k = fps * erp;
        let u = &self.u;
        let ax1 = get_axis(node, bodies, &u.axis1);
        let ax2 = get_axis2(node, bodies, &u.axis2);
        let mut perp = cross(&ax1, &ax2);
        normalize3(&mut perp);
        put3(rows, J1 + JA, &perp);
        let has1 = node[1].is_some();
        if has1 {
            put3(rows, J2 + JA, &neg3(&perp));
        }
        rows[RHS] = -k * dot3(&ax1, &ax2);
        let mut an1 = get_anchor(node, bodies, &u.anchor1);
        let mut an2 = get_anchor2(node, bodies, &u.anchor2);
        let mut ax_p = if reverse { get_axis2(node, bodies, &self.axis_p1) } else { get_axis(node, bodies, &self.axis_p1) };
        let sep = sub3(&an2, &an1);
        let (p, q) = plane_space(&ax_p);
        put3(rows, ROW + J1, &p);
        put3(rows, 2 * ROW + J1, &q);
        let b0 = node[0].unwrap();
        an1 = sub3(&an1, &bodies[b0].pos);
        put3(rows, ROW + J1 + JA, &cross(&an1, &p));
        put3(rows, 2 * ROW + J1 + JA, &cross(&an1, &q));
        if let Some(b1) = node[1] {
            put3(rows, ROW + J2, &neg3(&p));
            put3(rows, 2 * ROW + J2, &neg3(&q));
            an2 = sub3(&an2, &bodies[b1].pos);
            put3(rows, ROW + J2 + JA, &cross(&p, &an2));
            put3(rows, 2 * ROW + J2 + JA, &cross(&q, &an2));
        }
        rows[ROW + RHS] = k * dot3(&p, &sep);
        rows[2 * ROW + RHS] = k * dot3(&q, &sep);
        let mut r = 3 * ROW;
        if u.limot1.add_limot(node, bodies, fps, &mut rows[r..], &ax1, true) {
            r += ROW;
        }
        if u.limot2.add_limot(node, bodies, fps, &mut rows[r..], &ax2, true) {
            r += ROW;
        }
        if !(has1 || !reverse) {
            ax_p = neg3(&ax_p);
        }
        add_two_point_limot(&self.limot_p, node, bodies, fps, &mut rows[r..], &ax_p, &an1, &an2);
    }
    fn set_relative(&mut self, node: Node, reverse: bool, bodies: &[Body]) {
        let a = match node[1] {
            Some(_) => get_anchor2(node, bodies, &self.u.anchor2),
            None => self.u.anchor2,
        };
        set_anchors(node, bodies, a[0], a[1], a[2], &mut self.u.anchor1, &mut self.u.anchor2);
        let (ax1, ax2) = self.u.world_axes(node, reverse, bodies);
        let ax3 = get_axis(node, bodies, &self.axis_p1);
        self.u.set_axis(node, reverse, bodies, false, ax1[0], ax1[1], ax1[2]);
        self.u.set_axis(node, reverse, bodies, true, ax2[0], ax2[1], ax2[2]);
        set_axes(node, bodies, ax3[0], ax3[1], ax3[2], Some(&mut self.axis_p1), None);
        self.u.initial_rotations(node, bodies);
    }
}

// ---- piston

#[derive(Clone, Copy, Debug)]
pub struct Piston {
    pub anchor1: V3,
    pub anchor2: V3,
    pub axis1: V3,
    pub axis2: V3,
    pub qrel: Quat,
    pub limot_p: LimitMotor,
    pub limot_r: LimitMotor,
}
impl Piston {
    fn position(&self, node: Node, reverse: bool, bodies: &[Body]) -> Real {
        let Some(b0) = node[0] else { return 0.0 };
        let b0 = &bodies[b0];
        let mut q = mul0_331(&b0.r, &self.anchor1);
        if let Some(b1) = node[1] {
            let a2 = mul0_331(&bodies[b1].r, &self.anchor2);
            for i in 0..3 {
                q[i] = (b0.pos[i] + q[i]) - (bodies[b1].pos[i] + a2[i]);
            }
        } else {
            for i in 0..3 {
                q[i] = (b0.pos[i] + q[i]) - self.anchor2[i];
            }
            if reverse {
                q = neg3(&q);
            }
        }
        dot3(&mul0_331(&b0.r, &self.axis1), &q)
    }
    pub(crate) fn info1(&mut self, node: Node, reverse: bool, bodies: &[Body]) -> Info1 {
        let mut m = 4;
        self.limot_p.limit = 0;
        if lin_limited(&self.limot_p) {
            let p = self.position(node, reverse, bodies);
            self.limot_p.test_rotational_limit(p);
        }
        if self.limot_p.limit != 0 || self.limot_p.fmax > 0.0 {
            m += 1;
        }
        self.limot_r.limit = 0;
        if lin_limited(&self.limot_r) {
            let a = hinge_angle(bodies, node, &self.axis1, &self.qrel);
            self.limot_r.test_rotational_limit(a);
        }
        if self.limot_r.limit != 0 || self.limot_r.fmax > 0.0 {
            m += 1;
        }
        Info1 { m, nub: 4 }
    }
    pub(crate) fn info2(&self, node: Node, reverse: bool, bodies: &mut [Body], fps: Real, erp: Real, rows: &mut [Real]) {
        let k = fps * erp;
        let b0 = node[0].unwrap();
        let (pos1, r1) = (bodies[b0].pos, bodies[b0].r);
        let mut lanchor2 = [0.0; 4];
        let dist = if let Some(b1) = node[1] {
            lanchor2 = mul0_331(&bodies[b1].r, &self.anchor2);
            let p2 = bodies[b1].pos;
            v3(lanchor2[0] + p2[0] - pos1[0], lanchor2[1] + p2[1] - pos1[1], lanchor2[2] + p2[2] - pos1[2])
        } else if reverse {
            sub3(&pos1, &self.anchor2)
        } else {
            sub3(&self.anchor2, &pos1)
        };
        let ax1 = mul0_331(&r1, &self.axis1);
        let (p, q) = plane_space(&ax1);
        let has1 = node[1].is_some();
        put3(rows, J1 + JA, &p);
        if has1 {
            put3(rows, J2 + JA, &neg3(&p));
        }
        put3(rows, ROW + J1 + JA, &q);
        let b = if let Some(b1) = node[1] {
            put3(rows, ROW + J2 + JA, &neg3(&q));
            let ax2 = mul0_331(&bodies[b1].r, &self.axis2);
            cross(&ax1, &ax2)
        } else {
            cross(&ax1, &self.axis2)
        };
        rows[RHS] = k * dot3(&p, &b);
        rows[ROW + RHS] = k * dot3(&q, &b);
        for (rr, v) in [(2usize, p), (3usize, q)] {
            let r = rr * ROW;
            put3(rows, r + J1, &v);
            put3(rows, r + J1 + JA, &cross(&dist, &v));
            if has1 {
                put3(rows, r + J2, &neg3(&v));
                put3(rows, r + J2 + JA, &cross(&v, &lanchor2));
            }
        }
        let err = sub3(&dist, &mul0_331(&r1, &self.anchor1));
        rows[2 * ROW + RHS] = k * dot3(&p, &err);
        rows[3 * ROW + RHS] = k * dot3(&q, &err);
        let mut r = 4 * ROW;
        let axp = if has1 || !reverse { ax1 } else { neg3(&ax1) };
        if self.limot_p.add_limot(node, bodies, fps, &mut rows[r..], &axp, false) {
            r += ROW;
        }
        self.limot_r.add_limot(node, bodies, fps, &mut rows[r..], &ax1, true);
    }
    fn set_relative(&mut self, node: Node, reverse: bool, bodies: &[Body]) {
        let a = if reverse { get_anchor2(node, bodies, &self.anchor2) } else { get_anchor(node, bodies, &self.anchor1) };
        set_anchors(node, bodies, a[0], a[1], a[2], &mut self.anchor1, &mut self.anchor2);
        let ax = get_axis(node, bodies, &self.axis1);
        set_axes(node, bodies, ax[0], ax[1], ax[2], Some(&mut self.axis1), Some(&mut self.axis2));
        self.qrel = initial_relative_rotation(node, bodies, self.qrel);
    }
}

// ---- plane2d

#[derive(Clone, Copy, Debug)]
pub struct Plane2D {
    pub row_x: usize,
    pub row_y: usize,
    pub row_angle: usize,
    pub motor_x: LimitMotor,
    pub motor_y: LimitMotor,
    pub motor_angle: LimitMotor,
}
impl Plane2D {
    pub(crate) fn info1(&mut self) -> Info1 {
        let mut m = 3;
        let mut row = |l: &LimitMotor| {
            if l.fmax > 0.0 {
                m += 1;
                m as usize - 1
            } else {
                0
            }
        };
        self.row_x = row(&self.motor_x);
        self.row_y = row(&self.motor_y);
        self.row_angle = row(&self.motor_angle);
        Info1 { m, nub: 3 }
    }
    pub(crate) fn info2(&self, node: Node, bodies: &mut [Body], fps: Real, erp: Real, rows: &mut [Real]) {
        let eps = fps * erp;
        rows[J1 + 2] = 1.0;
        rows[ROW + J1 + JA] = 1.0;
        rows[2 * ROW + J1 + JA + 1] = 1.0;
        rows[RHS] = eps * -bodies[node[0].unwrap()].pos[2];
        let ident = [v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)];
        for (r, l, ax, rot) in [(self.row_x, &self.motor_x, 0, false), (self.row_y, &self.motor_y, 1, false), (self.row_angle, &self.motor_angle, 2, true)] {
            if r > 0 {
                l.add_limot(node, bodies, fps, &mut rows[r * ROW..], &ident[ax], rot);
            }
        }
    }
}

// ---- lmotor

#[derive(Clone, Copy, Debug)]
pub struct LMotor {
    pub num: usize,
    pub rel: [i32; 3],
    pub axis: [V3; 3],
    pub limot: [LimitMotor; 3],
}
impl LMotor {
    fn global_axes(&self, node: Node, bodies: &[Body]) -> [V3; 3] {
        let mut ax = [[0.0; 4]; 3];
        for i in 0..self.num {
            if self.rel[i] == 1 {
                ax[i] = mul0_331(r_of(bodies, node[0].unwrap()), &self.axis[i]);
            } else if self.rel[i] == 2 {
                if let Some(b1) = node[1] {
                    ax[i] = mul0_331(r_of(bodies, b1), &self.axis[i]);
                }
            } else {
                set3(&mut ax[i], &self.axis[i]);
            }
        }
        ax
    }
    pub(crate) fn info1(&self) -> Info1 {
        Info1 { m: self.limot[..self.num].iter().filter(|l| l.fmax > 0.0).count() as u32, nub: 0 }
    }
    pub(crate) fn info2(&self, node: Node, bodies: &mut [Body], fps: Real, rows: &mut [Real]) {
        let ax = self.global_axes(node, bodies);
        let mut r = 0;
        for i in 0..self.num {
            if self.limot[i].add_limot(node, bodies, fps, &mut rows[r..], &ax[i], false) {
                r += ROW;
            }
        }
    }
}

// ---- dhinge (a distance ball that also keeps two axes parallel)

#[derive(Clone, Copy, Debug)]
pub struct DHinge {
    pub ball: DBall,
    pub axis1: V3,
    pub axis2: V3,
}
impl DHinge {
    pub(crate) fn info2(&self, node: Node, bodies: &mut [Body], fps: Real, rows: &mut [Real]) {
        dball_info2(&self.ball, node, bodies, fps, rows);
        let b0 = &bodies[node[0].unwrap()];
        let ga1 = b0.vector_to_world(self.axis1[0], self.axis1[1], self.axis1[2]);
        let (p, q) = plane_space(&ga1);
        let has1 = node[1].is_some();
        put3(rows, ROW + J1 + JA, &p);
        if has1 {
            put3(rows, ROW + J2 + JA, &neg3(&p));
        }
        put3(rows, 2 * ROW + J1 + JA, &q);
        if has1 {
            put3(rows, 2 * ROW + J2 + JA, &neg3(&q));
        }
        let ga2 = match node[1] {
            Some(b1) => bodies[b1].vector_to_world(self.axis2[0], self.axis2[1], self.axis2[2]),
            None => v3(self.axis2[0], self.axis2[1], self.axis2[2]),
        };
        let u = cross(&ga1, &ga2);
        let k = fps * self.ball.erp;
        rows[ROW + RHS] = k * dot3(&u, &p);
        rows[2 * ROW + RHS] = k * dot3(&u, &q);
        put3(rows, 3 * ROW + J1, &ga1);
        if let Some(b1) = node[1] {
            let h = add_scaled_vectors3(&b0.pos, &bodies[b1].pos, -0.5, 0.5);
            let ja = cross(&h, &ga1);
            put3(rows, 3 * ROW + J1 + JA, &ja);
            put3(rows, 3 * ROW + J2, &neg3(&ga1));
            put3(rows, 3 * ROW + J2 + JA, &ja);
        }
        let a = &self.ball;
        let g1 = b0.rel_point_pos(a.anchor1[0], a.anchor1[1], a.anchor1[2]);
        let g2 = match node[1] {
            Some(b1) => bodies[b1].rel_point_pos(a.anchor2[0], a.anchor2[1], a.anchor2[2]),
            None => v3(a.anchor2[0], a.anchor2[1], a.anchor2[2]),
        };
        let d = sub3(&g1, &g2);
        rows[3 * ROW + RHS] = -k * dot3(&ga1, &d);
    }
}

// ---- transmission

pub const TRANSMISSION_PARALLEL_AXES: i32 = 0;
pub const TRANSMISSION_INTERSECTING_AXES: i32 = 1;
pub const TRANSMISSION_CHAIN_DRIVE: i32 = 2;

#[derive(Clone, Copy, Debug)]
pub struct Transmission {
    pub mode: i32,
    pub update: bool,
    pub contacts: [V3; 2],
    pub axes: [V3; 2],
    pub anchors: [V3; 2],
    pub reference: [M3; 2],
    pub phase: [Real; 2],
    pub radii: [Real; 2],
    pub backlash: Real,
    pub ratio: Real,
    pub erp: Real,
    pub cfm: Real,
}
impl Transmission {
    pub(crate) fn info1(&self) -> Info1 {
        Info1 { m: 1, nub: if self.backlash > 0.0 { 0 } else { 1 } }
    }
    pub(crate) fn info2(&mut self, node: Node, bodies: &[Body], fps: Real, rows: &mut [Real]) {
        let mut a = [[0.0; 4]; 2];
        let mut n = [[0.0; 4]; 2];
        let mut p = [[0.0; 4]; 2];
        let mut omega = [[0.0; 4]; 2];
        for i in 0..2 {
            let b = &bodies[node[i].unwrap()];
            a[i] = b.rel_point_pos(self.anchors[i][0], self.anchors[i][1], self.anchors[i][2]);
            n[i] = b.vector_to_world(self.axes[i][0], self.axes[i][1], self.axes[i][2]);
            p[i] = b.pos;
            omega[i] = b.avel;
        }
        if self.update && dot3(&n[0], &n[1]) < 0.0 {
            for k in 0..3 {
                self.axes[0][k] = -self.axes[0][k];
                n[0][k] = -n[0][k];
            }
        }
        let mut c = [[0.0; 4]; 2];
        let mut l = [[0.0; 4]; 2];
        match self.mode {
            TRANSMISSION_PARALLEL_AXES => {
                let mut d = sub3(&a[1], &a[0]);
                c[0] = add_scaled3(&a[0], &d, self.ratio / (1.0 + self.ratio));
                c[1] = c[0];
                normalize3(&mut d);
                for i in 0..2 {
                    l[i] = cross(&d, &n[i]);
                }
            }
            TRANSMISSION_INTERSECTING_AXES => {
                l[0] = cross(&n[0], &n[1]);
                l[1] = l[0];
                let nn = dot3(&n[0], &n[1]);
                let na0 = dot3(&n[0], &a[0]);
                let na1 = dot3(&n[1], &a[1]);
                let o = add_scaled_vectors3(&n[0], &n[1], (na0 - na1 * nn) / (1.0 - nn * nn), (na1 - na0 * nn) / (1.0 - nn * nn));
                for i in 0..2 {
                    let d = sub3(&a[i], &o);
                    let m = dot3(&d, &l[i]);
                    c[i] = add_scaled3(&o, &l[i], m);
                }
            }
            TRANSMISSION_CHAIN_DRIVE => {
                let mut d = sub3(&a[0], &a[1]);
                let m = length3(&d);
                let cosphi = ((self.radii[1] - self.radii[0]) / m).clamp(-1.0, 1.0);
                let sinphi = (1.0 - cosphi * cosphi).sqrt();
                normalize3(&mut d);
                for i in 0..2 {
                    let u = v3(self.radii[i] * cosphi, 0.0, self.radii[i] * sinphi);
                    let z = cross(&d, &n[i]);
                    let v = v3(dot3(&d, &u), dot3(&n[i], &u), dot3(&z, &u));
                    c[i] = add3(&a[i], &v);
                    l[i] = cross(&v, &n[i]);
                    normalize3(&mut l[i]);
                }
            }
            _ => {}
        }
        let mut r = [[0.0; 4]; 2];
        if self.update {
            for i in 0..2 {
                r[i] = sub3(&c[i], &a[i]);
                self.radii[i] = length3(&r[i]);
                let b = &bodies[node[i].unwrap()];
                let mut x = b.vector_from_world(r[i][0], r[i][1], r[i][2]);
                normalize3(&mut x);
                let rf = &mut self.reference[i];
                rf[0..3].copy_from_slice(&x[..3]);
                rf[8..11].copy_from_slice(&self.axes[i][..3]);
                let y = cross(&rf[8..11], &rf[0..3]);
                rf[4..7].copy_from_slice(&y[..3]);
                self.phase[i] = 0.0;
            }
            self.ratio = self.radii[0] / self.radii[1];
            self.update = false;
        }
        let two_pi = 2.0 * PI;
        for i in 0..2 {
            r[i] = sub3(&c[i], &a[i]);
            let b = &bodies[node[i].unwrap()];
            let s = b.vector_from_world(r[i][0], r[i][1], r[i][2]);
            let t = mul0_331(&self.reference[i], &s);
            let mut theta = t[1].atan2(t[0]);
            let phase_hat = self.phase[i] + dot3(&omega[i], &n[i]) / fps;
            if phase_hat > std::f64::consts::FRAC_PI_2 {
                if theta < 0.0 {
                    theta += two_pi;
                }
                theta += (phase_hat / two_pi).floor() * two_pi;
            } else if phase_hat < -std::f64::consts::FRAC_PI_2 {
                if theta > 0.0 {
                    theta -= two_pi;
                }
                theta += (phase_hat / two_pi).ceil() * two_pi;
            }
            self.phase[i] = if phase_hat - theta > PI {
                theta + two_pi
            } else if phase_hat - theta < -PI {
                theta - two_pi
            } else {
                theta
            };
        }
        let delta = if self.mode == TRANSMISSION_CHAIN_DRIVE {
            length3(&r[0]) * self.phase[0] - length3(&r[1]) * self.phase[1]
        } else {
            length3(&r[0]) * self.phase[0] + length3(&r[1]) * self.phase[1]
        };
        if self.mode == TRANSMISSION_CHAIN_DRIVE && delta < 0.0 {
            let d = sub3(&a[0], &a[1]);
            for i in 0..2 {
                let nn = cross(&n[i], &d);
                let aa = dot3(&nn, &nn);
                c[i] = add_scaled_vectors3(&c[i], &nn, 1.0, -2.0 * dot3(&c[i], &nn) / aa);
                l[i] = add_scaled_vectors3(&l[i], &nn, -1.0, 2.0 * dot3(&l[i], &nn) / aa);
            }
        }
        if self.backlash == 0.0 || delta.abs() > self.backlash {
            for i in 0..2 {
                r[i] = sub3(&c[i], &p[i]);
            }
            put3(rows, J1, &l[0]);
            put3(rows, J1 + JA, &cross(&r[0], &l[0]));
            put3(rows, J2, &neg3(&l[1]));
            put3(rows, J2 + JA, &cross(&l[1], &r[1]));
            if delta > 0.0 {
                if self.backlash > 0.0 {
                    rows[LO] = -INFINITY;
                    rows[HI] = 0.0;
                }
                rows[RHS] = -fps * self.erp * (delta - self.backlash);
            } else {
                if self.backlash > 0.0 {
                    rows[LO] = 0.0;
                    rows[HI] = INFINITY;
                }
                rows[RHS] = -fps * self.erp * (delta + self.backlash);
            }
        }
        rows[CFM] = self.cfm;
        self.contacts = c;
    }
}

// ---- dispatch from Joint

impl Joint {
    /// setRelativeValues for the joints in this file
    pub(crate) fn set_relative_values2(&mut self, bodies: &[Body]) {
        let (node, reverse) = (self.node, self.reverse);
        match &mut self.kind {
            JointKind::Hinge2(j) => j.set_relative(node, reverse, bodies),
            JointKind::Universal(j) => j.set_relative(node, reverse, bodies),
            JointKind::Pr(j) => j.set_relative(node, bodies),
            JointKind::Pu(j) => j.set_relative(node, reverse, bodies),
            JointKind::Piston(j) => j.set_relative(node, reverse, bodies),
            JointKind::DHinge(j) => j.ball.target_distance = dball_distance(node, bodies, &j.ball),
            _ => {}
        }
    }
}

// ---- API

impl World {
    fn new_limot(&self) -> LimitMotor {
        LimitMotor::new(&self.p)
    }
    pub fn joint_create_hinge2(&mut self, group: Option<super::world::GroupId>) -> JointId {
        let k = JointKind::Hinge2(Hinge2 {
            anchor1: [0.0; 4],
            anchor2: [0.0; 4],
            axis1: v3(1.0, 0.0, 0.0),
            axis2: v3(0.0, 1.0, 0.0),
            c0: 0.0,
            s0: 0.0,
            v1: v3(1.0, 0.0, 0.0),
            v2: v3(0.0, 1.0, 0.0),
            w1: [0.0; 4],
            w2: [0.0; 4],
            limot1: self.new_limot(),
            limot2: self.new_limot(),
            susp_erp: self.p.erp,
            susp_cfm: self.p.cfm,
        });
        self.add_joint(k, group)
    }
    fn universal_default(&self, axis1: V3, axis2: V3) -> Universal {
        Universal { anchor1: [0.0; 4], anchor2: [0.0; 4], axis1, axis2, qrel1: [0.0; 4], qrel2: [0.0; 4], limot1: self.new_limot(), limot2: self.new_limot() }
    }
    pub fn joint_create_universal(&mut self, group: Option<super::world::GroupId>) -> JointId {
        let k = JointKind::Universal(self.universal_default(v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0)));
        self.add_joint(k, group)
    }
    pub fn joint_create_pr(&mut self, group: Option<super::world::GroupId>) -> JointId {
        let k = JointKind::Pr(Pr {
            anchor2: [0.0; 4],
            axis_r1: v3(1.0, 0.0, 0.0),
            axis_r2: v3(1.0, 0.0, 0.0),
            axis_p1: v3(0.0, 1.0, 0.0),
            qrel: [0.0; 4],
            offset: [0.0; 4],
            limot_r: self.new_limot(),
            limot_p: self.new_limot(),
        });
        self.add_joint(k, group)
    }
    pub fn joint_create_pu(&mut self, group: Option<super::world::GroupId>) -> JointId {
        let k = JointKind::Pu(Pu { u: self.universal_default(v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)), axis_p1: v3(1.0, 0.0, 0.0), limot_p: self.new_limot() });
        self.add_joint(k, group)
    }
    pub fn joint_create_piston(&mut self, group: Option<super::world::GroupId>) -> JointId {
        let k = JointKind::Piston(Piston {
            anchor1: [0.0; 4],
            anchor2: [0.0; 4],
            axis1: v3(1.0, 0.0, 0.0),
            axis2: v3(1.0, 0.0, 0.0),
            qrel: [0.0; 4],
            limot_p: self.new_limot(),
            limot_r: self.new_limot(),
        });
        self.add_joint(k, group)
    }
    pub fn joint_create_plane2d(&mut self, group: Option<super::world::GroupId>) -> JointId {
        let k = JointKind::Plane2D(Plane2D { row_x: 0, row_y: 0, row_angle: 0, motor_x: self.new_limot(), motor_y: self.new_limot(), motor_angle: self.new_limot() });
        self.add_joint(k, group)
    }
    pub fn joint_create_lmotor(&mut self, group: Option<super::world::GroupId>) -> JointId {
        let lm = self.new_limot();
        let k = JointKind::LMotor(LMotor { num: 0, rel: [0; 3], axis: [[0.0; 4]; 3], limot: [lm; 3] });
        self.add_joint(k, group)
    }
    pub fn joint_create_dhinge(&mut self, group: Option<super::world::GroupId>) -> JointId {
        let ball = DBall { anchor1: [0.0; 4], anchor2: [0.0; 4], target_distance: 0.0, erp: self.p.erp, cfm: self.p.cfm };
        let k = JointKind::DHinge(DHinge { ball, axis1: [0.0; 4], axis2: [0.0; 4] });
        self.add_joint(k, group)
    }
    pub fn joint_create_transmission(&mut self, group: Option<super::world::GroupId>) -> JointId {
        let x = v3(1.0, 0.0, 0.0);
        let k = JointKind::Transmission(Transmission {
            mode: TRANSMISSION_PARALLEL_AXES,
            update: true,
            contacts: [[0.0; 4]; 2],
            axes: [x, x],
            anchors: [[0.0; 4]; 2],
            reference: [[0.0; 12]; 2],
            phase: [0.0; 2],
            radii: [0.0; 2],
            backlash: 0.0,
            ratio: 1.0,
            erp: self.p.erp,
            cfm: self.p.cfm,
        });
        self.add_joint(k, group)
    }
    pub fn joint_create_null(&mut self, group: Option<super::world::GroupId>) -> JointId {
        self.add_joint(JointKind::Null, group)
    }

    /// dJointSet{Hinge2,Universal,PU,Piston}Anchor and dJointSetPRAnchor
    pub fn joint_set_anchor2(&mut self, j: JointId, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        match &mut jt.kind {
            JointKind::Hinge2(h) => {
                set_anchors(node, bodies, x, y, z, &mut h.anchor1, &mut h.anchor2);
                h.make_v1_v2(node, bodies);
                h.make_w1_w2(node, bodies);
            }
            JointKind::Universal(u) => {
                set_anchors(node, bodies, x, y, z, &mut u.anchor1, &mut u.anchor2);
                u.initial_rotations(node, bodies);
            }
            JointKind::Pu(p) => {
                set_anchors(node, bodies, x, y, z, &mut p.u.anchor1, &mut p.u.anchor2);
                p.u.initial_rotations(node, bodies);
            }
            JointKind::Piston(p) => {
                set_anchors(node, bodies, x, y, z, &mut p.anchor1, &mut p.anchor2);
                p.qrel = initial_relative_rotation(node, bodies, p.qrel);
            }
            JointKind::Pr(p) => set_anchors(node, bodies, x, y, z, &mut p.offset, &mut p.anchor2),
            _ => {}
        }
    }
    /// dJointSetHinge2Axis1/2
    pub fn joint_set_hinge2_axis(&mut self, j: JointId, second: bool, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        if let JointKind::Hinge2(h) = &mut jt.kind {
            if second {
                set_axes(node, bodies, x, y, z, None, Some(&mut h.axis2));
            } else {
                set_axes(node, bodies, x, y, z, Some(&mut h.axis1), None);
            }
            h.update_axes(node, bodies);
        }
    }
    /// dJointSet{Universal,PU}Axis1/2, dJointSetPUAxis3 (`which` 0, 1, 2)
    pub fn joint_set_universal_axis(&mut self, j: JointId, which: usize, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let (node, rev) = (jt.node, jt.reverse);
        match &mut jt.kind {
            JointKind::Universal(u) if which < 2 => {
                u.set_axis(node, rev, bodies, which == 1, x, y, z);
                u.initial_rotations(node, bodies);
            }
            JointKind::Pu(p) => {
                if which < 2 {
                    p.u.set_axis(node, rev, bodies, which == 1, x, y, z);
                } else {
                    set_axes(node, bodies, x, y, z, Some(&mut p.axis_p1), None);
                }
                p.u.initial_rotations(node, bodies);
            }
            _ => {}
        }
    }
    /// dJointSetPRAxis1 (prismatic) / dJointSetPRAxis2 (rotoide)
    pub fn joint_set_pr_axis(&mut self, j: JointId, rotoide: bool, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        if let JointKind::Pr(p) = &mut jt.kind {
            if rotoide {
                set_axes(node, bodies, x, y, z, Some(&mut p.axis_r1), Some(&mut p.axis_r2));
            } else {
                set_axes(node, bodies, x, y, z, Some(&mut p.axis_p1), None);
            }
            p.qrel = initial_relative_rotation(node, bodies, p.qrel);
        }
    }
    /// dJointSetPistonAxis
    pub fn joint_set_piston_axis(&mut self, j: JointId, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        if let JointKind::Piston(p) = &mut jt.kind {
            set_axes(node, bodies, x, y, z, Some(&mut p.axis1), Some(&mut p.axis2));
            p.qrel = initial_relative_rotation(node, bodies, p.qrel);
        }
    }
    /// dJointSet{Hinge2,Universal,PR,PU,Piston,Plane2D*,LMotor,Transmission,DHinge}Param.
    /// Plane2D takes its motor groups as 0x000 x, 0x100 y, 0x200 angle.
    pub fn joint_set_param2(&mut self, j: JointId, parameter: i32, value: Real) {
        let group = parameter & 0xff00;
        let p = parameter & 0xff;
        match &mut self.joints[j].kind {
            JointKind::Hinge2(h) => {
                if group == 0x100 {
                    h.limot2.set(p, value);
                } else if parameter == PARAM_SUSPENSION_ERP {
                    h.susp_erp = value;
                } else if parameter == PARAM_SUSPENSION_CFM {
                    h.susp_cfm = value;
                } else {
                    h.limot1.set(parameter, value);
                }
            }
            JointKind::Universal(u) => {
                if group == 0x100 {
                    u.limot2.set(p, value);
                } else {
                    u.limot1.set(parameter, value);
                }
            }
            JointKind::Pr(r) => {
                if group == 0x100 {
                    r.limot_r.set(p, value);
                } else {
                    r.limot_p.set(parameter, value);
                }
            }
            JointKind::Piston(r) => {
                if group == 0x100 {
                    r.limot_r.set(p, value);
                } else {
                    r.limot_p.set(parameter, value);
                }
            }
            JointKind::Pu(u) => match group {
                0x000 => u.u.limot1.set(parameter, value),
                0x100 => u.u.limot2.set(p, value),
                0x200 => u.limot_p.set(p, value),
                _ => {}
            },
            JointKind::Plane2D(pl) => match group {
                0x000 => pl.motor_x.set(p, value),
                0x100 => pl.motor_y.set(p, value),
                0x200 => pl.motor_angle.set(p, value),
                _ => {}
            },
            JointKind::LMotor(l) => {
                let anum = (parameter >> 8).clamp(0, 2) as usize;
                l.limot[anum].set(p, value);
            }
            JointKind::Transmission(t) => match parameter {
                PARAM_CFM => t.cfm = value,
                PARAM_ERP => t.erp = value,
                _ => {}
            },
            JointKind::DHinge(d) => match parameter {
                PARAM_CFM => d.ball.cfm = value,
                PARAM_ERP => d.ball.erp = value,
                _ => {}
            },
            _ => {}
        }
    }
    /// dJointSetLMotorNumAxes
    pub fn joint_set_lmotor_num_axes(&mut self, j: JointId, num: i32) {
        if let JointKind::LMotor(l) = &mut self.joints[j].kind {
            l.num = num.clamp(0, 3) as usize;
        }
    }
    /// dJointSetLMotorAxis
    pub fn joint_set_lmotor_axis(&mut self, j: JointId, anum: i32, rel: i32, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        if let JointKind::LMotor(l) = &mut jt.kind {
            let anum = anum.clamp(0, 2) as usize;
            let rel = if node[1].is_none() && rel == 2 { 1 } else { rel };
            l.rel[anum] = rel;
            let r = v3(x, y, z);
            if rel > 0 {
                let b = if rel == 1 { node[0] } else { node[1] };
                l.axis[anum] = mul1_331(&bodies[b.unwrap()].r, &r);
            } else {
                set3(&mut l.axis[anum], &r);
            }
            normalize3(&mut l.axis[anum]);
        }
    }
    /// dJointSetDHingeAxis
    pub fn joint_set_dhinge_axis(&mut self, j: JointId, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        if let JointKind::DHinge(d) = &mut jt.kind {
            d.axis1 = bodies[node[0].unwrap()].vector_from_world(x, y, z);
            d.axis2 = match node[1] {
                Some(b1) => bodies[b1].vector_from_world(x, y, z),
                None => v3(x, y, z),
            };
            normalize3(&mut d.axis1);
            normalize3(&mut d.axis2);
        }
    }
    /// dJointSetDHingeAnchor1/2
    pub fn joint_set_dhinge_anchor(&mut self, j: JointId, second: bool, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        let slot = if jt.reverse { !second } else { second } as usize;
        if let JointKind::DHinge(d) = &mut jt.kind {
            let a = match node[slot] {
                Some(b) => bodies[b].pos_rel_point(x, y, z),
                None => v3(x, y, z),
            };
            if slot == 0 {
                d.ball.anchor1[..3].copy_from_slice(&a[..3]);
            } else {
                d.ball.anchor2[..3].copy_from_slice(&a[..3]);
            }
        }
        jt.set_relative_values(bodies);
    }
    /// dJointSetTransmission{Mode,Ratio,Radius1/2,Backlash}
    pub fn joint_set_transmission(&mut self, j: JointId, mode: i32, ratio: Real, radii: [Real; 2], backlash: Real) {
        if let JointKind::Transmission(t) = &mut self.joints[j].kind {
            t.mode = mode;
            t.ratio = ratio;
            t.radii = radii;
            t.backlash = backlash;
        }
    }
    /// dJointSetTransmissionAxis (both bodies) / Axis1 / Axis2 (`which` 2, 0, 1)
    pub fn joint_set_transmission_axis(&mut self, j: JointId, which: usize, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        if let JointKind::Transmission(t) = &mut jt.kind {
            for i in 0..2 {
                if which == i || which == 2 {
                    if let Some(b) = node[i] {
                        t.axes[i] = bodies[b].vector_from_world(x, y, z);
                        normalize3(&mut t.axes[i]);
                    }
                }
            }
            t.update = true;
        }
    }
    /// dJointSetTransmissionAnchor1/2
    pub fn joint_set_transmission_anchor(&mut self, j: JointId, i: usize, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        if let JointKind::Transmission(t) = &mut jt.kind {
            if let Some(b) = node[i] {
                t.anchors[i] = bodies[b].pos_rel_point(x, y, z);
            }
            t.update = true;
        }
    }

    // ---- getters

    pub fn joint_get_hinge2_angle1(&self, j: JointId) -> Real {
        match &self.joints[j].kind {
            JointKind::Hinge2(h) => h.measure_angle1(self.joints[j].node, &self.bodies),
            _ => 0.0,
        }
    }
    pub fn joint_get_hinge2_angle2(&self, j: JointId) -> Real {
        match &self.joints[j].kind {
            JointKind::Hinge2(h) => h.measure_angle2(self.joints[j].node, &self.bodies),
            _ => 0.0,
        }
    }
    /// dJointGetUniversalAngles / dJointGetPUAngles
    pub fn joint_get_universal_angles(&self, j: JointId) -> (Real, Real) {
        let jt = &self.joints[j];
        let (u, pu) = match &jt.kind {
            JointKind::Universal(u) => (u, false),
            JointKind::Pu(p) => (&p.u, true),
            _ => return (0.0, 0.0),
        };
        let (a1, a2) = u.get_angles(jt.node, &self.bodies);
        match (jt.reverse, pu) {
            (true, false) => (a2, -a1),
            (true, true) => (a2, a1),
            _ => (a1, a2),
        }
    }
    /// dJointGetUniversalAngle1/2
    pub fn joint_get_universal_angle(&self, j: JointId, second: bool) -> Real {
        let jt = &self.joints[j];
        let JointKind::Universal(u) = &jt.kind else { return 0.0 };
        match (jt.reverse, second) {
            (false, false) => u.angle1(jt.node, &self.bodies),
            (false, true) => u.angle2(jt.node, &self.bodies),
            (true, false) => u.angle2(jt.node, &self.bodies),
            (true, true) => -u.angle1(jt.node, &self.bodies),
        }
    }
    /// dJointGet{PR,PU,Piston}Position
    pub fn joint_get_position2(&self, j: JointId) -> Real {
        let jt = &self.joints[j];
        match &jt.kind {
            JointKind::Pr(p) => p.position(jt.node, jt.reverse, &self.bodies),
            JointKind::Pu(p) => p.position(jt.node, jt.reverse, &self.bodies),
            JointKind::Piston(p) => p.position(jt.node, jt.reverse, &self.bodies),
            _ => 0.0,
        }
    }
    /// dJointGet{PR,Piston}Angle
    pub fn joint_get_angle2(&self, j: JointId) -> Real {
        let jt = &self.joints[j];
        let (axis, qrel) = match &jt.kind {
            JointKind::Pr(p) => (p.axis_r1, p.qrel),
            JointKind::Piston(p) => (p.axis1, p.qrel),
            _ => return 0.0,
        };
        if jt.node[0].is_none() {
            return 0.0;
        }
        let a = hinge_angle(&self.bodies, jt.node, &axis, &qrel);
        if jt.reverse {
            -a
        } else {
            a
        }
    }
    /// dJointGetTransmissionAngle1/2 and ContactPoint1/2
    pub fn joint_get_transmission_state(&self, j: JointId) -> Option<([Real; 2], [V3; 2])> {
        match &self.joints[j].kind {
            JointKind::Transmission(t) => Some((t.phase, t.contacts)),
            _ => None,
        }
    }
}
