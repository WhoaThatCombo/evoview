//! ODE's joint API (dJointCreate*, dJointSet*/Get*), on the world.

use super::joints::*;
use super::math::*;
use super::world::{BodyId, GroupId, JointId, World};

impl World {
    pub fn joint_create_ball(&mut self, group: Option<GroupId>) -> JointId {
        let k = JointKind::Ball(Ball { anchor1: [0.0; 4], anchor2: [0.0; 4], erp: self.p.erp, cfm: self.p.cfm });
        self.add_joint(k, group)
    }
    pub fn joint_create_hinge(&mut self, group: Option<GroupId>) -> JointId {
        let k = JointKind::Hinge(Hinge {
            anchor1: [0.0; 4],
            anchor2: [0.0; 4],
            axis1: v3(1.0, 0.0, 0.0),
            axis2: v3(1.0, 0.0, 0.0),
            qrel: [0.0; 4],
            limot: LimitMotor::new(&self.p),
        });
        self.add_joint(k, group)
    }
    pub fn joint_create_slider(&mut self, group: Option<GroupId>) -> JointId {
        let k = JointKind::Slider(Slider { axis1: v3(1.0, 0.0, 0.0), qrel: [0.0; 4], offset: [0.0; 4], limot: LimitMotor::new(&self.p) });
        self.add_joint(k, group)
    }
    pub fn joint_create_fixed(&mut self, group: Option<GroupId>) -> JointId {
        let k = JointKind::Fixed(Fixed { qrel: [0.0; 4], offset: [0.0; 4], erp: self.p.erp, cfm: self.p.cfm });
        self.add_joint(k, group)
    }
    pub fn joint_create_dball(&mut self, group: Option<GroupId>) -> JointId {
        let k = JointKind::DBall(DBall { anchor1: [0.0; 4], anchor2: [0.0; 4], target_distance: 0.0, erp: self.p.erp, cfm: self.p.cfm });
        self.add_joint(k, group)
    }
    pub fn joint_create_contact(&mut self, group: Option<GroupId>, c: &Contact) -> JointId {
        let k = JointKind::Contact(ContactJoint { contact: *c, the_m: 0 });
        self.add_joint(k, group)
    }
    pub fn joint_create_amotor(&mut self, group: Option<GroupId>) -> JointId {
        let lm = LimitMotor::new(&self.p);
        let k = JointKind::AMotor(AMotor {
            mode: AMOTOR_USER,
            num: 0,
            rel: [REL_GLOBAL; 3],
            axis: [[0.0; 4]; 3],
            references: [[0.0; 4]; 2],
            angle: [0.0; 3],
            limot: [lm; 4],
        });
        self.add_joint(k, group)
    }

    /// dJointSetFeedback: record the forces this joint applies each step
    pub fn joint_enable_feedback(&mut self, j: JointId, on: bool) {
        self.joints[j].feedback = if on { Some(Feedback::default()) } else { None };
    }
    pub fn joint_feedback(&self, j: JointId) -> Option<Feedback> {
        self.joints[j].feedback
    }

    // ---- ball

    pub fn joint_set_ball_anchor(&mut self, j: JointId, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        if let JointKind::Ball(b) = &mut jt.kind {
            set_anchors(node, bodies, x, y, z, &mut b.anchor1, &mut b.anchor2);
        }
    }
    pub fn joint_get_ball_anchor(&self, j: JointId) -> V3 {
        let jt = &self.joints[j];
        match &jt.kind {
            JointKind::Ball(b) if jt.reverse => get_anchor2(jt.node, &self.bodies, &b.anchor2),
            JointKind::Ball(b) => get_anchor(jt.node, &self.bodies, &b.anchor1),
            _ => [0.0; 4],
        }
    }
    pub fn joint_get_ball_anchor2(&self, j: JointId) -> V3 {
        let jt = &self.joints[j];
        match &jt.kind {
            JointKind::Ball(b) if jt.reverse => get_anchor(jt.node, &self.bodies, &b.anchor1),
            JointKind::Ball(b) => get_anchor2(jt.node, &self.bodies, &b.anchor2),
            _ => [0.0; 4],
        }
    }

    // ---- hinge

    pub fn joint_set_hinge_anchor(&mut self, j: JointId, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        if let JointKind::Hinge(h) = &mut jt.kind {
            set_anchors(node, bodies, x, y, z, &mut h.anchor1, &mut h.anchor2);
            h.qrel = initial_relative_rotation(node, bodies, h.qrel);
        }
    }
    pub fn joint_set_hinge_axis(&mut self, j: JointId, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        if let JointKind::Hinge(h) = &mut jt.kind {
            set_axes(node, bodies, x, y, z, Some(&mut h.axis1), Some(&mut h.axis2));
            h.qrel = initial_relative_rotation(node, bodies, h.qrel);
        }
    }
    pub fn joint_get_hinge_anchor(&self, j: JointId) -> V3 {
        let jt = &self.joints[j];
        match &jt.kind {
            JointKind::Hinge(h) if jt.reverse => get_anchor2(jt.node, &self.bodies, &h.anchor2),
            JointKind::Hinge(h) => get_anchor(jt.node, &self.bodies, &h.anchor1),
            _ => [0.0; 4],
        }
    }
    pub fn joint_get_hinge_axis(&self, j: JointId) -> V3 {
        let jt = &self.joints[j];
        match &jt.kind {
            JointKind::Hinge(h) => get_axis(jt.node, &self.bodies, &h.axis1),
            _ => [0.0; 4],
        }
    }
    pub fn joint_set_hinge_param(&mut self, j: JointId, param: i32, value: Real) {
        if let JointKind::Hinge(h) = &mut self.joints[j].kind {
            h.limot.set(param, value);
        }
    }
    pub fn joint_get_hinge_angle(&self, j: JointId) -> Real {
        let jt = &self.joints[j];
        match (&jt.kind, jt.node[0]) {
            (JointKind::Hinge(h), Some(_)) => {
                let a = hinge_angle(&self.bodies, jt.node, &h.axis1, &h.qrel);
                if jt.reverse {
                    -a
                } else {
                    a
                }
            }
            _ => 0.0,
        }
    }
    pub fn joint_get_hinge_angle_rate(&self, j: JointId) -> Real {
        let jt = &self.joints[j];
        match (&jt.kind, jt.node[0]) {
            (JointKind::Hinge(h), Some(b0)) => {
                let axis = mul0_331(&self.bodies[b0].r, &h.axis1);
                let mut rate = dot3(&axis, &self.bodies[b0].avel);
                if let Some(b1) = jt.node[1] {
                    rate -= dot3(&axis, &self.bodies[b1].avel);
                }
                if jt.reverse {
                    -rate
                } else {
                    rate
                }
            }
            _ => 0.0,
        }
    }
    /// dJointAddHingeTorque
    pub fn joint_add_hinge_torque(&mut self, j: JointId, mut torque: Real) {
        let jt = &self.joints[j];
        if let JointKind::Hinge(h) = &jt.kind {
            if jt.reverse {
                torque = -torque;
            }
            let mut axis = get_axis(jt.node, &self.bodies, &h.axis1);
            axis[0] *= torque;
            axis[1] *= torque;
            axis[2] *= torque;
            let node = jt.node;
            if let Some(b) = node[0] {
                self.body_add_torque(b, axis[0], axis[1], axis[2]);
            }
            if let Some(b) = node[1] {
                self.body_add_torque(b, -axis[0], -axis[1], -axis[2]);
            }
        }
    }

    // ---- slider

    pub fn joint_set_slider_axis(&mut self, j: JointId, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        if let JointKind::Slider(s) = &mut jt.kind {
            set_axes(node, bodies, x, y, z, Some(&mut s.axis1), None);
            s.offset = slider_offset(node, bodies, s.offset);
            s.qrel = initial_relative_rotation(node, bodies, s.qrel);
        }
    }
    pub fn joint_set_slider_param(&mut self, j: JointId, param: i32, value: Real) {
        if let JointKind::Slider(s) = &mut self.joints[j].kind {
            s.limot.set(param, value);
        }
    }
    pub fn joint_get_slider_position(&self, j: JointId) -> Real {
        let jt = &self.joints[j];
        match &jt.kind {
            JointKind::Slider(s) => slider_position(&self.bodies, jt.node, jt.reverse, s),
            _ => 0.0,
        }
    }
    pub fn joint_get_slider_position_rate(&self, j: JointId) -> Real {
        let jt = &self.joints[j];
        match &jt.kind {
            JointKind::Slider(s) => {
                let b0 = &self.bodies[jt.node[0].unwrap()];
                let ax1 = mul0_331(&b0.r, &s.axis1);
                if let Some(b1) = jt.node[1] {
                    dot3(&ax1, &b0.lvel) - dot3(&ax1, &self.bodies[b1].lvel)
                } else {
                    let rate = dot3(&ax1, &b0.lvel);
                    if jt.reverse {
                        -rate
                    } else {
                        rate
                    }
                }
            }
            _ => 0.0,
        }
    }

    // ---- fixed

    /// dJointSetFixed: hold the current relative position and rotation
    pub fn joint_set_fixed(&mut self, j: JointId) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        if let JointKind::Fixed(f) = &mut jt.kind {
            if let Some(b0) = node[0] {
                if let Some(b1) = node[1] {
                    let mut ofs = [0.0; 4];
                    for i in 0..4 {
                        ofs[i] = bodies[b0].pos[i] - bodies[b1].pos[i];
                    }
                    f.offset = mul1_331(&bodies[b0].r, &ofs);
                } else {
                    let p = bodies[b0].pos;
                    f.offset[0] = p[0];
                    f.offset[1] = p[1];
                    f.offset[2] = p[2];
                }
            }
            f.qrel = initial_relative_rotation(node, bodies, f.qrel);
        }
    }

    // ---- dball

    pub fn joint_set_dball_anchor1(&mut self, j: JointId, x: Real, y: Real, z: Real) {
        self.dball_anchor(j, x, y, z, false);
    }
    pub fn joint_set_dball_anchor2(&mut self, j: JointId, x: Real, y: Real, z: Real) {
        self.dball_anchor(j, x, y, z, true);
    }
    fn dball_anchor(&mut self, j: JointId, x: Real, y: Real, z: Real, second: bool) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let node = jt.node;
        let slot = if jt.reverse { !second } else { second } as usize;
        if let JointKind::DBall(d) = &mut jt.kind {
            let a = match node[slot] {
                Some(b) => bodies[b].pos_rel_point(x, y, z),
                None => v3(x, y, z),
            };
            if slot == 0 {
                d.anchor1[..3].copy_from_slice(&a[..3]);
            } else {
                d.anchor2[..3].copy_from_slice(&a[..3]);
            }
        }
        jt.set_relative_values(bodies);
    }
    pub fn joint_set_dball_distance(&mut self, j: JointId, dist: Real) {
        if let JointKind::DBall(d) = &mut self.joints[j].kind {
            d.target_distance = dist;
        }
    }
    pub fn joint_get_dball_distance(&self, j: JointId) -> Real {
        match &self.joints[j].kind {
            JointKind::DBall(d) => d.target_distance,
            _ => 0.0,
        }
    }

    /// dJointSet{Ball,Fixed,DBall}Param: their own ERP / CFM
    pub fn joint_set_erp_cfm_param(&mut self, j: JointId, param: i32, value: Real) {
        let (erp, cfm) = match &mut self.joints[j].kind {
            JointKind::Ball(b) => (&mut b.erp, &mut b.cfm),
            JointKind::Fixed(f) => (&mut f.erp, &mut f.cfm),
            JointKind::DBall(d) => (&mut d.erp, &mut d.cfm),
            _ => return,
        };
        match param {
            PARAM_CFM => *cfm = value,
            PARAM_ERP => *erp = value,
            _ => {}
        }
    }

    // ---- amotor

    pub fn joint_set_amotor_mode(&mut self, j: JointId, mode: i32) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let (node, rev) = (jt.node, jt.reverse);
        if let JointKind::AMotor(a) = &mut jt.kind {
            a.mode = mode;
            if mode == AMOTOR_EULER {
                a.num = 3;
                amotor_set_euler_references(a, node, rev, bodies);
            }
        }
    }
    pub fn joint_set_amotor_num_axes(&mut self, j: JointId, num: usize) {
        if let JointKind::AMotor(a) = &mut self.joints[j].kind {
            a.num = if a.mode == AMOTOR_EULER { 3 } else { num.min(3) };
        }
    }
    /// dJointSetAMotorAxis; rel 0 = global, 1 = body 1, 2 = body 2
    pub fn joint_set_amotor_axis(&mut self, j: JointId, anum: usize, rel: i32, x: Real, y: Real, z: Real) {
        let (joints, bodies) = (&mut self.joints, &self.bodies);
        let jt = &mut joints[j];
        let (node, rev) = (jt.node, jt.reverse);
        if let JointKind::AMotor(a) = &mut jt.kind {
            let anum = anum.min(2);
            let r = v3(x, y, z);
            let mut rel = rel;
            let is_body = rel == REL_BODY1 || rel == REL_BODY2;
            if is_body && rev {
                rel = REL_BODY1 + REL_BODY2 - rel;
            }
            a.rel[anum] = rel;
            let mut assigned = false;
            if is_body {
                if rel == REL_BODY1 {
                    a.axis[anum] = mul1_331(&bodies[node[0].unwrap()].r, &r);
                    assigned = true;
                } else if let Some(b1) = node[1] {
                    a.axis[anum] = mul1_331(&bodies[b1].r, &r);
                    assigned = true;
                }
            }
            if !assigned {
                a.axis[anum] = r;
            }
            normalize3(&mut a.axis[anum]);
            if a.mode == AMOTOR_EULER {
                amotor_set_euler_references(a, node, rev, bodies);
            }
        }
    }
    pub fn joint_set_amotor_angle(&mut self, j: JointId, anum: usize, angle: Real) {
        if let JointKind::AMotor(a) = &mut self.joints[j].kind {
            if a.mode == AMOTOR_USER {
                a.angle[anum.min(2)] = angle;
            }
        }
    }
    /// dJointSetAMotorParam (axis in the high byte: dParamFMax2 etc.)
    pub fn joint_set_amotor_param(&mut self, j: JointId, param: i32, value: Real) {
        if let JointKind::AMotor(a) = &mut self.joints[j].kind {
            let anum = ((param >> 8).clamp(0, 2)) as usize;
            a.limot[anum].set(param & 0xff, value);
        }
    }
    pub fn joint_get_amotor_angle(&self, j: JointId, anum: usize) -> Real {
        match &self.joints[j].kind {
            JointKind::AMotor(a) => a.angle[anum.min(2)],
            _ => 0.0,
        }
    }
    /// dJointAddAMotorTorques
    pub fn joint_add_amotor_torques(&mut self, j: JointId, t1: Real, t2: Real, t3: Real) {
        let jt = &self.joints[j];
        let (node, rev) = (jt.node, jt.reverse);
        let JointKind::AMotor(a) = &jt.kind else { return };
        let num = a.num;
        if num == 0 {
            return;
        }
        let axes = amotor_global_axes(a, node, rev, &self.bodies);
        let tv = if !rev { v3(t1, t2, t3) } else { v3(-t1, -t2, -t3) };
        let sum = match num {
            3 => {
                let (a0, a1, a2) = (&axes[2], &axes[1], &axes[0]);
                let (s0, s1, s2) = (tv[2], tv[1], tv[0]);
                [s0 * a0[0] + s1 * a1[0] + s2 * a2[0], s0 * a0[1] + s1 * a1[1] + s2 * a2[1], s0 * a0[2] + s1 * a1[2] + s2 * a2[2], 0.0]
            }
            2 => add_scaled_vectors3(&axes[1], &axes[0], tv[1], tv[0]),
            _ => scaled3(&axes[0], tv[0]),
        };
        let b0: BodyId = node[0].unwrap();
        self.body_add_torque(b0, sum[0], sum[1], sum[2]);
        if let Some(b1) = node[1] {
            self.body_add_torque(b1, -sum[0], -sum[1], -sum[2]);
        }
    }
}
