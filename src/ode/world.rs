//! ODE's world and bodies (ode.cpp, objects.cpp, util.cpp).
//!
//! Bodies and joints live in slot vectors and are referred to by index.
//! ODE keeps them in singly linked lists with new objects at the head; the
//! island search walks those lists, so the order decides the order of the
//! rows the solver sees. `order` lists live objects oldest first and is
//! walked backwards, which is the same thing.

use super::joints::{Joint, JointKind};
use super::math::*;

pub type BodyId = usize;
pub type JointId = usize;
pub type GroupId = usize;

pub const FINITE_ROTATION: u32 = 1;
pub const FINITE_ROTATION_AXIS: u32 = 2;
pub const DISABLED: u32 = 4;
pub const NO_GRAVITY: u32 = 8;
pub const AUTO_DISABLE: u32 = 16;
pub const LINEAR_DAMPING: u32 = 32;
pub const ANGULAR_DAMPING: u32 = 64;
pub const MAX_ANGULAR_SPEED: u32 = 128;
pub const GYROSCOPIC: u32 = 256;

#[derive(Clone, Copy, Debug)]
pub struct AutoDisable {
    pub idle_time: Real,
    pub idle_steps: i32,
    pub average_samples: u32,
    pub linear_average_threshold: Real,
    pub angular_average_threshold: Real,
}
impl Default for AutoDisable {
    fn default() -> Self {
        AutoDisable { idle_time: 0.0, idle_steps: 10, average_samples: 1, linear_average_threshold: 0.01 * 0.01, angular_average_threshold: 0.01 * 0.01 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Damping {
    pub linear_scale: Real,
    pub angular_scale: Real,
    pub linear_threshold: Real,
    pub angular_threshold: Real,
}
impl Default for Damping {
    fn default() -> Self {
        Damping { linear_scale: 0.0, angular_scale: 0.0, linear_threshold: 0.01 * 0.01, angular_threshold: 0.01 * 0.01 }
    }
}

#[derive(Clone, Debug)]
pub struct Body {
    /// (joint, the body at its other end), most recently attached LAST
    /// (ODE's list head is our tail)
    pub(crate) joints: Vec<(JointId, Option<BodyId>)>,
    /// geoms on this body, most recently attached LAST (ODE's b->geom head)
    pub(crate) geoms: Vec<super::collision::GeomId>,
    pub flags: u32,
    pub mass: Mass,
    pub inv_i: M3,
    pub inv_mass: Real,
    pub pos: V3,
    pub r: M3,
    pub q: Quat,
    pub lvel: V3,
    pub avel: V3,
    pub facc: V3,
    pub tacc: V3,
    pub finite_rot_axis: V3,
    pub adis: AutoDisable,
    pub adis_timeleft: Real,
    pub adis_stepsleft: i32,
    average_lvel: Vec<V3>,
    average_avel: Vec<V3>,
    average_counter: u32,
    average_ready: bool,
    pub damping: Damping,
    pub max_angular_speed: Real,
    pub(crate) tag: i32,
    pub(crate) alive: bool,
}

/// dxQuickStepParameters
#[derive(Clone, Copy, Debug)]
pub struct QuickStep {
    pub num_iterations: u32,
    pub w: Real,
}
/// dxContactParameters
#[derive(Clone, Copy, Debug)]
pub struct ContactParams {
    pub max_vel: Real,
    pub min_depth: Real,
}

/// The parts of dxWorld a joint reads while building its rows.
#[derive(Clone, Copy, Debug)]
pub struct WorldParams {
    pub gravity: V3,
    pub erp: Real,
    pub cfm: Real,
    pub contactp: ContactParams,
}

pub struct World {
    pub bodies: Vec<Body>,
    /// the collision layer's geoms and spaces (collision.rs)
    pub geoms: Vec<super::collision::Geom>,
    pub joints: Vec<Joint>,
    pub(crate) body_order: Vec<BodyId>,
    pub(crate) joint_order: Vec<JointId>,
    /// joint groups: members in creation order
    pub(crate) groups: Vec<Vec<JointId>>,
    free_bodies: Vec<BodyId>,
    free_joints: Vec<JointId>,
    pub p: WorldParams,
    pub adis: AutoDisable,
    pub body_flags: u32,
    pub qs: QuickStep,
    pub damping: Damping,
    pub max_angular_speed: Real,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// dWorldCreate
    pub fn new() -> World {
        World {
            bodies: Vec::new(),
            geoms: Vec::new(),
            joints: Vec::new(),
            body_order: Vec::new(),
            joint_order: Vec::new(),
            groups: Vec::new(),
            free_bodies: Vec::new(),
            free_joints: Vec::new(),
            p: WorldParams { gravity: [0.0; 4], erp: 0.2, cfm: 1e-10, contactp: ContactParams { max_vel: INFINITY, min_depth: 0.0 } },
            adis: AutoDisable::default(),
            body_flags: 0,
            qs: QuickStep { num_iterations: 20, w: 1.3 },
            damping: Damping::default(),
            max_angular_speed: INFINITY,
        }
    }
    pub fn set_gravity(&mut self, x: Real, y: Real, z: Real) {
        self.p.gravity = [x, y, z, 0.0];
    }
    pub fn set_erp(&mut self, erp: Real) {
        self.p.erp = erp;
    }
    pub fn set_cfm(&mut self, cfm: Real) {
        self.p.cfm = cfm;
    }
    pub fn set_quickstep_iterations(&mut self, n: u32) {
        self.qs.num_iterations = n;
    }
    pub fn set_quickstep_w(&mut self, w: Real) {
        self.qs.w = w;
    }
    pub fn set_contact_max_correcting_vel(&mut self, v: Real) {
        self.p.contactp.max_vel = v;
    }
    pub fn set_contact_surface_layer(&mut self, d: Real) {
        self.p.contactp.min_depth = d;
    }

    // ---- bodies

    /// dBodyCreate
    pub fn body_create(&mut self) -> BodyId {
        let mut b = Body {
            joints: Vec::new(),
            geoms: Vec::new(),
            flags: 0,
            mass: Mass::parameters(1.0, [0.0; 3], 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            inv_i: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            inv_mass: 1.0,
            pos: [0.0; 4],
            r: IDENTITY_R,
            q: IDENTITY_Q,
            lvel: [0.0; 4],
            avel: [0.0; 4],
            facc: [0.0; 4],
            tacc: [0.0; 4],
            finite_rot_axis: [0.0; 4],
            adis: self.adis,
            adis_timeleft: 0.0,
            adis_stepsleft: 0,
            average_lvel: Vec::new(),
            average_avel: Vec::new(),
            average_counter: 0,
            average_ready: false,
            damping: self.damping,
            max_angular_speed: self.max_angular_speed,
            tag: 0,
            alive: true,
        };
        // dBodySetAutoDisableDefaults -> dBodySetAutoDisableFlag
        if self.body_flags & AUTO_DISABLE == 0 {
            b.flags &= !(AUTO_DISABLE | DISABLED);
            b.adis.idle_steps = self.adis.idle_steps;
            b.adis.idle_time = self.adis.idle_time;
            set_average_samples(&mut b, self.adis.average_samples);
        } else {
            b.flags |= AUTO_DISABLE;
        }
        b.adis_stepsleft = b.adis.idle_steps;
        b.adis_timeleft = b.adis.idle_time;
        b.average_counter = 0;
        b.average_ready = false;
        let n = b.adis.average_samples;
        set_average_samples(&mut b, n);
        // dBodySetDampingDefaults
        let mask = LINEAR_DAMPING | ANGULAR_DAMPING;
        b.flags &= !mask;
        b.flags |= self.body_flags & mask;
        b.flags |= self.body_flags & MAX_ANGULAR_SPEED;
        b.flags |= GYROSCOPIC;
        let id = if let Some(id) = self.free_bodies.pop() {
            self.bodies[id] = b;
            id
        } else {
            self.bodies.push(b);
            self.bodies.len() - 1
        };
        self.body_order.push(id);
        id
    }
    /// dBodyDestroy
    pub fn body_destroy(&mut self, id: BodyId) {
        for g in self.bodies[id].geoms.clone().into_iter().rev() {
            self.geom_set_body(g, None);
        }
        let list = std::mem::take(&mut self.bodies[id].joints);
        for &(j, _) in list.iter().rev() {
            let jt = &mut self.joints[j];
            let which = if jt.node[0] == Some(id) { 0 } else { 1 };
            jt.node[which] = None;
            self.remove_joint_references(j);
        }
        self.body_order.retain(|&b| b != id);
        self.bodies[id].alive = false;
        self.free_bodies.push(id);
    }
    pub fn body(&self, id: BodyId) -> &Body {
        &self.bodies[id]
    }
    pub fn body_mut(&mut self, id: BodyId) -> &mut Body {
        &mut self.bodies[id]
    }
    pub fn body_set_position(&mut self, id: BodyId, x: Real, y: Real, z: Real) {
        let b = &mut self.bodies[id];
        b.pos[0] = x;
        b.pos[1] = y;
        b.pos[2] = z;
        self.body_geoms_moved(id);
    }
    /// dBodySetRotation
    pub fn body_set_rotation(&mut self, id: BodyId, r: &M3) {
        let b = &mut self.bodies[id];
        b.r = *r;
        orthogonalize_r(&mut b.r);
        b.q = q_from_r(r);
        normalize4(&mut b.q);
        self.body_geoms_moved(id);
    }
    /// dBodySetQuaternion
    pub fn body_set_quaternion(&mut self, id: BodyId, q: &Quat) {
        let b = &mut self.bodies[id];
        b.q = *q;
        normalize4(&mut b.q);
        b.r = r_from_q(&b.q);
        self.body_geoms_moved(id);
    }
    pub fn body_set_linear_vel(&mut self, id: BodyId, x: Real, y: Real, z: Real) {
        let b = &mut self.bodies[id];
        b.lvel[0] = x;
        b.lvel[1] = y;
        b.lvel[2] = z;
    }
    pub fn body_set_angular_vel(&mut self, id: BodyId, x: Real, y: Real, z: Real) {
        let b = &mut self.bodies[id];
        b.avel[0] = x;
        b.avel[1] = y;
        b.avel[2] = z;
    }
    /// dBodySetMass
    pub fn body_set_mass(&mut self, id: BodyId, m: &Mass) {
        let b = &mut self.bodies[id];
        b.mass = *m;
        match invert_pd3(&b.mass.i) {
            Some(inv) => b.inv_i = inv,
            None => b.inv_i = IDENTITY_R,
        }
        b.inv_mass = recip(b.mass.mass);
    }
    pub fn body_set_kinematic(&mut self, id: BodyId) {
        let b = &mut self.bodies[id];
        b.inv_i = [0.0; 12];
        b.inv_mass = 0.0;
    }
    pub fn body_set_dynamic(&mut self, id: BodyId) {
        let m = self.bodies[id].mass;
        self.body_set_mass(id, &m);
    }
    pub fn body_enable(&mut self, id: BodyId) {
        let b = &mut self.bodies[id];
        b.flags &= !DISABLED;
        b.adis_stepsleft = b.adis.idle_steps;
        b.adis_timeleft = b.adis.idle_time;
    }
    pub fn body_disable(&mut self, id: BodyId) {
        self.bodies[id].flags |= DISABLED;
    }
    pub fn body_set_gravity_mode(&mut self, id: BodyId, on: bool) {
        let b = &mut self.bodies[id];
        if on {
            b.flags &= !NO_GRAVITY;
        } else {
            b.flags |= NO_GRAVITY;
        }
    }
    pub fn body_set_gyroscopic_mode(&mut self, id: BodyId, on: bool) {
        let b = &mut self.bodies[id];
        if on {
            b.flags |= GYROSCOPIC;
        } else {
            b.flags &= !GYROSCOPIC;
        }
    }
    pub fn body_set_finite_rotation_mode(&mut self, id: BodyId, on: bool) {
        let b = &mut self.bodies[id];
        b.flags &= !(FINITE_ROTATION | FINITE_ROTATION_AXIS);
        if on {
            b.flags |= FINITE_ROTATION;
            if b.finite_rot_axis[0] != 0.0 || b.finite_rot_axis[1] != 0.0 || b.finite_rot_axis[2] != 0.0 {
                b.flags |= FINITE_ROTATION_AXIS;
            }
        }
    }
    pub fn body_set_finite_rotation_axis(&mut self, id: BodyId, x: Real, y: Real, z: Real) {
        let b = &mut self.bodies[id];
        b.finite_rot_axis = v3(x, y, z);
        if x != 0.0 || y != 0.0 || z != 0.0 {
            normalize3(&mut b.finite_rot_axis);
            b.flags |= FINITE_ROTATION_AXIS;
        } else {
            b.flags &= !FINITE_ROTATION_AXIS;
        }
    }
    pub fn body_set_linear_damping(&mut self, id: BodyId, scale: Real) {
        let b = &mut self.bodies[id];
        if scale != 0.0 {
            b.flags |= LINEAR_DAMPING;
        } else {
            b.flags &= !LINEAR_DAMPING;
        }
        b.damping.linear_scale = scale;
    }
    pub fn body_set_angular_damping(&mut self, id: BodyId, scale: Real) {
        let b = &mut self.bodies[id];
        if scale != 0.0 {
            b.flags |= ANGULAR_DAMPING;
        } else {
            b.flags &= !ANGULAR_DAMPING;
        }
        b.damping.angular_scale = scale;
    }
    pub fn body_set_max_angular_speed(&mut self, id: BodyId, max: Real) {
        let b = &mut self.bodies[id];
        if max < INFINITY {
            b.flags |= MAX_ANGULAR_SPEED;
        } else {
            b.flags &= !MAX_ANGULAR_SPEED;
        }
        b.max_angular_speed = max;
    }
    pub fn body_set_auto_disable_flag(&mut self, id: BodyId, on: bool) {
        let (steps, time, samples) = (self.adis.idle_steps, self.adis.idle_time, self.adis.average_samples);
        let b = &mut self.bodies[id];
        if !on {
            b.flags &= !AUTO_DISABLE;
            b.flags &= !DISABLED;
            b.adis.idle_steps = steps;
            b.adis.idle_time = time;
            set_average_samples(b, samples);
        } else {
            b.flags |= AUTO_DISABLE;
        }
    }

    // ---- forces (ode.cpp dBodyAdd*)

    pub fn body_add_force(&mut self, id: BodyId, fx: Real, fy: Real, fz: Real) {
        let b = &mut self.bodies[id];
        b.facc[0] += fx;
        b.facc[1] += fy;
        b.facc[2] += fz;
    }
    pub fn body_add_torque(&mut self, id: BodyId, fx: Real, fy: Real, fz: Real) {
        let b = &mut self.bodies[id];
        b.tacc[0] += fx;
        b.tacc[1] += fy;
        b.tacc[2] += fz;
    }
    pub fn body_add_rel_force(&mut self, id: BodyId, fx: Real, fy: Real, fz: Real) {
        let b = &mut self.bodies[id];
        let t2 = mul0_331(&b.r, &v3(fx, fy, fz));
        b.facc[0] += t2[0];
        b.facc[1] += t2[1];
        b.facc[2] += t2[2];
    }
    pub fn body_add_rel_torque(&mut self, id: BodyId, fx: Real, fy: Real, fz: Real) {
        let b = &mut self.bodies[id];
        let t2 = mul0_331(&b.r, &v3(fx, fy, fz));
        b.tacc[0] += t2[0];
        b.tacc[1] += t2[1];
        b.tacc[2] += t2[2];
    }
    /// dBodyAddForceAtPos (world force, world point)
    pub fn body_add_force_at_pos(&mut self, id: BodyId, fx: Real, fy: Real, fz: Real, px: Real, py: Real, pz: Real) {
        let b = &mut self.bodies[id];
        b.facc[0] += fx;
        b.facc[1] += fy;
        b.facc[2] += fz;
        let f = v3(fx, fy, fz);
        let q = v3(px - b.pos[0], py - b.pos[1], pz - b.pos[2]);
        add_cross(&mut b.tacc, &q, &f);
    }
    /// dBodyAddForceAtRelPos (world force, body point)
    pub fn body_add_force_at_rel_pos(&mut self, id: BodyId, fx: Real, fy: Real, fz: Real, px: Real, py: Real, pz: Real) {
        let b = &mut self.bodies[id];
        let f = v3(fx, fy, fz);
        let p = mul0_331(&b.r, &v3(px, py, pz));
        b.facc[0] += f[0];
        b.facc[1] += f[1];
        b.facc[2] += f[2];
        add_cross(&mut b.tacc, &p, &f);
    }
    /// dBodyAddRelForceAtPos (body force, world point)
    pub fn body_add_rel_force_at_pos(&mut self, id: BodyId, fx: Real, fy: Real, fz: Real, px: Real, py: Real, pz: Real) {
        let b = &mut self.bodies[id];
        let f = mul0_331(&b.r, &v3(fx, fy, fz));
        b.facc[0] += f[0];
        b.facc[1] += f[1];
        b.facc[2] += f[2];
        let q = v3(px - b.pos[0], py - b.pos[1], pz - b.pos[2]);
        add_cross(&mut b.tacc, &q, &f);
    }
    /// dBodyAddRelForceAtRelPos (body force, body point)
    pub fn body_add_rel_force_at_rel_pos(&mut self, id: BodyId, fx: Real, fy: Real, fz: Real, px: Real, py: Real, pz: Real) {
        let b = &mut self.bodies[id];
        let f = mul0_331(&b.r, &v3(fx, fy, fz));
        let p = mul0_331(&b.r, &v3(px, py, pz));
        b.facc[0] += f[0];
        b.facc[1] += f[1];
        b.facc[2] += f[2];
        add_cross(&mut b.tacc, &p, &f);
    }

    // ---- joints

    pub(crate) fn add_joint(&mut self, kind: JointKind, group: Option<GroupId>) -> JointId {
        let j = Joint::new(kind, group.is_some());
        let id = if let Some(id) = self.free_joints.pop() {
            self.joints[id] = j;
            id
        } else {
            self.joints.push(j);
            self.joints.len() - 1
        };
        self.joint_order.push(id);
        if let Some(g) = group {
            self.groups[g].push(id);
        }
        id
    }
    /// dJointGroupCreate
    pub fn joint_group_create(&mut self) -> GroupId {
        self.groups.push(Vec::new());
        self.groups.len() - 1
    }
    /// dJointGroupEmpty: destroys the group's joints, newest first
    pub fn joint_group_empty(&mut self, g: GroupId) {
        let list = std::mem::take(&mut self.groups[g]);
        for &j in list.iter().rev() {
            self.joint_destroy_inner(j);
        }
    }
    /// dJointDestroy
    pub fn joint_destroy(&mut self, j: JointId) {
        if self.joints[j].in_group {
            return;
        }
        self.joint_destroy_inner(j);
    }
    fn joint_destroy_inner(&mut self, j: JointId) {
        if self.joints[j].node[0].is_some() || self.joints[j].node[1].is_some() {
            self.remove_joint_references(j);
        }
        self.joint_order.retain(|&x| x != j);
        self.joints[j].alive = false;
        self.free_joints.push(j);
    }
    /// removeJointReferencesFromAttachedBodies
    fn remove_joint_references(&mut self, j: JointId) {
        for i in 0..2 {
            if let Some(b) = self.joints[j].node[i] {
                let list = &mut self.bodies[b].joints;
                // ODE removes the first match walking from the head (our tail)
                if let Some(k) = list.iter().rposition(|&(jj, _)| jj == j) {
                    list.remove(k);
                }
            }
        }
        self.joints[j].node = [None, None];
    }
    /// dJointAttach
    pub fn joint_attach(&mut self, j: JointId, body1: Option<BodyId>, body2: Option<BodyId>) {
        if self.joints[j].node[0].is_some() || self.joints[j].node[1].is_some() {
            self.remove_joint_references(j);
        }
        let (b1, b2) = match (body1, body2) {
            (None, b2) => {
                self.joints[j].reverse = true;
                (b2, None)
            }
            (b1, b2) => {
                self.joints[j].reverse = false;
                (b1, b2)
            }
        };
        self.joints[j].node = [b1, b2];
        if let Some(b) = b1 {
            self.bodies[b].joints.push((j, b2));
        }
        if let Some(b) = b2 {
            self.bodies[b].joints.push((j, b1));
        }
        if b1.is_some() || b2.is_some() {
            let (joints, bodies) = (&mut self.joints, &self.bodies);
            joints[j].set_relative_values(bodies);
        }
    }
    pub fn joint_enable(&mut self, j: JointId) {
        self.joints[j].disabled = false;
    }
    pub fn joint_disable(&mut self, j: JointId) {
        self.joints[j].disabled = true;
    }
    /// dJointGetBody
    pub fn joint_get_body(&self, j: JointId, index: usize) -> Option<BodyId> {
        let jt = &self.joints[j];
        if jt.reverse {
            jt.node[1 - index]
        } else {
            jt.node[index]
        }
    }
    /// dAreConnected
    pub fn are_connected(&self, b1: BodyId, b2: BodyId) -> bool {
        self.bodies[b1].joints.iter().any(|&(_, o)| o == Some(b2))
    }
    /// dAreConnectedExcluding: any joint other than `kind` joints b1 and b2
    pub fn are_connected_excluding(&self, b1: BodyId, b2: BodyId, kind: super::joints::JointType) -> bool {
        self.bodies[b1].joints.iter().rev().any(|&(j, o)| self.joints[j].kind.type_of() != kind && o == Some(b2))
    }

    // ---- stepping

    /// dWorldQuickStep
    pub fn quick_step(&mut self, stepsize: Real) {
        self.handle_auto_disabling(stepsize);
        let islands = self.build_islands();
        for (bodies, joints) in islands {
            super::quickstep::quick_step_island(self, &bodies, &joints, stepsize);
        }
    }

    fn joint_is_enabled(&self, j: JointId) -> bool {
        let jt = &self.joints[j];
        !jt.disabled
            && (jt.node[0].map_or(false, |b| self.bodies[b].inv_mass > 0.0) || jt.node[1].map_or(false, |b| self.bodies[b].inv_mass > 0.0))
    }

    /// BuildIslandsAndEstimateStepperMemoryRequirements (minus the memory)
    fn build_islands(&mut self) -> Vec<(Vec<BodyId>, Vec<JointId>)> {
        for &b in &self.body_order {
            self.bodies[b].tag = 0;
        }
        for &j in &self.joint_order {
            self.joints[j].tag = 0;
        }
        let mut islands = Vec::new();
        let order: Vec<BodyId> = self.body_order.iter().rev().copied().collect();
        let mut stack: Vec<BodyId> = Vec::new();
        for bb in order {
            if self.bodies[bb].tag != 0 {
                continue;
            }
            if self.bodies[bb].flags & DISABLED != 0 {
                self.bodies[bb].tag = -1;
                continue;
            }
            self.bodies[bb].tag = 1;
            let mut ib = vec![bb];
            let mut ij = Vec::new();
            let mut b = bb;
            loop {
                let n = self.bodies[b].joints.len();
                for k in (0..n).rev() {
                    let (j, other) = self.bodies[b].joints[k];
                    if self.joints[j].tag != 0 {
                        continue;
                    }
                    if self.joint_is_enabled(j) {
                        self.joints[j].tag = 1;
                        ij.push(j);
                        if let Some(nb) = other {
                            if self.bodies[nb].tag <= 0 {
                                self.bodies[nb].tag = 1;
                                self.bodies[nb].flags &= !DISABLED;
                                stack.push(nb);
                            }
                        }
                    } else {
                        self.joints[j].tag = -1;
                    }
                }
                match stack.pop() {
                    Some(nb) => {
                        b = nb;
                        ib.push(b);
                    }
                    None => break,
                }
            }
            islands.push((ib, ij));
        }
        islands
    }

    /// dInternalHandleAutoDisabling
    fn handle_auto_disabling(&mut self, stepsize: Real) {
        for idx in (0..self.body_order.len()).rev() {
            let bb = &mut self.bodies[self.body_order[idx]];
            if bb.joints.is_empty() {
                continue;
            }
            if bb.flags & (AUTO_DISABLE | DISABLED) != AUTO_DISABLE {
                continue;
            }
            if bb.adis.average_samples == 0 {
                continue;
            }
            if bb.average_counter >= bb.adis.average_samples {
                bb.average_ready = false;
                bb.average_counter = 0;
            }
            let c = bb.average_counter as usize;
            bb.average_lvel[c] = [bb.lvel[0], bb.lvel[1], bb.lvel[2], 0.0];
            bb.average_avel[c] = [bb.avel[0], bb.avel[1], bb.avel[2], 0.0];
            bb.average_counter += 1;
            if bb.average_counter >= bb.adis.average_samples {
                bb.average_counter = 0;
                bb.average_ready = true;
            }
            let mut idle = false;
            if bb.average_ready {
                idle = true;
                let mut al = bb.average_lvel[0];
                let mut aa = bb.average_avel[0];
                if bb.adis.average_samples > 1 {
                    for i in 1..bb.adis.average_samples as usize {
                        for k in 0..3 {
                            al[k] += bb.average_lvel[i][k];
                            aa[k] += bb.average_avel[i][k];
                        }
                    }
                    let r1 = 1.0 / bb.adis.average_samples as Real;
                    for k in 0..3 {
                        al[k] *= r1;
                        aa[k] *= r1;
                    }
                }
                if dot3(&al, &al) > bb.adis.linear_average_threshold {
                    idle = false;
                } else if dot3(&aa, &aa) > bb.adis.angular_average_threshold {
                    idle = false;
                }
            }
            if idle {
                bb.adis_stepsleft -= 1;
                bb.adis_timeleft -= stepsize;
            } else {
                bb.adis_stepsleft = bb.adis.idle_steps;
                bb.adis_timeleft = bb.adis.idle_time;
            }
            if bb.adis_stepsleft <= 0 && bb.adis_timeleft <= 0.0 {
                bb.flags |= DISABLED;
                bb.lvel = [0.0; 4];
                bb.avel = [0.0; 4];
            }
        }
    }
}

fn set_average_samples(b: &mut Body, n: u32) {
    b.adis.average_samples = n;
    b.average_lvel = vec![[0.0; 4]; n as usize];
    b.average_avel = vec![[0.0; 4]; n as usize];
    b.average_counter = 0;
    b.average_ready = false;
}

impl Body {
    /// dBodyGetRelPointPos
    pub fn rel_point_pos(&self, px: Real, py: Real, pz: Real) -> V3 {
        let p = mul0_331(&self.r, &v3(px, py, pz));
        v3(p[0] + self.pos[0], p[1] + self.pos[1], p[2] + self.pos[2])
    }
    /// dBodyGetRelPointVel
    pub fn rel_point_vel(&self, px: Real, py: Real, pz: Real) -> V3 {
        let p = mul0_331(&self.r, &v3(px, py, pz));
        let mut r = v3(self.lvel[0], self.lvel[1], self.lvel[2]);
        add_cross(&mut r, &self.avel, &p);
        r
    }
    /// dBodyGetPointVel
    pub fn point_vel(&self, px: Real, py: Real, pz: Real) -> V3 {
        let p = v3(px - self.pos[0], py - self.pos[1], pz - self.pos[2]);
        let mut r = v3(self.lvel[0], self.lvel[1], self.lvel[2]);
        add_cross(&mut r, &self.avel, &p);
        r
    }
    /// dBodyGetPosRelPoint
    pub fn pos_rel_point(&self, px: Real, py: Real, pz: Real) -> V3 {
        mul1_331(&self.r, &v3(px - self.pos[0], py - self.pos[1], pz - self.pos[2]))
    }
    /// dBodyVectorToWorld
    pub fn vector_to_world(&self, px: Real, py: Real, pz: Real) -> V3 {
        mul0_331(&self.r, &v3(px, py, pz))
    }
    /// dBodyVectorFromWorld
    pub fn vector_from_world(&self, px: Real, py: Real, pz: Real) -> V3 {
        mul1_331(&self.r, &v3(px, py, pz))
    }
    pub fn is_enabled(&self) -> bool {
        self.flags & DISABLED == 0
    }
}

fn sinc(x: Real) -> Real {
    if x.abs() < 1.0e-4 {
        1.0 - x * x * 0.166666666666666666667
    } else {
        x.sin() / x
    }
}

/// dxStepBody: integrate one body's position and orientation over h
pub(crate) fn step_body(b: &mut Body, mut h: Real) {
    if b.flags & MAX_ANGULAR_SPEED != 0 {
        let max = b.max_angular_speed;
        let aspeed = dot3(&b.avel, &b.avel);
        if aspeed > max * max {
            let coef = max / aspeed.sqrt();
            b.avel[0] *= coef;
            b.avel[1] *= coef;
            b.avel[2] *= coef;
        }
    }
    for j in 0..3 {
        b.pos[j] += h * b.lvel[j];
    }
    if b.flags & FINITE_ROTATION != 0 {
        let mut irv = [0.0; 4];
        let mut q = [0.0; 4];
        if b.flags & FINITE_ROTATION_AXIS != 0 {
            let k = dot3(&b.finite_rot_axis, &b.avel);
            let frv = [b.finite_rot_axis[0] * k, b.finite_rot_axis[1] * k, b.finite_rot_axis[2] * k];
            irv[0] = b.avel[0] - frv[0];
            irv[1] = b.avel[1] - frv[1];
            irv[2] = b.avel[2] - frv[2];
            h *= 0.5;
            let theta = k * h;
            q[0] = theta.cos();
            let s = sinc(theta) * h;
            q[1] = frv[0] * s;
            q[2] = frv[1] * s;
            q[3] = frv[2] * s;
        } else {
            let wlen = (b.avel[0] * b.avel[0] + b.avel[1] * b.avel[1] + b.avel[2] * b.avel[2]).sqrt();
            h *= 0.5;
            let theta = wlen * h;
            q[0] = theta.cos();
            let s = sinc(theta) * h;
            q[1] = b.avel[0] * s;
            q[2] = b.avel[1] * s;
            q[3] = b.avel[2] * s;
        }
        b.q = q_mul0(&q, &b.q);
        if b.flags & FINITE_ROTATION_AXIS != 0 {
            let dq = dq_from_w(&irv, &b.q);
            for j in 0..4 {
                b.q[j] += h * dq[j];
            }
        }
    } else {
        let dq = dq_from_w(&b.avel, &b.q);
        for j in 0..4 {
            b.q[j] += h * dq[j];
        }
    }
    normalize4(&mut b.q);
    b.r = r_from_q(&b.q);
    if b.flags & LINEAR_DAMPING != 0 {
        let lin_speed = dot3(&b.lvel, &b.lvel);
        if lin_speed > b.damping.linear_threshold {
            let k = 1.0 - b.damping.linear_scale;
            b.lvel[0] *= k;
            b.lvel[1] *= k;
            b.lvel[2] *= k;
        }
    }
    if b.flags & ANGULAR_DAMPING != 0 {
        let ang_speed = dot3(&b.avel, &b.avel);
        if ang_speed > b.damping.angular_threshold {
            let k = 1.0 - b.damping.angular_scale;
            b.avel[0] *= k;
            b.avel[1] *= k;
            b.avel[2] *= k;
        }
    }
}

/// dInvertPDMatrix for n = 3 (row stride 4): Cholesky, then one solve per
/// column. None when A is not positive definite.
fn invert_pd3(a: &M3) -> Option<M3> {
    const N: usize = 3;
    const S: usize = 4;
    let mut l = *a;
    let mut rc = [0.0; N];
    // dxFactorCholesky
    for i in 0..N {
        let aa = i * S;
        for j in 0..i {
            let bb = j * S;
            let mut sum = l[aa + j];
            for k in 0..j {
                sum -= l[aa + k] * l[bb + k];
            }
            l[aa + j] = sum * rc[j];
        }
        let mut sum = l[aa + i];
        for k in 0..i {
            sum -= l[aa + k] * l[aa + k];
        }
        if sum <= 0.0 {
            return None;
        }
        let s = sum.sqrt();
        l[aa + i] = s;
        rc[i] = recip(s);
    }
    let mut inv = [0.0; 12];
    for c in 0..N {
        let mut x = [0.0; N];
        x[c] = 1.0;
        // dxSolveCholesky
        let mut y = [0.0; N];
        for i in 0..N {
            let mut sum = 0.0;
            for k in 0..i {
                sum += l[i * S + k] * y[k];
            }
            y[i] = (x[i] - sum) / l[i * S + i];
        }
        for i in (0..N).rev() {
            let mut sum = 0.0;
            for k in i + 1..N {
                sum += l[k * S + i] * x[k];
            }
            x[i] = (y[i] - sum) / l[i * S + i];
        }
        for r in 0..N {
            inv[r * S + c] = x[r];
        }
    }
    Some(inv)
}
