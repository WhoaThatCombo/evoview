//! The rig car's bodies in ODE (EVOVIEW_RIG_ODE=1): the chassis and the
//! four hubs as ODE bodies, each hub on a slider joint along the chassis'
//! up axis, stepped by QuickStep with the world settings AC EVO gives its
//! own (the code after dWorldCreate, FUN_14252b5e0's caller in the
//! 20260907 client):
//!
//! - gravity (0, -9.806, 0); ERP 0.6 and CFM 1e-7 (floats, widened);
//! - QuickStep 30 iterations, SOR w 1.4;
//! - contact max correcting velocity 3, surface layer 0; no damping.
//!
//! rigcar.rs keeps the car's state and computes every force as before
//! (coilovers, ARBs, tyres, aero); each substep writes that state into the
//! bodies, adds the forces, steps the world and reads the state back. What
//! changes against the explicit integrator: the hubs carry their full
//! weight and tyre force, and the slider constraint - not a roll-centre
//! shortcut - hands the off-axis part to the chassis; ODE integrates the
//! bodies (gyroscopic term included). The game's own wishbone rods are
//! rigwish.rs (EVOVIEW_RIG_ODE=links).

use crate::ode::{self, BodyId, Mass, World};
use glam::{Mat3, Quat, Vec3};

pub struct OdeRig {
    w: World,
    body: BodyId,
    hub: [BodyId; 4],
}

/// float values the game stores as doubles (0x3fe3333340000000 etc.); the
/// suspension joints get the same pair (FUN_142435fc0)
pub const ERP: f64 = 0.6f32 as f64;
pub const CFM: f64 = 1e-7f32 as f64;

/// A world with the game's settings.
pub fn evo_world() -> World {
    let mut w = World::new();
    w.set_gravity(0.0, -9.806, 0.0);
    w.set_erp(ERP);
    w.set_cfm(CFM);
    w.set_quickstep_iterations(30);
    w.set_quickstep_w(1.4);
    w.set_contact_max_correcting_vel(3.0);
    w.set_contact_surface_layer(0.0);
    w
}

fn d3(v: Vec3) -> [f64; 3] {
    [v.x as f64, v.y as f64, v.z as f64]
}

/// The four corners' state the rig reads back: compression and its rate.
#[derive(Clone, Copy, Default)]
pub struct HubState {
    pub x: f32,
    pub xv: f32,
}

/// What one substep integrates: the chassis' state and forces, each hub's
/// travel and forces, and the geometry to place the hubs.
pub struct Frame<'a> {
    pub cg: Vec3,
    pub q: Quat,
    pub v: Vec3,
    pub w: Vec3,
    pub cg_local: Vec3,
    pub anchor: &'a [Vec3; 4],
    pub x_static: &'a [f32; 4],
    pub x: [f32; 4],
    pub xv: [f32; 4],
    pub f_body: Vec3,
    pub t_body: Vec3,
    pub f_hub: [Vec3; 4],
    pub t_hub: [Vec3; 4],
}

impl OdeRig {
    pub fn enabled() -> bool {
        std::env::var("EVOVIEW_RIG_ODE").map_or(false, |v| v != "0" && v != "links")
    }

    /// `inertia`: the chassis' principal moments (body axes); hubs of
    /// `m_unsprung` kg at the mesh-space `anchor`s.
    pub fn new(mass: f32, inertia: Vec3, m_unsprung: [f32; 2], anchor: &[Vec3; 4], cg_local: Vec3) -> OdeRig {
        let mut w = evo_world();
        let body = w.body_create();
        let mut m = Mass { mass: mass as f64, ..Default::default() };
        m.i[0] = inertia.x as f64;
        m.i[5] = inertia.y as f64;
        m.i[10] = inertia.z as f64;
        w.body_set_mass(body, &m);
        let c = d3(cg_local);
        w.body_set_position(body, c[0], c[1], c[2]);
        let hub = [0usize, 1, 2, 3].map(|k| {
            let h = w.body_create();
            w.body_set_mass(h, &Mass::sphere_total(m_unsprung[k / 2] as f64, 0.3));
            let p = d3(anchor[k]);
            w.body_set_position(h, p[0], p[1], p[2]);
            let j = w.joint_create_slider(None);
            w.joint_attach(j, Some(body), Some(h));
            w.joint_set_slider_axis(j, 0.0, 1.0, 0.0);
            h
        });
        OdeRig { w, body, hub }
    }

    fn put(&mut self, b: BodyId, p: Vec3, q: Quat, v: Vec3, wv: Vec3) {
        let (p, v, wv) = (d3(p), d3(v), d3(wv));
        self.w.body_set_position(b, p[0], p[1], p[2]);
        self.w.body_set_quaternion(b, &[q.w as f64, q.x as f64, q.y as f64, q.z as f64]);
        self.w.body_set_linear_vel(b, v[0], v[1], v[2]);
        self.w.body_set_angular_vel(b, wv[0], wv[1], wv[2]);
    }

    /// One substep: the chassis' new (cg, q, v, w) and each hub's travel.
    pub fn step(&mut self, f: &Frame, dt: f32) -> (Vec3, Quat, Vec3, Vec3, [HubState; 4]) {
        let r = Mat3::from_quat(f.q);
        let up = r * Vec3::Y;
        let o = f.cg - r * f.cg_local;
        self.put(self.body, f.cg, f.q, f.v, f.w);
        for k in 0..4 {
            let p = o + r * f.anchor[k] + up * (f.x[k] - f.x_static[k]);
            let v = f.v + f.w.cross(p - f.cg) + up * f.xv[k];
            self.put(self.hub[k], p, f.q, v, f.w);
        }
        let add = |w: &mut World, b: BodyId, fv: Vec3, tv: Vec3| {
            let (fv, tv) = (d3(fv), d3(tv));
            w.body_add_force(b, fv[0], fv[1], fv[2]);
            w.body_add_torque(b, tv[0], tv[1], tv[2]);
        };
        add(&mut self.w, self.body, f.f_body, f.t_body);
        for k in 0..4 {
            add(&mut self.w, self.hub[k], f.f_hub[k], f.t_hub[k]);
        }
        self.w.quick_step(dt as f64);
        let get = |b: &ode::world::Body| {
            let v3 = |a: &[f64]| Vec3::new(a[0] as f32, a[1] as f32, a[2] as f32);
            (v3(&b.pos), Quat::from_xyzw(b.q[1] as f32, b.q[2] as f32, b.q[3] as f32, b.q[0] as f32).normalize(), v3(&b.lvel), v3(&b.avel))
        };
        let (cg, q, v, wv) = get(&self.w.bodies[self.body]);
        let r = Mat3::from_quat(q);
        let up = r * Vec3::Y;
        let o = cg - r * f.cg_local;
        let mut hubs = [HubState::default(); 4];
        for k in 0..4 {
            let (p, _, hv, _) = get(&self.w.bodies[self.hub[k]]);
            let a = o + r * f.anchor[k];
            let av = v + wv.cross(a - cg);
            hubs[k] = HubState { x: f.x_static[k] + (p - a).dot(up), xv: (hv - av).dot(up) };
        }
        (cg, q, v, wv, hubs)
    }
}
