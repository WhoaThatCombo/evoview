//! The Rust side of tools/ode_oracle: the same scenario as oracle.cpp on
//! the port, printed the same way. `ODE_ORACLE=<path to ode_oracle.exe>
//! cargo test ode::oracle` runs both and fails on the first differing bit.

use super::joints::*;
use super::math::*;
use super::world::*;
use std::fmt::Write;

fn hex(out: &mut String, v: f64) {
    let _ = write!(out, " {:016x}", v.to_bits());
}
fn dump_body(out: &mut String, b: &Body) {
    for i in 0..3 {
        hex(out, b.pos[i]);
    }
    for i in 0..4 {
        hex(out, b.q[i]);
    }
    for i in 0..3 {
        hex(out, b.lvel[i]);
    }
    for i in 0..3 {
        hex(out, b.avel[i]);
    }
    out.push('\n');
}

pub fn scenario(steps: usize) -> String {
    let mut out = String::new();
    rand_set_seed(12345);
    let mut w = World::new();
    w.set_gravity(0.0, -9.806, 0.0);
    w.set_erp(0.25);
    w.set_cfm(1e-7);
    w.set_quickstep_iterations(24);
    w.set_contact_max_correcting_vel(5.0);
    w.set_contact_surface_layer(0.001);
    let contacts = w.joint_group_create();
    let mut b = [0usize; 11];

    b[0] = w.body_create();
    w.body_set_mass(b[0], &Mass::box_total(3.0, 0.4, 1.2, 0.2));
    w.body_set_position(b[0], 0.0, 5.0, 0.0);
    w.body_set_angular_vel(b[0], 0.3, 7.0, 0.2);
    w.body_set_linear_vel(b[0], 1.0, 0.0, 0.0);

    b[1] = w.body_create();
    w.body_set_mass(b[1], &Mass::capsule_total(2.0, 3, 0.1, 0.8));
    w.body_set_position(b[1], 2.0, 3.0, 0.4);
    let r = r_from_axis_and_angle(0.2, 1.0, 0.3, 0.7);
    w.body_set_rotation(b[1], &r);
    let h = w.joint_create_hinge(None);
    w.joint_attach(h, Some(b[1]), None);
    w.joint_set_hinge_anchor(h, 2.0, 3.0, 0.0);
    w.joint_set_hinge_axis(h, 1.0, 0.0, 0.2);
    w.joint_set_hinge_param(h, PARAM_LO_STOP, -0.6);
    w.joint_set_hinge_param(h, PARAM_HI_STOP, 0.9);
    w.joint_set_hinge_param(h, PARAM_VEL, 1.5);
    w.joint_set_hinge_param(h, PARAM_FMAX, 4.0);
    w.joint_set_hinge_param(h, PARAM_BOUNCE, 0.3);
    w.joint_enable_feedback(h, true);
    b[2] = w.body_create();
    w.body_set_mass(b[2], &Mass::sphere_total(1.5, 0.15));
    w.body_set_position(b[2], 2.3, 2.2, 0.8);
    let ball = w.joint_create_ball(None);
    w.joint_attach(ball, Some(b[1]), Some(b[2]));
    w.joint_set_ball_anchor(ball, 2.1, 2.6, 0.7);
    w.joint_set_erp_cfm_param(ball, PARAM_ERP, 0.4);

    b[3] = w.body_create();
    w.body_set_mass(b[3], &Mass::box_total(5.0, 0.5, 0.5, 0.5));
    w.body_set_position(b[3], -2.0, 4.0, 0.0);
    b[4] = w.body_create();
    w.body_set_mass(b[4], &Mass::box_total(1.0, 0.2, 0.2, 0.6));
    w.body_set_position(b[4], -2.0, 3.2, 0.3);
    let r = r_from_euler_angles(0.1, -0.2, 0.3);
    w.body_set_rotation(b[4], &r);
    let sl = w.joint_create_slider(None);
    w.joint_attach(sl, Some(b[3]), Some(b[4]));
    w.joint_set_slider_axis(sl, 0.1, 1.0, 0.05);
    w.joint_set_slider_param(sl, PARAM_LO_STOP, -0.5);
    w.joint_set_slider_param(sl, PARAM_HI_STOP, 0.1);
    w.joint_set_slider_param(sl, PARAM_VEL, -0.4);
    w.joint_set_slider_param(sl, PARAM_FMAX, 30.0);
    w.joint_enable_feedback(sl, true);
    let db = w.joint_create_dball(None);
    w.joint_attach(db, Some(b[3]), None);
    w.joint_set_dball_anchor1(db, -2.0, 4.25, 0.0);
    w.joint_set_dball_anchor2(db, -2.2, 5.5, 0.1);

    b[5] = w.body_create();
    w.body_set_mass(b[5], &Mass::cylinder_total(2.0, 2, 0.2, 0.5));
    w.body_set_position(b[5], 4.0, 6.0, -1.0);
    w.body_set_angular_vel(b[5], 0.0, 2.0, 1.0);
    b[6] = w.body_create();
    let mut m6 = Mass::box_total(1.0, 0.3, 0.3, 0.3);
    m6.translate(0.0, 0.0, 0.0);
    w.body_set_mass(b[6], &m6);
    w.body_set_position(b[6], 4.2, 5.5, -1.1);
    let fx = w.joint_create_fixed(None);
    w.joint_attach(fx, Some(b[5]), Some(b[6]));
    w.joint_set_fixed(fx);
    w.joint_set_erp_cfm_param(fx, PARAM_CFM, 1e-5);
    let am = w.joint_create_amotor(None);
    w.joint_attach(am, Some(b[5]), Some(b[6]));
    w.joint_set_amotor_num_axes(am, 3);
    w.joint_set_amotor_axis(am, 0, 1, 1.0, 0.0, 0.0);
    w.joint_set_amotor_axis(am, 1, 0, 0.0, 1.0, 0.0);
    w.joint_set_amotor_axis(am, 2, 2, 0.0, 0.0, 1.0);
    w.joint_set_amotor_param(am, PARAM_VEL, 0.5);
    w.joint_set_amotor_param(am, PARAM_FMAX, 2.0);
    w.joint_set_amotor_param(am, 2 * PARAM_GROUP + PARAM_VEL, -0.3);
    w.joint_set_amotor_param(am, 2 * PARAM_GROUP + PARAM_FMAX, 1.0);
    let ae = w.joint_create_amotor(None);
    w.joint_attach(ae, Some(b[6]), None);
    w.joint_set_amotor_mode(ae, AMOTOR_EULER);
    w.joint_set_amotor_axis(ae, 0, 1, 1.0, 0.0, 0.0);
    w.joint_set_amotor_axis(ae, 2, 0, 0.0, 0.0, 1.0);
    w.joint_set_amotor_param(ae, PARAM_LO_STOP, -0.3);
    w.joint_set_amotor_param(ae, PARAM_HI_STOP, 0.3);
    w.joint_set_amotor_param(ae, PARAM_GROUP + PARAM_LO_STOP, -0.2);
    w.joint_set_amotor_param(ae, PARAM_GROUP + PARAM_HI_STOP, 0.2);
    w.joint_set_amotor_param(ae, 2 * PARAM_GROUP + PARAM_FMAX, 0.5);

    b[7] = w.body_create();
    w.body_set_mass(b[7], &Mass::box_total(10.0, 1.0, 0.5, 1.5));
    w.body_set_position(b[7], 0.0, 0.6, 3.0);
    let r = r_from_axis_and_angle(0.3, 0.1, 1.0, 0.4);
    w.body_set_rotation(b[7], &r);
    w.body_set_linear_vel(b[7], 2.0, 0.0, -1.0);
    w.body_set_angular_vel(b[7], 0.0, 1.0, 0.0);
    b[8] = w.body_create();
    w.body_set_mass(b[8], &Mass::box_total(4.0, 0.6, 0.6, 0.6));
    w.body_set_position(b[8], 3.0, 0.5, 3.0);
    w.body_set_linear_vel(b[8], -3.0, 0.0, 0.5);
    w.body_set_finite_rotation_mode(b[8], true);
    w.body_set_linear_damping(b[8], 0.002);
    w.body_set_angular_damping(b[8], 0.01);

    b[9] = w.body_create();
    w.body_set_mass(b[9], &Mass::sphere_total(1.0, 0.3));
    w.body_set_position(b[9], -5.0, 0.3, -3.0);
    w.body_set_linear_vel(b[9], 2.0, 0.0, 0.1);
    w.body_set_gravity_mode(b[9], false);
    b[10] = w.body_create();
    w.body_set_mass(b[10], &Mass::sphere_total(2.0, 0.4));
    w.body_set_position(b[10], -3.0, 0.35, -3.0);
    w.body_set_gravity_mode(b[10], false);
    w.body_set_angular_vel(b[10], 0.0, 0.0, 3.0);

    let half = [[0.5, 0.25, 0.75], [0.3, 0.3, 0.3]];
    let dt = 0.003;
    for s in 0..steps {
        w.body_add_force(b[3], 0.0, 0.0, 5.0 * (s as f64 * 0.01).sin());
        w.body_add_rel_torque(b[5], 0.1, 0.0, 0.0);
        w.body_add_force_at_rel_pos(b[7], 0.0, 0.0, 3.0, 0.3, 0.2, 0.0);
        w.joint_add_hinge_torque(h, 0.5);
        for k in 0..2 {
            let bb = b[7 + k];
            for c in 0..8 {
                let sx = if c & 1 != 0 { 1.0 } else { -1.0 };
                let sy = if c & 2 != 0 { 1.0 } else { -1.0 };
                let sz = if c & 4 != 0 { 1.0 } else { -1.0 };
                let p = w.body(bb).rel_point_pos(sx * half[k][0], sy * half[k][1], sz * half[k][2]);
                if p[1] < 0.0 {
                    let mut ct = Contact::default();
                    ct.surface.mode = CONTACT_APPROX1 | CONTACT_SOFT_ERP | CONTACT_SOFT_CFM | CONTACT_BOUNCE;
                    ct.surface.mu = if k == 0 { 0.8 } else { 0.3 };
                    ct.surface.bounce = 0.2;
                    ct.surface.bounce_vel = 0.1;
                    ct.surface.soft_erp = 0.3;
                    ct.surface.soft_cfm = 1e-5;
                    ct.geom.pos = [p[0], p[1], p[2], 0.0];
                    ct.geom.normal[1] = 1.0;
                    ct.geom.depth = -p[1];
                    let cj = w.joint_create_contact(Some(contacts), &ct);
                    w.joint_attach(cj, Some(bb), None);
                }
            }
        }
        {
            let (p1, p2) = (w.body(b[9]).pos, w.body(b[10]).pos);
            let d = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if len < 0.7 {
                let mut ct = Contact::default();
                ct.surface.mode = CONTACT_MU2 | CONTACT_FDIR1 | CONTACT_SLIP1;
                ct.surface.mu = 0.5;
                ct.surface.mu2 = 0.2;
                ct.surface.slip1 = 0.01;
                ct.geom.normal = [-d[0] / len, -d[1] / len, -d[2] / len, 0.0];
                ct.geom.pos = [p1[0] + d[0] / len * 0.3, p1[1] + d[1] / len * 0.3, p1[2] + d[2] / len * 0.3, 0.0];
                ct.geom.depth = 0.7 - len;
                ct.fdir1 = [0.0, 0.0, 1.0, 0.0];
                let cj = w.joint_create_contact(Some(contacts), &ct);
                w.joint_attach(cj, Some(b[9]), Some(b[10]));
            }
        }
        w.quick_step(dt);
        w.joint_group_empty(contacts);
        let _ = write!(out, "S{}", s);
        hex(&mut out, w.joint_get_hinge_angle(h));
        hex(&mut out, w.joint_get_slider_position(sl));
        let fh = w.joint_feedback(h).unwrap();
        let fs = w.joint_feedback(sl).unwrap();
        for i in 0..3 {
            hex(&mut out, fh.f1[i]);
        }
        for i in 0..3 {
            hex(&mut out, fs.t2[i]);
        }
        hex(&mut out, w.joint_get_amotor_angle(ae, 1));
        out.push('\n');
        for &bb in &b {
            dump_body(&mut out, w.body(bb));
        }
    }
    out
}

fn dump_contacts(out: &mut String, cs: &[super::joints::ContactGeom]) {
    for c in cs {
        out.push('C');
        for k in 0..3 {
            hex(out, c.pos[k]);
        }
        for k in 0..3 {
            hex(out, c.normal[k]);
        }
        hex(out, c.depth);
        out.push(char::from(10));
    }
}

fn near(w: &mut World, group: GroupId, o1: usize, o2: usize, out: &mut String, ray_hits: &mut usize, maxc: i32) {
    use super::collision::RAY;
    if w.geom_is_space(o1) || w.geom_is_space(o2) {
        for (a, b) in w.space_collide2(o1, o2) {
            near(w, group, a, b, out, ray_hits, maxc);
        }
        return;
    }
    let cs = w.collide(o1, o2, maxc);
    dump_contacts(out, &cs);
    if w.geom_class(o1) == RAY || w.geom_class(o2) == RAY {
        *ray_hits += cs.len();
        return;
    }
    for c in &cs {
        let mut ct = Contact::default();
        ct.surface.mode = CONTACT_APPROX1 | CONTACT_BOUNCE | CONTACT_SOFT_CFM;
        ct.surface.mu = 0.6;
        ct.surface.bounce = 0.1;
        ct.surface.bounce_vel = 0.2;
        ct.surface.soft_cfm = 1e-6;
        ct.geom = *c;
        let j = w.joint_create_contact(Some(group), &ct);
        let (b1, b2) = (w.geoms[o1].body, w.geoms[o2].body);
        w.joint_attach(j, b1, b2);
    }
}

pub fn collide_scenario(steps: usize) -> String {
    let mut out = String::new();
    rand_set_seed(777);
    let mut w = World::new();
    w.set_gravity(0.0, -9.806, 0.0);
    w.set_quickstep_iterations(20);
    w.set_contact_surface_layer(0.002);
    let group = w.joint_group_create();
    let space = w.hash_space_create(None);
    w.hash_space_set_levels(space, -4, 6);
    let inner = w.simple_space_create(Some(space));
    w.create_plane(Some(space), 0.0, 1.0, 0.0, 0.0);
    let mut bs = Vec::new();
    let mut r = IDENTITY_R;
    for i in 0..6 {
        let fi = i as f64;
        let b = w.body_create();
        w.body_set_mass(b, &Mass::box_total(2.0 + fi, 0.4 + 0.05 * fi, 0.3, 0.5));
        w.body_set_position(b, 0.3 * fi - 0.8, 0.5 + 0.45 * fi, 0.1 * (i % 3) as f64);
        r = r_from_axis_and_angle(0.3 * fi, 1.0, 0.2, 0.25 * fi);
        w.body_set_rotation(b, &r);
        w.body_set_angular_vel(b, 0.2 * fi, 0.0, -0.1 * fi);
        let g = w.create_box(Some(space), 0.4 + 0.05 * fi, 0.3, 0.5);
        w.geom_set_body(g, Some(b));
        bs.push(b);
    }
    for i in 0..4 {
        let fi = i as f64;
        let b = w.body_create();
        w.body_set_mass(b, &Mass::sphere_total(1.0, 0.15 + 0.03 * fi));
        w.body_set_position(b, -0.5 + 0.35 * fi, 3.2 + 0.3 * fi, -0.2 + 0.15 * fi);
        w.body_set_linear_vel(b, 0.3, -1.0, 0.1 * fi);
        let g = w.create_sphere(Some(inner), 0.15 + 0.03 * fi);
        w.geom_set_body(g, Some(b));
        bs.push(b);
    }
    for i in 0..3 {
        let fi = i as f64;
        let b = w.body_create();
        w.body_set_mass(b, &Mass::capsule_total(1.5, 3, 0.1, 0.5));
        w.body_set_position(b, 0.6 - 0.4 * fi, 4.5 + 0.4 * fi, 0.3);
        r = r_from_axis_and_angle(1.0, 0.1 * fi, 0.0, 1.2 + 0.3 * fi);
        w.body_set_rotation(b, &r);
        let g = w.create_capsule(Some(space), 0.1, 0.5);
        w.geom_set_body(g, Some(b));
        if i == 1 {
            w.geom_set_offset_position(g, 0.05, 0.0, 0.02);
            r = r_from_axis_and_angle(0.0, 1.0, 0.0, 0.3);
            w.geom_set_offset_rotation(g, &r);
        }
        bs.push(b);
    }
    let ray = w.create_ray(Some(space), 2.0);
    w.geom_set_body(ray, Some(bs[0]));
    w.geom_set_offset_rotation(ray, &r);
    for s in 0..steps {
        let mut ray_hits = 0;
        let _ = writeln!(out, "S{}", s);
        for (a, b) in w.space_collide(space) {
            near(&mut w, group, a, b, &mut out, &mut ray_hits, (s % 3 + 2) as i32);
        }
        for (a, b) in w.space_collide(inner) {
            near(&mut w, group, a, b, &mut out, &mut ray_hits, (s % 3 + 2) as i32);
        }
        w.quick_step(0.003);
        w.joint_group_empty(group);
        let _ = writeln!(out, "R{}", ray_hits);
        for &b in &bs {
            dump_body(&mut out, w.body(b));
        }
    }
    out
}

fn grid() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let mut v = Vec::new();
    for j in 0..13 {
        for i in 0..13 {
            let x = (i as f64 - 6.0) * 0.5;
            let z = (j as f64 - 6.0) * 0.5;
            v.push([x, 0.3 * (x * 1.3).sin() * (z * 0.7).cos() + 0.1 * x, z]);
        }
    }
    let mut t = Vec::new();
    for j in 0..12u32 {
        for i in 0..12u32 {
            let a = j * 13 + i;
            let (b, c) = (a + 1, a + 13);
            let d = c + 1;
            if (i + j) & 1 != 0 {
                t.push([a, c, b]);
                t.push([b, c, d]);
            } else {
                t.push([a, c, d]);
                t.push([a, d, b]);
            }
        }
    }
    (v, t)
}

pub fn trimesh_scenario(steps: usize) -> String {
    use super::collision::TriMeshData;
    let mut out = String::new();
    let mut w = World::new();
    let (v, t) = grid();
    let data = TriMeshData::build(v, t);
    let space = w.simple_space_create(None);
    let mesh = w.create_trimesh(Some(space), data);
    let ray = w.create_ray(Some(space), 1.0);
    let sph = w.create_sphere(Some(space), 0.3);
    let bx = w.create_box(Some(space), 0.6, 0.3, 0.9);
    let cap = w.create_capsule(Some(space), 0.2, 0.6);
    let pln = w.create_plane(None, 0.0, 1.0, 0.0, 0.0);
    for s in 0..steps {
        let fs = s as f64;
        let _ = writeln!(out, "S{}", s);
        let r = r_from_axis_and_angle(0.1, 1.0, 0.2, 0.05 * fs);
        w.geom_set_rotation(mesh, &r);
        w.geom_set_position(mesh, 0.01 * fs, -0.2 + 0.003 * fs, 0.02 * fs);
        let aabb = w.geom_aabb(mesh);
        for k in aabb {
            hex(&mut out, k);
        }
        out.push(char::from(10));
        for ri in 0..24 {
            let fr = ri as f64;
            let (mut ox, oz) = (-2.7 + 0.23 * fr + 0.01 * fs, 2.5 - 0.21 * fr);
            let (mut dx, mut dz) = (0.2 * (fr * 0.7).sin(), 0.15 * (fr * 1.1).cos());
            let mut oy = if ri % 5 == 4 { -1.5 } else { 1.5 };
            let mut dy = if ri % 5 == 4 { 1.0 } else { -1.0 };
            if ri % 7 == 3 {
                dx = 1.0;
                dy = -0.05;
                dz = 0.3;
                oy = 0.1;
                ox = -3.5;
            }
            let len = if ri % 3 == 0 { 1e30 } else { 1.6 + 0.4 * (ri % 4) as f64 };
            if let super::collision::Shape::Ray { length } = &mut w.geoms[ray].shape {
                *length = len;
            }
            w.geom_moved(ray);
            let first = (ri >> 1) & 1 != 0;
            w.geom_ray_set_params(ray, first, (ri >> 2) & 1 != 0, ri % 6 == 5 && !first);
            if ri % 7 == 3 {
                if let super::collision::Shape::Ray { length } = &mut w.geoms[ray].shape {
                    *length = if ri == 17 { 4.5 } else { 1e30 };
                }
                w.geom_moved(ray);
                w.geom_ray_set_params(ray, false, false, false);
            }
            w.geom_ray_set(ray, ox, oy, oz, dx, dy, dz);
            let maxc = 1 + (ri % 4) * 3;
            for (tag, a, b) in [('A', ray, mesh), ('B', mesh, ray)] {
                let cs = w.collide(a, b, maxc);
                let _ = write!(out, "{}{} {}", tag, ri, cs.len());
                for c in &cs {
                    let _ = write!(out, " {}/{}", c.side1, c.side2);
                }
                out.push(char::from(10));
                dump_contacts(&mut out, &cs);
            }
        }
        for q in 0..20 {
            let fq = q as f64;
            let x = -2.6 + 0.27 * fq;
            let z = 2.2 - 0.23 * fq + 0.005 * fs;
            let y = 0.25 * (fq * 0.9 + fs * 0.05).sin() + 0.05 * x;
            if let super::collision::Shape::Sphere { radius } = &mut w.geoms[sph].shape {
                *radius = 0.15 + 0.05 * (q % 6) as f64;
            }
            w.geom_moved(sph);
            w.geom_set_position(sph, x, y, z);
            let maxc = 1 + (q % 5) * 3;
            let cs = w.collide(sph, mesh, maxc);
            let _ = write!(out, "P{} {}", q, cs.len());
            for c in &cs {
                let _ = write!(out, " {}/{}", c.side1, c.side2);
            }
            out.push(char::from(10));
            dump_contacts(&mut out, &cs);
            let cs = w.collide(mesh, sph, maxc);
            let _ = writeln!(out, "Q{} {}", q, cs.len());
            dump_contacts(&mut out, &cs);
        }
        for q in 0..16 {
            let fq = q as f64;
            let x = -2.4 + 0.31 * fq;
            let z = 2.0 - 0.27 * fq + 0.004 * fs;
            let y = 0.2 * (fq * 1.3 + fs * 0.04).sin() + 0.05 * x;
            if let super::collision::Shape::Box { side } = &mut w.geoms[bx].shape {
                *side = [0.3 + 0.15 * (q % 4) as f64, 0.2 + 0.1 * (q % 3) as f64, 0.4 + 0.2 * (q % 5) as f64, 0.0];
            }
            w.geom_moved(bx);
            let rb = r_from_axis_and_angle((fq * 0.5).sin(), 1.0, (fq * 0.3).cos(), 0.4 * fq + 0.03 * fs);
            w.geom_set_rotation(bx, &rb);
            w.geom_set_position(bx, x, y, z);
            let maxc = 1 + (q % 5) * 3;
            let unimp = if q % 7 == 6 { super::collision::CONTACTS_UNIMPORTANT } else { 0 };
            let cs = w.collide(bx, mesh, maxc | unimp);
            let _ = write!(out, "X{} {}", q, cs.len());
            for c in &cs {
                let _ = write!(out, " {}/{}", c.side1, c.side2);
            }
            out.push(char::from(10));
            dump_contacts(&mut out, &cs);
            let cs = w.collide(mesh, bx, maxc);
            let _ = writeln!(out, "Y{} {}", q, cs.len());
            dump_contacts(&mut out, &cs);
        }
        for q in 0..16 {
            let fq = q as f64;
            let x = -2.5 + 0.33 * fq;
            let z = -2.1 + 0.26 * fq - 0.004 * fs;
            let y = 0.22 * (fq * 1.1 + fs * 0.05).cos() + 0.05 * x;
            if let super::collision::Shape::Capsule { radius, lz } = &mut w.geoms[cap].shape {
                *radius = 0.1 + 0.05 * (q % 4) as f64;
                *lz = 0.3 + 0.25 * (q % 3) as f64;
            }
            w.geom_moved(cap);
            let rc = r_from_axis_and_angle((fq * 0.7).cos(), 0.4, (fq * 0.2).sin(), 0.5 * fq + 0.02 * fs);
            w.geom_set_rotation(cap, &rc);
            w.geom_set_position(cap, x, y, z);
            let maxc = 1 + (q % 5) * 3;
            let unimp = if q % 6 == 5 { super::collision::CONTACTS_UNIMPORTANT } else { 0 };
            let cs = w.collide(cap, mesh, maxc | unimp);
            let _ = write!(out, "K{} {}", q, cs.len());
            for c in &cs {
                let _ = write!(out, " {}/{}", c.side1, c.side2);
            }
            out.push(char::from(10));
            dump_contacts(&mut out, &cs);
            let cs = w.collide(mesh, cap, maxc);
            let _ = writeln!(out, "L{} {}", q, cs.len());
            dump_contacts(&mut out, &cs);
        }
        for q in 0..4 {
            let fq = q as f64;
            w.geom_plane_set_params(pln, 0.1 * (fq + fs * 0.1).sin(), 1.0, 0.2 * (fq * 2.0).cos(), -0.15 + 0.1 * fq);
            let maxc = 2 + q * 3;
            let cs = w.collide(mesh, pln, maxc);
            let _ = write!(out, "M{} {}", q, cs.len());
            for c in &cs {
                let _ = write!(out, " {}/{}", c.side1, c.side2);
            }
            out.push(char::from(10));
            dump_contacts(&mut out, &cs);
            let cs = w.collide(pln, mesh, maxc);
            let _ = writeln!(out, "N{} {}", q, cs.len());
            dump_contacts(&mut out, &cs);
        }
    }
    out
}

#[test]
fn trimesh_matches_ode_0_15_2() {
    let steps = oracle_steps().min(100);
    let ours = trimesh_scenario(steps);
    compare(&ours, &[&steps.to_string(), "trimesh"]);
}

fn compare(ours: &str, args: &[&str]) {
    let Ok(exe) = std::env::var("ODE_ORACLE") else {
        eprintln!("ODE_ORACLE not set: ran the scenario only");
        return;
    };
    let o = std::process::Command::new(exe).args(args).output().expect("run ode_oracle");
    let theirs = String::from_utf8(o.stdout).unwrap();
    if let Ok(p) = std::env::var("ODE_ORACLE_DUMP") {
        let _ = std::fs::write(format!("{p}.{}.rs.txt", args.len()), ours);
        let _ = std::fs::write(format!("{p}.{}.c.txt", args.len()), &theirs);
    }
    for (n, (a, b)) in ours.lines().zip(theirs.lines()).enumerate() {
        assert_eq!(a, b, "first difference at output line {}", n + 1);
    }
    assert_eq!(ours.lines().count(), theirs.lines().count());
}

fn oracle_steps() -> usize {
    std::env::var("ODE_ORACLE_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(400)
}

#[test]
fn collision_matches_ode_0_15_2() {
    let steps = oracle_steps();
    let ours = collide_scenario(steps);
    compare(&ours, &[&steps.to_string(), "collide"]);
}

#[test]
fn matches_ode_0_15_2() {
    let steps = std::env::var("ODE_ORACLE_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(400);
    let ours = scenario(steps);
    let Ok(exe) = std::env::var("ODE_ORACLE") else {
        eprintln!("ODE_ORACLE not set: ran the scenario only");
        return;
    };
    let o = std::process::Command::new(exe).arg(steps.to_string()).output().expect("run ode_oracle");
    let theirs = String::from_utf8(o.stdout).unwrap();
    if let Ok(p) = std::env::var("ODE_ORACLE_DUMP") {
        let _ = std::fs::write(format!("{p}.rs.txt"), &ours);
        let _ = std::fs::write(format!("{p}.c.txt"), &theirs);
    }
    for (n, (a, b)) in ours.lines().zip(theirs.lines()).enumerate() {
        assert_eq!(a, b, "first difference at output line {} (each step is 1 + 11 lines)", n + 1);
    }
    assert_eq!(ours.lines().count(), theirs.lines().count());
}
