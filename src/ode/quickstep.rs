//! ODE's QuickStep (quickstep.cpp): the iterative SOR (projected
//! Gauss-Seidel) stepper. This is the only stepper AC EVO calls: its
//! physics core passes dWorldQuickStep's own estimator to the island
//! processor (3 * threads + 8 calls), and dWorldStep is never referenced.
//!
//! The single-threaded path, stage by stage. ODE's multithreaded path
//! orders rows by a dependency map so each body sees its rows in the same
//! order, which gives the same numbers.

use super::joints::{Info1, CFM, HI, J1, J2, LO, RHS, ROW};
use super::math::*;
use super::world::{step_body, BodyId, JointId, World, GYROSCOPIC, NO_GRAVITY};

/// iMJ row: inv(M) * J^T for body 1 (6) then body 2 (6)
const IMJ: usize = 12;
/// per-body constraint force / rhs accumulator
const CF: usize = 6;

pub(crate) fn quick_step_island(w: &mut World, bodies: &[BodyId], joints: &[JointId], h: Real) {
    let nb = bodies.len();

    // ---- stage 0, bodies: tags, gravity, world inverse inertia, gyroscopic torque
    for (i, &b) in bodies.iter().enumerate() {
        w.bodies[b].tag = i as i32;
    }
    for axis in 0..3 {
        let g = w.p.gravity[axis];
        if g != 0.0 {
            for &b in bodies {
                let b = &mut w.bodies[b];
                if b.flags & NO_GRAVITY == 0 {
                    b.facc[axis] += b.mass.mass * g;
                }
            }
        }
    }
    let mut inv_i: Vec<M3> = Vec::with_capacity(nb);
    for &bi in bodies {
        let b = &mut w.bodies[bi];
        let tmp = mul2_333(&b.inv_i, &b.r);
        inv_i.push(mul0_333(&b.r, &tmp));
        if b.flags & GYROSCOPIC != 0 && b.inv_mass > 0.0 {
            let tmp = mul2_333(&b.mass.i, &b.r);
            let ii = mul0_333(&b.r, &tmp);
            let mut l = mul0_331(&ii, &b.avel);
            let mut itild = [0.0; 12];
            set_cross_matrix_minus(&mut itild, &l, 4);
            for k in 0..12 {
                itild[k] = itild[k] * h + ii[k];
            }
            let rh = recip(h);
            l[0] *= rh;
            l[1] *= rh;
            l[2] *= rh;
            let mut it_inv = [0.0; 12];
            if invert3(&mut it_inv, &itild) != 0.0 {
                let mut it = mul0_333(&ii, &it_inv);
                it[0] -= 1.0;
                it[5] -= 1.0;
                it[10] -= 1.0;
                let tau0 = mul0_331(&it, &l);
                let t = add3(&b.tacc, &tau0);
                b.tacc[0] = t[0];
                b.tacc[1] = t[1];
                b.tacc[2] = t[2];
            }
        }
    }

    // ---- stage 0, joints: Info1, dropping joints with no rows
    let mut infos: Vec<(JointId, Info1)> = Vec::with_capacity(joints.len());
    let mut m = 0usize;
    let mut mfb = 0usize;
    for &j in joints {
        let info = {
            let (jt, bs) = (&mut w.joints[j], &w.bodies);
            jt.info1(bs)
        };
        if info.m != 0 {
            m += info.m as usize;
            if w.joints[j].feedback.is_some() {
                mfb += info.m as usize;
            }
            infos.push((j, info));
        }
    }
    let nj = infos.len();

    // ---- stage 1: row offsets
    if m > 0 {
        let mut mindex = Vec::with_capacity(nj + 1);
        let (mut moffs, mut fboffs) = (0usize, 0usize);
        mindex.push((0usize, 0usize));
        for &(j, info) in &infos {
            moffs += info.m as usize;
            if w.joints[j].feedback.is_some() {
                fboffs += info.m as usize;
            }
            mindex.push((moffs, fboffs));
        }
        let mut jm = vec![0.0; m * ROW];
        let mut findex = vec![-1i32; m];
        let mut jcopy = vec![0.0; mfb * IMJ];
        let mut jb: Vec<(i32, i32)> = vec![(0, 0); m];

        // ---- stage 2a: Jacobian rows
        let rh = recip(h);
        let (world_erp, world_cfm) = (w.p.erp, w.p.cfm);
        let params = w.p;
        let mut valid_findices = 0usize;
        for (ji, &(j, _)) in infos.iter().enumerate() {
            let ofsi = mindex[ji].0;
            let infom = mindex[ji + 1].0 - ofsi;
            let rows = &mut jm[ofsi * ROW..(ofsi + infom) * ROW];
            for r in 0..infom {
                let row = &mut rows[r * ROW..(r + 1) * ROW];
                row[..6].fill(0.0);
                row[RHS] = 0.0;
                row[CFM] = world_cfm;
                row[J2..J2 + 6].fill(0.0);
                row[LO] = -INFINITY;
                row[HI] = INFINITY;
            }
            let fx = &mut findex[ofsi..ofsi + infom];
            {
                let (jt, bs) = (&mut w.joints[j], &mut w.bodies);
                jt.info2(bs, &params, rh, world_erp, rows, fx);
            }
            for f in fx.iter_mut() {
                if *f != -1 {
                    *f += ofsi as i32;
                    valid_findices += 1;
                }
            }
            for r in 0..infom {
                rows[r * ROW + RHS] *= rh;
                rows[r * ROW + CFM] *= rh;
            }
            let fb0 = mindex[ji].1;
            let mfb_count = mindex[ji + 1].1 - fb0;
            for r in 0..mfb_count {
                let dst = &mut jcopy[(fb0 + r) * IMJ..(fb0 + r + 1) * IMJ];
                dst[..6].copy_from_slice(&rows[r * ROW + J1..r * ROW + J1 + 6]);
                dst[6..].copy_from_slice(&rows[r * ROW + J2..r * ROW + J2 + 6]);
            }
        }
        for (ji, &(j, _)) in infos.iter().enumerate() {
            let jt = &w.joints[j];
            let b1 = jt.node[0].map_or(-1, |b| w.bodies[b].tag);
            let b2 = jt.node[1].map_or(-1, |b| w.bodies[b].tag);
            for e in &mut jb[mindex[ji].0..mindex[ji + 1].0] {
                *e = (b1, b2);
            }
        }

        // ---- stage 2b: rhs_tmp = -(a + v / h), a = inv(M) * external forces
        let mut rhs_tmp = vec![0.0; nb * CF];
        for (bi, &b) in bodies.iter().enumerate() {
            let b = &w.bodies[b];
            let r = &mut rhs_tmp[bi * CF..(bi + 1) * CF];
            for k in 0..3 {
                r[k] = -(b.facc[k] * b.inv_mass + b.lvel[k] * rh);
            }
            let t = mul0_331(&inv_i[bi], &b.tacc);
            for k in 0..3 {
                r[3 + k] = -(b.avel[k] * rh) - t[k];
            }
        }

        // ---- stage 2c: rhs += J * rhs_tmp
        for i in 0..m {
            let (b1, b2) = jb[i];
            let row = &mut jm[i * ROW..(i + 1) * ROW];
            let mut sum = 0.0;
            let inp = &rhs_tmp[b1 as usize * CF..];
            for k in 0..6 {
                sum += row[J1 + k] * inp[k];
            }
            if b2 != -1 {
                let inp = &rhs_tmp[b2 as usize * CF..];
                for k in 0..6 {
                    sum += row[J2 + k] * inp[k];
                }
            }
            row[RHS] += sum;
        }

        // ---- stage 4: the LCP by SOR
        let mut lambda = vec![0.0; m];
        let mut imj = vec![0.0; m * IMJ];
        for i in 0..m {
            let (b1, b2) = jb[i];
            let row = &jm[i * ROW..(i + 1) * ROW];
            let o = &mut imj[i * IMJ..(i + 1) * IMJ];
            let k1 = w.bodies[bodies[b1 as usize]].inv_mass;
            for k in 0..3 {
                o[k] = k1 * row[J1 + k];
            }
            let a = mul0_331(&inv_i[b1 as usize], &row[J1 + 3..]);
            o[3..6].copy_from_slice(&a[..3]);
            if b2 != -1 {
                let k2 = w.bodies[bodies[b2 as usize]].inv_mass;
                for k in 0..3 {
                    o[6 + k] = k2 * row[J2 + k];
                }
                let a = mul0_331(&inv_i[b2 as usize], &row[J2 + 3..]);
                o[9..12].copy_from_slice(&a[..3]);
            }
        }
        let mut fc = vec![0.0; nb * CF];
        // Ad: scale each row by w / (J inv(M) J^T + cfm)
        let sor_w = w.qs.w;
        for i in 0..m {
            let b2 = jb[i].1;
            let o = &imj[i * IMJ..(i + 1) * IMJ];
            let row = &mut jm[i * ROW..(i + 1) * ROW];
            let mut sum = 0.0;
            for k in 0..6 {
                sum += o[k] * row[J1 + k];
            }
            if b2 != -1 {
                for k in 0..6 {
                    sum += o[6 + k] * row[J2 + k];
                }
            }
            let cfm_i = row[CFM];
            let ad = sor_w / (sum + cfm_i);
            row[CFM] = cfm_i * ad;
            row[RHS] *= ad;
            for k in 0..6 {
                row[J1 + k] *= ad;
            }
            if b2 != -1 {
                for k in 0..6 {
                    row[J2 + k] *= ad;
                }
            }
        }
        // rows without a friction index first, friction rows after
        let mut order = vec![0usize; m];
        {
            let (mut head, mut tail) = (0usize, m - valid_findices);
            for i in 0..m {
                if findex[i] == -1 {
                    order[head] = i;
                    head += 1;
                } else {
                    order[tail] = i;
                    tail += 1;
                }
            }
        }
        let head_n = m - valid_findices;
        for iteration in 0..w.qs.num_iterations {
            if iteration & 7 == 0 {
                shuffle(&mut order[..head_n]);
                shuffle(&mut order[head_n..]);
            }
            for &index in &order {
                iteration_step(index, &jm, &imj, &jb, &findex, &mut lambda, &mut fc);
            }
        }

        // ---- stage 4b: velocities from the constraint forces, feedback
        for (bi, &b) in bodies.iter().enumerate() {
            let b = &mut w.bodies[b];
            let c = &fc[bi * CF..(bi + 1) * CF];
            for k in 0..3 {
                b.lvel[k] += h * c[k];
                b.avel[k] += h * c[3 + k];
            }
        }
        if mfb > 0 {
            for (ji, &(j, info)) in infos.iter().enumerate() {
                if w.joints[j].feedback.is_none() {
                    continue;
                }
                let infom = info.m as usize;
                let lam = &lambda[mindex[ji].0..mindex[ji].0 + infom];
                let jc = &jcopy[mindex[ji].1 * IMJ..];
                let d1 = multiply1_12q1(&jc[0..], lam);
                let has2 = w.joints[j].node[1].is_some();
                let d2 = if has2 { Some(multiply1_12q1(&jc[6..], lam)) } else { None };
                let fb = w.joints[j].feedback.as_mut().unwrap();
                fb.f1 = [d1[0], d1[1], d1[2], fb.f1[3]];
                fb.t1 = [d1[3], d1[4], d1[5], fb.t1[3]];
                if let Some(d2) = d2 {
                    fb.f2 = [d2[0], d2[1], d2[2], fb.f2[3]];
                    fb.t2 = [d2[3], d2[4], d2[5], fb.t2[3]];
                }
            }
        }
    }

    // ---- stage 6a: velocities from the external forces
    for (bi, &b) in bodies.iter().enumerate() {
        let b = &mut w.bodies[b];
        let ims = h * b.inv_mass;
        for k in 0..3 {
            b.lvel[k] += ims * b.facc[k];
            b.tacc[k] *= h;
        }
        let t = mul0_331(&inv_i[bi], &b.tacc);
        let a = add3(&b.avel, &t);
        b.avel[0] = a[0];
        b.avel[1] = a[1];
        b.avel[2] = a[2];
    }
    // ---- stage 6b: integrate positions, clear the force accumulators
    for &bi in bodies {
        step_body(&mut w.bodies[bi], h);
        w.body_geoms_moved(bi);
        let b = &mut w.bodies[bi];
        b.facc[0] = 0.0;
        b.facc[1] = 0.0;
        b.facc[2] = 0.0;
        b.tacc[0] = 0.0;
        b.tacc[1] = 0.0;
        b.tacc[2] = 0.0;
    }
}

/// ConstraintsReorderingHelper (RANDOMLY): Fisher-Yates on dRandInt
fn shuffle(order: &mut [usize]) {
    for index in 1..order.len() {
        let swap = rand_int(index as i32 + 1) as usize;
        order.swap(index, swap);
    }
}

/// dxQuickStepIsland_Stage4LCP_IterationStep
#[inline]
fn iteration_step(index: usize, jm: &[Real], imj: &[Real], jb: &[(i32, i32)], findex: &[i32], lambda: &mut [Real], fc: &mut [Real]) {
    let row = &jm[index * ROW..(index + 1) * ROW];
    let old_lambda = lambda[index];
    let mut delta = row[RHS] - old_lambda * row[CFM];
    let (b1, b2) = jb[index];
    let f1 = b1 as usize * CF;
    {
        let c = &fc[f1..f1 + CF];
        delta -= c[0] * row[J1] + c[1] * row[J1 + 1] + c[2] * row[J1 + 2] + c[3] * row[J1 + 3] + c[4] * row[J1 + 4] + c[5] * row[J1 + 5];
    }
    let f2 = if b2 != -1 { Some(b2 as usize * CF) } else { None };
    if let Some(f2) = f2 {
        let c = &fc[f2..f2 + CF];
        delta -= c[0] * row[J2] + c[1] * row[J2 + 1] + c[2] * row[J2 + 2] + c[3] * row[J2 + 3] + c[4] * row[J2 + 4] + c[5] * row[J2 + 5];
    }
    let (hi_act, lo_act) = if findex[index] != -1 {
        let hi = (row[HI] * lambda[findex[index] as usize]).abs();
        (hi, -hi)
    } else {
        (row[HI], row[LO])
    };
    let new_lambda = old_lambda + delta;
    if new_lambda < lo_act {
        delta = lo_act - old_lambda;
        lambda[index] = lo_act;
    } else if new_lambda > hi_act {
        delta = hi_act - old_lambda;
        lambda[index] = hi_act;
    } else {
        lambda[index] = new_lambda;
    }
    let o = &imj[index * IMJ..(index + 1) * IMJ];
    for k in 0..6 {
        fc[f1 + k] += delta * o[k];
    }
    if let Some(f2) = f2 {
        for k in 0..6 {
            fc[f2 + k] += delta * o[6 + k];
        }
    }
}

/// Multiply1_12q1: sum over the joint's rows of Jcopy^T * lambda
fn multiply1_12q1(b: &[Real], c: &[Real]) -> [Real; 6] {
    let mut r = [0.0; 6];
    for (i, &s) in c.iter().enumerate() {
        let k = i * IMJ;
        for t in 0..6 {
            r[t] += b[k + t] * s;
        }
    }
    r
}
