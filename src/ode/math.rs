//! ODE's maths: vectors, 3x3 matrices, quaternions, mass and the random
//! generator (odemath.h/.cpp, rotation.cpp, mass.cpp, misc.cpp).
//!
//! Layouts are ODE's, so the stepper ports read like the C: a vector is
//! [x, y, z, pad], a 3x3 matrix is 3 rows of 4 (the 4th column unused), a
//! quaternion is [w, x, y, z]. Every expression keeps ODE's operation order,
//! which is what makes the results bit-identical.

pub type Real = f64;
pub type V3 = [Real; 4];
pub type M3 = [Real; 12];
pub type Quat = [Real; 4];

pub const INFINITY: Real = f64::INFINITY;

#[inline]
pub fn recip(x: Real) -> Real {
    1.0 / x
}
#[inline]
pub fn recip_sqrt(x: Real) -> Real {
    1.0 / x.sqrt()
}

#[inline]
pub fn dot3(a: &[Real], b: &[Real]) -> Real {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
/// _dCalcVectorDot3 with strides
#[inline]
pub fn dot3s(a: &[Real], b: &[Real], sa: usize, sb: usize) -> Real {
    a[0] * b[0] + a[sa] * b[sb] + a[2 * sa] * b[2 * sb]
}
#[inline]
pub fn dot3_14(a: &[Real], b: &[Real]) -> Real {
    dot3s(a, b, 1, 4)
}
#[inline]
pub fn dot3_41(a: &[Real], b: &[Real]) -> Real {
    dot3s(a, b, 4, 1)
}
#[inline]
pub fn dot3_44(a: &[Real], b: &[Real]) -> Real {
    dot3s(a, b, 4, 4)
}

/// _dCalcVectorCross3 (the result written after all three are computed)
#[inline]
pub fn cross3s(res: &mut [Real], a: &[Real], b: &[Real], sr: usize, sa: usize, sb: usize) {
    let r0 = a[sa] * b[2 * sb] - a[2 * sa] * b[sb];
    let r1 = a[2 * sa] * b[0] - a[0] * b[2 * sb];
    let r2 = a[0] * b[sb] - a[sa] * b[0];
    res[0] = r0;
    res[sr] = r1;
    res[2 * sr] = r2;
}
#[inline]
pub fn cross(a: &[Real], b: &[Real]) -> V3 {
    let mut r = [0.0; 4];
    cross3s(&mut r, a, b, 1, 1, 1);
    r
}
#[inline]
pub fn cross_into(res: &mut [Real], a: &[Real], b: &[Real]) {
    cross3s(res, a, b, 1, 1, 1);
}
/// dAddVectorCross3: res += a x b
#[inline]
pub fn add_cross(res: &mut [Real], a: &[Real], b: &[Real]) {
    let t = cross(a, b);
    let (r0, r1, r2) = (res[0] + t[0], res[1] + t[1], res[2] + t[2]);
    res[0] = r0;
    res[1] = r1;
    res[2] = r2;
}
/// dSubtractVectorCross3: res -= a x b
#[inline]
pub fn sub_cross(res: &mut [Real], a: &[Real], b: &[Real]) {
    let t = cross(a, b);
    let (r0, r1, r2) = (res[0] - t[0], res[1] - t[1], res[2] - t[2]);
    res[0] = r0;
    res[1] = r1;
    res[2] = r2;
}

#[inline]
pub fn v3(x: Real, y: Real, z: Real) -> V3 {
    [x, y, z, 0.0]
}
#[inline]
pub fn add3(a: &[Real], b: &[Real]) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], 0.0]
}
#[inline]
pub fn sub3(a: &[Real], b: &[Real]) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], 0.0]
}
/// dAddVectorScaledVector3: a + s*b
#[inline]
pub fn add_scaled3(a: &[Real], b: &[Real], s: Real) -> V3 {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2], 0.0]
}
/// dAddScaledVectors3: sa*a + sb*b
#[inline]
pub fn add_scaled_vectors3(a: &[Real], b: &[Real], sa: Real, sb: Real) -> V3 {
    [sa * a[0] + sb * b[0], sa * a[1] + sb * b[1], sa * a[2] + sb * b[2], 0.0]
}
/// dCopyScaledVector3: a * s
#[inline]
pub fn scaled3(a: &[Real], s: Real) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s, 0.0]
}
#[inline]
pub fn neg3(a: &[Real]) -> V3 {
    [-a[0], -a[1], -a[2], 0.0]
}
#[inline]
pub fn length3(a: &[Real]) -> Real {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
#[inline]
pub fn length_sq3(a: &[Real]) -> Real {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}
#[inline]
pub fn distance3(a: &[Real], b: &[Real]) -> Real {
    length3(&sub3(a, b))
}
/// dCalcPointDepth3
#[inline]
pub fn point_depth3(p: &[Real], plane_p: &[Real], plane_n: &[Real]) -> Real {
    (plane_p[0] - p[0]) * plane_n[0] + (plane_p[1] - p[1]) * plane_n[1] + (plane_p[2] - p[2]) * plane_n[2]
}
#[inline]
pub fn get_column3(a: &[Real], n: usize) -> V3 {
    [a[n], a[n + 4], a[n + 8], 0.0]
}

/// dSetCrossMatrixPlus (only the off-diagonal entries are written)
pub fn set_cross_matrix_plus(res: &mut [Real], a: &[Real], skip: usize) {
    let (a0, a1, a2) = (a[0], a[1], a[2]);
    res[1] = -a2;
    res[2] = a1;
    res[skip] = a2;
    res[skip + 2] = -a0;
    res[2 * skip] = -a1;
    res[2 * skip + 1] = a0;
}
/// dSetCrossMatrixMinus
pub fn set_cross_matrix_minus(res: &mut [Real], a: &[Real], skip: usize) {
    let (a0, a1, a2) = (a[0], a[1], a[2]);
    res[1] = a2;
    res[2] = -a1;
    res[skip] = -a2;
    res[skip + 2] = a0;
    res[2 * skip] = a1;
    res[2 * skip + 1] = -a0;
}

// dMultiply*: 0 = A*B, 1 = A^T*B, 2 = A*B^T.
/// dMultiply0_331: R * v
#[inline]
pub fn mul0_331(a: &[Real], b: &[Real]) -> V3 {
    [dot3(a, b), dot3(&a[4..], b), dot3(&a[8..], b), 0.0]
}
/// dMultiply1_331: R^T * v
#[inline]
pub fn mul1_331(a: &[Real], b: &[Real]) -> V3 {
    [dot3_41(a, b), dot3_41(&a[1..], b), dot3_41(&a[2..], b), 0.0]
}
/// dMultiply0_133: v^T * R (= dMultiplyHelper1_331(b, a))
#[inline]
pub fn mul0_133(a: &[Real], b: &[Real]) -> V3 {
    mul1_331(b, a)
}
/// dMultiplyHelper1_133
#[inline]
fn helper1_133(a: &[Real], b: &[Real]) -> V3 {
    [dot3_44(a, b), dot3_44(&a[1..], b), dot3_44(&a[2..], b), 0.0]
}
fn put_row(r: &mut M3, i: usize, v: V3) {
    r[i] = v[0];
    r[i + 1] = v[1];
    r[i + 2] = v[2];
}
/// dMultiply0_333: A * B
pub fn mul0_333(a: &[Real], b: &[Real]) -> M3 {
    let mut r = [0.0; 12];
    put_row(&mut r, 0, mul0_133(&a[0..], b));
    put_row(&mut r, 4, mul0_133(&a[4..], b));
    put_row(&mut r, 8, mul0_133(&a[8..], b));
    r
}
/// dMultiply1_333: A^T * B
pub fn mul1_333(a: &[Real], b: &[Real]) -> M3 {
    let mut r = [0.0; 12];
    put_row(&mut r, 0, helper1_133(b, &a[0..]));
    put_row(&mut r, 4, helper1_133(b, &a[1..]));
    put_row(&mut r, 8, helper1_133(b, &a[2..]));
    r
}
/// dMultiply2_333: A * B^T
pub fn mul2_333(a: &[Real], b: &[Real]) -> M3 {
    let mut r = [0.0; 12];
    put_row(&mut r, 0, mul0_331(b, &a[0..]));
    put_row(&mut r, 4, mul0_331(b, &a[4..]));
    put_row(&mut r, 8, mul0_331(b, &a[8..]));
    r
}

/// dCalcMatrix3Det
pub fn det3(m: &[Real]) -> Real {
    m[0] * (m[5] * m[10] - m[9] * m[6]) - m[1] * (m[4] * m[10] - m[8] * m[6]) + m[2] * (m[4] * m[9] - m[8] * m[5])
}
/// dInvertMatrix3: returns the determinant, 0 (dst untouched) if singular
pub fn invert3(dst: &mut [Real], ma: &[Real]) -> Real {
    let det = det3(ma);
    if det == 0.0 {
        return 0.0;
    }
    let r = recip(det);
    dst[0] = (ma[5] * ma[10] - ma[6] * ma[9]) * r;
    dst[1] = (ma[9] * ma[2] - ma[1] * ma[10]) * r;
    dst[2] = (ma[1] * ma[6] - ma[5] * ma[2]) * r;
    dst[4] = (ma[6] * ma[8] - ma[4] * ma[10]) * r;
    dst[5] = (ma[0] * ma[10] - ma[8] * ma[2]) * r;
    dst[6] = (ma[4] * ma[2] - ma[0] * ma[6]) * r;
    dst[8] = (ma[4] * ma[9] - ma[8] * ma[5]) * r;
    dst[9] = (ma[8] * ma[1] - ma[0] * ma[9]) * r;
    dst[10] = (ma[0] * ma[5] - ma[1] * ma[4]) * r;
    det
}

/// dxSafeNormalize3
pub fn safe_normalize3(a: &mut [Real]) -> bool {
    let (b0, b1, b2) = (a[0].abs(), a[1].abs(), a[2].abs());
    let idx = if b1 > b0 {
        if b2 > b1 {
            2
        } else {
            1
        }
    } else if b2 > b0 {
        2
    } else {
        if !(b0 > 0.0) {
            return false;
        }
        0
    };
    match idx {
        0 => {
            let r = recip(b0);
            let a1 = a[1] * r;
            let a2 = a[2] * r;
            let l = recip_sqrt(1.0 + a1 * a1 + a2 * a2);
            a[1] = a1 * l;
            a[2] = a2 * l;
            a[0] = l.copysign(a[0]);
        }
        1 => {
            let r = recip(b1);
            let a0 = a[0] * r;
            let a2 = a[2] * r;
            let l = recip_sqrt(1.0 + a0 * a0 + a2 * a2);
            a[0] = a0 * l;
            a[2] = a2 * l;
            a[1] = l.copysign(a[1]);
        }
        _ => {
            let r = recip(b2);
            let a0 = a[0] * r;
            let a1 = a[1] * r;
            let l = recip_sqrt(1.0 + a0 * a0 + a1 * a1);
            a[0] = a0 * l;
            a[1] = a1 * l;
            a[2] = l.copysign(a[2]);
        }
    }
    true
}
/// dNormalize3 (dxNormalize3 inside ODE)
pub fn normalize3(a: &mut [Real]) {
    if !safe_normalize3(a) {
        a[0] = 1.0;
        a[1] = 0.0;
        a[2] = 0.0;
    }
}
/// dxSafeNormalize4
pub fn safe_normalize4(a: &mut [Real]) -> bool {
    let l = a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3];
    if l > 0.0 {
        let l = recip_sqrt(l);
        a[0] *= l;
        a[1] *= l;
        a[2] *= l;
        a[3] *= l;
        true
    } else {
        false
    }
}
/// dNormalize4
pub fn normalize4(a: &mut [Real]) {
    if !safe_normalize4(a) {
        a[0] = 1.0;
        a[1] = 0.0;
        a[2] = 0.0;
        a[3] = 0.0;
    }
}
/// dPlaneSpace: two unit vectors perpendicular to n and to each other
pub fn plane_space(n: &[Real]) -> (V3, V3) {
    let mut p = [0.0; 4];
    let mut q = [0.0; 4];
    if n[2].abs() > std::f64::consts::FRAC_1_SQRT_2 {
        let a = n[1] * n[1] + n[2] * n[2];
        let k = recip_sqrt(a);
        p[0] = 0.0;
        p[1] = -n[2] * k;
        p[2] = n[1] * k;
        q[0] = a * k;
        q[1] = -n[0] * p[2];
        q[2] = n[0] * p[1];
    } else {
        let a = n[0] * n[0] + n[1] * n[1];
        let k = recip_sqrt(a);
        p[0] = -n[1] * k;
        p[1] = n[0] * k;
        p[2] = 0.0;
        q[0] = -n[2] * p[1];
        q[1] = n[2] * p[0];
        q[2] = a * k;
    }
    (p, q)
}
fn could_be_normalized3(a: &[Real]) -> bool {
    a[0] != 0.0 || a[1] != 0.0 || a[2] != 0.0
}
/// dOrthogonalizeR
pub fn orthogonalize_r(m: &mut M3) -> bool {
    if !could_be_normalized3(&m[0..]) {
        return false;
    }
    let n0 = length_sq3(&m[0..]);
    let mut row2: V3 = [m[4], m[5], m[6], 0.0];
    let proj = dot3(&m[0..], &m[4..]);
    if proj != 0.0 {
        let pd = proj / n0;
        row2 = [m[4] - pd * m[0], m[5] - pd * m[1], m[6] - pd * m[2], 0.0];
    }
    if !could_be_normalized3(&row2) {
        return false;
    }
    if n0 != 1.0 {
        safe_normalize3(&mut m[0..3]);
    }
    let n1 = length_sq3(&row2);
    if n1 != 1.0 {
        safe_normalize3(&mut row2);
    }
    // the second row stays as given when no projection was needed (ODE
    // normalises it in place through the row pointer then)
    if proj == 0.0 {
        m[4] = row2[0];
        m[5] = row2[1];
        m[6] = row2[2];
    }
    let z = cross(&m[0..], &row2);
    m[8] = z[0];
    m[9] = z[1];
    m[10] = z[2];
    m[3] = 0.0;
    m[7] = 0.0;
    m[11] = 0.0;
    true
}

// ---- rotation.cpp

pub const IDENTITY_R: M3 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
pub const IDENTITY_Q: Quat = [1.0, 0.0, 0.0, 0.0];

/// dQFromAxisAndAngle
pub fn q_from_axis_and_angle(ax: Real, ay: Real, az: Real, angle: Real) -> Quat {
    let l = ax * ax + ay * ay + az * az;
    if l > 0.0 {
        let angle = angle * 0.5;
        let w = angle.cos();
        let l = angle.sin() * recip_sqrt(l);
        [w, ax * l, ay * l, az * l]
    } else {
        IDENTITY_Q
    }
}
/// dRFromAxisAndAngle
pub fn r_from_axis_and_angle(ax: Real, ay: Real, az: Real, angle: Real) -> M3 {
    r_from_q(&q_from_axis_and_angle(ax, ay, az, angle))
}
/// dRFromEulerAngles
pub fn r_from_euler_angles(phi: Real, theta: Real, psi: Real) -> M3 {
    let (sphi, cphi) = (phi.sin(), phi.cos());
    let (st, ct) = (theta.sin(), theta.cos());
    let (spsi, cpsi) = (psi.sin(), psi.cos());
    [
        cpsi * ct,
        spsi * ct,
        -st,
        0.0,
        cpsi * st * sphi - spsi * cphi,
        spsi * st * sphi + cpsi * cphi,
        ct * sphi,
        0.0,
        cpsi * st * cphi + spsi * sphi,
        spsi * st * cphi - cpsi * sphi,
        ct * cphi,
        0.0,
    ]
}
/// dRFrom2Axes (None where ODE returns leaving R untouched)
pub fn r_from_2_axes(mut ax: Real, mut ay: Real, mut az: Real, mut bx: Real, mut by: Real, mut bz: Real) -> Option<M3> {
    let l = (ax * ax + ay * ay + az * az).sqrt();
    if l <= 0.0 {
        return None;
    }
    let l = recip(l);
    ax *= l;
    ay *= l;
    az *= l;
    let k = ax * bx + ay * by + az * bz;
    bx -= k * ax;
    by -= k * ay;
    bz -= k * az;
    let l = (bx * bx + by * by + bz * bz).sqrt();
    if l <= 0.0 {
        return None;
    }
    let l = recip(l);
    bx *= l;
    by *= l;
    bz *= l;
    Some([
        ax,
        bx,
        -by * az + ay * bz,
        0.0,
        ay,
        by,
        -bz * ax + az * bx,
        0.0,
        az,
        bz,
        -bx * ay + ax * by,
        0.0,
    ])
}
/// dRFromZAxis
pub fn r_from_z_axis(ax: Real, ay: Real, az: Real) -> M3 {
    let mut n = v3(ax, ay, az);
    normalize3(&mut n);
    let (p, q) = plane_space(&n);
    [p[0], q[0], n[0], 0.0, p[1], q[1], n[1], 0.0, p[2], q[2], n[2], 0.0]
}
/// dQMultiply0: qb * qc
pub fn q_mul0(qb: &[Real], qc: &[Real]) -> Quat {
    [
        qb[0] * qc[0] - qb[1] * qc[1] - qb[2] * qc[2] - qb[3] * qc[3],
        qb[0] * qc[1] + qb[1] * qc[0] + qb[2] * qc[3] - qb[3] * qc[2],
        qb[0] * qc[2] + qb[2] * qc[0] + qb[3] * qc[1] - qb[1] * qc[3],
        qb[0] * qc[3] + qb[3] * qc[0] + qb[1] * qc[2] - qb[2] * qc[1],
    ]
}
/// dQMultiply1: inv(qb) * qc
pub fn q_mul1(qb: &[Real], qc: &[Real]) -> Quat {
    [
        qb[0] * qc[0] + qb[1] * qc[1] + qb[2] * qc[2] + qb[3] * qc[3],
        qb[0] * qc[1] - qb[1] * qc[0] - qb[2] * qc[3] + qb[3] * qc[2],
        qb[0] * qc[2] - qb[2] * qc[0] - qb[3] * qc[1] + qb[1] * qc[3],
        qb[0] * qc[3] - qb[3] * qc[0] - qb[1] * qc[2] + qb[2] * qc[1],
    ]
}
/// dQMultiply2: qb * inv(qc)
pub fn q_mul2(qb: &[Real], qc: &[Real]) -> Quat {
    [
        qb[0] * qc[0] + qb[1] * qc[1] + qb[2] * qc[2] + qb[3] * qc[3],
        -qb[0] * qc[1] + qb[1] * qc[0] - qb[2] * qc[3] + qb[3] * qc[2],
        -qb[0] * qc[2] + qb[2] * qc[0] - qb[3] * qc[1] + qb[1] * qc[3],
        -qb[0] * qc[3] + qb[3] * qc[0] - qb[1] * qc[2] + qb[2] * qc[1],
    ]
}
/// dQMultiply3: inv(qb) * inv(qc)
pub fn q_mul3(qb: &[Real], qc: &[Real]) -> Quat {
    [
        qb[0] * qc[0] - qb[1] * qc[1] - qb[2] * qc[2] - qb[3] * qc[3],
        -qb[0] * qc[1] - qb[1] * qc[0] + qb[2] * qc[3] - qb[3] * qc[2],
        -qb[0] * qc[2] - qb[2] * qc[0] + qb[3] * qc[1] - qb[1] * qc[3],
        -qb[0] * qc[3] - qb[3] * qc[0] + qb[1] * qc[2] - qb[2] * qc[1],
    ]
}
/// dRfromQ
pub fn r_from_q(q: &[Real]) -> M3 {
    let qq1 = 2.0 * q[1] * q[1];
    let qq2 = 2.0 * q[2] * q[2];
    let qq3 = 2.0 * q[3] * q[3];
    [
        1.0 - qq2 - qq3,
        2.0 * (q[1] * q[2] - q[0] * q[3]),
        2.0 * (q[1] * q[3] + q[0] * q[2]),
        0.0,
        2.0 * (q[1] * q[2] + q[0] * q[3]),
        1.0 - qq1 - qq3,
        2.0 * (q[2] * q[3] - q[0] * q[1]),
        0.0,
        2.0 * (q[1] * q[3] - q[0] * q[2]),
        2.0 * (q[2] * q[3] + q[0] * q[1]),
        1.0 - qq1 - qq2,
        0.0,
    ]
}
/// dQfromR
pub fn q_from_r(r: &[Real]) -> Quat {
    let at = |i: usize, j: usize| r[i * 4 + j];
    let mut q = [0.0; 4];
    let tr = at(0, 0) + at(1, 1) + at(2, 2);
    if tr >= 0.0 {
        let s = (tr + 1.0).sqrt();
        q[0] = 0.5 * s;
        let s = 0.5 * recip(s);
        q[1] = (at(2, 1) - at(1, 2)) * s;
        q[2] = (at(0, 2) - at(2, 0)) * s;
        q[3] = (at(1, 0) - at(0, 1)) * s;
        return q;
    }
    let case = if at(1, 1) > at(0, 0) {
        if at(2, 2) > at(1, 1) {
            2
        } else {
            1
        }
    } else if at(2, 2) > at(0, 0) {
        2
    } else {
        0
    };
    match case {
        0 => {
            let s = ((at(0, 0) - (at(1, 1) + at(2, 2))) + 1.0).sqrt();
            q[1] = 0.5 * s;
            let s = 0.5 * recip(s);
            q[2] = (at(0, 1) + at(1, 0)) * s;
            q[3] = (at(2, 0) + at(0, 2)) * s;
            q[0] = (at(2, 1) - at(1, 2)) * s;
        }
        1 => {
            let s = ((at(1, 1) - (at(2, 2) + at(0, 0))) + 1.0).sqrt();
            q[2] = 0.5 * s;
            let s = 0.5 * recip(s);
            q[3] = (at(1, 2) + at(2, 1)) * s;
            q[1] = (at(0, 1) + at(1, 0)) * s;
            q[0] = (at(0, 2) - at(2, 0)) * s;
        }
        _ => {
            let s = ((at(2, 2) - (at(0, 0) + at(1, 1))) + 1.0).sqrt();
            q[3] = 0.5 * s;
            let s = 0.5 * recip(s);
            q[1] = (at(2, 0) + at(0, 2)) * s;
            q[2] = (at(1, 2) + at(2, 1)) * s;
            q[0] = (at(1, 0) - at(0, 1)) * s;
        }
    }
    q
}
/// dDQfromW: the quaternion rate for angular velocity w
pub fn dq_from_w(w: &[Real], q: &[Real]) -> Quat {
    [
        0.5 * (-w[0] * q[1] - w[1] * q[2] - w[2] * q[3]),
        0.5 * (w[0] * q[0] + w[1] * q[3] - w[2] * q[2]),
        0.5 * (-w[0] * q[3] + w[1] * q[0] + w[2] * q[1]),
        0.5 * (w[0] * q[2] - w[1] * q[1] + w[2] * q[0]),
    ]
}

// ---- mass.cpp

/// dMass: mass, centre of gravity (body frame), inertia about it
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mass {
    pub mass: Real,
    pub c: V3,
    pub i: M3,
}
impl Default for Mass {
    fn default() -> Self {
        Mass { mass: 0.0, c: [0.0; 4], i: [0.0; 12] }
    }
}
impl Mass {
    /// dMassSetParameters
    pub fn parameters(m: Real, cg: [Real; 3], i11: Real, i22: Real, i33: Real, i12: Real, i13: Real, i23: Real) -> Mass {
        let mut r = Mass { mass: m, c: [cg[0], cg[1], cg[2], 0.0], ..Default::default() };
        r.i[0] = i11;
        r.i[5] = i22;
        r.i[10] = i33;
        r.i[1] = i12;
        r.i[2] = i13;
        r.i[6] = i23;
        r.i[4] = i12;
        r.i[8] = i13;
        r.i[9] = i23;
        r
    }
    /// dMassSetSphereTotal
    pub fn sphere_total(total: Real, radius: Real) -> Mass {
        let mut m = Mass { mass: total, ..Default::default() };
        let ii = 0.4 * total * radius * radius;
        m.i[0] = ii;
        m.i[5] = ii;
        m.i[10] = ii;
        m
    }
    /// dMassSetSphere
    pub fn sphere(density: Real, radius: Real) -> Mass {
        Self::sphere_total((4.0 / 3.0) * std::f64::consts::PI * radius * radius * radius * density, radius)
    }
    /// dMassSetCapsule; direction 1 = x, 2 = y, 3 = z
    pub fn capsule(density: Real, direction: usize, radius: Real, length: Real) -> Mass {
        use std::f64::consts::PI;
        let mut m = Mass::default();
        let m1 = PI * radius * radius * length * density;
        let m2 = (4.0 / 3.0) * PI * radius * radius * radius * density;
        m.mass = m1 + m2;
        let ia = m1 * (0.25 * radius * radius + (1.0 / 12.0) * length * length)
            + m2 * (0.4 * radius * radius + 0.375 * radius * length + 0.25 * length * length);
        let ib = (m1 * 0.5 + m2 * 0.4) * radius * radius;
        m.i[0] = ia;
        m.i[5] = ia;
        m.i[10] = ia;
        m.i[(direction - 1) * 5] = ib;
        m
    }
    /// dMassSetCapsuleTotal
    pub fn capsule_total(total: Real, direction: usize, radius: Real, length: Real) -> Mass {
        let mut m = Self::capsule(1.0, direction, radius, length);
        m.adjust(total);
        m
    }
    /// dMassSetCylinderTotal
    pub fn cylinder_total(total: Real, direction: usize, radius: Real, length: Real) -> Mass {
        let mut m = Mass { mass: total, ..Default::default() };
        let r2 = radius * radius;
        let i = total * (0.25 * r2 + (1.0 / 12.0) * length * length);
        m.i[0] = i;
        m.i[5] = i;
        m.i[10] = i;
        m.i[(direction - 1) * 5] = total * 0.5 * r2;
        m
    }
    /// dMassSetCylinder
    pub fn cylinder(density: Real, direction: usize, radius: Real, length: Real) -> Mass {
        Self::cylinder_total(std::f64::consts::PI * radius * radius * length * density, direction, radius, length)
    }
    /// dMassSetBoxTotal
    pub fn box_total(total: Real, lx: Real, ly: Real, lz: Real) -> Mass {
        let mut m = Mass { mass: total, ..Default::default() };
        m.i[0] = total / 12.0 * (ly * ly + lz * lz);
        m.i[5] = total / 12.0 * (lx * lx + lz * lz);
        m.i[10] = total / 12.0 * (lx * lx + ly * ly);
        m
    }
    /// dMassSetBox
    pub fn box_density(density: Real, lx: Real, ly: Real, lz: Real) -> Mass {
        Self::box_total(lx * ly * lz * density, lx, ly, lz)
    }
    /// dMassAdjust
    pub fn adjust(&mut self, newmass: Real) {
        let scale = newmass / self.mass;
        self.mass = newmass;
        for i in 0..3 {
            for j in 0..3 {
                self.i[i * 4 + j] *= scale;
            }
        }
    }
    /// dMassTranslate
    pub fn translate(&mut self, x: Real, y: Real, z: Real) {
        let mut chat = [0.0; 12];
        set_cross_matrix_plus(&mut chat, &self.c, 4);
        let a = [x + self.c[0], y + self.c[1], z + self.c[2]];
        let mut ahat = [0.0; 12];
        set_cross_matrix_plus(&mut ahat, &a, 4);
        let t1 = mul0_333(&ahat, &ahat);
        let t2 = mul0_333(&chat, &chat);
        for i in 0..3 {
            for j in 0..3 {
                self.i[i * 4 + j] += self.mass * (t2[i * 4 + j] - t1[i * 4 + j]);
            }
        }
        self.i[4] = self.i[1];
        self.i[8] = self.i[2];
        self.i[9] = self.i[6];
        self.c[0] += x;
        self.c[1] += y;
        self.c[2] += z;
    }
    /// dMassRotate
    pub fn rotate(&mut self, r: &M3) {
        let t1 = mul2_333(&self.i, r);
        self.i = mul0_333(r, &t1);
        self.i[4] = self.i[1];
        self.i[8] = self.i[2];
        self.i[9] = self.i[6];
        let t2 = mul0_331(r, &self.c);
        self.c[0] = t2[0];
        self.c[1] = t2[1];
        self.c[2] = t2[2];
    }
    /// dMassAdd
    pub fn add(&mut self, b: &Mass) {
        let denom = recip(self.mass + b.mass);
        for i in 0..3 {
            self.c[i] = (self.c[i] * self.mass + b.c[i] * b.mass) * denom;
        }
        self.mass += b.mass;
        for i in 0..12 {
            self.i[i] += b.i[i];
        }
    }
}

// ---- misc.cpp: the random generator quickstep reorders constraints with.
// One global seed, as in ODE.

use std::sync::atomic::{AtomicU32, Ordering};
static SEED: AtomicU32 = AtomicU32::new(0);

/// dRand
pub fn rand() -> u32 {
    let s = SEED.load(Ordering::Relaxed).wrapping_mul(1664525).wrapping_add(1013904223);
    SEED.store(s, Ordering::Relaxed);
    s
}
pub fn rand_get_seed() -> u32 {
    SEED.load(Ordering::Relaxed)
}
pub fn rand_set_seed(s: u32) {
    SEED.store(s, Ordering::Relaxed);
}
/// dRandInt: 0..n
pub fn rand_int(n: i32) -> i32 {
    let mut r = rand();
    let un = n as u32;
    if un <= 0x10 {
        r ^= r >> 16;
        r ^= r >> 8;
        r ^= r >> 4;
        if un <= 0x2 {
            r ^= r >> 2;
            r ^= r >> 1;
            (r & (un >> 1)) as i32
        } else if un <= 0x4 {
            r ^= r >> 2;
            (((r & 0x03) * un) >> 2) as i32
        } else {
            (((r & 0x0F) * un) >> 4) as i32
        }
    } else if un <= 0x100 {
        r ^= r >> 16;
        r ^= r >> 8;
        (((r & 0xFF) * un) >> 8) as i32
    } else if un <= 0x10000 {
        r ^= r >> 16;
        (((r & 0xFFFF) * un) >> 16) as i32
    } else {
        ((r as u64 * un as u64) >> 32) as i32
    }
}
/// dRandReal
pub fn rand_real() -> Real {
    rand() as f64 / 0xffffffffu32 as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rand_matches_ode_self_test() {
        let old = rand_get_seed();
        rand_set_seed(0);
        let got: Vec<u32> = (0..5).map(|_| rand()).collect();
        rand_set_seed(old);
        assert_eq!(got, [0x3c6ef35f, 0x47502932, 0xd1ccf6e9, 0xaaf95334, 0x6252e503]);
    }
}
