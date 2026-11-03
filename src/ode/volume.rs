//! OPCODE's volume colliders (sphere, OBB, LSS) over the no-leaf tree:
//! VolumeCollider's walk plus each volume's overlap tests, in single
//! precision as in OPCODE.

use super::opcode::{dot, invert_pr, transform_point, Child, MeshData, Model, M44, P3};

#[inline]
fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn sqmag(a: P3) -> f32 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}

/// One query volume: the per-node and per-triangle tests
pub trait Volume {
    fn box_overlap(&mut self, c: P3, e: P3) -> bool;
    fn contains_box(&self, c: P3, e: P3) -> bool;
    fn tri_overlap(&mut self, v0: P3, v1: P3, v2: P3) -> bool;
}

/// VolumeCollider: overlap test per node, a whole subtree dumped when the
/// volume contains its box, primitive tests at the leaves. `touched` holds
/// the triangles in OPCODE's order.
pub struct VolumeQuery<'a, V: Volume> {
    pub vol: V,
    mesh: &'a MeshData,
    pub contact: bool,
    pub touched: Vec<u32>,
}

impl<'a, V: Volume> VolumeQuery<'a, V> {
    pub fn new(vol: V, mesh: &'a MeshData) -> Self {
        VolumeQuery { vol, mesh, contact: false, touched: Vec::new() }
    }
    fn prim(&mut self, p: u32) {
        let v = self.mesh.tri(p as usize);
        if self.vol.tri_overlap(v[0], v[1], v[2]) {
            self.contact = true;
            self.touched.push(p);
        }
    }
    /// _Dump(const AABBNoLeafNode*)
    fn dump(&mut self, m: &Model, n: u32) {
        let node = m.nodes[n as usize];
        match node.pos {
            Child::Prim(p) => self.touched.push(p),
            Child::Node(c) => self.dump(m, c),
        }
        match node.neg {
            Child::Prim(p) => self.touched.push(p),
            Child::Node(c) => self.dump(m, c),
        }
    }
    /// _Collide(const AABBNoLeafNode*)
    fn walk(&mut self, m: &Model, n: u32) {
        let node = m.nodes[n as usize];
        if !self.vol.box_overlap(node.center, node.extents) {
            return;
        }
        if self.vol.contains_box(node.center, node.extents) {
            self.contact = true;
            self.dump(m, n);
            return;
        }
        match node.pos {
            Child::Prim(p) => self.prim(p),
            Child::Node(c) => self.walk(m, c),
        }
        match node.neg {
            Child::Prim(p) => self.prim(p),
            Child::Node(c) => self.walk(m, c),
        }
    }
    /// Collide over the whole model (no first-contact, no coherence)
    pub fn run(&mut self, m: &Model) {
        self.contact = false;
        self.touched.clear();
        if m.single_node {
            self.prim(0);
            return;
        }
        self.walk(m, 0);
    }
}

/// SphereCollider's volume
pub struct SphereVol {
    pub center: P3,
    pub radius2: f32,
}
impl SphereVol {
    /// InitQuery: the sphere into mesh space (worldm = the mesh matrix)
    pub fn new(center: P3, radius: f32, worldm: &M44) -> SphereVol {
        let inv = invert_pr(worldm);
        SphereVol { center: transform_point(center, &inv), radius2: radius * radius }
    }
    fn sqdist(&self, p: P3) -> f32 {
        let c = self.center;
        (c[0] - p[0]) * (c[0] - p[0]) + (c[1] - p[1]) * (c[1] - p[1]) + (c[2] - p[2]) * (c[2] - p[2])
    }
}
impl Volume for SphereVol {
    fn box_overlap(&mut self, center: P3, extents: P3) -> bool {
        let mut d = 0.0f32;
        for a in 0..3 {
            let tmp = self.center[a] - center[a];
            let s = tmp + extents[a];
            if s < 0.0 {
                d += s * s;
                if d > self.radius2 {
                    return false;
                }
            } else {
                let s = tmp - extents[a];
                if s > 0.0 {
                    d += s * s;
                    if d > self.radius2 {
                        return false;
                    }
                }
            }
        }
        d <= self.radius2
    }
    fn contains_box(&self, bc: P3, be: P3) -> bool {
        let corners = [
            [bc[0] + be[0], bc[1] + be[1], bc[2] + be[2]],
            [bc[0] - be[0], bc[1] + be[1], bc[2] + be[2]],
            [bc[0] + be[0], bc[1] - be[1], bc[2] + be[2]],
            [bc[0] - be[0], bc[1] - be[1], bc[2] + be[2]],
            [bc[0] + be[0], bc[1] + be[1], bc[2] - be[2]],
            [bc[0] - be[0], bc[1] + be[1], bc[2] - be[2]],
            [bc[0] + be[0], bc[1] - be[1], bc[2] - be[2]],
            [bc[0] - be[0], bc[1] - be[1], bc[2] - be[2]],
        ];
        corners.iter().all(|&p| self.sqdist(p) < self.radius2)
    }
    /// SphereTriOverlap (Eberly's point-triangle distance)
    fn tri_overlap(&mut self, v0: P3, v1: P3, v2: P3) -> bool {
        let r2 = self.radius2;
        let mut kdiff = sub(v2, self.center);
        let mut fc = sqmag(kdiff);
        if fc <= r2 {
            return true;
        }
        kdiff = sub(v1, self.center);
        fc = sqmag(kdiff);
        if fc <= r2 {
            return true;
        }
        kdiff = sub(v0, self.center);
        fc = sqmag(kdiff);
        if fc <= r2 {
            return true;
        }
        let e0 = sub(v1, v0);
        let e1 = sub(v2, v0);
        let a00 = sqmag(e0);
        let a01 = dot(e0, e1);
        let a11 = sqmag(e1);
        let b0 = dot(kdiff, e0);
        let b1 = dot(kdiff, e1);
        let det = (a00 * a11 - a01 * a01).abs();
        let mut u = a01 * b1 - a11 * b0;
        let mut v = a01 * b0 - a00 * b1;
        let sq;
        if u + v <= det {
            if u < 0.0 {
                if v < 0.0 {
                    if b0 < 0.0 {
                        if -b0 >= a00 {
                            sq = a00 + 2.0 * b0 + fc;
                        } else {
                            u = -b0 / a00;
                            sq = b0 * u + fc;
                        }
                    } else if b1 >= 0.0 {
                        sq = fc;
                    } else if -b1 >= a11 {
                        sq = a11 + 2.0 * b1 + fc;
                    } else {
                        v = -b1 / a11;
                        sq = b1 * v + fc;
                    }
                } else if b1 >= 0.0 {
                    sq = fc;
                } else if -b1 >= a11 {
                    sq = a11 + 2.0 * b1 + fc;
                } else {
                    v = -b1 / a11;
                    sq = b1 * v + fc;
                }
            } else if v < 0.0 {
                if b0 >= 0.0 {
                    sq = fc;
                } else if -b0 >= a00 {
                    sq = a00 + 2.0 * b0 + fc;
                } else {
                    u = -b0 / a00;
                    sq = b0 * u + fc;
                }
            } else if det == 0.0 {
                sq = f32::MAX;
            } else {
                let inv = 1.0 / det;
                u *= inv;
                v *= inv;
                sq = u * (a00 * u + a01 * v + 2.0 * b0) + v * (a01 * u + a11 * v + 2.0 * b1) + fc;
            }
        } else if u < 0.0 {
            let t0 = a01 + b0;
            let t1 = a11 + b1;
            if t1 > t0 {
                let num = t1 - t0;
                let den = a00 - 2.0 * a01 + a11;
                if num >= den {
                    sq = a00 + 2.0 * b0 + fc;
                } else {
                    u = num / den;
                    v = 1.0 - u;
                    sq = u * (a00 * u + a01 * v + 2.0 * b0) + v * (a01 * u + a11 * v + 2.0 * b1) + fc;
                }
            } else if t1 <= 0.0 {
                sq = a11 + 2.0 * b1 + fc;
            } else if b1 >= 0.0 {
                sq = fc;
            } else {
                v = -b1 / a11;
                sq = b1 * v + fc;
            }
        } else if v < 0.0 {
            let t0 = a01 + b1;
            let t1 = a00 + b0;
            if t1 > t0 {
                let num = t1 - t0;
                let den = a00 - 2.0 * a01 + a11;
                if num >= den {
                    sq = a11 + 2.0 * b1 + fc;
                } else {
                    v = num / den;
                    u = 1.0 - v;
                    sq = u * (a00 * u + a01 * v + 2.0 * b0) + v * (a01 * u + a11 * v + 2.0 * b1) + fc;
                }
            } else if t1 <= 0.0 {
                sq = a00 + 2.0 * b0 + fc;
            } else if b0 >= 0.0 {
                sq = fc;
            } else {
                u = -b0 / a00;
                sq = b0 * u + fc;
            }
        } else {
            let num = a11 + b1 - a01 - b0;
            if num <= 0.0 {
                sq = a11 + 2.0 * b1 + fc;
            } else {
                let den = a00 - 2.0 * a01 + a11;
                if num >= den {
                    sq = a00 + 2.0 * b0 + fc;
                } else {
                    u = num / den;
                    v = 1.0 - u;
                    sq = u * (a00 * u + a01 * v + 2.0 * b0) + v * (a01 * u + a11 * v + 2.0 * b1) + fc;
                }
            }
        }
        sq.abs() < r2
    }
}

/// OBBCollider's volume. Box and mesh matrices are OPCODE row-vector 4x4s.
pub struct ObbVol {
    extents: P3,
    r_m2b: [[f32; 3]; 3],
    t_m2b: P3,
    r_b2m: [[f32; 3]; 3],
    t_b2m: P3,
    ar: [[f32; 3]; 3],
    b0: P3,
    b1: P3,
    bbx1: f32,
    bby1: f32,
    bbz1: f32,
    bb: [f32; 9],
}

pub fn mul44(a: &M44, b: &M44) -> M44 {
    let mut r = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    r
}
pub fn m33(m: &M44) -> [[f32; 3]; 3] {
    [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]]
}
#[inline]
fn greater(x: f32, y: f32) -> bool {
    x.abs() > y
}
#[inline]
fn min3(a: f32, b: f32, c: f32) -> f32 {
    if a < b {
        if a < c {
            a
        } else {
            c
        }
    } else if b < c {
        b
    } else {
        c
    }
}
#[inline]
fn max3(a: f32, b: f32, c: f32) -> f32 {
    if a > b {
        if a > c {
            a
        } else {
            c
        }
    } else if b > c {
        b
    } else {
        c
    }
}

impl ObbVol {
    /// InitQuery(box, worldb = null, worldm): `rot` rows are OPCODE box.mRot
    /// rows (the columns of the ODE rotation)
    pub fn new(center: P3, extents: P3, rot: [[f32; 3]; 3], worldm: &M44) -> ObbVol {
        let mut world_b: M44 = [[rot[0][0], rot[0][1], rot[0][2], 0.0], [rot[1][0], rot[1][1], rot[1][2], 0.0], [rot[2][0], rot[2][1], rot[2][2], 0.0], [0.0, 0.0, 0.0, 1.0]];
        world_b[3][0] = center[0];
        world_b[3][1] = center[1];
        world_b[3][2] = center[2];
        let inv_b = invert_pr(&world_b);
        let inv_m = invert_pr(worldm);
        let b2m = mul44(&world_b, &inv_m);
        let m2b = mul44(worldm, &inv_b);
        let r_m2b = m33(&m2b);
        let t_m2b = [m2b[3][0], m2b[3][1], m2b[3][2]];
        let r_b2m = m33(&b2m);
        let t_b2m = [b2m[3][0], b2m[3][1], b2m[3][2]];
        let mut ar = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                ar[i][j] = 1e-6 + r_b2m[i][j].abs();
            }
        }
        let e = extents;
        let b0 = sub(e, t_m2b);
        let b1 = sub([-e[0], -e[1], -e[2]], t_m2b);
        ObbVol {
            extents,
            r_m2b,
            t_m2b,
            r_b2m,
            t_b2m,
            ar,
            b0,
            b1,
            bbx1: e[0] * ar[0][0] + e[1] * ar[1][0] + e[2] * ar[2][0],
            bby1: e[0] * ar[0][1] + e[1] * ar[1][1] + e[2] * ar[2][1],
            bbz1: e[0] * ar[0][2] + e[1] * ar[1][2] + e[2] * ar[2][2],
            bb: [
                e[1] * ar[2][0] + e[2] * ar[1][0],
                e[0] * ar[2][0] + e[2] * ar[0][0],
                e[0] * ar[1][0] + e[1] * ar[0][0],
                e[1] * ar[2][1] + e[2] * ar[1][1],
                e[0] * ar[2][1] + e[2] * ar[0][1],
                e[0] * ar[1][1] + e[1] * ar[0][1],
                e[1] * ar[2][2] + e[2] * ar[1][2],
                e[0] * ar[2][2] + e[2] * ar[0][2],
                e[0] * ar[1][2] + e[1] * ar[0][2],
            ],
        }
    }
    /// TransformPoint into box space
    fn to_box(&self, s: P3) -> P3 {
        let (r, t) = (&self.r_m2b, &self.t_m2b);
        [
            t[0] + s[0] * r[0][0] + s[1] * r[1][0] + s[2] * r[2][0],
            t[1] + s[0] * r[0][1] + s[1] * r[1][1] + s[2] * r[2][1],
            t[2] + s[0] * r[0][2] + s[1] * r[1][2] + s[2] * r[2][2],
        ]
    }
}

/// planeBoxOverlap
fn plane_box_overlap(normal: P3, d: f32, maxbox: P3) -> bool {
    let mut vmin = [0.0f32; 3];
    let mut vmax = [0.0f32; 3];
    for q in 0..3 {
        if normal[q] > 0.0 {
            vmin[q] = -maxbox[q];
            vmax[q] = maxbox[q];
        } else {
            vmin[q] = maxbox[q];
            vmax[q] = -maxbox[q];
        }
    }
    if dot(normal, vmin) + d > 0.0 {
        return false;
    }
    dot(normal, vmax) + d >= 0.0
}

/// TriBoxOverlap for a box at the origin with half-sizes `ext`, on
/// triangle vertices already relative to its centre (OPC_USE_FCOMI path)
pub fn tri_box_overlap(v0: P3, v1: P3, v2: P3, ext: P3) -> bool {
    for a in 0..3 {
        if min3(v0[a], v1[a], v2[a]) > ext[a] {
            return false;
        }
        if max3(v0[a], v1[a], v2[a]) < -ext[a] {
            return false;
        }
    }
    let e0 = sub(v1, v0);
    let e1 = sub(v2, v1);
    let normal = super::opcode::cross(e0, e1);
    let d = dot([-normal[0], -normal[1], -normal[2]], v0);
    if !plane_box_overlap(normal, d, ext) {
        return false;
    }
    // IMPLEMENT_CLASS3_TESTS
    let test = |mut mn: f32, mut mx: f32, rad: f32| -> bool {
        if mn > mx {
            std::mem::swap(&mut mn, &mut mx);
        }
        !(mn > rad || mx < -rad)
    };
    let x01 = |a: f32, b: f32, fa: f32, fb: f32| test(a * v0[1] - b * v0[2], a * v2[1] - b * v2[2], fa * ext[1] + fb * ext[2]);
    let x2 = |a: f32, b: f32, fa: f32, fb: f32| test(a * v0[1] - b * v0[2], a * v1[1] - b * v1[2], fa * ext[1] + fb * ext[2]);
    let y02 = |a: f32, b: f32, fa: f32, fb: f32| test(b * v0[2] - a * v0[0], b * v2[2] - a * v2[0], fa * ext[0] + fb * ext[2]);
    let y1 = |a: f32, b: f32, fa: f32, fb: f32| test(b * v0[2] - a * v0[0], b * v1[2] - a * v1[0], fa * ext[0] + fb * ext[2]);
    let z12 = |a: f32, b: f32, fa: f32, fb: f32| test(a * v1[0] - b * v1[1], a * v2[0] - b * v2[1], fa * ext[0] + fb * ext[1]);
    let z0 = |a: f32, b: f32, fa: f32, fb: f32| test(a * v0[0] - b * v0[1], a * v1[0] - b * v1[1], fa * ext[0] + fb * ext[1]);
    let (fey0, fez0) = (e0[1].abs(), e0[2].abs());
    if !x01(e0[2], e0[1], fez0, fey0) {
        return false;
    }
    let fex0 = e0[0].abs();
    if !y02(e0[2], e0[0], fez0, fex0) || !z12(e0[1], e0[0], fey0, fex0) {
        return false;
    }
    let (fey1, fez1) = (e1[1].abs(), e1[2].abs());
    if !x01(e1[2], e1[1], fez1, fey1) {
        return false;
    }
    let fex1 = e1[0].abs();
    if !y02(e1[2], e1[0], fez1, fex1) || !z0(e1[1], e1[0], fey1, fex1) {
        return false;
    }
    let e2 = sub(v0, v2);
    let (fey2, fez2) = (e2[1].abs(), e2[2].abs());
    if !x2(e2[2], e2[1], fez2, fey2) {
        return false;
    }
    let fex2 = e2[0].abs();
    y1(e2[2], e2[0], fez2, fex2) && z12(e2[1], e2[0], fey2, fex2)
}

impl Volume for ObbVol {
    /// OBBCollider::BoxBoxOverlap (full test)
    fn box_overlap(&mut self, center: P3, extents: P3) -> bool {
        let (r, ar, t) = (&self.r_b2m, &self.ar, &self.t_b2m);
        let tx = t[0] - center[0];
        if greater(tx, extents[0] + self.bbx1) {
            return false;
        }
        let ty = t[1] - center[1];
        if greater(ty, extents[1] + self.bby1) {
            return false;
        }
        let tz = t[2] - center[2];
        if greater(tz, extents[2] + self.bbz1) {
            return false;
        }
        let be = self.extents;
        for i in 0..3 {
            let tt = tx * r[i][0] + ty * r[i][1] + tz * r[i][2];
            let t2 = extents[0] * ar[i][0] + extents[1] * ar[i][1] + extents[2] * ar[i][2] + be[i];
            if greater(tt, t2) {
                return false;
            }
        }
        let bb = &self.bb;
        let e = extents;
        let tests = [
            (tz * r[0][1] - ty * r[0][2], e[1] * ar[0][2] + e[2] * ar[0][1] + bb[0]),
            (tz * r[1][1] - ty * r[1][2], e[1] * ar[1][2] + e[2] * ar[1][1] + bb[1]),
            (tz * r[2][1] - ty * r[2][2], e[1] * ar[2][2] + e[2] * ar[2][1] + bb[2]),
            (tx * r[0][2] - tz * r[0][0], e[0] * ar[0][2] + e[2] * ar[0][0] + bb[3]),
            (tx * r[1][2] - tz * r[1][0], e[0] * ar[1][2] + e[2] * ar[1][0] + bb[4]),
            (tx * r[2][2] - tz * r[2][0], e[0] * ar[2][2] + e[2] * ar[2][0] + bb[5]),
            (ty * r[0][0] - tx * r[0][1], e[0] * ar[0][1] + e[1] * ar[0][0] + bb[6]),
            (ty * r[1][0] - tx * r[1][1], e[0] * ar[1][1] + e[1] * ar[1][0] + bb[7]),
            (ty * r[2][0] - tx * r[2][1], e[0] * ar[2][1] + e[1] * ar[2][0] + bb[8]),
        ];
        tests.iter().all(|&(tt, t2)| !greater(tt, t2))
    }
    /// OBBContainsBox
    fn contains_box(&self, bc: P3, be: P3) -> bool {
        let r = &self.r_m2b;
        for a in 0..3 {
            let nc = bc[0] * r[0][a] + bc[1] * r[1][a] + bc[2] * r[2][a];
            let ne = (r[0][a] * be[0]).abs() + (r[1][a] * be[1]).abs() + (r[2][a] * be[2]).abs();
            if self.b0[a] < nc + ne || self.b1[a] > nc - ne {
                return false;
            }
        }
        true
    }
    /// OBB_PRIM: the triangle into box space, then TriBoxOverlap
    fn tri_overlap(&mut self, v0: P3, v1: P3, v2: P3) -> bool {
        let (a, b, c) = (self.to_box(v0), self.to_box(v1), self.to_box(v2));
        tri_box_overlap(a, b, c, self.extents)
    }
}
