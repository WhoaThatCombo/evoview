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
