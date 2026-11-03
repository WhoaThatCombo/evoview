//! The parts of OPCODE 1.3 (Pierre Terdiman, as bundled with ODE 0.15.2)
//! that ODE's trimesh colliders use: the AABB tree build (splatter-points
//! split at the geometric centre, mLimit 1), its "no-leaf" optimised form,
//! and the queries. All of it runs in single precision, like OPCODE.
//!
//! The tree decides which triangles a query returns and in which order, so
//! the build follows OPCODE step by step.

/// OPCODE's Point
pub type P3 = [f32; 3];

const INV3: f32 = 0.333_333_34;
pub const IEEE_MAX_FLOAT: u32 = 0x7f7f_ffff;

#[inline]
pub fn ir(x: f32) -> u32 {
    x.to_bits()
}
#[inline]
fn add(a: P3, b: P3) -> P3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn mul(a: P3, s: f32) -> P3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn dot(a: P3, b: P3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
/// Point::operator^ (cross)
#[inline]
pub fn cross(a: P3, b: P3) -> P3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline]
fn sqmag(a: P3) -> f32 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}
#[inline]
fn fmin(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}
#[inline]
fn fmax(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

/// The mesh as OPCODE sees it: float triangles fetched by index.
pub struct MeshData {
    /// vertices as given (ODE keeps double data in double for its own uses)
    pub verts: Vec<[f64; 3]>,
    pub tris: Vec<[u32; 3]>,
}
impl MeshData {
    /// MeshInterface::GetTriangle (double data is converted to float)
    #[inline]
    pub fn tri(&self, i: usize) -> [P3; 3] {
        let t = self.tris[i];
        let f = |k: u32| -> P3 {
            let v = self.verts[k as usize];
            [v[0] as f32, v[1] as f32, v[2] as f32]
        };
        [f(t[0]), f(t[1]), f(t[2])]
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Child {
    Node(u32),
    Prim(u32),
}

/// AABBNoLeafNode: a box whose children are nodes or single triangles
#[derive(Clone, Copy, Debug)]
pub struct NoLeafNode {
    pub center: P3,
    pub extents: P3,
    pub pos: Child,
    pub neg: Child,
}

/// Model: the no-leaf tree (empty for a single-triangle mesh, which OPCODE
/// flags OPC_SINGLE_NODE)
pub struct Model {
    pub nodes: Vec<NoLeafNode>,
    pub single_node: bool,
}

/// AABBTreeNode during the build
struct BNode {
    center: P3,
    extents: P3,
    start: usize,
    count: usize,
    /// first of the two children in the pool
    pos: Option<usize>,
}

struct Builder<'a> {
    mesh: &'a MeshData,
    prims: Vec<u32>,
    pool: Vec<BNode>,
}

impl Builder<'_> {
    /// ComputeGlobalBox
    fn global_box(&self, start: usize, count: usize) -> (P3, P3) {
        let mut mn = [f32::MAX; 3];
        let mut mx = [-f32::MAX; 3];
        for &p in &self.prims[start..start + count] {
            let v = self.mesh.tri(p as usize);
            for k in 0..3 {
                for a in 0..3 {
                    mn[a] = fmin(mn[a], v[k][a]);
                }
            }
            for k in 0..3 {
                for a in 0..3 {
                    mx[a] = fmax(mx[a], v[k][a]);
                }
            }
        }
        // AABB::SetMinMax
        (mul(add(mx, mn), 0.5), mul(sub(mx, mn), 0.5))
    }
    /// GetSplittingValues: the triangle's centroid
    fn centroid(&self, p: u32) -> P3 {
        let v = self.mesh.tri(p as usize);
        mul(add(add(v[0], v[1]), v[2]), INV3)
    }
    /// AABBTreeNode::Split with GetSplittingValue at the geometric centre
    fn split(&mut self, start: usize, count: usize, axis: usize) -> usize {
        let mut sv = 0.0f32;
        for &p in &self.prims[start..start + count] {
            let v = self.mesh.tri(p as usize);
            sv += v[0][axis];
            sv += v[1][axis];
            sv += v[2][axis];
        }
        let split_value = sv / (count * 3) as f32;
        let mut nb_pos = 0;
        for i in 0..count {
            let p = self.prims[start + i];
            let v = self.mesh.tri(p as usize);
            let pv = (v[0][axis] + v[1][axis] + v[2][axis]) * INV3;
            if pv > split_value {
                self.prims.swap(start + i, start + nb_pos);
                nb_pos += 1;
            }
        }
        nb_pos
    }
    /// _BuildHierarchy + Subdivide (SPLIT_SPLATTER_POINTS, mLimit 1)
    fn build(&mut self, node: usize) {
        let (start, count) = (self.pool[node].start, self.pool[node].count);
        let (c, e) = self.global_box(start, count);
        self.pool[node].center = c;
        self.pool[node].extents = e;
        if count > 1 {
            let mut means = [0.0f32; 3];
            for i in 0..count {
                means = add(means, self.centroid(self.prims[start + i]));
            }
            means = mul(means, 1.0 / count as f32);
            let mut vars = [0.0f32; 3];
            for i in 0..count {
                let d = sub(self.centroid(self.prims[start + i]), means);
                vars = add(vars, [d[0] * d[0], d[1] * d[1], d[2] * d[2]]);
            }
            vars = mul(vars, 1.0 / (count - 1) as f32);
            let mut axis = 0;
            if vars[1] > vars[axis] {
                axis = 1;
            }
            if vars[2] > vars[axis] {
                axis = 2;
            }
            let mut nb_pos = self.split(start, count, axis);
            if nb_pos == 0 || nb_pos == count {
                nb_pos = count >> 1;
            }
            let p = self.pool.len();
            self.pool.push(BNode { center: [0.0; 3], extents: [0.0; 3], start, count: nb_pos, pos: None });
            self.pool.push(BNode { center: [0.0; 3], extents: [0.0; 3], start: start + nb_pos, count: count - nb_pos, pos: None });
            self.pool[node].pos = Some(p);
            self.build(p);
            self.build(p + 1);
        }
    }
}

impl Model {
    /// Model::Build for ODE's OPCODECREATE(noLeaf = true, quantized = false)
    pub fn build(mesh: &MeshData) -> Model {
        let n = mesh.tris.len();
        if n == 1 {
            return Model { nodes: Vec::new(), single_node: true };
        }
        let mut b = Builder { mesh, prims: (0..n as u32).collect(), pool: Vec::with_capacity(2 * n) };
        b.pool.push(BNode { center: [0.0; 3], extents: [0.0; 3], start: 0, count: n, pos: None });
        b.build(0);
        // _BuildNoLeafTree: depth first, pos before neg
        let mut nodes = vec![NoLeafNode { center: [0.0; 3], extents: [0.0; 3], pos: Child::Prim(0), neg: Child::Prim(0) }; n.saturating_sub(1)];
        let mut cur = 1u32;
        fn walk(b: &Builder, nodes: &mut [NoLeafNode], box_id: u32, cur: &mut u32, node: usize) {
            let src = &b.pool[node];
            let p = src.pos.unwrap();
            nodes[box_id as usize].center = src.center;
            nodes[box_id as usize].extents = src.extents;
            for (k, child) in [p, p + 1].into_iter().enumerate() {
                let c = &b.pool[child];
                let link = if c.pos.is_none() {
                    Child::Prim(b.prims[c.start])
                } else {
                    let id = *cur;
                    *cur += 1;
                    walk(b, nodes, id, cur, child);
                    Child::Node(id)
                };
                if k == 0 {
                    nodes[box_id as usize].pos = link;
                } else {
                    nodes[box_id as usize].neg = link;
                }
            }
        }
        if n > 0 {
            walk(&b, &mut nodes, 0, &mut cur, 0);
        }
        Model { nodes, single_node: false }
    }
}

/// CollisionFace
#[derive(Clone, Copy, Debug, Default)]
pub struct Face {
    pub face_id: u32,
    pub distance: f32,
    pub u: f32,
    pub v: f32,
}

/// A 4x4 OPCODE matrix (row vectors): rows 0..2 are the rotation's columns,
/// row 3 the translation (ODE's MakeMatrix)
pub type M44 = [[f32; 4]; 4];

pub fn make_matrix(pos: &[f64], r: &[f64]) -> M44 {
    [
        [r[0] as f32, r[4] as f32, r[8] as f32, 0.0],
        [r[1] as f32, r[5] as f32, r[9] as f32, 0.0],
        [r[2] as f32, r[6] as f32, r[10] as f32, 0.0],
        [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
    ]
}
/// InvertPRMatrix
pub fn invert_pr(s: &M44) -> M44 {
    let mut d = [[0.0f32; 4]; 4];
    d[0][0] = s[0][0];
    d[1][0] = s[0][1];
    d[2][0] = s[0][2];
    d[3][0] = -(s[3][0] * s[0][0] + s[3][1] * s[0][1] + s[3][2] * s[0][2]);
    d[0][1] = s[1][0];
    d[1][1] = s[1][1];
    d[2][1] = s[1][2];
    d[3][1] = -(s[3][0] * s[1][0] + s[3][1] * s[1][1] + s[3][2] * s[1][2]);
    d[0][2] = s[2][0];
    d[1][2] = s[2][1];
    d[2][2] = s[2][2];
    d[3][2] = -(s[3][0] * s[2][0] + s[3][1] * s[2][1] + s[3][2] * s[2][2]);
    d[3][3] = 1.0;
    d
}
/// Point * Matrix4x4 (row vector, with translation)
pub fn transform_point(p: P3, m: &M44) -> P3 {
    [
        p[0] * m[0][0] + p[1] * m[1][0] + p[2] * m[2][0] + m[3][0],
        p[0] * m[0][1] + p[1] * m[1][1] + p[2] * m[2][1] + m[3][1],
        p[0] * m[0][2] + p[1] * m[1][2] + p[2] * m[2][2] + m[3][2],
    ]
}
/// Matrix3x3 * Point (rows dotted with the point)
pub fn rotate_m3(m: &M44, p: P3) -> P3 {
    [
        m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
        m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
        m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
    ]
}
/// Point * Matrix3x3 (row vector)
pub fn transform_m3(p: P3, m: &M44) -> P3 {
    [
        p[0] * m[0][0] + p[1] * m[1][0] + p[2] * m[2][0],
        p[0] * m[0][1] + p[1] * m[1][1] + p[2] * m[2][1],
        p[0] * m[0][2] + p[1] * m[1][2] + p[2] * m[2][2],
    ]
}

/// RayCollider (no callbacks, stabbed faces kept in a list)
pub struct RayCollider<'a> {
    pub first_contact: bool,
    pub closest_hit: bool,
    pub culling: bool,
    pub max_dist: f32,
    mesh: &'a MeshData,
    origin: P3,
    dir: P3,
    fdir: P3,
    data: P3,
    data2: P3,
    contact: bool,
    stabbed: Face,
    pub faces: Vec<Face>,
}

impl<'a> RayCollider<'a> {
    pub fn new(mesh: &'a MeshData) -> Self {
        RayCollider {
            first_contact: false,
            closest_hit: false,
            culling: true,
            max_dist: f32::MAX,
            mesh,
            origin: [0.0; 3],
            dir: [0.0; 3],
            fdir: [0.0; 3],
            data: [0.0; 3],
            data2: [0.0; 3],
            contact: false,
            stabbed: Face::default(),
            faces: Vec::new(),
        }
    }
    fn contact_found(&self) -> bool {
        self.first_contact && self.contact
    }

    /// RayCollider::Collide(world_ray, model, world)
    pub fn collide(&mut self, orig: P3, dir: P3, model: &Model, world: &M44) {
        self.contact = false;
        self.faces.clear();
        self.dir = rotate_m3(world, dir);
        let w = invert_pr(world);
        self.origin = transform_point(orig, &w);
        if model.single_node {
            self.segment_prim(0);
            return;
        }
        let segment = ir(self.max_dist) != IEEE_MAX_FLOAT;
        if segment {
            self.data = mul(mul(self.dir, 0.5), self.max_dist);
            self.data2 = add(self.origin, self.data);
            self.fdir = [self.data[0].abs(), self.data[1].abs(), self.data[2].abs()];
            self.segment_stab(model, 0);
        } else {
            self.fdir = [self.dir[0].abs(), self.dir[1].abs(), self.dir[2].abs()];
            self.ray_stab(model, 0);
        }
    }

    fn segment_aabb(&self, c: P3, e: P3) -> bool {
        let dx = self.data2[0] - c[0];
        if dx.abs() > e[0] + self.fdir[0] {
            return false;
        }
        let dy = self.data2[1] - c[1];
        if dy.abs() > e[1] + self.fdir[1] {
            return false;
        }
        let dz = self.data2[2] - c[2];
        if dz.abs() > e[2] + self.fdir[2] {
            return false;
        }
        let d = self.data;
        let f = d[1] * dz - d[2] * dy;
        if f.abs() > e[1] * self.fdir[2] + e[2] * self.fdir[1] {
            return false;
        }
        let f = d[2] * dx - d[0] * dz;
        if f.abs() > e[0] * self.fdir[2] + e[2] * self.fdir[0] {
            return false;
        }
        let f = d[0] * dy - d[1] * dx;
        if f.abs() > e[0] * self.fdir[1] + e[1] * self.fdir[0] {
            return false;
        }
        true
    }
    fn ray_aabb(&self, c: P3, e: P3) -> bool {
        let (o, d) = (self.origin, self.dir);
        let dx = o[0] - c[0];
        if dx.abs() > e[0] && dx * d[0] >= 0.0 {
            return false;
        }
        let dy = o[1] - c[1];
        if dy.abs() > e[1] && dy * d[1] >= 0.0 {
            return false;
        }
        let dz = o[2] - c[2];
        if dz.abs() > e[2] && dz * d[2] >= 0.0 {
            return false;
        }
        let f = d[1] * dz - d[2] * dy;
        if f.abs() > e[1] * self.fdir[2] + e[2] * self.fdir[1] {
            return false;
        }
        let f = d[2] * dx - d[0] * dz;
        if f.abs() > e[0] * self.fdir[2] + e[2] * self.fdir[0] {
            return false;
        }
        let f = d[0] * dy - d[1] * dx;
        if f.abs() > e[0] * self.fdir[1] + e[1] * self.fdir[0] {
            return false;
        }
        true
    }
    /// RayTriOverlap (Moller-Trumbore)
    fn ray_tri(&mut self, v0: P3, v1: P3, v2: P3) -> bool {
        const EPS: f32 = 0.000001;
        let edge1 = sub(v1, v0);
        let edge2 = sub(v2, v0);
        let pvec = cross(self.dir, edge2);
        let det = dot(edge1, pvec);
        let neg = |x: f32| x.to_bits() & 0x8000_0000 != 0;
        if self.culling {
            if det <= EPS * fmin(sqmag(edge1), sqmag(edge2)) {
                return false;
            }
            let tvec = sub(self.origin, v0);
            self.stabbed.u = dot(tvec, pvec);
            if neg(self.stabbed.u) || ir(self.stabbed.u) > ir(det) {
                return false;
            }
            let qvec = cross(tvec, edge1);
            self.stabbed.v = dot(self.dir, qvec);
            if neg(self.stabbed.v) || self.stabbed.u + self.stabbed.v > det {
                return false;
            }
            self.stabbed.distance = dot(edge2, qvec);
            if neg(self.stabbed.distance) {
                return false;
            }
            let inv = 1.0 / det;
            self.stabbed.distance *= inv;
            self.stabbed.u *= inv;
            self.stabbed.v *= inv;
        } else {
            if det.abs() <= EPS * fmin(sqmag(edge1), sqmag(edge2)) {
                return false;
            }
            let inv = 1.0 / det;
            let tvec = sub(self.origin, v0);
            self.stabbed.u = dot(tvec, pvec) * inv;
            if neg(self.stabbed.u) || ir(self.stabbed.u) > 0x3f80_0000 {
                return false;
            }
            let qvec = cross(tvec, edge1);
            self.stabbed.v = dot(self.dir, qvec) * inv;
            if neg(self.stabbed.v) || self.stabbed.u + self.stabbed.v > 1.0 {
                return false;
            }
            self.stabbed.distance = dot(edge2, qvec) * inv;
            if neg(self.stabbed.distance) {
                return false;
            }
        }
        true
    }
    /// HANDLE_CONTACT with a stabbed-faces list
    fn handle_contact(&mut self, prim: u32) {
        self.contact = true;
        self.stabbed.face_id = prim;
        if !self.closest_hit || self.faces.is_empty() {
            self.faces.push(self.stabbed);
        } else if self.stabbed.distance < self.faces[0].distance {
            self.faces[0] = self.stabbed;
        }
    }
    fn segment_prim(&mut self, prim: u32) {
        let v = self.mesh.tri(prim as usize);
        if self.ray_tri(v[0], v[1], v[2]) && ir(self.stabbed.distance) < ir(self.max_dist) {
            self.handle_contact(prim);
        }
    }
    fn ray_prim(&mut self, prim: u32) {
        let v = self.mesh.tri(prim as usize);
        if self.ray_tri(v[0], v[1], v[2]) {
            self.handle_contact(prim);
        }
    }
    fn segment_stab(&mut self, m: &Model, n: u32) {
        let node = m.nodes[n as usize];
        if !self.segment_aabb(node.center, node.extents) {
            return;
        }
        match node.pos {
            Child::Prim(p) => self.segment_prim(p),
            Child::Node(c) => self.segment_stab(m, c),
        }
        if self.contact_found() {
            return;
        }
        match node.neg {
            Child::Prim(p) => self.segment_prim(p),
            Child::Node(c) => self.segment_stab(m, c),
        }
    }
    fn ray_stab(&mut self, m: &Model, n: u32) {
        let node = m.nodes[n as usize];
        if !self.ray_aabb(node.center, node.extents) {
            return;
        }
        match node.pos {
            Child::Prim(p) => self.ray_prim(p),
            Child::Node(c) => self.ray_stab(m, c),
        }
        if self.contact_found() {
            return;
        }
        match node.neg {
            Child::Prim(p) => self.ray_prim(p),
            Child::Node(c) => self.ray_stab(m, c),
        }
    }
}

/// OPCODE's TransformPoint (translation first in the sum)
#[inline]
fn tp(s: P3, r: &[[f32; 3]; 3], t: P3) -> P3 {
    [
        t[0] + s[0] * r[0][0] + s[1] * r[1][0] + s[2] * r[2][0],
        t[1] + s[0] * r[0][1] + s[1] * r[1][1] + s[2] * r[2][1],
        t[2] + s[0] * r[0][2] + s[1] * r[1][2] + s[2] * r[2][2],
    ]
}
#[inline]
fn fabs(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fff_ffff)
}

/// AABBTreeCollider on two no-leaf trees, as ODE's trimesh-trimesh sets it
/// up: no first-contact exit, no temporal coherence, full box-box and
/// prim-box tests. Collects every overlapping triangle pair.
pub struct TreeCollider<'a> {
    mesh0: &'a MeshData,
    mesh1: &'a MeshData,
    r0to1: [[f32; 3]; 3],
    r1to0: [[f32; 3]; 3],
    t0to1: P3,
    t1to0: P3,
    ar: [[f32; 3]; 3],
    leaf: [P3; 3],
    leaf_index: u32,
    pub pairs: Vec<(u32, u32)>,
}

impl<'a> TreeCollider<'a> {
    /// AABBTreeCollider::Collide(BVTCache&, world0, world1)
    pub fn collide(mesh0: &'a MeshData, model0: &Model, world0: &M44, mesh1: &'a MeshData, model1: &Model, world1: &M44) -> Vec<(u32, u32)> {
        use super::volume::mul44;
        // InitQuery
        let inv0 = invert_pr(world0);
        let inv1 = invert_pr(world1);
        let w01 = mul44(world0, &inv1);
        let w10 = mul44(world1, &inv0);
        let m3 = |m: &M44| [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]];
        let r1to0 = m3(&w10);
        let mut ar = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                ar[i][j] = 1e-6f32 + r1to0[i][j].abs();
            }
        }
        let mut c = TreeCollider {
            mesh0,
            mesh1,
            r0to1: m3(&w01),
            r1to0,
            t0to1: [w01[3][0], w01[3][1], w01[3][2]],
            t1to0: [w10[3][0], w10[3][1], w10[3][2]],
            ar,
            leaf: [[0.0; 3]; 3],
            leaf_index: 0,
            pairs: Vec::new(),
        };
        if model0.nodes.is_empty() || model1.nodes.is_empty() {
            return c.pairs;
        }
        c.nodes(&model0.nodes, 0, &model1.nodes, 0);
        c.pairs
    }

    #[allow(clippy::needless_range_loop)]
    fn box_box(&self, ea: P3, ca: P3, eb: P3, cb: P3) -> bool {
        let r = &self.r1to0;
        let ar = &self.ar;
        let tx = (r[0][0] * cb[0] + r[1][0] * cb[1] + r[2][0] * cb[2]) + self.t1to0[0] - ca[0];
        let t = ea[0] + eb[0] * ar[0][0] + eb[1] * ar[1][0] + eb[2] * ar[2][0];
        if fabs(tx) > t {
            return false;
        }
        let ty = (r[0][1] * cb[0] + r[1][1] * cb[1] + r[2][1] * cb[2]) + self.t1to0[1] - ca[1];
        let t = ea[1] + eb[0] * ar[0][1] + eb[1] * ar[1][1] + eb[2] * ar[2][1];
        if fabs(ty) > t {
            return false;
        }
        let tz = (r[0][2] * cb[0] + r[1][2] * cb[1] + r[2][2] * cb[2]) + self.t1to0[2] - ca[2];
        let t = ea[2] + eb[0] * ar[0][2] + eb[1] * ar[1][2] + eb[2] * ar[2][2];
        if fabs(tz) > t {
            return false;
        }
        for i in 0..3 {
            let t = tx * r[i][0] + ty * r[i][1] + tz * r[i][2];
            let t2 = ea[0] * ar[i][0] + ea[1] * ar[i][1] + ea[2] * ar[i][2] + eb[i];
            if fabs(t) > t2 {
                return false;
            }
        }
        // mFullBoxBoxTest
        let tests: [(f32, f32); 9] = [
            (tz * r[0][1] - ty * r[0][2], ea[1] * ar[0][2] + ea[2] * ar[0][1] + eb[1] * ar[2][0] + eb[2] * ar[1][0]),
            (tz * r[1][1] - ty * r[1][2], ea[1] * ar[1][2] + ea[2] * ar[1][1] + eb[0] * ar[2][0] + eb[2] * ar[0][0]),
            (tz * r[2][1] - ty * r[2][2], ea[1] * ar[2][2] + ea[2] * ar[2][1] + eb[0] * ar[1][0] + eb[1] * ar[0][0]),
            (tx * r[0][2] - tz * r[0][0], ea[0] * ar[0][2] + ea[2] * ar[0][0] + eb[1] * ar[2][1] + eb[2] * ar[1][1]),
            (tx * r[1][2] - tz * r[1][0], ea[0] * ar[1][2] + ea[2] * ar[1][0] + eb[0] * ar[2][1] + eb[2] * ar[0][1]),
            (tx * r[2][2] - tz * r[2][0], ea[0] * ar[2][2] + ea[2] * ar[2][0] + eb[0] * ar[1][1] + eb[1] * ar[0][1]),
            (ty * r[0][0] - tx * r[0][1], ea[0] * ar[0][1] + ea[1] * ar[0][0] + eb[1] * ar[2][2] + eb[2] * ar[1][2]),
            (ty * r[1][0] - tx * r[1][1], ea[0] * ar[1][1] + ea[1] * ar[1][0] + eb[0] * ar[2][2] + eb[2] * ar[0][2]),
            (ty * r[2][0] - tx * r[2][1], ea[0] * ar[2][1] + ea[1] * ar[2][0] + eb[0] * ar[1][2] + eb[1] * ar[0][2]),
        ];
        for (t, t2) in tests {
            if fabs(t) > t2 {
                return false;
            }
        }
        true
    }

    fn tri_box(&self, center: P3, extents: P3) -> bool {
        let v = |k: usize| sub(self.leaf[k], center);
        super::volume::tri_box_overlap(v(0), v(1), v(2), extents)
    }

    fn leaf_from0(&mut self, id: u32) {
        self.leaf_index = id;
        let t = self.mesh0.tri(id as usize);
        for k in 0..3 {
            self.leaf[k] = tp(t[k], &self.r0to1, self.t0to1);
        }
    }
    fn leaf_from1(&mut self, id: u32) {
        self.leaf_index = id;
        let t = self.mesh1.tri(id as usize);
        for k in 0..3 {
            self.leaf[k] = tp(t[k], &self.r1to0, self.t1to0);
        }
    }
    /// PrimTestTriIndex
    fn prim_tri_index(&mut self, id1: u32) {
        let u = self.mesh1.tri(id1 as usize);
        if tri_tri_overlap(self.leaf[0], self.leaf[1], self.leaf[2], u[0], u[1], u[2]) {
            self.pairs.push((self.leaf_index, id1));
        }
    }
    /// PrimTestIndexTri
    fn prim_index_tri(&mut self, id0: u32) {
        let u = self.mesh0.tri(id0 as usize);
        if tri_tri_overlap(self.leaf[0], self.leaf[1], self.leaf[2], u[0], u[1], u[2]) {
            self.pairs.push((id0, self.leaf_index));
        }
    }
    /// _CollideTriBox (tree 1 against the leaf triangle of tree 0)
    fn tri_box_node(&mut self, nodes: &[NoLeafNode], b: usize) {
        let n = nodes[b];
        if !self.tri_box(n.center, n.extents) {
            return;
        }
        match n.pos {
            Child::Prim(p) => self.prim_tri_index(p),
            Child::Node(c) => self.tri_box_node(nodes, c as usize),
        }
        match n.neg {
            Child::Prim(p) => self.prim_tri_index(p),
            Child::Node(c) => self.tri_box_node(nodes, c as usize),
        }
    }
    /// _CollideBoxTri (tree 0 against the leaf triangle of tree 1)
    fn box_tri_node(&mut self, nodes: &[NoLeafNode], b: usize) {
        let n = nodes[b];
        if !self.tri_box(n.center, n.extents) {
            return;
        }
        match n.pos {
            Child::Prim(p) => self.prim_index_tri(p),
            Child::Node(c) => self.box_tri_node(nodes, c as usize),
        }
        match n.neg {
            Child::Prim(p) => self.prim_index_tri(p),
            Child::Node(c) => self.box_tri_node(nodes, c as usize),
        }
    }
    /// one side of tree 1 (`bc`) against a leaf of tree 0 already loaded
    fn b_side_vs_leaf(&mut self, n1: &[NoLeafNode], bc: Child) {
        match bc {
            Child::Prim(p) => self.prim_tri_index(p),
            Child::Node(c) => self.tri_box_node(n1, c as usize),
        }
    }
    /// child `ac` of tree 0 (a node) against one side `bc` of tree 1
    fn a_node_vs_b_side(&mut self, n0: &[NoLeafNode], ac: u32, n1: &[NoLeafNode], bc: Child) {
        match bc {
            Child::Prim(p) => {
                self.leaf_from1(p);
                self.box_tri_node(n0, ac as usize);
            }
            Child::Node(c) => self.nodes(n0, ac as usize, n1, c as usize),
        }
    }
    /// _Collide(AABBNoLeafNode*, AABBNoLeafNode*)
    fn nodes(&mut self, n0: &[NoLeafNode], a: usize, n1: &[NoLeafNode], b: usize) {
        let (na, nb) = (n0[a], n1[b]);
        if !self.box_box(na.extents, na.center, nb.extents, nb.center) {
            return;
        }
        for ac in [na.pos, na.neg] {
            match ac {
                Child::Prim(p) => {
                    self.leaf_from0(p);
                    self.b_side_vs_leaf(n1, nb.pos);
                    self.b_side_vs_leaf(n1, nb.neg);
                }
                Child::Node(c) => {
                    self.a_node_vs_b_side(n0, c, n1, nb.pos);
                    self.a_node_vs_b_side(n0, c, n1, nb.neg);
                }
            }
        }
    }
}

/// CoplanarTriTri
fn coplanar_tri_tri(n: P3, v0: P3, v1: P3, v2: P3, u0: P3, u1: P3, u2: P3) -> bool {
    let a = [fabs(n[0]), fabs(n[1]), fabs(n[2])];
    let (i0, i1) = if a[0] > a[1] {
        if a[0] > a[2] {
            (1, 2)
        } else {
            (0, 1)
        }
    } else if a[2] > a[1] {
        (0, 1)
    } else {
        (0, 2)
    };
    // EDGE_AGAINST_TRI_EDGES
    let edge_against = |va: P3, vb: P3| -> bool {
        let ax = vb[i0] - va[i0];
        let ay = vb[i1] - va[i1];
        let edge_edge = |ua: P3, ub: P3| -> bool {
            let bx = ua[i0] - ub[i0];
            let by = ua[i1] - ub[i1];
            let cx = va[i0] - ua[i0];
            let cy = va[i1] - ua[i1];
            let f = ay * bx - ax * by;
            let d = by * cx - bx * cy;
            if (f > 0.0 && d >= 0.0 && d <= f) || (f < 0.0 && d <= 0.0 && d >= f) {
                let e = ax * cy - ay * cx;
                if f > 0.0 {
                    if e >= 0.0 && e <= f {
                        return true;
                    }
                } else if e <= 0.0 && e >= f {
                    return true;
                }
            }
            false
        };
        edge_edge(u0, u1) || edge_edge(u1, u2) || edge_edge(u2, u0)
    };
    if edge_against(v0, v1) || edge_against(v1, v2) || edge_against(v2, v0) {
        return true;
    }
    // POINT_IN_TRI
    let point_in = |p: P3, t0: P3, t1: P3, t2: P3| -> bool {
        let mut a = t1[i1] - t0[i1];
        let mut b = -(t1[i0] - t0[i0]);
        let mut c = -a * t0[i0] - b * t0[i1];
        let d0 = a * p[i0] + b * p[i1] + c;
        a = t2[i1] - t1[i1];
        b = -(t2[i0] - t1[i0]);
        c = -a * t1[i0] - b * t1[i1];
        let d1 = a * p[i0] + b * p[i1] + c;
        a = t0[i1] - t2[i1];
        b = -(t0[i0] - t2[i0]);
        c = -a * t2[i0] - b * t2[i1];
        let d2 = a * p[i0] + b * p[i1] + c;
        d0 * d1 > 0.0 && d0 * d2 > 0.0
    };
    point_in(v0, u0, u1, u2) || point_in(u0, v0, v1, v2)
}

/// NEWCOMPUTE_INTERVALS: Some((a, b, c, x0, x1)), or None for coplanar
#[allow(clippy::too_many_arguments)]
fn intervals(vv0: f32, vv1: f32, vv2: f32, d0: f32, d1: f32, d2: f32, d0d1: f32, d0d2: f32) -> Option<(f32, f32, f32, f32, f32)> {
    if d0d1 > 0.0 {
        Some((vv2, (vv0 - vv2) * d2, (vv1 - vv2) * d2, d2 - d0, d2 - d1))
    } else if d0d2 > 0.0 {
        Some((vv1, (vv0 - vv1) * d1, (vv2 - vv1) * d1, d1 - d0, d1 - d2))
    } else if d1 * d2 > 0.0 || d0 != 0.0 {
        Some((vv0, (vv1 - vv0) * d0, (vv2 - vv0) * d0, d0 - d1, d0 - d2))
    } else if d1 != 0.0 {
        Some((vv1, (vv0 - vv1) * d1, (vv2 - vv1) * d1, d1 - d0, d1 - d2))
    } else if d2 != 0.0 {
        Some((vv2, (vv0 - vv2) * d2, (vv1 - vv2) * d2, d2 - d0, d2 - d1))
    } else {
        None
    }
}

/// AABBTreeCollider::TriTriOverlap (Moller, with OPCODE's epsilon test)
pub fn tri_tri_overlap(v0: P3, v1: P3, v2: P3, u0: P3, u1: P3, u2: P3) -> bool {
    let neg = |a: P3| [-a[0], -a[1], -a[2]];
    let n1 = cross(sub(v1, v0), sub(v2, v0));
    let d1 = dot(neg(n1), v0);
    let mut du0 = dot(n1, u0) + d1;
    let mut du1 = dot(n1, u1) + d1;
    let mut du2 = dot(n1, u2) + d1;
    let (absd1, sq1) = (fabs(d1), sqmag(n1));
    if absd1 >= sq1 {
        if fabs(du0) <= 0.000001 * absd1 {
            du0 = 0.0;
        }
        if fabs(du1) <= 0.000001 * absd1 {
            du1 = 0.0;
        }
        if fabs(du2) <= 0.000001 * absd1 {
            du2 = 0.0;
        }
    } else {
        if fabs(du0) <= 0.000001 * fmax(absd1, fmin(sq1, sqmag(u0))) {
            du0 = 0.0;
        }
        if fabs(du1) <= 0.000001 * fmax(absd1, fmin(sq1, sqmag(u1))) {
            du1 = 0.0;
        }
        if fabs(du2) <= 0.000001 * fmax(absd1, fmin(sq1, sqmag(u2))) {
            du2 = 0.0;
        }
    }
    let du0du1 = du0 * du1;
    let du0du2 = du0 * du2;
    if du0du1 > 0.0 && du0du2 > 0.0 {
        return false;
    }
    let n2 = cross(sub(u1, u0), sub(u2, u0));
    let d2 = dot(neg(n2), u0);
    let mut dv0 = dot(n2, v0) + d2;
    let mut dv1 = dot(n2, v1) + d2;
    let mut dv2 = dot(n2, v2) + d2;
    let (absd2, sq2) = (fabs(d2), sqmag(n2));
    if absd2 >= sq2 {
        if fabs(dv0) <= 0.000001 * absd2 {
            dv0 = 0.0;
        }
        if fabs(dv1) <= 0.000001 * absd2 {
            dv1 = 0.0;
        }
        if fabs(dv2) <= 0.000001 * absd2 {
            dv2 = 0.0;
        }
    } else {
        if fabs(dv0) <= 0.000001 * fmax(absd2, fmin(sq2, sqmag(v0))) {
            dv0 = 0.0;
        }
        if fabs(dv1) <= 0.000001 * fmax(absd2, fmin(sq2, sqmag(v1))) {
            dv1 = 0.0;
        }
        if fabs(dv2) <= 0.000001 * fmax(absd2, fmin(sq2, sqmag(v2))) {
            dv2 = 0.0;
        }
    }
    let dv0dv1 = dv0 * dv1;
    let dv0dv2 = dv0 * dv2;
    if dv0dv1 > 0.0 && dv0dv2 > 0.0 {
        return false;
    }
    let dd = cross(n1, n2);
    let mut max = dd[0].abs();
    let mut index = 0;
    let bb = dd[1].abs();
    let cc = dd[2].abs();
    if bb > max {
        max = bb;
        index = 1;
    }
    if cc > max {
        index = 2;
    }
    let (vp0, vp1, vp2) = (v0[index], v1[index], v2[index]);
    let (up0, up1, up2) = (u0[index], u1[index], u2[index]);
    let Some((a, b, c, x0, x1)) = intervals(vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2) else {
        return coplanar_tri_tri(n1, v0, v1, v2, u0, u1, u2);
    };
    let Some((d, e, f, y0, y1)) = intervals(up0, up1, up2, du0, du1, du2, du0du1, du0du2) else {
        return coplanar_tri_tri(n1, v0, v1, v2, u0, u1, u2);
    };
    let xx = x0 * x1;
    let yy = y0 * y1;
    let xxyy = xx * yy;
    let mut tmp = a * xxyy;
    let mut i1 = [tmp + b * x1 * yy, tmp + c * x0 * yy];
    tmp = d * xxyy;
    let mut i2 = [tmp + e * xx * y1, tmp + f * xx * y0];
    if i1[0] > i1[1] {
        i1.swap(0, 1);
    }
    if i2[0] > i2[1] {
        i2.swap(0, 1);
    }
    !(i1[1] < i2[0] || i2[1] < i1[0])
}
