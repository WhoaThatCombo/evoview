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
