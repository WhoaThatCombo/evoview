//! ODE's collision layer (collision_kernel.cpp, collision_space.cpp,
//! collision_util.cpp, sphere/box/capsule/plane/ray.cpp): geoms, simple
//! and hash spaces, and the primitive colliders.
//!
//! Geoms live on the World next to the bodies. A space keeps its geoms in
//! ODE's list order, head first; a geom that moves goes back to the head
//! (dxSpace::dirty), which is what decides the order pairs come out in.

use super::joints::ContactGeom;
use super::math::*;
use super::opcode;
use super::world::{BodyId, World};
use std::sync::Arc;

pub type GeomId = usize;

pub const GEOM_DIRTY: u32 = 1;
pub const GEOM_POSR_BAD: u32 = 2;
pub const GEOM_AABB_BAD: u32 = 4;
pub const GEOM_PLACEABLE: u32 = 8;
pub const GEOM_ENABLED: u32 = 16;
pub const GEOM_ZERO_SIZED: u32 = 32;
pub const RAY_FIRSTCONTACT: u32 = 0x10000;
pub const RAY_BACKFACECULL: u32 = 0x20000;
pub const RAY_CLOSEST_HIT: u32 = 0x40000;

/// the low 16 bits of dCollide's flags: how many contacts may be written
pub const NUMC_MASK: i32 = 0xffff;
pub const CONTACTS_UNIMPORTANT: i32 = 0x80000000u32 as i32;

// dGeomClass numbers (the collider table is indexed by them)
pub const SPHERE: usize = 0;
pub const BOX: usize = 1;
pub const CAPSULE: usize = 2;
pub const CYLINDER: usize = 3;
pub const PLANE: usize = 4;
pub const RAY: usize = 5;
pub const TRIMESH: usize = 8;
pub const SIMPLE_SPACE: usize = 10;
pub const HASH_SPACE: usize = 11;
pub const SAP_SPACE: usize = 12;
pub const QUADTREE_SPACE: usize = 13;
pub const CONVEX: usize = 6;
pub const TRANSFORM: usize = 7;
pub const HEIGHTFIELD: usize = 9;
const NUM_CLASSES: usize = 14;

#[derive(Clone, Copy, Debug)]
pub struct PosR {
    pub pos: V3,
    pub r: M3,
}
impl Default for PosR {
    fn default() -> Self {
        PosR { pos: [0.0; 4], r: IDENTITY_R }
    }
}

#[derive(Clone, Debug)]
pub enum SpaceKind {
    Simple,
    Hash { min_level: i32, max_level: i32 },
    QuadTree { blocks: Vec<QtBlock>, dirty: Vec<GeomId> },
    Sap(SapData),
}

/// collision_quadtreespace.cpp Block (x and the AABB's second axis)
#[derive(Clone, Debug)]
pub struct QtBlock {
    min_x: Real,
    max_x: Real,
    min_z: Real,
    max_z: Real,
    /// member geoms, ODE's list head LAST
    list: Vec<GeomId>,
    count: i32,
    parent: Option<usize>,
    children: Option<usize>,
}

/// collision_sapspace.cpp dxSAPSpace state
#[derive(Clone, Debug)]
pub struct SapData {
    dirty: Vec<GeomId>,
    geoms: Vec<GeomId>,
    ax: [usize; 3],
    sort: RadixSort,
}

/// RaixSortContext: ranks kept from the previous sort (temporal coherence)
#[derive(Clone, Debug, Default)]
pub struct RadixSort {
    size: usize,
    utilization: usize,
    valid: bool,
    buf: Vec<u32>,
    primary: usize,
}
#[derive(Clone, Debug)]
pub struct Space {
    pub kind: SpaceKind,
    /// member geoms, head (ODE's `first`) at index 0
    pub list: Vec<GeomId>,
    pub sublevel: i32,
}

#[derive(Clone, Debug)]
pub enum Shape {
    Sphere { radius: Real },
    Box { side: V3 },
    Capsule { radius: Real, lz: Real },
    Plane { p: [Real; 4] },
    Ray { length: Real },
    TriMesh(Arc<TriMeshData>),
    Cylinder { radius: Real, lz: Real },
    Convex(Arc<ConvexData>),
    Heightfield(Arc<HeightfieldData>),
    /// dxGeomTransform: `obj` placed by this geom; `tx` is the cached final
    /// placement (computeFinalTx)
    Transform { obj: Option<GeomId>, cleanup: bool, info: bool, tx: PosR },
    Space(Space),
}

/// dxTriMeshData: the mesh, its OPCODE tree and its model-space AABB
pub struct TriMeshData {
    pub mesh: opcode::MeshData,
    pub model: opcode::Model,
    pub aabb_center: V3,
    pub aabb_extents: V3,
}
impl std::fmt::Debug for TriMeshData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TriMeshData({} tris)", self.mesh.tris.len())
    }
}
impl TriMeshData {
    /// dGeomTriMeshDataBuild{Single,Double}: float data passes through
    /// exactly, double data is kept for the AABB and rounded for OPCODE
    pub fn build(verts: Vec<[f64; 3]>, tris: Vec<[u32; 3]>) -> Arc<TriMeshData> {
        let mesh = opcode::MeshData { verts, tris };
        let model = opcode::Model::build(&mesh);
        let mut mx = [-INFINITY; 3];
        let mut mn = [INFINITY; 3];
        for v in &mesh.verts {
            for a in 0..3 {
                if v[a] > mx[a] {
                    mx[a] = v[a];
                }
                if v[a] < mn[a] {
                    mn[a] = v[a];
                }
            }
        }
        let mut c = add3(&mn, &mx);
        c[0] *= 0.5;
        c[1] *= 0.5;
        c[2] *= 0.5;
        let e = sub3(&mx, &c);
        Arc::new(TriMeshData { mesh, model, aabb_center: c, aabb_extents: e })
    }
}
impl Shape {
    fn class(&self) -> usize {
        match self {
            Shape::Sphere { .. } => SPHERE,
            Shape::Box { .. } => BOX,
            Shape::Capsule { .. } => CAPSULE,
            Shape::Plane { .. } => PLANE,
            Shape::Ray { .. } => RAY,
            Shape::TriMesh(_) => TRIMESH,
            Shape::Cylinder { .. } => CYLINDER,
            Shape::Convex(_) => CONVEX,
            Shape::Heightfield(_) => HEIGHTFIELD,
            Shape::Transform { .. } => TRANSFORM,
            Shape::Space(s) => match s.kind {
                SpaceKind::Simple => SIMPLE_SPACE,
                SpaceKind::Hash { .. } => HASH_SPACE,
                SpaceKind::QuadTree { .. } => QUADTREE_SPACE,
                SpaceKind::Sap(_) => SAP_SPACE,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct Geom {
    pub shape: Shape,
    pub gflags: u32,
    pub body: Option<BodyId>,
    /// own position (unattached geoms, and attached geoms with an offset)
    posr: PosR,
    offset: Option<PosR>,
    pub parent: Option<GeomId>,
    pub aabb: [Real; 6],
    pub category_bits: u64,
    pub collide_bits: u64,
    alive: bool,
    /// next_ex / tome_ex: quadtree block, or SAP (dirty index, geom index)
    ex: (i32, i32),
}

fn is_space(g: &Geom) -> bool {
    matches!(g.shape, Shape::Space(_))
}
fn enabled(g: &Geom) -> bool {
    g.gflags & (GEOM_ENABLED | GEOM_ZERO_SIZED) == GEOM_ENABLED
}

impl World {
    fn geom_new(&mut self, shape: Shape, space: Option<GeomId>, placeable: bool) -> GeomId {
        let mut gflags = GEOM_DIRTY | GEOM_AABB_BAD | GEOM_ENABLED;
        if placeable {
            gflags |= GEOM_PLACEABLE;
        }
        let g = Geom {
            shape,
            gflags,
            body: None,
            posr: PosR::default(),
            offset: None,
            parent: None,
            aabb: [0.0; 6],
            category_bits: !0,
            collide_bits: !0,
            alive: true,
            ex: (-1, -1),
        };
        self.geoms.push(g);
        let id = self.geoms.len() - 1;
        if let Some(s) = space {
            self.space_add(s, id);
        }
        id
    }
    fn zero_sized(&mut self, g: GeomId, z: bool) {
        let f = &mut self.geoms[g].gflags;
        *f = if z { *f | GEOM_ZERO_SIZED } else { *f & !GEOM_ZERO_SIZED };
    }
    /// dCreateSphere
    pub fn create_sphere(&mut self, space: Option<GeomId>, radius: Real) -> GeomId {
        let g = self.geom_new(Shape::Sphere { radius }, space, true);
        self.zero_sized(g, radius == 0.0);
        g
    }
    /// dCreateBox
    pub fn create_box(&mut self, space: Option<GeomId>, lx: Real, ly: Real, lz: Real) -> GeomId {
        let g = self.geom_new(Shape::Box { side: [lx, ly, lz, 0.0] }, space, true);
        self.zero_sized(g, lx == 0.0 || ly == 0.0 || lz == 0.0);
        g
    }
    /// dCreateCapsule (axis along the geom's local z)
    pub fn create_capsule(&mut self, space: Option<GeomId>, radius: Real, length: Real) -> GeomId {
        let g = self.geom_new(Shape::Capsule { radius, lz: length }, space, true);
        self.zero_sized(g, radius == 0.0);
        g
    }
    /// dCreatePlane: a*x + b*y + c*z = d
    pub fn create_plane(&mut self, space: Option<GeomId>, a: Real, b: Real, c: Real, d: Real) -> GeomId {
        let mut p = [a, b, c, d];
        plane_unit_normal(&mut p);
        self.geom_new(Shape::Plane { p }, space, false)
    }
    /// dCreateRay (along the geom's local z)
    pub fn create_ray(&mut self, space: Option<GeomId>, length: Real) -> GeomId {
        self.geom_new(Shape::Ray { length }, space, true)
    }
    /// dCreateTriMesh
    pub fn create_trimesh(&mut self, space: Option<GeomId>, data: Arc<TriMeshData>) -> GeomId {
        self.geom_new(Shape::TriMesh(data), space, true)
    }
    /// dGeomPlaneSetParams
    pub fn geom_plane_set_params(&mut self, g: GeomId, a: Real, b: Real, c: Real, d: Real) {
        if let Shape::Plane { p } = &mut self.geoms[g].shape {
            *p = [a, b, c, d];
            plane_unit_normal(p);
        }
        self.geom_moved(g);
    }
    /// dGeomRaySetParams / dGeomRaySetClosestHit
    pub fn geom_ray_set_params(&mut self, g: GeomId, first_contact: bool, backface_cull: bool, closest_hit: bool) {
        let f = &mut self.geoms[g].gflags;
        *f &= !(RAY_FIRSTCONTACT | RAY_BACKFACECULL | RAY_CLOSEST_HIT);
        if first_contact {
            *f |= RAY_FIRSTCONTACT;
        }
        if backface_cull {
            *f |= RAY_BACKFACECULL;
        }
        if closest_hit {
            *f |= RAY_CLOSEST_HIT;
        }
    }
    /// dCreateCylinder (axis along the geom's local z)
    pub fn create_cylinder(&mut self, space: Option<GeomId>, radius: Real, length: Real) -> GeomId {
        let g = self.geom_new(Shape::Cylinder { radius, lz: length }, space, true);
        self.zero_sized(g, radius == 0.0 || length == 0.0);
        g
    }
    /// dCreateConvex
    pub fn create_convex(&mut self, space: Option<GeomId>, data: Arc<ConvexData>) -> GeomId {
        self.geom_new(Shape::Convex(data), space, true)
    }
    /// dCreateHeightfield
    pub fn create_heightfield(&mut self, space: Option<GeomId>, data: Arc<HeightfieldData>, placeable: bool) -> GeomId {
        self.geom_new(Shape::Heightfield(data), space, placeable)
    }
    /// dCreateGeomTransform
    pub fn create_geom_transform(&mut self, space: Option<GeomId>) -> GeomId {
        self.geom_new(Shape::Transform { obj: None, cleanup: false, info: false, tx: PosR::default() }, space, true)
    }
    /// dGeomTransformSetGeom / SetInfo
    pub fn geom_transform_set(&mut self, g: GeomId, obj: Option<GeomId>, info: bool) {
        if let Shape::Transform { obj: o, info: i, .. } = &mut self.geoms[g].shape {
            *o = obj;
            *i = info;
        }
    }
    /// dQuadTreeSpaceCreate
    pub fn quadtree_space_create(&mut self, space: Option<GeomId>, center: [Real; 3], extents: [Real; 3], depth: i32) -> GeomId {
        let n = ((1usize << (2 * (depth + 1))) - 1) / 3;
        let mut blocks = vec![QtBlock { min_x: 0.0, max_x: 0.0, min_z: 0.0, max_z: 0.0, list: Vec::new(), count: 0, parent: None, children: None }; n];
        let mut cursor = 1usize;
        qt_create(&mut blocks, 0, center[0] - extents[0], (center[0] + extents[0]).next_up(), center[1] - extents[1], (center[1] + extents[1]).next_up(), None, depth, &mut cursor);
        let g = self.geom_new(Shape::Space(Space { kind: SpaceKind::QuadTree { blocks, dirty: Vec::new() }, list: Vec::new(), sublevel: 0 }), space, false);
        self.geoms[g].aabb = [-INFINITY, INFINITY, -INFINITY, INFINITY, -INFINITY, INFINITY];
        g
    }
    /// dSweepAndPruneSpaceCreate
    pub fn sap_space_create(&mut self, space: Option<GeomId>, axisorder: i32) -> GeomId {
        let ax = [((axisorder & 3) << 1) as usize, (((axisorder >> 2) & 3) << 1) as usize, (((axisorder >> 4) & 3) << 1) as usize];
        let g = self.geom_new(Shape::Space(Space { kind: SpaceKind::Sap(SapData { dirty: Vec::new(), geoms: Vec::new(), ax, sort: RadixSort::default() }), list: Vec::new(), sublevel: 0 }), space, false);
        self.geoms[g].aabb = [-INFINITY, INFINITY, -INFINITY, INFINITY, -INFINITY, INFINITY];
        g
    }
    /// dSimpleSpaceCreate
    pub fn simple_space_create(&mut self, space: Option<GeomId>) -> GeomId {
        self.geom_new(Shape::Space(Space { kind: SpaceKind::Simple, list: Vec::new(), sublevel: 0 }), space, false)
    }
    /// dHashSpaceCreate
    pub fn hash_space_create(&mut self, space: Option<GeomId>) -> GeomId {
        self.geom_new(Shape::Space(Space { kind: SpaceKind::Hash { min_level: -3, max_level: 10 }, list: Vec::new(), sublevel: 0 }), space, false)
    }
    pub fn hash_space_set_levels(&mut self, s: GeomId, min: i32, max: i32) {
        if let Shape::Space(sp) = &mut self.geoms[s].shape {
            sp.kind = SpaceKind::Hash { min_level: min, max_level: max };
        }
    }
    /// dGeomDestroy
    pub fn geom_destroy(&mut self, g: GeomId) {
        if let Some(p) = self.geoms[g].parent {
            self.space_remove(p, g);
        }
        self.body_remove(g);
        self.geoms[g].alive = false;
    }

    fn space_mut(&mut self, s: GeomId) -> &mut Space {
        match &mut self.geoms[s].shape {
            Shape::Space(sp) => sp,
            _ => panic!("not a space"),
        }
    }
    fn space_ref(&self, s: GeomId) -> &Space {
        match &self.geoms[s].shape {
            Shape::Space(sp) => sp,
            _ => panic!("not a space"),
        }
    }
    /// dSpaceAdd
    pub fn space_add(&mut self, s: GeomId, g: GeomId) {
        let gaabb = self.geoms[g].aabb;
        let mut ex = self.geoms[g].ex;
        match &mut self.space_mut(s).kind {
            SpaceKind::QuadTree { blocks, dirty } => {
                dirty.push(g);
                let b = qt_get_block(blocks, 0, &gaabb);
                qt_add(blocks, b, g);
                ex.0 = b as i32;
            }
            SpaceKind::Sap(d) => {
                ex = (d.dirty.len() as i32, -1);
                d.dirty.push(g);
            }
            _ => {}
        }
        self.geoms[g].ex = ex;
        self.geoms[g].parent = Some(s);
        self.space_mut(s).list.insert(0, g);
        self.geom_moved(s);
    }
    /// dSpaceRemove
    pub fn space_remove(&mut self, s: GeomId, g: GeomId) {
        let ex = self.geoms[g].ex;
        let mut moved: Vec<(GeomId, (i32, i32))> = Vec::new();
        match &mut self.space_mut(s).kind {
            SpaceKind::QuadTree { blocks, dirty } => {
                qt_del(blocks, ex.0 as usize, g);
                dirty.retain(|&x| x != g);
            }
            SpaceKind::Sap(d) => {
                if ex.0 != -1 {
                    let last = d.dirty.len() - 1;
                    if ex.0 as usize != last {
                        let lg = d.dirty[last];
                        d.dirty[ex.0 as usize] = lg;
                        moved.push((lg, (ex.0, -2)));
                    }
                    d.dirty.pop();
                } else {
                    let last = d.geoms.len() - 1;
                    if ex.1 as usize != last {
                        let lg = d.geoms[last];
                        d.geoms[ex.1 as usize] = lg;
                        moved.push((lg, (-2, ex.1)));
                    }
                    d.geoms.pop();
                }
            }
            _ => {}
        }
        for (lg, (a, b)) in moved {
            if a != -2 {
                self.geoms[lg].ex.0 = a;
            }
            if b != -2 {
                self.geoms[lg].ex.1 = b;
            }
        }
        self.geoms[g].ex = (-1, -1);
        self.space_mut(s).list.retain(|&x| x != g);
        self.geoms[g].parent = None;
        self.geom_moved(s);
    }
    pub fn space_geoms(&self, s: GeomId) -> &[GeomId] {
        &self.space_ref(s).list
    }

    fn mark_aabb_bad(&mut self, g: GeomId) {
        self.geoms[g].gflags |= GEOM_DIRTY | GEOM_AABB_BAD;
    }
    /// dGeomMoved
    pub fn geom_moved(&mut self, mut g: GeomId) {
        if self.geoms[g].offset.is_some() {
            self.geoms[g].gflags |= GEOM_POSR_BAD;
        }
        let mut parent = self.geoms[g].parent;
        while let Some(p) = parent {
            if self.geoms[g].gflags & GEOM_DIRTY != 0 {
                break;
            }
            self.mark_aabb_bad(g);
            self.space_dirty(p, g);
            g = p;
            parent = self.geoms[p].parent;
        }
        let mut cur = Some(g);
        while let Some(c) = cur {
            self.mark_aabb_bad(c);
            cur = self.geoms[c].parent;
        }
    }
    /// dxSpace::dirty and its overrides
    fn space_dirty(&mut self, p: GeomId, g: GeomId) {
        let ex = self.geoms[g].ex;
        let mut set: Vec<(GeomId, i32, i32)> = Vec::new();
        let sp = self.space_mut(p);
        match &mut sp.kind {
            SpaceKind::QuadTree { dirty, .. } => dirty.push(g),
            SpaceKind::Sap(d) => {
                if ex.0 != -1 {
                    return;
                }
                let last = d.geoms.len() - 1;
                if ex.1 as usize != last {
                    let lg = d.geoms[last];
                    d.geoms[ex.1 as usize] = lg;
                    set.push((lg, -2, ex.1));
                }
                d.geoms.pop();
                set.push((g, d.dirty.len() as i32, -1));
                d.dirty.push(g);
            }
            _ => {
                // back to the head of the list
                let list = &mut sp.list;
                if let Some(k) = list.iter().position(|&x| x == g) {
                    list.remove(k);
                }
                list.insert(0, g);
            }
        }
        for (x, a, b) in set {
            if a != -2 {
                self.geoms[x].ex.0 = a;
            }
            self.geoms[x].ex.1 = b;
        }
    }
    pub(crate) fn body_geoms_moved(&mut self, b: BodyId) {
        for k in (0..self.bodies[b].geoms.len()).rev() {
            let g = self.bodies[b].geoms[k];
            self.geom_moved(g);
        }
    }

    // ---- placement

    fn body_remove(&mut self, g: GeomId) {
        if let Some(b) = self.geoms[g].body {
            let list = &mut self.bodies[b].geoms;
            if let Some(k) = list.iter().rposition(|&x| x == g) {
                list.remove(k);
            }
            self.geoms[g].body = None;
        }
    }
    /// dGeomSetBody
    pub fn geom_set_body(&mut self, g: GeomId, b: Option<BodyId>) {
        match b {
            Some(b) => {
                if self.geoms[g].body != Some(b) {
                    self.geoms[g].offset = None;
                    self.body_remove(g);
                    self.geoms[g].body = Some(b);
                    self.bodies[b].geoms.push(g);
                }
                self.geom_moved(g);
            }
            None => {
                if let Some(ob) = self.geoms[g].body {
                    if self.geoms[g].offset.is_some() {
                        self.recompute_posr(g);
                        self.geoms[g].offset = None;
                    } else {
                        let body = &self.bodies[ob];
                        self.geoms[g].posr = PosR { pos: body.pos, r: body.r };
                    }
                    self.body_remove(g);
                }
            }
        }
    }
    /// the geom's world position and rotation (final_posr)
    pub fn geom_posr(&self, g: GeomId) -> PosR {
        let gg = &self.geoms[g];
        match (gg.body, &gg.offset) {
            (Some(b), None) => PosR { pos: self.bodies[b].pos, r: self.bodies[b].r },
            _ => gg.posr,
        }
    }
    fn recompute_posr(&mut self, g: GeomId) {
        if self.geoms[g].gflags & GEOM_POSR_BAD != 0 {
            let (Some(b), Some(off)) = (self.geoms[g].body, self.geoms[g].offset) else { return };
            let body = &self.bodies[b];
            let mut pos = mul0_331(&body.r, &off.pos);
            pos[0] += body.pos[0];
            pos[1] += body.pos[1];
            pos[2] += body.pos[2];
            let r = mul0_333(&body.r, &off.r);
            let gg = &mut self.geoms[g];
            gg.posr.pos[0] = pos[0];
            gg.posr.pos[1] = pos[1];
            gg.posr.pos[2] = pos[2];
            gg.posr.r = r;
            gg.gflags &= !GEOM_POSR_BAD;
        }
    }
    /// dGeomSetPosition
    pub fn geom_set_position(&mut self, g: GeomId, x: Real, y: Real, z: Real) {
        let gg = &self.geoms[g];
        if let (Some(b), Some(off)) = (gg.body, gg.offset) {
            let wo = mul0_331(&self.bodies[b].r, &off.pos);
            self.body_set_position(b, x - wo[0], y - wo[1], z - wo[2]);
        } else if let Some(b) = gg.body {
            self.body_set_position(b, x, y, z);
        } else {
            let p = &mut self.geoms[g].posr.pos;
            p[0] = x;
            p[1] = y;
            p[2] = z;
            self.geom_moved(g);
        }
    }
    /// dGeomSetRotation
    pub fn geom_set_rotation(&mut self, g: GeomId, r: &M3) {
        if self.geoms[g].offset.is_some() && self.geoms[g].body.is_some() {
            self.recompute_posr(g);
            let off = self.geoms[g].offset.unwrap();
            let fin = PosR { pos: self.geoms[g].posr.pos, r: *r };
            let bp = body_posr(&off, &fin);
            let b = self.geoms[g].body.unwrap();
            self.body_set_rotation(b, &bp.r);
            self.body_set_position(b, bp.pos[0], bp.pos[1], bp.pos[2]);
        } else if let Some(b) = self.geoms[g].body {
            self.body_set_rotation(b, r);
        } else {
            self.geoms[g].posr.r = *r;
            self.geom_moved(g);
        }
    }
    fn create_offset(&mut self, g: GeomId) {
        if self.geoms[g].offset.is_none() {
            self.geoms[g].offset = Some(PosR::default());
            self.geoms[g].gflags |= GEOM_POSR_BAD;
        }
    }
    /// dGeomSetOffsetPosition (body frame)
    pub fn geom_set_offset_position(&mut self, g: GeomId, x: Real, y: Real, z: Real) {
        self.create_offset(g);
        let o = self.geoms[g].offset.as_mut().unwrap();
        o.pos[0] = x;
        o.pos[1] = y;
        o.pos[2] = z;
        self.geom_moved(g);
    }
    /// dGeomSetOffsetRotation (body frame)
    pub fn geom_set_offset_rotation(&mut self, g: GeomId, r: &M3) {
        self.create_offset(g);
        self.geoms[g].offset.as_mut().unwrap().r = *r;
        self.geom_moved(g);
    }
    pub fn geom_set_category_bits(&mut self, g: GeomId, bits: u64) {
        self.geoms[g].category_bits = bits;
    }
    pub fn geom_set_collide_bits(&mut self, g: GeomId, bits: u64) {
        self.geoms[g].collide_bits = bits;
    }
    pub fn geom_enable(&mut self, g: GeomId, on: bool) {
        if on {
            self.geoms[g].gflags |= GEOM_ENABLED;
        } else {
            self.geoms[g].gflags &= !GEOM_ENABLED;
        }
    }
    /// dGeomRaySet: start point and direction (normalised into R's z column)
    pub fn geom_ray_set(&mut self, g: GeomId, px: Real, py: Real, pz: Real, dx: Real, dy: Real, dz: Real) {
        self.recompute_posr(g);
        let mut n = v3(dx, dy, dz);
        normalize3(&mut n);
        // a ray on a body writes through the body's posr, as in ODE
        let (pos, r) = match (self.geoms[g].body, self.geoms[g].offset.is_some()) {
            (Some(b), false) => {
                let body = &mut self.bodies[b];
                (&mut body.pos, &mut body.r)
            }
            _ => {
                let gg = &mut self.geoms[g];
                (&mut gg.posr.pos, &mut gg.posr.r)
            }
        };
        pos[0] = px;
        pos[1] = py;
        pos[2] = pz;
        r[2] = n[0];
        r[6] = n[1];
        r[10] = n[2];
        self.geom_moved(g);
    }
    /// dGeomIsSpace
    pub fn geom_is_space(&self, g: GeomId) -> bool {
        is_space(&self.geoms[g])
    }
    /// dGeomGetClass
    pub fn geom_class(&self, g: GeomId) -> usize {
        self.geoms[g].shape.class()
    }
    pub fn geom_aabb(&mut self, g: GeomId) -> [Real; 6] {
        self.recompute_aabb(g);
        self.geoms[g].aabb
    }

    // ---- AABBs

    fn recompute_aabb(&mut self, g: GeomId) {
        if self.geoms[g].gflags & GEOM_AABB_BAD == 0 {
            return;
        }
        self.recompute_posr(g);
        let aabb = match &self.geoms[g].shape {
            Shape::Space(sp) => match sp.kind {
                // computeAABB does nothing for these: the box stays infinite
                SpaceKind::QuadTree { .. } | SpaceKind::Sap(_) => self.geoms[g].aabb,
                _ => {
                    let list = sp.list.clone();
                    if list.is_empty() {
                        [0.0; 6]
                    } else {
                        let mut a = [INFINITY, -INFINITY, INFINITY, -INFINITY, INFINITY, -INFINITY];
                        for c in list {
                            self.recompute_aabb(c);
                            let ga = self.geoms[c].aabb;
                            for i in (0..6).step_by(2) {
                                if ga[i] < a[i] {
                                    a[i] = ga[i];
                                }
                            }
                            for i in (1..6).step_by(2) {
                                if ga[i] > a[i] {
                                    a[i] = ga[i];
                                }
                            }
                        }
                        a
                    }
                }
            },
            Shape::Transform { obj, .. } => match *obj {
                None => [0.0; 6],
                Some(o) => {
                    let tx = self.transform_final_tx(g, o);
                    if let Shape::Transform { tx: t, .. } = &mut self.geoms[g].shape {
                        *t = tx;
                    }
                    geom_shape_aabb(&self.geoms[o].shape, self.geoms[o].gflags, &tx)
                }
            },
            shape => geom_shape_aabb(shape, self.geoms[g].gflags, &self.geom_posr(g)),
        };
        let gg = &mut self.geoms[g];
        gg.aabb = aabb;
        gg.gflags &= !GEOM_AABB_BAD;
    }
    /// computeFinalTx
    fn transform_final_tx(&self, g: GeomId, obj: GeomId) -> PosR {
        let me = self.geom_posr(g);
        let o = self.geom_posr(obj);
        let mut pos = mul0_331(&me.r, &o.pos);
        pos[0] += me.pos[0];
        pos[1] += me.pos[1];
        pos[2] += me.pos[2];
        PosR { pos, r: mul0_333(&me.r, &o.r) }
    }

    fn clean_geoms(&mut self, s: GeomId) {
        match &self.space_ref(s).kind {
            SpaceKind::QuadTree { dirty, .. } => {
                let d = dirty.clone();
                for g in d {
                    if is_space(&self.geoms[g]) {
                        self.clean_geoms(g);
                    }
                    self.recompute_aabb(g);
                    self.geoms[g].gflags &= !GEOM_DIRTY;
                    let (aabb, b0) = (self.geoms[g].aabb, self.geoms[g].ex.0 as usize);
                    let nb = if let SpaceKind::QuadTree { blocks, .. } = &mut self.space_mut(s).kind {
                        let nb = qt_get_block(blocks, b0, &aabb);
                        if nb != b0 {
                            qt_del(blocks, b0, g);
                            qt_add(blocks, nb, g);
                        }
                        nb
                    } else {
                        b0
                    };
                    self.geoms[g].ex.0 = nb as i32;
                }
                if let SpaceKind::QuadTree { dirty, .. } = &mut self.space_mut(s).kind {
                    dirty.clear();
                }
                return;
            }
            SpaceKind::Sap(d) => {
                let d = d.dirty.clone();
                if d.is_empty() {
                    return;
                }
                let base = if let SpaceKind::Sap(sd) = &self.space_ref(s).kind { sd.geoms.len() } else { 0 };
                for (i, &g) in d.iter().enumerate() {
                    if is_space(&self.geoms[g]) {
                        self.clean_geoms(g);
                    }
                    self.recompute_aabb(g);
                    self.geoms[g].gflags &= !GEOM_DIRTY;
                    self.geoms[g].ex = (-1, (base + i) as i32);
                }
                if let SpaceKind::Sap(sd) = &mut self.space_mut(s).kind {
                    sd.geoms.extend_from_slice(&d);
                    sd.dirty.clear();
                }
                return;
            }
            _ => {}
        }
        let list = self.space_ref(s).list.clone();
        for g in list {
            if self.geoms[g].gflags & GEOM_DIRTY == 0 {
                break;
            }
            if is_space(&self.geoms[g]) {
                self.clean_geoms(g);
            }
            self.recompute_aabb(g);
            self.geoms[g].gflags &= !GEOM_DIRTY;
        }
    }

    fn collide_aabbs(&self, g1: GeomId, g2: GeomId, out: &mut Vec<(GeomId, GeomId)>) {
        let (a, b) = (&self.geoms[g1], &self.geoms[g2]);
        if a.body == b.body && a.body.is_some() {
            return;
        }
        if (a.category_bits & b.collide_bits) == 0 && (b.category_bits & a.collide_bits) == 0 {
            return;
        }
        let (b1, b2) = (&a.aabb, &b.aabb);
        if b1[0] > b2[1] || b1[1] < b2[0] || b1[2] > b2[3] || b1[3] < b2[2] || b1[4] > b2[5] || b1[5] < b2[4] {
            return;
        }
        out.push((g1, g2));
    }

    /// dSpaceCollide: the candidate pairs, in the order ODE would call the
    /// near callback with them
    pub fn space_collide(&mut self, s: GeomId) -> Vec<(GeomId, GeomId)> {
        let mut out = Vec::new();
        self.space_collide_into(s, &mut out);
        out
    }
    fn space_collide_into(&mut self, s: GeomId, out: &mut Vec<(GeomId, GeomId)>) {
        match self.space_ref(s).kind {
            SpaceKind::Simple => {
                self.clean_geoms(s);
                let list = self.space_ref(s).list.clone();
                for (i, &g1) in list.iter().enumerate() {
                    if !enabled(&self.geoms[g1]) {
                        continue;
                    }
                    for &g2 in &list[i + 1..] {
                        if enabled(&self.geoms[g2]) {
                            self.collide_aabbs(g1, g2, out);
                        }
                    }
                }
            }
            SpaceKind::Hash { min_level, max_level } => self.hash_collide(s, min_level, max_level, out),
            SpaceKind::QuadTree { .. } => {
                self.clean_geoms(s);
                if let SpaceKind::QuadTree { blocks, .. } = &self.space_ref(s).kind {
                    let blocks = blocks.clone();
                    self.qt_collide(&blocks, 0, out);
                }
            }
            SpaceKind::Sap(_) => self.sap_collide(s, out),
        }
    }

    /// dSpaceCollide2: geom (or space) against geom (or space)
    pub fn space_collide2(&mut self, g1: GeomId, g2: GeomId) -> Vec<(GeomId, GeomId)> {
        let mut out = Vec::new();
        let mut s1 = is_space(&self.geoms[g1]).then_some(g1);
        let mut s2 = is_space(&self.geoms[g2]).then_some(g2);
        if let (Some(a), Some(b)) = (s1, s2) {
            let (l1, l2) = (self.space_ref(a).sublevel, self.space_ref(b).sublevel);
            if l1 != l2 {
                if l1 > l2 {
                    s2 = None;
                } else {
                    s1 = None;
                }
            }
        }
        match (s1, s2) {
            (Some(a), Some(b)) if a == b => self.space_collide_into(a, &mut out),
            (Some(a), Some(b)) => {
                if self.space_ref(a).list.len() < self.space_ref(b).list.len() {
                    for g in self.space_ref(a).list.clone() {
                        let mut tmp = Vec::new();
                        self.space_collide_geom(b, g, &mut tmp);
                        out.extend(tmp.into_iter().map(|(x, y)| (y, x)));
                    }
                } else {
                    for g in self.space_ref(b).list.clone() {
                        self.space_collide_geom(a, g, &mut out);
                    }
                }
            }
            (Some(a), None) => self.space_collide_geom(a, g2, &mut out),
            (None, Some(b)) => {
                let mut tmp = Vec::new();
                self.space_collide_geom(b, g1, &mut tmp);
                out.extend(tmp.into_iter().map(|(x, y)| (y, x)));
            }
            (None, None) => {
                self.recompute_aabb(g1);
                self.recompute_aabb(g2);
                self.collide_aabbs(g1, g2, &mut out);
            }
        }
        out
    }
    /// collide2 of either space kind: every member against one geom
    fn space_collide_geom(&mut self, s: GeomId, geom: GeomId, out: &mut Vec<(GeomId, GeomId)>) {
        self.clean_geoms(s);
        self.recompute_aabb(geom);
        if let SpaceKind::QuadTree { blocks, .. } = &self.space_ref(s).kind {
            let blocks = blocks.clone();
            let mut tmp = Vec::new();
            if self.geoms[geom].parent == Some(s) {
                let mut cb = self.geoms[geom].ex.0 as usize;
                let first: Vec<GeomId> = blocks[cb].list.iter().rev().copied().collect();
                self.qt_collide_pair(&blocks, cb, geom, &first, &mut tmp);
                out.extend(tmp.into_iter().map(|(a, b)| (b, a)));
                while let Some(p) = blocks[cb].parent {
                    cb = p;
                    for &g1 in blocks[cb].list.iter().rev() {
                        if enabled(&self.geoms[g1]) {
                            self.collide_aabbs(g1, geom, out);
                        }
                    }
                }
            } else {
                let first: Vec<GeomId> = blocks[0].list.iter().rev().copied().collect();
                self.qt_collide_pair(&blocks, 0, geom, &first, &mut tmp);
                out.extend(tmp.into_iter().map(|(a, b)| (b, a)));
            }
            return;
        }
        if let SpaceKind::Sap(d) = &self.space_ref(s).kind {
            for g in d.geoms.clone() {
                if enabled(&self.geoms[g]) {
                    self.collide_aabbs(g, geom, out);
                }
            }
            return;
        }
        for g in self.space_ref(s).list.clone() {
            if enabled(&self.geoms[g]) {
                self.collide_aabbs(g, geom, out);
            }
        }
    }

    /// dxHashSpace::collide
    fn hash_collide(&mut self, s: GeomId, global_min: i32, global_max: i32, out: &mut Vec<(GeomId, GeomId)>) {
        if self.space_ref(s).list.len() < 2 {
            return;
        }
        self.clean_geoms(s);
        struct HBox {
            level: i32,
            dbounds: [i32; 6],
            geom: GeomId,
        }
        let mut hash_boxes: Vec<HBox> = Vec::new();
        let mut big_boxes: Vec<GeomId> = Vec::new();
        let mut maxlevel = global_min - 1;
        for g in self.space_ref(s).list.clone() {
            if !enabled(&self.geoms[g]) {
                continue;
            }
            let aabb = self.geoms[g].aabb;
            let mut level = find_level(&aabb);
            if level < global_min {
                level = global_min;
            }
            if level <= global_max {
                if level > maxlevel {
                    maxlevel = level;
                }
                let cell_recip = recip(ldexp1(level));
                let mut db = [0i32; 6];
                for i in 0..6 {
                    db[i] = (aabb[i] * cell_recip).floor() as i32;
                }
                hash_boxes.push(HBox { level, dbounds: db, geom: g });
            } else {
                big_boxes.push(g);
            }
        }
        const PRIMES: [u32; 31] = [
            1, 2, 3, 7, 13, 31, 61, 127, 251, 509, 1021, 2039, 4093, 8191, 16381, 32749, 65521, 131071, 262139, 524287, 1048573, 2097143, 4194301,
            8388593, 16777213, 33554393, 67108859, 134217689, 268435399, 536870909, 1073741789,
        ];
        let n = hash_boxes.len();
        let rowsize = (n + 7) >> 3;
        let mut tested = vec![0u8; n * rowsize];
        let mut i = 0;
        while i < PRIMES.len() {
            if PRIMES[i] as usize >= 8 * n {
                break;
            }
            i += 1;
        }
        if i >= PRIMES.len() {
            i = PRIMES.len() - 1;
        }
        let sz = PRIMES[i];
        // nodes (x, y, z, box index); ODE prepends, so each chain is walked backwards
        let mut table: Vec<Vec<(i32, i32, i32, usize)>> = vec![Vec::new(); sz as usize];
        let vab = |level: i32, x: i32, y: i32| -> u32 { (level as u32).wrapping_mul(1000).wrapping_add((x as u32).wrapping_mul(100)).wrapping_add((y as u32).wrapping_mul(10)) };
        for (bi, hb) in hash_boxes.iter().enumerate() {
            let d = hb.dbounds;
            for xi in d[0]..=d[1] {
                for yi in d[2]..=d[3] {
                    let zbegin = d[4];
                    let mut hi = vab(hb.level, xi, yi).wrapping_add(zbegin as u32) % sz;
                    for zi in zbegin..=d[5] {
                        table[hi as usize].push((xi, yi, zi, bi));
                        hi = if hi + 1 != sz { hi + 1 } else { 0 };
                    }
                }
            }
        }
        for (ai, hb) in hash_boxes.iter().enumerate() {
            let mut db = hb.dbounds;
            let mut level = hb.level;
            loop {
                for xi in db[0]..=db[1] {
                    for yi in db[2]..=db[3] {
                        let zbegin = db[4];
                        let mut hi = vab(level, xi, yi).wrapping_add(zbegin as u32) % sz;
                        for zi in zbegin..=db[5] {
                            for &(nx, ny, nz, nb) in table[hi as usize].iter().rev() {
                                if nb == ai {
                                    continue;
                                }
                                if hash_boxes[nb].level == level && nx == xi && ny == yi && nz == zi {
                                    let (k, mask) = if ai <= nb { (ai * rowsize + (nb >> 3), 1u8 << (nb & 7)) } else { (nb * rowsize + (ai >> 3), 1u8 << (ai & 7)) };
                                    if tested[k] & mask == 0 {
                                        tested[k] |= mask;
                                        self.collide_aabbs(hb.geom, hash_boxes[nb].geom, out);
                                    }
                                }
                            }
                            hi = if hi + 1 != sz { hi + 1 } else { 0 };
                        }
                    }
                }
                if level == maxlevel {
                    break;
                }
                level += 1;
                for v in db.iter_mut() {
                    *v >>= 1;
                }
            }
        }
        for hb in &hash_boxes {
            for &bg in &big_boxes {
                self.collide_aabbs(hb.geom, bg, out);
            }
        }
        for (k, &a) in big_boxes.iter().enumerate() {
            for &b in &big_boxes[k + 1..] {
                self.collide_aabbs(a, b, out);
            }
        }
    }

    // ---- dCollide

    /// dCollide: contacts between two geoms (at most `flags & NUMC_MASK`)
    pub fn collide(&mut self, o1: GeomId, o2: GeomId, flags: i32) -> Vec<ContactGeom> {
        let mut out = Vec::new();
        if flags & NUMC_MASK == 0 || o1 == o2 {
            return out;
        }
        if self.geoms[o1].body == self.geoms[o2].body && self.geoms[o1].body.is_some() {
            return out;
        }
        self.recompute_posr(o1);
        self.recompute_posr(o2);
        let (c1, c2) = (self.geoms[o1].shape.class(), self.geoms[o2].shape.class());
        if c1 >= SIMPLE_SPACE || c2 >= SIMPLE_SPACE {
            // dCollideSpaceGeom: every overlapping pair below, up to the limit
            let pairs = self.space_collide2(o1, o2);
            let mut left = flags;
            for (a, b) in pairs {
                if left & NUMC_MASK == 0 {
                    break;
                }
                let c = self.collide(a, b, left);
                left -= c.len() as i32;
                out.extend(c);
            }
            return out;
        }
        // a transform's cached placement is refreshed when its AABB is stale
        for g in [o1, o2] {
            if let Shape::Transform { obj: Some(obj), .. } = self.geoms[g].shape {
                if self.geoms[g].gflags & GEOM_AABB_BAD != 0 {
                    let tx = self.transform_final_tx(g, obj);
                    if let Shape::Transform { tx: t, .. } = &mut self.geoms[g].shape {
                        *t = tx;
                    }
                }
            }
        }
        let (a, b) = (self.prim_of(o1), self.prim_of(o2));
        self.dispatch(&a, &b, flags)
    }

    /// the geom as colliders see it
    fn prim_of(&self, g: GeomId) -> Prim<'_> {
        let gg = &self.geoms[g];
        let pr = self.geom_posr(g);
        let q = match (gg.body, &gg.offset) {
            (Some(b), None) => self.bodies[b].q,
            _ => q_from_r(&pr.r),
        };
        Prim { shape: &gg.shape, pr, id: g, flags: gg.gflags, q, body: gg.body }
    }

    /// dCollide's table lookup (after the space case), with dCollideTransform
    fn dispatch(&self, a: &Prim, b: &Prim, flags: i32) -> Vec<ContactGeom> {
        let (c1, c2) = (a.shape.class(), b.shape.class());
        if c1 == TRANSFORM || c2 == TRANSFORM {
            let (tr, other, rev) = if c1 == TRANSFORM { (a, b, false) } else { (b, a, true) };
            let mut out = self.collide_transform(tr, other, flags);
            if rev {
                flip(&mut out);
            }
            return out;
        }
        let Some((reverse, f)) = collider(c1, c2) else { return Vec::new() };
        let mut out = if reverse { f(b, a, flags) } else { f(a, b, flags) };
        if reverse {
            flip(&mut out);
        }
        out
    }

    /// dCollideTransform: the held geom, placed by the transform, collided
    /// through dCollide
    fn collide_transform(&self, tr: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
        let Shape::Transform { obj: Some(obj), info, tx, .. } = tr.shape else { return Vec::new() };
        let og = &self.geoms[*obj];
        let q = if tr.body.is_some() && og.offset.is_none() { self.bodies[tr.body.unwrap()].q } else { q_from_r(&tx.r) };
        let p = Prim { shape: &og.shape, pr: *tx, id: *obj, flags: og.gflags, q, body: tr.body };
        if flags & NUMC_MASK == 0 || p.id == o2.id || (p.body == o2.body && p.body.is_some()) {
            return Vec::new();
        }
        let mut out = self.dispatch(&p, o2, flags);
        if *info {
            for c in &mut out {
                c.g1 = tr.id;
            }
        }
        out
    }
}

fn flip(out: &mut [ContactGeom]) {
    for c in out {
        c.normal[0] = -c.normal[0];
        c.normal[1] = -c.normal[1];
        c.normal[2] = -c.normal[2];
        std::mem::swap(&mut c.g1, &mut c.g2);
        std::mem::swap(&mut c.side1, &mut c.side2);
    }
}

/// a geom's own AABB (computeAABB) at placement `pr`
fn geom_shape_aabb(shape: &Shape, gflags: u32, pr: &PosR) -> [Real; 6] {
    match shape {
        Shape::Heightfield(d) => heightfield_aabb(d, gflags & GEOM_PLACEABLE != 0, pr),
        _ => shape_aabb(shape, pr),
    }
}

// ---- quadtree blocks

#[allow(clippy::too_many_arguments)]
fn qt_create(blocks: &mut [QtBlock], me: usize, min_x: Real, max_x: Real, min_z: Real, max_z: Real, parent: Option<usize>, depth: i32, cursor: &mut usize) {
    blocks[me] = QtBlock { min_x, max_x, min_z, max_z, list: Vec::new(), count: 0, parent, children: None };
    if depth > 0 {
        let ch = *cursor;
        blocks[me].children = Some(ch);
        *cursor += 4;
        let ex = (max_x - min_x) / 2.0;
        let ez = (max_z - min_z) / 2.0;
        let mut index = 0;
        let mut rx = min_x;
        for i in 0..2 {
            let lx = rx;
            rx = if i != 1 { lx + ex } else { max_x };
            let mut rz = min_z;
            for j in 0..2 {
                let lz = rz;
                rz = if j != 1 { lz + ez } else { max_z };
                qt_create(blocks, ch + index, lx, rx, lz, rz, Some(me), depth - 1, cursor);
                index += 1;
            }
        }
    }
}
fn qt_inside(b: &QtBlock, a: &[Real; 6]) -> bool {
    a[0] >= b.min_x && a[1] < b.max_x && a[2] >= b.min_z && a[3] < b.max_z
}
fn qt_get_block_child(blocks: &[QtBlock], me: usize, a: &[Real; 6]) -> usize {
    if let Some(ch) = blocks[me].children {
        for i in 0..4 {
            if qt_inside(&blocks[ch + i], a) {
                return qt_get_block_child(blocks, ch + i, a);
            }
        }
    }
    me
}
fn qt_get_block(blocks: &[QtBlock], me: usize, a: &[Real; 6]) -> usize {
    if qt_inside(&blocks[me], a) {
        qt_get_block_child(blocks, me, a)
    } else if let Some(p) = blocks[me].parent {
        qt_get_block(blocks, p, a)
    } else {
        me
    }
}
fn qt_add(blocks: &mut [QtBlock], me: usize, g: GeomId) {
    blocks[me].list.push(g);
    let mut b = Some(me);
    while let Some(x) = b {
        blocks[x].count += 1;
        b = blocks[x].parent;
    }
}
fn qt_del(blocks: &mut [QtBlock], me: usize, g: GeomId) {
    if let Some(k) = blocks[me].list.iter().rposition(|&x| x == g) {
        blocks[me].list.remove(k);
    }
    let mut b = Some(me);
    while let Some(x) = b {
        blocks[x].count -= 1;
        b = blocks[x].parent;
    }
}

impl World {
    /// Block::Collide(UserData, Callback)
    fn qt_collide(&self, blocks: &[QtBlock], me: usize, out: &mut Vec<(GeomId, GeomId)>) {
        let list: Vec<GeomId> = blocks[me].list.iter().rev().copied().collect();
        for (k, &g) in list.iter().enumerate() {
            if enabled(&self.geoms[g]) {
                self.qt_collide_pair(blocks, me, g, &list[k + 1..], out);
            }
        }
        if let Some(ch) = blocks[me].children {
            for i in 0..4 {
                if blocks[ch + i].count <= 1 {
                    continue;
                }
                self.qt_collide(blocks, ch + i, out);
            }
        }
    }
    /// Block::Collide(g1, g2 list, ...)
    fn qt_collide_pair(&self, blocks: &[QtBlock], me: usize, g1: GeomId, g2s: &[GeomId], out: &mut Vec<(GeomId, GeomId)>) {
        for &g2 in g2s {
            if enabled(&self.geoms[g2]) {
                self.collide_aabbs(g1, g2, out);
            }
        }
        if let Some(ch) = blocks[me].children {
            for i in 0..4 {
                let c = &blocks[ch + i];
                if c.count == 0 {
                    continue;
                }
                if c.count != 1 {
                    let a = &self.geoms[g1].aabb;
                    if a[0] >= c.max_x || a[1] < c.min_x || a[2] >= c.max_z || a[3] < c.min_z {
                        continue;
                    }
                }
                let first: Vec<GeomId> = c.list.iter().rev().copied().collect();
                self.qt_collide_pair(blocks, ch + i, g1, &first, out);
            }
        }
    }

    /// collideGeomsNoAABBs
    fn collide_no_aabbs(&self, g1: GeomId, g2: GeomId, out: &mut Vec<(GeomId, GeomId)>) {
        let (a, b) = (&self.geoms[g1], &self.geoms[g2]);
        if a.body == b.body && a.body.is_some() {
            return;
        }
        if (a.category_bits & b.collide_bits) == 0 && (b.category_bits & a.collide_bits) == 0 {
            return;
        }
        out.push((g1, g2));
    }

    /// dxSAPSpace::collide
    fn sap_collide(&mut self, s: GeomId, out: &mut Vec<(GeomId, GeomId)>) {
        self.clean_geoms(s);
        let SpaceKind::Sap(d) = &self.space_ref(s).kind else { return };
        let (geoms, ax) = (d.geoms.clone(), d.ax);
        let mut tmp: Vec<GeomId> = Vec::new();
        let mut inf: Vec<GeomId> = Vec::new();
        for &g in &geoms {
            if !enabled(&self.geoms[g]) {
                continue;
            }
            if self.geoms[g].aabb[ax[0] + 1] == INFINITY {
                inf.push(g);
            } else {
                tmp.push(g);
            }
        }
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        if !tmp.is_empty() {
            let poslist: Vec<f32> = tmp.iter().map(|&g| self.geoms[g].aabb[ax[0]] as f32).collect();
            let sorted = if let SpaceKind::Sap(d) = &mut self.space_mut(s).kind { d.sort.sort(&poslist) } else { Vec::new() };
            let last = sorted.len();
            let mut si = 0usize;
            let mut running = 0usize;
            'outer: while si < last {
                let id0 = sorted[si];
                si += 1;
                loop {
                    let v = poslist[sorted[running] as usize];
                    running += 1;
                    if !(v < poslist[id0 as usize]) {
                        break;
                    }
                    if running == last {
                        break 'outer;
                    }
                }
                if running == last {
                    break;
                }
                let a0 = self.geoms[tmp[id0 as usize]].aabb;
                let max0 = a0[ax[0] + 1] as f32;
                let (max1, max2) = (a0[ax[1] + 1], a0[ax[2] + 1]);
                let mut r2 = running;
                loop {
                    let id1 = sorted[r2];
                    r2 += 1;
                    if !(poslist[id1 as usize] <= max0) {
                        break;
                    }
                    let a1 = &self.geoms[tmp[id1 as usize]].aabb;
                    if max1 >= a1[ax[1]] && a1[ax[1] + 1] >= a0[ax[1]] && max2 >= a1[ax[2]] && a1[ax[2] + 1] >= a0[ax[2]] {
                        pairs.push((id0, id1));
                    }
                    if r2 == last {
                        break;
                    }
                }
            }
        }
        for (i0, i1) in pairs {
            self.collide_no_aabbs(tmp[i0 as usize], tmp[i1 as usize], out);
        }
        for m in 0..inf.len() {
            for n in m + 1..inf.len() {
                self.collide_no_aabbs(inf[m], inf[n], out);
            }
            for &g2 in &tmp {
                self.collide_no_aabbs(inf[m], g2, out);
            }
        }
    }
}

impl RadixSort {
    fn ranks2(&self) -> usize {
        self.size + self.size - self.primary
    }
    /// RaixSortContext::RadixSort: indices of `input` in ascending order,
    /// reusing (and checking) the previous call's ranks
    fn sort(&mut self, input: &[f32]) -> Vec<u32> {
        let nb = input.len();
        // ReallocateRanksIfNecessary
        if nb != self.utilization {
            if nb > self.size {
                self.buf = vec![0; 2 * nb];
                self.primary = 0;
                self.size = nb;
            }
            self.valid = false;
            self.utilization = nb;
        }
        let bytes: Vec<u8> = input.iter().flat_map(|f| f.to_bits().to_le_bytes()).collect();
        let words: Vec<u32> = input.iter().map(|f| f.to_bits()).collect();
        let mut hist = [0u32; 1024];
        let mut p = 0usize;
        let pe = nb * 4;
        let mut already = true;
        if !self.valid {
            let mut prev = input[0];
            let mut k = 0;
            while p != pe {
                let v = input[k];
                k += 1;
                if v < prev {
                    already = false;
                    break;
                }
                prev = v;
                for c in 0..4 {
                    hist[c * 256 + bytes[p] as usize] += 1;
                    p += 1;
                }
            }
            if already {
                let r1 = self.primary;
                for i in 0..nb {
                    self.buf[r1 + i] = i as u32;
                }
                return self.buf[r1..r1 + nb].to_vec();
            }
        } else {
            let r1 = self.primary;
            let mut k = 0;
            let mut prev = input[self.buf[r1] as usize];
            while p != pe {
                let v = input[self.buf[r1 + k] as usize];
                k += 1;
                if v < prev {
                    already = false;
                    break;
                }
                prev = v;
                for c in 0..4 {
                    hist[c * 256 + bytes[p] as usize] += 1;
                    p += 1;
                }
            }
            if already {
                return self.buf[r1..r1 + nb].to_vec();
            }
        }
        while p != pe {
            for c in 0..4 {
                hist[c * 256 + bytes[p] as usize] += 1;
                p += 1;
            }
        }
        let mut negatives = 0u32;
        for i in 128..256 {
            negatives += hist[768 + i];
        }
        for j in 0..4 {
            let cur = &hist[j * 256..j * 256 + 256];
            let unique = bytes[j];
            let perform = cur[unique as usize] as usize != nb;
            if j != 3 {
                if perform {
                    let r2 = self.ranks2();
                    let mut link = [0usize; 256];
                    link[0] = r2;
                    for i in 1..256 {
                        link[i] = link[i - 1] + cur[i - 1] as usize;
                    }
                    if !self.valid {
                        for i in 0..nb {
                            let b = bytes[(i << 2) + j] as usize;
                            self.buf[link[b]] = i as u32;
                            link[b] += 1;
                        }
                        self.valid = true;
                    } else {
                        let r1 = self.primary;
                        for k in 0..nb {
                            let id = self.buf[r1 + k];
                            let b = bytes[((id as usize) << 2) + j] as usize;
                            self.buf[link[b]] = id;
                            link[b] += 1;
                        }
                    }
                    self.primary = self.ranks2();
                }
            } else if perform {
                let r2 = self.ranks2();
                let mut link = [0usize; 256];
                link[0] = r2 + negatives as usize;
                for i in 1..128 {
                    link[i] = link[i - 1] + cur[i - 1] as usize;
                }
                link[255] = r2;
                for i in 0..127 {
                    link[254 - i] = link[255 - i] + cur[255 - i] as usize;
                }
                for i in 128..256 {
                    link[i] += cur[i] as usize;
                }
                if !self.valid {
                    for i in 0..nb {
                        let radix = (words[i] >> 24) as usize;
                        if radix < 128 {
                            self.buf[link[radix]] = i as u32;
                            link[radix] += 1;
                        } else {
                            link[radix] -= 1;
                            self.buf[link[radix]] = i as u32;
                        }
                    }
                    self.valid = true;
                } else {
                    let r1 = self.primary;
                    for i in 0..nb {
                        let id = self.buf[r1 + i];
                        let radix = (words[id as usize] >> 24) as usize;
                        if radix < 128 {
                            self.buf[link[radix]] = id;
                            link[radix] += 1;
                        } else {
                            link[radix] -= 1;
                            self.buf[link[radix]] = id;
                        }
                    }
                }
                self.primary = self.ranks2();
            } else if unique >= 128 {
                let r2 = self.ranks2();
                if !self.valid {
                    for i in 0..nb {
                        self.buf[r2 + i] = (nb - i - 1) as u32;
                    }
                    self.valid = true;
                } else {
                    let r1 = self.primary;
                    for i in 0..nb {
                        self.buf[r2 + i] = self.buf[r1 + nb - i - 1];
                    }
                }
                self.primary = self.ranks2();
            }
        }
        let r1 = self.primary;
        self.buf[r1..r1 + nb].to_vec()
    }
}

/// getBodyPosr: the body placement that puts an offset geom at `fin`
fn body_posr(off: &PosR, fin: &PosR) -> PosR {
    let mut inv = off.r;
    inv.swap(4, 1);
    inv.swap(2, 8);
    inv.swap(9, 6);
    let r = mul0_333(&fin.r, &inv);
    let wo = mul0_331(&r, &off.pos);
    PosR { pos: v3(fin.pos[0] - wo[0], fin.pos[1] - wo[1], fin.pos[2] - wo[2]), r }
}

fn plane_unit_normal(p: &mut [Real; 4]) {
    let l = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    if l > 0.0 {
        let l = recip_sqrt(l);
        p[0] *= l;
        p[1] *= l;
        p[2] *= l;
        p[3] *= l;
    } else {
        *p = [1.0, 0.0, 0.0, 0.0];
    }
}

fn shape_aabb(shape: &Shape, pr: &PosR) -> [Real; 6] {
    let (r, pos) = (&pr.r, &pr.pos);
    match shape {
        Shape::Sphere { radius } => [pos[0] - radius, pos[0] + radius, pos[1] - radius, pos[1] + radius, pos[2] - radius, pos[2] + radius],
        Shape::Box { side } => {
            let xr = 0.5 * ((r[0] * side[0]).abs() + (r[1] * side[1]).abs() + (r[2] * side[2]).abs());
            let yr = 0.5 * ((r[4] * side[0]).abs() + (r[5] * side[1]).abs() + (r[6] * side[2]).abs());
            let zr = 0.5 * ((r[8] * side[0]).abs() + (r[9] * side[1]).abs() + (r[10] * side[2]).abs());
            [pos[0] - xr, pos[0] + xr, pos[1] - yr, pos[1] + yr, pos[2] - zr, pos[2] + zr]
        }
        Shape::Capsule { radius, lz } => {
            let xr = (r[2] * lz).abs() * 0.5 + radius;
            let yr = (r[6] * lz).abs() * 0.5 + radius;
            let zr = (r[10] * lz).abs() * 0.5 + radius;
            [pos[0] - xr, pos[0] + xr, pos[1] - yr, pos[1] + yr, pos[2] - zr, pos[2] + zr]
        }
        Shape::Plane { p } => {
            let mut a = [-INFINITY, INFINITY, -INFINITY, INFINITY, -INFINITY, INFINITY];
            if p[1] == 0.0 && p[2] == 0.0 {
                a[0] = if p[0] > 0.0 { -INFINITY } else { -p[3] };
                a[1] = if p[0] > 0.0 { p[3] } else { INFINITY };
            } else if p[0] == 0.0 && p[2] == 0.0 {
                a[2] = if p[1] > 0.0 { -INFINITY } else { -p[3] };
                a[3] = if p[1] > 0.0 { p[3] } else { INFINITY };
            } else if p[0] == 0.0 && p[1] == 0.0 {
                a[4] = if p[2] > 0.0 { -INFINITY } else { -p[3] };
                a[5] = if p[2] > 0.0 { p[3] } else { INFINITY };
            }
            a
        }
        Shape::Ray { length } => {
            let e = [pos[0] + r[2] * length, pos[1] + r[6] * length, pos[2] + r[10] * length];
            let mut a = [0.0; 6];
            for i in 0..3 {
                if pos[i] < e[i] {
                    a[2 * i] = pos[i];
                    a[2 * i + 1] = e[i];
                } else {
                    a[2 * i] = e[i];
                    a[2 * i + 1] = pos[i];
                }
            }
            a
        }
        Shape::TriMesh(d) => {
            // dxTriMesh::computeAABB
            let c = mul0_331(r, &d.aabb_center);
            let e = &d.aabb_extents;
            let xr = (r[0] * e[0]).abs() + (r[1] * e[1]).abs() + (r[2] * e[2]).abs();
            let yr = (r[4] * e[0]).abs() + (r[5] * e[1]).abs() + (r[6] * e[2]).abs();
            let zr = (r[8] * e[0]).abs() + (r[9] * e[1]).abs() + (r[10] * e[2]).abs();
            [c[0] + pos[0] - xr, c[0] + pos[0] + xr, c[1] + pos[1] - yr, c[1] + pos[1] + yr, c[2] + pos[2] - zr, c[2] + pos[2] + zr]
        }
        Shape::Cylinder { radius, lz } => cylinder_aabb(*radius, *lz, pr),
        Shape::Convex(c) => convex_aabb(c, pr),
        Shape::Heightfield(d) => heightfield_aabb(d, true, pr),
        Shape::Transform { .. } | Shape::Space(_) => [0.0; 6],
    }
}

/// findLevel: frexp's exponent of the AABB's largest side
fn find_level(b: &[Real; 6]) -> i32 {
    if b[0] <= -INFINITY || b[1] >= INFINITY || b[2] <= -INFINITY || b[3] >= INFINITY || b[4] <= -INFINITY || b[5] >= INFINITY {
        return i32::MAX;
    }
    let mut q = b[1] - b[0];
    let q2 = b[3] - b[2];
    if q2 > q {
        q = q2;
    }
    let q2 = b[5] - b[4];
    if q2 > q {
        q = q2;
    }
    frexp_exp(q)
}
fn frexp_exp(x: Real) -> i32 {
    if x == 0.0 || !x.is_finite() {
        return 0;
    }
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32;
    if e == 0 {
        frexp_exp(x * (1u64 << 54) as f64) - 54
    } else {
        e - 1022
    }
}
fn ldexp1(level: i32) -> Real {
    2f64.powi(level)
}

// ---- the colliders

/// one geom as a collider sees it
pub struct Prim<'a> {
    pub shape: &'a Shape,
    pub pr: PosR,
    pub id: GeomId,
    pub flags: u32,
    /// dGeomGetQuaternion
    pub q: Quat,
    pub body: Option<BodyId>,
}
type ColliderFn = fn(&Prim, &Prim, i32) -> Vec<ContactGeom>;

/// dInitColliders' table: (reverse, function) for a class pair
fn collider(c1: usize, c2: usize) -> Option<(bool, ColliderFn)> {
    // registration order, first wins (setCollider)
    let table: [(usize, usize, ColliderFn); 37] = [
        (SPHERE, SPHERE, collide_sphere_sphere),
        (SPHERE, BOX, collide_sphere_box),
        (SPHERE, PLANE, collide_sphere_plane),
        (BOX, BOX, collide_box_box),
        (BOX, PLANE, collide_box_plane),
        (CAPSULE, SPHERE, collide_capsule_sphere),
        (CAPSULE, BOX, collide_capsule_box),
        (CAPSULE, CAPSULE, collide_capsule_capsule),
        (CAPSULE, PLANE, collide_capsule_plane),
        (RAY, SPHERE, collide_ray_sphere),
        (RAY, BOX, collide_ray_box),
        (RAY, CAPSULE, collide_ray_capsule),
        (RAY, PLANE, collide_ray_plane),
        (RAY, CYLINDER, collide_ray_cylinder),
        (TRIMESH, SPHERE, collide_sphere_trimesh),
        (TRIMESH, BOX, collide_box_trimesh),
        (TRIMESH, RAY, collide_ray_trimesh),
        (TRIMESH, TRIMESH, collide_trimesh_trimesh),
        (TRIMESH, CAPSULE, collide_capsule_trimesh),
        (TRIMESH, PLANE, collide_trimesh_plane),
        (CYLINDER, TRIMESH, collide_cylinder_trimesh),
        (CONVEX, TRIMESH, collide_convex_trimesh),
        (CYLINDER, BOX, collide_cylinder_box),
        (CYLINDER, SPHERE, collide_cylinder_sphere),
        (CYLINDER, PLANE, collide_cylinder_plane),
        (CONVEX, BOX, collide_convex_box),
        (CONVEX, CAPSULE, collide_convex_capsule),
        (SPHERE, CONVEX, collide_sphere_convex),
        (CONVEX, CONVEX, collide_convex_convex),
        (CONVEX, PLANE, collide_convex_plane),
        (RAY, CONVEX, collide_ray_convex),
        (HEIGHTFIELD, RAY, collide_heightfield),
        (HEIGHTFIELD, SPHERE, collide_heightfield),
        (HEIGHTFIELD, BOX, collide_heightfield),
        (HEIGHTFIELD, CAPSULE, collide_heightfield),
        (HEIGHTFIELD, CYLINDER, collide_heightfield),
        (HEIGHTFIELD, CONVEX, collide_heightfield),
    ];
    let hf_tri: (usize, usize, ColliderFn) = (HEIGHTFIELD, TRIMESH, collide_heightfield);
    for &(i, j, f) in table.iter().chain(std::iter::once(&hf_tri)) {
        if i == c1 && j == c2 {
            return Some((false, f));
        }
        if j == c1 && i == c2 {
            return Some((true, f));
        }
    }
    None
}

fn cg(o1: &Prim, o2: &Prim) -> ContactGeom {
    ContactGeom { g1: o1.id, g2: o2.id, side1: -1, side2: -1, ..Default::default() }
}

/// dCollideSpheres
fn collide_spheres(p1: &[Real], r1: Real, p2: &[Real], r2: Real, c: &mut ContactGeom) -> bool {
    let d = distance3(p1, p2);
    if d > r1 + r2 {
        return false;
    }
    if d <= 0.0 {
        c.pos = [p1[0], p1[1], p1[2], 0.0];
        c.normal = [1.0, 0.0, 0.0, 0.0];
        c.depth = r1 + r2;
    } else {
        let d1 = recip(d);
        c.normal[0] = (p1[0] - p2[0]) * d1;
        c.normal[1] = (p1[1] - p2[1]) * d1;
        c.normal[2] = (p1[2] - p2[2]) * d1;
        let k = 0.5 * (r2 - r1 - d);
        c.pos[0] = p1[0] + c.normal[0] * k;
        c.pos[1] = p1[1] + c.normal[1] * k;
        c.pos[2] = p1[2] + c.normal[2] * k;
        c.depth = r1 + r2 - d;
    }
    true
}

fn one(c: ContactGeom, hit: bool) -> Vec<ContactGeom> {
    if hit {
        vec![c]
    } else {
        Vec::new()
    }
}

fn radius_of(s: &Shape) -> Real {
    match s {
        Shape::Sphere { radius } | Shape::Capsule { radius, .. } => *radius,
        _ => 0.0,
    }
}

fn collide_sphere_sphere(o1: &Prim, o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    let mut c = cg(o1, o2);
    let hit = collide_spheres(&o1.pr.pos, radius_of(o1.shape), &o2.pr.pos, radius_of(o2.shape), &mut c);
    one(c, hit)
}

fn collide_sphere_box(o1: &Prim, o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    let Shape::Box { side } = o2.shape else { return Vec::new() };
    let radius = radius_of(o1.shape);
    let mut c = cg(o1, o2);
    let (p1, p2, r2) = (&o1.pr.pos, &o2.pr.pos, &o2.pr.r);
    let p = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2], 0.0];
    let mut l = [0.0; 3];
    let mut t = [0.0; 4];
    let mut onborder = false;
    l[0] = side[0] * 0.5;
    t[0] = dot3_14(&p, &r2[0..]);
    if t[0] < -l[0] {
        t[0] = -l[0];
        onborder = true;
    }
    if t[0] > l[0] {
        t[0] = l[0];
        onborder = true;
    }
    l[1] = side[1] * 0.5;
    t[1] = dot3_14(&p, &r2[1..]);
    if t[1] < -l[1] {
        t[1] = -l[1];
        onborder = true;
    }
    if t[1] > l[1] {
        t[1] = l[1];
        onborder = true;
    }
    t[2] = dot3_14(&p, &r2[2..]);
    l[2] = side[2] * 0.5;
    if t[2] < -l[2] {
        t[2] = -l[2];
        onborder = true;
    }
    if t[2] > l[2] {
        t[2] = l[2];
        onborder = true;
    }
    if !onborder {
        let mut min_distance = l[0] - t[0].abs();
        let mut mini = 0;
        for i in 1..3 {
            let fd = l[i] - t[i].abs();
            if fd < min_distance {
                min_distance = fd;
                mini = i;
            }
        }
        c.pos = [p1[0], p1[1], p1[2], 0.0];
        let mut tmp = [0.0; 4];
        tmp[mini] = if t[mini] > 0.0 { 1.0 } else { -1.0 };
        c.normal = mul0_331(r2, &tmp);
        c.depth = min_distance + radius;
        return vec![c];
    }
    t[3] = 0.0;
    let q = mul0_331(r2, &t);
    let r = [p[0] - q[0], p[1] - q[1], p[2] - q[2], 0.0];
    let depth = radius - dot3(&r, &r).sqrt();
    if depth < 0.0 {
        return Vec::new();
    }
    c.pos = [q[0] + p2[0], q[1] + p2[1], q[2] + p2[2], 0.0];
    c.normal = r;
    normalize3(&mut c.normal);
    c.depth = depth;
    vec![c]
}

fn collide_sphere_plane(o1: &Prim, o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    let Shape::Plane { p } = o2.shape else { return Vec::new() };
    let radius = radius_of(o1.shape);
    let mut c = cg(o1, o2);
    let pos = &o1.pr.pos;
    let k = dot3(pos, p);
    let depth = p[3] - k + radius;
    if depth >= 0.0 {
        c.normal = [p[0], p[1], p[2], 0.0];
        c.pos = [pos[0] - p[0] * radius, pos[1] - p[1] * radius, pos[2] - p[2] * radius, 0.0];
        c.depth = depth;
        vec![c]
    } else {
        Vec::new()
    }
}

/// intersectRectQuad: clip a quad against the rectangle |x| <= h0, |y| <= h1
fn intersect_rect_quad(h: [Real; 2], p: &[Real; 8], ret: &mut [Real; 16]) -> usize {
    let mut nq = 4usize;
    let mut nr = 0usize;
    let mut buffer = [0.0; 16];
    let mut qbuf = [0.0; 16];
    qbuf[..8].copy_from_slice(p);
    // the current input polygon is q, output r; ODE ping-pongs ret/buffer
    'outer: for dir in 0..=1usize {
        for sign in [-1.0, 1.0] {
            let mut r = [0.0; 16];
            nr = 0;
            let mut pq = 0usize;
            let mut pr = 0usize;
            let mut i = nq;
            while i > 0 {
                if sign * qbuf[pq + dir] < h[dir] {
                    r[pr] = qbuf[pq];
                    r[pr + 1] = qbuf[pq + 1];
                    pr += 2;
                    nr += 1;
                    if nr & 8 != 0 {
                        qbuf = r;
                        break 'outer;
                    }
                }
                let nextq = if i > 1 { pq + 2 } else { 0 };
                if (sign * qbuf[pq + dir] < h[dir]) ^ (sign * qbuf[nextq + dir] < h[dir]) {
                    r[pr + 1 - dir] = qbuf[pq + 1 - dir] + (qbuf[nextq + 1 - dir] - qbuf[pq + 1 - dir]) / (qbuf[nextq + dir] - qbuf[pq + dir]) * (sign * h[dir] - qbuf[pq + dir]);
                    r[pr + dir] = sign * h[dir];
                    pr += 2;
                    nr += 1;
                    if nr & 8 != 0 {
                        qbuf = r;
                        break 'outer;
                    }
                }
                pq += 2;
                i -= 1;
            }
            qbuf = r;
            nq = nr;
        }
    }
    let _ = &mut buffer;
    ret[..nr * 2].copy_from_slice(&qbuf[..nr * 2]);
    nr
}

/// cullPoints: pick m of the n points, spread around their centroid
fn cull_points(n: usize, p: &[Real], m: usize, i0: usize, iret: &mut [usize]) {
    let (cx, cy);
    if n == 1 {
        cx = p[0];
        cy = p[1];
    } else if n == 2 {
        cx = 0.5 * (p[0] + p[2]);
        cy = 0.5 * (p[1] + p[3]);
    } else {
        let mut a = 0.0;
        let mut sx = 0.0;
        let mut sy = 0.0;
        for i in 0..n - 1 {
            let q = p[i * 2] * p[i * 2 + 3] - p[i * 2 + 2] * p[i * 2 + 1];
            a += q;
            sx += q * (p[i * 2] + p[i * 2 + 2]);
            sy += q * (p[i * 2 + 1] + p[i * 2 + 3]);
        }
        let q = p[n * 2 - 2] * p[1] - p[0] * p[n * 2 - 1];
        let a = recip(3.0 * (a + q));
        cx = a * (sx + q * (p[n * 2 - 2] + p[0]));
        cy = a * (sy + q * (p[n * 2 - 1] + p[1]));
    }
    let mut aa = [0.0; 8];
    for i in 0..n {
        aa[i] = (p[i * 2 + 1] - cy).atan2(p[i * 2] - cx);
    }
    let mut avail = [true; 8];
    avail[i0] = false;
    iret[0] = i0;
    let pi = std::f64::consts::PI;
    for j in 1..m {
        let mut a = j as Real * (2.0 * pi / m as Real) + aa[i0];
        if a > pi {
            a -= 2.0 * pi;
        }
        let mut maxdiff = 1e9;
        iret[j] = i0;
        for i in 0..n {
            if avail[i] {
                let mut diff = (aa[i] - a).abs();
                if diff > pi {
                    diff = 2.0 * pi - diff;
                }
                if diff < maxdiff {
                    maxdiff = diff;
                    iret[j] = i;
                }
            }
        }
        avail[iret[j]] = false;
    }
}

/// dLineClosestApproach
fn line_closest_approach(pa: &[Real], ua: &[Real], pb: &[Real], ub: &[Real]) -> (Real, Real) {
    let p = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    let uaub = dot3(ua, ub);
    let q1 = dot3(ua, &p);
    let q2 = -dot3(ub, &p);
    let d = 1.0 - uaub * uaub;
    if d <= 0.0001 {
        (0.0, 0.0)
    } else {
        let d = recip(d);
        ((q1 + uaub * q2) * d, (uaub * q1 + q2) * d)
    }
}

/// dBoxBox: separating-axis test plus contact generation. Returns the
/// contacts (positions and depths), the normal and the axis code.
#[allow(clippy::too_many_arguments)]
fn box_box(p1: &[Real], r1: &[Real], side1: &[Real], p2: &[Real], r2: &[Real], side2: &[Real], flags: i32) -> Option<(Vec<ContactGeom>, V3, Real, i32)> {
    let fudge_factor = 1.05;
    let mut normal_c = [0.0; 4];
    let mut normal_r: Option<usize> = None;
    let p = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2], 0.0];
    let pp = mul1_331(r1, &p);
    let a = [side1[0] * 0.5, side1[1] * 0.5, side1[2] * 0.5];
    let b = [side2[0] * 0.5, side2[1] * 0.5, side2[2] * 0.5];
    let r11 = dot3_44(&r1[0..], &r2[0..]);
    let r12 = dot3_44(&r1[0..], &r2[1..]);
    let r13 = dot3_44(&r1[0..], &r2[2..]);
    let r21 = dot3_44(&r1[1..], &r2[0..]);
    let r22 = dot3_44(&r1[1..], &r2[1..]);
    let r23 = dot3_44(&r1[1..], &r2[2..]);
    let r31 = dot3_44(&r1[2..], &r2[0..]);
    let r32 = dot3_44(&r1[2..], &r2[1..]);
    let r33 = dot3_44(&r1[2..], &r2[2..]);
    let (q11, q12, q13) = (r11.abs(), r12.abs(), r13.abs());
    let (q21, q22, q23) = (r21.abs(), r22.abs(), r23.abs());
    let (q31, q32, q33) = (r31.abs(), r32.abs(), r33.abs());

    let mut s = -INFINITY;
    let mut invert_normal = false;
    let mut code = 0;
    // which matrix the face normal comes from: 1 = R1, 2 = R2 (column)
    let mut normal_from_r2 = false;
    'tests: {
        macro_rules! tst {
            ($e1:expr, $e2:expr, $col:expr, $r2:expr, $cc:expr) => {
                let e1 = $e1;
                let s2 = e1.abs() - ($e2);
                if s2 > 0.0 {
                    return None;
                }
                if s2 > s {
                    s = s2;
                    normal_r = Some($col);
                    normal_from_r2 = $r2;
                    invert_normal = e1 < 0.0;
                    code = $cc;
                    if flags & CONTACTS_UNIMPORTANT != 0 {
                        break 'tests;
                    }
                }
            };
        }
        tst!(pp[0], a[0] + b[0] * q11 + b[1] * q12 + b[2] * q13, 0, false, 1);
        tst!(pp[1], a[1] + b[0] * q21 + b[1] * q22 + b[2] * q23, 1, false, 2);
        tst!(pp[2], a[2] + b[0] * q31 + b[1] * q32 + b[2] * q33, 2, false, 3);
        tst!(dot3_41(&r2[0..], &p), a[0] * q11 + a[1] * q21 + a[2] * q31 + b[0], 0, true, 4);
        tst!(dot3_41(&r2[1..], &p), a[0] * q12 + a[1] * q22 + a[2] * q32 + b[1], 1, true, 5);
        tst!(dot3_41(&r2[2..], &p), a[0] * q13 + a[1] * q23 + a[2] * q33 + b[2], 2, true, 6);
        macro_rules! tst2 {
            ($e1:expr, $e2:expr, $n1:expr, $n2:expr, $n3:expr, $cc:expr) => {
                let e1 = $e1;
                let mut s2 = e1.abs() - ($e2);
                if s2 > 0.0 {
                    return None;
                }
                let (n1, n2, n3): (Real, Real, Real) = ($n1, $n2, $n3);
                let l = (n1 * n1 + n2 * n2 + n3 * n3).sqrt();
                if l > 0.0 {
                    s2 /= l;
                    if s2 * fudge_factor > s {
                        s = s2;
                        normal_r = None;
                        normal_c = [n1 / l, n2 / l, n3 / l, 0.0];
                        invert_normal = e1 < 0.0;
                        code = $cc;
                        if flags & CONTACTS_UNIMPORTANT != 0 {
                            break 'tests;
                        }
                    }
                }
            };
        }
        tst2!(pp[2] * r21 - pp[1] * r31, a[1] * q31 + a[2] * q21 + b[1] * q13 + b[2] * q12, 0.0, -r31, r21, 7);
        tst2!(pp[2] * r22 - pp[1] * r32, a[1] * q32 + a[2] * q22 + b[0] * q13 + b[2] * q11, 0.0, -r32, r22, 8);
        tst2!(pp[2] * r23 - pp[1] * r33, a[1] * q33 + a[2] * q23 + b[0] * q12 + b[1] * q11, 0.0, -r33, r23, 9);
        tst2!(pp[0] * r31 - pp[2] * r11, a[0] * q31 + a[2] * q11 + b[1] * q23 + b[2] * q22, r31, 0.0, -r11, 10);
        tst2!(pp[0] * r32 - pp[2] * r12, a[0] * q32 + a[2] * q12 + b[0] * q23 + b[2] * q21, r32, 0.0, -r12, 11);
        tst2!(pp[0] * r33 - pp[2] * r13, a[0] * q33 + a[2] * q13 + b[0] * q22 + b[1] * q21, r33, 0.0, -r13, 12);
        tst2!(pp[1] * r11 - pp[0] * r21, a[0] * q21 + a[1] * q11 + b[1] * q33 + b[2] * q32, -r21, r11, 0.0, 13);
        tst2!(pp[1] * r12 - pp[0] * r22, a[0] * q22 + a[1] * q12 + b[0] * q33 + b[2] * q31, -r22, r12, 0.0, 14);
        tst2!(pp[1] * r13 - pp[0] * r23, a[0] * q23 + a[1] * q13 + b[0] * q32 + b[1] * q31, -r23, r13, 0.0, 15);
    }
    if code == 0 {
        return None;
    }
    let mut normal = match normal_r {
        Some(col) => {
            let m = if normal_from_r2 { r2 } else { r1 };
            [m[col], m[col + 4], m[col + 8], 0.0]
        }
        None => mul0_331(r1, &normal_c),
    };
    if invert_normal {
        normal = neg3(&normal);
    }
    let depth = -s;

    if code > 6 {
        let mut pa = [p1[0], p1[1], p1[2], 0.0];
        for j in 0..3 {
            let sign = if dot3_14(&normal, &r1[j..]) > 0.0 { 1.0 } else { -1.0 };
            for i in 0..3 {
                pa[i] += sign * a[j] * r1[i * 4 + j];
            }
        }
        let mut pb = [p2[0], p2[1], p2[2], 0.0];
        for j in 0..3 {
            let sign = if dot3_14(&normal, &r2[j..]) > 0.0 { -1.0 } else { 1.0 };
            for i in 0..3 {
                pb[i] += sign * b[j] * r2[i * 4 + j];
            }
        }
        let ci = ((code - 7) / 3) as usize;
        let cj = ((code - 7) % 3) as usize;
        let ua = [r1[ci], r1[ci + 4], r1[ci + 8]];
        let ub = [r2[cj], r2[cj + 4], r2[cj + 8]];
        let (alpha, beta) = line_closest_approach(&pa, &ua, &pb, &ub);
        for i in 0..3 {
            pa[i] += ua[i] * alpha;
        }
        for i in 0..3 {
            pb[i] += ub[i] * beta;
        }
        let mut c = ContactGeom::default();
        for i in 0..3 {
            c.pos[i] = 0.5 * (pa[i] + pb[i]);
        }
        c.depth = depth;
        return Some((vec![c], normal, depth, code));
    }

    let (ra, rb, pa, pb, sa, sb) = if code <= 3 { (r1, r2, p1, p2, a, b) } else { (r2, r1, p2, p1, b, a) };
    let normal2 = if code <= 3 { normal } else { neg3(&normal) };
    let nr = mul1_331(rb, &normal2);
    let anr = [nr[0].abs(), nr[1].abs(), nr[2].abs()];
    let (lanr, a1, a2);
    if anr[1] > anr[0] {
        if anr[1] > anr[2] {
            a1 = 0;
            lanr = 1;
            a2 = 2;
        } else {
            a1 = 0;
            a2 = 1;
            lanr = 2;
        }
    } else if anr[0] > anr[2] {
        lanr = 0;
        a1 = 1;
        a2 = 2;
    } else {
        a1 = 0;
        a2 = 1;
        lanr = 2;
    }
    let mut center = [0.0; 4];
    if nr[lanr] < 0.0 {
        for i in 0..3 {
            center[i] = pb[i] - pa[i] + sb[lanr] * rb[i * 4 + lanr];
        }
    } else {
        for i in 0..3 {
            center[i] = pb[i] - pa[i] - sb[lanr] * rb[i * 4 + lanr];
        }
    }
    let code_n = if code <= 3 { code - 1 } else { code - 4 } as usize;
    let (code1, code2) = match code_n {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let c1 = dot3_14(&center, &ra[code1..]);
    let c2 = dot3_14(&center, &ra[code2..]);
    let mut m11 = dot3_44(&ra[code1..], &rb[a1..]);
    let mut m12 = dot3_44(&ra[code1..], &rb[a2..]);
    let mut m21 = dot3_44(&ra[code2..], &rb[a1..]);
    let mut m22 = dot3_44(&ra[code2..], &rb[a2..]);
    let quad = {
        let k1 = m11 * sb[a1];
        let k2 = m21 * sb[a1];
        let k3 = m12 * sb[a2];
        let k4 = m22 * sb[a2];
        [c1 - k1 - k3, c2 - k2 - k4, c1 - k1 + k3, c2 - k2 + k4, c1 + k1 + k3, c2 + k2 + k4, c1 + k1 - k3, c2 + k2 - k4]
    };
    let rect = [sa[code1], sa[code2]];
    let mut ret = [0.0; 16];
    let n = intersect_rect_quad(rect, &quad, &mut ret);
    if n < 1 {
        return None;
    }
    let mut point = [0.0; 24];
    let mut dep = [0.0; 8];
    let det1 = recip(m11 * m22 - m12 * m21);
    m11 *= det1;
    m12 *= det1;
    m21 *= det1;
    m22 *= det1;
    let mut cnum = 0usize;
    for j in 0..n {
        let k1 = m22 * (ret[j * 2] - c1) - m12 * (ret[j * 2 + 1] - c2);
        let k2 = -m21 * (ret[j * 2] - c1) + m11 * (ret[j * 2 + 1] - c2);
        for i in 0..3 {
            point[cnum * 3 + i] = center[i] + k1 * rb[i * 4 + a1] + k2 * rb[i * 4 + a2];
        }
        dep[cnum] = sa[code_n] - dot3(&normal2, &point[cnum * 3..]);
        if dep[cnum] >= 0.0 {
            ret[cnum * 2] = ret[j * 2];
            ret[cnum * 2 + 1] = ret[j * 2 + 1];
            cnum += 1;
            if ((cnum as i32) | CONTACTS_UNIMPORTANT) == (flags & (NUMC_MASK | CONTACTS_UNIMPORTANT)) {
                break;
            }
        }
    }
    if cnum < 1 {
        return None;
    }
    let mut maxc = (flags & NUMC_MASK) as usize;
    if maxc > cnum {
        maxc = cnum;
    }
    if maxc < 1 {
        maxc = 1;
    }
    let mut out = Vec::new();
    if cnum <= maxc {
        for j in 0..cnum {
            let mut c = ContactGeom::default();
            for i in 0..3 {
                c.pos[i] = point[j * 3 + i] + pa[i];
            }
            c.depth = dep[j];
            out.push(c);
        }
    } else {
        let mut i1 = 0;
        let mut maxdepth = dep[0];
        for i in 1..cnum {
            if dep[i] > maxdepth {
                maxdepth = dep[i];
                i1 = i;
            }
        }
        let mut iret = [0usize; 8];
        cull_points(cnum, &ret, maxc, i1, &mut iret);
        for j in 0..maxc {
            let mut c = ContactGeom::default();
            for i in 0..3 {
                c.pos[i] = point[iret[j] * 3 + i] + pa[i];
            }
            c.depth = dep[iret[j]];
            out.push(c);
        }
    }
    Some((out, normal, depth, code))
}

fn collide_box_box(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    let (Shape::Box { side: s1 }, Shape::Box { side: s2 }) = (o1.shape, o2.shape) else { return Vec::new() };
    let Some((mut cs, normal, _, _)) = box_box(&o1.pr.pos, &o1.pr.r, s1, &o2.pr.pos, &o2.pr.r, s2, flags) else { return Vec::new() };
    for c in &mut cs {
        c.normal = neg3(&normal);
        c.g1 = o1.id;
        c.g2 = o2.id;
        c.side1 = -1;
        c.side2 = -1;
    }
    cs
}

fn collide_box_plane(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    let (Shape::Box { side }, Shape::Plane { p: n }) = (o1.shape, o2.shape) else { return Vec::new() };
    let r = &o1.pr.r;
    let pos = &o1.pr.pos;
    let q1 = dot3_14(n, &r[0..]);
    let q2 = dot3_14(n, &r[1..]);
    let q3 = dot3_14(n, &r[2..]);
    let aa = [side[0] * q1, side[1] * q2, side[2] * q3];
    let bb = [aa[0].abs(), aa[1].abs(), aa[2].abs()];
    let depth = n[3] + 0.5 * (bb[0] + bb[1] + bb[2]) - dot3(n, pos);
    if depth < 0.0 {
        return Vec::new();
    }
    let mut maxc = flags & NUMC_MASK;
    if maxc > 4 {
        maxc = 4;
    }
    let mut p = [pos[0], pos[1], pos[2], 0.0];
    for i in 0..3 {
        // BAR(i, i + 1): step from the centre to the deepest corner
        let k = 0.5 * side[i];
        if aa[i] > 0.0 {
            p[0] -= k * r[i];
            p[1] -= k * r[4 + i];
            p[2] -= k * r[8 + i];
        } else {
            p[0] += k * r[i];
            p[1] += k * r[4 + i];
            p[2] += k * r[8 + i];
        }
    }
    let mut cs: Vec<ContactGeom> = Vec::new();
    cs.push(ContactGeom { pos: p, depth, ..Default::default() });
    // BAR(ctact, side, sideinc) of the second macro set
    let push = |cs: &mut Vec<ContactGeom>, sd: usize| -> bool {
        if depth - bb[sd] < 0.0 {
            return false;
        }
        let mut c = ContactGeom::default();
        for k in 0..3 {
            c.pos[k] = if aa[sd] > 0.0 { p[k] + side[sd] * r[k * 4 + sd] } else { p[k] - side[sd] * r[k * 4 + sd] };
        }
        c.depth = depth - bb[sd];
        cs.push(c);
        true
    };
    'done: {
        if maxc == 1 {
            break 'done;
        }
        let (b1, b2, b3) = (bb[0], bb[1], bb[2]);
        let second: usize;
        if b1 < b2 {
            if b3 < b1 {
                if !push(&mut cs, 2) {
                    break 'done;
                }
                if maxc == 2 {
                    break 'done;
                }
                second = if b1 < b2 { 0 } else { 1 };
            } else {
                if !push(&mut cs, 0) {
                    break 'done;
                }
                if maxc == 2 {
                    break 'done;
                }
                second = if b2 < b3 { 1 } else { 2 };
            }
        } else if b3 < b2 {
            if !push(&mut cs, 2) {
                break 'done;
            }
            if maxc == 2 {
                break 'done;
            }
            second = if b1 < b2 { 0 } else { 1 };
        } else {
            if !push(&mut cs, 1) {
                break 'done;
            }
            if maxc == 2 {
                break 'done;
            }
            second = if b1 < b3 { 0 } else { 2 };
        }
        push(&mut cs, second);
    }
    if maxc == 4 && cs.len() == 3 {
        let d4 = cs[1].depth + cs[2].depth - depth;
        if d4 > 0.0 {
            let mut c = ContactGeom::default();
            for k in 0..3 {
                c.pos[k] = cs[1].pos[k] + cs[2].pos[k] - p[k];
            }
            c.depth = d4;
            cs.push(c);
        }
    }
    for c in &mut cs {
        c.g1 = o1.id;
        c.g2 = o2.id;
        c.side1 = -1;
        c.side2 = -1;
        c.normal = [n[0], n[1], n[2], 0.0];
    }
    cs
}

fn collide_capsule_sphere(o1: &Prim, o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    let Shape::Capsule { radius, lz } = *o1.shape else { return Vec::new() };
    let mut c = cg(o1, o2);
    let (p1, r1, p2) = (&o1.pr.pos, &o1.pr.r, &o2.pr.pos);
    let mut alpha = r1[2] * (p2[0] - p1[0]) + r1[6] * (p2[1] - p1[1]) + r1[10] * (p2[2] - p1[2]);
    let lz2 = lz * 0.5;
    if alpha > lz2 {
        alpha = lz2;
    }
    if alpha < -lz2 {
        alpha = -lz2;
    }
    let p = [p1[0] + alpha * r1[2], p1[1] + alpha * r1[6], p1[2] + alpha * r1[10]];
    let hit = collide_spheres(&p, radius, p2, radius_of(o2.shape), &mut c);
    one(c, hit)
}

/// dClosestLineBoxPoints
fn closest_line_box_points(p1: &[Real], p2: &[Real], c: &[Real], r: &[Real], side: &[Real]) -> (V3, V3) {
    let mut tmp = [p1[0] - c[0], p1[1] - c[1], p1[2] - c[2], 0.0];
    let mut s = mul1_331(r, &tmp);
    tmp = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2], 0.0];
    let mut v = mul1_331(r, &tmp);
    let mut sign = [0.0; 3];
    for i in 0..3 {
        if v[i] < 0.0 {
            s[i] = -s[i];
            v[i] = -v[i];
            sign[i] = -1.0;
        } else {
            sign[i] = 1.0;
        }
    }
    let v2 = [v[0] * v[0], v[1] * v[1], v[2] * v[2]];
    let h = [0.5 * side[0], 0.5 * side[1], 0.5 * side[2]];
    let mut region = [0i32; 3];
    let mut tanchor = [0.0; 3];
    let tanchor_eps = 1e-307;
    for i in 0..3 {
        if v[i] > tanchor_eps {
            if s[i] < -h[i] {
                region[i] = -1;
                tanchor[i] = (-h[i] - s[i]) / v[i];
            } else {
                region[i] = (s[i] > h[i]) as i32;
                tanchor[i] = (h[i] - s[i]) / v[i];
            }
        } else {
            region[i] = 0;
            tanchor[i] = 2.0;
        }
    }
    let mut t = 0.0;
    let mut dd2dt = 0.0;
    for i in 0..3 {
        dd2dt -= (if region[i] != 0 { v2[i] } else { 0.0 }) * tanchor[i];
    }
    'got: {
        if dd2dt >= 0.0 {
            break 'got;
        }
        loop {
            let mut next_t = 1.0;
            for i in 0..3 {
                if tanchor[i] > t && tanchor[i] < 1.0 && tanchor[i] < next_t {
                    next_t = tanchor[i];
                }
            }
            let mut next_dd2dt = 0.0;
            for i in 0..3 {
                next_dd2dt += (if region[i] != 0 { v2[i] } else { 0.0 }) * (next_t - tanchor[i]);
            }
            if next_dd2dt >= 0.0 {
                let m = (next_dd2dt - dd2dt) / (next_t - t);
                t -= dd2dt / m;
                break 'got;
            }
            for i in 0..3 {
                if tanchor[i] == next_t {
                    tanchor[i] = (h[i] - s[i]) / v[i];
                    region[i] += 1;
                }
            }
            t = next_t;
            dd2dt = next_dd2dt;
            if !(t < 1.0) {
                break;
            }
        }
        t = 1.0;
    }
    let lret = v3(p1[0] + t * tmp[0], p1[1] + t * tmp[1], p1[2] + t * tmp[2]);
    for i in 0..3 {
        tmp[i] = sign[i] * (s[i] + t * v[i]);
        if tmp[i] < -h[i] {
            tmp[i] = -h[i];
        } else if tmp[i] > h[i] {
            tmp[i] = h[i];
        }
    }
    let s = mul0_331(r, &tmp);
    let bret = v3(s[0] + c[0], s[1] + c[1], s[2] + c[2]);
    (lret, bret)
}

fn collide_capsule_box(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    let (Shape::Capsule { radius, lz }, Shape::Box { side }) = (o1.shape, o2.shape) else { return Vec::new() };
    let (pos1, r1) = (&o1.pr.pos, &o1.pr.r);
    let clen = lz * 0.5;
    let p1 = [pos1[0] + clen * r1[2], pos1[1] + clen * r1[6], pos1[2] + clen * r1[10]];
    let p2 = [pos1[0] - clen * r1[2], pos1[1] - clen * r1[6], pos1[2] - clen * r1[10]];
    let (c, r) = (&o2.pr.pos, &o2.pr.r);
    let (pl, pb) = closest_line_box_points(&p1, &p2, c, r, side);
    let mindist = 1e-15;
    if distance3(&pl, &pb) < mindist {
        let r2x = radius * 2.0;
        let capboxside = [r2x, r2x, lz + r2x];
        let Some((mut cs, normal, _, _)) = box_box(c, r, side, pos1, r1, &capboxside, flags) else { return Vec::new() };
        for cc in &mut cs {
            cc.normal = normal;
            cc.g1 = o1.id;
            cc.g2 = o2.id;
            cc.side1 = -1;
            cc.side2 = -1;
        }
        cs
    } else {
        let mut ct = cg(o1, o2);
        let hit = collide_spheres(&pl, *radius, &pb, 0.0, &mut ct);
        one(ct, hit)
    }
}

/// dClosestLineSegmentPoints
fn closest_line_segment_points(a1: &[Real], a2: &[Real], b1: &[Real], b2: &[Real]) -> (V3, V3) {
    let sub = |x: &[Real], y: &[Real]| -> V3 { [x[0] - y[0], x[1] - y[1], x[2] - y[2], 0.0] };
    let cp = |x: &[Real]| -> V3 { [x[0], x[1], x[2], 0.0] };
    let madd = |x: &[Real], k: Real, y: &[Real]| -> V3 { [x[0] + k * y[0], x[1] + k * y[1], x[2] + k * y[2], 0.0] };
    let msub = |x: &[Real], k: Real, y: &[Real]| -> V3 { [x[0] - k * y[0], x[1] - k * y[1], x[2] - k * y[2], 0.0] };
    let a1a2 = sub(a2, a1);
    let b1b2 = sub(b2, b1);
    let a1b1 = sub(b1, a1);
    let da1 = dot3(&a1a2, &a1b1);
    let db1 = dot3(&b1b2, &a1b1);
    if da1 <= 0.0 && db1 >= 0.0 {
        return (cp(a1), cp(b1));
    }
    let a1b2 = sub(b2, a1);
    let da2 = dot3(&a1a2, &a1b2);
    let db2 = dot3(&b1b2, &a1b2);
    if da2 <= 0.0 && db2 <= 0.0 {
        return (cp(a1), cp(b2));
    }
    let a2b1 = sub(b1, a2);
    let da3 = dot3(&a1a2, &a2b1);
    let db3 = dot3(&b1b2, &a2b1);
    if da3 >= 0.0 && db3 >= 0.0 {
        return (cp(a2), cp(b1));
    }
    let a2b2 = sub(b2, a2);
    let da4 = dot3(&a1a2, &a2b2);
    let db4 = dot3(&b1b2, &a2b2);
    if da4 >= 0.0 && db4 <= 0.0 {
        return (cp(a2), cp(b2));
    }
    let la = dot3(&a1a2, &a1a2);
    if da1 >= 0.0 && da3 <= 0.0 {
        let k = da1 / la;
        let n = msub(&a1b1, k, &a1a2);
        if dot3(&b1b2, &n) >= 0.0 {
            return (madd(a1, k, &a1a2), cp(b1));
        }
    }
    if da2 >= 0.0 && da4 <= 0.0 {
        let k = da2 / la;
        let n = msub(&a1b2, k, &a1a2);
        if dot3(&b1b2, &n) <= 0.0 {
            return (madd(a1, k, &a1a2), cp(b2));
        }
    }
    let lb = dot3(&b1b2, &b1b2);
    if db1 <= 0.0 && db2 >= 0.0 {
        let k = -db1 / lb;
        let n = msub(&neg3(&a1b1), k, &b1b2);
        if dot3(&a1a2, &n) >= 0.0 {
            return (cp(a1), madd(b1, k, &b1b2));
        }
    }
    if db3 <= 0.0 && db4 >= 0.0 {
        let k = -db3 / lb;
        let n = msub(&neg3(&a2b1), k, &b1b2);
        if dot3(&a1a2, &n) <= 0.0 {
            return (cp(a2), madd(b1, k, &b1b2));
        }
    }
    let k = dot3(&a1a2, &b1b2);
    let det = la * lb - k * k;
    if det <= 0.0 {
        return (cp(a1), cp(b1));
    }
    let det = recip(det);
    let alpha = (lb * da1 - k * db1) * det;
    let beta = (k * da1 - la * db1) * det;
    (madd(a1, alpha, &a1a2), madd(b1, beta, &b1b2))
}

fn collide_capsule_capsule(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    let (Shape::Capsule { radius: rad1, lz: l1 }, Shape::Capsule { radius: rad2, lz: l2 }) = (o1.shape, o2.shape) else { return Vec::new() };
    let tolerance = 1e-5;
    let mut c = cg(o1, o2);
    let lz1 = l1 * 0.5;
    let lz2 = l2 * 0.5;
    let (pos1, pos2) = (&o1.pr.pos, &o2.pr.pos);
    let axis1 = [o1.pr.r[2], o1.pr.r[6], o1.pr.r[10]];
    let mut axis2 = [o2.pr.r[2], o2.pr.r[6], o2.pr.r[10]];
    let a1a2 = dot3(&axis1, &axis2);
    let det = 1.0 - a1a2 * a1a2;
    if det < tolerance {
        if a1a2 < 0.0 {
            axis2 = [-axis2[0], -axis2[1], -axis2[2]];
        }
        let q = [pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]];
        let k = dot3(&axis1, &q);
        let a1lo = -lz1;
        let a1hi = lz1;
        let a2lo = -lz2 - k;
        let a2hi = lz2 - k;
        let lo = if a1lo > a2lo { a1lo } else { a2lo };
        let hi = if a1hi < a2hi { a1hi } else { a2hi };
        if lo <= hi {
            let num_contacts = flags & NUMC_MASK;
            if num_contacts >= 2 && lo < hi {
                let s1: Vec<Real> = (0..3).map(|i| pos1[i] + lo * axis1[i]).collect();
                let s2: Vec<Real> = (0..3).map(|i| pos2[i] + (lo + k) * axis2[i]).collect();
                let n1 = collide_spheres(&s1, *rad1, &s2, *rad2, &mut c);
                if n1 {
                    let s1: Vec<Real> = (0..3).map(|i| pos1[i] + hi * axis1[i]).collect();
                    let s2: Vec<Real> = (0..3).map(|i| pos2[i] + (hi + k) * axis2[i]).collect();
                    let mut c2 = cg(o1, o2);
                    if collide_spheres(&s1, *rad1, &s2, *rad2, &mut c2) {
                        return vec![c, c2];
                    }
                }
            }
            let alpha1 = (lo + hi) * 0.5;
            let alpha2 = alpha1 + k;
            let s1: Vec<Real> = (0..3).map(|i| pos1[i] + alpha1 * axis1[i]).collect();
            let s2: Vec<Real> = (0..3).map(|i| pos2[i] + alpha2 * axis2[i]).collect();
            let hit = collide_spheres(&s1, *rad1, &s2, *rad2, &mut c);
            return one(c, hit);
        }
    }
    let a1 = [pos1[0] + axis1[0] * lz1, pos1[1] + axis1[1] * lz1, pos1[2] + axis1[2] * lz1];
    let a2 = [pos1[0] - axis1[0] * lz1, pos1[1] - axis1[1] * lz1, pos1[2] - axis1[2] * lz1];
    let b1 = [pos2[0] + axis2[0] * lz2, pos2[1] + axis2[1] * lz2, pos2[2] + axis2[2] * lz2];
    let b2 = [pos2[0] - axis2[0] * lz2, pos2[1] - axis2[1] * lz2, pos2[2] - axis2[2] * lz2];
    let (s1, s2) = closest_line_segment_points(&a1, &a2, &b1, &b2);
    let hit = collide_spheres(&s1, *rad1, &s2, *rad2, &mut c);
    one(c, hit)
}

fn collide_capsule_plane(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    let (Shape::Capsule { radius, lz }, Shape::Plane { p: pl }) = (o1.shape, o2.shape) else { return Vec::new() };
    let (pos, r) = (&o1.pr.pos, &o1.pr.r);
    let sign = if dot3_14(pl, &r[2..]) > 0.0 { -1.0 } else { 1.0 };
    let mut p = [pos[0] + r[2] * lz * 0.5 * sign, pos[1] + r[6] * lz * 0.5 * sign, pos[2] + r[10] * lz * 0.5 * sign];
    let k = dot3(&p, pl);
    let depth = pl[3] - k + radius;
    if depth < 0.0 {
        return Vec::new();
    }
    let mut c = cg(o1, o2);
    c.normal = [pl[0], pl[1], pl[2], 0.0];
    c.pos = [p[0] - pl[0] * radius, p[1] - pl[1] * radius, p[2] - pl[2] * radius, 0.0];
    c.depth = depth;
    let mut out = vec![c];
    if flags & NUMC_MASK >= 2 {
        p = [pos[0] - r[2] * lz * 0.5 * sign, pos[1] - r[6] * lz * 0.5 * sign, pos[2] - r[10] * lz * 0.5 * sign];
        let k = dot3(&p, pl);
        let depth = pl[3] - k + radius;
        if depth >= 0.0 {
            let mut c2 = cg(o1, o2);
            c2.normal = [pl[0], pl[1], pl[2], 0.0];
            c2.pos = [p[0] - pl[0] * radius, p[1] - pl[1] * radius, p[2] - pl[2] * radius, 0.0];
            c2.depth = depth;
            out.push(c2);
        }
    }
    out
}

fn ray_len(s: &Shape) -> Real {
    match s {
        Shape::Ray { length } => *length,
        _ => 0.0,
    }
}

/// ray_sphere_helper
fn ray_sphere_helper(ray: &Prim, sphere_pos: &[Real], radius: Real, c: &mut ContactGeom, mode: bool) -> bool {
    let (rp, rr) = (&ray.pr.pos, &ray.pr.r);
    let q = [rp[0] - sphere_pos[0], rp[1] - sphere_pos[1], rp[2] - sphere_pos[2], 0.0];
    let b = dot3_14(&q, &rr[2..]);
    let cc = dot3(&q, &q) - radius * radius;
    let k = b * b - cc;
    if k < 0.0 {
        return false;
    }
    let k = k.sqrt();
    let mut alpha;
    if mode && cc >= 0.0 {
        alpha = -b + k;
        if alpha < 0.0 {
            return false;
        }
    } else {
        alpha = -b - k;
        if alpha < 0.0 {
            alpha = -b + k;
            if alpha < 0.0 {
                return false;
            }
        }
    }
    if alpha > ray_len(ray.shape) {
        return false;
    }
    c.pos = [rp[0] + alpha * rr[2], rp[1] + alpha * rr[6], rp[2] + alpha * rr[10], 0.0];
    let nsign = if cc < 0.0 || mode { -1.0 } else { 1.0 };
    c.normal = [nsign * (c.pos[0] - sphere_pos[0]), nsign * (c.pos[1] - sphere_pos[1]), nsign * (c.pos[2] - sphere_pos[2]), 0.0];
    normalize3(&mut c.normal);
    c.depth = alpha;
    true
}

fn collide_ray_sphere(o1: &Prim, o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    let mut c = cg(o1, o2);
    let hit = ray_sphere_helper(o1, &o2.pr.pos, radius_of(o2.shape), &mut c, false);
    one(c, hit)
}

fn collide_ray_box(o1: &Prim, o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    let Shape::Box { side } = o2.shape else { return Vec::new() };
    let mut c = cg(o1, o2);
    let (rp, rr) = (&o1.pr.pos, &o1.pr.r);
    let (bp, br) = (&o2.pr.pos, &o2.pr.r);
    let mut tmp = [rp[0] - bp[0], rp[1] - bp[1], rp[2] - bp[2], 0.0];
    let mut s = mul1_331(br, &tmp);
    tmp = [rr[2], rr[6], rr[10], 0.0];
    let mut v = mul1_331(br, &tmp);
    let mut sign = [0.0; 3];
    for i in 0..3 {
        if v[i] < 0.0 {
            s[i] = -s[i];
            v[i] = -v[i];
            sign[i] = 1.0;
        } else {
            sign[i] = -1.0;
        }
    }
    let h = [0.5 * side[0], 0.5 * side[1], 0.5 * side[2]];
    if (s[0] < -h[0] && v[0] <= 0.0) || s[0] > h[0] || (s[1] < -h[1] && v[1] <= 0.0) || s[1] > h[1] || (s[2] < -h[2] && v[2] <= 0.0) || s[2] > h[2] || (v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0) {
        return Vec::new();
    }
    let mut lo = -INFINITY;
    let mut hi = INFINITY;
    let (mut nlo, mut nhi) = (0, 0);
    for i in 0..3 {
        if v[i] != 0.0 {
            let k = (-h[i] - s[i]) / v[i];
            if k > lo {
                lo = k;
                nlo = i;
            }
            let k = (h[i] - s[i]) / v[i];
            if k < hi {
                hi = k;
                nhi = i;
            }
        }
    }
    if lo > hi {
        return Vec::new();
    }
    let (alpha, n) = if lo >= 0.0 { (lo, nlo) } else { (hi, nhi) };
    if alpha < 0.0 || alpha > ray_len(o1.shape) {
        return Vec::new();
    }
    c.pos = [rp[0] + alpha * rr[2], rp[1] + alpha * rr[6], rp[2] + alpha * rr[10], 0.0];
    c.normal = [br[n] * sign[n], br[4 + n] * sign[n], br[8 + n] * sign[n], 0.0];
    c.depth = alpha;
    vec![c]
}

fn collide_ray_capsule(o1: &Prim, o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    let Shape::Capsule { radius, lz } = *o2.shape else { return Vec::new() };
    let mut c = cg(o1, o2);
    let (rp, rr) = (&o1.pr.pos, &o1.pr.r);
    let (cp, cr) = (&o2.pr.pos, &o2.pr.r);
    let lz2 = lz * 0.5;
    let cs = [rp[0] - cp[0], rp[1] - cp[1], rp[2] - cp[2], 0.0];
    let mut k = dot3_41(&cr[2..], &cs);
    let mut q = [k * cr[2] - cs[0], k * cr[6] - cs[1], k * cr[10] - cs[2], 0.0];
    let cc = dot3(&q, &q) - radius * radius;
    let mut inside = false;
    if cc < 0.0 {
        if k < -lz2 {
            k = -lz2;
        } else if k > lz2 {
            k = lz2;
        }
        let r = [cp[0] + k * cr[2], cp[1] + k * cr[6], cp[2] + k * cr[10]];
        if (rp[0] - r[0]) * (rp[0] - r[0]) + (rp[1] - r[1]) * (rp[1] - r[1]) + (rp[2] - r[2]) * (rp[2] - r[2]) < radius * radius {
            inside = true;
        }
    }
    if !inside && cc < 0.0 {
        k = if k < 0.0 { -lz2 } else { lz2 };
    } else {
        let uv = dot3_44(&cr[2..], &rr[2..]);
        let r = [uv * cr[2] - rr[2], uv * cr[6] - rr[6], uv * cr[10] - rr[10], 0.0];
        let a = dot3(&r, &r);
        if a == 0.0 {
            k = if uv < 0.0 { -lz2 } else { lz2 };
        } else {
            let b = 2.0 * dot3(&q, &r);
            k = b * b - 4.0 * a * cc;
            if k < 0.0 {
                if !inside {
                    return Vec::new();
                }
                k = if uv < 0.0 { -lz2 } else { lz2 };
            } else {
                k = k.sqrt();
                let a = recip(2.0 * a);
                let mut alpha = (-b - k) * a;
                if alpha < 0.0 {
                    alpha = (-b + k) * a;
                    if alpha < 0.0 {
                        return Vec::new();
                    }
                }
                if alpha > ray_len(o1.shape) {
                    return Vec::new();
                }
                c.pos = [rp[0] + alpha * rr[2], rp[1] + alpha * rr[6], rp[2] + alpha * rr[10], 0.0];
                q = [c.pos[0] - cp[0], c.pos[1] - cp[1], c.pos[2] - cp[2], 0.0];
                k = dot3_14(&q, &cr[2..]);
                let nsign = if inside { -1.0 } else { 1.0 };
                if k >= -lz2 && k <= lz2 {
                    c.normal = [
                        nsign * (c.pos[0] - (cp[0] + k * cr[2])),
                        nsign * (c.pos[1] - (cp[1] + k * cr[6])),
                        nsign * (c.pos[2] - (cp[2] + k * cr[10])),
                        0.0,
                    ];
                    normalize3(&mut c.normal);
                    c.depth = alpha;
                    return vec![c];
                }
                k = if k < 0.0 { -lz2 } else { lz2 };
            }
        }
    }
    let q = [cp[0] + k * cr[2], cp[1] + k * cr[6], cp[2] + k * cr[10]];
    let hit = ray_sphere_helper(o1, &q, radius, &mut c, inside);
    one(c, hit)
}

fn collide_ray_plane(o1: &Prim, o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    let Shape::Plane { p } = o2.shape else { return Vec::new() };
    let (rp, rr) = (&o1.pr.pos, &o1.pr.r);
    let mut alpha = p[3] - dot3(p, rp);
    let nsign = if alpha > 0.0 { -1.0 } else { 1.0 };
    let k = dot3_14(p, &rr[2..]);
    if k == 0.0 {
        return Vec::new();
    }
    alpha /= k;
    if alpha < 0.0 || alpha > ray_len(o1.shape) {
        return Vec::new();
    }
    let mut c = cg(o1, o2);
    c.pos = [rp[0] + alpha * rr[2], rp[1] + alpha * rr[6], rp[2] + alpha * rr[10], 0.0];
    c.normal = [nsign * p[0], nsign * p[1], nsign * p[2], 0.0];
    c.depth = alpha;
    vec![c]
}

// ---- trimesh colliders (collision_trimesh_*.cpp)

fn mesh_of<'a>(s: &'a Shape) -> &'a TriMeshData {
    match s {
        Shape::TriMesh(d) => d,
        _ => panic!("not a trimesh"),
    }
}

/// dxTriMesh::fetchMeshTriangle: a triangle in world space (vertices pass
/// through OPCODE's float fetch first)
pub fn fetch_triangle(d: &TriMeshData, index: usize, pos: &[Real], r: &[Real]) -> [V3; 3] {
    let t = d.mesh.tri(index);
    let mut out = [[0.0; 4]; 3];
    for i in 0..3 {
        let v = [t[i][0] as Real, t[i][1] as Real, t[i][2] as Real, 0.0];
        let w = mul0_331(r, &v);
        out[i] = add3(&w, pos);
    }
    out
}

/// dCollideRTL
fn collide_ray_trimesh(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    let d = mesh_of(o1.shape);
    let length = ray_len(o2.shape);
    let mut col = opcode::RayCollider::new(&d.mesh);
    col.first_contact = o2.flags & RAY_FIRSTCONTACT != 0;
    col.closest_hit = o2.flags & RAY_CLOSEST_HIT != 0;
    col.culling = o2.flags & RAY_BACKFACECULL != 0;
    col.max_dist = length as f32;
    let (tlpos, tlrot) = (&o1.pr.pos, &o1.pr.r);
    let mesh_matrix = opcode::make_matrix(&[0.0; 3], tlrot);
    let origin = o2.pr.pos;
    let dir = [o2.pr.r[2], o2.pr.r[6], o2.pr.r[10]];
    let off = sub3(&origin, tlpos);
    col.collide([off[0] as f32, off[1] as f32, off[2] as f32], [dir[0] as f32, dir[1] as f32, dir[2] as f32], &d.model, &mesh_matrix);
    let mut out = Vec::new();
    for f in &col.faces {
        let tri = f.face_id as usize;
        let dv = fetch_triangle(d, tri, tlpos, tlrot);
        let vu = sub3(&dv[1], &dv[0]);
        let vv = sub3(&dv[2], &dv[0]);
        let mut n = cross(&vv, &vu);
        if safe_normalize3(&mut n) {
            let t = f.distance as Real;
            let mut c = ContactGeom { g1: o1.id, g2: o2.id, side1: tri as i32, side2: -1, ..Default::default() };
            c.normal = n;
            c.pos = [origin[0] + dir[0] * t, origin[1] + dir[1] * t, origin[2] + dir[2] * t, 0.0];
            c.depth = t;
            out.push(c);
            if out.len() as i32 >= flags & NUMC_MASK {
                break;
            }
        }
    }
    out
}

/// GetContactData (collision_trimesh_sphere.cpp): sphere vs triangle by
/// Eberly's closest point; returns (depth, u, v) when within the radius
fn sphere_tri_contact(center: &[Real], radius: Real, origin: &[Real], e0: &[Real], e1: &[Real]) -> Option<(Real, Real, Real)> {
    let diff = [origin[0] - center[0], origin[1] - center[1], origin[2] - center[2]];
    let a00 = dot3(e0, e0);
    let a01 = dot3(e0, e1);
    let a11 = dot3(e1, e1);
    let b0 = dot3(&diff, e0);
    let b1 = dot3(&diff, e1);
    let c = dot3(&diff, &diff);
    let det = (a00 * a11 - a01 * a01).abs();
    let mut u = a01 * b1 - a11 * b0;
    let mut v = a01 * b0 - a00 * b1;
    let dist_sq;
    if u + v <= det {
        if u < 0.0 {
            if v < 0.0 {
                if b0 < 0.0 {
                    v = 0.0;
                    if -b0 >= a00 {
                        u = 1.0;
                        dist_sq = a00 + 2.0 * b0 + c;
                    } else {
                        u = -b0 / a00;
                        dist_sq = b0 * u + c;
                    }
                } else {
                    u = 0.0;
                    if b1 >= 0.0 {
                        v = 0.0;
                        dist_sq = c;
                    } else if -b1 >= a11 {
                        v = 1.0;
                        dist_sq = a11 + 2.0 * b1 + c;
                    } else {
                        v = -b1 / a11;
                        dist_sq = b1 * v + c;
                    }
                }
            } else {
                u = 0.0;
                if b1 >= 0.0 {
                    v = 0.0;
                    dist_sq = c;
                } else if -b1 >= a11 {
                    v = 1.0;
                    dist_sq = a11 + 2.0 * b1 + c;
                } else {
                    v = -b1 / a11;
                    dist_sq = b1 * v + c;
                }
            }
        } else if v < 0.0 {
            v = 0.0;
            if b0 >= 0.0 {
                u = 0.0;
                dist_sq = c;
            } else if -b0 >= a00 {
                u = 1.0;
                dist_sq = a00 + 2.0 * b0 + c;
            } else {
                u = -b0 / a00;
                dist_sq = b0 * u + c;
            }
        } else if det == 0.0 {
            u = 0.0;
            v = 0.0;
            dist_sq = f32::MAX as Real;
        } else {
            let inv = 1.0 / det;
            u *= inv;
            v *= inv;
            dist_sq = u * (a00 * u + a01 * v + 2.0 * b0) + v * (a01 * u + a11 * v + 2.0 * b1) + c;
        }
    } else if u < 0.0 {
        let t0 = a01 + b0;
        let t1 = a11 + b1;
        if t1 > t0 {
            let num = t1 - t0;
            let den = a00 - 2.0 * a01 + a11;
            if num >= den {
                u = 1.0;
                v = 0.0;
                dist_sq = a00 + 2.0 * b0 + c;
            } else {
                u = num / den;
                v = 1.0 - u;
                dist_sq = u * (a00 * u + a01 * v + 2.0 * b0) + v * (a01 * u + a11 * v + 2.0 * b1) + c;
            }
        } else {
            u = 0.0;
            if t1 <= 0.0 {
                v = 1.0;
                dist_sq = a11 + 2.0 * b1 + c;
            } else if b1 >= 0.0 {
                v = 0.0;
                dist_sq = c;
            } else {
                v = -b1 / a11;
                dist_sq = b1 * v + c;
            }
        }
    } else if v < 0.0 {
        let t0 = a01 + b1;
        let t1 = a00 + b0;
        if t1 > t0 {
            let num = t1 - t0;
            let den = a00 - 2.0 * a01 + a11;
            if num >= den {
                v = 1.0;
                u = 0.0;
                dist_sq = a11 + 2.0 * b1 + c;
            } else {
                v = num / den;
                u = 1.0 - v;
                dist_sq = u * (a00 * u + a01 * v + 2.0 * b0) + v * (a01 * u + a11 * v + 2.0 * b1) + c;
            }
        } else {
            v = 0.0;
            if t1 <= 0.0 {
                u = 1.0;
                dist_sq = a00 + 2.0 * b0 + c;
            } else if b0 >= 0.0 {
                u = 0.0;
                dist_sq = c;
            } else {
                u = -b0 / a00;
                dist_sq = b0 * u + c;
            }
        }
    } else {
        let num = a11 + b1 - a01 - b0;
        if num <= 0.0 {
            u = 0.0;
            v = 1.0;
            dist_sq = a11 + 2.0 * b1 + c;
        } else {
            let den = a00 - 2.0 * a01 + a11;
            if num >= den {
                u = 1.0;
                v = 0.0;
                dist_sq = a00 + 2.0 * b0 + c;
            } else {
                u = num / den;
                v = 1.0 - u;
                dist_sq = u * (a00 * u + a01 * v + 2.0 * b0) + v * (a01 * u + a11 * v + 2.0 * b1) + c;
            }
        }
    }
    let dist = dist_sq.abs().sqrt();
    if dist <= radius {
        Some((radius - dist, u, v))
    } else {
        None
    }
}

/// dCollideSTL (contacts not merged: MERGE_NORMALS__SPHERE_DEFAULT)
fn collide_sphere_trimesh(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    use super::volume::{SphereVol, VolumeQuery};
    let d = mesh_of(o1.shape);
    let (tlpos, tlrot) = (&o1.pr.pos, &o1.pr.r);
    let mesh_matrix = opcode::make_matrix(&[0.0; 3], tlrot);
    let position = o2.pr.pos;
    let radius = radius_of(o2.shape);
    let off = sub3(&position, tlpos);
    let vol = SphereVol::new([off[0] as f32, off[1] as f32, off[2] as f32], radius as f32, &mesh_matrix);
    let mut q = VolumeQuery::new(vol, &d.mesh);
    q.run(&d.model);
    let mut out = Vec::new();
    if !q.contact {
        return out;
    }
    for &tri in &q.touched {
        if out.len() as i32 == flags & NUMC_MASK {
            break;
        }
        let dv = fetch_triangle(d, tri as usize, tlpos, tlrot);
        let (v0, v1, v2) = (&dv[0], &dv[1], &dv[2]);
        let vu = sub3(v1, v0);
        let vv = sub3(v2, v0);
        let mut plane = cross(&vu, &vv);
        if !safe_normalize3(&mut plane) {
            continue;
        }
        let side = dot3(&plane, &position) - dot3(&plane, v0);
        if side < 0.0 {
            continue;
        }
        let Some((depth, u, v)) = sphere_tri_contact(&position, radius, v0, &vu, &vv) else { continue };
        if depth < 0.0 {
            continue;
        }
        let w = 1.0 - u - v;
        let cp = [(v0[0] * w) + (v1[0] * u) + (v2[0] * v), (v0[1] * w) + (v1[1] * u) + (v2[1] * v), (v0[2] * w) + (v1[2] * u) + (v2[2] * v), 0.0];
        let dir = sub3(&position, &cp);
        let dir_proj = dot3(&dir, &plane) / dot3(&dir, &dir).sqrt();
        if dir_proj < 0.0 {
            continue;
        }
        let mut c = ContactGeom { g1: o1.id, g2: o2.id, side1: tri as i32, side2: -1, ..Default::default() };
        c.pos = cp;
        c.normal = [-plane[0], -plane[1], -plane[2], 0.0];
        c.depth = depth * dir_proj;
        out.push(c);
    }
    out
}

// ---- dCollideBTL (collision_trimesh_box.cpp)

/// sTrimeshBoxColliderData
struct Btl {
    rot: M3,
    pos: V3,
    half: V3,
    best_normal: V3,
    best_depth: Real,
    best_axis: i32,
    e0: V3,
    e1: V3,
    e2: V3,
    n: V3,
    flags: i32,
    g1: GeomId,
    g2: GeomId,
    contacts: Vec<ContactGeom>,
}

/// dPointPlaneDistance
fn plane_dist(p: &[Real], pl: &[Real; 4]) -> Real {
    pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] + pl[3]
}
fn mk_plane(n: &[Real], d: Real) -> [Real; 4] {
    [n[0], n[1], n[2], d]
}

/// _cldClipPolyToPlane
fn clip_poly_to_plane(input: &[V3], out: &mut Vec<V3>, pl: &[Real; 4]) {
    out.clear();
    let n = input.len();
    if n == 0 {
        return;
    }
    let mut i0 = n - 1;
    for i1 in 0..n {
        let d0 = plane_dist(&input[i0], pl);
        let d1 = plane_dist(&input[i1], pl);
        if d0 >= 0.0 {
            out.push([input[i0][0], input[i0][1], input[i0][2], 0.0]);
        }
        if (d0 > 0.0 && d1 < 0.0) || (d0 < 0.0 && d1 > 0.0) {
            let a = &input[i0];
            let b = &input[i1];
            out.push([
                a[0] - (a[0] - b[0]) * d0 / (d0 - d1),
                a[1] - (a[1] - b[1]) * d0 / (d0 - d1),
                a[2] - (a[2] - b[2]) * d0 / (d0 - d1),
                0.0,
            ]);
        }
        i0 = i1;
    }
}

impl Btl {
    fn test_normal(&mut self, fp0: Real, fr: Real, n: &V3, axis: i32) -> bool {
        let depth = fr + fp0;
        if depth < 0.0 {
            return false;
        }
        let len = length3(n);
        if len > 0.0 {
            let inv = 1.0 / len;
            let depth = depth * inv;
            if depth < self.best_depth {
                self.best_normal = [-n[0] * inv, -n[1] * inv, -n[2] * inv, 0.0];
                self.best_axis = axis;
                self.best_depth = depth;
            }
        }
        true
    }
    fn test_face(&mut self, fp0: Real, fp1: Real, fp2: Real, fr: Real, n: &mut V3, axis: i32) -> bool {
        let fmin = if fp0 < fp1 {
            if fp0 < fp2 {
                fp0
            } else {
                fp2
            }
        } else if fp1 < fp2 {
            fp1
        } else {
            fp2
        };
        let fmax = if fp0 > fp1 {
            if fp0 > fp2 {
                fp0
            } else {
                fp2
            }
        } else if fp1 > fp2 {
            fp1
        } else {
            fp2
        };
        let dmin = fr - fmin;
        let dmax = fmax + fr;
        if dmin < 0.0 || dmax < 0.0 {
            return false;
        }
        let depth;
        if dmin > dmax {
            depth = dmax;
            *n = neg3(n);
        } else {
            depth = dmin;
        }
        if depth < self.best_depth {
            self.best_normal = [n[0], n[1], n[2], 0.0];
            self.best_axis = axis;
            self.best_depth = depth;
        }
        true
    }
    fn test_edge(&mut self, fp0: Real, fp1: Real, fr: Real, n: &mut V3, axis: i32) -> bool {
        let m = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
        if m <= f64::EPSILON {
            return true;
        }
        let (fmin, fmax) = if fp0 < fp1 { (fp0, fp1) } else { (fp1, fp0) };
        let dmin = fr - fmin;
        let dmax = fmax + fr;
        if dmin < 0.0 || dmax < 0.0 {
            return false;
        }
        let mut depth;
        if dmin > dmax {
            depth = dmax;
            *n = neg3(n);
        } else {
            depth = dmin;
        }
        let len = length3(n);
        if len > 0.0 {
            let inv = 1.0 / len;
            depth *= inv;
            if depth * 1.5 < self.best_depth {
                self.best_normal = [n[0] * inv, n[1] * inv, n[2] * inv, 0.0];
                self.best_axis = axis;
                self.best_depth = depth;
            }
        }
        true
    }

    /// _cldTestSeparatingAxes
    fn separating_axes(&mut self, v0: &V3, v1: &V3, v2: &V3) -> bool {
        self.best_axis = 0;
        self.best_depth = f64::MAX;
        self.e0 = sub3(v1, v0);
        self.e1 = sub3(v2, v0);
        self.e2 = sub3(&self.e1, &self.e0);
        self.n = cross(&self.e0, &self.e1);
        let nlen = length3(&self.n);
        if nlen == 0.0 {
            return false;
        }
        let a0 = get_column3(&self.rot, 0);
        let a1 = get_column3(&self.rot, 1);
        let a2 = get_column3(&self.rot, 2);
        let (fa0, fa1, fa2) = (self.half[0], self.half[1], self.half[2]);
        let vd = sub3(v0, &self.pos);
        let (e0, e1, e2, n) = (self.e0, self.e1, self.e2, self.n);

        let vl = n;
        let fp0 = dot3(&vl, &vd);
        let fr = fa0 * dot3(&n, &a0).abs() + fa1 * dot3(&n, &a1).abs() + fa2 * dot3(&n, &a2).abs();
        if !self.test_normal(fp0, fr, &vl, 1) {
            return false;
        }
        for (k, (a, fr)) in [(a0, fa0), (a1, fa1), (a2, fa2)].into_iter().enumerate() {
            let mut vl = a;
            let fp0 = dot3(&vl, &vd);
            let fp1 = fp0 + dot3(&a, &e0);
            let fp2 = fp0 + dot3(&a, &e1);
            if !self.test_face(fp0, fp1, fp2, fr, &mut vl, 2 + k as i32) {
                return false;
            }
        }
        // edges: box axis x triangle edge
        let edge_tests: [(V3, V3, Real, Real, V3, V3); 3] = [(a0, a1, fa1, fa2, a2, a1), (a1, a0, fa0, fa2, a2, a0), (a2, a0, fa0, fa1, a1, a0)];
        let mut axis = 5;
        for &(a, _, fx, fy, pa, pb) in &edge_tests {
            // vs e0
            let mut vl = cross(&a, &e0);
            let fp0 = dot3(&vl, &vd);
            let fp2 = fp0 + dot3(&a, &n);
            let fr = fx * dot3(&pa, &e0).abs() + fy * dot3(&pb, &e0).abs();
            // ODE passes (fp1 = fp0, fp2) for the first two box axes' e0 edge
            if !self.test_edge(fp0, fp2, fr, &mut vl, axis) {
                return false;
            }
            axis += 1;
            // vs e1
            let mut vl = cross(&a, &e1);
            let fp0 = dot3(&vl, &vd);
            let fp1 = fp0 - dot3(&a, &n);
            let fr = fx * dot3(&pa, &e1).abs() + fy * dot3(&pb, &e1).abs();
            if !self.test_edge(fp0, fp1, fr, &mut vl, axis) {
                return false;
            }
            axis += 1;
            // vs e2
            let mut vl = cross(&a, &e2);
            let fp0 = dot3(&vl, &vd);
            let fp1 = fp0 - dot3(&a, &n);
            let fr = fx * dot3(&pa, &e2).abs() + fy * dot3(&pb, &e2).abs();
            if !self.test_edge(fp0, fp1, fr, &mut vl, axis) {
                return false;
            }
            axis += 1;
        }
        true
    }

    /// GenerateContact: add, merge with a duplicate, or replace the
    /// shallowest when full
    fn generate(&mut self, tri: i32, pos: &V3, normal: &V3, depth: Real) {
        let maxc = (self.flags & NUMC_MASK) as usize;
        let mut target: Option<usize> = None;
        if self.flags & CONTACTS_UNIMPORTANT == 0 {
            let mut min_depth = INFINITY;
            let mut min_contact: Option<usize> = None;
            for i in 0..self.contacts.len() {
                let c = &mut self.contacts[i];
                let diff = sub3(pos, &c.pos);
                if dot3(&diff, &diff) < f64::EPSILON && 1.0 - dot3(normal, &c.normal) < f64::EPSILON {
                    if depth > c.depth {
                        c.depth = depth;
                        c.side1 = tri;
                    }
                    return;
                }
                if c.depth < min_depth {
                    min_depth = c.depth;
                    min_contact = Some(i);
                }
            }
            if self.contacts.len() == maxc {
                if !(min_depth < depth) {
                    return;
                }
                target = min_contact;
            }
        }
        let idx = match target {
            Some(i) => i,
            None => {
                self.contacts.push(ContactGeom { g1: self.g1, g2: self.g2, side2: -1, ..Default::default() });
                self.contacts.len() - 1
            }
        };
        let c = &mut self.contacts[idx];
        c.pos = [pos[0], pos[1], pos[2], 0.0];
        c.normal = [normal[0], normal[1], normal[2], 0.0];
        c.depth = depth;
        c.side1 = tri;
    }
    fn full(&self) -> bool {
        ((self.contacts.len() as i32) | CONTACTS_UNIMPORTANT) == (self.flags & (NUMC_MASK | CONTACTS_UNIMPORTANT))
    }

    /// _cldClipping
    fn clipping(&mut self, v0: &V3, v1: &V3, v2: &V3, tri: i32) {
        if self.best_axis > 4 {
            let mut pa = self.pos;
            for i in 0..3 {
                let col = get_column3(&self.rot, i);
                let sign = if dot3(&self.best_normal, &col) > 0.0 { 1.0 } else { -1.0 };
                pa[0] += sign * self.half[i] * col[0];
                pa[1] += sign * self.half[i] * col[1];
                pa[2] += sign * self.half[i] * col[2];
            }
            let edge = (self.best_axis - 5) % 3;
            let (mut pb, mut ub) = match edge {
                0 => (*v0, self.e0),
                1 => (*v2, self.e1),
                _ => (*v1, self.e2),
            };
            normalize3(&mut ub);
            let col = ((self.best_axis - 5) / 3) as usize;
            let ua = get_column3(&self.rot, col);
            // _cldClosestPointOnTwoLines
            let vp = sub3(&pb, &pa);
            let uaub = dot3(&ua, &ub);
            let q1 = dot3(&ua, &vp);
            let q2 = -dot3(&ub, &vp);
            let mut d = 1.0 - uaub * uaub;
            let (p1, p2) = if d > 0.0 {
                d = 1.0 / d;
                ((q1 + uaub * q2) * d, (uaub * q1 + q2) * d)
            } else {
                (0.0, 0.0)
            };
            for k in 0..3 {
                pa[k] += ua[k] * p1;
            }
            for k in 0..3 {
                pb[k] += ub[k] * p2;
            }
            let mut p = add3(&pa, &pb);
            p[0] *= 0.5;
            p[1] *= 0.5;
            p[2] *= 0.5;
            let (bn, bd) = (self.best_normal, self.best_depth);
            self.generate(tri, &p, &bn, bd);
        } else if self.best_axis == 1 {
            let n2 = neg3(&self.best_normal);
            let m = &self.rot;
            let nr = [
                m[0] * n2[0] + m[4] * n2[1] + m[8] * n2[2],
                m[1] * n2[0] + m[5] * n2[1] + m[9] * n2[2],
                m[2] * n2[0] + m[6] * n2[1] + m[10] * n2[2],
            ];
            let an = [nr[0].abs(), nr[1].abs(), nr[2].abs()];
            let (b0, b1, b2) = if an[1] > an[0] {
                if an[1] > an[2] {
                    (1, 0, 2)
                } else {
                    (2, 0, 1)
                }
            } else if an[0] > an[2] {
                (0, 1, 2)
            } else {
                (2, 0, 1)
            };
            let rc = get_column3(m, b0);
            let mut center = [0.0; 4];
            for k in 0..3 {
                center[k] = if nr[b0] > 0.0 { self.pos[k] - v0[k] - self.half[b0] * rc[k] } else { self.pos[k] - v0[k] + self.half[b0] * rc[k] };
            }
            let r1 = get_column3(m, b1);
            let r2 = get_column3(m, b2);
            let mut pts = [[0.0; 4]; 4];
            for x in 0..3 {
                pts[0][x] = center[x] + (self.half[b1] * r1[x]) - (self.half[b2] * r2[x]);
                pts[1][x] = center[x] - (self.half[b1] * r1[x]) - (self.half[b2] * r2[x]);
                pts[2][x] = center[x] - (self.half[b1] * r1[x]) + (self.half[b2] * r2[x]);
                pts[3][x] = center[x] + (self.half[b1] * r1[x]) + (self.half[b2] * r2[x]);
            }
            let mut t1 = Vec::with_capacity(9);
            let mut t2 = Vec::with_capacity(9);
            let mut vt = neg3(&self.n);
            normalize3(&mut vt);
            clip_poly_to_plane(&pts, &mut t1, &mk_plane(&vt, 0.0));
            let tmp2 = sub3(v1, v0);
            let mut vt = cross(&self.n, &tmp2);
            normalize3(&mut vt);
            clip_poly_to_plane(&t1, &mut t2, &mk_plane(&vt, 0.0));
            let tmp2 = sub3(v2, v1);
            let mut vt = cross(&self.n, &tmp2);
            normalize3(&mut vt);
            let tmp2 = sub3(v0, v2);
            clip_poly_to_plane(&t2, &mut t1, &mk_plane(&vt, dot3(&tmp2, &vt)));
            let tmp2 = sub3(v0, v2);
            let mut vt = cross(&self.n, &tmp2);
            normalize3(&mut vt);
            clip_poly_to_plane(&t1, &mut t2, &mk_plane(&vt, 0.0));
            for p in &t2 {
                let mut d = dot3(&n2, p);
                if d > 0.0 {
                    d = 0.0;
                }
                let pt = add3(p, v0);
                let bn = self.best_normal;
                self.generate(tri, &pt, &bn, -d);
                if self.full() {
                    break;
                }
            }
        } else {
            let n2 = self.best_normal;
            let a0 = (self.best_axis - 2) as usize;
            let (a1, a2) = match a0 {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            let pts = [sub3(v0, &self.pos), sub3(v1, &self.pos), sub3(v2, &self.pos)];
            let mut t1 = Vec::with_capacity(9);
            let mut t2 = Vec::with_capacity(9);
            let vt = neg3(&n2);
            clip_poly_to_plane(&pts, &mut t1, &mk_plane(&vt, self.half[a0]));
            let vt = get_column3(&self.rot, a1);
            clip_poly_to_plane(&t1, &mut t2, &mk_plane(&vt, self.half[a1]));
            let vt = neg3(&get_column3(&self.rot, a1));
            clip_poly_to_plane(&t2, &mut t1, &mk_plane(&vt, self.half[a1]));
            let vt = get_column3(&self.rot, a2);
            clip_poly_to_plane(&t1, &mut t2, &mk_plane(&vt, self.half[a2]));
            let vt = neg3(&get_column3(&self.rot, a2));
            clip_poly_to_plane(&t2, &mut t1, &mk_plane(&vt, self.half[a2]));
            for p in &t1 {
                let mut d = dot3(&n2, p) - self.half[a0];
                if d > 0.0 {
                    d = 0.0;
                }
                let pt = add3(p, &self.pos);
                let bn = self.best_normal;
                self.generate(tri, &pt, &bn, -d);
                if self.full() {
                    break;
                }
            }
        }
    }
}

/// dCollideBTL
fn collide_box_trimesh(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    use super::volume::{ObbVol, VolumeQuery};
    let d = mesh_of(o1.shape);
    let Shape::Box { side } = o2.shape else { return Vec::new() };
    let (mesh_pos, mesh_rot) = (&o1.pr.pos, &o1.pr.r);
    let (box_pos, box_rot) = (&o2.pr.pos, &o2.pr.r);
    let mut c = Btl {
        rot: *box_rot,
        pos: *box_pos,
        half: [side[0] * 0.5, side[1] * 0.5, side[2] * 0.5, side[3]],
        best_normal: [0.0; 4],
        best_depth: f64::MAX,
        best_axis: 0,
        e0: [0.0; 4],
        e1: [0.0; 4],
        e2: [0.0; 4],
        n: [0.0; 4],
        flags,
        g1: o1.id,
        g2: o2.id,
        contacts: Vec::new(),
    };
    let mesh_matrix = opcode::make_matrix(&[0.0; 3], mesh_rot);
    let off = sub3(box_pos, mesh_pos);
    let r = box_rot;
    let rot = [[r[0] as f32, r[4] as f32, r[8] as f32], [r[1] as f32, r[5] as f32, r[9] as f32], [r[2] as f32, r[6] as f32, r[10] as f32]];
    let vol = ObbVol::new([off[0] as f32, off[1] as f32, off[2] as f32], [c.half[0] as f32, c.half[1] as f32, c.half[2] as f32], rot, &mesh_matrix);
    let mut q = VolumeQuery::new(vol, &d.mesh);
    q.run(&d.model);
    if !q.contact {
        return Vec::new();
    }
    for &tri in &q.touched {
        let dv = fetch_triangle(d, tri as usize, mesh_pos, mesh_rot);
        if c.separating_axes(&dv[0], &dv[1], &dv[2]) && c.best_axis != 0 {
            c.clipping(&dv[0], &dv[1], &dv[2], tri as i32);
            if c.full() {
                break;
            }
        }
    }
    c.contacts
}

// ---- dCollideCCTL (collision_trimesh_ccylinder.cpp)

#[derive(Clone, Copy)]
struct LocalContact {
    pos: V3,
    normal: V3,
    depth: Real,
    tri: i32,
    keep: bool,
}

/// sTrimeshCapsuleColliderData (no edge/vertex use flags: every feature
/// is tested, as ODE does for meshes that were not preprocessed)
struct Cctl {
    pos: V3,
    axis: V3,
    radius: Real,
    size: Real,
    e0: V3,
    e1: V3,
    e2: V3,
    normal: V3,
    best_depth: Real,
    best_center: Real,
    best_rt: Real,
    best_axis: i32,
    n: V3,
    v0: V3,
    v1: V3,
    v2: V3,
    flags: i32,
    local: Vec<LocalContact>,
}

/// _cldClipEdgeToPlane
fn clip_edge_to_plane(p0: &mut V3, p1: &mut V3, pl: &[Real; 4]) -> bool {
    let d0 = plane_dist(p0, pl);
    let d1 = plane_dist(p1, pl);
    if d0 < 0.0 && d1 < 0.0 {
        false
    } else if d0 > 0.0 && d1 > 0.0 {
        true
    } else if (d0 > 0.0 && d1 < 0.0) || (d0 < 0.0 && d1 > 0.0) {
        let ip = [
            p0[0] - (p0[0] - p1[0]) * d0 / (d0 - d1),
            p0[1] - (p0[1] - p1[1]) * d0 / (d0 - d1),
            p0[2] - (p0[2] - p1[2]) * d0 / (d0 - d1),
            0.0,
        ];
        if d0 < 0.0 {
            p0[0] = ip[0];
            p0[1] = ip[1];
            p0[2] = ip[2];
        } else {
            p1[0] = ip[0];
            p1[1] = ip[1];
            p1[2] = ip[2];
        }
        true
    } else {
        true
    }
}

/// _CalculateAxis: ((v1 - v2) x v3) x v4
fn calculate_axis(v1: &V3, v2: &V3, v3: &V3, v4: &V3) -> V3 {
    let t1 = sub3(v1, v2);
    let t2 = cross(&t1, v3);
    cross(&t2, v4)
}

impl Cctl {
    fn half_len(&self) -> Real {
        self.size * 0.5 - self.radius
    }
    /// _cldTestAxis
    fn test_axis(&mut self, axis: &mut V3, iaxis: i32, no_flip: bool) -> bool {
        let fl = length3(axis);
        if fl < 1e-5 {
            return true;
        }
        normalize3(axis);
        let frc = dot3(&self.axis, axis).abs() * (self.size * 0.5 - self.radius) + self.radius;
        let afv = [dot3(&self.v0, axis), dot3(&self.v1, axis), dot3(&self.v2, axis)];
        let mut fmin = f64::MAX;
        let mut fmax = -f64::MAX;
        for &a in &afv {
            if a < fmin {
                fmin = a;
            }
            if a > fmax {
                fmax = a;
            }
        }
        let center = (fmin + fmax) * 0.5;
        let tri_r = (fmax - fmin) * 0.5;
        if center.abs() > frc + tri_r {
            return false;
        }
        let depth = center.abs() - (frc + tri_r);
        if depth > self.best_depth {
            self.best_depth = depth;
            self.best_center = center;
            self.best_rt = tri_r;
            self.normal = [axis[0], axis[1], axis[2], 0.0];
            self.best_axis = iaxis;
            if center < 0.0 && !no_flip {
                self.normal = neg3(&self.normal);
                self.best_center = -center;
            }
        }
        true
    }

    /// _cldTestSeparatingAxesOfCapsule
    fn separating_axes(&mut self, v0: &V3, v1: &V3, v2: &V3) -> bool {
        let h = self.half_len();
        let (p, a) = (self.pos, self.axis);
        let cp0 = [p[0] + a[0] * h, p[1] + a[1] * h, p[2] + a[2] * h, 0.0];
        let cp1 = [p[0] - a[0] * h, p[1] - a[1] * h, p[2] - a[2] * h, 0.0];
        self.best_axis = 0;
        self.best_depth = -f64::MAX;
        let eps = 1e-6f32 as Real;
        self.v0 = sub3(v0, &p);
        self.v1 = sub3(v1, &p);
        self.v2 = sub3(v2, &p);
        let mut ax = neg3(&self.n);
        if !self.test_axis(&mut ax, 1, true) {
            return false;
        }
        let (e0, e1, e2) = (self.e0, self.e1, self.e2);
        let mut cands: Vec<V3> = Vec::with_capacity(18);
        cands.push(cross(&a, &e0));
        cands.push(cross(&a, &e1));
        cands.push(cross(&a, &e2));
        cands.push(calculate_axis(&cp0, v0, &e0, &e0));
        cands.push(calculate_axis(&cp0, v1, &e1, &e1));
        cands.push(calculate_axis(&cp0, v2, &e2, &e2));
        cands.push(calculate_axis(&cp1, v0, &e0, &e0));
        cands.push(calculate_axis(&cp1, v1, &e1, &e1));
        cands.push(calculate_axis(&cp1, v2, &e2, &e2));
        cands.push(calculate_axis(v0, &cp0, &a, &a));
        cands.push(calculate_axis(v1, &cp0, &a, &a));
        cands.push(calculate_axis(v2, &cp0, &a, &a));
        cands.push(sub3(v0, &cp0));
        cands.push(sub3(v1, &cp0));
        cands.push(sub3(v2, &cp0));
        cands.push(sub3(v0, &cp1));
        cands.push(sub3(v1, &cp1));
        cands.push(sub3(v2, &cp1));
        for (k, mut ax) in cands.into_iter().enumerate() {
            if length_sq3(&ax) > eps && !self.test_axis(&mut ax, 2 + k as i32, false) {
                return false;
            }
        }
        true
    }

    /// _cldTestOneTriangleVSCapsule
    fn test_triangle(&mut self, v0: &V3, v1: &V3, v2: &V3, tri: i32) {
        let first_new = self.local.len();
        self.test_triangle_inner(v0, v1, v2);
        for c in &mut self.local[first_new..] {
            c.tri = tri;
        }
    }
    fn test_triangle_inner(&mut self, v0: &V3, v1: &V3, v2: &V3) {
        self.e0 = sub3(v1, v0);
        self.e1 = sub3(v2, v1);
        self.e2 = sub3(v0, v2);
        let minus_e0 = sub3(v0, v1);
        self.n = cross(&self.e1, &minus_e0);
        if !safe_normalize3(&mut self.n) {
            return;
        }
        let pl_d = -dot3(v0, &self.n);
        let tri_plane = mk_plane(&self.n, pl_d);
        let dist = plane_dist(&self.pos, &tri_plane);
        if dist < 0.0 {
            return;
        }
        let (p0, p1, p2) = (*v0, *v1, *v2);
        if !self.separating_axes(&p0, &p1, &p2) {
            return;
        }
        if self.best_axis == 0 {
            return;
        }
        let h = self.half_len();
        let (p, a, nrm, r) = (self.pos, self.axis, self.normal, self.radius);
        let ct = [p[0] + nrm[0] * r, p[1] + nrm[1] * r, p[2] + nrm[2] * r, 0.0];
        let mut ep0 = [ct[0] + a[0] * h, ct[1] + a[1] * h, ct[2] + a[2] * h, 0.0];
        let mut ep1 = [ct[0] - a[0] * h, ct[1] - a[1] * h, ct[2] - a[2] * h, 0.0];
        for k in 0..3 {
            ep0[k] -= p0[k];
        }
        for k in 0..3 {
            ep1[k] -= p0[k];
        }
        let mn = neg3(&self.n);
        if !clip_edge_to_plane(&mut ep0, &mut ep1, &mk_plane(&mn, 0.0)) {
            return;
        }
        let t = cross(&self.n, &self.e0);
        if !clip_edge_to_plane(&mut ep0, &mut ep1, &mk_plane(&t, 1e-5)) {
            return;
        }
        let t = cross(&self.n, &self.e1);
        if !clip_edge_to_plane(&mut ep0, &mut ep1, &mk_plane(&t, -(dot3(&self.e0, &t) - 1e-5))) {
            return;
        }
        let t = cross(&self.n, &self.e2);
        if !clip_edge_to_plane(&mut ep0, &mut ep1, &mk_plane(&t, 1e-5)) {
            return;
        }
        for k in 0..3 {
            ep0[k] += p0[k];
        }
        for k in 0..3 {
            ep1[k] += p0[k];
        }
        let t = sub3(&ep0, &p);
        let mut d0 = dot3(&t, &nrm) - (self.best_center - self.best_rt);
        let t = sub3(&ep1, &p);
        let mut d1 = dot3(&t, &nrm) - (self.best_center - self.best_rt);
        if d0 < 0.0 {
            d0 = 0.0;
        }
        if d1 < 0.0 {
            d1 = 0.0;
        }
        let maxc = (self.flags & NUMC_MASK) as usize;
        self.local.push(LocalContact { pos: ep0, normal: nrm, depth: d0, tri: 0, keep: true });
        if self.local.len() < maxc {
            self.local.push(LocalContact { pos: ep1, normal: nrm, depth: d1, tri: 0, keep: true });
        }
    }
}

/// _IsNearContacts
fn near_contacts(c1: &LocalContact, c2: &LocalContact) -> bool {
    let e = 0.0001;
    let d = sub3(&c1.pos, &c2.pos);
    let pos_near = d[0].abs() < e && d[1].abs() < e && d[2].abs() < e;
    let d = sub3(&c1.normal, &c2.normal);
    let same_dir = d[0].abs() < e && d[1].abs() < e && d[2].abs() < e;
    pos_near && same_dir
}

fn collide_capsule_trimesh(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    use super::volume::{ObbVol, VolumeQuery};
    let d = mesh_of(o1.shape);
    let Shape::Capsule { radius, lz } = *o2.shape else { return Vec::new() };
    let (mesh_pos, mesh_rot) = (&o1.pr.pos, &o1.pr.r);
    let (cpos, crot) = (&o2.pr.pos, &o2.pr.r);
    let mut c = Cctl {
        pos: *cpos,
        axis: [crot[2], crot[6], crot[10], 0.0],
        radius,
        size: lz + 2.0 * radius,
        e0: [0.0; 4],
        e1: [0.0; 4],
        e2: [0.0; 4],
        normal: [0.0; 4],
        best_depth: -f64::MAX,
        best_center: 0.0,
        best_rt: 0.0,
        best_axis: 0,
        n: [0.0; 4],
        v0: [0.0; 4],
        v1: [0.0; 4],
        v2: [0.0; 4],
        flags,
        local: Vec::new(),
    };
    let mesh_matrix = opcode::make_matrix(&[0.0; 3], mesh_rot);
    let off = sub3(cpos, mesh_pos);
    let half_axis = c.size * 0.5;
    let r = crot;
    let rot = [[r[0] as f32, r[4] as f32, r[8] as f32], [r[1] as f32, r[5] as f32, r[9] as f32], [r[2] as f32, r[6] as f32, r[10] as f32]];
    let vol = ObbVol::new([off[0] as f32, off[1] as f32, off[2] as f32], [radius as f32, radius as f32, half_axis as f32], rot, &mesh_matrix);
    let mut q = VolumeQuery::new(vol, &d.mesh);
    q.run(&d.model);
    if !q.contact || q.touched.is_empty() {
        return Vec::new();
    }
    let maxc = (flags & NUMC_MASK) as usize;
    for &tri in &q.touched {
        let dv = fetch_triangle(d, tri as usize, mesh_pos, mesh_rot);
        c.test_triangle(&dv[0], &dv[1], &dv[2], tri as i32);
        if c.local.len() >= maxc {
            break;
        }
    }
    let mut out = Vec::new();
    if c.local.is_empty() {
        return out;
    }
    // _ProcessLocalContacts
    let n = c.local.len();
    if n > 1 && flags & CONTACTS_UNIMPORTANT == 0 {
        for i in 0..n.saturating_sub(1) {
            for j in i + 1..n {
                if near_contacts(&c.local[i], &c.local[j]) {
                    if c.local[j].depth > c.local[i].depth {
                        c.local[i].keep = false;
                    } else {
                        c.local[j].keep = false;
                    }
                }
            }
        }
    }
    for lc in &c.local {
        if out.len() >= maxc {
            break;
        }
        if lc.keep {
            out.push(ContactGeom { pos: lc.pos, normal: lc.normal, depth: lc.depth, g1: o1.id, g2: o2.id, side1: lc.tri, side2: -1 });
        }
    }
    out
}

/// dCollideTrimeshPlane: every vertex (each once) below the plane
fn collide_trimesh_plane(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    let d = mesh_of(o1.shape);
    let Shape::Plane { p } = o2.shape else { return Vec::new() };
    let (pos, r) = (&o1.pr.pos, &o1.pr.r);
    let maxc = (flags & NUMC_MASK) as usize;
    let mut used = vec![false; d.mesh.verts.len()];
    let mut out = Vec::new();
    for (t, tri) in d.mesh.tris.iter().enumerate() {
        let fv = d.mesh.tri(t);
        for v in 0..3 {
            let idx = tri[v] as usize;
            if used[idx] {
                continue;
            }
            used[idx] = true;
            let iv = [fv[v][0] as Real, fv[v][1] as Real, fv[v][2] as Real, 0.0];
            let mut vx = mul0_331(r, &iv);
            vx[0] += pos[0];
            vx[1] += pos[1];
            vx[2] += pos[2];
            let alpha = p[3] - dot3(p, &vx);
            if alpha > 0.0 {
                out.push(ContactGeom { pos: vx, normal: [p[0], p[1], p[2], 0.0], depth: alpha, g1: o1.id, g2: o2.id, side1: t as i32, side2: -1 });
                if out.len() >= maxc {
                    return out;
                }
            }
        }
    }
    out
}

// ---- heightfield.cpp

/// dxHeightfieldData. Samples are stored as given (byte, short, float or
/// double data all convert to f64 exactly), row-major with x fastest.
#[derive(Debug)]
pub struct HeightfieldData {
    pub width: Real,
    pub depth: Real,
    pub sample_width: Real,
    pub sample_depth: Real,
    pub sample_zx_aspect: Real,
    pub inv_sample_width: Real,
    pub inv_sample_depth: Real,
    pub half_width: Real,
    pub half_depth: Real,
    pub min_height: Real,
    pub max_height: Real,
    pub thickness: Real,
    pub scale: Real,
    pub offset: Real,
    pub width_samples: i32,
    pub depth_samples: i32,
    pub wrap: bool,
    pub samples: Vec<Real>,
}

impl HeightfieldData {
    /// dGeomHeightfieldDataBuild{Byte,Short,Single,Double} (SetData +
    /// ComputeHeightBounds)
    #[allow(clippy::too_many_arguments)]
    pub fn build(samples: Vec<Real>, width: Real, depth: Real, width_samples: i32, depth_samples: i32, scale: Real, offset: Real, thickness: Real, wrap: bool) -> Arc<HeightfieldData> {
        let sw = width / (width_samples as Real - 1.0);
        let sd = depth / (depth_samples as Real - 1.0);
        let mut mn = INFINITY;
        let mut mx = -INFINITY;
        for &h in &samples[..(width_samples * depth_samples) as usize] {
            if h < mn {
                mn = h;
            }
            if h > mx {
                mx = h;
            }
        }
        mn *= scale;
        mx *= scale;
        mn += offset;
        mx += offset;
        mn -= thickness;
        Arc::new(HeightfieldData {
            width,
            depth,
            sample_width: sw,
            sample_depth: sd,
            sample_zx_aspect: sd / sw,
            inv_sample_width: 1.0 / sw,
            inv_sample_depth: 1.0 / sd,
            half_width: width / 2.0,
            half_depth: depth / 2.0,
            min_height: mn,
            max_height: mx,
            thickness,
            scale,
            offset,
            width_samples,
            depth_samples,
            wrap,
            samples,
        })
    }
    /// GetHeight(int x, int z)
    pub fn height(&self, mut x: i32, mut z: i32) -> Real {
        if !self.wrap {
            if x < 0 {
                x = 0;
            }
            if z < 0 {
                z = 0;
            }
            if x > self.width_samples - 1 {
                x = self.width_samples - 1;
            }
            if z > self.depth_samples - 1 {
                z = self.depth_samples - 1;
            }
        } else {
            x %= self.width_samples - 1;
            z %= self.depth_samples - 1;
            if x < 0 {
                x += self.width_samples - 1;
            }
            if z < 0 {
                z += self.depth_samples - 1;
            }
        }
        let h = self.samples[(x + z * self.width_samples) as usize];
        (h * self.scale) + self.offset
    }
    /// IsOnHeightfield2
    fn is_on2(&self, corner: &HfVertex, pos: &[Real], is_abc: bool) -> bool {
        if is_abc {
            let min_x = corner.vertex[0];
            if pos[0] < min_x {
                return false;
            }
            let max_x = (corner.coords.0 + 1) as Real * self.sample_width;
            if pos[0] >= max_x {
                return false;
            }
            let min_z = corner.vertex[2];
            if pos[2] < min_z {
                return false;
            }
            let max_z = (corner.coords.1 + 1) as Real * self.sample_depth;
            if pos[2] >= max_z {
                return false;
            }
            (max_z - pos[2]) > (pos[0] - min_x) * self.sample_zx_aspect
        } else {
            let max_x = corner.vertex[0];
            if pos[0] >= max_x {
                return false;
            }
            let min_x = (corner.coords.0 - 1) as Real * self.sample_width;
            if pos[0] < min_x {
                return false;
            }
            let max_z = corner.vertex[2];
            if pos[2] >= max_z {
                return false;
            }
            let min_z = (corner.coords.1 - 1) as Real * self.sample_depth;
            if pos[2] < min_z {
                return false;
            }
            (max_z - pos[2]) <= (pos[0] - min_x) * self.sample_zx_aspect
        }
    }
}

#[derive(Clone, Copy, Default)]
struct HfVertex {
    vertex: V3,
    coords: (i32, i32),
    state: bool,
}
#[derive(Clone, Copy)]
struct HfTri {
    v: [usize; 3],
    plane: [Real; 4],
    max_aaab: Real,
    is_up: bool,
    state: bool,
}
struct HfPlane {
    tris: Vec<usize>,
    max_aaab: Real,
    plane: [Real; 4],
}

fn dmin(a: Real, b: Real) -> Real {
    if a > b {
        b
    } else {
        a
    }
}
fn dmax(a: Real, b: Real) -> Real {
    if a > b {
        a
    } else {
        b
    }
}
fn dmin3(a: Real, b: Real, c: Real) -> Real {
    if a < b {
        dmin(a, c)
    } else {
        dmin(b, c)
    }
}
fn dmax3(a: Real, b: Real, c: Real) -> Real {
    if a > b {
        dmax(a, c)
    } else {
        dmax(b, c)
    }
}

/// dxHeightfield::computeAABB
fn heightfield_aabb(d: &HeightfieldData, placeable: bool, pr: &PosR) -> [Real; 6] {
    if !d.wrap {
        if placeable {
            let r = &pr.r;
            let mut dy = [0.0; 6];
            if d.min_height != -INFINITY {
                dy[0] = r[1] * d.min_height;
                dy[1] = r[5] * d.min_height;
                dy[2] = r[9] * d.min_height;
            } else {
                dy[0] = if r[1] != 0.0 { r[1] * -INFINITY } else { 0.0 };
                dy[1] = if r[5] != 0.0 { r[5] * -INFINITY } else { 0.0 };
                dy[2] = if r[9] != 0.0 { r[9] * -INFINITY } else { 0.0 };
            }
            if d.max_height != INFINITY {
                dy[3] = r[1] * d.max_height;
                dy[4] = r[5] * d.max_height;
                dy[5] = r[9] * d.max_height;
            } else {
                dy[3] = if r[1] != 0.0 { r[1] * INFINITY } else { 0.0 };
                dy[4] = if r[5] != 0.0 { r[5] * INFINITY } else { 0.0 };
                dy[5] = if r[9] != 0.0 { r[9] * INFINITY } else { 0.0 };
            }
            let dx = [r[0] * -d.half_width, r[4] * -d.half_width, r[8] * -d.half_width, r[0] * d.half_width, r[4] * d.half_width, r[8] * d.half_width];
            let dz = [r[2] * -d.half_depth, r[6] * -d.half_depth, r[10] * -d.half_depth, r[2] * d.half_depth, r[6] * d.half_depth, r[10] * d.half_depth];
            let p = &pr.pos;
            let mut a = [0.0; 6];
            for k in 0..3 {
                a[2 * k] = p[k] + dmin3(dmin(dx[k], dx[k + 3]), dmin(dy[k], dy[k + 3]), dmin(dz[k], dz[k + 3]));
                a[2 * k + 1] = p[k] + dmax3(dmax(dx[k], dx[k + 3]), dmax(dy[k], dy[k + 3]), dmax(dz[k], dz[k + 3]));
            }
            a
        } else {
            [-d.half_width, d.half_width, d.min_height, d.max_height, -d.half_depth, d.half_depth]
        }
    } else if placeable {
        [-INFINITY, INFINITY, -INFINITY, INFINITY, -INFINITY, INFINITY]
    } else {
        [-INFINITY, INFINITY, d.min_height, d.max_height, -INFINITY, INFINITY]
    }
}

/// dGeomSpherePointDepth / dGeomBoxPointDepth / dGeomCapsulePointDepth
fn point_depth(g: &Prim, x: Real, y: Real, z: Real) -> Option<Real> {
    let (pos, r) = (&g.pr.pos, &g.pr.r);
    match g.shape {
        Shape::Sphere { radius } => Some(radius - ((x - pos[0]) * (x - pos[0]) + (y - pos[1]) * (y - pos[1]) + (z - pos[2]) * (z - pos[2])).sqrt()),
        Shape::Box { side } => {
            let p = v3(x - pos[0], y - pos[1], z - pos[2]);
            let q = mul1_331(r, &p);
            let mut dist = [0.0; 6];
            let mut inside = true;
            for i in 0..3 {
                let s = side[i] * 0.5;
                dist[i] = s - q[i];
                dist[i + 3] = s + q[i];
                if dist[i] < 0.0 || dist[i + 3] < 0.0 {
                    inside = false;
                }
            }
            if inside {
                let mut smallest = u32::MAX as Real;
                for &d in &dist {
                    if d < smallest {
                        smallest = d;
                    }
                }
                return Some(smallest);
            }
            let mut largest = 0.0;
            for &d in &dist {
                if d > largest {
                    largest = d;
                }
            }
            Some(-largest)
        }
        Shape::Capsule { radius, lz } => {
            let a = [x - pos[0], y - pos[1], z - pos[2]];
            let mut beta = dot3_14(&a, &r[2..]);
            let lz2 = lz * 0.5;
            if beta < -lz2 {
                beta = -lz2;
            } else if beta > lz2 {
                beta = lz2;
            }
            let a = [pos[0] + beta * r[2], pos[1] + beta * r[6], pos[2] + beta * r[10]];
            Some(radius - ((x - a[0]) * (x - a[0]) + (y - a[1]) * (y - a[1]) + (z - a[2]) * (z - a[2])).sqrt())
        }
        _ => None,
    }
}

/// the colliders the heightfield runs against its planes and rays
fn hf_plane_collider(c: usize) -> Option<ColliderFn> {
    match c {
        RAY => Some(collide_ray_plane),
        SPHERE => Some(collide_sphere_plane),
        BOX => Some(collide_box_plane),
        CAPSULE => Some(collide_capsule_plane),
        CYLINDER => Some(collide_cylinder_plane),
        CONVEX => Some(collide_convex_plane),
        TRIMESH => Some(collide_trimesh_plane),
        _ => None,
    }
}
/// ray first, as the heightfield calls them
fn hf_ray_collider(c: usize) -> Option<fn(&Prim, &Prim, i32) -> Vec<ContactGeom>> {
    match c {
        SPHERE => Some(collide_ray_sphere),
        BOX => Some(collide_ray_box),
        CAPSULE => Some(collide_ray_capsule),
        CYLINDER => Some(collide_ray_cylinder),
        CONVEX => Some(collide_ray_convex),
        TRIMESH => Some(|ray: &Prim, mesh: &Prim, f: i32| collide_ray_trimesh(mesh, ray, f)),
        _ => None,
    }
}

const DEPS: Real = f64::EPSILON;

/// dxHeightfield::dCollideHeightfieldZone, in heightfield space
fn hf_zone(d: &HeightfieldData, min_x: i32, max_x: i32, min_z: i32, max_z: i32, o2: &Prim, o2_aabb: &[Real; 6], max_contacts: i32, flags: i32) -> Vec<ContactGeom> {
    let num_x = (max_x - min_x + 1) as usize;
    let num_z = (max_z - min_z + 1) as usize;
    let min_o2 = o2_aabb[2];
    let max_o2 = o2_aabb[3];
    let mut max_y = -INFINITY;
    let mut min_y = INFINITY;
    let mut verts = vec![HfVertex::default(); num_x * num_z];
    for xl in 0..num_x {
        let x = min_x + xl as i32;
        let xpos = x as Real * d.sample_width;
        for zl in 0..num_z {
            let z = min_z + zl as i32;
            let ypos = z as Real * d.sample_depth;
            let h = d.height(x, z);
            let v = &mut verts[xl * num_z + zl];
            v.vertex = [xpos, h, ypos, 0.0];
            v.coords = (x, z);
            max_y = if max_y > h { max_y } else { h };
            min_y = if min_y > h { h } else { min_y };
        }
    }
    let mut out = Vec::new();
    if min_o2 - max_y > -DEPS {
        return out;
    }
    if min_y - max_o2 > -DEPS {
        out.push(ContactGeom { pos: [o2.pr.pos[0], min_y, o2.pr.pos[2], 0.0], normal: [0.0, -1.0, 0.0, 0.0], depth: min_y - max_o2, side1: -1, side2: -1, ..Default::default() });
        return out;
    }
    let c2 = o2.shape.class();
    let plane_fn = hf_plane_collider(c2).unwrap();
    let ray_fn = hf_ray_collider(c2);
    let plane_prim = |p: [Real; 4]| -> (Shape, PosR) { (Shape::Plane { p }, PosR::default()) };
    if max_y - min_y < DEPS {
        let tp = [0.0, 1.0, 0.0, min_y];
        let (ps, pr) = plane_prim(tp);
        let pl = Prim { shape: &ps, pr, id: usize::MAX, flags: 0, q: [1.0, 0.0, 0.0, 0.0], body: None };
        let mut cs = plane_fn(o2, &pl, flags);
        for c in &mut cs {
            c.normal = [-tp[0], -tp[1], -tp[2], c.normal[3]];
        }
        return cs;
    }
    let mut need_further = c2 == TRIMESH;
    if c2 != RAY && !need_further {
        let xratio = (o2_aabb[1] - o2_aabb[0]) * d.inv_sample_width;
        if xratio > 1.5 {
            need_further = true;
        } else {
            let zratio = (o2_aabb[5] - o2_aabb[4]) * d.inv_sample_depth;
            if zratio > 1.5 {
                need_further = true;
            }
        }
    }
    let mut tris: Vec<HfTri> = Vec::new();
    let mx_l = (max_x - min_x) as usize;
    let mz_l = (max_z - min_z) as usize;
    let idx = |xl: usize, zl: usize| xl * num_z + zl;
    for xl in 0..mx_l {
        let mut c = idx(xl, 0);
        let mut dd = idx(xl + 1, 0);
        for zl in 0..mz_l {
            let a = c;
            let b = dd;
            c = idx(xl, zl + 1);
            dd = idx(xl + 1, zl + 1);
            let (ah, bh, ch, dh) = (verts[a].vertex[1], verts[b].vertex[1], verts[c].vertex[1], verts[dd].vertex[1]);
            let (ac, bc, cc, dc) = (ah > min_o2, bh > min_o2, ch > min_o2, dh > min_o2);
            verts[a].state = !ac;
            verts[b].state = !bc;
            verts[c].state = !cc;
            verts[dd].state = !dc;
            let tri_max = |vs: &[usize; 3]| -> Real {
                let m = if verts[vs[0]].vertex[1] > verts[vs[1]].vertex[1] { verts[vs[0]].vertex[1] } else { verts[vs[1]].vertex[1] };
                if verts[vs[2]].vertex[1] > m {
                    verts[vs[2]].vertex[1]
                } else {
                    m
                }
            };
            if ac || bc || cc {
                let v = [a, b, c];
                tris.push(HfTri { v, plane: [0.0; 4], max_aaab: tri_max(&v), is_up: true, state: false });
            }
            if bc || cc || dc {
                let v = [dd, b, c];
                tris.push(HfTri { v, plane: [0.0; 4], max_aaab: tri_max(&v), is_up: false, state: false });
            }
            if need_further && (bc || cc) && (ah > ch && ah > bh && dh > ch && dh > bh) {
                verts[b].state = true;
                verts[c].state = true;
            }
        }
    }
    for t in &mut tris {
        let e1 = sub3(&verts[t.v[2]].vertex, &verts[t.v[0]].vertex);
        let e2 = sub3(&verts[t.v[1]].vertex, &verts[t.v[0]].vertex);
        let mut tp = if t.is_up { cross(&e1, &e2) } else { cross(&e2, &e1) };
        let inv = 1.0 / length3(&tp);
        tp[0] *= inv;
        tp[1] *= inv;
        tp[2] *= inv;
        let d3 = dot3(&tp, &verts[t.v[0]].vertex);
        t.plane = [tp[0], tp[1], tp[2], d3];
    }
    let mut planes: Vec<HfPlane> = Vec::new();
    for k in 0..tris.len() {
        if tris[k].state {
            continue;
        }
        let base = tris[k].plane;
        let mut list = vec![k];
        for m in k + 1..tris.len() {
            if tris[m].state {
                continue;
            }
            let tp = tris[m].plane;
            if (base[1] - tp[1]).abs() < DEPS && (base[3] - tp[3]).abs() < DEPS && (base[0] - tp[0]).abs() < DEPS && (base[2] - tp[2]).abs() < DEPS {
                list.push(m);
                tris[m].state = true;
            }
        }
        tris[k].state = true;
        let mut mx = tris[list[0]].max_aaab;
        for &t in &list[1..] {
            if tris[t].max_aaab > mx {
                mx = tris[t].max_aaab;
            }
        }
        planes.push(HfPlane { tris: list, max_aaab: mx, plane: base });
    }
    // sortPlanes: a bubble sort on max height (ascending)
    if planes.len() > 1 {
        loop {
            let mut swapped = false;
            for i in 0..planes.len() - 1 {
                if planes[i].max_aaab - planes[i + 1].max_aaab > DEPS {
                    planes.swap(i, i + 1);
                    swapped = true;
                }
            }
            if !swapped {
                break;
            }
        }
    }
    let plane_flags = (flags & !NUMC_MASK) | 10;
    for pl in &planes {
        let (ps, pr) = plane_prim(pl.plane);
        let plp = Prim { shape: &ps, pr, id: usize::MAX, flags: 0, q: [1.0, 0.0, 0.0, 0.0], body: None };
        let pcs = plane_fn(o2, &plp, plane_flags);
        let mut did = false;
        for pc in &pcs {
            for &t in &pl.tris {
                if d.is_on2(&verts[tris[t].v[0]], &pc.pos, tris[t].is_up) {
                    out.push(ContactGeom { pos: [pc.pos[0], pc.pos[1], pc.pos[2], 0.0], normal: [-pl.plane[0], -pl.plane[1], -pl.plane[2], 0.0], depth: pc.depth, side1: pc.side1, side2: pc.side2, ..Default::default() });
                    if out.len() as i32 == max_contacts {
                        return out;
                    }
                    did = true;
                    break;
                }
            }
        }
        if did {
            for &t in &pl.tris {
                for i in 0..3 {
                    verts[tris[t].v[i]].state = true;
                }
            }
        } else {
            for &t in &pl.tris {
                tris[t].state = false;
            }
        }
    }
    if need_further {
        let ray_flags = (flags & !NUMC_MASK) | 1;
        for k in 0..tris.len() {
            if tris[k].state {
                continue;
            }
            for i in 0..3 {
                let vi = tris[k].v[i];
                if verts[vi].state {
                    continue;
                }
                let tv = verts[vi].vertex;
                let mut depth = 0.0;
                let mut collided = false;
                if let Some(dep) = point_depth(o2, tv[0], tv[1], tv[2]) {
                    depth = dep;
                    if dep > DEPS {
                        collided = true;
                    }
                } else if let Some(rf) = ray_fn {
                    let len = (min_o2 - tv[1]) * 1000.0;
                    let rs = Shape::Ray { length: len };
                    let mut rr = IDENTITY_R;
                    rr[2] = tris[k].plane[0];
                    rr[6] = tris[k].plane[1];
                    rr[10] = tris[k].plane[2];
                    let ray = Prim { shape: &rs, pr: PosR { pos: tv, r: rr }, id: usize::MAX, flags: 0, q: [1.0, 0.0, 0.0, 0.0], body: None };
                    let rc = rf(&ray, o2, ray_flags);
                    if !rc.is_empty() {
                        depth = rc[0].depth;
                        collided = true;
                    }
                }
                if collided {
                    let p = tris[k].plane;
                    out.push(ContactGeom { pos: tv, normal: [-p[0], -p[1], -p[2], 0.0], depth, side1: -1, side2: -1, ..Default::default() });
                    if out.len() as i32 == max_contacts {
                        return out;
                    }
                    verts[vi].state = true;
                }
            }
        }
    }
    out
}

/// dCollideHeightfield: o2 moved into heightfield space, the zone it
/// covers collided, contacts moved back
fn collide_heightfield(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    let (Shape::Heightfield(d), placeable) = (o1.shape, o1.flags & GEOM_PLACEABLE != 0) else { return Vec::new() };
    let max_contacts = flags & NUMC_MASK;
    let mut local = o2.pr;
    if placeable {
        let pos0 = sub3(&o2.pr.pos, &o1.pr.pos);
        let pos1 = mul1_331(&o1.pr.r, &pos0);
        local.r = mul1_333(&o1.pr.r, &o2.pr.r);
        local.pos = [pos1[0], pos1[1], pos1[2], local.pos[3]];
    }
    local.pos[0] += d.half_width;
    local.pos[2] += d.half_depth;
    let o2l = Prim { shape: o2.shape, pr: local, id: o2.id, flags: o2.flags, q: o2.q, body: o2.body };
    let aabb = shape_aabb(o2.shape, &local);
    let mut out = Vec::new();
    'zone: {
        if !d.wrap {
            if aabb[0] > d.width || aabb[4] > d.depth {
                break 'zone;
            }
            if aabb[1] < 0.0 || aabb[5] < 0.0 {
                break 'zone;
            }
        }
        let mut nmin_x = (aabb[0] * d.inv_sample_width).next_down().floor() as i32;
        let mut nmax_x = (aabb[1] * d.inv_sample_width).next_up().ceil() as i32;
        let mut nmin_z = (aabb[4] * d.inv_sample_depth).next_down().floor() as i32;
        let mut nmax_z = (aabb[5] * d.inv_sample_depth).next_up().ceil() as i32;
        if !d.wrap {
            nmin_x = if nmin_x > 0 { nmin_x } else { 0 };
            nmax_x = if nmax_x > d.width_samples - 1 { d.width_samples - 1 } else { nmax_x };
            nmin_z = if nmin_z > 0 { nmin_z } else { 0 };
            nmax_z = if nmax_z > d.depth_samples - 1 { d.depth_samples - 1 } else { nmax_z };
        }
        out = hf_zone(d, nmin_x, nmax_x, nmin_z, nmax_z, &o2l, &aabb, max_contacts, flags);
        for c in &mut out {
            c.g1 = o1.id;
            c.g2 = o2.id;
        }
    }
    for c in &mut out {
        if placeable {
            let mut pos0 = c.pos;
            pos0[0] -= d.half_width;
            pos0[2] -= d.half_depth;
            let p = mul0_331(&o1.pr.r, &pos0);
            c.pos = add3(&p, &o1.pr.pos);
            let n0 = c.normal;
            c.normal = mul0_331(&o1.pr.r, &n0);
        } else {
            c.pos[0] -= d.half_width;
            c.pos[2] -= d.half_depth;
        }
    }
    out
}

// ---- cylinder.cpp, collision_cylinder_{sphere,plane,box}.cpp, ray.cpp

fn cylinder_aabb(radius: Real, lz: Real, pr: &PosR) -> [Real; 6] {
    let (r, pos) = (&pr.r, &pr.pos);
    let range = |c: Real| -> Real {
        let one = 1.0 - c * c;
        (c * lz * 0.5).abs() + radius * (if 0.0 > one { 0.0 } else { one }).sqrt()
    };
    let (xr, yr, zr) = (range(r[2]), range(r[6]), range(r[10]));
    [pos[0] - xr, pos[0] + xr, pos[1] - yr, pos[1] + yr, pos[2] - zr, pos[2] + zr]
}

fn cyl_params(s: &Shape) -> (Real, Real) {
    match s {
        Shape::Cylinder { radius, lz } => (*radius, *lz),
        _ => (0.0, 0.0),
    }
}

/// dCollideCylinderSphere
fn collide_cylinder_sphere(o1: &Prim, o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    let (radius, length) = cyl_params(o1.shape);
    let cp = &o1.pr.pos;
    let radius2 = radius_of(o2.shape);
    let sp = &o2.pr.pos;
    let d = [o1.pr.r[2], o1.pr.r[6], o1.pr.r[10]];
    let s = length * 0.5;
    let p2 = [d[0] * s + cp[0], d[1] * s + cp[1], d[2] * s + cp[2]];
    let p1 = [d[0] * -s + cp[0], d[1] * -s + cp[1], d[2] * -s + cp[2]];
    let s = (sp[0] - p1[0]) * d[0] - (p1[1] - sp[1]) * d[1] - (p1[2] - sp[2]) * d[2];
    if s < -radius2 || s > length + radius2 {
        return Vec::new();
    }
    let mut c = [s * d[0] + p1[0] - sp[0], s * d[1] + p1[1] - sp[1], s * d[2] + p1[2] - sp[2], 0.0];
    let t = length3(&c);
    if t > radius + radius2 {
        return Vec::new();
    }
    let mut ct = cg(o1, o2);
    if t > radius && (s < 0.0 || s > length) {
        let (end, ds) = if s <= 0.0 { (p1, s) } else { (p2, s - length) };
        ct.depth = radius2 - (ds * ds + (t - radius) * (t - radius)).sqrt();
        if ct.depth < 0.0 {
            return Vec::new();
        }
        for k in 0..3 {
            ct.pos[k] = c[k] / t * -radius + end[k];
        }
        for k in 0..3 {
            ct.normal[k] = (ct.pos[k] - sp[k]) / (radius2 - ct.depth);
        }
        return vec![ct];
    } else if (radius - t) <= s && (radius - t) <= (length - s) {
        if t > radius2 + 0.0000001 {
            ct.depth = (radius2 + radius) - t;
            if ct.depth < 0.0 {
                return Vec::new();
            }
            c[0] /= t;
            c[1] /= t;
            c[2] /= t;
            for k in 0..3 {
                ct.pos[k] = c[k] * radius2 + sp[k];
                ct.normal[k] = c[k];
            }
        } else {
            ct.depth = (radius2 + radius) - t;
            if ct.depth < 0.0 {
                return Vec::new();
            }
            for k in 0..3 {
                ct.pos[k] = c[k] + sp[k];
                ct.normal[k] = c[k] / t;
            }
        }
        return vec![ct];
    }
    if s <= length * 0.5 {
        ct.depth = s + radius2;
        if ct.depth < 0.0 {
            return Vec::new();
        }
        for k in 0..3 {
            ct.pos[k] = radius2 * d[k] + sp[k];
            ct.normal[k] = d[k];
        }
    } else {
        ct.depth = radius2 + length - s;
        if ct.depth < 0.0 {
            return Vec::new();
        }
        for k in 0..3 {
            ct.pos[k] = radius2 * -d[k] + sp[k];
            ct.normal[k] = -d[k];
        }
    }
    vec![ct]
}

/// dCollideCylinderPlane
fn collide_cylinder_plane(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    let (radius, length) = cyl_params(o1.shape);
    let Shape::Plane { p: pv } = o2.shape else { return Vec::new() };
    let cp = &o1.pr.pos;
    let pn = [pv[0], pv[1], pv[2], 0.0];
    let d = [o1.pr.r[2], o1.pr.r[6], o1.pr.r[10], 0.0];
    let s = length * 0.5;
    let p2 = [d[0] * s + cp[0], d[1] * s + cp[1], d[2] * s + cp[2], 0.0];
    let p1 = [d[0] * -s + cp[0], d[1] * -s + cp[1], d[2] * -s + cp[2], 0.0];
    let tol = 0.0000001;
    let maxc = (flags & NUMC_MASK) as usize;
    let mut out = Vec::new();
    let mut s = d[0] * pn[0] + d[1] * pn[1] + d[2] * pn[2];
    if s < 0.0 {
        s += 1.0;
    } else {
        s -= 1.0;
    }
    let mut try_push = |pos: V3, strict: bool, depth: Real, out: &mut Vec<ContactGeom>| -> bool {
        let ok = if strict { depth > 0.0 } else { depth >= 0.0 };
        if ok {
            let mut c = cg(o1, o2);
            c.pos = pos;
            c.depth = depth;
            c.normal = pn;
            out.push(c);
            if out.len() >= maxc {
                return true;
            }
        }
        false
    };
    if s < tol && s > -tol {
        let s = pv[3] - dot3(pv, &p1);
        let t = pv[3] - dot3(pv, &p2);
        let p = if s >= t {
            if s >= 0.0 {
                p1
            } else {
                return out;
            }
        } else if t >= 0.0 {
            p2
        } else {
            return out;
        };
        let mut v1 = if d[0] < tol && d[0] > -tol { [d[0] + 1.0, d[1], d[2], 0.0] } else { [d[0], d[1] + 1.0, d[2], 0.0] };
        let mut v2 = cross(&v1, &d);
        let t = radius / length3(&v2);
        v2[0] *= t;
        v2[1] *= t;
        v2[2] *= t;
        v1 = cross(&v2, &d);
        for pos in [add3(&p, &v1), sub3(&p, &v1), add3(&p, &v2), sub3(&p, &v2)] {
            let depth = pv[3] - dot3(pv, &pos);
            if try_push(pos, true, depth, &mut out) {
                return out;
            }
        }
    } else {
        let t = dot3(&pn, &d);
        let mut c = [d[0] * t - pn[0], d[1] * t - pn[1], d[2] * t - pn[2], 0.0];
        let s = radius / length3(&c);
        c[0] *= s;
        c[1] *= s;
        c[2] *= s;
        let pos = add3(&c, &p1);
        let depth = pv[3] - dot3(pv, &pos);
        if try_push(pos, false, depth, &mut out) {
            return out;
        }
        let pos = add3(&c, &p2);
        let depth = pv[3] - pv[0] * pos[0] - pv[1] * pos[1] - pv[2] * pos[2];
        if try_push(pos, false, depth, &mut out) {
            return out;
        }
    }
    out
}

/// dCollideRayCylinder
fn collide_ray_cylinder(o1: &Prim, o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    let (radius, lz) = cyl_params(o2.shape);
    let (rp, rr) = (&o1.pr.pos, &o1.pr.r);
    let (cp, cr) = (&o2.pr.pos, &o2.pr.r);
    let half = lz * 0.5;
    let tmp = sub3(rp, cp);
    let mut pos = mul1_331(cr, &tmp);
    let tmp = [rr[2], rr[6], rr[10], 0.0];
    let mut dir = mul1_331(cr, &tmp);
    let r2 = radius * radius;
    let c = pos[0] * pos[0] + pos[1] * pos[1] - r2;
    let parallel = dir[0] == 0.0 && dir[1] == 0.0;
    let perpendicular = dir[2] == 0.0;
    let in_radius = c <= 0.0;
    let in_caps = pos[2].abs() <= half;
    let check_caps = !perpendicular && (!in_caps || in_radius);
    let mut check_cyl = !parallel && (!in_radius || in_caps);
    let flip_normals = in_caps && in_radius;
    let mut tt = -INFINITY;
    let mut tn = [f64::NAN, f64::NAN, f64::NAN, 0.0];
    let mut tp = [0.0; 4];
    let len = ray_len(o1.shape);
    if check_caps {
        let mut flip_dir = false;
        if (dir[2] < 0.0 && flip_normals) || (dir[2] > 0.0 && !flip_normals) {
            flip_dir = true;
            dir[2] = -dir[2];
            pos[2] = -pos[2];
        }
        tt = (half - pos[2]) / dir[2];
        if tt >= 0.0 && tt <= len {
            tp[0] = pos[0] + tt * dir[0];
            tp[1] = pos[1] + tt * dir[1];
            if tp[0] * tp[0] + tp[1] * tp[1] <= r2 {
                tp[2] = if flip_dir { -half } else { half };
                tn = [0.0, 0.0, if flip_dir != flip_normals { -1.0 } else { 1.0 }, 0.0];
                check_cyl = false;
            } else {
                tt = -INFINITY;
            }
        } else {
            tt = -INFINITY;
        }
        if flip_dir {
            dir[2] = -dir[2];
            pos[2] = -pos[2];
        }
    }
    if check_cyl {
        let mut a = dir[0] * dir[0] + dir[1] * dir[1];
        let b = 2.0 * (pos[0] * dir[0] + pos[1] * dir[1]);
        let mut k = b * b - 4.0 * a * c;
        if k >= 0.0 && (b < 0.0 || b * b <= k) {
            k = k.sqrt();
            a = 1.0 / (2.0 * a);
            tt = if b.abs() <= k { (-b + k) * a } else { (-b - k) * a };
            if tt <= len {
                tp[2] = pos[2] + tt * dir[2];
                if tp[2].abs() <= half {
                    tp[0] = pos[0] + tt * dir[0];
                    tp[1] = pos[1] + tt * dir[1];
                    tn = [tp[0] / radius, tp[1] / radius, 0.0, 0.0];
                    if flip_normals {
                        tn[0] = -tn[0];
                        tn[1] = -tn[1];
                    }
                } else {
                    tt = -INFINITY;
                }
            } else {
                tt = -INFINITY;
            }
        }
    }
    if tt > 0.0 {
        let mut ct = cg(o1, o2);
        ct.depth = tt;
        tn[3] = 0.0;
        tp[3] = 0.0;
        ct.normal = mul0_331(cr, &tn);
        let p = mul0_331(cr, &tp);
        ct.pos = add3(&p, cp);
        return vec![ct];
    }
    Vec::new()
}

/// dMatrix3Inv (collision_util.h)
fn matrix3_inv(ma: &[Real]) -> M3 {
    let det = ma[0] * (ma[5] * ma[10] - ma[9] * ma[6]) - ma[1] * (ma[4] * ma[10] - ma[8] * ma[6]) + ma[2] * (ma[4] * ma[9] - ma[8] * ma[5]);
    if det.abs() < 0.0005 {
        return IDENTITY_R;
    }
    let r = 1.0 / det;
    let mut d = [0.0; 12];
    d[0] = (ma[5] * ma[10] - ma[6] * ma[9]) * r;
    d[1] = (ma[9] * ma[2] - ma[1] * ma[10]) * r;
    d[2] = (ma[1] * ma[6] - ma[5] * ma[2]) * r;
    d[4] = (ma[6] * ma[8] - ma[4] * ma[10]) * r;
    d[5] = (ma[0] * ma[10] - ma[8] * ma[2]) * r;
    d[6] = (ma[4] * ma[2] - ma[0] * ma[6]) * r;
    d[8] = (ma[4] * ma[9] - ma[8] * ma[5]) * r;
    d[9] = (ma[8] * ma[1] - ma[0] * ma[9]) * r;
    d[10] = (ma[0] * ma[5] - ma[1] * ma[4]) * r;
    d
}

/// dClipEdgeToPlane (collision_util.cpp)
fn clip_edge_util(p0: &mut V3, p1: &mut V3, pl: &[Real; 4]) -> bool {
    clip_edge_to_plane(p0, p1, pl)
}

/// sCylinderBoxData
struct CylBox {
    crot: M3,
    cpos: V3,
    caxis: V3,
    cradius: Real,
    csize: Real,
    cnormals: [V3; 8],
    brot: M3,
    bpos: V3,
    bhalf: V3,
    bverts: [V3; 8],
    diff: V3,
    normal: V3,
    best_depth: Real,
    best_rb: Real,
    best_rc: Real,
    best_axis: i32,
}

impl CylBox {
    fn test_axis(&mut self, n: &mut V3, iaxis: i32) -> bool {
        let fl = length3(n);
        if fl < 1e-5 {
            return true;
        }
        normalize3(n);
        let d1 = dot3(&self.caxis, n);
        let frc = if d1 > 1.0 || d1 < -1.0 { self.csize * 0.5 } else { (d1 * (self.csize * 0.5)).abs() + self.cradius * (1.0 - d1 * d1).sqrt() };
        let mut frb = dot3(&get_column3(&self.brot, 0), n).abs() * self.bhalf[0];
        frb += dot3(&get_column3(&self.brot, 1), n).abs() * self.bhalf[1];
        frb += dot3(&get_column3(&self.brot, 2), n).abs() * self.bhalf[2];
        let fd = dot3(&self.diff, n);
        let mut depth = frc + frb;
        if fd.abs() > depth {
            return false;
        }
        depth -= fd.abs();
        if depth < self.best_depth {
            self.best_depth = depth;
            self.normal = *n;
            self.best_axis = iaxis;
            self.best_rb = frb;
            self.best_rc = frc;
            if fd > 0.0 {
                self.normal = neg3(&self.normal);
            }
        }
        true
    }
    fn test_edge_circle(&mut self, center: &V3, vx0: &V3, vx1: &V3, iaxis: i32) -> bool {
        let mut dir = sub3(vx1, vx0);
        normalize3(&mut dir);
        let start = *vx0;
        let d2 = dot3(&dir, &self.caxis);
        if d2.abs() < 1e-5 {
            return true;
        }
        let t1 = sub3(center, &start);
        let d1 = dot3(&t1, &self.caxis);
        let vp = [start[0] + dir[0] * (d1 / d2), start[1] + dir[1] * (d1 / d2), start[2] + dir[2] * (d1 / d2), 0.0];
        let t1 = sub3(center, &vp);
        let tangent = cross(&t1, &self.caxis);
        let mut axis = cross(&tangent, &dir);
        self.test_axis(&mut axis, iaxis)
    }
    fn separating_axes(&mut self) -> bool {
        self.best_depth = INFINITY;
        self.best_axis = 0;
        self.best_rb = 0.0;
        self.best_rc = 0.0;
        let eps = 1e-6;
        for k in 0..3 {
            let mut a = get_column3(&self.brot, k);
            if !self.test_axis(&mut a, 1 + k as i32) {
                return false;
            }
        }
        let mut a = self.caxis;
        if !self.test_axis(&mut a, 4) {
            return false;
        }
        for k in 0..3 {
            let mut a = [0.0; 4];
            cross3s(&mut a, &self.caxis, &self.brot[k..], 1, 1, 4);
            if length_sq3(&a) > eps && !self.test_axis(&mut a, 5 + k as i32) {
                return false;
            }
        }
        for i in 0..8 {
            let t1 = sub3(&self.bverts[i], &self.cpos);
            let t2 = cross(&self.caxis, &t1);
            let mut a = cross(&self.caxis, &t2);
            if length_sq3(&a) > eps && !self.test_axis(&mut a, 8 + i as i32) {
                return false;
            }
        }
        let edges = [(1, 0), (1, 3), (2, 3), (2, 0), (4, 1), (4, 7), (0, 7), (5, 3), (5, 6), (2, 6), (4, 5), (6, 7)];
        let h = self.csize * 0.5;
        for (cap, sign) in [(0, 1.0), (1, -1.0)] {
            let vcc = [self.cpos[0] + sign * self.caxis[0] * h, self.cpos[1] + sign * self.caxis[1] * h, self.cpos[2] + sign * self.caxis[2] * h, 0.0];
            // the minus cap is written as pos - axis * h
            let vcc = if sign < 0.0 { [self.cpos[0] - self.caxis[0] * h, self.cpos[1] - self.caxis[1] * h, self.cpos[2] - self.caxis[2] * h, 0.0] } else { vcc };
            for (k, &(a, b)) in edges.iter().enumerate() {
                let (va, vb) = (self.bverts[a], self.bverts[b]);
                if !self.test_edge_circle(&vcc, &va, &vb, 16 + cap * 12 + k as i32) {
                    return false;
                }
            }
        }
        true
    }
    fn clip_cylinder_to_box(&self, o1: &Prim, o2: &Prim, flags: i32, out: &mut Vec<ContactGeom>) -> bool {
        let t1 = dot3(&self.caxis, &self.normal);
        let mut vn = [self.normal[0] - self.caxis[0] * t1, self.normal[1] - self.caxis[1] * t1, self.normal[2] - self.caxis[2] * t1, 0.0];
        normalize3(&mut vn);
        let ct = [self.cpos[0] + vn[0] * self.cradius, self.cpos[1] + vn[1] * self.cradius, self.cpos[2] + vn[2] * self.cradius, 0.0];
        let h = self.csize * 0.5;
        let mut ep0 = [ct[0] + self.caxis[0] * h, ct[1] + self.caxis[1] * h, ct[2] + self.caxis[2] * h, 0.0];
        let mut ep1 = [ct[0] - self.caxis[0] * h, ct[1] - self.caxis[1] * h, ct[2] - self.caxis[2] * h, 0.0];
        for k in 0..3 {
            ep0[k] -= self.bpos[k];
        }
        for k in 0..3 {
            ep1[k] -= self.bpos[k];
        }
        for k in 0..3 {
            let c = get_column3(&self.brot, k);
            if !clip_edge_util(&mut ep0, &mut ep1, &mk_plane(&c, self.bhalf[k])) {
                return false;
            }
        }
        for k in 0..3 {
            let c = neg3(&get_column3(&self.brot, k));
            if !clip_edge_util(&mut ep0, &mut ep1, &mk_plane(&c, self.bhalf[k])) {
                return false;
            }
        }
        let mut d0 = self.best_rb + dot3(&ep0, &self.normal);
        let mut d1 = self.best_rb + dot3(&ep1, &self.normal);
        if d0 < 0.0 {
            d0 = 0.0;
        }
        if d1 < 0.0 {
            d1 = 0.0;
        }
        for k in 0..3 {
            ep0[k] += self.bpos[k];
        }
        for k in 0..3 {
            ep1[k] += self.bpos[k];
        }
        let mut c = cg(o1, o2);
        c.depth = d0;
        c.normal = neg3(&self.normal);
        c.pos = ep0;
        out.push(c);
        if out.len() as i32 != flags & NUMC_MASK {
            let mut c = cg(o1, o2);
            c.depth = d1;
            c.normal = neg3(&self.normal);
            c.pos = ep1;
            out.push(c);
        }
        true
    }
    fn clip_box_to_cylinder(&self, o1: &Prim, o2: &Prim, flags: i32, out: &mut Vec<ContactGeom>) {
        let h = self.csize * 0.5;
        let (cpos, crel) = if dot3(&self.caxis, &self.normal) > 0.0 {
            ([self.cpos[0] + self.caxis[0] * h, self.cpos[1] + self.caxis[1] * h, self.cpos[2] + self.caxis[2] * h, 0.0], [0.0, 0.0, -1.0, 0.0])
        } else {
            ([self.cpos[0] - self.caxis[0] * h, self.cpos[1] - self.caxis[1] * h, self.cpos[2] - self.caxis[2] * h, 0.0], [0.0, 0.0, 1.0, 0.0])
        };
        let binv = matrix3_inv(&self.brot);
        let nr = mul0_331(&binv, &self.normal);
        let an = [nr[0].abs(), nr[1].abs(), nr[2].abs()];
        let (b0, b1, b2) = if an[1] > an[0] {
            if an[0] > an[2] {
                (1, 0, 2)
            } else if an[1] > an[2] {
                (1, 2, 0)
            } else {
                (2, 1, 0)
            }
        } else if an[1] > an[2] {
            (0, 1, 2)
        } else if an[0] > an[2] {
            (0, 2, 1)
        } else {
            (2, 0, 1)
        };
        let t = get_column3(&self.brot, b0);
        let center = if nr[b0] > 0.0 {
            [self.bpos[0] - self.bhalf[b0] * t[0], self.bpos[1] - self.bhalf[b0] * t[1], self.bpos[2] - self.bhalf[b0] * t[2], 0.0]
        } else {
            [self.bpos[0] + self.bhalf[b0] * t[0], self.bpos[1] + self.bhalf[b0] * t[1], self.bpos[2] + self.bhalf[b0] * t[2], 0.0]
        };
        let a1 = get_column3(&self.brot, b1);
        let a2 = get_column3(&self.brot, b2);
        let (h1, h2) = (self.bhalf[b1], self.bhalf[b2]);
        let mut pts = [[0.0; 4]; 4];
        for k in 0..3 {
            pts[0][k] = center[k] + h1 * a1[k] - h2 * a2[k];
            pts[1][k] = center[k] - h1 * a1[k] - h2 * a2[k];
            pts[2][k] = center[k] - h1 * a1[k] + h2 * a2[k];
            pts[3][k] = center[k] + h1 * a1[k] + h2 * a2[k];
        }
        let cinv = matrix3_inv(&self.crot);
        for p in &mut pts {
            let t = sub3(p, &cpos);
            *p = mul0_331(&cinv, &t);
        }
        let mut t1 = Vec::new();
        let mut t2 = Vec::new();
        clip_poly_to_plane(&pts, &mut t1, &mk_plane(&crel, 0.0));
        for seg in 0..8 {
            let pl = mk_plane(&self.cnormals[seg], self.cradius);
            if seg % 2 == 0 {
                clip_poly_to_plane(&t1, &mut t2, &pl);
            } else {
                clip_poly_to_plane(&t2, &mut t1, &pl);
            }
        }
        // 8 segments: the result is back in t1
        for p in &t1 {
            let mut vp = mul0_331(&self.crot, p);
            vp[0] += cpos[0];
            vp[1] += cpos[1];
            vp[2] += cpos[2];
            let t = sub3(&vp, &self.cpos);
            let depth = self.best_rc - dot3(&t, &self.normal);
            if depth > 0.0 {
                let mut c = cg(o1, o2);
                c.depth = depth;
                c.normal = neg3(&self.normal);
                c.pos = vp;
                out.push(c);
                if out.len() as i32 == flags & NUMC_MASK {
                    break;
                }
            }
        }
    }
}

/// dCollideCylinderBox
fn collide_cylinder_box(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    let (cradius, csize) = cyl_params(o1.shape);
    let Shape::Box { side } = o2.shape else { return Vec::new() };
    let crot = o1.pr.r;
    let bhalf = [side[0] * 0.5, side[1] * 0.5, side[2] * 0.5, 0.0];
    let corners: [[Real; 3]; 8] = [[-1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]];
    let mut bverts = [[0.0; 4]; 8];
    for i in 0..8 {
        let v = [
            if corners[i][0] < 0.0 { -bhalf[0] } else { bhalf[0] },
            if corners[i][1] < 0.0 { -bhalf[1] } else { bhalf[1] },
            if corners[i][2] < 0.0 { -bhalf[2] } else { bhalf[2] },
            0.0,
        ];
        let t = mul0_331(&o2.pr.r, &v);
        bverts[i] = add3(&t, &o2.pr.pos);
    }
    let mut cnormals = [[0.0; 4]; 8];
    let mut ang = std::f64::consts::PI / 8.0;
    let inc = ang * 2.0;
    for n in &mut cnormals {
        *n = [-ang.cos(), -ang.sin(), 0.0, 0.0];
        ang += inc;
    }
    let mut c = CylBox {
        crot,
        cpos: o1.pr.pos,
        caxis: get_column3(&crot, 2),
        cradius,
        csize,
        cnormals,
        brot: o2.pr.r,
        bpos: o2.pr.pos,
        bhalf,
        bverts,
        diff: sub3(&o1.pr.pos, &o2.pr.pos),
        normal: [0.0; 4],
        best_depth: INFINITY,
        best_rb: 0.0,
        best_rc: 0.0,
        best_axis: 0,
    };
    if !c.separating_axes() || c.best_axis == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let fdot = dot3(&c.normal, &c.caxis);
    if fdot.abs() < 0.9 {
        if !c.clip_cylinder_to_box(o1, o2, flags, &mut out) {
            return Vec::new();
        }
    } else {
        c.clip_box_to_cylinder(o1, o2, flags, &mut out);
    }
    out
}

// ---- collision_cylinder_trimesh.cpp

/// dQuatTransform
fn quat_transform(q: &Quat, s: &V3) -> V3 {
    let x0 = s[0] * q[0] + s[2] * q[2] - s[1] * q[3];
    let x1 = s[1] * q[0] + s[0] * q[3] - s[2] * q[1];
    let x2 = s[2] * q[0] + s[1] * q[1] - s[0] * q[2];
    let x3 = s[0] * q[1] + s[1] * q[2] + s[2] * q[3];
    [q[0] * x0 + q[1] * x3 + q[2] * x2 - q[3] * x1, q[0] * x1 + q[2] * x3 + q[3] * x0 - q[1] * x2, q[0] * x2 + q[3] * x3 + q[1] * x1 - q[2] * x0, 0.0]
}
/// dQuatInv
fn quat_inv(s: &Quat) -> Quat {
    let norm = s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3];
    if norm > 0.0 {
        let n = -1.0 / norm;
        [-s[0] * n, s[1] * n, s[2] * n, s[3] * n]
    } else {
        IDENTITY_Q
    }
}

/// sCylinderTrimeshColliderData
struct CylTri {
    crot: M3,
    cq: Quat,
    cpos: V3,
    caxis: V3,
    cradius: Real,
    csize: Real,
    cnormals: [V3; 8],
    best_depth: Real,
    best_center: Real,
    best_rt: Real,
    best_axis: i32,
    cnormal: V3,
    normal: V3,
    e0: V3,
    e1: V3,
    e2: V3,
    flags: i32,
    local: Vec<LocalContact>,
}

impl CylTri {
    fn maxc(&self) -> usize {
        (self.flags & NUMC_MASK) as usize
    }
    fn test_axis(&mut self, v0: &V3, v1: &V3, v2: &V3, a: &mut V3, iaxis: i32, no_flip: bool) -> bool {
        let fl = length3(a);
        if fl < 1e-5 {
            return true;
        }
        a[0] /= fl;
        a[1] /= fl;
        a[2] /= fl;
        let d1 = dot3(&self.caxis, a);
        let frc = if d1.abs() > 1.0 { (self.csize * 0.5).abs() } else { ((self.csize * 0.5) * d1).abs() + self.cradius * (1.0 - d1 * d1).sqrt() };
        let afv = [dot3(&sub3(v0, &self.cpos), a), dot3(&sub3(v1, &self.cpos), a), dot3(&sub3(v2, &self.cpos), a)];
        let mut fmin = INFINITY;
        let mut fmax = -INFINITY;
        for &x in &afv {
            if x < fmin {
                fmin = x;
            }
            if x > fmax {
                fmax = x;
            }
        }
        let center = (fmin + fmax) * 0.5;
        let tri_r = (fmax - fmin) * 0.5;
        if center.abs() > frc + tri_r {
            return false;
        }
        let depth = -(center.abs() - (frc + tri_r));
        if depth < self.best_depth {
            self.best_depth = depth;
            self.best_center = center;
            self.best_rt = frc;
            self.cnormal = *a;
            self.best_axis = iaxis;
            if center < 0.0 && !no_flip {
                self.cnormal = neg3(&self.cnormal);
                self.best_center = -center;
            }
        }
        true
    }
    #[allow(clippy::too_many_arguments)]
    fn circle_to_edge(&mut self, v0: &V3, v1: &V3, v2: &V3, center: &V3, vx0: &V3, vx1: &V3, iaxis: i32) -> bool {
        let mut kl = sub3(vx1, vx0);
        normalize3(&mut kl);
        let ol = *vx0;
        let ax = self.caxis;
        let d2 = dot3(&kl, &ax);
        if d2.abs() < 1e-5 {
            return true;
        }
        let t = sub3(center, &ol);
        let d1 = dot3(&t, &ax);
        let vp = [ol[0] + kl[0] * d1 / d2, ol[1] + kl[1] * d1 / d2, ol[2] + kl[2] * d1 / d2, 0.0];
        let t = sub3(center, &vp);
        let tangent = cross(&t, &ax);
        let mut a = cross(&tangent, &kl);
        self.test_axis(v0, v1, v2, &mut a, iaxis, false)
    }
    fn separating_axes(&mut self, v0: &V3, v1: &V3, v2: &V3) -> bool {
        self.e0 = sub3(v1, v0);
        self.e2 = sub3(v0, v2);
        let h = self.csize * 0.5;
        let (p, ax) = (self.cpos, self.caxis);
        let cp0 = [p[0] + ax[0] * h, p[1] + ax[1] * h, p[2] + ax[2] * h, 0.0];
        self.best_axis = 0;
        let mut a = neg3(&self.normal);
        if !self.test_axis(v0, v1, v2, &mut a, 1, true) {
            return false;
        }
        for (k, e) in [self.e0, self.e1, self.e2].into_iter().enumerate() {
            let mut a = cross(&ax, &e);
            if !self.test_axis(v0, v1, v2, &mut a, 2 + k as i32, false) {
                return false;
            }
        }
        for (k, v) in [v0, v1, v2].into_iter().enumerate() {
            // _CalculateAxis(v, cp0, axis): ((v - cp0) x axis) x axis
            let t1 = sub3(v, &cp0);
            let t2 = cross(&t1, &ax);
            let mut a = cross(&t2, &ax);
            if !self.test_axis(v0, v1, v2, &mut a, 11 + k as i32, false) {
                return false;
            }
        }
        let mut a = ax;
        if !self.test_axis(v0, v1, v2, &mut a, 14, false) {
            return false;
        }
        let top = [p[0] + ax[0] * h, p[1] + ax[1] * h, p[2] + ax[2] * h, 0.0];
        let bottom = [p[0] - ax[0] * h, p[1] - ax[1] * h, p[2] - ax[2] * h, 0.0];
        let mut k = 15;
        for c in [top, bottom] {
            for (a0, a1) in [(v0, v1), (v1, v2), (v0, v2)] {
                if !self.circle_to_edge(v0, v1, v2, &c, a0, a1, k) {
                    return false;
                }
                k += 1;
            }
        }
        true
    }
    fn clip_edge_to_triangle(&mut self, v0: &V3) -> bool {
        let ax = self.caxis;
        let t = dot3(&ax, &self.cnormal);
        let mut n2 = [self.cnormal[0] - ax[0] * t, self.cnormal[1] - ax[1] * t, self.cnormal[2] - ax[2] * t, 0.0];
        let t = length3(&n2);
        if t < 1e-5 {
            return false;
        }
        n2[0] /= t;
        n2[1] /= t;
        n2[2] /= t;
        let r = self.cradius;
        let ct = [self.cpos[0] + n2[0] * r, self.cpos[1] + n2[1] * r, self.cpos[2] + n2[2] * r, 0.0];
        let h = self.csize * 0.5;
        let mut p0 = [ct[0] + ax[0] * h, ct[1] + ax[1] * h, ct[2] + ax[2] * h, 0.0];
        let mut p1 = [ct[0] - ax[0] * h, ct[1] - ax[1] * h, ct[2] - ax[2] * h, 0.0];
        for k in 0..3 {
            p0[k] -= v0[k];
        }
        for k in 0..3 {
            p1[k] -= v0[k];
        }
        let pn = neg3(&self.normal);
        if !clip_edge_util(&mut p0, &mut p1, &mk_plane(&pn, 0.0)) {
            return false;
        }
        let pn = cross(&self.normal, &self.e0);
        if !clip_edge_util(&mut p0, &mut p1, &mk_plane(&pn, 1e-5)) {
            return false;
        }
        let pn = cross(&self.normal, &self.e1);
        let t = dot3(&self.e0, &pn) - 1e-5;
        if !clip_edge_util(&mut p0, &mut p1, &mk_plane(&pn, -t)) {
            return false;
        }
        let pn = cross(&self.normal, &self.e2);
        if !clip_edge_util(&mut p0, &mut p1, &mk_plane(&pn, 1e-5)) {
            return false;
        }
        for k in 0..3 {
            p0[k] += v0[k];
        }
        for k in 0..3 {
            p1[k] += v0[k];
        }
        let t = sub3(&p0, &self.cpos);
        let rest0 = -dot3(&t, &self.cnormal) + self.best_rt;
        let t = sub3(&p1, &self.cpos);
        let rest1 = -dot3(&t, &self.cnormal) + self.best_rt;
        let mut d0 = self.best_depth - rest0;
        let mut d1 = self.best_depth - rest1;
        if d0 < 0.0 {
            d0 = 0.0;
        }
        if d1 < 0.0 {
            d1 = 0.0;
        }
        let n = self.cnormal;
        self.local.push(LocalContact { pos: p0, normal: n, depth: d0, tri: 0, keep: true });
        if self.local.len() >= self.maxc() {
            return true;
        }
        self.local.push(LocalContact { pos: p1, normal: n, depth: d1, tri: 0, keep: true });
        true
    }
    fn clip_to_triangle(&mut self, v0: &V3, v1: &V3, v2: &V3) {
        let ax = self.caxis;
        let h = self.csize * 0.5;
        let (cpos, crel) = if dot3(&ax, &self.cnormal) > 0.0 {
            ([self.cpos[0] + ax[0] * h, self.cpos[1] + ax[1] * h, self.cpos[2] + ax[2] * h, 0.0], [0.0, 0.0, -1.0, 0.0])
        } else {
            ([self.cpos[0] - ax[0] * h, self.cpos[1] - ax[1] * h, self.cpos[2] - ax[2] * h, 0.0], [0.0, 0.0, 1.0, 0.0])
        };
        let qinv = quat_inv(&self.cq);
        let mut pts = [*v0, *v1, *v2];
        for p in &mut pts {
            let t = sub3(p, &cpos);
            *p = quat_transform(&qinv, &t);
        }
        let mut t1 = Vec::new();
        let mut t2 = Vec::new();
        clip_poly_to_plane(&pts, &mut t1, &mk_plane(&crel, 0.0));
        for seg in 0..8 {
            let pl = mk_plane(&self.cnormals[seg], self.cradius);
            if seg % 2 == 0 {
                clip_poly_to_plane(&t1, &mut t2, &pl);
            } else {
                clip_poly_to_plane(&t2, &mut t1, &pl);
            }
        }
        for p in &t1 {
            let mut vp = quat_transform(&self.cq, p);
            vp[0] += cpos[0];
            vp[1] += cpos[1];
            vp[2] += cpos[2];
            let t = sub3(&vp, &self.cpos);
            let depth = self.best_rt - dot3(&t, &self.cnormal).abs();
            if depth > 0.0 {
                let n = self.cnormal;
                self.local.push(LocalContact { pos: vp, normal: n, depth, tri: 0, keep: true });
                if self.local.len() >= self.maxc() {
                    return;
                }
            }
        }
    }
    fn test_triangle(&mut self, v0: &V3, v1: &V3, v2: &V3) {
        self.e1 = sub3(v2, v1);
        let t = sub3(v0, v1);
        self.normal = cross(&self.e1, &t);
        if !safe_normalize3(&mut self.normal) {
            return;
        }
        let pl = mk_plane(&self.normal, -dot3(v0, &self.normal));
        let dist = plane_dist(&self.cpos, &pl);
        if dist < 0.0 {
            return;
        }
        self.best_depth = INFINITY;
        if !self.separating_axes(v0, v1, v2) || self.best_axis == 0 {
            return;
        }
        let fdot = dot3(&self.cnormal, &self.caxis);
        if fdot.abs() < 0.9 {
            self.clip_edge_to_triangle(v0);
        } else {
            self.clip_to_triangle(v0, v1, v2);
        }
    }
}

/// dCollideCylinderTrimesh (o1 cylinder, o2 trimesh)
fn collide_cylinder_trimesh(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    use super::volume::{ObbVol, VolumeQuery};
    let (cradius, csize) = cyl_params(o1.shape);
    let d = mesh_of(o2.shape);
    let mut cnormals = [[0.0; 4]; 8];
    let mut ang = std::f64::consts::PI / 8.0;
    let inc = ang * 2.0;
    for n in &mut cnormals {
        *n = [-ang.cos(), -ang.sin(), 0.0, 0.0];
        ang += inc;
    }
    let crot = o1.pr.r;
    let mut c = CylTri {
        crot,
        cq: o1.q,
        cpos: o1.pr.pos,
        caxis: get_column3(&crot, 2),
        cradius,
        csize,
        cnormals,
        best_depth: 0.0,
        best_center: 0.0,
        best_rt: 0.0,
        best_axis: 0,
        cnormal: [0.0; 4],
        normal: [0.0; 4],
        e0: [0.0; 4],
        e1: [0.0; 4],
        e2: [0.0; 4],
        flags,
        local: Vec::new(),
    };
    let _ = &c.crot;
    let (mpos, mrot) = (&o2.pr.pos, &o2.pr.r);
    let mesh_matrix = opcode::make_matrix(&[0.0; 3], mrot);
    let off = sub3(&c.cpos, mpos);
    let r = &crot;
    let rot = [[r[0] as f32, r[4] as f32, r[8] as f32], [r[1] as f32, r[5] as f32, r[9] as f32], [r[2] as f32, r[6] as f32, r[10] as f32]];
    let half = csize * 0.5;
    let vol = ObbVol::new([off[0] as f32, off[1] as f32, off[2] as f32], [cradius as f32, cradius as f32, half as f32], rot, &mesh_matrix);
    let mut q = VolumeQuery::new(vol, &d.mesh);
    q.run(&d.model);
    if q.touched.is_empty() {
        return Vec::new();
    }
    let maxc = (flags & NUMC_MASK) as usize;
    for &tri in &q.touched {
        let dv = fetch_triangle(d, tri as usize, mpos, mrot);
        let first = c.local.len();
        c.test_triangle(&dv[0], &dv[1], &dv[2]);
        for lc in &mut c.local[first..] {
            lc.tri = tri as i32;
        }
        if c.local.len() >= maxc {
            break;
        }
    }
    let n = c.local.len();
    if n > 1 && flags & CONTACTS_UNIMPORTANT == 0 {
        for i in 0..n - 1 {
            for j in i + 1..n {
                if near_contacts(&c.local[i], &c.local[j]) {
                    if c.local[j].depth > c.local[i].depth {
                        c.local[i].keep = false;
                    } else {
                        c.local[j].keep = false;
                    }
                }
            }
        }
    }
    c.local
        .iter()
        .filter(|lc| lc.keep)
        .map(|lc| ContactGeom { pos: lc.pos, normal: neg3(&lc.normal), depth: lc.depth, g1: o1.id, g2: o2.id, side1: -1, side2: lc.tri })
        .collect()
}

// ---- convex.cpp (the non-libccd colliders; convex-box and convex-capsule
// are stubs returning nothing in that build, convex-trimesh likewise)

/// dxConvex: planes (a, b, c, d per face, body frame), points (x, y, z) and
/// polygons ([n, i0 .. in-1] per face)
#[derive(Debug)]
pub struct ConvexData {
    pub planes: Vec<Real>,
    pub points: Vec<Real>,
    pub polygons: Vec<u32>,
    pub edges: Vec<(u32, u32)>,
}
impl ConvexData {
    /// dCreateConvex (FillEdges)
    pub fn new(planes: Vec<Real>, points: Vec<Real>, polygons: Vec<u32>) -> Arc<ConvexData> {
        let planecount = planes.len() / 4;
        let mut edges: Vec<(u32, u32)> = Vec::new();
        let mut at = 0usize;
        for _ in 0..planecount {
            let n = polygons[at] as usize;
            let idx = &polygons[at + 1..at + 1 + n];
            for j in 0..n {
                let (a, b) = (idx[j], idx[(j + 1) % n]);
                let e = (if a > b { b } else { a }, if a > b { a } else { b });
                if !edges.contains(&e) {
                    edges.push(e);
                }
            }
            at += n + 1;
        }
        Arc::new(ConvexData { planes, points, polygons, edges })
    }
    fn planecount(&self) -> usize {
        self.planes.len() / 4
    }
    fn pointcount(&self) -> usize {
        self.points.len() / 3
    }
    fn point(&self, i: usize) -> V3 {
        [self.points[i * 3], self.points[i * 3 + 1], self.points[i * 3 + 2], 0.0]
    }
    fn plane(&self, i: usize) -> [Real; 4] {
        [self.planes[i * 4], self.planes[i * 4 + 1], self.planes[i * 4 + 2], self.planes[i * 4 + 3]]
    }
    /// start index of face `i` in `polygons`
    fn poly(&self, i: usize) -> usize {
        let mut at = 0;
        for _ in 0..i {
            at += self.polygons[at] as usize + 1;
        }
        at
    }
}

fn convex_of(s: &Shape) -> &ConvexData {
    match s {
        Shape::Convex(c) => c,
        _ => panic!("not a convex"),
    }
}

fn convex_aabb(c: &ConvexData, pr: &PosR) -> [Real; 6] {
    let (r, pos) = (&pr.r, &pr.pos);
    let p = mul0_331(r, &c.point(0));
    let mut a = [p[0] + pos[0], p[0] + pos[0], p[1] + pos[1], p[1] + pos[1], p[2] + pos[2], p[2] + pos[2]];
    for i in 1..c.pointcount() {
        let p = mul0_331(r, &c.point(i));
        for k in 0..3 {
            let v = p[k] + pos[k];
            a[2 * k] = if a[2 * k] > v { v } else { a[2 * k] };
            a[2 * k + 1] = if a[2 * k + 1] > v { a[2 * k + 1] } else { v };
        }
    }
    a
}

/// world vertex i of a convex
fn cvx_world(c: &ConvexData, pr: &PosR, i: usize) -> V3 {
    let b = mul0_331(&pr.r, &c.point(i));
    [pr.pos[0] + b[0], pr.pos[1] + b[1], pr.pos[2] + b[2], 0.0]
}

/// IsPointInPolygon: Ok(()) inside, Err(closest point on the offending edge)
fn point_in_polygon(p: &V3, c: &ConvexData, pr: &PosR, poly: usize, plane: &[Real]) -> Result<(), V3> {
    let n = c.polygons[poly] as usize;
    let idx = &c.polygons[poly + 1..poly + 1 + n];
    let mut b = cvx_world(c, pr, idx[n - 1] as usize);
    for i in 0..n {
        let a = b;
        b = cvx_world(c, pr, idx[i] as usize);
        let ab = sub3(&b, &a);
        let ap = sub3(p, &a);
        let v = cross(&ab, plane);
        if dot3(&ap, &v) > 0.0 {
            let ab_m2 = dot3(&ab, &ab);
            let s = if ab_m2 != 0.0 { dot3(&ab, &ap) / ab_m2 } else { 0.0 };
            let out = if s <= 0.0 {
                a
            } else if s >= 1.0 {
                b
            } else {
                [a[0] + ab[0] * s, a[1] + ab[1] * s, a[2] + ab[2] * s, 0.0]
            };
            return Err(out);
        }
    }
    Ok(())
}

/// dCollideConvexPlane
fn collide_convex_plane(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    let c = convex_of(o1.shape);
    let Shape::Plane { p } = o2.shape else { return Vec::new() };
    let maxc = (flags & NUMC_MASK) as u32;
    let mut out = Vec::new();
    let mut total = 0u32;
    for i in 0..c.pointcount() {
        let v = mul0_331(&o1.pr.r, &c.point(i));
        let v = add3(&o1.pr.pos, &v);
        let mut sign = 0x2000_0000u32;
        let d2 = dot3(p, &v) - p[3];
        if d2 <= 0.0 {
            sign = if d2 != 0.0 { 0x1000_0000 } else { 0x3000_0000 };
            if out.len() as u32 != maxc {
                let mut ct = cg(o1, o2);
                ct.normal = [p[0], p[1], p[2], 0.0];
                ct.pos = v;
                ct.depth = -d2;
                out.push(ct);
            }
        }
        total |= sign;
        if ((out.len() as u32 ^ maxc) | total) == 0x3000_0000 {
            break;
        }
    }
    if total == 0x3000_0000 {
        out
    } else {
        Vec::new()
    }
}

/// dCollideSphereConvex
fn collide_sphere_convex(o1: &Prim, o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    let radius = radius_of(o1.shape);
    let c = convex_of(o2.shape);
    let sp = &o1.pr.pos;
    let off = sub3(sp, &o2.pr.pos);
    let mut closest = INFINITY;
    let mut closest_plane = 0usize;
    let mut inside = true;
    let mut poly = 0usize;
    for i in 0..c.planecount() {
        let mut plane = mul0_331(&o2.pr.r, &c.planes[i * 4..]);
        plane[3] = c.planes[i * 4 + 3];
        let dist = dot3(&plane, &off) - plane[3];
        if dist > 0.0 {
            if dist < radius {
                match point_in_polygon(sp, c, &o2.pr, poly, &plane) {
                    Ok(()) => {
                        let mut ct = cg(o1, o2);
                        ct.normal = [plane[0], plane[1], plane[2], 0.0];
                        for k in 0..3 {
                            ct.pos[k] = sp[k] + (-ct.normal[k] * radius);
                        }
                        ct.depth = radius - dist;
                        return vec![ct];
                    }
                    Err(out) => {
                        let t = [sp[0] - out[0], sp[1] - out[1], sp[2] - out[2]];
                        let d = (t[0] * t[0]) + (t[1] * t[1]) + (t[2] * t[2]);
                        if d < radius * radius {
                            let d = d.sqrt();
                            let mut ct = cg(o1, o2);
                            ct.normal = [t[0] / d, t[1] / d, t[2] / d, 0.0];
                            for k in 0..3 {
                                ct.pos[k] = sp[k] + (-ct.normal[k] * radius);
                            }
                            ct.depth = radius - d;
                            return vec![ct];
                        }
                    }
                }
            }
            inside = false;
        }
        if inside && closest > dist.abs() {
            closest = dist.abs();
            closest_plane = i;
        }
        poly += c.polygons[poly] as usize + 1;
    }
    if inside {
        let mut ct = cg(o1, o2);
        ct.normal = mul0_331(&o2.pr.r, &c.planes[closest_plane * 4..]);
        ct.pos = [sp[0], sp[1], sp[2], 0.0];
        ct.depth = closest + radius;
        return vec![ct];
    }
    Vec::new()
}

fn collide_convex_box(_o1: &Prim, _o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    Vec::new()
}
fn collide_convex_capsule(_o1: &Prim, _o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    Vec::new()
}
fn collide_convex_trimesh(_o1: &Prim, _o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    Vec::new()
}

/// IntersectSegmentPlane
fn intersect_segment_plane(a: &V3, b: &V3, p: &[Real; 4]) -> Option<(Real, V3)> {
    let ab = sub3(b, a);
    let t = (p[3] - dot3(p, a)) / dot3(p, &ab);
    if (0.0..=1.0).contains(&t) {
        Some((t, [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2], 0.0]))
    } else {
        None
    }
}

fn clamp01(x: Real) -> Real {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// ClosestPointBetweenSegments: squared distance and the two points
fn closest_between_segments(p1: &V3, q1: &V3, p2: &V3, q2: &V3) -> (Real, V3, V3) {
    let d1 = sub3(q1, p1);
    let d2 = sub3(q2, p2);
    let r = sub3(p1, p2);
    let a = dot3(&d1, &d1);
    let e = dot3(&d2, &d2);
    let f = dot3(&d2, &r);
    let eps = f64::EPSILON;
    let (s, t);
    let sq = |c1: &V3, c2: &V3| (c1[0] - c2[0]) * (c1[0] - c2[0]) + (c1[1] - c2[1]) * (c1[1] - c2[1]) + (c1[2] - c2[2]) * (c1[2] - c2[2]);
    if a <= eps && e <= eps {
        return (sq(p1, p2), *p1, *p2);
    }
    if a <= eps {
        s = 0.0;
        t = clamp01(f / e);
    } else {
        let c = dot3(&d1, &r);
        if e <= eps {
            t = 0.0;
            s = clamp01(-c / a);
        } else {
            let b = dot3(&d1, &d2);
            let denom = a * e - b * b;
            let mut ss = if denom != 0.0 { clamp01((b * f - c * e) / denom) } else { 0.0 };
            let tnom = b * ss + f;
            let tt;
            if tnom < 0.0 {
                tt = 0.0;
                ss = clamp01(-c / a);
            } else if tnom > e {
                tt = 1.0;
                ss = clamp01((b - c) / a);
            } else {
                tt = tnom / e;
            }
            s = ss;
            t = tt;
        }
    }
    let c1 = [p1[0] + d1[0] * s, p1[1] + d1[1] * s, p1[2] + d1[2] * s, 0.0];
    let c2 = [p2[0] + d2[0] * t, p2[1] + d2[1] * t, p2[2] + d2[2] * t, 0.0];
    (sq(&c1, &c2), c1, c2)
}

/// ComputeInterval
fn compute_interval(c: &ConvexData, pr: &PosR, axis: &[Real; 4]) -> (Real, Real) {
    let p = cvx_world(c, pr, 0);
    let mut mn = dot3(&p, axis) - axis[3];
    let mut mx = mn;
    for i in 1..c.pointcount() {
        let p = cvx_world(c, pr, i);
        let v = dot3(&p, axis) - axis[3];
        if v < mn {
            mn = v;
        } else if v > mx {
            mx = v;
        }
    }
    (mn, mx)
}

/// a face plane of a convex in world space (normalised, offset moved)
fn cvx_world_plane(c: &ConvexData, pr: &PosR, i: usize) -> [Real; 4] {
    let mut p = mul0_331(&pr.r, &c.planes[i * 4..]);
    normalize3(&mut p);
    let d = c.planes[i * 4 + 3] + ((p[0] * pr.pos[0]) + (p[1] * pr.pos[1]) + (p[2] * pr.pos[2]));
    [p[0], p[1], p[2], d]
}

struct Ccso {
    min_depth: Real,
    depth_type: i32,
    dist: V3,
    e1a: V3,
    e1b: V3,
    e2a: V3,
    e2b: V3,
}

fn sat_faces(c1: &ConvexData, p1: &PosR, c2: &ConvexData, p2: &PosR, o: &mut Ccso) -> bool {
    for i in 0..c1.planecount() {
        let plane = cvx_world_plane(c1, p1, i);
        let (min1, max1) = compute_interval(c1, p1, &plane);
        let (min2, max2) = compute_interval(c2, p2, &plane);
        if max2 < min1 || max1 < min2 {
            return false;
        }
        let mn = if min1 > min2 { min1 } else { min2 };
        let mx = if max1 > max2 { max2 } else { max1 };
        let depth = mx - mn;
        if (max2 * min2) <= 0.0 && depth.abs() < o.min_depth.abs() {
            o.min_depth = -depth;
            o.depth_type = 1;
        }
    }
    true
}

/// SupportIndex
fn support_index(c: &ConvexData, pr: &PosR, dir: &V3) -> usize {
    let rdir = mul1_331(&pr.r, dir);
    let mut idx = 0;
    let mut mx = dot3(&c.points[0..], &rdir);
    for i in 1..c.pointcount() {
        let t = dot3(&c.points[i * 3..], &rdir);
        if t > mx {
            idx = i;
            mx = t;
        }
    }
    idx
}

fn sat_edges(c1: &ConvexData, p1: &PosR, c2: &ConvexData, p2: &PosR, o: &mut Ccso) -> bool {
    let mut dist = o.dist;
    let s1 = support_index(c1, p1, &dist) as u32;
    dist = neg3(&dist);
    let s2 = support_index(c2, p2, &dist) as u32;
    for &(f1, s1b) in &c1.edges {
        if f1 != s1 && s1b != s1 {
            continue;
        }
        let e1a = mul0_331(&p1.r, &c1.point(f1 as usize));
        let e1b = mul0_331(&p1.r, &c1.point(s1b as usize));
        let e1 = sub3(&e1b, &e1a);
        for &(f2, s2b) in &c2.edges {
            if f2 != s2 && s2b != s2 {
                continue;
            }
            let e2a = mul0_331(&p2.r, &c2.point(f2 as usize));
            let e2b = mul0_331(&p2.r, &c2.point(s2b as usize));
            let e2 = sub3(&e2b, &e2a);
            let mut pl = cross(&e1, &e2);
            if dot3(&pl, &pl) < f64::EPSILON {
                continue;
            }
            normalize3(&mut pl);
            let plane = [pl[0], pl[1], pl[2], 0.0];
            let (min1, max1) = compute_interval(c1, p1, &plane);
            let (min2, max2) = compute_interval(c2, p2, &plane);
            if max2 < min1 || max1 < min2 {
                return false;
            }
            let mn = if min1 > min2 { min1 } else { min2 };
            let mx = if max1 > max2 { max2 } else { max1 };
            let depth = mx - mn;
            if (depth.abs() + f64::EPSILON) < o.min_depth.abs() {
                o.min_depth = depth;
                o.depth_type = 2;
                o.e1a = add3(&e1a, &p1.pos);
                o.e1b = add3(&e1b, &p1.pos);
                o.e2a = add3(&e2a, &p2.pos);
                o.e2b = add3(&e2b, &p2.pos);
            }
        }
    }
    true
}

/// GetSupportSide
fn support_side(dir: &V3, c: &ConvexData, pr: &PosR) -> usize {
    let mut t = *dir;
    normalize3(&mut t);
    let dics = mul1_331(&pr.r, &t);
    let mut saved = dot3(&dics, &c.planes[0..]);
    let mut side = 0;
    for i in 1..c.planecount() {
        let d = dot3(&dics, &c.planes[i * 4..]);
        if d > saved {
            saved = d;
            side = i;
        }
    }
    side
}

/// dCollideConvexConvex (TestConvexIntersection)
fn collide_convex_convex(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    let (c1, c2) = (convex_of(o1.shape), convex_of(o2.shape));
    let (p1, p2) = (&o1.pr, &o2.pr);
    let mut o = Ccso { min_depth: INFINITY, depth_type: 0, dist: sub3(&p2.pos, &p1.pos), e1a: [0.0; 4], e1b: [0.0; 4], e2a: [0.0; 4], e2b: [0.0; 4] };
    let maxc = (flags & NUMC_MASK) as usize;
    if !sat_faces(c1, p1, c2, p2, &mut o) || !sat_faces(c2, p2, c1, p1, &mut o) || !sat_edges(c1, p1, c2, p2, &mut o) {
        return Vec::new();
    }
    let mut out: Vec<ContactGeom> = Vec::new();
    let push = |out: &mut Vec<ContactGeom>, pos: V3, n: &[Real; 4], d: Real| {
        let mut ct = cg(o1, o2);
        ct.pos = pos;
        ct.normal = [n[0], n[1], n[2], 0.0];
        ct.depth = d;
        out.push(ct);
    };
    if o.depth_type == 1 {
        let mut dist = o.dist;
        let reference = support_side(&dist, c1, p1);
        dist = neg3(&dist);
        let incident = support_side(&dist, c2, p2);
        let mut rplane = cvx_world_plane(c1, p1, reference);
        for k in 0..4 {
            rplane[k] = -rplane[k];
        }
        let ip = c2.poly(incident);
        let n_inc = c2.polygons[ip] as usize;
        let ipts = &c2.polygons[ip + 1..ip + 1 + n_inc];
        let mut i2 = cvx_world(c2, p2, ipts[0] as usize);
        let mut r2 = mul1_331(&p1.r, &sub3(&i2, &p1.pos));
        for i in 0..n_inc {
            let i1 = i2;
            let r1 = r2;
            i2 = cvx_world(c2, p2, ipts[(i + 1) % n_inc] as usize);
            r2 = mul1_331(&p1.r, &sub3(&i2, &p1.pos));
            let mut outside = false;
            for j in 0..c1.planecount() {
                let plane = c1.plane(j);
                let d1 = r1[0] * plane[0] + r1[1] * plane[1] + r1[2] * plane[2] - plane[3];
                let d2 = r2[0] * plane[0] + r2[1] * plane[1] + r2[2] * plane[2] - plane[3];
                if d1 * d2 < 0.0 {
                    let mut point = None;
                    if let Some((_, p)) = intersect_segment_plane(&r1, &r2, &plane) {
                        let mut inside = true;
                        for k in 0..c1.planecount() {
                            let pk = c1.plane(k);
                            let d = p[0] * pk[0] + p[1] * pk[1] + p[2] * pk[2] - pk[3];
                            if d > 0.0 {
                                inside = false;
                                break;
                            }
                        }
                        if inside {
                            point = Some(p);
                        }
                    }
                    if let Some(p) = point {
                        let t = mul0_331(&p1.r, &p);
                        let p = add3(&t, &p1.pos);
                        let d = p[0] * rplane[0] + p[1] * rplane[1] + p[2] * rplane[2] - rplane[3];
                        if d > 0.0 {
                            push(&mut out, p, &rplane, d);
                            if out.len() == maxc {
                                return out;
                            }
                        }
                    }
                }
                if d1 > 0.0 {
                    outside = true;
                }
            }
            if outside {
                continue;
            }
            let d = i1[0] * rplane[0] + i1[1] * rplane[1] + i1[2] * rplane[2] - rplane[3];
            if d > 0.0 {
                push(&mut out, i1, &rplane, d);
                if out.len() == maxc {
                    return out;
                }
            }
        }
        let iplane = cvx_world_plane(c2, p2, incident);
        let rp = c1.poly(reference);
        let n_ref = c1.polygons[rp] as usize;
        for i in 0..n_ref {
            let mut i1 = cvx_world(c1, p1, c1.polygons[rp + 1 + i] as usize);
            // dVector3Add(pos, i1) rather than (i1, pos): same sum
            let t = -(i1[0] * iplane[0] + i1[1] * iplane[1] + i1[2] * iplane[2] - iplane[3]);
            i1[0] += iplane[0] * t;
            i1[1] += iplane[1] * t;
            i1[2] += iplane[2] * t;
            let r1 = mul1_331(&p2.r, &sub3(&i1, &p2.pos));
            let mut outp = false;
            for j in 0..c2.planecount() {
                let pj = c2.plane(j);
                let d = r1[0] * pj[0] + r1[1] * pj[1] + r1[2] * pj[2] - pj[3];
                if d >= 0.0 {
                    outp = true;
                    break;
                }
            }
            if !outp {
                let dup = out.iter().any(|c| c.pos[0] == i1[0] && c.pos[1] == i1[1] && c.pos[2] == i1[2]);
                if !dup {
                    let d = i1[0] * rplane[0] + i1[1] * rplane[1] + i1[2] * rplane[2] - rplane[3];
                    if d > 0.0 {
                        push(&mut out, i1, &rplane, d);
                        if out.len() == maxc {
                            return out;
                        }
                    }
                }
            }
        }
    } else if o.depth_type == 2 {
        let (sq, c1p, c2p) = closest_between_segments(&o.e1a, &o.e1b, &o.e2a, &o.e2b);
        let mut ct = cg(o1, o2);
        ct.depth = sq.sqrt();
        ct.pos = c1p;
        ct.normal = sub3(&c2p, &c1p);
        normalize3(&mut ct.normal);
        out.push(ct);
    }
    out
}

/// dCollideRayConvex (planes are used as given, in the convex frame, as
/// ODE does)
fn collide_ray_convex(o1: &Prim, o2: &Prim, flags: i32) -> Vec<ContactGeom> {
    let c = convex_of(o2.shape);
    let (rp, rr) = (&o1.pr.pos, &o1.pr.r);
    let len = ray_len(o1.shape);
    let mut flag = false;
    for i in 0..c.planecount() {
        let plane = &c.planes[i * 4..];
        if dot3(plane, rp) - plane[3] >= 0.0 {
            flag = true;
            break;
        }
    }
    let nsign = if flag { 1.0 } else { -1.0 };
    let mut ct = cg(o1, o2);
    ct.depth = INFINITY;
    for i in 0..c.planecount() {
        let plane = &c.planes[i * 4..i * 4 + 4];
        let alpha = nsign * (dot3(plane, rp) - plane[3]);
        // dCalcVectorDot3_13: ODE reads R[2], R[5], R[8] here
        let beta = dot3s(plane, &rr[2..], 1, 3) * nsign;
        if beta < -f64::EPSILON && alpha >= 0.0 && alpha <= len && alpha < ct.depth {
            let pos = [rp[0] + alpha * rr[2], rp[1] + alpha * rr[6], rp[2] + alpha * rr[10], 0.0];
            ct.pos = pos;
            let mut out = false;
            for j in 0..c.planecount() {
                if i == j {
                    continue;
                }
                let pj = &c.planes[j * 4..];
                if dot3(pj, &pos) - plane[3] > f64::EPSILON {
                    out = true;
                    break;
                }
            }
            if !out {
                ct.normal = [nsign * plane[0], nsign * plane[1], nsign * plane[2], 0.0];
                ct.depth = alpha;
                if flags & CONTACTS_UNIMPORTANT != 0 && ct.depth <= len {
                    break;
                }
            }
        }
    }
    if ct.depth <= len {
        vec![ct]
    } else {
        Vec::new()
    }
}

/// dCollideTTL: not ported yet
fn collide_trimesh_trimesh(_o1: &Prim, _o2: &Prim, _flags: i32) -> Vec<ContactGeom> {
    Vec::new()
}
