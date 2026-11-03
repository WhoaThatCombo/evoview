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
            Shape::Space(s) => match s.kind {
                SpaceKind::Simple => SIMPLE_SPACE,
                SpaceKind::Hash { .. } => HASH_SPACE,
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
        self.geoms[g].parent = Some(s);
        self.space_mut(s).list.insert(0, g);
        self.geom_moved(s);
    }
    /// dSpaceRemove
    pub fn space_remove(&mut self, s: GeomId, g: GeomId) {
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
            // dxSpace::dirty: back to the head of the list
            let list = &mut self.space_mut(p).list;
            if let Some(k) = list.iter().position(|&x| x == g) {
                list.remove(k);
            }
            list.insert(0, g);
            g = p;
            parent = self.geoms[p].parent;
        }
        let mut cur = Some(g);
        while let Some(c) = cur {
            self.mark_aabb_bad(c);
            cur = self.geoms[c].parent;
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
            Shape::Space(sp) => {
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
            shape => {
                let pr = self.geom_posr(g);
                shape_aabb(shape, &pr)
            }
        };
        let gg = &mut self.geoms[g];
        gg.aabb = aabb;
        gg.gflags &= !GEOM_AABB_BAD;
    }

    fn clean_geoms(&mut self, s: GeomId) {
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
        let Some((reverse, f)) = collider(c1, c2) else { return out };
        let (a, b) = if reverse { (o2, o1) } else { (o1, o2) };
        let ga = Prim { shape: &self.geoms[a].shape, pr: self.geom_posr(a), id: a, flags: self.geoms[a].gflags };
        let gb = Prim { shape: &self.geoms[b].shape, pr: self.geom_posr(b), id: b, flags: self.geoms[b].gflags };
        out = f(&ga, &gb, flags);
        if reverse {
            for c in &mut out {
                c.normal[0] = -c.normal[0];
                c.normal[1] = -c.normal[1];
                c.normal[2] = -c.normal[2];
                std::mem::swap(&mut c.g1, &mut c.g2);
                std::mem::swap(&mut c.side1, &mut c.side2);
            }
        }
        out
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
        Shape::Space(_) => [0.0; 6],
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
}
type ColliderFn = fn(&Prim, &Prim, i32) -> Vec<ContactGeom>;

/// dInitColliders' table: (reverse, function) for a class pair
fn collider(c1: usize, c2: usize) -> Option<(bool, ColliderFn)> {
    // registration order, first wins (setCollider)
    let table: [(usize, usize, ColliderFn); 15] = [
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
        (TRIMESH, SPHERE, collide_sphere_trimesh),
        (TRIMESH, RAY, collide_ray_trimesh),
    ];
    let _ = NUM_CLASSES;
    for &(i, j, f) in &table {
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
