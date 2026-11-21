//! CPU frustum + screen-size culling for static scenery.
//!
//! Tracks are vertex bound, not fill bound: Brands Hatch submits ~78 M
//! instanced triangles a frame (barriers, tyre walls, trees, grass) whether
//! or not they are on screen, and the frame time does not move between
//! 1600x900 and 400x225. Culling each INSTANCE against the view frustum and
//! dropping ones smaller than a pixel cuts that without touching the shader,
//! the vertex layout or the bind groups: visible instances are compacted into
//! the mesh's existing instance buffer and the draw uses the shorter count.
//!
//! Only static meshes are culled. Skinned car parts move with animation and
//! replay cars rewrite their own instance buffers, so both are left alone.

use crate::asset::Car;
use crate::render::GpuMesh;
use glam::{Mat4, Vec3, Vec4};
use std::collections::HashMap;
use std::sync::Arc;

/// Objects whose projected radius is below this many pixels are dropped.
/// Half a pixel: at that size a mesh covers at most a sample or two and
/// its triangles are pure vertex cost.
const MIN_PIXELS: f32 = 0.5;

struct Entry {
    gpu: usize,
    /// world-space bounding spheres, one per instance (xyz centre, w radius)
    spheres: Vec<Vec4>,
    /// The instance rows (16 matrix floats + facing, bone base, wind
    /// height bounds), same order, as the placements' matrices - ONE list
    /// (the same Arc) for every LOD level of a prop standing at the same
    /// places, not 80 B per row per level (2.64 M rows for the
    /// Nordschleife's 527 k placements) - and this mesh's floats 16..20: one
    /// for all rows, or one per row (an animated prop's variant per instance).
    place: Arc<Vec<[f32; 16]>>,
    tail: Vec<[f32; 4]>,
    /// Shadow caster rows past `place`: placements of levels that send their
    /// casters here (shadow_to) where this mesh itself does not stand.
    extra: Vec<[f32; 16]>,
    extra_tail: Vec<[f32; 4]>,
    /// This level's instance -> its caster's row in the shadow_to mesh, when
    /// that does not stand at the same places (None: the same index).
    cast_map: Option<Vec<u32>>,
    /// visible instances last written, to skip identical uploads
    last: Vec<u32>,
    /// The shadow pass sees a different set (the light's box, not the
    /// camera's frustum), so it gets its own compacted buffer. Created on
    /// first use, only for instanced meshes.
    /// per shadow cascade (0 near, 1 far)
    /// per extra view: 0 near shadow, 1 far shadow, 2 mirror, 3 reflection probe, 4 rear-view camera,
    /// 5 middle shadow cascade
    shadow_buf: [Option<wgpu::Buffer>; 6],
    shadow_count: [u32; 6],
    /// the instances last uploaded per view, to skip identical uploads
    shadow_last: [Vec<u32>; 6],
    /// squared [near, far) distance band when this is one LOD of a prop,
    /// x the current lodDistanceScale
    lod: Option<[f32; 2]>,
    /// that band and its fades in raw MeshLodData metres (lod_fade layout)
    raw_lod: Option<[f32; 2]>,
    raw_fade: [f32; 4],
    /// shadow cascades: whose buffer this LOD's casters go to (MeshEntry
    /// shadow_mesh as an entry index): None = its own, Some(None) = none
    shadow_to: Option<Option<usize>>,
    /// that, still as a Car::meshes index (resolved after build)
    shadow_src: Option<Option<usize>>,
    /// the dithered crossfade around that band (camera view only)
    band: Option<Band>,
    /// per visible instance last written: its dither-out amount (row[3])
    last_fade: Vec<f32>,
    /// Instance indices grouped by grid cell (Grid), each run in ascending
    /// order: (cell, start, end) into `order`. Empty for meshes with few
    /// instances, which are tested one by one.
    runs: Vec<(u32, u32, u32)>,
    /// per run: its largest instance radius (a run too small to show is
    /// skipped whole, see `Tiny`)
    run_r: Vec<f32>,
    order: Vec<u32>,
    /// mesh name (EVOVIEW_LOD_LOG)
    name: String,
}

impl Entry {
    /// Rows a shadow caster buffer of this mesh can hold: its own and the
    /// casters' `extra`.
    fn cast_len(&self) -> usize {
        // spheres: one per placement, in the Entry itself (place.len() is
        // behind the Arc - a cache miss per mesh per view)
        self.spheres.len() + self.extra.len()
    }

    /// Append rows `ks` (`push_row`; the common case - one tail, no extras -
    /// without the per-row branches).
    fn push_rows(&self, ks: &[u32], out: &mut Vec<f32>) {
        out.reserve(ks.len() * 20);
        if self.tail.len() == 1 && self.extra.is_empty() {
            let t = self.tail[0];
            for &k in ks {
                out.extend_from_slice(&self.place[k as usize]);
                out.extend_from_slice(&t);
            }
        } else {
            for &k in ks {
                self.push_row(k as usize, out);
            }
        }
    }

    /// Append row `k` (past `place`: an `extra` row).
    fn push_row(&self, k: usize, out: &mut Vec<f32>) {
        let n = self.place.len();
        if k < n {
            out.extend_from_slice(&self.place[k]);
            out.extend_from_slice(&self.tail[if self.tail.len() > 1 { k } else { 0 }]);
        } else {
            out.extend_from_slice(&self.extra[k - n]);
            out.extend_from_slice(&self.extra_tail[k - n]);
        }
    }
}

/// A matrix's bits (placement identity; NaN-safe, unlike f32 ==).
fn bits(m: &[f32; 16]) -> [u32; 16] {
    m.map(f32::to_bits)
}

/// The rows' floats 16..20 for mesh `src` at `n` rows: facing, then what
/// track::skin_rows / wind_rows write (as render.rs does for its buffer).
fn tails(car: &Car, src: usize, facing: f32, animated: bool, n: usize) -> Vec<[f32; 4]> {
    let mut flat = vec![0.0f32; n * 20];
    for r in flat.chunks_exact_mut(20) {
        r[16] = facing;
    }
    if animated || car.track_env.is_some() {
        crate::track::skin_rows(car, src, &mut flat);
        if car.track_env.is_some() {
            crate::track::wind_rows(car, src, &mut flat);
        }
    }
    flat.chunks_exact(20).map(|r| [r[16], r[17], r[18], r[19]]).collect()
}

/// A coarse world grid over the instances' bounding spheres, so a view
/// rejects whole cells (frustum, LOD distance) before testing instances.
/// The Nordschleife places 529k instances; testing each one against every
/// view (main, mirror, probe, rear camera) was 20 ms of CPU a frame.
/// Results are identical: a cell passes whenever any sphere in it could,
/// and the visible instances are kept in their original order.
///
/// Only OCCUPIED cells are stored and walked (ascending by grid index): a
/// view walking every cell of the square cost +0.4 ms per Nordschleife-sized
/// tile of map (cells over empty land), and each mesh walking all of its
/// runs cost the same again. A view now walks the occupied cells once and,
/// in those that pass, the meshes' runs there (`refs`).
struct Grid {
    origin: Vec3,
    cell: f32,
    nx: u32,
    nz: u32,
    /// grid index -> occupied cell (u32::MAX: empty)
    dense_of: Vec<u32>,
    /// per occupied cell: the box of its spheres (centre +- radius) and of their centres
    boxes: Vec<[Vec3; 4]>,
    /// per occupied cell: its meshes' runs, refs[at[c]..at[c + 1]] as (entry, run)
    at: Vec<u32>,
    refs: Vec<(u32, u32)>,
    /// per occupied cell this update: frustum test passed, (min, max) squared distance
    vis: Vec<bool>,
    dist: Vec<[f32; 2]>,
    /// per occupied cell this update (main view): the least view depth
    /// (`row3` . centre) over its centres' box
    wmin: Vec<f32>,
}

/// Grid cell size in metres; meshes with fewer instances skip the grid.
const CELL: f32 = 96.0;
/// The most cells a grid gets (1 M: a 98 km square at CELL).
const GRID_MAX: f32 = 1_048_576.0;
const GRID_MIN: usize = 64;

impl Grid {
    fn build(spheres: &[&[Vec4]]) -> Option<Grid> {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for s in spheres.iter().flat_map(|s| s.iter()) {
            lo = lo.min(s.truncate());
            hi = hi.max(s.truncate());
        }
        if lo.x > hi.x || !lo.is_finite() || !hi.is_finite() {
            return None;
        }
        // at most GRID_MAX cells, growing the cell for huge extents: one
        // stray instance ~400 km out made 4096 x 4096 cells (~800 MB, walked
        // four times a frame). In float: `as u32 + 1` overflowed past 4e11 m
        let (ex, ez) = ((hi.x - lo.x).max(0.0), (hi.z - lo.z).max(0.0));
        let mut cell = CELL;
        while ((ex / cell).floor() + 1.0) * ((ez / cell).floor() + 1.0) > GRID_MAX {
            cell *= 2.0;
        }
        let nx = ((ex / cell).floor() + 1.0).min(4096.0) as u32;
        let nz = ((ez / cell).floor() + 1.0).min(4096.0) as u32;
        let empty = [Vec3::splat(f32::MAX), Vec3::splat(f32::MIN), Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)];
        let mut g = Grid { origin: lo, cell, nx, nz, dense_of: vec![u32::MAX; (nx * nz) as usize], boxes: Vec::new(),
                           at: Vec::new(), refs: Vec::new(), vis: Vec::new(), dist: Vec::new(), wmin: Vec::new() };
        // the occupied cells, numbered in grid order
        for s in spheres.iter().flat_map(|s| s.iter()) {
            let c = g.full_cell(s.truncate()) as usize;
            g.dense_of[c] = 0;
        }
        let mut n = 0u32;
        for d in g.dense_of.iter_mut().filter(|d| **d != u32::MAX) {
            *d = n;
            n += 1;
        }
        g.boxes = vec![empty; n as usize];
        for s in spheres.iter().flat_map(|s| s.iter()) {
            let c = g.cell_of(s.truncate());
            let b = &mut g.boxes[c as usize];
            let (p, r) = (s.truncate(), Vec3::splat(s.w));
            b[0] = b[0].min(p - r);
            b[1] = b[1].max(p + r);
            b[2] = b[2].min(p);
            b[3] = b[3].max(p);
        }
        Some(g)
    }

    /// `wmin` for this update's view row (clip w = row3 . (p, 1)).
    fn update_depth(&mut self, row3: Vec4) {
        self.wmin.clear();
        for b in &self.boxes {
            let (lo, hi) = (b[2], b[3]);
            let v = Vec3::new(if row3.x >= 0.0 { lo.x } else { hi.x },
                              if row3.y >= 0.0 { lo.y } else { hi.y },
                              if row3.z >= 0.0 { lo.z } else { hi.z });
            self.wmin.push(row3.truncate().dot(v) + row3.w);
        }
    }

    fn full_cell(&self, p: Vec3) -> u32 {
        let x = (((p.x - self.origin.x) / self.cell).max(0.0) as u32).min(self.nx - 1);
        let z = (((p.z - self.origin.z) / self.cell).max(0.0) as u32).min(self.nz - 1);
        z * self.nx + x
    }

    /// The occupied cell holding `p` (one of the spheres the grid was built
    /// from; any other point may land in an empty cell: u32::MAX).
    fn cell_of(&self, p: Vec3) -> u32 {
        self.dense_of[self.full_cell(p) as usize]
    }

    /// Per occupied cell: whether its sphere box is inside every plane, and
    /// the squared distance range from `eye` to its centres.
    fn update(&mut self, planes: &[Vec4], eye: Vec3) {
        self.vis.clear();
        self.dist.clear();
        for b in &self.boxes {
            let (lo, hi) = (b[0], b[1]);
            let inside = planes.iter().all(|p| {
                // the box corner furthest along the plane normal
                let v = Vec3::new(if p.x >= 0.0 { hi.x } else { lo.x },
                                  if p.y >= 0.0 { hi.y } else { lo.y },
                                  if p.z >= 0.0 { hi.z } else { lo.z });
                p.truncate().dot(v) + p.w >= 0.0
            });
            self.vis.push(inside);
            let (clo, chi) = (b[2], b[3]);
            let near = eye.clamp(clo, chi) - eye;
            let far = (eye - clo).abs().max((chi - eye).abs());
            self.dist.push([near.length_squared(), far.length_squared()]);
        }
    }
}

/// The instances of `e` to test this update: false (test them all) when it
/// has no grid runs, else true with `out` holding those in cells that passed
/// and whose distance range meets the squared window [lo2, hi2) (distances
/// x s2, the LOD scale squared), in their original order.
/// A whole grid run whose largest instance would fail the caller's own
/// too-small test at the run's nearest point is skipped unread - every
/// instance in it fails that test too, so the visible set is unchanged.
#[derive(Clone, Copy)]
enum Tiny {
    No,
    /// radius < k x distance from the grid's focus (update_view): k squared
    Dist(f32),
    /// radius x focal < MIN_PIXELS x view depth (update): focal
    Proj(f32),
}

/// Each level whose shadow caster (shadow_to) stands at other places: its
/// `cast_map`, and the caster's `extra` rows for places it lacks.
fn map_casters(entries: &mut [Entry]) {
    let mut index: HashMap<usize, HashMap<[u32; 16], u32>> = HashMap::new();
    for ei in 0..entries.len() {
        let Some(Some(t)) = entries[ei].shadow_to else { continue };
        if t == ei || Arc::ptr_eq(&entries[ei].place, &entries[t].place) {
            continue;
        }
        let idx = index.entry(t).or_insert_with(|| {
            let mut h = HashMap::new();
            for (k, m) in entries[t].place.iter().enumerate() {
                h.entry(bits(m)).or_insert(k as u32);
            }
            h
        });
        let src = entries[ei].place.clone();
        let mut map = Vec::with_capacity(src.len());
        for m in src.iter() {
            let k = *idx.entry(bits(m)).or_insert_with(|| {
                let tg = &mut entries[t];
                tg.extra.push(*m);
                (tg.cast_len() - 1) as u32
            });
            map.push(k);
        }
        entries[ei].cast_map = Some(map);
    }
    let n = entries.iter().filter(|e| e.cast_map.is_some()).count();
    if n > 0 {
        println!("cull: {n} level(s) cast through a caster standing elsewhere; {} caster row(s) added",
                 entries.iter().map(|e| e.extra.len()).sum::<usize>());
    }
}

/// The grid over the entries with at least GRID_MIN instances, and their
/// runs (instances grouped by occupied cell) and the cells' refs to them.
fn grid_entries(entries: &mut [Entry]) -> Option<Grid> {
    let mut grid = {
        let big: Vec<&[Vec4]> = entries.iter().filter(|e| e.spheres.len() >= GRID_MIN).map(|e| e.spheres.as_slice()).collect();
        Grid::build(&big)
    };
    if let Some(g) = grid.as_ref() {
        for e in entries.iter_mut().filter(|e| e.spheres.len() >= GRID_MIN) {
            let mut keyed: Vec<(u32, u32)> = e.spheres.iter().enumerate()
                .map(|(k, s)| (g.cell_of(s.truncate()), k as u32)).collect();
            keyed.sort_unstable();
            e.order = keyed.iter().map(|k| k.1).collect();
            let mut i = 0;
            while i < keyed.len() {
                let c = keyed[i].0;
                let mut j = i;
                while j < keyed.len() && keyed[j].0 == c {
                    j += 1;
                }
                e.runs.push((c, i as u32, j as u32));
                e.run_r.push(keyed[i..j].iter().map(|k| e.spheres[k.1 as usize].w).fold(0.0f32, f32::max));
                i = j;
            }
        }
    }
    // each occupied cell's (entry, run) refs
    if let Some(g) = grid.as_mut() {
        let n = g.boxes.len();
        let mut at = vec![0u32; n + 1];
        for e in entries.iter() {
            for r in &e.runs {
                at[r.0 as usize + 1] += 1;
            }
        }
        for c in 0..n {
            at[c + 1] += at[c];
        }
        let mut fill = at.clone();
        let mut refs = vec![(0u32, 0u32); at[n] as usize];
        for (ei, e) in entries.iter().enumerate() {
            for (ri, r) in e.runs.iter().enumerate() {
                let f = &mut fill[r.0 as usize];
                refs[*f as usize] = (ei as u32, ri as u32);
                *f += 1;
            }
        }
        g.at = at;
        g.refs = refs;
    }
    grid
}

/// Per gridded entry: the instances in cells that passed this update and
/// whose distance range meets the entry's squared window `win(e)` (distances
/// x s2, the LOD scale squared), in their original order. Walks the occupied
/// cells once; a cell that fails costs nothing per mesh.
fn gather(g: &Grid, entries: &[Entry], win: impl Fn(&Entry) -> Option<[f32; 2]>, s2: f32, tiny: Tiny,
          out: &mut Vec<Vec<u32>>) {
    out.resize_with(entries.len(), Vec::new);
    for v in out.iter_mut() {
        v.clear();
    }
    for c in 0..g.boxes.len() {
        if !g.vis[c] {
            continue;
        }
        let [dmin, dmax] = g.dist[c];
        for &(ei, ri) in &g.refs[g.at[c] as usize..g.at[c + 1] as usize] {
            let e = &entries[ei as usize];
            let rmax = e.run_r[ri as usize];
            match tiny {
                Tiny::Dist(k2) if rmax * rmax < k2 * dmin => continue,
                Tiny::Proj(focal) => {
                    if let Some(&w) = g.wmin.get(c) {
                        if w > rmax && rmax * focal < MIN_PIXELS * w {
                            continue;
                        }
                    }
                }
                _ => {}
            }
            if let Some([lo2, hi2]) = win(e) {
                if dmax * s2 < lo2 || dmin * s2 >= hi2 {
                    continue;
                }
            }
            let (_, a, b) = e.runs[ri as usize];
            out[ei as usize].extend_from_slice(&e.order[a as usize..b as usize]);
        }
    }
    // back to the original order (the instances' draw order)
    for v in out.iter_mut().filter(|v| v.len() > 1) {
        v.sort_unstable();
    }
}

/// One LOD's camera-distance window with the game's crossfades (visibility
/// compute shader 033aa890): LOD L draws from lodIn[L] - dlt[L] to
/// lodIn[L+1] + dlt[L+1], fading IN over the first dlt[L] metres and OUT
/// over the last dlt[L+1]; the incoming level reaches full coverage at the
/// switch distance where the outgoing one starts to leave, so the pair never
/// shows a hole. Past lodOut the whole prop fades over ditheredLodOut.
/// Distances in metres, already x lodDistanceScale; the widths are not.
#[derive(Clone, Copy)]
struct Band {
    near: f32,
    far: f32,
    din: f32,
    dout: f32,
    out: f32,
    dlo: f32,
    /// squared: drawn at all [lo2, hi2), fully opaque [op_lo2, op_hi2)
    lo2: f32,
    hi2: f32,
    op_lo2: f32,
    op_hi2: f32,
}

impl Band {
    fn new(lod: [f32; 2], f: [f32; 4]) -> Option<Self> {
        let [near, far] = lod;
        let [din, dout, out, dlo] = f;
        if din <= 0.0 && dout <= 0.0 && out <= 0.0 {
            return None;
        }
        let sq = |x: f32| if x.is_finite() { x.max(0.0) * x.max(0.0) } else { f32::INFINITY };
        let mut hi = far + dout;
        let mut op_hi = far;
        if out > 0.0 {
            hi = hi.min(out + dlo);
            op_hi = op_hi.min(out);
        }
        Some(Band { near, far, din, dout, out, dlo,
                    lo2: sq(near - din), hi2: sq(hi), op_lo2: sq(near), op_hi2: sq(op_hi) })
    }

    /// Coverage 0..1 at `d2` (squared metres) inside [lo2, hi2), in the
    /// shader's 6-bit steps.
    fn coverage(&self, d2: f32) -> f32 {
        if d2 >= self.op_lo2 && d2 < self.op_hi2 {
            return 1.0;
        }
        let d = d2.sqrt();
        let mut t = 1.0f32;
        if self.out > 0.0 && d >= self.out && self.dlo > 0.0 {
            t = 1.0 - (d - self.out) / self.dlo;
        }
        if self.far.is_finite() && self.dout > 0.0 && d + self.dout >= self.far {
            t = 1.0 - (d - self.far) / self.dout;
        }
        if d < self.near && self.din > 0.0 {
            t = (d - (self.near - self.din)) / self.din;
        }
        (t.clamp(0.0, 1.0) * 63.0).floor() / 63.0
    }
}

pub struct Culler {
    entries: Vec<Entry>,
    grid: Option<Grid>,
    /// scratch: per entry, the instances a view tests (Grid)
    cands: Vec<Vec<u32>>,
    /// visible instance count per GPU mesh; None = not culled, draw as is
    counts: Vec<Option<u32>>,
    /// GPU mesh -> entry index
    entry_of: Vec<Option<usize>>,
    scratch: Vec<f32>,
    vis: Vec<u32>,
    fade: Vec<f32>,
    /// shadow cascades: per entry, the instances it casts for (sources
    /// redirected by shadow_to)
    cast: Vec<Vec<u32>>,
    pub stats: (usize, usize),
    /// EVOVIEW_LOD_LOG: the previous main-view eye and the frame number
    log_eye: Option<Vec3>,
    log_frame: u32,
    /// the lodDistanceScale the bands were last built with
    scale: f32,
    /// rows the device's largest buffer holds
    max_rows: Option<usize>,
}

/// Local bounding sphere of a mesh's positions: AABB centre, max distance.
fn local_sphere(pos: &[f32]) -> Option<(Vec3, f32)> {
    if pos.len() < 3 {
        return None;
    }
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for p in pos.chunks_exact(3) {
        let v = Vec3::new(p[0], p[1], p[2]);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let c = (lo + hi) * 0.5;
    let r = pos
        .chunks_exact(3)
        .map(|p| Vec3::new(p[0], p[1], p[2]).distance_squared(c))
        .fold(0.0f32, f32::max)
        .sqrt();
    (r.is_finite() && c.is_finite()).then_some((c, r))
}

impl Culler {
    /// `skip` marks GPU meshes that must never be culled (replay cars).
    pub fn build(car: &Car, meshes: &[GpuMesh], skip: &[usize]) -> Self {
        let mut entries = Vec::new();
        // placement lists already built, by (length, sampled hash)
        let mut shared: HashMap<(usize, u64), Vec<Arc<Vec<[f32; 16]>>>> = HashMap::new();
        for (gi, m) in meshes.iter().enumerate() {
            if skip.contains(&gi) {
                continue;
            }
            let Some(src) = car.meshes.get(m.src) else { continue };
            // skinned geometry moves; its rest-pose bounds are not safe -
            // except the track's animated props, which stay where they stand
            // (their sphere gets some slack for waving arms and flags)
            let animated = crate::track::is_animated(car, m.src);
            if !animated && src.bone_wgt.is_some() && !car.floats(src.bone_wgt).iter().all(|&w| w == 0.0) {
                continue;
            }
            let Some((c, r)) = local_sphere(&car.floats(src.positions)) else { continue };
            let r = if animated { r * 1.5 + 0.5 } else { r };
            // Camera-facing cards rotate about their origin in the shader, so
            // bound them by a sphere around the origin instead.
            let (c, r) = if src.facing != 0 { (Vec3::ZERO, c.length() + r) } else { (c, r) };
            let facing = src.facing as f32;
            let xfs: Vec<[f32; 16]> = if src.xforms.is_empty() {
                vec![Mat4::IDENTITY.to_cols_array()]
            } else {
                src.xforms.clone()
            };
            let mut spheres = Vec::with_capacity(xfs.len());
            for xf in &xfs {
                let mm = Mat4::from_cols_array(xf);
                let wc = mm.transform_point3(c);
                let s = mm.x_axis.truncate().length()
                    .max(mm.y_axis.truncate().length())
                    .max(mm.z_axis.truncate().length());
                spheres.push(wc.extend(r * s));
            }
            let mut tail = tails(car, m.src, facing, animated, xfs.len());
            if tail.windows(2).all(|w| w[0] == w[1]) {
                tail.truncate(1);
            }
            // the LOD levels of a prop share one placement list
            let key = {
                let mut h: u64 = 0xcbf29ce484222325;
                for xf in xfs.iter().step_by((xfs.len() / 64).max(1)) {
                    for b in bits(xf) {
                        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
                    }
                }
                (xfs.len(), h)
            };
            let same = shared.entry(key).or_default();
            let flat_bits = |x: &[[f32; 16]]| -> Vec<u32> { x.iter().flat_map(bits).collect() };
            let place = match same.iter().find(|a| a.len() == xfs.len() && flat_bits(a) == flat_bits(&xfs)) {
                Some(a) => a.clone(),
                None => {
                    let a = Arc::new(xfs);
                    same.push(a.clone());
                    a
                }
            };
            entries.push(Entry {
                gpu: gi, spheres, place, tail, extra: Vec::new(), extra_tail: Vec::new(), cast_map: None,
                last: Vec::new(), shadow_buf: Default::default(), shadow_count: [0; 6], shadow_last: Default::default(),
                lod: None,
                band: None,
                raw_lod: src.lod,
                raw_fade: src.lod_fade,
                shadow_to: None,
                shadow_src: src.shadow_mesh,
                last_fade: Vec::new(),
                runs: Vec::new(),
                run_r: Vec::new(),
                order: Vec::new(),
                name: src.name.clone(),
            });
        }
        // EVOVIEW_CULL_GRID=0: test every instance in every view
        let grid = if crate::envc::var("EVOVIEW_CULL_GRID").map_or(true, |v| v != "0") {
            grid_entries(&mut entries)
        } else {
            None
        };
        let mut entry_of = vec![None; meshes.len()];
        for (k, e) in entries.iter().enumerate() {
            entry_of[e.gpu] = Some(k);
        }
        // shadow casters by Car::meshes index -> entry index
        let by_src: std::collections::HashMap<usize, usize> = meshes.iter().enumerate()
            .filter_map(|(gi, m)| entry_of[gi].map(|k| (m.src, k))).collect();
        for e in entries.iter_mut() {
            e.shadow_to = match e.shadow_src {
                None => None,
                Some(None) => Some(None),
                // a target that was not built (culled away) keeps the old way
                Some(Some(t)) => by_src.get(&t).map(|&k| Some(k)),
            };
        }
        // a level whose shadow caster does not stand where it does: its
        // casters go to that mesh's rows by placement, the missing ones
        // added after its own (indexing the caster's rows with the level's
        // own instance numbers drew the wrong places - or ran off the end)
        map_casters(&mut entries);
        for e in entries.iter_mut().filter(|e| !e.extra.is_empty()) {
            let src = meshes[e.gpu].src;
            let animated = crate::track::is_animated(car, src);
            let all = tails(car, src, e.tail[0][0], animated, e.cast_len());
            e.extra_tail = all[e.place.len()..].to_vec();
        }
        Culler {
            entries,
            grid,
            cands: Vec::new(),
            entry_of,
            counts: vec![None; meshes.len()],
            scratch: Vec::new(),
            vis: Vec::new(),
            fade: Vec::new(),
            cast: Vec::new(),
            stats: (0, 0),
            log_eye: None,
            log_frame: 0,
            scale: 0.0,
            max_rows: None,
        }.with_scale(lod_scale(None))
    }

    /// Rebuild every LOD band for lodDistanceScale `s` (switch distances and
    /// lodOut scale; the dither widths are plain metres, as in 033aa890).
    fn with_scale(mut self, s: f32) -> Self {
        self.set_scale(s);
        self
    }

    fn set_scale(&mut self, s: f32) {
        if (s - self.scale).abs() <= 1e-4 * s.max(1.0) {
            return;
        }
        self.scale = s;
        if lod_log() || crate::envc::var_os("EVOVIEW_LOD_DUMP").is_some() {
            println!("lodDistanceScale {s:.3}");
        }
        for e in self.entries.iter_mut() {
            let Some([a, b]) = e.raw_lod else { continue };
            let (a, b) = (a * s, b * s);
            e.lod = Some([a * a, if b.is_finite() { b * b } else { f32::INFINITY }]);
            let f = e.raw_fade;
            e.band = Band::new([a, b], [f[0], f[1], f[2] * s, f[3]]);
            // ⚠ no `e.last.clear()`: update's vis + fade compare already
            // catches every real change, and clearing it re-uploaded every
            // LOD'd instance buffer each frame under a zooming TV camera
        }
    }

    /// The lodDistanceScale in use (EVOVIEW_LOD_LOG, HUD).
    pub fn lod_scale(&self) -> f32 {
        self.scale
    }

    /// Instances to draw for GPU mesh `i`.
    pub fn count(&self, i: usize, default: u32) -> u32 {
        self.counts.get(i).copied().flatten().unwrap_or(default)
    }

    /// Cull for the shadow pass against an ORTHOGRAPHIC light matrix (all six
    /// planes, no size floor) into the per-mesh shadow buffers.
    /// LODs are chosen by distance from `focus`, the shadow box centre, so
    /// the cached map does not depend on where the camera is.
    /// Cull cascade `c` (0 near, 1 far) to `light_vp`'s box. Casters whose
    /// bounding radius is under `min_radius` metres are dropped (the far
    /// cascade's texels are a metre wide).
    /// `tiny`: (eye, k) drops casters whose radius over their distance from
    /// the eye is under k - their shadow would cover a pixel or two on
    /// screen (grass tufts across the field were half the cascade's cost).
    pub fn update_shadow(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, c: usize,
                         light_vp: Mat4, focus: Vec3, min_radius: f32, tiny: Option<(Vec3, f32)>) {
        self.update_view(device, queue, c, light_vp, focus, min_radius, tiny, 1.0);
    }

    /// `update_shadow` for a small perspective view (a mirror, a probe face)
    /// with its LODs picked as if `lod_scale` times further away: the main
    /// camera's focal length over this view's, so a prop gets the LOD the
    /// main view would give it at the same size in pixels. A 240 px mirror
    /// drew the full-detail track behind the car (EVOVIEW_VIEW_LOD=0: 1).
    pub fn update_view(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, c: usize,
                       light_vp: Mat4, focus: Vec3, min_radius: f32, tiny: Option<(Vec3, f32)>, lod_scale: f32) {
        let s2 = lod_scale * lod_scale;
        let _t = crate::bench::cpu(["cull shadow", "cull far", "cull mirror", "cull probe", "cull rear cam", "cull mid"][c.min(5)]);
        let rows = [light_vp.row(0), light_vp.row(1), light_vp.row(2), light_vp.row(3)];
        let mut planes = [
            rows[3] + rows[0],
            rows[3] - rows[0],
            rows[3] + rows[1],
            rows[3] - rows[1],
            rows[2],
            rows[3] - rows[2],
        ];
        for p in planes.iter_mut() {
            let l = p.truncate().length();
            if l > 0.0 {
                *p /= l;
            }
        }
        if let Some(g) = self.grid.as_mut() {
            g.update(&planes, focus);
        }
        // runs too small to show skipped whole, when the size test is from
        // the grid's own centre (the views; not the shadow maps)
        let tiny_run = match tiny {
            Some((eye, k)) if eye == focus && min_radius <= 0.0 => Tiny::Dist(k * k),
            _ => Tiny::No,
        };
        // the shadow cascades draw each LOD's casters with its shadow
        // level's geometry (load.rs forceShadowLod); other views draw as seen
        let redirect = matches!(c, 0 | 1 | 5);
        self.cast.resize_with(self.entries.len(), Vec::new);
        for v in self.cast.iter_mut() {
            v.clear();
        }
        if let Some(g) = self.grid.as_ref() {
            gather(g, &self.entries, |e| e.lod, s2, tiny_run, &mut self.cands);
        }
        for ei in 0..self.entries.len() {
            let e = &self.entries[ei];
            self.vis.clear();
            let gridded = self.grid.is_some() && !e.runs.is_empty();
            let (cand, all) = if gridded { (&self.cands[ei][..], 0..0) } else { (&[][..], 0..e.spheres.len() as u32) };
            for k in cand.iter().copied().chain(all) {
                let s = &e.spheres[k as usize];
                if s.w < min_radius {
                    continue;
                }
                if let Some((eye, k)) = tiny {
                    if s.w < k * s.truncate().distance(eye) {
                        continue;
                    }
                }
                let centre = s.truncate().extend(1.0);
                if planes.iter().any(|p| p.dot(centre) < -s.w) {
                    continue;
                }
                if let Some([a, b]) = e.lod {
                    let d2 = s.truncate().distance_squared(focus) * s2;
                    if d2 < a || d2 >= b {
                        continue;
                    }
                }
                self.vis.push(k);
            }
            let to = if redirect { e.shadow_to.unwrap_or(Some(ei)) } else { Some(ei) };
            if let Some(t) = to {
                // to the caster's rows: the same placements, its numbering
                if let Some(map) = e.cast_map.as_ref().filter(|_| t != ei) {
                    for k in self.vis.iter_mut() {
                        *k = map[*k as usize];
                    }
                    self.vis.sort_unstable();
                    self.vis.dedup();
                }
                if self.cast[t].is_empty() {
                    std::mem::swap(&mut self.cast[t], &mut self.vis);
                } else {
                    self.cast[t].extend_from_slice(&self.vis);
                    self.cast[t].sort_unstable();
                    self.cast[t].dedup();
                }
            }
        }
        // a view's buffer holds what it draws (+50 %), not a row for every
        // placement: six of those per mesh were ~150 MB on the Nordschleife,
        // and past 3.36 M placements one was over the 256 MiB buffer limit
        let max_rows = *self.max_rows.get_or_insert_with(|| (device.limits().max_buffer_size / 80) as usize);
        for (e, vis) in self.entries.iter_mut().zip(self.cast.iter()) {
            let mut n = vis.len();
            // a single placement with LOD bands gets its own buffer too:
            // the mesh's own one carries the MAIN view's dither-out amount
            if n > 0 && (e.cast_len() > 1 || e.band.is_some()) {
                n = n.min(max_rows);
                if e.shadow_last[c] != *vis {
                    e.shadow_last[c].clone_from(vis);
                    self.scratch.clear();
                    e.push_rows(&vis[..n], &mut self.scratch);
                    let need = (n * 80) as u64;
                    if e.shadow_buf[c].as_ref().map_or(true, |b| b.size() < need) {
                        let cap = (n + n / 2).max(256).min(e.cast_len()).min(max_rows).max(n);
                        e.shadow_buf[c] = Some(device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("shadow-inst"),
                            size: (cap * 80) as u64,
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }));
                    }
                    queue.write_buffer(e.shadow_buf[c].as_ref().unwrap(), 0, bytemuck::cast_slice(&self.scratch));
                }
            }
            e.shadow_count[c] = n as u32;
        }
    }

    /// What the shadow pass should draw for GPU mesh `i`: None = not culled
    /// (use the mesh's own buffer and count); Some((buffer, count)) where a
    /// None buffer means the mesh's own (single identity row).
    pub fn shadow(&self, c: usize, i: usize) -> Option<(Option<&wgpu::Buffer>, u32)> {
        let e = &self.entries[self.entry_of.get(i).copied().flatten()?];
        // (a banded single placement: its own fade-0 copy, not the main
        // buffer - mirrors / probe / rear view drew it main-view dithered)
        Some((e.shadow_buf[c].as_ref().filter(|_| e.cast_len() > 1 || e.band.is_some()), e.shadow_count[c]))
    }

    /// Cull against `view_proj` (wgpu clip space, z in 0..1) for a viewport
    /// `height` pixels tall, and rewrite the instance buffers that changed.
    pub fn update(&mut self, queue: &wgpu::Queue, meshes: &[GpuMesh], view_proj: Mat4, eye: Vec3,
                  height: u32) {
        let _t = crate::bench::cpu("cull main");
        let rows = [view_proj.row(0), view_proj.row(1), view_proj.row(2), view_proj.row(3)];
        let mut planes = [
            rows[3] + rows[0],
            rows[3] - rows[0],
            rows[3] + rows[1],
            rows[3] - rows[1],
            rows[2], // near; the far plane is at infinity
        ];
        for p in planes.iter_mut() {
            let l = p.truncate().length();
            if l > 0.0 {
                *p /= l;
            }
        }
        // focal length in pixels: |row1.xyz| is 1/tan(fov/2) for an
        // orthonormal view, and clip w is the view depth
        let focal = rows[1].truncate().length() * height as f32 * 0.5;
        // the game's lodDistanceScale follows the main view's vertical fov
        let fov_y = 2.0 * (1.0 / rows[1].truncate().length().max(1e-6)).atan();
        self.set_scale(lod_scale(Some(fov_y)));
        let (mut shown, mut total) = (0usize, 0usize);
        if lod_log() {
            self.log_switches(&planes, rows[3], focal, eye);
        }
        if let Some(g) = self.grid.as_mut() {
            g.update(&planes, eye);
            g.update_depth(rows[3]);
        }
        let tiny_run = if focal > 0.0 { Tiny::Proj(focal) } else { Tiny::No };
        if let Some(g) = self.grid.as_ref() {
            gather(g, &self.entries, |e| e.band.map(|b| [b.lo2, b.hi2]).or(e.lod), 1.0, tiny_run, &mut self.cands);
        }
        let gridded_on = self.grid.is_some();
        for (ei, e) in self.entries.iter_mut().enumerate() {
            self.vis.clear();
            self.fade.clear();
            let gridded = gridded_on && !e.runs.is_empty();
            let (cand, all) = if gridded { (&self.cands[ei][..], 0..0) } else { (&[][..], 0..e.spheres.len() as u32) };
            for k in cand.iter().copied().chain(all) {
                let s = &e.spheres[k as usize];
                let c = s.truncate().extend(1.0);
                let r = s.w;
                if planes.iter().any(|p| p.dot(c) < -r) {
                    continue;
                }
                let mut out = 0.0f32;
                if let Some(b) = &e.band {
                    let d2 = s.truncate().distance_squared(eye);
                    if d2 < b.lo2 || d2 >= b.hi2 {
                        continue;
                    }
                    out = 1.0 - b.coverage(d2);
                    if out >= 1.0 {
                        continue;
                    }
                } else if let Some([a, b]) = e.lod {
                    let d2 = s.truncate().distance_squared(eye);
                    if d2 < a || d2 >= b {
                        continue;
                    }
                }
                let w = rows[3].dot(c);
                if w > r && r * focal < MIN_PIXELS * w {
                    continue;
                }
                self.vis.push(k);
                self.fade.push(out);
            }
            total += e.spheres.len();
            shown += self.vis.len();
            self.counts[e.gpu] = Some(self.vis.len() as u32);
            if self.vis.is_empty() || (self.vis == e.last && self.fade == e.last_fade) {
                continue;
            }
            // A single un-instanced mesh only needs its count; its one row
            // (identity) is already in the buffer - unless it fades.
            if e.spheres.len() > 1 || e.band.is_some() {
                self.scratch.clear();
                e.push_rows(&self.vis, &mut self.scratch);
                for (row, &f) in self.scratch.chunks_exact_mut(20).zip(&self.fade) {
                    // the matrix's unused [0][3]: the shader reads it as the
                    // instance's dither-out amount (track.wgsl lod_dither)
                    row[3] = f;
                }
                queue.write_buffer(&meshes[e.gpu].instances, 0, bytemuck::cast_slice(&self.scratch));
            }
            std::mem::swap(&mut e.last, &mut self.vis);
            std::mem::swap(&mut e.last_fade, &mut self.fade);
        }
        self.stats = (shown, total);
    }
}

/// The game's lodDistanceScale for the main view (FUN_141dbce80 + the
/// VideoSettings getters, build 20260907): the LOD preset's scale divided by
/// min(1, vertical fov / 80 deg), so a narrow view keeps detail further out
/// (a 40 deg view switches twice as far as an 80 deg one; wider views do
/// not go below the preset).
///   preset: Low 0.5, Medium 1.0, High 1.25, Ultra 1.5, Custom = the
///   mainLodDistanceScale field, "experimental static LOD" 4.0.
/// EVOVIEW_LOD_QUALITY=low|medium|high|ultra|static|<number> picks the
/// preset (default medium, Josh's setting). EVOVIEW_LOD_SCALE=x fixes the
/// whole scale with no fov term (the old behaviour was 3).
pub fn lod_scale(fov_y: Option<f32>) -> f32 {
    static FIXED: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    static PRESET: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    if let Some(x) = *FIXED.get_or_init(|| crate::envc::var("EVOVIEW_LOD_SCALE").ok().and_then(|v| v.parse().ok())) {
        return x;
    }
    let p = *PRESET.get_or_init(|| {
        match crate::envc::var("EVOVIEW_LOD_QUALITY").unwrap_or_default().trim().to_lowercase().as_str() {
            "low" => 0.5,
            "high" => 1.25,
            "ultra" => 1.5,
            "static" => 4.0,
            v => v.parse().unwrap_or(1.0),
        }
    });
    // FUN_141dbce80: fVar33 = 1 / min(1, max(0, fov / 80))
    let k = fov_y.map_or(1.0, |f| (f.to_degrees() / 80.0).clamp(1e-3, 1.0));
    p / k
}

fn lod_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::envc::var_os("EVOVIEW_LOD_LOG").is_some())
}

impl Culler {
    /// EVOVIEW_LOD_LOG=1: every on-screen instance whose LOD band or lodOut
    /// state changed since the last main-view update, as
    /// `LODSW frame mesh inst dist px in|out|fade` (px = projected radius).
    fn log_switches(&mut self, planes: &[Vec4], w_row: Vec4, focal: f32, eye: Vec3) {
        let prev = self.log_eye.replace(eye);
        self.log_frame += 1;
        let Some(prev) = prev else { return };
        for e in &self.entries {
            let Some([a, b]) = e.lod else { continue };
            let win = e.band.map(|b| [b.lo2, b.hi2]).unwrap_or([a, b]);
            for (k, s) in e.spheres.iter().enumerate() {
                let c = s.truncate().extend(1.0);
                if planes.iter().any(|p| p.dot(c) < -s.w) {
                    continue;
                }
                let (d0, d1) = (s.truncate().distance_squared(prev), s.truncate().distance_squared(eye));
                let inside = |d2: f32| d2 >= win[0] && d2 < win[1];
                let full = |d2: f32| e.band.map_or(inside(d2), |b| b.coverage(d2) >= 1.0);
                let what = if inside(d0) != inside(d1) {
                    if inside(d1) { "in" } else { "out" }
                } else if full(d0) != full(d1) {
                    "fade"
                } else {
                    continue;
                };
                let w = w_row.dot(c).max(1e-3);
                // hard: it appears (or leaves) at full coverage - a pop
                let cov = |d2: f32| e.band.map_or(1.0, |b| b.coverage(d2));
                let hard = match what { "in" => cov(d1) >= 0.99, "out" => cov(d0) >= 0.99, _ => false };
                println!("LODSW {} {} {} {:.1} {:.1} {} band {:.1}..{:.1} {}", self.log_frame, e.name, k, d1.sqrt(),
                         s.w * focal / w, what, win[0].sqrt(), win[1].sqrt(), if hard { "hard" } else { "soft" });
            }
        }
    }
}

/// Front-to-back order for the opaque and cutout draws of a car.
///
/// The car fragment shader is the expensive part of a car frame (area-light
/// rig, IBL, several maps), and a car is many layers deep: the body shell in
/// front of the cabin, the cabin in front of the engine bay. Drawn in load
/// order, much of that is shaded and then covered. Sorting by distance lets
/// the depth test reject hidden layers before they are shaded. Pixels come
/// out identical; only the amount of work changes.
pub struct DrawOrder {
    /// (gpu mesh, draw index within it, rest-pose centre of the draw)
    items: Vec<(usize, usize, Vec3)>,
    keyed: Vec<(f32, usize, usize)>,
}

impl DrawOrder {
    pub fn build(car: &Car, meshes: &[GpuMesh]) -> Self {
        let mut items = Vec::new();
        for (gi, m) in meshes.iter().enumerate() {
            let src = car.meshes.get(m.src);
            let pos = src.map(|s| car.floats(s.positions)).unwrap_or_default();
            let idx = src.map(|s| car.u32s(s.indices)).unwrap_or_default();
            for (k, &(start, count, _)) in m.draws.iter().enumerate() {
                let (s, e) = (start as usize, (start + count) as usize);
                let mut c = Vec3::ZERO;
                let mut n = 0.0f32;
                if e <= idx.len() && pos.len() >= 3 {
                    let step = ((e - s) / 256).max(1);
                    for &v in idx[s..e].iter().step_by(step) {
                        let v = v as usize * 3;
                        if v + 2 < pos.len() {
                            c += Vec3::new(pos[v], pos[v + 1], pos[v + 2]);
                            n += 1.0;
                        }
                    }
                }
                // no geometry to look at (the studio floor): draw it last
                let c = if n > 0.0 { c / n } else { Vec3::splat(f32::INFINITY) };
                items.push((gi, k, c));
            }
        }
        DrawOrder { items, keyed: Vec::new() }
    }

    /// (gpu mesh, draw index) pairs, nearest first.
    pub fn sorted(&mut self, eye: Vec3) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.keyed.clear();
        self.keyed.extend(self.items.iter().map(|&(g, k, c)| {
            let d = if c.is_finite() { c.distance_squared(eye) } else { f32::INFINITY };
            (d, g, k)
        }));
        self.keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.keyed.iter().map(|&(_, g, k)| (g, k))
    }
}

#[cfg(test)]
mod grid_tests {
    use super::*;

    #[test]
    fn a_stray_far_instance_does_not_blow_up_the_grid() {
        let s = [Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(400_000.0, 0.0, 400_000.0, 1.0)];
        let g = Grid::build(&[&s[..]]).unwrap();
        assert!(g.boxes.len() as f32 <= GRID_MAX);
        let far = [Vec4::new(-1e12, 0.0, -1e12, 1.0), Vec4::new(1e12, 0.0, 1e12, 1.0)];
        let g = Grid::build(&[&far[..]]).unwrap();
        assert!(g.nx >= 1 && !g.boxes.is_empty() && g.boxes.len() as f32 <= GRID_MAX);
        let _ = g.cell_of(Vec3::ZERO);
    }

    #[test]
    fn a_track_sized_grid_keeps_its_cell() {
        let s = [Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(5000.0, 0.0, 3000.0, 1.0)];
        let g = Grid::build(&[&s[..]]).unwrap();
        assert_eq!(g.cell, CELL);
    }

    /// A small LCG: the tests' scenes are random but repeatable.
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    pub(super) fn test_entry(spheres: Vec<Vec4>, lod: Option<[f32; 2]>) -> Entry {
        let n = spheres.len();
        Entry {
            gpu: 0, spheres, place: Arc::new(vec![[0.0; 16]; n]), tail: vec![[0.0; 4]], extra: Vec::new(),
            extra_tail: Vec::new(), cast_map: None, last: Vec::new(), shadow_buf: Default::default(),
            shadow_count: [0; 6], shadow_last: Default::default(), lod, raw_lod: None, raw_fade: [0.0; 4],
            shadow_to: None, shadow_src: None, band: None, last_fade: Vec::new(), runs: Vec::new(),
            run_r: Vec::new(), order: Vec::new(), name: String::new(),
        }
    }

    /// Meshes scattered in clumps over a `span` metre square with gaps of
    /// empty cells, some LOD-banded, some too few for the grid.
    fn scene(r: &mut Rng, span: f32) -> Vec<Entry> {
        let mut out = Vec::new();
        for m in 0..24 {
            let n = if m % 5 == 0 { 20 } else { 60 + (r.f() * 400.0) as usize };
            let lod = match m % 3 {
                0 => None,
                1 => Some([0.0, (200.0 + r.f() * 800.0).powi(2)]),
                _ => Some([(100.0 + r.f() * 300.0).powi(2), f32::INFINITY]),
            };
            let mut sp = Vec::with_capacity(n);
            let clumps: Vec<(f32, f32)> = (0..6).map(|_| (r.f() * span, r.f() * span)).collect();
            for k in 0..n {
                let (cx, cz) = clumps[k % clumps.len()];
                sp.push(Vec4::new(cx + r.f() * 150.0, r.f() * 30.0, cz + r.f() * 150.0, 0.05 + r.f() * r.f() * 6.0));
            }
            out.push(test_entry(sp, lod));
        }
        out
    }

    fn frustum(eye: Vec3, at: Vec3) -> [Vec4; 6] {
        let vp = Mat4::perspective_rh(1.0, 1.6, 0.1, 5000.0) * Mat4::look_at_rh(eye, at, Vec3::Y);
        let rows = [vp.row(0), vp.row(1), vp.row(2), vp.row(3)];
        let mut planes = [rows[3] + rows[0], rows[3] - rows[0], rows[3] + rows[1], rows[3] - rows[1], rows[2], rows[3] - rows[2]];
        for p in planes.iter_mut() {
            *p /= p.truncate().length();
        }
        planes
    }

    /// The per-instance test of update_view.
    fn passes(e: &Entry, s: Vec4, planes: &[Vec4], eye: Vec3, k: f32) -> bool {
        let c = s.truncate().extend(1.0);
        if s.w < k * s.truncate().distance(eye) || planes.iter().any(|p| p.dot(c) < -s.w) {
            return false;
        }
        e.lod.map_or(true, |[a, b]| {
            let d2 = s.truncate().distance_squared(eye);
            d2 >= a && d2 < b
        })
    }

    /// A level whose shadow caster stands elsewhere sends its casters to
    /// the caster's rows BY PLACEMENT; places the caster lacks are added.
    #[test]
    fn shadow_casters_map_by_placement() {
        let m = |x: f32| { let mut a = Mat4::IDENTITY.to_cols_array(); a[12] = x; a };
        let mut a = test_entry(vec![Vec4::ONE; 3], None);
        a.place = Arc::new(vec![m(1.0), m(2.0), m(3.0)]);
        a.shadow_to = Some(Some(1));
        let mut b = test_entry(vec![Vec4::ONE; 2], None);
        b.place = Arc::new(vec![m(2.0), m(4.0)]);
        // a level at the same places as its caster keeps its numbering
        let mut c = test_entry(vec![Vec4::ONE; 2], None);
        c.place = b.place.clone();
        c.shadow_to = Some(Some(1));
        let mut es = vec![a, b, c];
        map_casters(&mut es);
        assert_eq!(es[0].cast_map.as_deref(), Some(&[2, 0, 3][..]));
        assert_eq!(es[1].extra, vec![m(1.0), m(3.0)]);
        assert_eq!(es[1].cast_len(), 4);
        assert!(es[2].cast_map.is_none());
        es[1].extra_tail = vec![[0.0; 4]; 2];
        let mut row = Vec::new();
        es[1].push_row(3, &mut row);
        assert_eq!(row[12], 3.0);
    }

    /// The occupied-cell walk keeps every instance the per-instance test
    /// passes, and hands each mesh exactly what walking its own runs did.
    #[test]
    fn occupied_cells_give_the_same_candidates() {
        let mut r = Rng(7);
        for trial in 0..12 {
            let span = if trial < 6 { 3000.0 } else { 40_000.0 };
            let mut entries = scene(&mut r, span);
            let mut g = grid_entries(&mut entries).unwrap();
            // only occupied cells are stored
            assert!(g.boxes.len() < (g.nx * g.nz) as usize);
            assert_eq!(g.refs.len(), entries.iter().map(|e| e.runs.len()).sum::<usize>());
            for v in 0..8 {
                let eye = Vec3::new(r.f() * span, 2.0 + r.f() * 40.0, r.f() * span);
                let at = eye + Vec3::new(r.f() - 0.5, -0.1, r.f() - 0.5);
                let planes = frustum(eye, at);
                let k = if v % 2 == 0 { 0.0 } else { 0.002 };
                g.update(&planes, eye);
                let tiny = if k > 0.0 { Tiny::Dist(k * k) } else { Tiny::No };
                let mut out = Vec::new();
                gather(&g, &entries, |e| e.lod, 1.0, tiny, &mut out);
                for (ei, e) in entries.iter().enumerate() {
                    if e.runs.is_empty() {
                        assert!(out[ei].is_empty());
                        continue;
                    }
                    // the old way: this mesh's own runs, every one checked
                    let mut old = Vec::new();
                    for (ri, &(c, a, b)) in e.runs.iter().enumerate() {
                        let c = c as usize;
                        if !g.vis[c] || (k > 0.0 && e.run_r[ri].powi(2) < k * k * g.dist[c][0]) {
                            continue;
                        }
                        if let Some([lo2, hi2]) = e.lod {
                            if g.dist[c][1] < lo2 || g.dist[c][0] >= hi2 {
                                continue;
                            }
                        }
                        old.extend_from_slice(&e.order[a as usize..b as usize]);
                    }
                    old.sort_unstable();
                    assert_eq!(out[ei], old, "trial {trial} view {v} entry {ei}");
                    for (i, s) in e.spheres.iter().enumerate() {
                        if passes(e, *s, &planes, eye, k) {
                            assert!(out[ei].binary_search(&(i as u32)).is_ok(), "visible instance dropped");
                        }
                    }
                }
            }
        }
    }
}
