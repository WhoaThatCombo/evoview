//! Per-frame culling and LOD choice for the moving cars: AI opponents,
//! replay cars, online players.
//!
//! cull.rs culls static scenery only. The cars rewrite their instance rows
//! every frame, and none of them was culled: a 53-car Nordschleife replay
//! pushed every car's whole model (an M4 GT3 is ~0.5 M triangles, its cabin
//! alone more than its body) through the main pass, the shadow maps and the
//! motion-vector pass, wherever the car was. Those passes are vertex bound,
//! and most of the grid is kilometres away or behind the camera.
//!
//! The frame loop now hands each car mesh's FULL rows (one per car, in a
//! fixed order) to `put`. `flush` runs once the frame's camera is final, and
//! for each mesh it:
//! - keeps the cars that can show: inside the view frustum, or close enough
//!   that their shadow can fall into it (the car's bounding sphere swept
//!   along the sun to the ground);
//! - drops cars that are not there (a zero matrix: a replay stream that has
//!   ended, a driver hidden in the onboard view);
//! - for "detail" meshes (cabin, driver, belts, nets), also drops the cars
//!   too far away for them to show;
//! - sends each kept car to one of the mesh's LODs (the game's own coarser
//!   meshes, load.rs CAR_LODS) by its distance, scaled by the camera's zoom;
//! - writes each LOD's cars packed, and sets its draw count;
//! - gives motion.rs LAST frame's rows of those same cars, in the same order
//!   (a GPU copy of the buffer would pair one car's old matrix with another
//!   car's new one whenever the visible set changes).
//!
//! The shadow passes also ask `casts_into` per mesh: a mesh whose drawn cars
//! all lie outside a light's box draws nothing into that map.
//!
//! The kept rows of every mesh and LOD go into ONE buffer (`rows`), and the
//! same cars' previous rows into another: two uploads a frame instead of two
//! per mesh and LOD (1,300 queue writes a frame with 69 different cars on
//! screen, each its own staging buffer and copy at submit), and passes that
//! draw many car meshes keep one instance buffer bound. A mesh's own
//! instance buffer is still written when something outside this scheme reads
//! it (`own`: the bloom's glow pass draws the emissive batches).
//! EVOVIEW_CAR_ROWS_SHARED=0: each mesh's own buffer, as before.
//!
//! Within a mesh's rows, the cars inside the near shadow cascade's box come
//! first (`upload`), and the near shadow map draws only those
//! (`near_count`): a mesh used to cast with ALL its drawn cars as soon as
//! one of them reached the box, and with a grid spread over the track every
//! look had one (200 cars: 1.6 ms of the GPU's frame in a map a handful of
//! them can reach). A car outside the box's sides leaves nothing in the
//! map, so the map is the same. EVOVIEW_CAR_SHADOW_NEAR=0: all of them.
//!
//! EVOVIEW_CAR_CULL=0: no view test (every car drawn wherever it is; the
//! cabin distance rule, the LODs and the dropping of absent cars still
//! apply). EVOVIEW_CAR_LOD_SCALE: the LOD distances' multiplier (2; 0 =
//! always LOD0).

use glam::{Mat4, Vec3, Vec4};

/// Floats per instance row (matrix 16 + facing 4).
const ROW: usize = 20;
/// Added to each mesh's bounding radius: skinning moves vertices a little
/// past their rest place (suspension travel, steering, the driver's arms).
const SKIN_MARGIN: f32 = 1.0;
/// Every car is at least this big: a part placed by its bones far from the
/// mesh origin still lies inside.
const MIN_RADIUS: f32 = 4.0;
/// For callers testing a whole car (`visible`): its meshes' largest
/// radius plus this, at least MIN_REACH - never less than a mesh's own test.
pub const REACH_MARGIN: f32 = SKIN_MARGIN + 0.5;
pub const MIN_REACH: f32 = MIN_RADIUS + 0.5;
/// Shadows longer than this are not followed off screen (a sun within ~1
/// degree of the horizon).
const MAX_SHADOW: f32 = 150.0;
/// The height a car's shadow is cast from (roof plus a wing or a driver's
/// head, generously).
const CAR_HEIGHT: f32 = 2.5;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Detail {
    /// drawn wherever the car is drawn
    No,
    /// an AI car's cabin: within `detail_m` of the eye (race.rs), further
    /// through a lens narrower than 60 degrees
    Near,
    /// a replay car's cabin: within `detail_m` at a 60-degree view, scaled
    /// by the camera's zoom (a TV camera's long lens keeps it)
    Seen,
}

struct Mesh {
    /// this frame's full rows
    cur: Vec<f32>,
    /// `cur` holds this frame's rows
    put: bool,
    /// last frame's full rows, and the frame they were put on
    prev: Vec<f32>,
    prev_frame: u64,
    detail: Detail,
    /// coarser LODs: (switch distance, gpu mesh), nearest first
    lods: Vec<(f32, usize)>,
    /// bounding sphere of the rows drawn this frame (w = 0: none), and the
    /// frame it was set on
    bound: Vec4,
    bound_frame: u64,
    /// per car: the LOD it was drawn with (HYST_UNSET: none yet) and whether
    /// its detail rule kept it, for the hysteresis
    last_li: Vec<u8>,
    last_keep: Vec<bool>,
}

/// LOD / cabin switches need this far past their distance (5 % either
/// way): a car holding station at a switch distance flipped every frame.
const HYST: f32 = 0.05;
const HYST_UNSET: u8 = u8::MAX;

/// One LOD's rows being gathered in `flush`.
#[derive(Default)]
struct Level {
    rows: Vec<f32>,
    prev: Vec<f32>,
    lo: Vec3,
    hi: Vec3,
    r: f32,
}

pub struct CarCull {
    /// the kept rows of every mesh in one buffer (see the module notes)
    shared_on: bool,
    /// this frame's kept rows, every mesh and LOD one after the other, and
    /// last frame's rows of the same cars in the same places
    rows: Vec<f32>,
    prev_rows: Vec<f32>,
    rows_buf: Option<wgpu::Buffer>,
    prev_buf: Option<wgpu::Buffer>,
    /// `prev_buf` holds the last flush's previous rows
    has_prev: bool,
    /// per gpu mesh: the frame its rows last went into `rows`, and its
    /// first row there
    at: Vec<(u64, u32)>,
    /// per gpu mesh: the largest bounding radius among those rows' cars
    reach: Vec<f32>,
    /// per gpu mesh: the frame `upload` sorted its rows on, and how many of
    /// them (the first ones) are inside the near shadow box
    near: Vec<(u64, u32)>,
    near_on: bool,
    on: bool,
    lod_scale: f32,
    meshes: Vec<Mesh>,
    frame: u64,
    levels: Vec<Level>,
    /// the last flush's view test: frustum planes and shadow offset
    planes: [Vec4; 6],
    shadow: Vec3,
    /// the next flush pairs no previous rows (a replay seek / loop: every
    /// car jumped, and last frame's rows would be screen-long motion)
    cut: bool,
    /// rows drawn / put in the last flush, and how many went to a coarse LOD
    pub drawn: usize,
    pub total: usize,
    pub coarse: usize,
}

impl Default for CarCull {
    fn default() -> Self {
        CarCull {
            shared_on: crate::envc::var("EVOVIEW_CAR_ROWS_SHARED").map_or(true, |v| v != "0"),
            rows: Vec::new(),
            prev_rows: Vec::new(),
            rows_buf: None,
            prev_buf: None,
            has_prev: false,
            at: Vec::new(),
            reach: Vec::new(),
            near: Vec::new(),
            near_on: crate::envc::var("EVOVIEW_CAR_SHADOW_NEAR").map_or(true, |v| v != "0"),
            on: crate::envc::var("EVOVIEW_CAR_CULL").map_or(true, |v| v != "0"),
            lod_scale: crate::envc::var("EVOVIEW_CAR_LOD_SCALE").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(2.0),
            meshes: Vec::new(),
            frame: 0,
            levels: Vec::new(),
            planes: [Vec4::ZERO; 6],
            shadow: Vec3::ZERO,
            cut: false,
            drawn: 0,
            total: 0,
            coarse: 0,
        }
    }
}

fn frustum(vp: Mat4) -> [Vec4; 6] {
    let r = [vp.row(0), vp.row(1), vp.row(2), vp.row(3)];
    let mut p = [r[3] + r[0], r[3] - r[0], r[3] + r[1], r[3] - r[1], r[2], r[3] - r[2]];
    for q in p.iter_mut() {
        let l = q.truncate().length();
        if l > 0.0 {
            *q /= l;
        }
    }
    p
}

fn sphere_in(planes: &[Vec4], c: Vec3, r: f32) -> bool {
    planes.iter().all(|p| p.truncate().dot(c) + p.w >= -r)
}

impl CarCull {
    fn mesh(&mut self, gi: usize) -> &mut Mesh {
        if self.meshes.len() <= gi {
            self.meshes.resize_with(gi + 1, || Mesh {
                cur: Vec::new(),
                put: false,
                prev: Vec::new(),
                prev_frame: 0,
                detail: Detail::No,
                lods: Vec::new(),
                bound: Vec4::ZERO,
                bound_frame: 0,
                last_li: Vec::new(),
                last_keep: Vec::new(),
            });
        }
        &mut self.meshes[gi]
    }

    /// Mark mesh `gi` as a cabin / driver part (see `Detail`).
    pub fn set_detail(&mut self, gi: usize, d: Detail) {
        self.mesh(gi).detail = d;
    }

    /// `lod` (a gpu mesh) is a coarser LOD of `gi`, from `dist` (the game's
    /// switch distance) on. It draws nothing until `flush` gives it cars.
    pub fn add_lod(&mut self, meshes: &mut [crate::render::GpuMesh], gi: usize, dist: f32, lod: usize) {
        if gi >= meshes.len() || lod >= meshes.len() {
            return;
        }
        meshes[lod].instance_count = 0;
        self.mesh(lod);
        let m = self.mesh(gi);
        m.lods.push((dist, lod));
        m.lods.sort_by(|a, b| a.0.total_cmp(&b.0));
    }

    /// Every coarse LOD mesh (they move with their cars: motion.rs draws them).
    pub fn lod_meshes(&self) -> impl Iterator<Item = usize> + '_ {
        self.meshes.iter().flat_map(|m| m.lods.iter().map(|l| l.1))
    }

    /// This frame's rows for car mesh `gi`: one per car, always in the same
    /// order (uploaded by `flush`).
    pub fn put(&mut self, gi: usize, rows: &[f32]) {
        let m = self.mesh(gi);
        m.cur.clear();
        m.cur.extend_from_slice(rows);
        m.put = true;
    }

    /// Whether any car drawn from mesh `gi` this frame lies inside the
    /// orthographic light box `light_vp` across (its sides only: casters
    /// nearer the light than its near plane still cast). Meshes this does
    /// not manage always do.
    pub fn casts_into(&self, gi: usize, light_vp: Mat4) -> bool {
        let Some(m) = self.meshes.get(gi).filter(|m| self.on && m.bound_frame == self.frame && self.frame > 0) else {
            return true;
        };
        if m.bound.w <= 0.0 {
            return true;
        }
        let p = frustum(light_vp);
        sphere_in(&p[..4], m.bound.truncate(), m.bound.w)
    }

    /// Every car teleported (a replay seek or loop): the next flush gives
    /// motion.rs each car's current row as its previous one.
    pub fn cut(&mut self) {
        self.cut = true;
    }

    /// Whether mesh `gi` is a car mesh this culls (given rows, or a coarse
    /// LOD): in drive mode, the opponents' meshes and not the player's.
    pub fn manages(&self, gi: usize) -> bool {
        self.meshes.get(gi).map_or(false, |m| m.prev_frame > 0 || m.bound_frame > 0 || !m.lods.is_empty())
    }

    /// Whether a car at `c` (bounding radius `r`) passes this frame's view
    /// test (`flush` sets it up): what decides if its rig needs posing.
    pub fn visible(&self, c: Vec3, r: f32) -> bool {
        !self.on || self.frame == 0 || sphere_in(&self.planes, c, r)
            || (self.shadow != Vec3::ZERO && sphere_in(&self.planes, c + self.shadow * 0.5, r + self.shadow.length() * 0.5))
    }

    /// Where mesh `gi`'s instance rows are this frame, when `flush` put them
    /// in the shared buffers: (this frame's rows, last frame's rows of the
    /// same cars when a motion pass was given, the mesh's first row). Its
    /// `instance_count` rows from there; None: the mesh's own buffer.
    pub fn rows(&self, gi: usize) -> Option<(&wgpu::Buffer, Option<&wgpu::Buffer>, u32)> {
        let &(frame, first) = self.at.get(gi)?;
        if frame != self.frame || frame == 0 {
            return None;
        }
        Some((self.rows_buf.as_ref()?, self.prev_buf.as_ref().filter(|_| self.has_prev), first))
    }

    /// How many of mesh `gi`'s rows (the first ones, from `rows`) hold cars
    /// that can cast into the near shadow map `upload` was given. None: not
    /// sorted - all of them may.
    pub fn near_count(&self, gi: usize) -> Option<u32> {
        self.near.get(gi).filter(|n| n.0 == self.frame && self.frame > 0).map(|n| n.1)
    }

    /// Upload the rows `flush` gathered into the shared buffers: once a
    /// frame, after `flush` and before anything is drawn. `near`: the near
    /// shadow cascade's light matrix - within each mesh's rows, the cars
    /// inside that box (its sides: see `casts_into`) are moved to the front
    /// and counted (`near_count`). Their order is free: every pass draws a
    /// mesh's rows as one instanced range, and last frame's rows move with
    /// them.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, meshes: &[crate::render::GpuMesh], near: Option<Mat4>) {
        if self.rows.is_empty() {
            return;
        }
        let _t = crate::bench::cpu("car cull");
        if let Some(lv) = near.filter(|_| self.near_on && self.on) {
            let p = frustum(lv);
            if self.near.len() < self.at.len() {
                self.near.resize(self.at.len(), (0, 0));
            }
            let with_prev = self.has_prev && self.prev_rows.len() == self.rows.len();
            for g in 0..self.at.len().min(meshes.len()) {
                let (frame, first) = self.at[g];
                if frame != self.frame {
                    continue;
                }
                let (first, n) = (first as usize, meshes[g].instance_count as usize);
                if (first + n) * ROW > self.rows.len() {
                    continue;
                }
                let r = self.reach.get(g).copied().unwrap_or(f32::MAX);
                let mut k = 0;
                for i in 0..n {
                    let o = (first + i) * ROW;
                    let c = Vec3::new(self.rows[o + 12], self.rows[o + 13], self.rows[o + 14]);
                    if sphere_in(&p[..4], c, r) {
                        if i != k {
                            let q = (first + k) * ROW;
                            for j in 0..ROW {
                                self.rows.swap(o + j, q + j);
                                if with_prev {
                                    self.prev_rows.swap(o + j, q + j);
                                }
                            }
                        }
                        k += 1;
                    }
                }
                self.near[g] = (self.frame, k as u32);
            }
        }
        // every mesh's rows in one write (and their previous rows in another)
        let write = |buf: &mut Option<wgpu::Buffer>, rows: &[f32], label: &str| {
            let need = (rows.len() * 4) as u64;
            if buf.as_ref().map_or(true, |b| b.size() < need) {
                *buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: need.next_power_of_two().max(64 * 1024),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            if let Some(b) = buf.as_ref() {
                queue.write_buffer(b, 0, bytemuck::cast_slice(rows));
            }
        };
        crate::bench::det("car_rows", &self.rows);
        write(&mut self.rows_buf, &self.rows, "car rows");
        if self.has_prev {
            write(&mut self.prev_buf, &self.prev_rows, "car rows (last frame)");
        }
    }

    /// Cull every mesh `put` this frame, and gather (shared rows: `upload`
    /// sends them) or upload its rows. `vp` / `eye`: the camera
    /// as drawn; `sun`: toward the key light; `detail_m`: see `Detail`.
    /// `own(material)`: a batch of this material is drawn by something that
    /// reads its mesh's own instance buffer (the glow pass).
    #[allow(clippy::too_many_arguments)]
    pub fn flush(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, meshes: &mut [crate::render::GpuMesh],
                 mut motion: Option<&mut crate::motion::Motion>, own: &dyn Fn(usize) -> bool,
                 vp: Mat4, eye: Vec3, sun: Vec3, detail_m: f32) {
        let _t = crate::bench::cpu("car cull");
        self.frame += 1;
        let planes = frustum(vp);
        // where a point CAR_HEIGHT up casts its shadow, relative to it (0: no sun)
        let shadow = if sun.y > 0.01 {
            let s = -sun / sun.y;
            let l = s.length() * CAR_HEIGHT;
            s * (CAR_HEIGHT * (MAX_SHADOW / l).min(1.0))
        } else {
            Vec3::ZERO
        };
        self.planes = planes;
        self.shadow = shadow;
        // 1/tan(half the vertical fov) over a 60-degree view's (1.732): a
        // long lens sees a far car as a near one
        let zoom = (vp.row(1).truncate().length() / 1.732).max(1e-3);
        self.drawn = 0;
        self.total = 0;
        self.coarse = 0;
        let cut = std::mem::take(&mut self.cut);
        let want_prev = motion.is_some();
        self.rows.clear();
        self.prev_rows.clear();
        if self.at.len() < meshes.len() {
            self.at.resize(meshes.len(), (0, 0));
            self.reach.resize(meshes.len(), 0.0);
        }
        for gi in 0..self.meshes.len().min(meshes.len()) {
            if !self.meshes[gi].put {
                // a car mesh not given rows this frame (an AI look parked
                // off the track): its LODs draw nothing either
                for &(_, l) in &self.meshes[gi].lods {
                    if let Some(x) = meshes.get_mut(l) {
                        x.instance_count = 0;
                    }
                }
                // (its last rows were in the shared buffer, which now holds
                // other meshes': there is nothing left to draw it from)
                if self.at[gi].0 > 0 && self.at[gi].0 + 1 == self.frame {
                    meshes[gi].instance_count = 0;
                }
                continue;
            }
            let radius = (meshes[gi].radius + SKIN_MARGIN).max(MIN_RADIUS);
            let frame = self.frame;
            let on = self.on;
            let lod_scale = self.lod_scale;
            let m = &mut self.meshes[gi];
            m.put = false;
            let n = m.cur.len() / ROW;
            m.last_li.resize(n, HYST_UNSET);
            m.last_keep.resize(n, true);
            let prev_ok = !cut && m.prev_frame + 1 == frame && m.prev.len() == m.cur.len();
            let nl = m.lods.len() + 1;
            if self.levels.len() < nl {
                self.levels.resize_with(nl, Level::default);
            }
            for l in self.levels[..nl].iter_mut() {
                l.rows.clear();
                l.prev.clear();
                l.lo = Vec3::splat(f32::MAX);
                l.hi = Vec3::splat(f32::MIN);
                l.r = 0.0;
            }
            for k in 0..n {
                let row = &m.cur[k * ROW..(k + 1) * ROW];
                let x = Mat4::from_cols_slice(&row[..16]);
                // a zero matrix: this car is not there
                if x.w_axis.w == 0.0 && x.x_axis == Vec4::ZERO {
                    continue;
                }
                let c = x.w_axis.truncate();
                let scale = x.x_axis.truncate().length().max(x.y_axis.truncate().length()).max(x.z_axis.truncate().length());
                let r = radius * scale.max(1.0);
                let d = c.distance(eye);
                // kept: it must go 5 % past the distance to be dropped
                let hk = if m.last_keep[k] { 1.0 + HYST } else { 1.0 - HYST };
                let keep = match m.detail {
                    Detail::No => true,
                    // a long lens (TV cameras) fills the screen with a car far
                    // past detail_m: keep its cabin as far as the zoom reaches
                    Detail::Near => d < detail_m * zoom.max(1.0) * hk,
                    Detail::Seen => d < detail_m * zoom * hk,
                };
                m.last_keep[k] = keep;
                if !keep {
                    continue;
                }
                let seen = !on || sphere_in(&planes, c, r)
                    || (shadow != Vec3::ZERO && sphere_in(&planes, c + shadow * 0.5, r + shadow.length() * 0.5));
                if !seen {
                    continue;
                }
                // the coarsest LOD whose switch distance it is past
                let de = d / zoom;
                // (a switch toward the level it was on is 5 % easier)
                let was = m.last_li[k];
                let li = if lod_scale > 0.0 {
                    m.lods.iter().enumerate().take_while(|(j, l)| {
                        let h = if was == HYST_UNSET { 1.0 } else if (*j as u8) < was { 1.0 - HYST } else { 1.0 + HYST };
                        de >= l.0 * lod_scale * h
                    }).count()
                } else { 0 };
                m.last_li[k] = li as u8;
                let l = &mut self.levels[li];
                l.rows.extend_from_slice(row);
                // ⚠ a car not there last frame (zero row: a stream starting,
                // a driver un-hidden, a car leaving its parking spot) moved
                // from the origin as far as motion.rs could tell: its own row
                let pr = if prev_ok { &m.prev[k * ROW..(k + 1) * ROW] } else { row };
                let gone = pr[15] == 0.0 && pr[..4].iter().all(|&v| v == 0.0);
                let parked = (Vec3::new(pr[12], pr[13], pr[14]) - c).length_squared() > 50.0 * 50.0;
                l.prev.extend_from_slice(if gone || parked { row } else { pr });
                l.lo = l.lo.min(c);
                l.hi = l.hi.max(c);
                l.r = l.r.max(r);
            }
            self.total += n;
            std::mem::swap(&mut m.prev, &mut m.cur);
            m.prev_frame = frame;
            for li in 0..nl {
                let g = if li == 0 { gi } else { self.meshes[gi].lods[li - 1].1 };
                let l = &self.levels[li];
                let cnt = l.rows.len() / ROW;
                self.drawn += cnt;
                if li > 0 {
                    self.coarse += cnt;
                }
                let b = if cnt > 0 { ((l.lo + l.hi) * 0.5).extend((l.hi - l.lo).length() * 0.5 + l.r) } else { Vec4::ZERO };
                let me = &mut self.meshes[g];
                me.bound = b;
                me.bound_frame = frame;
                if g >= meshes.len() {
                    continue;
                }
                // (never more rows than the mesh's own buffer holds)
                let keep = cnt.min(meshes[g].instances.size() as usize / (ROW * 4));
                if keep > 0 && self.shared_on {
                    self.at[g] = (frame, (self.rows.len() / ROW) as u32);
                    self.reach[g] = l.r;
                    self.rows.extend_from_slice(&l.rows[..keep * ROW]);
                    if want_prev {
                        self.prev_rows.extend_from_slice(&l.prev[..keep * ROW]);
                    }
                    if meshes[g].draws.iter().any(|d| own(d.2)) {
                        queue.write_buffer(&meshes[g].instances, 0, bytemuck::cast_slice(&l.rows[..keep * ROW]));
                    }
                } else if keep > 0 {
                    queue.write_buffer(&meshes[g].instances, 0, bytemuck::cast_slice(&l.rows[..keep * ROW]));
                    if let Some(mo) = motion.as_deref_mut() {
                        mo.set_prev(device, queue, meshes, g, &l.prev);
                    }
                }
                meshes[g].instance_count = keep as u32;
            }
        }
        self.has_prev = want_prev;
        // EVOVIEW_CARCULL_LOG=1: the counts, twice a second at 60 fps
        if self.frame % 30 == 0 && crate::envc::var_os("EVOVIEW_CARCULL_LOG").is_some() {
            println!("carcull: {} of {} rows drawn, {} on a coarse LOD", self.drawn, self.total, self.coarse);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_keeps_in_front_and_drops_behind() {
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y);
        let proj = Mat4::perspective_rh(1.0, 1.5, 0.1, 5000.0);
        let p = frustum(proj * view);
        assert!(sphere_in(&p, Vec3::new(0.0, 0.0, -50.0), 4.0));
        assert!(!sphere_in(&p, Vec3::new(0.0, 0.0, 50.0), 4.0));
        // just behind the eye but its sphere reaches in front
        assert!(sphere_in(&p, Vec3::new(0.0, 0.0, 3.0), 4.0));
        assert!(!sphere_in(&p, Vec3::new(500.0, 0.0, -50.0), 4.0));
    }

    #[test]
    fn lod_names_round_trip() {
        assert_eq!(crate::load::car_lod_of("x_exterior.mesh@lod2@20"), Some(("x_exterior.mesh", 20.0)));
        assert_eq!(crate::load::car_lod_of("x_exterior.mesh"), None);
    }
}
