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
    /// an AI car's cabin: within `detail_m` of the eye (race.rs)
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
}

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
    on: bool,
    lod_scale: f32,
    meshes: Vec<Mesh>,
    frame: u64,
    levels: Vec<Level>,
    /// rows drawn / put in the last flush, and how many went to a coarse LOD
    pub drawn: usize,
    pub total: usize,
    pub coarse: usize,
}

impl Default for CarCull {
    fn default() -> Self {
        CarCull {
            on: std::env::var("EVOVIEW_CAR_CULL").map_or(true, |v| v != "0"),
            lod_scale: std::env::var("EVOVIEW_CAR_LOD_SCALE").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(2.0),
            meshes: Vec::new(),
            frame: 0,
            levels: Vec::new(),
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

    /// Cull and upload every mesh `put` this frame. `vp` / `eye`: the camera
    /// as drawn; `sun`: toward the key light; `detail_m`: see `Detail`.
    #[allow(clippy::too_many_arguments)]
    pub fn flush(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, meshes: &mut [crate::render::GpuMesh],
                 mut motion: Option<&mut crate::motion::Motion>, vp: Mat4, eye: Vec3, sun: Vec3, detail_m: f32) {
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
        // 1/tan(half the vertical fov) over a 60-degree view's (1.732): a
        // long lens sees a far car as a near one
        let zoom = (vp.row(1).truncate().length() / 1.732).max(1e-3);
        self.drawn = 0;
        self.total = 0;
        self.coarse = 0;
        for gi in 0..self.meshes.len().min(meshes.len()) {
            if !self.meshes[gi].put {
                // a car mesh not given rows this frame (an AI look parked
                // off the track): its LODs draw nothing either
                for &(_, l) in &self.meshes[gi].lods {
                    if let Some(x) = meshes.get_mut(l) {
                        x.instance_count = 0;
                    }
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
            let prev_ok = m.prev_frame + 1 == frame && m.prev.len() == m.cur.len();
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
                let keep = match m.detail {
                    Detail::No => true,
                    Detail::Near => d < detail_m,
                    Detail::Seen => d < detail_m * zoom,
                };
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
                let li = if lod_scale > 0.0 { m.lods.iter().take_while(|l| de >= l.0 * lod_scale).count() } else { 0 };
                let l = &mut self.levels[li];
                l.rows.extend_from_slice(row);
                l.prev.extend_from_slice(if prev_ok { &m.prev[k * ROW..(k + 1) * ROW] } else { row });
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
                if cnt > 0 {
                    let cap = (meshes[g].instances.size() as usize / 4).min(l.rows.len());
                    queue.write_buffer(&meshes[g].instances, 0, bytemuck::cast_slice(&l.rows[..cap]));
                    if let Some(mo) = motion.as_deref_mut() {
                        mo.set_prev(device, queue, meshes, g, &l.prev);
                    }
                }
                meshes[g].instance_count = (cnt.min(meshes[g].instances.size() as usize / (ROW * 4))) as u32;
            }
        }
        // EVOVIEW_CARCULL_LOG=1: the counts, twice a second at 60 fps
        if self.frame % 30 == 0 && std::env::var_os("EVOVIEW_CARCULL_LOG").is_some() {
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
