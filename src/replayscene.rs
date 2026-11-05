//! Building one renderable scene out of a track plus the cars in a replay.
//!
//! The renderer draws a single `asset::Car`, so the track and every car model
//! the replay needs are merged into one. Each source contributes its meshes,
//! textures and materials; blob ranges and texture indices are shifted as they
//! are concatenated.
//!
//! Car models are loaded ONCE each, however many drivers use them - a 24-car
//! GT3 grid is usually six or seven distinct cars. Each model's meshes are then
//! given one GPU instance per driver using it, and those instance matrices are
//! rewritten every frame from the replay. That is why `build` hands back
//! `slots`: the renderer needs to know which meshes belong to which model.

use crate::asset::{Car, MeshEntry};
use crate::replay::Replay;
use glam::{Mat4, Vec3};
use std::path::Path;

/// Where one car model landed in the merged scene.
#[derive(Clone)]
pub struct Slot {
    pub model: String,
    /// Body meshes - everything that moves rigidly with the hull.
    pub meshes: Vec<usize>,
    /// Rim and tyre meshes per replay wheel index (0 FR, 1 FL, 2 RR, 3 RL).
    pub wheels: [Vec<usize>; 4],
    /// Where each wheel is authored in car space. The meshes are baked at
    /// their bone position, so they have to be brought back to the origin
    /// before being re-placed at the replay's world wheel position.
    pub centres: [Vec3; 4],
    /// Each wheel's axle in car space (unit, +X side): the tyre's thinnest
    /// direction. ⚠ Not always X: the F1 cars' wheels carry 3 deg of static
    /// camber, and spinning them about X wobbled them once a turn.
    pub axles: [Vec3; 4],
    /// Rolling radius, from the tyre mesh, used to turn distance into spin.
    pub radius: f32,
    /// Mesh-space shift that lines the body up with its own wheels.
    pub offset: Vec3,
    /// Which replay cars are drawn with this model, in instance order.
    pub cars: Vec<usize>,
    /// The model's skeleton with its rig (replays): each car's rig is posed
    /// from its own recording - steering wheel, driver, needles, suspension,
    /// body roll - and the wheels ride their bones, not `wheel_matrix`
    pub skel: Option<std::sync::Arc<crate::load::Skeleton>>,
    /// The model's own cockpit camera (actor cameras, mesh space): the
    /// replay's onboard view
    pub cockpit: Option<crate::fxdata::CarCam>,
    /// Coarser LODs of body meshes (load.rs CAR_LODS): (the LOD0 mesh in
    /// `meshes`, switch distance, the LOD's mesh). Not in `meshes`: only
    /// carcull.rs draws them, in place of LOD0 on far cars.
    pub lods: Vec<(usize, f32, usize)>,
}

/// Pair each `@lod` mesh in `range` with its LOD0 (same base name).
fn collect_lods(meshes: &[MeshEntry], range: std::ops::Range<usize>, lods: &mut Vec<(usize, f32, usize)>) {
    for i in range.clone() {
        let Some((base, dist)) = crate::load::car_lod_of(&meshes[i].name) else { continue };
        if let Some(j) = range.clone().find(|&j| meshes[j].name == base) {
            lods.push((j, dist, i));
        }
    }
}

/// Mesh-name corner code -> replay wheel index. The replay stores wheels
/// front-RIGHT, front-left, rear-RIGHT, rear-left; evoview names them by
/// corner, so the two orders have to be matched up explicitly.
fn corner_index(name: &str) -> Option<usize> {
    if !(name.starts_with("rim_") || name.starts_with("tyre_")) {
        return None;
    }
    match name.rsplit('_').next()? {
        "RF" => Some(0),
        "LF" => Some(1),
        "RR" => Some(2),
        "LR" => Some(3),
        _ => None,
    }
}

pub struct Scene {
    pub car: Car,
    pub slots: Vec<Slot>,
    /// Number of merged meshes that belong to the track rather than a car.
    pub track_meshes: usize,
    /// build_single only: the one car's skeleton (with its rig) and clips,
    /// so a driven car keeps its steering, needles, lights and doors
    pub skeleton: Option<crate::load::Skeleton>,
    pub animations: Vec<crate::load::Animation>,
}

/// Per-car wheel animation, precomputed once so scrubbing and looping stay
/// exact - integrating spin frame by frame during playback would drift and
/// would be wrong the moment the user scrubs backwards.
pub struct Wheels {
    /// Cumulative wheel rotation per frame, in radians.
    pub dist: Vec<Vec<f32>>,
    /// Front steer angle per frame, in radians.
    pub steer: Vec<Vec<f32>>,
}

impl Wheels {
    /// Both tables come from the car's own telemetry now that the base-frame
    /// block is decoded: steering is the recorded wheel angle rather than a
    /// bicycle-model guess, and spin is integrated from the recorded per-wheel
    /// angular velocity rather than from distance travelled - so a locked or
    /// spinning wheel looks locked or spinning instead of always matching road
    /// speed.
    pub fn build(rep: &Replay) -> Wheels {
        let rate = rep.rate();
        let dt = 1.0 / rate;
        let mut dist = Vec::with_capacity(rep.cars.len());
        let mut steer = Vec::with_capacity(rep.cars.len());
        for (ci, c) in rep.cars.iter().enumerate() {
            let n = c.n_matrix as usize;
            let mut d = Vec::with_capacity(n);
            let mut s = Vec::with_capacity(n);
            let mut spin = 0.0f32;
            for f in 0..n {
                let t = f as f32 * dt;
                match rep.telemetry(ci, t) {
                    Some(tel) => {
                        // Average the four wheels: one spin angle drives all of
                        // them, which is wrong only while a single wheel locks.
                        let w = tel.wheel.iter().sum::<f32>() / 4.0;
                        spin += w * dt;
                        // The recorded angle is at the STEERING WHEEL. A GT3
                        // runs roughly 190 degrees of rim for 15 at the tyre,
                        // hence the ratio.
                        s.push((tel.steer_deg / 13.0).to_radians().clamp(-0.6, 0.6));
                    }
                    None => {
                        s.push(*s.last().unwrap_or(&0.0));
                    }
                }
                d.push(spin);
            }
            dist.push(d);
            steer.push(s);
        }
        Wheels { dist, steer }
    }

    /// Linearly sampled value from a per-frame table.
    fn at(table: &[f32], x: f32) -> f32 {
        if table.is_empty() {
            return 0.0;
        }
        let i = (x.floor().max(0.0) as usize).min(table.len() - 1);
        let j = (i + 1).min(table.len() - 1);
        table[i] + (table[j] - table[i]) * (x - i as f32).clamp(0.0, 1.0)
    }
}

/// Body matrix plus the four wheel matrices for one car at one instant.
pub fn car_pose(
    rep: &Replay, w: &Wheels, slot: &Slot, car: usize, t: f32,
) -> Option<(Mat4, [Mat4; 4])> {
    let p = rep.sample(car, t)?;
    let x = t * rep.rate();
    let spin = Wheels::at(&w.dist[car], x);
    let steer = Wheels::at(&w.steer[car], x);
    let mut wm = [Mat4::ZERO; 4];
    for k in 0..4 {
        wm[k] = wheel_matrix(&p, k, slot.centres[k], slot.axles[k], spin, steer);
    }
    Some((car_matrix(&p, slot.offset), wm))
}

/// Shift every blob range in a mesh by `d`.
fn shift_mesh(m: &mut MeshEntry, d: u64) {
    for r in [
        &mut m.positions, &mut m.normals, &mut m.uvs,
        &mut m.tangents, &mut m.bone_idx, &mut m.bone_wgt, &mut m.indices, &mut m.colors, &mut m.uv1,
    ] {
        if let Some(v) = r.as_mut() {
            v[0] += d;
        }
    }
}

/// Merge `src` into the accumulating scene.
///
/// ⚠ Materials are looked up by BASENAME, not by path, so two cars that both
/// have `EXT_SKIN.material` would collide and the second car would render with
/// the first one's paint. Car materials are therefore renamed with a per-model
/// tag, and every batch that referenced them is rewritten to match. The track
/// is merged untagged: its material names drive the terrain and ground
/// heuristics in material.rs, and renaming them would break those.
fn merge(
    meshes: &mut Vec<MeshEntry>,
    textures: &mut Vec<crate::asset::TextureEntry>,
    // each texture's content identity for the GPU cache (Car::tex_src), in
    // step with `textures`. ⚠ Without it the merged scene had none and the
    // cache fell back to name + format + size + length: every GT car's
    // same-named, same-sized livery maps (and any matching track texture)
    // collapsed into the first one uploaded - wrong liveries that got worse
    // with every car added to the grid.
    tex_src: &mut Vec<String>,
    materials: &mut Vec<crate::asset::MaterialEntry>,
    blob: &mut Vec<u8>,
    src: &Car,
    tag: Option<&str>,
) -> std::ops::Range<usize> {
    let blob_off = blob.len() as u64;
    let tex_off = textures.len();
    blob.extend_from_slice(src.blob());

    tex_src.resize(tex_off, String::new());
    for i in 0..src.textures.len() {
        tex_src.push(src.tex_src.get(i).filter(|s| !s.is_empty()).cloned()
            // no vfs key: unique to this copy, never shared by name
            .unwrap_or_else(|| format!("merged:{}:{i}", tag.unwrap_or("track"))));
    }
    for t in &src.textures {
        let mut t = t.clone();
        t.data[0] += blob_off;
        textures.push(t);
    }
    for m in &src.materials {
        let mut m = m.clone();
        for v in m.slots.values_mut() {
            *v += tex_off;
        }
        if let Some(tag) = tag {
            m.file = format!("{tag}__{}", base_of(&m.file));
        }
        materials.push(m);
    }
    let start = meshes.len();
    for m in &src.meshes {
        let mut m = m.clone();
        shift_mesh(&mut m, blob_off);
        if let Some(tag) = tag {
            for b in &mut m.batches {
                b.material = format!("{tag}__{}", base_of(&b.material));
            }
        }
        meshes.push(m);
    }
    start..meshes.len()
}

/// `merge` for an AI car, unskinned, keeping only bytes the scene does not
/// hold yet: a texture whose content key (tex_src) is already in it points
/// at that copy, and the meshes copy their own ranges rather than the whole
/// blob. A second livery of a model then adds the maps that differ, not a
/// whole car's (13 GB commit -> ~8 at 4 looks of an M4 GT3).
fn merge_shared(
    meshes: &mut Vec<MeshEntry>,
    textures: &mut Vec<crate::asset::TextureEntry>,
    tex_src: &mut Vec<String>,
    materials: &mut Vec<crate::asset::MaterialEntry>,
    blob: &mut Vec<u8>,
    src: &Car,
    tag: &str,
) -> std::ops::Range<usize> {
    tex_src.resize(textures.len(), String::new());
    let have: std::collections::HashMap<String, usize> = tex_src.iter().enumerate()
        .filter(|(_, k)| !k.is_empty() && !k.starts_with("merged:"))
        .map(|(i, k)| (k.clone(), i))
        .collect();
    let mut copy = |blob: &mut Vec<u8>, r: [u64; 2]| -> u64 {
        while blob.len() % 16 != 0 {
            blob.push(0);
        }
        let off = blob.len() as u64;
        blob.extend_from_slice(src.slice(Some(r)));
        off
    };
    let mut map = Vec::with_capacity(src.textures.len());
    for (i, t) in src.textures.iter().enumerate() {
        let key = src.tex_src.get(i).filter(|s| !s.is_empty());
        if let Some(&j) = key.and_then(|k| have.get(k)) {
            map.push(j);
            continue;
        }
        let mut t = t.clone();
        t.data[0] = copy(blob, t.data);
        textures.push(t);
        tex_src.push(key.cloned().unwrap_or_else(|| format!("merged:{tag}:{i}")));
        map.push(textures.len() - 1);
    }
    for m in &src.materials {
        let mut m = m.clone();
        for v in m.slots.values_mut() {
            *v = map.get(*v).copied().unwrap_or(*v);
        }
        m.file = format!("{tag}__{}", base_of(&m.file));
        materials.push(m);
    }
    let start = meshes.len();
    for m in &src.meshes {
        let mut m = m.clone();
        m.bone_idx = None;
        m.bone_wgt = None;
        for r in [&mut m.positions, &mut m.normals, &mut m.uvs, &mut m.tangents, &mut m.indices, &mut m.colors, &mut m.uv1] {
            if let Some(v) = r.as_mut() {
                v[0] = copy(blob, *v);
            }
        }
        for b in &mut m.batches {
            b.material = format!("{tag}__{}", base_of(&b.material));
        }
        meshes.push(m);
    }
    start..meshes.len()
}

fn base_of(p: &str) -> &str {
    p.rsplit(|c| c == '\\' || c == '/').next().unwrap_or(p)
}

fn mesh_bounds(car: &Car, m: &MeshEntry) -> (Vec3, Vec3) {
    let p = car.floats(m.positions);
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for v in p.chunks_exact(3) {
        let v = Vec3::new(v[0], v[1], v[2]);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if lo.x > hi.x { (Vec3::ZERO, Vec3::ZERO) } else { (lo, hi) }
}

/// A tyre mesh's axle: the direction of least spread of its vertices (the
/// smallest eigenvector of their covariance), turned to +X. Falls back to X
/// when it is not near X (EVOVIEW_SPIN_AXIS=car: always X).
pub(crate) fn tyre_axle(car: &Car, m: &MeshEntry) -> Vec3 {
    if crate::envc::var("EVOVIEW_SPIN_AXIS").map_or(false, |v| v == "car") {
        return Vec3::X;
    }
    let p = car.floats(m.positions);
    let n = (p.len() / 3) as f64;
    if n < 16.0 {
        return Vec3::X;
    }
    let mut mean = glam::DVec3::ZERO;
    for v in p.chunks_exact(3) {
        mean += glam::DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64);
    }
    mean /= n;
    let mut c = glam::DMat3::ZERO;
    for v in p.chunks_exact(3) {
        let d = glam::DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64) - mean;
        c += glam::DMat3::from_cols(d * d.x, d * d.y, d * d.z);
    }
    c *= 1.0 / n;
    // power iteration on (tr I - C): its largest eigenvector is C's smallest
    let tr = c.x_axis.x + c.y_axis.y + c.z_axis.z;
    let m2 = glam::DMat3::from_diagonal(glam::DVec3::splat(tr)) - c;
    let mut a = glam::DVec3::X;
    for _ in 0..64 {
        a = (m2 * a).normalize_or_zero();
    }
    let a = a.as_vec3() * if a.x < 0.0 { -1.0 } else { 1.0 };
    if a.x > 0.985 { a } else { Vec3::X }
}

/// A car's world matrix for a pose, built from the wheel-derived basis.
///
/// evoview's car meshes face -Z (the same convention the game authors them
/// in), so the basis columns are laid out to match rather than to make the
/// maths look tidy.
/// Rotation-only part of the body transform.
///
/// ⚠ evoview's car meshes are authored with FRONT at +Z and RIGHT at -X, which
/// is 180 degrees about Y from the obvious `(right, up, -forward)` guess. That
/// guess renders every car backwards, and a grid of cars all facing the same
/// wrong way looks perfectly fine, so it does not show up by eye. Measured:
/// the mesh centre of `rim_RF` is (-0.845, 0.339, +1.332) while the replay puts
/// that same wheel at (+0.865, -0.027, -1.463) in the guessed basis - matching
/// magnitudes, flipped X and Z.
fn car_basis(p: &crate::replay::Pose) -> Mat4 {
    Mat4::from_cols(
        (-p.right).extend(0.0),
        p.up.extend(0.0),
        p.forward.extend(0.0),
        glam::Vec4::W,
    )
}

/// The particle materials' texture indices moved by `base` (where merge
/// put this car's textures).
fn shift_fx_textures(mut fx: Box<crate::fxdata::CarFx>, base: usize) -> Box<crate::fxdata::CarFx> {
    for d in fx.emitter_defs.iter_mut() {
        d.material.color_tex = d.material.color_tex.map(|i| i + base);
        d.material.opacity_tex = d.material.opacity_tex.map(|i| i + base);
    }
    // ⚠ the dash pages' images index car textures too: unshifted, every
    // digital dash on a track drew one of the TRACK's textures (the M4 GT3's
    // DDU 10 came out a lavender normal map, a washed-out white-ish screen)
    for pg in fx.dash.iter_mut() {
        for im in pg.images.iter_mut() {
            if let crate::dash::Image::Tex(i, _, _) = im {
                *i += base;
            }
        }
    }
    fx
}

/// Body transform. `offset` shifts the mesh so its wheels land on the replay's
/// wheel positions: the mesh origin sits on the ground between the wheels,
/// while the replay's body position is the centre of mass, about 0.37 m up and
/// slightly rearward. Without it the body floats above its own wheels.
pub fn car_matrix(p: &crate::replay::Pose, offset: Vec3) -> Mat4 {
    Mat4::from_translation(p.pos) * car_basis(p) * Mat4::from_translation(offset)
}

/// One wheel's world matrix.
///
/// Position and suspension travel are real - they come straight from the
/// replay's world-space wheel positions, so ride height, dive, squat and roll
/// all fall out for free. Spin and steer are DERIVED:
///
/// ponytail: steer is a bicycle-model estimate, `atan(wheelbase * yaw_rate /
/// speed)`, because the matrix frame does not carry a steering angle - no lane
/// in it correlates with yaw rate (best |r| = 0.24). Real steer is presumably
/// in the undecoded variable-length base-frame block; decode that if the
/// estimate ever looks wrong at low speed, where it is weakest.
pub(crate) fn wheel_matrix(
    body: &crate::replay::Pose, k: usize, centre: Vec3, axle: Vec3, spin: f32, steer: f32,
) -> Mat4 {
    let mut rot = car_basis(body);
    if k < 2 {
        rot *= Mat4::from_rotation_y(steer);   // front pair only
    }
    // The wheel turns about the car's lateral axis, which is local X. Positive:
    // a right-handed rotation about +X carries the top of the wheel toward +Z,
    // and +Z is the front of the mesh, so that is rolling forwards.
    // About the wheel's own axle (X but for a cambered wheel, see Slot::axles).
    rot *= Mat4::from_axis_angle(axle, spin);
    // Translate to the replay's world wheel position, NOT to a position
    // implied by the body - the difference between them is the suspension
    // travel, and using the body would throw it away.
    Mat4::from_translation(body.wheels[k]) * rot * Mat4::from_translation(-centre)
}

impl Scene {
    /// Load the track named by the replay plus every car model it uses.
    ///
    /// `pkg` is the game's content.kspkg; cars and tracks are read in place
    /// from it, so nothing is extracted to disk.
    pub fn build(rep: &Replay, pkg: &Path, base: Option<&Path>) -> Result<Scene, String> {
        let track = rep.track.as_deref()
            .ok_or("this replay does not name a track")?;
        let mut t = crate::load::load_track_any(pkg, track, base)
            .map_err(|e| format!("track {track}: {e}"))?;
        let track_env = t.car.track_env.take();

        let mut meshes = Vec::new();
        let mut textures = Vec::new();
        let mut materials = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut tex_src: Vec<String> = Vec::new();
        merge(&mut meshes, &mut textures, &mut tex_src, &mut materials, &mut blob, &t.car, None);
        let track_meshes = meshes.len();

        // Group drivers by model so each model is loaded and uploaded once.
        let mut slots: Vec<Slot> = Vec::new();
        for (i, c) in rep.cars.iter().enumerate() {
            if rep.is_placeholder(i) {
                continue;
            }
            // A car whose own model could not be resolved still has to be
            // drawn as something; the recording driver's car is the best
            // available guess and keeps the grid complete.
            let model = c.model.clone().unwrap_or_else(|| rep.model.clone());
            match slots.iter_mut().find(|s| s.model == model) {
                Some(s) => s.cars.push(i),
                None => slots.push(Slot {
                    model,
                    meshes: Vec::new(),
                    wheels: Default::default(),
                    centres: [Vec3::ZERO; 4],
                    axles: [Vec3::X; 4],
                    radius: 0.33,
                    offset: Vec3::ZERO,
                    cars: vec![i],
                    skel: None,
                    cockpit: None,
                    lods: Vec::new(),
                }),
            }
        }

        let mut kept: Vec<Slot> = Vec::new();
        let mut first_fx = None;
        // big grids: the cars' textures a mip or two down. A 75-car Ring
        // replay of ~10 models took 9 GB at full size on a 12 GB card, it
        // paged, and every pass ran 15-30x slow (4 fps; TAA alone 11 ms).
        // EVOVIEW_REPLAY_TEX_SKIP=n overrides.
        let skip = crate::envc::var("EVOVIEW_REPLAY_TEX_SKIP").ok().and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(match slots.len() { 0..=2 => 0, 3..=6 => 1, _ => 2 });
        if skip > 0 {
            println!("replay: {} car models, their textures {skip} mip(s) down", slots.len());
        }
        crate::load::CAR_TEX_SKIP.store(skip, std::sync::atomic::Ordering::Relaxed);
        crate::load::set_car_lods(true);
        for (k, mut s) in slots.into_iter().enumerate() {
            // a mod car is in its own package (Saved Games/ACE/mods/<model>
            // .kspkg) over the base game, as the race path (below) opens it
            let loaded = crate::load::load_from_package(pkg, &s.model, base).or_else(|e| {
                let m = crate::envc::var_os("USERPROFILE").map(|h| std::path::Path::new(&h).join("Saved Games")
                    .join("ACE").join("mods").join(format!("{}.kspkg", s.model)));
                match m.filter(|m| m.exists()) {
                    Some(m) => {
                        println!("car {}: the mod {}", s.model, m.display());
                        crate::load::load_from_package(&m, &s.model, Some(pkg))
                    }
                    None => Err(e),
                }
            });
            let loaded = match loaded {
                Ok(l) => l,
                Err(e) => {
                    // One missing car must not sink the whole replay.
                    eprintln!("  skipping {}: {e}", s.model);
                    continue;
                }
            };
            if first_fx.is_none() {
                // ⚠ the effects' texture indices are into THIS car's table,
                // which merge appends after the track's: unshifted, the smoke
                // sampled asphalt and line paint (dark square shards)
                let base = textures.len();
                first_fx = loaded.car.fx.clone().map(|fx| shift_fx_textures(fx, base));
            }
            let tag = format!("c{k}");
            let range = merge(&mut meshes, &mut textures, &mut tex_src, &mut materials, &mut blob,
                &loaded.car, Some(&tag));
            // One instance per driver using this model. build_scene sizes the
            // instance buffer from `xforms`, so seeding them here is what makes
            // room for the per-frame rewrite; the values are replaced on the
            // first frame.
            let ident = Mat4::IDENTITY.to_cols_array();
            collect_lods(&meshes, range.clone(), &mut s.lods);
            for i in range.clone() {
                meshes[i].xforms = vec![ident; s.cars.len()];
                if crate::load::car_lod_of(&meshes[i].name).is_some() {
                    continue;
                }
                match corner_index(&meshes[i].name) {
                    Some(k) => {
                        // Wheel meshes are authored AT the wheel bone, so their
                        // bounds give both the centre to rotate about and the
                        // rolling radius - no skeleton lookup needed here.
                        //
                        // ⚠ Measure the SOURCE mesh, not the merged one: the
                        // merged copy's blob ranges have already been shifted
                        // to index the combined blob, so reading them against
                        // the source car's blob returns garbage bounds and the
                        // wheels fly off the car.
                        let (lo, hi) =
                            mesh_bounds(&loaded.car, &loaded.car.meshes[i - range.start]);
                        s.centres[k] = (lo + hi) * 0.5;
                        if meshes[i].name.starts_with("tyre_") {
                            s.radius = s.radius.max((hi.y - lo.y) * 0.5);
                            s.axles[k] = tyre_axle(&loaded.car, &loaded.car.meshes[i - range.start]);
                        }
                        s.wheels[k].push(i);
                    }
                    None => s.meshes.push(i),
                }
            }
            // Line the body up with its own wheels: average, over the four
            // corners, of where the replay puts the wheel minus where the mesh
            // has it. Measured per model rather than hard-coded, because the
            // gap between mesh origin and centre of mass differs per car.
            {
                let p = rep.pose(s.cars[0], 0);
                let basis = car_basis(&p).transpose();
                let mut d = Vec3::ZERO;
                for k in 0..4 {
                    let local = basis.transform_vector3(p.wheels[k] - p.pos);
                    d += local - s.centres[k];
                    if crate::envc::var("EVOVIEW_REPLAY_LOG").is_ok() {
                        eprintln!(
                            "{} wheel{k}: mesh ({:6.3},{:6.3},{:6.3})  replay ({:6.3},{:6.3},{:6.3})  r={:.3}",
                            s.model, s.centres[k].x, s.centres[k].y, s.centres[k].z,
                            local.x, local.y, local.z, s.radius);
                    }
                }
                s.offset = d / 4.0;
            }
            if s.wheels.iter().any(|w| w.is_empty()) {
                eprintln!("  {}: only {} wheel corner(s) found; they will not turn",
                          s.model, s.wheels.iter().filter(|w| !w.is_empty()).count());
            }
            s.cockpit = loaded.car.fx.as_ref().and_then(|f| {
                let c = &f.actor.cameras;
                c.iter().find(|c| c.name.eq_ignore_ascii_case("cockpit")).or_else(|| c.iter().find(|c| c.mode == 1)).cloned()
            });
            // EVOVIEW_REPLAY_RIG=0: the old rigid bodies and wheels
            if loaded.skeleton.rig.is_some() && crate::envc::var("EVOVIEW_REPLAY_RIG").map_or(true, |v| v != "0") {
                s.skel = Some(std::sync::Arc::new(loaded.skeleton));
            }
            kept.push(s);
        }
        crate::load::CAR_TEX_SKIP.store(0, std::sync::atomic::Ordering::Relaxed);
        crate::load::set_car_lods(false);
        if kept.is_empty() {
            return Err("none of the replay's cars could be loaded".into());
        }
        let mut car = Car::from_parts(meshes, textures, materials, blob);
        car.tex_src = tex_src;
        car.track_env = track_env;
        // The first model's effects serve every car (particles.rs emits per
        // car in world space): the particle packs are the shared common_assets
        // ones, only the tyre / exhaust positions are that car's. ⚠ Mixed
        // grids had no effects at all before - no smoke, dust or backfires.
        car.fx = first_fx;
        Ok(Scene {
            car,
            slots: kept,
            track_meshes,
            skeleton: None,
            animations: Vec::new(),
        })
    }

    /// Bake every car's transform at time `t` into the mesh instance data.
    ///
    /// The live path overwrites these on the GPU each frame, but the headless
    /// screenshot path renders whatever `xforms` held at upload time - so this
    /// is what lets a still be taken at a chosen moment of the replay.
    pub fn pose_at(&mut self, rep: &Replay, w: &Wheels, t: f32) {
        for s in &self.slots {
            let mut body = Vec::with_capacity(s.cars.len());
            let mut wheels: [Vec<[f32; 16]>; 4] = Default::default();
            for &c in &s.cars {
                match car_pose(rep, w, s, c, t) {
                    Some((b, wm)) => {
                        body.push(b.to_cols_array());
                        for k in 0..4 {
                            wheels[k].push(wm[k].to_cols_array());
                        }
                    }
                    None => {
                        body.push([0.0; 16]);
                        for k in 0..4 {
                            wheels[k].push([0.0; 16]);
                        }
                    }
                }
            }
            for &mi in &s.meshes {
                self.car.meshes[mi].xforms = body.clone();
            }
            for k in 0..4 {
                for &mi in &s.wheels[k] {
                    self.car.meshes[mi].xforms = wheels[k].clone();
                }
            }
        }
    }

    /// Where the action is at t=0, so the camera can open looking at the grid
    /// rather than at the middle of the track.
    pub fn start_focus(&self, rep: &Replay) -> Option<Vec3> {
        let mut sum = Vec3::ZERO;
        let mut n = 0.0f32;
        for s in &self.slots {
            for &c in &s.cars {
                sum += rep.pose(c, 0).pos;
                n += 1.0;
            }
        }
        (n > 0.0).then(|| sum / n)
    }
}

impl Scene {
    /// A track and ONE car to drive on it (trackdrive.rs): the car's meshes
    /// get a single instance each, rewritten every frame from the physics.
    /// Wheel centres and rolling radius are measured from the wheel meshes;
    /// `offset` lifts the mesh so its wheel centres sit one radius above the
    /// ground point the physics places the car at.
    pub fn build_single(pkg: &Path, base: Option<&Path>, track: &str, model: &str)
            -> Result<(Scene, crate::load::Vfs), String> {
        Self::build_race(pkg, base, track, model, &[], &[]).map(|(s, v, _, _)| (s, v))
    }

    /// build_single plus the AI cars (race.rs), one model per entry of `ai`.
    /// Each distinct LOOK (model + livery + body colour, livery::ai_looks)
    /// is one more slot, its meshes with one instance per car wearing it.
    /// Returns each distinct model's vfs (its physics) and per AI car its
    /// (model index into that list, slot, instance). The AI copies are
    /// loaded apart from the player's car even for the same model: their
    /// skinning is dropped (the one bone buffer is the player's rig - a
    /// second car would wear its steering, wheel spin and needles, a
    /// different model garbage), so they draw in their bind pose with
    /// per-wheel matrices, as a replay's cars do. Textures are shared
    /// through the GPU cache (tex_src): a second livery of a model uploads
    /// only the maps that differ.
    /// EVOVIEW_AI_LIVERIES=n: at most n looks per model (default 6; 1 = every
    /// AI of a model in its first livery other than the player's, 0 = the
    /// default livery, as before).
    #[allow(clippy::type_complexity)]
    /// `own`: per opponent, a CarFinalStateData to wear (online: each
    /// player's own design and paints, mp.rs) instead of a picked look.
    pub fn build_race(pkg: &Path, base: Option<&Path>, track: &str, model: &str, ai: &[String], own: &[Option<Vec<u8>>])
            -> Result<(Scene, crate::load::Vfs, Vec<(String, crate::load::Vfs)>, Vec<(usize, usize, usize)>), String> {
        let mut t = crate::load::load_track_any(pkg, track, base)
            .map_err(|e| format!("track {track}: {e}"))?;
        let track_env = t.car.track_env.take();
        let mut meshes = Vec::new();
        let mut textures = Vec::new();
        let mut materials = Vec::new();
        let mut blob: Vec<u8> = Vec::new();
        let mut tex_src: Vec<String> = Vec::new();
        merge(&mut meshes, &mut textures, &mut tex_src, &mut materials, &mut blob, &t.car, None);
        let track_meshes = meshes.len();
        // a mod car (Saved Games/ACE/mods/<model>.kspkg) on the base game,
        // when the package does not carry it
        let open = |model: &str| -> Result<crate::load::Vfs, String> {
            match crate::load::package_vfs(pkg, model, base) {
                Ok(v) => Ok(v),
                Err(e) => {
                    let m = crate::envc::var_os("USERPROFILE").map(|h| std::path::Path::new(&h).join("Saved Games").join("ACE").join("mods").join(format!("{model}.kspkg")));
                    match m.filter(|m| m.exists()) {
                        Some(m) => {
                            println!("car {model}: the mod {}", m.display());
                            crate::load::package_vfs(&m, model, Some(pkg))
                        }
                        None => Err(e),
                    }
                }
            }
        };
        let mut vfs = open(model)?;
        let loaded = crate::load::load(&mut vfs).map_err(|e| format!("car {model}: {e}"))?;
        let car_tex_base = textures.len();
        let range = merge(&mut meshes, &mut textures, &mut tex_src, &mut materials, &mut blob, &loaded.car, Some("c0"));
        let s = measure_slot(&mut meshes, range, &loaded.car, model, 1);
        let mut slots = vec![s];
        let player_look = crate::load::pick_visual_preset(&mut vfs).map(|k| crate::livery::stem_of(&k));
        let max_looks = crate::envc::var("EVOVIEW_AI_LIVERIES").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(6);
        let mut ai_vfs: Vec<(String, crate::load::Vfs)> = Vec::new();
        let mut place = vec![(0usize, 0usize, 0usize); ai.len()];
        // the env the loader reads a look from, put back after
        let saved = (crate::envc::var("EVOVIEW_VISUAL").ok(), crate::envc::var("EVOVIEW_PAINT").ok());
        let restore = |k: &str, v: &Option<String>| match v {
            Some(v) => crate::envc::set_var(k, v),
            None => crate::envc::remove_var(k),
        };
        let t_ai = std::time::Instant::now();
        // the opponents' textures EVOVIEW_AI_TEX_SKIP mips down: 1 (a quarter
        // of the memory), 2 online (a whole server of liveries)
        let ai_skip = crate::envc::var("EVOVIEW_AI_TEX_SKIP").ok().and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(if own.iter().any(|o| o.is_some()) { 2 } else { 1 });
        crate::load::CAR_TEX_SKIP.store(ai_skip, std::sync::atomic::Ordering::Relaxed);
        crate::load::set_car_lods(true);
        let mut models: Vec<&String> = Vec::new();
        for m in ai {
            if !models.contains(&m) {
                models.push(m);
            }
        }
        for (mi, m) in models.iter().enumerate() {
            let cars: Vec<usize> = (0..ai.len()).filter(|&i| &ai[i] == *m).collect();
            let mut v = open(m)?;
            let list = crate::livery::list(&mut v);
            let looks: Vec<Option<crate::livery::Look>> = if max_looks == 0 {
                vec![None; cars.len()]
            } else {
                let paints = crate::load::body_paints(&mut v);
                let mine = if *m == model { player_look.as_deref() } else { None };
                // each livery's own paints (env-selected, as the loader reads them)
                let own: Vec<(String, Vec<String>)> = list.iter().map(|l| {
                    crate::envc::set_var("EVOVIEW_VISUAL", &l.stem);
                    let p = crate::load::preset_paints(&mut v);
                    (l.stem.clone(), p)
                }).collect();
                restore("EVOVIEW_VISUAL", &saved.0);
                let own_of = |s: &str| own.iter().find(|o| o.0 == s).map_or(Vec::new(), |o| o.1.clone());
                println!("AI model {m}: {} livery(s), {} body colour(s)", list.len(), paints.len());
                if crate::envc::var_os("EVOVIEW_AI_LOOK_LOG").is_some() {
                    for ext in ["design", "cardesign", "visualcarpreset", "oemmultilayercolor"] {
                        for d in ["skins", "presets", "", "skins/designs"] {
                            for k in v.list(d, ext) {
                                println!("  look file {k}");
                            }
                        }
                    }
                    println!("  body paints {paints:?}");
                }
                let l = crate::livery::ai_looks(&list, mine, &paints, &own_of, cars.len(), max_looks, mi * 3);
                if l.is_empty() { vec![None; cars.len()] } else { l.into_iter().map(Some).collect() }
            };
            // online: a player's own look, as a visual preset made from their
            // final state (named by a hash of it, so equal looks share a slot)
            let looks: Vec<Option<crate::livery::Look>> = cars.iter().zip(looks).map(|(&c, picked)| {
                let Some(fs) = own.get(c).and_then(|o| o.as_ref()) else { return picked };
                let Some(bytes) = crate::load::netcar_visual(&mut v, fs) else { return picked };
                let mut h: u64 = 0xcbf29ce484222325;
                for b in &bytes {
                    h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
                }
                let stem = format!("mp_{h:016x}");
                v.add_file(&format!("presets/{stem}.visualcarpreset"), bytes);
                Some(crate::livery::Look { visual: stem, paint: None })
            }).collect();
            let mut distinct: Vec<Option<crate::livery::Look>> = Vec::new();
            for l in &looks {
                if !distinct.contains(l) {
                    distinct.push(l.clone());
                }
            }
            if distinct.len() > 1 {
                // the decoded textures stay for the next look's load
                v.keep_textures();
            }
            for look in &distinct {
                if let Some(l) = look {
                    crate::envc::set_var("EVOVIEW_VISUAL", &l.visual);
                    match &l.paint {
                        Some(p) => crate::envc::set_var("EVOVIEW_PAINT", p),
                        None => restore("EVOVIEW_PAINT", &saved.1),
                    }
                }
                let r = crate::load::load(&mut v);
                restore("EVOVIEW_VISUAL", &saved.0);
                restore("EVOVIEW_PAINT", &saved.1);
                let l = r.map_err(|e| format!("AI car {m}: {e}"))?;
                let si = slots.len();
                let mine: Vec<usize> = cars.iter().zip(&looks).filter(|(_, x)| *x == look).map(|(&c, _)| c).collect();
                let tag = format!("a{}", si - 1);
                let range = if crate::envc::var("EVOVIEW_AI_SHARE_TEX").map_or(true, |v| v != "0") {
                    merge_shared(&mut meshes, &mut textures, &mut tex_src, &mut materials, &mut blob, &l.car, &tag)
                } else {
                    let r = merge(&mut meshes, &mut textures, &mut tex_src, &mut materials, &mut blob, &l.car, Some(&tag));
                    for i in r.clone() {
                        meshes[i].bone_idx = None;
                        meshes[i].bone_wgt = None;
                    }
                    r
                };
                slots.push(measure_slot(&mut meshes, range, &l.car, m, mine.len()));
                for (inst, &c) in mine.iter().enumerate() {
                    place[c] = (mi, si, inst);
                }
                println!("AI look {}: {m} {} {} x{}", si - 1,
                         look.as_ref().map_or("(default)", |l| l.visual.as_str()),
                         look.as_ref().and_then(|l| l.paint.as_deref()).unwrap_or("-"), mine.len());
            }
            ai_vfs.push((m.to_string(), v));
        }
        crate::load::CAR_TEX_SKIP.store(0, std::sync::atomic::Ordering::Relaxed);
        crate::load::set_car_lods(false);
        if !ai.is_empty() {
            println!("AI cars: {} model(s), {} look(s), loaded in {:.1}s", models.len(), slots.len() - 1, t_ai.elapsed().as_secs_f32());
        }
        let mut car = Car::from_parts(meshes, textures, materials, blob);
        car.tex_src = tex_src;
        car.track_env = track_env;
        // lights, LEDs, dash: nothing in CarFx indexes meshes, so it carries
        // over; the particle textures index the car's table, now after the track's
        car.fx = loaded.car.fx.map(|fx| shift_fx_textures(fx, car_tex_base));
        car.rain_tex = loaded.car.rain_tex.map(|t| t.map(|i| i + car_tex_base));
        car.rain_streaks = loaded.car.rain_streaks.clone();
        let has_rig = loaded.skeleton.rig.is_some();
        Ok((Scene { car, slots, track_meshes,
                    skeleton: has_rig.then_some(loaded.skeleton), animations: loaded.animations }, vfs, ai_vfs, place))
    }
}

/// Online (mp.rs): a late joiner's car - `model` in the look `own` (their
/// CarFinalStateData), at the opponents' texture size - as a scene of its
/// own (meshes, materials and textures tagged `tag`) and its slot, to be
/// appended to the running one. Runs off the render thread.
pub fn load_net_car(pkg: &Path, base: Option<&Path>, model: &str, own: Option<&[u8]>, tag: &str) -> Result<(Car, Slot), String> {
    let mut v = crate::load::package_vfs(pkg, model, base).map_err(|e| format!("car {model}: {e}"))?;
    let saved = crate::envc::var("EVOVIEW_VISUAL").ok();
    if let Some(fs) = own {
        if let Some(bytes) = crate::load::netcar_visual(&mut v, fs) {
            let stem = format!("mp_{tag}");
            v.add_file(&format!("presets/{stem}.visualcarpreset"), bytes);
            crate::envc::set_var("EVOVIEW_VISUAL", &stem);
        }
    }
    let skip = crate::envc::var("EVOVIEW_AI_TEX_SKIP").ok().and_then(|x| x.parse::<u32>().ok()).unwrap_or(2);
    crate::load::CAR_TEX_SKIP.store(skip, std::sync::atomic::Ordering::Relaxed);
    crate::load::set_car_lods(true);
    let r = crate::load::load(&mut v);
    crate::load::CAR_TEX_SKIP.store(0, std::sync::atomic::Ordering::Relaxed);
    crate::load::set_car_lods(false);
    match saved {
        Some(x) => crate::envc::set_var("EVOVIEW_VISUAL", x),
        None => crate::envc::remove_var("EVOVIEW_VISUAL"),
    }
    let l = r.map_err(|e| format!("car {model}: {e}"))?;
    let (mut meshes, mut textures, mut materials, mut blob, mut tex_src) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let range = merge(&mut meshes, &mut textures, &mut tex_src, &mut materials, &mut blob, &l.car, Some(tag));
    for i in range.clone() {
        meshes[i].bone_idx = None;
        meshes[i].bone_wgt = None;
    }
    let slot = measure_slot(&mut meshes, range, &l.car, model, 1);
    let mut car = Car::from_parts(meshes, textures, materials, blob);
    car.tex_src = tex_src;
    Ok((car, slot))
}

/// One merged car's slot: body / wheel meshes, wheel centres, axles and
/// rolling radius from the wheel meshes, `n` instances per mesh; `offset`
/// lifts the mesh so its wheel centres sit one radius above the ground
/// point the physics places the car at.
fn measure_slot(meshes: &mut [MeshEntry], range: std::ops::Range<usize>, src: &Car, model: &str, n: usize) -> Slot {
    let mut s = Slot {
        model: model.to_string(),
        meshes: Vec::new(),
        wheels: Default::default(),
        centres: [Vec3::ZERO; 4],
        axles: [Vec3::X; 4],
        radius: 0.0,
        offset: Vec3::ZERO,
        cars: (0..n).collect(),
        skel: None,
        cockpit: None,
        lods: Vec::new(),
    };
    let ident = Mat4::IDENTITY.to_cols_array();
    collect_lods(meshes, range.clone(), &mut s.lods);
    for i in range.clone() {
        meshes[i].xforms = vec![ident; n];
        if crate::load::car_lod_of(&meshes[i].name).is_some() {
            continue;
        }
        match corner_index(&meshes[i].name) {
            Some(k) => {
                let (lo, hi) = mesh_bounds(src, &src.meshes[i - range.start]);
                s.centres[k] = (lo + hi) * 0.5;
                if meshes[i].name.starts_with("tyre_") {
                    s.radius = s.radius.max((hi.y - lo.y) * 0.5);
                    s.axles[k] = tyre_axle(src, &src.meshes[i - range.start]);
                }
                s.wheels[k].push(i);
            }
            None => s.meshes.push(i),
        }
    }
    if s.radius <= 0.05 {
        s.radius = 0.33;
    }
    let mean_y = s.centres.iter().map(|c| c.y).sum::<f32>() / 4.0;
    s.offset = Vec3::new(0.0, s.radius - mean_y, 0.0);
    if s.wheels.iter().any(|w| w.is_empty()) {
        eprintln!("  {model}: only {} wheel corner(s) found",
                  s.wheels.iter().filter(|w| !w.is_empty()).count());
    }
    s
}
