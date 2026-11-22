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
    /// build_race: the driving ground of the track's colliders, built on a
    /// thread of its own while the cars load (~4 s on the Nordschleife)
    pub ground: Option<std::sync::Arc<crate::trackdrive::Ground>>,
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
    grow(blob, src.blob().len());
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
    geo: &mut std::collections::HashMap<(String, usize, u64), [u64; 2]>,
) -> std::ops::Range<usize> {
    tex_src.resize(textures.len(), String::new());
    // at most the whole source blob plus alignment padding
    grow(blob, src.blob().len() + 16 * (src.textures.len() + 8 * src.meshes.len()));
    let have: std::collections::HashMap<String, usize> = tex_src.iter().enumerate()
        .filter(|(_, k)| !k.is_empty() && !k.starts_with("merged:"))
        .map(|(i, k)| (k.clone(), i))
        .collect();
    // the copy's range: ⚠ its own length, not the source's - a bogus
    // source range copies nothing, and keeping its length read whatever
    // came next in the blob
    let copy = |blob: &mut Vec<u8>, r: [u64; 2]| -> [u64; 2] {
        while blob.len() % 16 != 0 {
            blob.push(0);
        }
        let off = blob.len() as u64;
        let b = src.slice(Some(r));
        blob.extend_from_slice(b);
        [off, b.len() as u64]
    };
    let mut map = Vec::with_capacity(src.textures.len());
    for (i, t) in src.textures.iter().enumerate() {
        let key = src.tex_src.get(i).filter(|s| !s.is_empty());
        if let Some(&j) = key.and_then(|k| have.get(k)) {
            map.push(j);
            continue;
        }
        let mut t = t.clone();
        t.data = copy(blob, t.data);
        textures.push(t);
        tex_src.push(key.cloned().unwrap_or_else(|| format!("merged:{tag}:{i}")));
        map.push(textures.len() - 1);
    }
    for m in &src.materials {
        let mut m = m.clone();
        // a slot naming no texture of this car is dropped: left unshifted
        // it pointed at some other car's (or the track's) texture
        m.slots.retain(|_, v| map.get(*v).map(|&j| *v = j).is_some());
        m.file = format!("{tag}__{}", base_of(&m.file));
        materials.push(m);
    }
    let start = meshes.len();
    for m in &src.meshes {
        let mut m = m.clone();
        m.bone_idx = None;
        m.bone_wgt = None;
        // Another look of the same model has the same geometry: its bytes
        // are already in the blob (keyed by mesh name, stream and length,
        // and compared before reuse). A copy per look was ~110 MB of load
        // RAM each, uploaded once anyway (render.rs shares geometry).
        for (fi, r) in [&mut m.positions, &mut m.normals, &mut m.uvs, &mut m.tangents, &mut m.indices, &mut m.colors, &mut m.uv1]
            .into_iter().enumerate() {
            if let Some(v) = r.as_mut() {
                let bytes = src.slice(Some(*v));
                let key = (m.name.clone(), fi, bytes.len() as u64);
                match geo.get(&key) {
                    Some(&[off, len]) if blob.get(off as usize..(off + len) as usize) == Some(bytes) => *v = [off, len],
                    _ => {
                        *v = copy(blob, *v);
                        geo.insert(key, *v);
                    }
                }
            }
        }
        for b in &mut m.batches {
            b.material = format!("{tag}__{}", base_of(&b.material));
        }
        meshes.push(m);
    }
    start..meshes.len()
}

/// Room for `n` more bytes in the scene blob. ⚠ Not plain `extend`: its
/// doubling turned a 3 GB track blob plus one car into a 6 GB allocation
/// (and a copy). 25 % headroom keeps the reallocations few without that.
/// EVOVIEW_RAM_LOG=1: the process's memory and the scene blob at a load step.
pub(crate) fn ram_log(what: &str, blob: Option<&Vec<u8>>) {
    if crate::envc::var_os("EVOVIEW_RAM_LOG").is_none() {
        return;
    }
    let (peak, commit, ws) = crate::perf::ram();
    let mb = |b: u64| b as f64 / 1048576.0;
    let (bl, bc) = blob.map_or((0, 0), |b| (b.len() as u64, b.capacity() as u64));
    println!("RAM {what}: ws {:.0} peak {:.0} commit {:.0} MB; blob {:.0} / cap {:.0} MB", mb(ws), mb(peak), mb(commit), mb(bl), mb(bc));
}

fn grow(blob: &mut Vec<u8>, n: usize) {
    if blob.capacity() - blob.len() < n {
        blob.reserve_exact(n.max(blob.len() / 4));
    }
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
        // ⚠ the normal map too (smoke lighting, spray refraction): unshifted
        // it sampled a TRACK texture, and uploaded a linear copy of it
        d.material.normal_tex = d.material.normal_tex.map(|i| i + base);
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
        let mut tex_src: Vec<String> = Vec::new();
        // ⚠ the track's blob is MOVED in (merged at offset 0 with nothing to
        // copy), not copied: the copy and the loaded track both lived to the
        // end of the build, 2x a multi-GB blob at the load peak
        let mut blob: Vec<u8> = Vec::new();
        merge(&mut meshes, &mut textures, &mut tex_src, &mut materials, &mut blob, &t.car, None);
        let mut blob = t.car.take_blob();
        drop(t);
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
                    // measured below (0.33 if no tyre): starting at 0.33 and
                    // only growing, small wheels never got their own radius
                    radius: 0.0,
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
        // The models load on worker threads (EVOVIEW_LOAD_THREADS, 2: a load
        // is half disk, half decode) and are merged here strictly in order,
        // so the scene is the same as loading them one by one.
        let models: Vec<String> = slots.iter().map(|s| s.model.clone()).collect();
        let load_one = |model: &str| -> Result<crate::load::Loaded, String> {
            // a mod car is in its own package (Saved Games/ACE/mods/<model>
            // .kspkg) over the base game, as the race path (below) opens it
            crate::load::set_car_lods(true);
            let r = crate::load::load_from_package(pkg, model, base).or_else(|e| {
                let m = crate::envc::var_os("USERPROFILE").map(|h| std::path::Path::new(&h).join("Saved Games")
                    .join("ACE").join("mods").join(format!("{}.kspkg", model)));
                match m.filter(|m| m.exists()) {
                    Some(m) => {
                        println!("car {}: the mod {}", model, m.display());
                        crate::load::load_from_package(&m, model, Some(pkg))
                    }
                    None => Err(e),
                }
            });
            crate::load::set_car_lods(false);
            r
        };
        let threads = crate::envc::var("EVOVIEW_LOAD_THREADS").ok().and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2).clamp(1, 4).min(models.len().max(1));
        let next = std::sync::atomic::AtomicUsize::new(0);
        // ⚠ back-pressure: the merge below drains the channel eagerly, so the
        // channel bound alone let the other workers load EVERY later model
        // while the wanted one was still loading - all held decoded at once.
        // A worker starts model i only once i < merged + threads.
        let gate = (std::sync::Mutex::new(0usize), std::sync::Condvar::new());
        let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Result<crate::load::Loaded, String>)>(threads);
        std::thread::scope(|sc| {
            for _ in 0..threads {
                let tx = tx.clone();
                let (models, load_one, next, gate) = (&models, &load_one, &next, &gate);
                sc.spawn(move || loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(m) = models.get(i) else { break };
                    {
                        let mut w = gate.0.lock().unwrap_or_else(|e| e.into_inner());
                        while i >= *w + threads {
                            w = gate.1.wait(w).unwrap_or_else(|e| e.into_inner());
                        }
                    }
                    if tx.send((i, load_one(m))).is_err() {
                        break;
                    }
                });
            }
            drop(tx);
            let mut want = 0usize;
            let mut slots = slots.into_iter().enumerate();
            let mut pending: std::collections::BTreeMap<usize, Result<crate::load::Loaded, String>> = Default::default();
            for (i, r) in rx {
                pending.insert(i, r);
                while let Some(loaded) = pending.remove(&want) {
                    want += 1;
                    *gate.0.lock().unwrap_or_else(|e| e.into_inner()) = want;
                    gate.1.notify_all();
                    let Some((k, mut s)) = slots.next() else { break };
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
                    if s.radius <= 0.05 {
                        s.radius = 0.33;
                    }
                    {
                        // ⚠ the first frame the car is really there: one that
                        // joined after the recording started sits at the
                        // origin with a zero basis at frame 0, which made the
                        // offset minus the mean wheel centre for the model
                        let p = s.cars.iter()
                            .flat_map(|&c| (0..rep.cars[c].n_matrix as usize).map(move |f| (c, f)))
                            .map(|(c, f)| rep.pose(c, f))
                            .find(|p| p.pos.length_squared() >= 1.0 && p.forward.length_squared() > 0.5)
                            .unwrap_or_else(|| rep.pose(s.cars[0], 0));
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
            }
        });
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
            ground: None,
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
        // The cars (the player's, then the opponents') load on a thread of
        // their own while this one loads the track: neither reads the
        // other's data, and the loader globals the cars set (CAR_TEX_SKIP,
        // the look env, the car LODs) are read by the car loader only.
        // EVOVIEW_LOAD_SERIAL=1: one after the other, as before.
        let serial = crate::envc::var_os("EVOVIEW_LOAD_SERIAL").is_some();
        std::thread::scope(|sc| {
            let job = if serial {
                None
            } else {
                std::thread::Builder::new().name("car load".into())
                    .spawn_scoped(sc, || race_cars(pkg, base, model, ai, own)).ok()
            };
            let mut t = crate::load::load_track_any(pkg, track, base).map_err(|e| format!("track {track}: {e}"));
            crate::perf::stamp("track loaded");
            // said now: the error is returned once the car thread is done
            // (the scope waits for it), 15 s or more with a full grid
            if let Err(e) = &t {
                eprintln!("could not load: {e} (finishing the car load first)");
            }
            // the driving ground (one more thread) while the cars finish:
            // it needs only the colliders; the env comes back with it
            let ground_job = t.as_mut().ok().and_then(|t| t.car.track_env.take()).map(|e| std::thread::spawn(move || {
                let t0 = std::time::Instant::now();
                let g = e.collision.as_ref().map(|c| std::sync::Arc::new(crate::trackdrive::Ground::new(c, &e.surface_defs)));
                crate::perf::add("ground (thread)", t0.elapsed().as_secs_f64());
                (e, g)
            }));
            let cars = match job {
                Some(j) => j.join().map_err(|_| "the car loader crashed".to_string())?,
                None => race_cars(pkg, base, model, ai, own),
            };
            crate::perf::stamp("cars loaded");
            // (the cars failed: the ground thread is joined, not left
            // detached holding the track's colliders)
            let cars = match cars {
                Ok(c) => c,
                Err(e) => {
                    if let Some(g) = ground_job {
                        let _ = g.join();
                    }
                    return Err(t.err().unwrap_or(e));
                }
            };
            Self::race_scene(t?, cars, ground_job, model, track)
        })
    }

    /// build_race's second half: the track, the player's car and the
    /// opponents (merged among themselves by `race_cars`) into one scene,
    /// the same as merging them one after the other.
    #[allow(clippy::type_complexity)]
    fn race_scene(mut t: crate::load::Loaded, cars: RaceCars, ground_job: Option<GroundJob>, model: &str, track: &str)
            -> Result<(Scene, crate::load::Vfs, Vec<(String, crate::load::Vfs)>, Vec<(usize, usize, usize)>), String> {
        let RaceCars { vfs, mut loaded, ai_vfs, place, ai_meshes, ai_textures, ai_tex_src, ai_materials, ai_blob, ai_slots, share } = cars;
        let mut meshes = Vec::new();
        let mut textures = Vec::new();
        let mut materials = Vec::new();
        let mut tex_src: Vec<String> = Vec::new();
        // the track's blob moved in, not copied (see Scene::build)
        let mut blob: Vec<u8> = Vec::new();
        merge(&mut meshes, &mut textures, &mut tex_src, &mut materials, &mut blob, &t.car, None);
        let mut blob = t.car.take_blob();
        drop(t);
        let track_meshes = meshes.len();
        let car_tex_base = textures.len();
        // room for the player's car and the opponents at once: one move of
        // the GBs of track blob, not one per append
        grow(&mut blob, loaded.car.blob().len() + ai_blob.len() + 64);
        let range = merge(&mut meshes, &mut textures, &mut tex_src, &mut materials, &mut blob, &loaded.car, Some("c0"));
        let s = measure_slot(&mut meshes, range, &loaded.car, model, 1);
        // merged and measured: its blob is a dead copy from here on (only
        // fx / rain / skeleton are taken from `loaded` at the end)
        drop(loaded.car.take_blob());
        ram_log("track + player car", Some(&blob));
        let mut slots = vec![s];
        let (ai_tex_base, ai_mesh_base) = (textures.len(), meshes.len());
        // the opponents: a texture whose content the track or the player's
        // car already has points at that copy (as merge_shared, merging
        // after them, did); the rest, the meshes and the slots shift past
        // the track's and the player's
        let have: std::collections::HashMap<&str, usize> = tex_src.iter().enumerate()
            .filter(|(_, k)| share && !k.is_empty() && !k.starts_with("merged:"))
            .map(|(i, k)| (k.as_str(), i))
            .collect();
        let mut to = Vec::with_capacity(ai_textures.len());
        let mut kept = 0usize;
        for k in &ai_tex_src {
            match have.get(k.as_str()) {
                Some(&j) => to.push(j),
                None => {
                    to.push(ai_tex_base + kept);
                    kept += 1;
                }
            }
        }
        drop(have);
        grow(&mut blob, ai_blob.len() + 16);
        while blob.len() % 16 != 0 {
            blob.push(0);
        }
        let off = blob.len() as u64;
        blob.extend_from_slice(&ai_blob);
        drop(ai_blob);
        tex_src.resize(textures.len(), String::new());
        for (i, (mut t, k)) in ai_textures.into_iter().zip(ai_tex_src).enumerate() {
            if to[i] >= ai_tex_base {
                t.data[0] += off;
                textures.push(t);
                tex_src.push(k);
            }
        }
        for mut m in ai_materials {
            // (a slot past its car's textures - possible with
            // EVOVIEW_AI_SHARE_TEX=0's plain merge - is dropped, not a panic)
            m.slots.retain(|_, v| match to.get(*v) {
                Some(&j) => {
                    *v = j;
                    true
                }
                None => false,
            });
            materials.push(m);
        }
        for mut m in ai_meshes {
            shift_mesh(&mut m, off);
            meshes.push(m);
        }
        let sh = |v: &mut Vec<usize>| v.iter_mut().for_each(|i| *i += ai_mesh_base);
        for mut s in ai_slots {
            sh(&mut s.meshes);
            s.wheels.iter_mut().for_each(sh);
            s.lods.iter_mut().for_each(|l| {
                l.0 += ai_mesh_base;
                l.2 += ai_mesh_base;
            });
            slots.push(s);
        }
        // the other cars' textures further down when the session would not
        // fit in the VRAM free (vram.rs)
        {
            let base = crate::vram::tex_bytes(&textures[..ai_tex_base], 0) + crate::vram::mesh_bytes(&meshes[..track_meshes]);
            // an extra look of a model shares its geometry (merge_shared): once
            let opp_mesh = crate::vram::mesh_bytes_shared(&meshes[ai_mesh_base..]);
            crate::vram::plan(base, &mut textures[ai_tex_base..], &mut blob, opp_mesh, slots.len() - 1);
        }
        ram_log("opponents planned", Some(&blob));
        crate::perf::stamp("opponents loaded + planned");
        let (track_env, ground) = match ground_job.map(|j| j.join()) {
            Some(Ok((e, g))) => (Some(e), g),
            Some(Err(_)) => return Err(format!("track {track}: building the ground failed")),
            None => (None, None),
        };
        crate::perf::stamp("ground joined");
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
                    skeleton: has_rig.then_some(loaded.skeleton), animations: loaded.animations, ground }, vfs, ai_vfs, place))
    }
}

/// build_race's cars, loaded beside the track: the player's (not merged
/// yet) and the opponents, merged among themselves (`ai_*`: blob offsets
/// into `ai_blob`, mesh indices into `ai_meshes`, texture indices into
/// `ai_textures`).
/// build_race's ground thread: the track's env (back) and its ground.
type GroundJob = std::thread::JoinHandle<(Box<crate::track::TrackEnv>, Option<std::sync::Arc<crate::trackdrive::Ground>>)>;

struct RaceCars {
    vfs: crate::load::Vfs,
    loaded: crate::load::Loaded,
    ai_vfs: Vec<(String, crate::load::Vfs)>,
    place: Vec<(usize, usize, usize)>,
    ai_meshes: Vec<MeshEntry>,
    ai_textures: Vec<crate::asset::TextureEntry>,
    ai_tex_src: Vec<String>,
    ai_materials: Vec<crate::asset::MaterialEntry>,
    ai_blob: Vec<u8>,
    ai_slots: Vec<Slot>,
    /// EVOVIEW_AI_SHARE_TEX: textures shared by content key
    share: bool,
}

fn race_cars(pkg: &Path, base: Option<&Path>, model: &str, ai: &[String], own: &[Option<Vec<u8>>]) -> Result<RaceCars, String> {
    let open = |model: &str| open_car(pkg, model, base);
    let mut vfs = open(model)?;
    let loaded = crate::load::load(&mut vfs).map_err(|e| format!("car {model}: {e}"))?;
    crate::perf::stamp("player car loaded");
    let (mut meshes, mut textures, mut materials, mut tex_src, mut blob) = (Vec::new(), Vec::new(), Vec::new(), Vec::<String>::new(), Vec::new());
    let mut slots: Vec<Slot> = Vec::new();
    let share = crate::envc::var("EVOVIEW_AI_SHARE_TEX").map_or(true, |v| v != "0");
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
    // put the loader's globals back on every way out, an early `?` included
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            crate::load::CAR_TEX_SKIP.store(0, std::sync::atomic::Ordering::Relaxed);
            crate::load::set_car_lods(false);
        }
    }
    let _restore = Restore;
    let mut models: Vec<&String> = Vec::new();
    for m in ai {
        if !models.contains(&m) {
            models.push(m);
        }
    }
    for (mi, m) in models.iter().enumerate() {
        // this model's geometry already in the blob (merge_shared)
        let mut geo = std::collections::HashMap::new();
        let cars: Vec<usize> = (0..ai.len()).filter(|&i| &ai[i] == *m).collect();
        // ⚠ one opponent model that cannot be opened (a stub or
        // unreleased car id on a server) must not end the session: those
        // cars are drawn (and driven) as the player's model instead
        let (mut v, m) = match open(m) {
            Ok(v) => (v, m.as_str()),
            Err(e) => {
                eprintln!("  AI car {m}: {e}; using {model} instead");
                (open(model)?, model)
            }
        };
        let list = crate::livery::list(&mut v);
        let looks: Vec<Option<crate::livery::Look>> = if max_looks == 0 {
            vec![None; cars.len()]
        } else {
            let paints = crate::load::body_paints(&mut v);
            let mine = if m == model { player_look.as_deref() } else { None };
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
            // a look that fails: the model's default look, then the
            // player's model - never the whole session
            let r = r.or_else(|e| {
                eprintln!("  AI car {m}: {e}; trying its default look");
                crate::load::load(&mut v)
            }).or_else(|e| {
                eprintln!("  AI car {m}: {e}; drawn as {model}");
                open(model).and_then(|mut pv| crate::load::load(&mut pv))
            });
            let l = r.map_err(|e| format!("AI car {m}: {e}"))?;
            let si = slots.len() + 1;
            let mine: Vec<usize> = cars.iter().zip(&looks).filter(|(_, x)| *x == look).map(|(&c, _)| c).collect();
            let tag = format!("a{}", si - 1);
            let range = if crate::envc::var("EVOVIEW_AI_SHARE_TEX").map_or(true, |v| v != "0") {
                merge_shared(&mut meshes, &mut textures, &mut tex_src, &mut materials, &mut blob, &l.car, &tag, &mut geo)
            } else {
                let r = merge(&mut meshes, &mut textures, &mut tex_src, &mut materials, &mut blob, &l.car, Some(&tag));
                for i in r.clone() {
                    meshes[i].bone_idx = None;
                    meshes[i].bone_wgt = None;
                }
                r
            };
            slots.push(measure_slot(&mut meshes, range, &l.car, m, mine.len()));
            ram_log(&format!("look {} {m}", si - 1), Some(&blob));
            for (inst, &c) in mine.iter().enumerate() {
                place[c] = (mi, si, inst);
            }
            println!("AI look {}: {m} {} {} x{}", si - 1,
                     look.as_ref().map_or("(default)", |l| l.visual.as_str()),
                     look.as_ref().and_then(|l| l.paint.as_deref()).unwrap_or("-"), mine.len());
        }
        // ⚠ the look caches (every decoded texture and file read) were
        // only for this model's next look: kept, every model's stayed
        // resident while all later models loaded and merged - only the
        // physics is read from this vfs from here on
        v.drop_kept();
        ai_vfs.push((m.to_string(), v));
    }
    drop(_restore);
    if !ai.is_empty() {
        println!("AI cars: {} model(s), {} look(s), loaded in {:.1}s", models.len(), slots.len(), t_ai.elapsed().as_secs_f32());
    }

    Ok(RaceCars { vfs, loaded, ai_vfs, place, ai_meshes: meshes, ai_textures: textures, ai_tex_src: tex_src,
                  ai_materials: materials, ai_blob: blob, ai_slots: slots, share })
}

/// A car's vfs: from the package, else a mod car (Saved Games/ACE/mods/
/// <model>.kspkg) on the base game.
fn open_car(pkg: &Path, model: &str, base: Option<&Path>) -> Result<crate::load::Vfs, String> {
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
}

/// Online (mp.rs): a late joiner's car - `model` in the look `own` (their
/// CarFinalStateData), at the opponents' texture size - as a scene of its
/// own (meshes, materials and textures tagged `tag`) and its slot, to be
/// appended to the running one. Runs off the render thread.
pub fn load_net_car(pkg: &Path, base: Option<&Path>, model: &str, own: Option<&[u8]>, tag: &str) -> Result<(Car, Slot), String> {
    // ⚠ with the mod fallback, as build_race opens cars: without it a player
    // joining after us in a mod car stayed on a borrowed spare model
    let mut v = open_car(pkg, model, base).map_err(|e| format!("car {model}: {e}"))?;
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
    crate::vram::late_drop(&mut textures, &mut blob, crate::vram::mesh_bytes(&meshes));
    let mut car = Car::from_parts(meshes, textures, materials, blob);
    car.tex_src = tex_src;
    car.appended = true;
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
