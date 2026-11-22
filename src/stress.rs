//! EVOVIEW_STRESS: scale a loaded scene up until something breaks or gets
//! slow (limit finding). Unset, nothing here runs.
//!
//! EVOVIEW_STRESS=tiles=9,copies=2,tex=400,texdrop=2,pitch=8000,cars=200
//! - `tiles=N`: the track N times over, on a square spiral around the real
//!   one, `pitch` metres apart (default: the track's own extent + 200 m).
//!   Tiles are extra placements of the track's own meshes (same geometry,
//!   materials and textures): instance counts, the culler, the instance
//!   buffers and f32 precision far from the origin.
//! - `copies=C`: the first C of those tiles are deep copies instead - their
//!   own mesh entries (own vertex / index / instance buffers, own draws) and
//!   their own materials (own bind groups): draw calls, buffers, VRAM.
//! - `tex=K`: the copies' materials bind up to K copies of the track's
//!   textures (uploaded again under their own cache key): texture count and
//!   VRAM. `texdrop=k` drops the top k mips of each copy (count without the
//!   memory).
//! - `cars=N`: N opponents (`--ai` stops at 24), over EVOVIEW_AI_CARS and
//!   EVOVIEW_AI_LIVERIES as usual.
//! - `join=N,join_at=s,join_gap=s`: an offline stand-in online session
//!   (mp.rs `Mp::fake`) in which N players join while driving, the first
//!   at `join_at` s (default 3) after the first frame, then every
//!   `join_gap` s (default 2.5): the late-join load and its frame hitch,
//!   repeatable. Their models: EVOVIEW_JOIN_CARS, else EVOVIEW_AI_CARS,
//!   each model's presets in turn. The opponents become net cars too.
//!   `join_swap=s`: the first joiner changes to the second's car s seconds
//!   after the first join (a number re-announced while its car loads).
//! Collision, the racing line, lamps and the irradiance volumes are not
//! tiled: cars drive on the real track in the middle.

use crate::asset::Car;

#[derive(Clone, Copy, Default)]
pub struct Opts {
    pub tiles: usize,
    pub copies: usize,
    /// texture copies allowed in all (usize::MAX: every one a copy binds)
    pub tex: usize,
    pub texdrop: u32,
    /// 0: the track's extent
    pub pitch: f32,
    pub cars: Option<usize>,
    pub join: usize,
    pub join_at: f32,
    pub join_gap: f32,
    pub join_swap: Option<f32>,
}

pub fn opts() -> Option<Opts> {
    static O: std::sync::OnceLock<Option<Opts>> = std::sync::OnceLock::new();
    *O.get_or_init(|| {
        let v = crate::envc::var("EVOVIEW_STRESS").ok().filter(|v| !v.trim().is_empty())?;
        let mut o = Opts { tiles: 1, join_at: 3.0, join_gap: 2.5, ..Default::default() };
        for kv in v.split(',') {
            let (k, x) = kv.split_once('=').unwrap_or((kv, ""));
            let n = x.trim().parse::<f64>().unwrap_or(0.0).max(0.0);
            match k.trim() {
                "tiles" => o.tiles = (n as usize).max(1),
                "copies" => o.copies = n as usize,
                "tex" => o.tex = n as usize,
                "texdrop" => o.texdrop = n as u32,
                "pitch" => o.pitch = n as f32,
                "cars" => o.cars = Some(n as usize),
                "join" => o.join = n as usize,
                "join_at" => o.join_at = n as f32,
                "join_gap" => o.join_gap = n as f32,
                "join_swap" => o.join_swap = Some(n as f32),
                k => eprintln!("EVOVIEW_STRESS: unknown key {k:?}"),
            }
        }
        o.tiles = o.tiles.max(o.copies + 1);
        if o.copies > 0 && !v.contains("tex=") {
            o.tex = usize::MAX;
        }
        Some(o)
    })
}

/// `cars=N`: the opponent count, past `--ai`'s 24.
pub fn cars() -> Option<usize> {
    opts().and_then(|o| o.cars)
}

/// `join=N`: (N, first join s, gap s), when asked for.
pub fn joins() -> Option<(usize, f32, f32)> {
    opts().filter(|o| o.join > 0).map(|o| (o.join, o.join_at, o.join_gap))
}

/// `join_swap=s`: the first joiner changes car s seconds after the first join.
pub fn join_swap() -> Option<f32> {
    opts().and_then(|o| o.join_swap)
}

/// The stress joins' models: EVOVIEW_JOIN_CARS, else EVOVIEW_AI_CARS.
pub fn join_models() -> Vec<String> {
    let v = crate::envc::var("EVOVIEW_JOIN_CARS").or_else(|_| crate::envc::var("EVOVIEW_AI_CARS")).unwrap_or_default();
    v.split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect()
}

/// Tile k's place on the spiral, in pitches: (0, 0), then ring after ring.
fn spiral(n: usize) -> Vec<(i32, i32)> {
    let mut v = vec![(0, 0)];
    let mut r = 1i32;
    while v.len() < n {
        for z in -r..=r {
            for x in -r..=r {
                if x.abs().max(z.abs()) == r {
                    v.push((x, z));
                }
            }
        }
        r += 1;
    }
    v.truncate(n);
    v
}

/// The track's larger XZ extent: its colliders', else its placements'.
fn extent(car: &Car) -> f32 {
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    let mut eat = |x: f32, z: f32| {
        lo = [lo[0].min(x), lo[1].min(z)];
        hi = [hi[0].max(x), hi[1].max(z)];
    };
    match car.track_env.as_ref().and_then(|e| e.collision.as_ref()) {
        Some(c) if !c.verts.is_empty() => c.verts.iter().for_each(|v| eat(v[0], v[2])),
        _ => car.meshes.iter().flat_map(|m| m.xforms.iter()).for_each(|x| eat(x[12], x[14])),
    }
    if lo[0] > hi[0] { 5000.0 } else { (hi[0] - lo[0]).max(hi[1] - lo[1]) }
}

/// Apply `tiles` / `copies` / `tex` to a freshly loaded track (before it is
/// merged into a scene: its materials are still all the scene has).
pub fn track(car: &mut Car) {
    let Some(o) = opts().filter(|o| o.tiles > 1) else { return };
    const IDENT: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let pitch = if o.pitch > 0.0 { o.pitch } else { extent(car) + 200.0 };
    let at = spiral(o.tiles);
    let moved = |base: &[[f32; 16]], k: usize, out: &mut Vec<[f32; 16]>| {
        let (dx, dz) = (at[k].0 as f32 * pitch, at[k].1 as f32 * pitch);
        for x in if base.is_empty() { std::slice::from_ref(&IDENT) } else { base } {
            let mut x = *x;
            x[12] += dx;
            x[14] += dz;
            out.push(x);
        }
    };
    let n0 = car.meshes.len();
    let n_mat = car.track_env.as_ref().map_or(car.materials.len(), |e| e.n_materials.min(car.materials.len()));
    car.tex_src.resize(car.textures.len(), String::new());
    // the track animates its props by mesh index: a copy would stand skinned
    // with no bones, so those stay with the instanced tiles
    let mut animated: Vec<bool> = (0..n0).map(|i| crate::track::is_animated(car, i)).collect();
    // ... and so does whatever shares a shadow caster with one: cull.rs
    // sends a level's instance indices to its caster's rows, so both must
    // hold the same placements
    loop {
        let mut grew = false;
        for i in 0..n0 {
            if let Some(Some(t)) = car.meshes[i].shadow_mesh {
                if t < n0 && animated[i] != animated[t] {
                    animated[i] = true;
                    animated[t] = true;
                    grew = true;
                }
            }
        }
        if !grew {
            break;
        }
    }
    let (mut tex_left, mut n_tex, mut mats) = (o.tex, 0usize, Vec::new());
    for c in 1..=o.copies {
        let pre = format!("stress{c}/");
        let mut tmap: std::collections::HashMap<usize, usize> = Default::default();
        for i in 0..n_mat {
            let mut m = car.materials[i].clone();
            m.path = format!("{pre}{}", m.path);
            // in name order: the map's own order differs from run to run
            let mut slots: Vec<(String, usize)> = m.slots.iter().map(|(k, v)| (k.clone(), *v)).collect();
            slots.sort();
            for (k, v) in slots {
                let j = *tmap.entry(v).or_insert_with(|| {
                    if tex_left == 0 || v >= car.textures.len() {
                        return v;
                    }
                    tex_left -= 1;
                    n_tex += 1;
                    let mut t = car.textures[v].clone();
                    crate::vram::drop_top_mips(&mut t, o.texdrop);
                    car.textures.push(t);
                    car.tex_src.push(format!("stress{c}:{v}"));
                    car.textures.len() - 1
                });
                m.slots.insert(k, j);
            }
            mats.push(m);
        }
        let map: Vec<Option<usize>> = {
            let mut next = car.meshes.len();
            animated.iter().map(|&a| (!a).then(|| { next += 1; next - 1 })).collect()
        };
        for i in 0..n0 {
            if animated[i] {
                continue;
            }
            let mut m = car.meshes[i].clone();
            let base = std::mem::take(&mut m.xforms);
            moved(&base, c, &mut m.xforms);
            for b in &mut m.batches {
                b.material = format!("{pre}{}", crate::load::norm_mat_path(&b.material));
            }
            if let Some(Some(t)) = m.shadow_mesh {
                m.shadow_mesh = map.get(t).copied().flatten().map(Some);
            }
            car.meshes.push(m);
        }
    }
    // the copies' materials stay inside the track's index range (track.rs
    // owned_material_count)
    let n_new = mats.len();
    car.materials.splice(n_mat..n_mat, mats);
    if let Some(e) = car.track_env.as_mut() {
        e.n_materials = n_mat + n_new;
    }
    let mut placed = 0usize;
    for (i, m) in car.meshes[..n0].iter_mut().enumerate() {
        let first = if animated[i] { 1 } else { o.copies + 1 };
        if first >= o.tiles {
            continue;
        }
        let base = m.xforms.clone();
        if base.is_empty() {
            m.xforms.push(IDENT);
        }
        for k in first..o.tiles {
            moved(&base, k, &mut m.xforms);
        }
        placed += m.xforms.len();
    }
    let reach = at.iter().map(|a| a.0.abs().max(a.1.abs())).max().unwrap_or(0) as f32 * pitch;
    println!("stress: {} tile(s) {pitch:.0} m apart (out to {:.1} km), {} deep cop{}: +{} mesh(es), +{n_new} material(s), +{n_tex} texture(s); {placed} placement(s) on the tiled meshes",
             o.tiles, reach / 1000.0, o.copies, if o.copies == 1 { "y" } else { "ies" }, car.meshes.len() - n0);
}

#[cfg(test)]
mod tests {
    #[test]
    fn spiral_is_distinct_and_starts_at_the_origin() {
        let s = super::spiral(30);
        assert_eq!(s[0], (0, 0));
        assert_eq!(s.len(), 30);
        let mut d = s.clone();
        d.sort();
        d.dedup();
        assert_eq!(d.len(), 30);
        assert!(s[1..9].iter().all(|a| a.0.abs().max(a.1.abs()) == 1));
    }
}
