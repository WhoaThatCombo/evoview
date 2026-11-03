//! A "debug world" for materials.
//!
//! Minecraft ships a debug world that lays out every block state in a flat
//! grid, because the fastest way to find the one that is wrong is to see all of
//! them at once. Cars and tracks have the same problem: a car is 90-odd
//! materials and a circuit is 200, and when one of them is wrong the symptom is
//! a single white patch somewhere inside a mesh you cannot easily isolate.
//!
//! `--materials` replaces the scene with one sphere per material, laid out
//! alphabetically, lit by exactly the same shader path the real content uses -
//! so what you see here is what that material does on the car or the track,
//! not an approximation of it.
//!
//! Finding the roll-bar padding bug took four `EVOVIEW_MAT_HL` passes to work
//! out which material even owned the white pixels. On a sheet it is one glance,
//! and the legend gives the name.

use crate::asset::Car;
use crate::material::{is_cabin, key_of, Blend, Materials};
use crate::render::{pack_instances, GpuMesh, FLOATS_PER_VERTEX};
use glam::Vec3;
use wgpu::util::DeviceExt;

/// Was `--materials` asked for?
pub fn wanted() -> bool {
    std::env::args().any(|a| a == "--materials")
}

/// Sphere radius in metres. It matters: triplanar and world-tiled materials
/// (track terrain, roads) scale their UVs by WORLD size, so a 1 m ball shows a
/// fraction of one tile and reads as flat colour. Default 1.5 m shows a couple
/// of tiles; `--grid-radius` overrides it.
fn radius() -> f32 {
    let mut it = std::env::args();
    while let Some(a) = it.next() {
        if a == "--grid-radius" {
            if let Some(v) = it.next().and_then(|v| v.parse().ok()) {
                return v;
            }
        }
    }
    1.5
}

/// `--materials-only` filter list, lowercased.
fn only() -> Option<Vec<String>> {
    let mut it = std::env::args();
    while let Some(a) = it.next() {
        if a == "--materials-only" {
            return it.next().map(|v| {
                v.split(',')
                    .map(|x| x.trim().to_lowercase())
                    .filter(|x| !x.is_empty())
                    .collect()
            });
        }
    }
    None
}

/// A UV sphere: positions, normals, UVs and real tangents, so normal maps and
/// tiling behave the way they do on the real mesh.
fn sphere(r: f32, at: Vec3, segs: usize, rings: usize) -> (Vec<f32>, Vec<u32>) {
    let mut verts = vec![0f32; (segs + 1) * (rings + 1) * FLOATS_PER_VERTEX];
    for j in 0..=rings {
        let phi = std::f32::consts::PI * j as f32 / rings as f32;
        let (sp, cp) = phi.sin_cos();
        for i in 0..=segs {
            let theta = std::f32::consts::TAU * i as f32 / segs as f32;
            let (st, ct) = theta.sin_cos();
            let n = Vec3::new(sp * ct, cp, sp * st);
            let p = at + n * r;
            let o = (j * (segs + 1) + i) * FLOATS_PER_VERTEX;
            verts[o..o + 3].copy_from_slice(&[p.x, p.y, p.z]);
            verts[o + 3..o + 6].copy_from_slice(&[n.x, n.y, n.z]);
            verts[o + 6] = i as f32 / segs as f32;
            verts[o + 7] = j as f32 / rings as f32;
            // dP/du, which is what a tangent-space normal map expects.
            verts[o + 8..o + 12].copy_from_slice(&[-st, 0.0, ct, 1.0]);
            // zero bone weights: "leave this vertex alone"
        }
    }
    let mut idx = Vec::with_capacity(segs * rings * 6);
    for j in 0..rings {
        for i in 0..segs {
            let a = (j * (segs + 1) + i) as u32;
            let b = a + segs as u32 + 1;
            idx.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    (verts, idx)
}

/// Build the sheet. Same return shape as `build_scene`, so the caller's camera
/// framing works unchanged.
pub fn build(
    device: &wgpu::Device,
    car: &Car,
    mats: &Materials,
) -> (Vec<GpuMesh>, Vec3, Vec3, u32) {
    // Alphabetical, so families land together and an odd one out in a run of
    // `int_elec_*` is obvious.
    let mut names: Vec<String> = car.materials.iter().map(|m| m.file.clone()).collect();
    names.sort_by_key(|a| a.to_lowercase());
    names.dedup();
    // `--materials-only a,b,c` narrows the sheet to matching names. Once the
    // full sheet has told you WHICH materials are suspect, a sheet of just
    // those is what you actually stare at while fixing them.
    if let Some(want) = only() {
        names.retain(|n| {
            let l = n.to_lowercase();
            want.iter().any(|w| l.contains(w.as_str()))
        });
        if names.is_empty() {
            println!("--materials-only matched nothing; showing everything instead");
            names = car.materials.iter().map(|m| m.file.clone()).collect();
            names.sort_by_key(|a| a.to_lowercase());
            names.dedup();
        }
    }

    let r = radius();
    let step = r * 2.6;
    let cols = (names.len() as f32).sqrt().ceil().max(1.0) as usize;
    let rows = names.len().div_ceil(cols);

    let mut meshes = Vec::new();
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    let mut tris = 0u32;

    for (k, name) in names.iter().enumerate() {
        let (col, row) = (k % cols, k / cols);
        // A wall in XY, not a field on the ground: a flat sheet seen head-on
        // wastes no pixels on perspective foreshortening.
        let at = Vec3::new(
            (col as f32 - (cols - 1) as f32 * 0.5) * step,
            ((rows - 1) as f32 * 0.5 - row as f32) * step,
            0.0,
        );
        let (verts, idx) = sphere(r, at, 32, 16);
        lo = lo.min(at - Vec3::splat(r));
        hi = hi.max(at + Vec3::splat(r));
        tris += (idx.len() / 3) as u32;
        let mi = mats.lookup(name);
        meshes.push(GpuMesh {
            draws: vec![(0, idx.len() as u32, mi)],
            vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: bytemuck::cast_slice(&idx),
                usage: wgpu::BufferUsages::INDEX,
            }),
            instances: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("grid instance"),
                contents: bytemuck::cast_slice(&pack_instances(&[], 0)),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            instance_count: 1,
            index_count: idx.len() as u32,
            src: usize::MAX,
            extra: None,
        });
    }

    // A material file existing does not mean any mesh draws it. Spa ships 277
    // and a good few are legacy; without this you go hunting a white sphere
    // that is never on screen.
    let used: std::collections::HashSet<String> = car
        .meshes
        .iter()
        .flat_map(|m| m.batches.iter())
        .map(|b| {
            let base = b
                .material
                .rsplit(['\\', '/'])
                .next()
                .unwrap_or(&b.material);
            key_of(base)
        })
        .collect();
    // ⚠ Honesty flag. A splat or triplanar material derives its colour from a
    // SpatMap and world position, and a sphere sitting at the origin has
    // neither in any meaningful way - so those cells are NOT representative and
    // must not be read as bugs. On Spa that is 30-odd UberBiplanar materials.
    let world_driven: std::collections::HashSet<String> = car
        .materials
        .iter()
        .filter(|m| {
            m.slot(&["SpatMap", "SplatMap"]).is_some()
                || m.param(&["Base_Triplanar"], 0.0) > 0.5
                || m.param(&["Red_Triplanar"], 0.0) > 0.5
        })
        .map(|m| key_of(&m.file))
        .collect();
    legend(&names, cols, mats, &used, &world_driven);
    (meshes, lo, hi, tris)
}

/// The half that makes the picture actionable: which cell is which, and what
/// the loader decided about it. Row/column match the sheet exactly.
fn legend(
    names: &[String],
    cols: usize,
    mats: &Materials,
    used: &std::collections::HashSet<String>,
    world_driven: &std::collections::HashSet<String>,
) {
    println!(
        "\nmaterial sheet: {} materials, {} columns, sphere radius {:.2} m",
        names.len(),
        cols,
        radius()
    );
    println!(
        "  flags: F=fallback (unbound)  A=alpha  C=cutout  I=cabin/interior           U=unused (no mesh draws it)"
    );
    println!(
        "  S = splat/triplanar: colour comes from a SpatMap and world position,          so the
      sphere is NOT representative - check those on the real          mesh, not here."
    );
    let mut n_fallback = 0;
    let mut n_unused = 0;
    let mut n_world = 0;
    for (k, name) in names.iter().enumerate() {
        if k % cols == 0 {
            print!("\n r{:<3}", k / cols);
        }
        let mi = mats.lookup(name);
        let mut f = String::new();
        if mi == mats.fallback {
            f.push('F');
            n_fallback += 1;
        }
        if is_cabin(&name.to_lowercase()) {
            f.push('I');
        }
        if world_driven.contains(&key_of(name)) {
            f.push('S');
            n_world += 1;
        }
        if !used.contains(&key_of(name)) {
            f.push('U');
            n_unused += 1;
        }
        match mats.blend.get(mi) {
            Some(Blend::Alpha) => f.push('A'),
            Some(Blend::Cutout) => f.push('C'),
            Some(Blend::Glass) => f.push('G'),
            _ => {}
        }
        let short = name.trim_end_matches(".material");
        print!(" [c{:<2}{:>4} {}]", k % cols, f, short);
    }
    println!();
    if n_world > 0 {
        println!("{n_world} material(s) are splat/triplanar (S) - judge those in the scene.");
    }
    if n_unused > 0 {
        println!("{n_unused} material(s) are on the sheet but no mesh draws them (U).");
    }
    if n_fallback > 0 {
        println!(
            "⚠ {n_fallback} material(s) are unbound and drew on the fallback - \
             those spheres are grey by definition, not by authoring."
        );
    }
}
