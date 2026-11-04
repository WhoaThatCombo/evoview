//! Ground queries through the ODE port's ray collider (EVOVIEW_GROUND_ODE=1)
//! - the OPCODE tree and dCollideRTL the game asks for its ground: the
//! physics engine keeps one ray (dCreateRay, length 100, backface culling;
//! FUN_142516460) and its raycast (FUN_1425154d0) collides it with the
//! track space and keeps the nearest hit.
//!
//! Here the track is two trimeshes: the tagged surfaces (surface_id: what
//! the game collides with) and the untagged ones (terrain tiles, verges),
//! asked only where nothing tagged lies below. Not decoded: how many rays
//! a tyre casts and where - trackdrive's tyre envelope still picks the
//! sample points; each sample is now this ray.

use crate::ode::collision::{GeomId, TriMeshData};
use crate::ode::World;
use glam::Vec3;
use std::sync::Mutex;

struct Mesh {
    geom: GeomId,
    /// the trimesh's triangle -> the collision's
    tri: Vec<u32>,
}

pub struct OdeGround {
    w: Mutex<World>,
    ray: GeomId,
    mesh: [Option<Mesh>; 2],
}

impl OdeGround {
    pub fn enabled() -> bool {
        std::env::var_os("EVOVIEW_GROUND_ODE").is_some()
    }

    /// `keep[ti]`: the triangle is ground at all; `tagged[ti]`: on a
    /// surface with a surface_id.
    pub fn build(verts: &[[f32; 3]], tris: &[[u32; 3]], keep: impl Fn(usize) -> bool, tagged: impl Fn(usize) -> bool) -> OdeGround {
        let t0 = std::time::Instant::now();
        let mut w = World::new();
        let v: Vec<[f64; 3]> = verts.iter().map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect();
        let mut mesh = [None, None];
        for (slot, want) in [true, false].into_iter().enumerate() {
            let sel: Vec<u32> = (0..tris.len()).filter(|&ti| keep(ti) && tagged(ti) == want).map(|ti| ti as u32).collect();
            if sel.is_empty() {
                continue;
            }
            let t: Vec<[u32; 3]> = sel.iter().map(|&ti| tris[ti as usize]).collect();
            let geom = w.create_trimesh(None, TriMeshData::build(v.clone(), t));
            mesh[slot] = Some(Mesh { geom, tri: sel });
        }
        // the game's ray: backface culled; the nearest hit (the game takes
        // the first contact per geom and the nearest over its many geoms;
        // with one geom for the whole track, the nearest)
        let ray = w.create_ray(None, 100.0);
        w.geom_ray_set_params(ray, false, true, true);
        println!("ground: ODE ray ground, {} tagged + {} untagged triangles, tree built in {:.1} s",
            mesh[0].as_ref().map_or(0, |m| m.tri.len()), mesh[1].as_ref().map_or(0, |m| m.tri.len()), t0.elapsed().as_secs_f32());
        OdeGround { w: Mutex::new(w), ray, mesh }
    }

    /// The ground straight below (x, top, z): height, normal (up), the
    /// collision triangle.
    pub fn probe(&self, x: f32, z: f32, top: f32) -> Option<(f32, Vec3, u32)> {
        let mut w = self.w.lock().unwrap();
        w.geom_ray_set(self.ray, x as f64, top as f64, z as f64, 0.0, -1.0, 0.0);
        for m in self.mesh.iter().flatten() {
            let cs = w.collide(self.ray, m.geom, 1);
            if let Some(c) = cs.first() {
                let i = c.side1.max(c.side2);
                if i < 0 {
                    continue;
                }
                let mut n = Vec3::new(c.normal[0] as f32, c.normal[1] as f32, c.normal[2] as f32);
                if n.y < 0.0 {
                    n = -n;
                }
                return Some((c.pos[1] as f32, n, m.tri[i as usize]));
            }
        }
        None
    }
}
