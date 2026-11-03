//! The game's own contact shadows: textured decals on the ground under the
//! body and under each tyre.
//!
//! EVO does not compute the dark patch under a car. Every car's `.actor`
//! names a pre-drawn shadow texture and the size of the quad it is stamped on:
//!
//! ```text
//! actor ... f28 body shadow   { f1 width(x) f2 length(z) f3 y? f4 texture
//!                               f5 z offset f6 ? f7 opacity }
//!           f29 front tyres   { same shape, texture Tyre_Shadow }
//!           f30 rear tyres
//! ```
//!
//! M2 coupe: body 2.12 x 4.957 m `textures/body_shadow.texture` (512^2 BC1),
//! tyres 0.38 x 0.73 m `common_assets/parts/tyres/textures/common/
//! Tyre_Shadow.texture`. The M4 GT3 has a body shadow and empty tyre entries.
//!
//! The decals are drawn as ordinary alpha-blended meshes with a material the
//! shader recognises (`misc.w == 4`): it writes black with the texture as
//! coverage, so the floor (or track) under it is darkened and nothing else.
//! That needs no new pass, binding or vertex layout.

use crate::proto::{fields, text, Val};

/// Shader name on the synthetic material, matched in material.rs.
pub const SHADER: &str = "EvoviewContactShadow";

#[derive(Debug, Clone)]
pub struct Decal {
    pub texture: String,
    pub width: f32,
    pub length: f32,
    pub z_offset: f32,
    pub opacity: f32,
}

/// Body, front-tyre and rear-tyre decals, as authored in the actor.
#[derive(Debug, Default, Clone)]
pub struct Decals {
    pub body: Option<Decal>,
    pub front: Option<Decal>,
    pub rear: Option<Decal>,
}

fn parse_decal(d: &[u8], s: usize, e: usize) -> Option<Decal> {
    let mut out = Decal { texture: String::new(), width: 0.0, length: 0.0,
                          z_offset: 0.0, opacity: 1.0 };
    let mut ok = true;
    // ⚠ Some actors (the 964) follow the real fields with zero-length
    // copies of 1/2/3/5/6/7; read as numbers they zeroed the width and the
    // body shadow was rejected. Numbers only come from number wire types.
    fields(d, s, e, |f, v| match f {
        1 | 2 | 5 | 7 if matches!(v, Val::Bytes(..)) => {}
        1 => out.width = v.f32(),
        2 => out.length = v.f32(),
        4 => match v.range() {
            Some(r) => out.texture = text(d, r),
            None => ok = false,
        },
        5 => out.z_offset = v.f32(),
        7 => out.opacity = v.f32(),
        _ => {}
    });
    let tex = out.texture.to_lowercase();
    if !ok || !tex.ends_with(".texture") || !tex.contains("shadow") {
        return None;
    }
    if !(out.width > 0.01 && out.width < 20.0 && out.length > 0.01 && out.length < 40.0) {
        return None;
    }
    Some(out)
}

/// Find the shadow block structurally: any message whose fields 28/29/30
/// parse as shadow decals. Read by shape rather than by a fixed path because
/// the actor's outer nesting is not decoded.
fn scan(d: &[u8], s: usize, e: usize, depth: u32, out: &mut Decals) {
    if depth > 6 || out.body.is_some() {
        return;
    }
    let mut found = Decals::default();
    let mut subs = Vec::new();
    fields(d, s, e, |f, v| {
        if let Some((a, b)) = v.range() {
            // ⚠ A singular message written twice MERGES (protobuf), it does not
            // replace: the 964's actor has its body decal and then an empty
            // second copy, and taking the last one dropped its body shadow.
            let slot = match f {
                28 => Some(&mut found.body),
                29 => Some(&mut found.front),
                30 => Some(&mut found.rear),
                _ => None,
            };
            if let (Some(slot), Some(x)) = (slot, parse_decal(d, a, b)) {
                *slot = Some(x);
            }
            if b > a + 2 {
                subs.push((a, b));
            }
        }
    });
    if found.body.is_some() || found.front.is_some() || found.rear.is_some() {
        *out = found;
        return;
    }
    for (a, b) in subs {
        scan(d, a, b, depth + 1, out);
        if out.body.is_some() || out.front.is_some() {
            return;
        }
    }
}

pub fn read_actor(d: &[u8]) -> Decals {
    let mut out = Decals::default();
    scan(d, 0, d.len(), 0, &mut out);
    out
}

/// One flat quad, `FLOATS` layout identical to the loader's separate arrays:
/// positions, normals, uvs, tangents, indices.
pub struct Quad {
    pub positions: Vec<u8>,
    pub normals: Vec<u8>,
    pub uvs: Vec<u8>,
    pub tangents: Vec<u8>,
    pub indices: Vec<u32>,
}

/// A horizontal quad centred on (cx, y, cz), `w` along X and `l` along Z.
pub fn quad(cx: f32, y: f32, cz: f32, w: f32, l: f32) -> Quad {
    let (hw, hl) = (w * 0.5, l * 0.5);
    let p: [f32; 12] = [
        cx - hw, y, cz - hl,
        cx + hw, y, cz - hl,
        cx + hw, y, cz + hl,
        cx - hw, y, cz + hl,
    ];
    let n: [f32; 12] = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let uv: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let t: [f32; 16] = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                        1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    Quad {
        positions: bytemuck::cast_slice(&p).to_vec(),
        normals: bytemuck::cast_slice(&n).to_vec(),
        uvs: bytemuck::cast_slice(&uv).to_vec(),
        tangents: bytemuck::cast_slice(&t).to_vec(),
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}
