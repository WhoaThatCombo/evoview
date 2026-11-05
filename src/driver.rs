//! The driver in the seat.
//!
//! The game builds a driver from PARTS - head, suit, gloves, shoes, helmet -
//! each its own skinned `.mesh` under `content/characters/`, and poses them
//! with the car's own driver animations. Nothing about the driver lives in the
//! car's meshes, so without this every cockpit is empty.
//!
//! WHO SITS IN THE CAR
//!   * The visual preset's field 12 names a `.driverfinalstate` when the car
//!     dresses its driver per livery (SF-25: one per race, helmet + suit +
//!     gloves re-materialled). AC1-converted cars point it at their own.
//!   * Otherwise the game's default driver, `system/saveddrivers/<kind>/
//!     default.driverfinalstate`. `<kind>` is `racing` when the mechanical
//!     preset sets field 18 (every race car: GT3, F1, cup cars) and `street`
//!     otherwise (road cars set only field 17).
//!
//! `.driverfinalstate`
//!   f200 { f20 part... f21 additive part... }   racing outfit
//!   f100 { f10 part... }                        street outfit
//!   f20 / f10 at the top level                  the same lists, older layout
//!   part { 1: mesh path, 100: kind (0 head, 1 suit/top, 2 gloves/legs,
//!          3 shoes, 4 helmet), 3|4: { 1: batch name, 2: { 1: material } } }
//!   A per-livery file overrides materials by BATCH name (3 for Uber
//!   materials, 4 otherwise - both mean the same thing here).
//!
//! POSE AND PLACEMENT
//!   The actor's f103.31 lists the driver animation layers by slot: 109 is
//!   `position` (the seated pose), 101 `steer`. Their tracks are bone LOCALS
//!   for the shared driver rig (COG > A_Hips > ... > A_Head) and are already in
//!   CAR space - A_Hips carries the seat position - so posing IS placing. Steer
//!   is 201 frames from full left to full right; its middle frame is the wheel
//!   straight ahead and is laid over `position` for the arms.
//!
//!   ⚠ Every part has its OWN bind skeleton and they do not agree (the
//!   gloves' A_Head rest sits 1.5 m below the suit's). Skin each part against
//!   its own bind; only the animated locals are shared.
//!
//!   ⚠ Head and helmet are not skinned to the rig at all. Their bones
//!   (MALE_HEAD_HD, HELMET_BASE...) are not in any animation: the engine hangs
//!   them on A_Head. Their vertices are head-centred but unrotated, and the
//!   rig's A_Head carries a fixed axis swap (the converter's `tGl`), so the
//!   part goes through the inverse of that swap and then A_Head's posed world.
//!   Stock bind matrices equal the swap; converted ones are identity, which is
//!   why the swap is used whenever the root bind is identity.
//!
//! The result is baked on the CPU into car-space meshes at the BASE pose
//! (seated, wheel straight). Each vertex keeps its skin weights, re-pointed at
//! a car-wide copy of the rig (bones named `drv:<bone>`, returned alongside),
//! whose bind is that same base pose - so the skin matrices are identity until
//! carrig.rs drives the rig (steer, pedals, shifts), and then the GPU moves
//! the driver like any other skinned part of the car.

use crate::load::{affine_inverse, base_of, mat_mul, read_animation, read_mesh, Animation, Bone, Lod,
                  Vfs, IDENTITY};
use crate::proto::{fields, text, Val};
use std::collections::HashMap;

/// Mesh names of driver parts start with this, so the renderer can find them.
pub const PREFIX: &str = "driver:";
/// Rig bones are merged into the car skeleton under this prefix, so a driver
/// `A_Head` can never collide with a car bone.
pub const BONE_PREFIX: &str = "drv:";

/// The A_Head axis swap (rows): x' = y, y' = z, z' = x.
const HEAD_SWAP: [f32; 16] = [
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// EVOVIEW_DRIVER: unset/1 = on, 0 = off, `racing` / `street` force an outfit.
pub fn setting() -> Option<Option<bool>> {
    match crate::envc::var("EVOVIEW_DRIVER").unwrap_or_default().trim().to_lowercase().as_str() {
        "0" | "off" | "false" | "no" => None,
        "racing" | "race" => Some(Some(true)),
        "street" | "road" => Some(Some(false)),
        _ => Some(None),
    }
}

struct Part {
    mesh: String,
    /// batch name -> material path
    overrides: Vec<(String, String)>,
}

fn read_part(d: &[u8], s: usize, e: usize) -> Option<Part> {
    let mut p = Part { mesh: String::new(), overrides: Vec::new() };
    fields(d, s, e, |f, v| match (f, v) {
        (1, Val::Bytes(a, b)) => p.mesh = text(d, (a, b)),
        (3 | 4, Val::Bytes(a, b)) => {
            let (mut sec, mut mat) = (String::new(), String::new());
            fields(d, a, b, |sf, sv| match (sf, sv) {
                (1, Val::Bytes(x, y)) => sec = text(d, (x, y)),
                (2, Val::Bytes(x, y)) => fields(d, x, y, |mf, mv| {
                    if let (1, Val::Bytes(i, j)) = (mf, mv) {
                        mat = text(d, (i, j));
                    }
                }),
                _ => {}
            });
            if !sec.is_empty() && !mat.is_empty() {
                p.overrides.push((sec, mat));
            }
        }
        _ => {}
    });
    (!p.mesh.is_empty()).then_some(p)
}

/// (racing parts, street parts) of a `.driverfinalstate`.
fn read_finalstate(d: &[u8]) -> (Vec<Part>, Vec<Part>) {
    let (mut race, mut street) = (Vec::new(), Vec::new());
    let (mut race_top, mut street_top) = (Vec::new(), Vec::new());
    fields(d, 0, d.len(), |f, v| {
        let Some((s, e)) = v.range() else { return };
        match f {
            200 => fields(d, s, e, |pf, pv| {
                if let (20 | 21, Some((a, b))) = (pf, pv.range()) {
                    race.extend(read_part(d, a, b));
                }
            }),
            100 => fields(d, s, e, |pf, pv| {
                if let (10 | 11, Some((a, b))) = (pf, pv.range()) {
                    street.extend(read_part(d, a, b));
                }
            }),
            20 | 21 => race_top.extend(read_part(d, s, e)),
            10 | 11 => street_top.extend(read_part(d, s, e)),
            _ => {}
        }
    });
    if race.is_empty() {
        race = race_top;
    }
    if street.is_empty() {
        street = street_top;
    }
    (race, street)
}

/// Driver animation paths from the actor's f103.31, by slot.
fn actor_driver_anims(vfs: &mut Vfs) -> HashMap<u32, String> {
    let mut out = HashMap::new();
    let Some(a) = vfs.list("", "actor").into_iter().next() else { return out };
    let d = vfs.read(&a);
    // a path is either slot { 1: { 1: path } } (current) or slot { 1: path } (legacy 1..9)
    let path_in = |s: usize, e: usize| -> Option<String> {
        let mut hit = None;
        fields(&d, s, e, |f, v| {
            if let (1, Some((a, b))) = (f, v.range()) {
                let t = &d[a..b];
                if t.iter().all(|c| (0x20..0x7f).contains(c)) && !t.is_empty() {
                    hit = Some(text(&d, (a, b)));
                } else {
                    fields(&d, a, b, |g, w| {
                        if let (1, Some((x, y))) = (g, w.range()) {
                            hit = Some(text(&d, (x, y)));
                        }
                    });
                }
            }
        });
        hit.filter(|p| p.to_lowercase().ends_with(".animation"))
    };
    fields(&d, 0, d.len(), |f, v| {
        let (103, Some((s, e))) = (f, v.range()) else { return };
        fields(&d, s, e, |g, w| {
            let (31, Some((s2, e2))) = (g, w.range()) else { return };
            fields(&d, s2, e2, |slot, sv| {
                if let Some((a, b)) = sv.range() {
                    if let Some(p) = path_in(a, b) {
                        out.entry(slot).or_insert(p);
                    }
                }
            });
        });
    });
    // legacy slots 9 / 1 stand in for 109 / 101 when the new ones are empty
    for (new, old) in [(109u32, 9u32), (101, 1)] {
        if !out.contains_key(&new) {
            if let Some(p) = out.get(&old).cloned() {
                out.insert(new, p);
            }
        }
    }
    out
}

fn first_frame_locals(a: &Animation, frame: Frame, into: &mut HashMap<String, [f32; 16]>) {
    for t in &a.tracks {
        if t.frames.is_empty() {
            continue;
        }
        let f = match frame {
            Frame::First => 0,
            Frame::Middle => (t.frames.len() - 1) / 2,
        };
        into.insert(t.bone.clone(), t.frames[f]);
    }
}

#[derive(Clone, Copy)]
enum Frame {
    First,
    Middle,
}

fn near_identity(m: &[f32; 16]) -> bool {
    m.iter().zip(IDENTITY.iter()).all(|(a, b)| (a - b).abs() < 1e-4)
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn put(buf: &mut [u8], i: usize, v: f32) {
    buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
}

/// Skin `lod` in place with one matrix per local bone (row-vector).
fn bake(lod: &mut Lod, skin: &[[f32; 16]]) {
    let n = lod.positions.len() / 12;
    let pos = f32s(&lod.positions);
    let nrm = f32s(&lod.normals);
    let tan = f32s(&lod.tangents);
    let idx = f32s(&lod.bone_idx);
    let wgt = f32s(&lod.bone_wgt);
    let has_skin = idx.len() >= n * 4 && wgt.len() >= n * 4;
    let xf = |m: &[f32; 16], v: [f32; 3], w: f32| -> [f32; 3] {
        let mut o = [0.0f32; 3];
        for c in 0..3 {
            o[c] = v[0] * m[c] + v[1] * m[4 + c] + v[2] * m[8 + c] + w * m[12 + c];
        }
        o
    };
    let norm = |v: [f32; 3]| -> [f32; 3] {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if l > 1e-8 { [v[0] / l, v[1] / l, v[2] / l] } else { v }
    };
    for i in 0..n {
        // blended matrix: sum of w * M, weights normalised
        let mut m = [0.0f32; 16];
        let mut total = 0.0f32;
        if has_skin {
            for k in 0..4 {
                let w = wgt[i * 4 + k];
                if w <= 0.0 {
                    continue;
                }
                let b = idx[i * 4 + k] as usize;
                let s = skin.get(b).unwrap_or(&IDENTITY);
                for e in 0..16 {
                    m[e] += w * s[e];
                }
                total += w;
            }
        }
        if total <= 1e-6 {
            m = *skin.first().unwrap_or(&IDENTITY);
        } else {
            for e in m.iter_mut() {
                *e /= total;
            }
        }
        let p = xf(&m, [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]], 1.0);
        for a in 0..3 {
            put(&mut lod.positions, i * 3 + a, p[a]);
        }
        if nrm.len() >= n * 3 {
            let q = norm(xf(&m, [nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]], 0.0));
            for a in 0..3 {
                put(&mut lod.normals, i * 3 + a, q[a]);
            }
        }
        if tan.len() >= n * 4 {
            let q = norm(xf(&m, [tan[i * 4], tan[i * 4 + 1], tan[i * 4 + 2]], 0.0));
            for a in 0..3 {
                put(&mut lod.tangents, i * 4 + a, q[a]);
            }
        }
    }
    lod.bone_idx.clear();
    lod.bone_wgt.clear();
    lod.bones.clear();
}

/// Load, pose and place the car's driver. Returns (mesh name, car-space LOD)
/// per part, with livery materials already set on the batches, plus the rig
/// bones those LODs' bone indices point into (parents are indices within the
/// returned list). Empty when the car has no driver seat (no `position`
/// animation) or EVOVIEW_DRIVER=0.
pub fn load(vfs: &mut Vfs) -> (Vec<(String, Lod)>, Vec<Bone>) {
    let none = || (Vec::new(), Vec::new());
    let Some(force) = setting() else {
        println!("driver: off (EVOVIEW_DRIVER=0)");
        return none();
    };
    let anims = actor_driver_anims(vfs);
    let Some(pos_path) = anims.get(&109) else {
        println!("driver: none (the actor has no driver position animation)");
        return none();
    };
    let read_anim = |vfs: &mut Vfs, p: &str| -> Option<Animation> {
        let k = vfs.find(p, None)?;
        let b = vfs.read(&k);
        read_animation(&b, &base_of(&k))
    };
    let Some(position) = read_anim(vfs, pos_path) else {
        println!("driver: none ({pos_path} does not resolve)");
        return none();
    };
    let steer = anims.get(&101).and_then(|p| read_anim(vfs, p));

    // posed locals: seated, then the arms on a centred wheel
    let mut posed: HashMap<String, [f32; 16]> = HashMap::new();
    first_frame_locals(&position, Frame::First, &mut posed);
    if let Some(s) = steer.as_ref() {
        first_frame_locals(s, Frame::Middle, &mut posed);
    }

    // who: the livery's own driver, else the default one for the car's kind
    let racing = force.unwrap_or_else(|| {
        let Some(m) = crate::load::pick_mech_preset(vfs) else { return true };
        let d = vfs.read(&m);
        let mut r = false;
        fields(&d, 0, d.len(), |f, v| {
            if let (18, Val::Varint(n)) = (f, v) {
                r |= n != 0;
            }
        });
        r
    });
    let mut fs_key = crate::load::pick_visual_preset(vfs).and_then(|vp| {
        let d = vfs.read(&vp);
        let mut p = None;
        fields(&d, 0, d.len(), |f, v| {
            if let (12, Some(r)) = (f, v.range()) {
                p = Some(text(&d, r));
            }
        });
        p.filter(|s| !s.is_empty()).and_then(|s| vfs.find(&s, None))
    });
    let livery = fs_key.is_some();
    if fs_key.is_none() {
        let kind = if racing { "racing" } else { "street" };
        fs_key = vfs.find(&format!("system/saveddrivers/{kind}/default.driverfinalstate"), None)
            .filter(|k| k.contains(kind));
    }
    let Some(fs_key) = fs_key else {
        println!("driver: no .driverfinalstate found");
        return none();
    };
    let fsd = vfs.read(&fs_key);
    let (race, street) = read_finalstate(&fsd);
    let parts = match (racing, race.is_empty(), street.is_empty()) {
        (true, false, _) | (false, false, true) => race,
        _ => street,
    };
    println!("driver: {} outfit from {}{} ({} parts), pose {}{}",
        if racing { "racing" } else { "street" }, fs_key,
        if livery { " (livery)" } else { "" }, parts.len(), base_of(pos_path),
        if steer.is_some() { " + steer centre" } else { "" });

    // read every part first: the rig's rest locals come from all of them
    let mut lods: Vec<(String, Lod, Vec<(String, String)>)> = Vec::new();
    for p in parts {
        let Some(k) = vfs.find(&p.mesh, None) else {
            println!("driver: part {} not found", p.mesh);
            continue;
        };
        let b = vfs.read(&k);
        let Some(lod) = read_mesh(&b) else { continue };
        if lod.positions.is_empty() || lod.indices.is_empty() {
            continue;
        }
        lods.push((base_of(&k), lod, p.overrides));
    }
    // rig-wide rest: name -> (parent name, local), first part to define a bone wins
    let mut rest: HashMap<String, (Option<String>, [f32; 16])> = HashMap::new();
    for (_, lod, _) in &lods {
        for b in &lod.bones {
            rest.entry(b.name.clone()).or_insert_with(|| {
                (b.parent.map(|p| lod.bones[p].name.clone()), b.local)
            });
        }
    }
    let rig_world = |name: &str| -> Option<[f32; 16]> {
        let mut m = *posed.get(name).or_else(|| rest.get(name).map(|r| &r.1))?;
        let mut cur = rest.get(name).and_then(|r| r.0.clone());
        let mut guard = 0;
        while let Some(p) = cur {
            let l = posed.get(&p).copied().or_else(|| rest.get(&p).map(|r| r.1))
                .unwrap_or(IDENTITY);
            m = mat_mul(&m, &l);
            cur = rest.get(&p).and_then(|r| r.0.clone());
            guard += 1;
            if guard > 64 {
                break;
            }
        }
        Some(m)
    };
    let head = rig_world("A_Head");

    // The car-wide rig: every bone any part names, local = the base pose.
    let mut rig_names: Vec<String> = rest.keys().cloned().collect();
    for k in posed.keys() {
        if !rest.contains_key(k) {
            rig_names.push(k.clone());
        }
    }
    rig_names.sort();
    let rig_index: HashMap<String, usize> =
        rig_names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();
    let rig_bones: Vec<Bone> = rig_names
        .iter()
        .map(|n| Bone {
            name: format!("{BONE_PREFIX}{n}"),
            parent: rest.get(n).and_then(|r| r.0.as_ref()).and_then(|p| rig_index.get(p).copied()),
            local: posed.get(n).copied().or_else(|| rest.get(n).map(|r| r.1)).unwrap_or(IDENTITY),
        })
        .collect();
    let head_rig = rig_index.get("A_Head").copied().unwrap_or(0);
    if let Some(h) = head {
        println!("driver: head at ({:.2}, {:.2}, {:.2})", h[12], h[13], h[14]);
    }

    let mut out = Vec::new();
    for (name, mut lod, overrides) in lods {
        let nb = lod.bones.len();
        // per local bone: does it (or an ancestor) take an animation track?
        let mut animated = vec![false; nb];
        let mut root = vec![0usize; nb];
        let mut bind = vec![IDENTITY; nb];
        let mut world = vec![IDENTITY; nb];
        for i in 0..nb {
            let b = &lod.bones[i];
            let local = posed.get(&b.name).copied().unwrap_or(b.local);
            match b.parent.filter(|&p| p < i) {
                Some(p) => {
                    animated[i] = animated[p] || posed.contains_key(&b.name);
                    root[i] = root[p];
                    bind[i] = mat_mul(&b.local, &bind[p]);
                    world[i] = mat_mul(&local, &world[p]);
                }
                None => {
                    animated[i] = posed.contains_key(&b.name);
                    root[i] = i;
                    bind[i] = b.local;
                    world[i] = local;
                }
            }
        }
        let mut skin = Vec::with_capacity(nb.max(1));
        let mut rigid = 0;
        for i in 0..nb {
            if animated[i] {
                skin.push(mat_mul(&affine_inverse(&bind[i]), &world[i]));
                continue;
            }
            // hung on A_Head (head, helmet)
            let Some(h) = head else {
                skin.push(IDENTITY);
                continue;
            };
            let rb = bind[root[i]];
            let c = if near_identity(&rb) { affine_inverse(&HEAD_SWAP) } else { affine_inverse(&rb) };
            skin.push(mat_mul(&c, &h));
            rigid += 1;
        }
        if nb == 0 {
            skin.push(head.unwrap_or(IDENTITY));
        }
        // which rig bone each local bone follows from here on
        let to_rig: Vec<f32> = (0..nb.max(1))
            .map(|i| {
                let follows = match lod.bones.get(i) {
                    Some(b) if animated[i] => rig_index.get(&b.name).copied().unwrap_or(head_rig),
                    _ => head_rig,
                };
                follows as f32
            })
            .collect();
        let old_idx = f32s(&lod.bone_idx);
        let old_wgt = f32s(&lod.bone_wgt);
        let nv = lod.positions.len() / 12;
        bake(&mut lod, &skin);
        let skinned = old_idx.len() >= nv * 4 && old_wgt.len() >= nv * 4;
        let mut idx = Vec::with_capacity(nv * 16);
        let mut wgt = Vec::with_capacity(nv * 16);
        for i in 0..nv {
            let total: f32 = if skinned { (0..4).map(|k| old_wgt[i * 4 + k].max(0.0)).sum() } else { 0.0 };
            for k in 0..4 {
                let (b, w) = if total > 1e-6 {
                    let w = old_wgt[i * 4 + k].max(0.0) / total;
                    (to_rig.get(old_idx[i * 4 + k] as usize).copied().unwrap_or(head_rig as f32), w)
                } else if k == 0 {
                    (to_rig[0], 1.0)
                } else {
                    (0.0, 0.0)
                };
                idx.extend_from_slice(&b.to_le_bytes());
                wgt.extend_from_slice(&w.to_le_bytes());
            }
        }
        lod.bone_idx = idx;
        lod.bone_wgt = wgt;
        for b in lod.batches.iter_mut() {
            if let Some((_, m)) = overrides.iter().find(|(s, _)| s.eq_ignore_ascii_case(&b.name)) {
                b.material = m.clone();
            }
        }
        println!("driver: {name} ({} verts{})", lod.positions.len() / 12,
            if rigid > 0 { ", on A_Head" } else { "" });
        out.push((format!("{PREFIX}{}", name.trim_end_matches(".mesh")), lod));
    }
    (out, rig_bones)
}
