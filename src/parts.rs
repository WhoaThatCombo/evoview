//! Body-kit and equipment parts: what the active presets FIT, where each one
//! goes, and which stock geometry it REPLACES.
//!
//! The schema is the game's own (protobuf descriptors embedded in
//! AssettoCorsaEVO.exe, Customization.proto):
//!
//!   MechanicalCarPresetData
//!     10 forced_compatible_part_paths  parts this preset always fits
//!     9  compatible_part_paths         parts it fits by default but the
//!                                      player may remove: the MX5 NA RS roll
//!                                      bar + wing, the Caterham race cage, the
//!                                      Dallara's coupe roof / spider screen are
//!                                      all f9 and all on the thumbnails.
//!                                      Rivals are settled by the carpart's
//!                                      not_allowed list (M2 wing 1 vs wing 2)
//!     11 not_compatible_part_paths
//!   VisualCarPresetData
//!     10 part_states {1 part, 2 part_design}   the DESIGN a trim gives a part
//!                                      IF it is fitted - not a fitment: the
//!                                      MX5 NA's visual_1 names the RS roll bar
//!                                      and its thumbnail has none
//!   CompatiblePartData (.compatiblepart)
//!     1 car_part_path  2 attachment_type {0 Replacement, 1 Additive}
//!     3 attachment_bone  5 attachment_relative_transform {1 pos, 2 euler, 3 scale}
//!   CarPartData (.carpart)
//!     3 part_type  5 mesh_path  7 design_paths  31 not_allowed_part_paths
//!
//! There is no hide list anywhere in the data. A REPLACEMENT part replaces
//! what the stock car has on its attachment bone: the stock meshes skin every
//! piece of swappable bodywork to its own bone (FRONT_BUMPER_TRIM, DOOR_L_TRIM,
//! TRIM, EXHAUST, SEATS, STEER_BASE, DISC_LF, SUSP_LF ...), and across all 94
//! replacement fitments in the fleet every vertex on those bones is weighted
//! 1.0 to them - never blended - so the bone IS the hide mask. The bone's
//! children go too (the AE86's stock rear wiper hangs off TRIM and the Kouki
//! body brings its own), which is what scaling a bone to zero does in the
//! engine. Additive parts (side skirts, wings, fog lights on CAR_BODY /
//! FRONT_BUMPER) hide nothing.
//!
//! Part meshes are authored in their attachment bone's frame, like a
//! steering wheel: placed by bone bind x relative transform.

use crate::load::{base_name_lower, euler_rot, parse_transform, Vfs};
use crate::proto::{fields, text, Val};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct FittedPart {
    /// vfs key of the .compatiblepart
    pub compat: String,
    /// vfs key of the part mesh
    pub mesh: String,
    pub bone: String,
    pub additive: bool,
    /// attachment_relative_transform as a row-vector matrix (v' = v * m), in
    /// the bone's frame
    pub local: [f32; 16],
    /// the design the visual preset dresses it with, if any
    pub design: Option<String>,
}

struct Compat {
    part: Option<String>,
    bone: Option<String>,
    additive: bool,
    local: [f32; 16],
}

fn read_compat(d: &[u8]) -> Compat {
    let mut c = Compat { part: None, bone: None, additive: false, local: IDENTITY };
    fields(d, 0, d.len(), |f, v| match (f, v) {
        (1, Val::Bytes(a, b)) => c.part = Some(text(d, (a, b))),
        (2, Val::Varint(n)) => c.additive = n == 1,
        (3, Val::Bytes(a, b)) => c.bone = Some(text(d, (a, b))),
        (5, Val::Bytes(a, b)) => c.local = local_matrix(parse_transform(d, a, b)),
        _ => {}
    });
    c
}

const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// (pos, euler deg, scale) -> row-major row-vector matrix. `euler_rot` is the
/// column-vector rotation the scene loader uses, so its transpose goes in the
/// rows here.
pub(crate) fn local_matrix((p, e, s): ([f32; 3], [f32; 3], [f32; 3])) -> [f32; 16] {
    let r = euler_rot(e);
    let mut m = IDENTITY;
    for i in 0..3 {
        for c in 0..3 {
            m[i * 4 + c] = r[c][i] * s[i];
        }
    }
    m[12] = p[0];
    m[13] = p[1];
    m[14] = p[2];
    m
}

/// Every part the active mech + visual presets fit, with its mesh resolved.
/// Parts with no mesh (engine maps, gear ratios, ECU) are physics only.
///
/// `final_parts`: the parts of the pair's `.carfinalstate` (load.rs
/// final_state). They are used only when the mech preset fits nothing
/// itself (no f9 / f10): the case of cars built by third-party tools, whose
/// body kits exist nowhere else. Stock cars that do ship a final state keep
/// the preset rules above - their final states list rival bodies together
/// (the Dallara coupe's names coupe, targa AND spider), so they are not a
/// clean fitment list on their own.
pub fn fitted_parts(vfs: &mut Vfs, mech: Option<&str>, visual: Option<&str>,
                    final_parts: &[String]) -> Vec<FittedPart> {
    // (path, forced)
    let mut want: Vec<(String, bool)> = Vec::new();
    let mut designs: Vec<(String, String)> = Vec::new();
    let mut banned: HashSet<String> = HashSet::new();
    if let Some(m) = mech {
        let d = vfs.read(m);
        fields(&d, 0, d.len(), |f, v| {
            let Some(r) = v.range() else { return };
            match f {
                10 => want.insert(want.iter().take_while(|w| w.1).count(), (text(&d, r), true)),
                9 => want.push((text(&d, r), false)),
                11 => {
                    banned.insert(base_name_lower(&text(&d, r)));
                }
                _ => {}
            }
        });
    }
    if want.is_empty() {
        want.extend(final_parts.iter().map(|p| (p.clone(), false)));
    }
    if let Some(vp) = visual {
        let d = vfs.read(vp);
        fields(&d, 0, d.len(), |f, v| {
            let (10, Some((a, b))) = (f, v.range()) else { return };
            let (mut part, mut design) = (None, None);
            fields(&d, a, b, |sf, sv| match (sf, sv.range()) {
                (1, Some(r)) => part = Some(text(&d, r)),
                (2, Some(r)) => design = Some(text(&d, r)),
                _ => {}
            });
            if let (Some(p), Some(dz)) = (part, design) {
                designs.push((base_name_lower(&p), dz));
            }
        });
    }
    let mut out: Vec<FittedPart> = Vec::new();
    // carparts fitted so far, and the carparts they rule out
    let mut chosen: HashSet<String> = HashSet::new();
    let mut ruled_out: HashSet<String> = HashSet::new();
    for (path, forced) in want {
        let low = base_name_lower(&path);
        let design = designs.iter().find(|(p, _)| *p == low).map(|(_, dz)| dz.clone());
        if banned.contains(&low) || out.iter().any(|p| base_name_lower(&p.compat) == low) {
            continue;
        }
        let Some(ck) = vfs.find(&path, None) else {
            eprintln!("part {path}: not found");
            continue;
        };
        let c = read_compat(&vfs.read(&ck));
        let (Some(part), Some(bone)) = (c.part, c.bone) else { continue };
        let Some(pk) = vfs.find(&part, None) else { continue };
        let pd = vfs.read(&pk);
        let mut mesh = None;
        let mut not_allowed: Vec<String> = Vec::new();
        fields(&pd, 0, pd.len(), |f, v| match (f, v.range()) {
            (5, Some(r)) => {
                mesh.get_or_insert_with(|| text(&pd, r));
            }
            (31, Some(r)) => not_allowed.push(base_name_lower(&text(&pd, r))),
            _ => {}
        });
        let me = base_name_lower(&part);
        if !forced {
            // a default option yields to anything already fitted that rules
            // it out, or that already replaces its bone
            let clash = ruled_out.contains(&me)
                || not_allowed.iter().any(|n| chosen.contains(n))
                || (!c.additive && out.iter().any(|p| !p.additive && p.bone == bone));
            if clash {
                continue;
            }
        }
        chosen.insert(me);
        ruled_out.extend(not_allowed);
        let Some(mesh) = mesh.and_then(|m| vfs.find(&m, None)) else { continue };
        let design = design.and_then(|d| vfs.find(&d, None));
        out.push(FittedPart { compat: ck, mesh, bone, additive: c.additive, local: c.local,
                              design });
    }
    out
}

/// Per skeleton bone: is it hidden by a replacement part? A bone is hidden
/// when it, or any ancestor, is a replacement part's attachment bone.
pub fn hidden_bones(parts: &[FittedPart], names: &[String], parents: &[Option<usize>]) -> Vec<bool> {
    let roots: HashSet<&str> = parts.iter().filter(|p| !p.additive).map(|p| p.bone.as_str()).collect();
    (0..names.len())
        .map(|mut i| {
            let mut guard = 0;
            loop {
                if roots.contains(names[i].as_str()) {
                    return true;
                }
                match parents[i] {
                    Some(p) if guard < 256 => {
                        i = p;
                        guard += 1;
                    }
                    _ => return false,
                }
            }
        })
        .collect()
}

/// Drop every triangle that has a vertex skinned to a hidden bone, batch by
/// batch so each batch's index range stays contiguous. Returns triangles
/// removed. `bone_idx` / `bone_wgt` are 4 f32 per vertex, car-wide indices.
pub fn hide_triangles(
    indices: &mut Vec<u32>,
    batches: &mut [crate::asset::Batch],
    bone_idx: &[u8],
    bone_wgt: &[u8],
    hidden: &[bool],
) -> usize {
    if bone_idx.is_empty() || !hidden.iter().any(|h| *h) {
        return 0;
    }
    let f = |b: &[u8], i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
    let nv = bone_idx.len() / 16;
    let gone: Vec<bool> = (0..nv)
        .map(|v| {
            let mut w = 0.0;
            for k in 0..4 {
                let bi = f(bone_idx, v * 4 + k) as usize;
                if hidden.get(bi).copied().unwrap_or(false) {
                    w += if bone_wgt.len() >= nv * 16 { f(bone_wgt, v * 4 + k) } else { 1.0 };
                }
            }
            w > 0.5
        })
        .collect();
    if !gone.iter().any(|g| *g) {
        return 0;
    }
    let mut out: Vec<u32> = Vec::with_capacity(indices.len());
    let mut removed = 0;
    for b in batches.iter_mut() {
        let s = b.start as usize;
        let e = (s + b.count as usize).min(indices.len());
        let start = out.len();
        for t in indices[s.min(e)..e].chunks_exact(3) {
            if t.iter().any(|&v| gone.get(v as usize).copied().unwrap_or(false)) {
                removed += 1;
            } else {
                out.extend_from_slice(t);
            }
        }
        b.start = start as u32;
        b.count = (out.len() - start) as u32;
    }
    *indices = out;
    removed
}
