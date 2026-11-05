//! Where the viewer had to guess.
//!
//! Every place that substitutes a default, a name heuristic or a fallback
//! for car data the game would read differently records it here, so a
//! conversion that only LOOKS right in evoview can be told apart from one
//! the game draws the same way. `--check` lists every record (check
//! `guesses`); with `--strict` (or EVOVIEW_STRICT=1) the parts that relied
//! on a guess are drawn with warning stripes, and heuristics the game has no
//! equivalent of are switched off.
//!
//! Ids follow the guess inventory: T = textures, P = material params,
//! S = shading model, L = paint / livery, G = geometry, A = alpha / blend.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static ON: AtomicBool = AtomicBool::new(false);
static GUESSES: Mutex<Vec<Guess>> = Mutex::new(Vec::new());
/// Set while the viewer handles its OWN materials (contact shadow) or the
/// stock driver: their guesses say nothing about the car being checked.
static MUTED: AtomicBool = AtomicBool::new(false);

pub fn mute(v: bool) {
    MUTED.store(v, Ordering::Relaxed);
}

#[derive(Clone, Debug, PartialEq)]
pub struct Guess {
    pub id: &'static str,
    /// the material (file name) or part the guess applies to
    pub what: String,
    pub detail: String,
}

/// Strict rendering requested (`--strict` or EVOVIEW_STRICT=1).
pub fn on() -> bool {
    ON.load(Ordering::Relaxed) || crate::envc::var_os("EVOVIEW_STRICT").map_or(false, |v| v != "0")
}

pub fn set_on(v: bool) {
    ON.store(v, Ordering::Relaxed);
}

/// Record a guess (once per id + subject + detail).
pub fn note(id: &'static str, what: &str, detail: impl Into<String>) {
    if MUTED.load(Ordering::Relaxed) {
        return;
    }
    let g = Guess { id, what: what.to_string(), detail: detail.into() };
    let mut v = GUESSES.lock().unwrap();
    if !v.contains(&g) {
        v.push(g);
    }
}

/// Did anything about this material (or part) have to be guessed?
pub fn touches(what: &str) -> bool {
    GUESSES.lock().unwrap().iter().any(|g| g.what == what && !info_only(g.id))
}

pub fn all() -> Vec<Guess> {
    GUESSES.lock().unwrap().clone()
}

/// Reloads (livery switch, file watch) start from a clean slate.
pub fn clear() {
    GUESSES.lock().unwrap().clear();
}

/// Guesses that are recorded for the report but do not change the look
/// (the game skips the same placeholder, for instance).
pub fn info_only(id: &str) -> bool {
    matches!(id, "T8")
}

/// `--check` on a track: every guess as a record (same shape as the car
/// report's records).
pub fn write_report(out: &std::path::Path, name: &str) {
    let gs = all();
    let mut ids: Vec<&'static str> = gs.iter().map(|g| g.id).collect();
    ids.sort();
    ids.dedup();
    let mut recs = Vec::new();
    for id in &ids {
        let mut whats: Vec<String> = gs.iter().filter(|g| g.id == *id)
            .map(|g| if g.detail.is_empty() { g.what.clone() } else { format!("{} ({})", g.what, g.detail) }).collect();
        whats.sort();
        whats.dedup();
        let n = whats.len();
        whats.truncate(12);
        recs.push(serde_json::json!({"car": name, "check": format!("guess_{id}"),
            "severity": if info_only(id) { "info" } else { "warn" },
            "detail": format!("{}: {n} - {}{}", describe(id), whats.join(", "), if n > 12 { ", ..." } else { "" })}));
        println!("check {:5} guess_{:<6} {} ({n})", if info_only(id) { "info" } else { "warn" }, id, describe(id));
    }
    let shown = ids.iter().filter(|id| !info_only(id)).count();
    recs.insert(0, serde_json::json!({"car": name, "check": "guesses", "severity": if shown == 0 { "ok" } else { "warn" },
        "detail": format!("{shown} kind(s) of guess")}));
    let _ = std::fs::write(out, serde_json::to_string_pretty(&serde_json::json!({"findings": recs})).unwrap_or_default());
    println!("wrote {}", out.display());
}

/// One line on what a guess id means.
pub fn describe(id: &str) -> &'static str {
    match id {
        "T2" => "texture found but could not be decoded; the slot was dropped",
        "T5" => "block-compressed texture with unaligned top mips; started at a smaller mip",
        "T7" => "16-bit float texture (format 9)",
        "T8" => "slot points at an editor / default_material placeholder (skipped, as the game does)",
        "T10" => "texture name matched several files; picked one by a path guess",
        "T11" => "albedo texture missing; the flat base colour is drawn instead",
        "T12" => "normal / AO map missing or failed; flat stand-in",
        "T13" => "roughness / global-AO / clear-coat map missing or failed; the constant is used",
        "T16" => "splat map missing or failed; only the base layer shows",
        "T17" => "emissive / function / dirt map missing or failed",
        "P3" => "no base colour at all; neutral grey",
        "P4" => "albedo tinted by Base_Basecolor because of its file name",
        "P5" => "no Roughness param; a per-shader default",
        "S1" => "shader template the viewer does not model for cars; generic PBR",
        "S3" => "no splat map; the Red layer's map is taken as the albedo",
        "S4" => "material name looks like track terrain (ignored on cars)",
        "A1" => "blend mode not 1-4; transparency guessed from the name or maps",
        "A2" => "file name says glass / transp but the shader is not glass",
        "A3" => "cutout with no a2cCutoff; 0.5 assumed",
        "A5" => "glass with no tint colour; a grey quarter-opaque default",
        "TR20" => "track haze density unit unknown; evoview's scale follows the track's ratio to Spa",
        "SKY4" => "cloud noise channel roles and density scale are guessed, not decoded from the game's cloud shader",
        "TR50" => "camera-facing cards turned to the view direction; the game's facing rule is not decoded",
        "SKY5" => "how the weather map is laid over the world (200 x 30 km, wind scroll) is guessed",
        "TR63" => "Fence / Grass AO map read the way the Subsurface shader reads it",
        "TR60" => "DynamicTrack road from its own maps; the vertex variation weight is a constant",
        "TL1" => "track spot lights are taken to shine along their local -Y",
        "TL2" => "track lamp intensity units are unknown; EVOVIEW_LAMP_K scales them",
        "SKY1" => "sun / sky colour and brightness come from a transmittance model, not the game's atmosphere",
        "SKY2" => "the scene average the auto exposure adapts to is modelled, not measured",
        "SKY3" => "track emissives are divided back by most of the night exposure",
        "G1" => "material not found; grey fallback (glass turns solid)",
        "TR3" => "track texture missing or undecodable; the slot was dropped",
        "TR5" => "road with no albedo; the shared asphalt map was substituted",
        "TR6" => "texture adopted from a sibling file by name (the material names none)",
        "TR7" => "fence cutout taken from a <albedo>_alpha sibling the material does not name",
        "TR12" => "track surface with no albedo and no base colour; grey",
        "TR14" => "track surface with no roughness; 0.8",
        "TR15" => "splat layer with no roughness; 0.9",
        "TR17" => "WorldAlign layer drawn triplanar",
        "TR18" => "Splatmap_MultiplyBaseColor ignored (layers replace)",
        "TR19" => "Splatmap_UseHeightBlend approximated by a contrast curve",
        "TR20" => "splat material with no weight source; Red layer everywhere",
        "TR22" => "glows only because its file name says emissive",
        "TR25" => "track shader template the viewer does not model; generic surface",
        "TR26" => "DynamicTrackMultilayer road drawn as a generic surface (runtime layers not modelled)",
        "TR41" => "DynamicTrack road drawn from a single asphalt map",
        "TR44" => "foliage decided by the file name, not the shader",
        "TR45" => "blendMode 3 (translucent) drawn as a cutout",
        _ => "guess",
    }
}
