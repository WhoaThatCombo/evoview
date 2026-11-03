//! Finding the player's own Assetto Corsa EVO install.
//!
//! evoview ships no game content. Tyres, paints, the studio cubemaps and the
//! shared track textures all live in the player's `content.kspkg`, and this
//! module is how we find it. Redistributing those files would mean shipping
//! Kunos' assets, which is not ours to do.
//!
//! Resolution order, first hit wins:
//!   1. `EVOVIEW_KSPKG`            explicit override, a file or a folder
//!   2. `EVOVIEW_ASSETS`           extracted content roots (folders), legacy
//!   3. the Steam library folders   parsed from libraryfolders.vdf
//!   4. a short list of default install paths

use std::path::{Path, PathBuf};

const GAME_DIR: &str = "Assetto Corsa EVO";

fn kspkg_in(dir: &Path) -> Option<PathBuf> {
    for cand in [
        dir.join("content.kspkg"),
        dir.join(GAME_DIR).join("content.kspkg"),
        dir.join("steamapps").join("common").join(GAME_DIR).join("content.kspkg"),
    ] {
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

/// Steam records every library root in `libraryfolders.vdf`. Parsing the
/// handful of `"path"` lines is enough - no VDF library needed, and it keeps
/// working when the game lives on a second drive, which is the common case.
fn steam_libraries() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let mut steam: Vec<PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("ProgramFiles(x86)") {
        steam.push(PathBuf::from(p).join("Steam"));
    }
    if let Ok(p) = std::env::var("ProgramFiles") {
        steam.push(PathBuf::from(p).join("Steam"));
    }
    for s in &steam {
        let vdf = s.join("steamapps").join("libraryfolders.vdf");
        let Ok(text) = std::fs::read_to_string(&vdf) else { continue };
        for line in text.lines() {
            let line = line.trim();
            if !line.starts_with("\"path\"") {
                continue;
            }
            // "path"		"D:\\SteamLibrary"
            if let Some(v) = line.split('"').nth(3) {
                roots.push(PathBuf::from(v.replace("\\\\", "\\")));
            }
        }
        roots.push(s.clone());
    }
    roots
}

/// The player's `content.kspkg`, or `None` if the game cannot be found.
pub fn find_content_kspkg() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("EVOVIEW_KSPKG") {
        let p = PathBuf::from(&v);
        if p.is_file() {
            return Some(p);
        }
        if let Some(f) = kspkg_in(&p) {
            return Some(f);
        }
        // Falling through silently here would look like the override worked
        // while a different install was quietly used instead.
        eprintln!("EVOVIEW_KSPKG={v} has no content.kspkg - searching Steam instead");
    }
    for lib in steam_libraries() {
        if let Some(f) = kspkg_in(&lib) {
            return Some(f);
        }
    }
    let mut guesses: Vec<PathBuf> = Vec::new();
    for drive in ["C", "D", "E", "F", "G", "H"] {
        guesses.push(PathBuf::from(format!("{drive}:\\SteamLibrary")));
        guesses.push(PathBuf::from(format!("{drive}:\\Games")));
        guesses.push(PathBuf::from(format!(
            "{drive}:\\Program Files (x86)\\Steam"
        )));
    }
    guesses.into_iter().find_map(|g| kspkg_in(&g))
}

/// What to tell the user when the game is nowhere to be found. Shared assets
/// missing is not fatal - the car still renders - so this is a warning, not an
/// error, and it says exactly how to fix it.
pub fn missing_note() -> String {
    format!(
        "shared assets: no {GAME_DIR} install found.\n  \
         Tyres, paints and the studio environment come from your own copy of the game.\n  \
         Point evoview at it with  EVOVIEW_KSPKG=<path to content.kspkg>  or  --base <path>."
    )
}
