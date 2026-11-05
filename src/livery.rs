//! Liveries: a car's visual presets (trims / skins), listed and switched at
//! runtime.
//!
//! A livery IS a `presets/*.visualcarpreset`. Its fields that matter here:
//!   (VisualCarPresetData, Customization.proto in the exe)
//!   f1 brand   f2 description ("296 GT3 Visual 1")
//!   f3 note - the name the UI shows ("296 GT3 Presentation", "44 Canada")
//!   f13 race number
//!
//! Switching does not duplicate any of the per-car selection logic (paints,
//! designs, rim style, tyre brand, body-kit parts): it sets EVOVIEW_VISUAL -
//! the one knob every one of those functions in load.rs already reads - and
//! runs `load::load` again on a Vfs that is kept open, with its decoded
//! textures cached. The GPU side keeps its uploaded textures too
//! (`material::TexCache` is keyed by content, not index), so a switch only
//! re-reads protobuf and re-uploads geometry.

use crate::load::{Loaded, Vfs};
use crate::proto::{fields, text};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Livery {
    /// file stem, what EVOVIEW_VISUAL takes (preset_296gt3_visual_1)
    pub stem: String,
    /// the game's UI name (f3), else the internal name (f2), else the stem
    pub display: String,
    /// race number (f13), when there is one
    pub number: Option<u64>,
}

pub(crate) fn stem_of(key: &str) -> String {
    key.replace('\\', "/")
        .rsplit('/')
        .next()
        .unwrap_or(key)
        .to_lowercase()
        .trim_end_matches(".visualcarpreset")
        .to_string()
}

/// Every visual preset of the car in `vfs`, in the order load.rs falls back
/// through (sorted by key: the first one is the default livery).
pub fn list(vfs: &mut Vfs) -> Vec<Livery> {
    let keys = vfs.list("presets", "visualcarpreset");
    let mut descr: Vec<String> = Vec::new();
    let mut out: Vec<Livery> = keys.iter()
        .map(|k| {
            let d = vfs.read(k);
            let (mut internal, mut ui, mut number) = (String::new(), String::new(), None);
            fields(&d, 0, d.len(), |f, v| match (f, v) {
                (2, crate::proto::Val::Bytes(a, b)) => internal = text(&d, (a, b)),
                (3, crate::proto::Val::Bytes(a, b)) => ui = text(&d, (a, b)),
                (13, crate::proto::Val::Varint(n)) => number = Some(n),
                _ => {}
            });
            let stem = stem_of(k);
            descr.push(internal.trim().to_string());
            let display = [ui, internal]
                .into_iter()
                .map(|s| s.trim().to_string())
                .find(|s| !s.is_empty())
                .unwrap_or_else(|| stem.clone());
            Livery { stem, display, number }
        })
        .collect();
    // AC1 cars converted to mods (CM-style: one visual preset per AC1 skin)
    // call every skin's note "Livery" and keep the skin's name in f2
    // ("DGFX Vinyl", "Falken"). A note shared by several presets names
    // nothing, so those show the description instead.
    let names: Vec<String> = out.iter().map(|l| l.display.clone()).collect();
    for (i, l) in out.iter_mut().enumerate() {
        let dup = names.iter().filter(|n| **n == l.display).count() > 1;
        if dup && !descr[i].is_empty() {
            l.display = descr[i].clone();
        }
    }
    out
}

/// A mod ships `<package stem>.json` beside its `.kspkg`, and its
/// `"cars": [{"name": "preset_..._mech_N"}]` is the MECH preset the game
/// lists the car as (one entry per preset the mod offers). Default
/// EVOVIEW_MECH to the first entry that is one of this car's mech presets,
/// so a mod whose package also holds other presets opens on the one the game
/// shows - not on whichever sorts first. An explicit EVOVIEW_MECH wins.
fn mod_json_preset(pkg: &Path, vfs: &mut Vfs) {
    if crate::envc::var("EVOVIEW_MECH").map_or(false, |v| !v.trim().is_empty()) {
        return;
    }
    let Ok(text) = std::fs::read_to_string(pkg.with_extension("json")) else { return };
    let Ok(j) = serde_json::from_str::<serde_json::Value>(&text) else { return };
    let mechs: Vec<String> = vfs.list("presets", "mechanicalcarpreset").iter()
        .map(|k| crate::load::base_name_lower(k).trim_end_matches(".mechanicalcarpreset").to_string())
        .collect();
    let names = j.get("cars").and_then(|c| c.as_array()).cloned().unwrap_or_default();
    for c in names {
        let Some(n) = c.get("name").and_then(|n| n.as_str()) else { continue };
        let n = n.trim().to_lowercase();
        let n = n.trim_end_matches(".mechanicalcarpreset");
        if mechs.iter().any(|m| m == n) {
            println!("mod preset: {n} (from {})", pkg.with_extension("json").display());
            crate::envc::set_var("EVOVIEW_MECH", n);
            return;
        }
    }
}

/// The livery load.rs would pick right now (EVOVIEW_VISUAL, else the first).
fn current_index(list: &[Livery]) -> usize {
    let want = crate::envc::var("EVOVIEW_VISUAL").unwrap_or_default().to_lowercase();
    let want = want.trim().trim_end_matches(".visualcarpreset");
    list.iter().position(|l| l.stem == want).unwrap_or(0)
}

/// A car kept open so its liveries can be switched.
pub struct Switcher {
    vfs: Vfs,
    pub list: Vec<Livery>,
    pub cur: usize,
    /// car id or folder name, for the title
    pub name: String,
}

impl Switcher {
    pub fn from_package(pkg: &Path, car: &str, base: Option<&Path>) -> Result<Self, String> {
        let mut vfs = crate::load::package_vfs(pkg, car, base)?;
        mod_json_preset(pkg, &mut vfs);
        Ok(Self::new(vfs, car.to_string()))
    }

    pub fn from_dir(dir: &Path, asset_roots: &[PathBuf]) -> Result<Self, String> {
        let name = dir.file_name().and_then(|s| s.to_str()).unwrap_or("car").to_string();
        Ok(Self::new(crate::load::car_dir_vfs(dir, asset_roots)?, name))
    }

    fn new(mut vfs: Vfs, name: String) -> Self {
        let list = list(&mut vfs);
        // the livery load.rs would pick (EVOVIEW_VISUAL, else the default of
        // the chosen mech preset, else the first)
        let cur = crate::load::pick_visual_preset(&mut vfs)
            .map(|k| stem_of(&k))
            .and_then(|s| list.iter().position(|l| l.stem == s))
            .unwrap_or_else(|| current_index(&list));
        Switcher { vfs, list, cur, name }
    }

    /// Keep decoded textures between loads (interactive viewer only: a
    /// single shot has no second load to speed up).
    pub fn keep_textures(&mut self) {
        self.vfs.keep_textures();
    }

    /// Load the car wearing the current livery.
    pub fn load(&mut self) -> Result<Loaded, String> {
        if let Some(l) = self.list.get(self.cur) {
            crate::envc::set_var("EVOVIEW_VISUAL", &l.stem);
        }
        let before = crate::perf::phases();
        let r = crate::load::load(&mut self.vfs);
        // EVOVIEW_LIVERY_PERF=1: where a (re)load spent its time
        if crate::envc::var_os("EVOVIEW_LIVERY_PERF").is_some() {
            for (n, s, c) in crate::perf::phases() {
                let (s0, c0) = before.iter().find(|e| e.0 == n).map_or((0.0, 0), |e| (e.1, e.2));
                if s - s0 > 0.002 {
                    println!("  perf {n:32} {:7.1} ms  x{}", (s - s0) * 1000.0, c - c0);
                }
            }
        }
        r
    }

    /// The viewer has these textures (vfs keys) on the GPU already.
    pub fn mark_resident(&mut self, keys: &[String]) {
        self.vfs.mark_resident(keys);
    }

    /// Select livery `i` (wrapping) and load it.
    pub fn select(&mut self, i: isize) -> Result<Loaded, String> {
        let n = self.list.len().max(1) as isize;
        self.cur = i.rem_euclid(n) as usize;
        self.load()
    }

    /// Select by stem (exact) or by 1-based number "3"; None if no match.
    pub fn find(&self, what: &str) -> Option<usize> {
        let w = what.trim().to_lowercase();
        let w = w.trim_end_matches(".visualcarpreset");
        if let Some(i) = self.list.iter().position(|l| l.stem == w) {
            return Some(i);
        }
        w.parse::<usize>().ok().filter(|&n| n >= 1 && n <= self.list.len()).map(|n| n - 1)
    }

    /// "296 GT3 Presentation (preset_296gt3_visual_1) 1/2"
    pub fn label(&self) -> String {
        match self.list.get(self.cur) {
            Some(l) => format!("livery {}/{}: {} ({})", self.cur + 1, self.list.len(), l.display, l.stem),
            None => "no liveries".to_string(),
        }
    }
}

/// What one AI car wears (race.rs): a visual preset and, for variety past
/// the car's few liveries, one of its design's other body colours.
#[derive(Clone, Debug, PartialEq)]
pub struct Look {
    pub visual: String,
    pub paint: Option<String>,
}

/// One look per AI car of a model, in driver order, the same every run:
/// the liveries the player is not wearing first, then those in the design's
/// other body colours, then the player's livery recoloured, the player's
/// exact look last. At most `max` distinct looks (each is its own set of
/// meshes and materials); drivers past that share them round-robin.
/// `salt` rotates the colours so two models do not pick alike; `own` is a
/// livery's own paints (one it already wears is no new look).
pub fn ai_looks(list: &[Livery], player: Option<&str>, paints: &[String], own: &dyn Fn(&str) -> Vec<String>,
                n: usize, max: usize, salt: usize) -> Vec<Look> {
    if list.is_empty() || n == 0 {
        return Vec::new();
    }
    let mine: Vec<&Livery> = list.iter().filter(|l| Some(l.stem.as_str()) == player).collect();
    let others: Vec<&Livery> = list.iter().filter(|l| Some(l.stem.as_str()) != player).collect();
    let mut pool: Vec<Look> = Vec::new();
    let look = |l: &Livery, p: Option<&String>| Look { visual: l.stem.clone(), paint: p.cloned() };
    for l in &others {
        pool.push(look(l, None));
    }
    let k = paints.len();
    for i in 0..k {
        let p = &paints[(i + salt) % k];
        for l in others.iter().chain(mine.iter()) {
            if !own(&l.stem).contains(p) {
                pool.push(look(l, Some(p)));
            }
        }
    }
    for l in &mine {
        pool.push(look(l, None));
    }
    pool.truncate(max.max(1));
    (0..n).map(|i| pool[i % pool.len()].clone()).collect()
}

/// `--list-liveries`: one `stem<TAB>display<TAB>number` line per preset.
pub fn print_list(list: &[Livery]) {
    for l in list {
        println!("{}\t{}\t{}", l.stem, l.display, l.number.map(|n| n.to_string()).unwrap_or_default());
    }
}

/// Commands on stdin, one per line, so a launcher (ACECM) can drive a running
/// viewer without a socket:
///   livery <stem | n>     switch to that livery
///   livery next | prev
///   quit
/// Lines arrive on a channel the render loop polls.
pub fn stdin_commands() -> std::sync::mpsc::Receiver<String> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut line = String::new();
        loop {
            line.clear();
            match stdin.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let l = line.trim();
                    if !l.is_empty() && tx.send(l.to_string()).is_err() {
                        break;
                    }
                }
            }
        }
    });
    rx
}
