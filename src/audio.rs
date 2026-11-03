//! Engine sound: the game's own FMOD, driven by the replay.
//!
//! ⚠ Two earlier attempts are worth remembering as dead ends. Synthesising the
//! note from rpm sounds like a kazoo, because an engine is a specific machine
//! and not a waveform. Crossfading the bank's raw PCM loops by hand is closer
//! but still wrong, because the loops are only raw material: the game runs them
//! through FMOD event graphs with separate interior and exterior layers,
//! filters, curves, distance attenuation and spatialisation.
//!
//! So nothing is reimplemented. The user's own `fmod.dll` / `fmodstudio.dll`
//! play the game's own banks, and the replay supplies the same parameters the
//! game supplies.
//!
//! ⭐ The engine event takes **speed (km/h) and gear**, NOT rpm - there is no
//! rpm parameter on the event and no global parameters at all. The bank works
//! out the note from road speed and the gear it is in, which it can do because
//! the bank is per-car and carries that car's own ratios.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};

use crate::fmodsys::Fmod;

/// Bank bytes read off-thread: (model, master, strings, car).
type Banks = (String, Vec<u8>, Vec<u8>, Vec<u8>);

pub struct Engine {
    fmod: Fmod,
    rx: Receiver<Banks>,
    /// Model whose bank is loaded or on its way, so a car change only triggers
    /// one read of an 80 MB bank.
    pub loaded: String,
    pub muted: bool,
    have_master: bool,
    /// True once an event is actually playing.
    pub live: bool,
    /// The car's two engine events; onboard uses the interior mix.
    ext: String,
    int: String,
    /// Which of the two is currently playing.
    inside: bool,
}

impl Engine {
    /// `dir` is the game folder with the FMOD dlls beside content.kspkg.
    pub fn new(dir: &Path) -> Result<Engine, String> {
        let (_tx, rx) = mpsc::channel();
        drop(_tx);
        Ok(Engine {
            fmod: Fmod::new(dir)?,
            rx,
            loaded: String::new(),
            muted: false,
            have_master: false,
            live: false,
            ext: String::new(),
            int: String::new(),
            inside: false,
        })
    }

    /// Start reading a car's banks. The kspkg read is the slow part (indexing a
    /// 74 GB archive), so it happens on its own thread; the FMOD load itself
    /// happens in `poll` on the main thread.
    pub fn load_car(&mut self, pkg: &Path, model: &str) {
        if model.is_empty() || self.loaded == model {
            return;
        }
        self.loaded = model.to_string();
        let (tx, rx) = mpsc::channel();
        self.rx = rx;
        let (pkg, model, need_master) = (pkg.to_path_buf(), model.to_string(), !self.have_master);
        std::thread::spawn(move || {
            let mut p = match crate::pkg::Package::open(&pkg) {
                Ok(p) => p,
                Err(e) => return eprintln!("audio: {e}"),
            };
            let mut grab = |n: String| -> Vec<u8> {
                p.by_path.get(&format!(r"content\sfx\{n}")).copied()
                    .map(|e| p.read(e)).unwrap_or_default()
            };
            let master = if need_master { grab("master_bank.bank".into()) } else { Vec::new() };
            let strings = if need_master { grab("master_bank.strings.bank".into()) } else { Vec::new() };
            let car = grab(format!("{}.bank", model.to_ascii_lowercase()));
            if car.is_empty() {
                return eprintln!("audio: no bank for {model}");
            }
            let _ = tx.send((model, master, strings, car));
        });
    }

    /// Hand any arrived banks to FMOD and start the engine event. Call once a
    /// frame; it does nothing until a load completes.
    pub fn poll(&mut self) {
        let Ok((model, master, strings, car)) = self.rx.try_recv() else { return };
        match self.fmod.load_banks(&master, &strings, &car) {
            Ok(events) => {
                self.have_master = true;
                let pick = |suffix: &str| events.iter()
                    .find(|e| e.ends_with(suffix)).cloned().unwrap_or_default();
                self.ext = pick("engine_ext");
                self.int = pick("engine_int");
                if self.ext.is_empty() {
                    self.ext = events.first().cloned().unwrap_or_default();
                }
                self.inside = false;
                match self.start(&self.ext.clone()) {
                    Ok(()) => println!("audio: {model} -> {} (+{} interior)",
                                       self.ext, if self.int.is_empty() { "no" } else { "1" }),
                    Err(e) => eprintln!("audio: {e}"),
                }
            }
            Err(e) => eprintln!("audio: bank load: {e}"),
        }
    }

    fn start(&mut self, ev: &str) -> Result<(), String> {
        if ev.is_empty() {
            return Err("no engine event in this bank".into());
        }
        self.fmod.play(ev, &["speed", "gear", "limiter", "boost"])?;
        self.live = true;
        Ok(())
    }

    /// Swap between the interior and exterior mixes. They are separate FMOD
    /// events, not a parameter, so this restarts the instance.
    pub fn set_inside(&mut self, inside: bool) {
        if !self.live || inside == self.inside || self.int.is_empty() {
            return;
        }
        self.inside = inside;
        let ev = if inside { self.int.clone() } else { self.ext.clone() };
        if let Err(e) = self.start(&ev) {
            eprintln!("audio: {e}");
        }
    }

    /// Feed the event what the game would feed it.
    pub fn update(&self, speed_kmh: f32, gear: u8, limiter: f32, gain: f32) {
        if !self.live {
            return;
        }
        self.fmod.set("speed", speed_kmh.clamp(0.0, 500.0));
        self.fmod.set("gear", gear.min(8) as f32);
        self.fmod.set("limiter", limiter.clamp(0.0, 1.0));
        self.fmod.set_volume(if self.muted { 0.0 } else { gain.clamp(0.0, 1.0) });
        self.fmod.update();
    }
}

/// Where the FMOD dlls live, given the content package path.
pub fn game_dir(pkg: &Path) -> PathBuf {
    pkg.parent().unwrap_or(Path::new(".")).to_path_buf()
}
