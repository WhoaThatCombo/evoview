//! The driven car's sound on a track (`--drive-car`): the game's own FMOD
//! events, fed the game's own inputs from the drive model (trackdrive.rs).
//!
//! ⚠ What made the replay engine (audio.rs) sound wrong: it fed the engine
//! event only `speed` and `gear`, after a probe that guessed the name "rpm"
//! concluded there was no engine-speed input. There is - it is `rpms` - and
//! the event also takes throttle_position, torque, limiter, boost, gear_up /
//! gear_dn, ignition and more (`--fmodprobe-all` lists every event's real
//! parameters). The car bank's events also lean on the shared banks (turbo,
//! transmission, gearbox, backfire ...), so those are loaded too.
//!
//! Events used, all 3D at the car with the camera as listener:
//!   evo_cars/<model>/engine_ext / engine_int  (outside / onboard mix)
//!   wheel/wheel_ext / wheel_int               (rolling, slip, surface)
//!   parts/brakes/brakes_street                (brake noise)
//!   environment/wind                          (wind at speed)
//!   environment/track_ambience                (2D bed)

use std::path::Path;

use crate::fmodsys::{Fmod, Inst};

/// Shared banks the car's events can reference, in load order. free_roam
/// (100 MB) and gui are left out.
const SHARED: &[&str] = &["wheel", "wheel_hub", "turbo", "transmission", "gearbox", "brakes", "backfire",
                          "bodywork", "damage", "environment", "horn", "indicators", "effect_chain"];

/// One frame of input, in the game's units.
#[derive(Clone, Copy, Debug, Default)]
pub struct Feed {
    pub rpm: f32,
    pub throttle: f32,
    pub brake: f32,
    /// -1 reverse, 0 neutral, 1.. gears
    pub gear: i32,
    pub speed_kmh: f32,
    pub torque: f32,
    pub boost: f32,
    pub limiter: bool,
    /// tyre slip angle (rad, largest axle) and longitudinal use (0..2)
    pub slip: f32,
    pub spin: f32,
    /// 1 asphalt .. lower off the road
    pub grip: f32,
    pub load: f32,
    pub wheel_radius: f32,
    pub tc_active: bool,
    pub abs_active: bool,
    /// world placement of the car and the listener (right-handed)
    pub car_pos: [f32; 3],
    pub car_vel: [f32; 3],
    pub car_fwd: [f32; 3],
    pub car_up: [f32; 3],
    pub cam_pos: [f32; 3],
    pub cam_fwd: [f32; 3],
    pub cam_up: [f32; 3],
    pub onboard: bool,
    /// engine load, % of the torque curve's peak (negative on the overrun)
    pub load_pct: f32,
    pub horn: bool,
}

pub struct CarSound {
    pub fmod: Fmod,
    engine_ext: Option<Inst>,
    engine_int: Option<Inst>,
    wheel_ext: Option<Inst>,
    wheel_int: Option<Inst>,
    brakes: Option<Inst>,
    wind: Option<Inst>,
    ambience: Option<Inst>,
    /// the car's other parts (AudioEventDataType, instance): bodywork 7,
    /// turbo 8, backfire 9, wheel hub 11, gearbox 12, transmission 13,
    /// horn 14, indicators 16
    parts: Vec<(u32, Inst)>,
    last_gear: i32,
    shift_t: f32,
    shift_up: bool,
    pub muted: bool,
    pub volume: f32,
    /// the opponents' engines (race.rs)
    pub ai: AiEngines,
    pkg: std::path::PathBuf,
    model: String,
    /// the player's model's exterior engine event
    engine_event: Option<String>,
}

/// The engine's inputs on one instance (the player's ext / int, an AI's).
fn engine_params(fm: &Fmod, i: Inst, f: &Feed, shift: Option<bool>, d: f32, vol: f32) {
    let gear = f.gear.max(0) as f32;
    fm.pset(i, "rpms", f.rpm);
    fm.pset(i, "throttle_position", f.throttle);
    fm.pset(i, "gear", gear);
    fm.pset(i, "speed", f.speed_kmh.abs());
    fm.pset(i, "drivetrain_speed", f.speed_kmh.abs());
    // torque is the engine's load in % (-200..200: the overrun is
    // negative), not N m - in N m every car sat pinned at 200
    fm.pset(i, "torque", f.load_pct.clamp(-200.0, 200.0));
    fm.pset(i, "limiter", if f.limiter { 1.0 } else { 0.0 });
    fm.pset(i, "boost", f.boost.clamp(0.0, 3.0));
    fm.pset(i, "bov", if f.throttle < 0.1 && f.boost > 0.3 { 1.0 } else { 0.0 });
    fm.pset(i, "gear_up", if shift == Some(true) { 1.0 } else { 0.0 });
    fm.pset(i, "gear_dn", if shift == Some(false) { 1.0 } else { 0.0 });
    fm.pset(i, "ignition", crate::envc::var("EVOVIEW_SND_IGN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0));
    fm.pset(i, "traction_control", if f.tc_active { 1.0 } else { 0.0 });
    fm.pset(i, "brake_position", f.brake);
    fm.pset(i, "engine_distance", d);
    fm.p3d(i, f.car_pos, f.car_vel, f.car_fwd, f.car_up);
    fm.pvol(i, vol);
}

/// A model's car bank out of the game's package (empty: none).
fn car_bank(pkg: &Path, model: &str) -> Vec<u8> {
    let Ok(p) = crate::pkg::Package::open_filtered(pkg, &[SFX_DIR]) else { return Vec::new() };
    let key = format!("{SFX_DIR}{}.bank", model.to_ascii_lowercase());
    p.by_path.get(&key).copied().map(|e| p.read(e)).unwrap_or_default()
}

const SFX_DIR: &str = "content\\sfx\\";

/// One AI car's engine: its exterior engine event while it is among the
/// nearest (EVOVIEW_AI_SOUND_N, default 5, within EVOVIEW_AI_SOUND_M m,
/// default 300) - started when it gets in, stopped (with the event's own
/// fade) a second after it drops out, so a car on the edge does not flap.
#[derive(Default)]
struct AiVoice {
    inst: Option<Inst>,
    /// seconds it has been out of the nearest set
    out_t: f32,
    last_gear: i32,
    shift_t: f32,
    shift_up: bool,
}

#[derive(Default)]
pub struct AiEngines {
    voices: Vec<AiVoice>,
    /// engine_ext event per AI model (None: no bank / no such event)
    events: Vec<(String, Option<String>)>,
    /// seconds to the next EVOVIEW_AI_SND_LOG line
    log_t: f32,
    pub starts: u32,
    pub stops: u32,
}

/// One AI car for `CarSound::update_ai`: its model and this frame's feed
/// (placed in the world, like the player's).
pub struct AiCar<'a> {
    pub name: &'a str,
    pub model: &'a str,
    pub feed: Feed,
}

impl CarSound {
    /// Load the banks out of `pkg` and start every event. `wav`: render
    /// offline into that file (see Fmod::new_opts).
    /// `events`: the car actor's audio_events (fxdata ActorFx) - the exact
    /// set the game plays for this car; empty = the generic set.
    pub fn new(pkg: &Path, model: &str, events: &[(String, u32, String)], wav: Option<&Path>) -> Result<CarSound, String> {
        let dir = pkg.parent().unwrap_or(Path::new("."));
        let mut p = crate::pkg::Package::open_filtered(pkg, &[r"content\sfx\"])
            .map_err(|e| format!("audio: {e}"))?;
        let mut grab = |n: &str| -> Vec<u8> {
            p.by_path.get(&format!(r"content\sfx\{n}.bank")).copied().map(|e| p.read(e)).unwrap_or_default()
        };
        let master = grab("master_bank");
        let strings = grab("master_bank.strings");
        let mut car = grab(&model.to_ascii_lowercase());
        // a mod car carries its own FMOD project inside its kspkg
        // (content/cars/<id>/sfx/master_bank + strings; the actor's events
        // point there), loaded next to the game's
        let mut mod_strings = Vec::new();
        if car.is_empty() {
            let id = model.to_ascii_lowercase();
            let mp = crate::envc::var_os("USERPROFILE").map(|h| Path::new(&h).join("Saved Games").join("ACE").join("mods").join(format!("{id}.kspkg")));
            let pre = format!(r"content\cars\{id}\sfx\");
            if let Some(Ok(m)) = mp.filter(|p| p.exists()).map(|p| crate::pkg::Package::open_filtered(&p, &[pre.as_str()])) {
                let rd = |n: &str| m.by_path.get(&format!("{pre}{n}.bank")).copied().map(|e| m.read(e)).unwrap_or_default();
                car = rd("master_bank");
                mod_strings = rd("master_bank.strings");
            }
        }
        if car.is_empty() {
            return Err(format!("no sound bank for {model}"));
        }
        let mut fmod = Fmod::new_opts(dir, wav)?;
        let events_of_car = fmod.load_banks(&master, &strings, &car)?;
        if !mod_strings.is_empty() {
            if let Err(e) = fmod.load_extra(&mod_strings) {
                eprintln!("audio: mod strings bank: {e}");
            }
        }
        // EVOVIEW_SND_NO_SHARED: the car bank alone (comparing what the
        // shared banks add)
        let shared: &[&str] = if crate::envc::var_os("EVOVIEW_SND_NO_SHARED").is_some() { &[] } else { SHARED };
        for b in shared {
            let d = grab(b);
            if !d.is_empty() {
                if let Err(e) = fmod.load_extra(&d) {
                    eprintln!("audio: {b}.bank: {e}");
                }
            }
        }
        let pick = |suffix: &str| events_of_car.iter().find(|e| e.ends_with(suffix)).cloned();
        let by_type = |ty: u32| events.iter().find(|e| e.1 == ty).map(|e| e.0.clone());
        let ev = |ty: u32, fallback: Option<String>| by_type(ty).or(fallback);
        let mut parts = Vec::new();
        for (e, ty, _) in events {
            if matches!(ty, 7 | 8 | 9 | 11 | 12 | 13 | 14 | 16) {
                match fmod.spawn(e) {
                    Some(i) => parts.push((*ty, i)),
                    None => eprintln!("audio: {e} did not start"),
                }
            }
        }
        let mut s = CarSound {
            engine_ext: ev(1, pick("engine_ext")).and_then(|e| fmod.spawn(&e)),
            engine_int: ev(0, pick("engine_int")).and_then(|e| fmod.spawn(&e)),
            wheel_ext: ev(3, Some("event:/wheel/wheel_ext".into())).and_then(|e| fmod.spawn(&e)),
            wheel_int: ev(2, Some("event:/wheel/wheel_int".into())).and_then(|e| fmod.spawn(&e)),
            brakes: ev(10, Some("event:/parts/brakes/brakes_street".into())).and_then(|e| fmod.spawn(&e)),
            wind: ev(6, Some("event:/environment/wind".into())).and_then(|e| fmod.spawn(&e)),
            ambience: fmod.spawn("event:/environment/track_ambience"),
            parts,
            fmod,
            last_gear: 1,
            shift_t: 9.0,
            shift_up: true,
            muted: false,
            volume: 1.0,
            ai: AiEngines::default(),
            pkg: pkg.to_path_buf(),
            model: model.to_ascii_lowercase(),
            engine_event: None,
        };
        s.engine_event = ev(1, pick("engine_ext"));
        if let Some(a) = s.ambience {
            s.fmod.pset(a, "sunny", 1.0);
        }
        println!("car sound: {} part event(s) from the actor: {}", s.parts.len(),
                 events.iter().filter(|e| matches!(e.1, 7 | 8 | 9 | 11 | 12 | 13 | 14 | 16))
                     .map(|e| e.0.rsplit('/').next().unwrap_or("")).collect::<Vec<_>>().join(", "));
        println!("car sound: {model} via the game's FMOD (engine {}{}, tyres, brakes, wind, ambience)",
                 if s.engine_ext.is_some() { "ext" } else { "-" },
                 if s.engine_int.is_some() { " + int" } else { "" });
        Ok(s)
    }

    /// Feed one frame and let FMOD mix. `dt` seconds since the last call.
    pub fn update(&mut self, f: &Feed, dt: f32) {
        let fm = &self.fmod;
        // gear changes: gear_up / gear_dn are held for a short pulse
        if f.gear != self.last_gear {
            if self.last_gear != 0 && f.gear > 0 {
                self.shift_up = f.gear > self.last_gear;
                self.shift_t = 0.0;
            }
            self.last_gear = f.gear;
        }
        self.shift_t += dt;
        let shifting = self.shift_t < 0.18;
        let gear = f.gear.max(0) as f32;
        let d = {
            let v = [f.car_pos[0] - f.cam_pos[0], f.car_pos[1] - f.cam_pos[1], f.car_pos[2] - f.cam_pos[2]];
            (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
        };
        let master = if self.muted { 0.0 } else { self.volume };
        for (inst, inside) in [(self.engine_ext, false), (self.engine_int, true)] {
            let Some(i) = inst else { continue };
            engine_params(fm, i, f, shifting.then_some(self.shift_up), d, master * if inside == f.onboard { 1.0 } else { 0.0 });
        }
        let ang = if f.wheel_radius > 0.05 { f.speed_kmh.abs() / 3.6 / f.wheel_radius } else { 0.0 };
        for (inst, inside) in [(self.wheel_ext, false), (self.wheel_int, true)] {
            let Some(i) = inst else { continue };
            fm.pset(i, "speed", f.speed_kmh.abs());
            fm.pset(i, "wheel_angular_speed", ang.min(500.0));
            fm.pset(i, "slip_ratio", (f.spin - 0.9).clamp(0.0, 1.0));
            fm.pset(i, "nd_slip", (f.slip.abs() / 0.14 + (f.spin - 0.9).max(0.0) * 4.0).clamp(0.0, 10.0));
            fm.pset(i, "tyre_load", f.load.clamp(0.0, 1.0));
            // 0 asphalt; off the road the grip drops (grass / gravel)
            fm.pset(i, "surface_type", if f.grip > 0.85 { 0.0 } else if f.grip > 0.65 { 2.0 } else { 3.0 });
            fm.pset(i, "wheel_ext_distance", d);
            fm.p3d(i, f.car_pos, f.car_vel, f.car_fwd, f.car_up);
            fm.pvol(i, master * if inside == f.onboard { 1.0 } else { 0.0 });
        }
        if let Some(i) = self.brakes {
            fm.pset(i, "brake_position", f.brake);
            fm.pset(i, "speed", f.speed_kmh.abs());
            fm.pset(i, "wheel_angular_speed", ang.min(500.0));
            fm.pset(i, "abs", if f.abs_active { 1.0 } else { 0.0 });
            fm.p3d(i, f.car_pos, f.car_vel, f.car_fwd, f.car_up);
            fm.pvol(i, master);
        }
        if let Some(i) = self.wind {
            fm.pset(i, "speed", f.speed_kmh.abs());
            fm.pset(i, "Distance", d);
            fm.pset(i, "air_pressure", 1.0);
            fm.p3d(i, f.car_pos, f.car_vel, f.car_fwd, f.car_up);
            fm.pvol(i, master);
        }
        if let Some(i) = self.ambience {
            fm.pvol(i, master);
        }
        // the parts, each with the inputs its AudioData enum lists
        let up = if shifting && self.shift_up { 1.0 } else { 0.0 };
        let dn = if shifting && !self.shift_up { 1.0 } else { 0.0 };
        let bov = if f.throttle < 0.1 && f.boost > 0.3 { 1.0 } else { 0.0 };
        for &(ty, i) in &self.parts {
            match ty {
                7 => {
                    fm.pset(i, "speed", f.speed_kmh.abs());
                    fm.pset(i, "Distance", d);
                    fm.pset(i, "susp_travel_speed", (f.slip.abs() * 2.0 + f.speed_kmh.abs() / 400.0).clamp(0.0, 1.0));
                }
                8 => {
                    fm.pset(i, "boost", f.boost.clamp(0.0, 3.0));
                    fm.pset(i, "bov", bov);
                    fm.pset(i, "throttle_position", f.throttle);
                    fm.pset(i, "rpms", f.rpm);
                    fm.pset(i, "engine_distance", d);
                }
                9 => {
                    fm.pset(i, "backfire_type", 1.0);
                    fm.pset(i, "rpms", f.rpm);
                    fm.pset(i, "throttle_position", f.throttle);
                    fm.pset(i, "limiter", if f.limiter { 1.0 } else { 0.0 });
                    fm.pset(i, "boost", f.boost.clamp(0.0, 3.0));
                    fm.pset(i, "gear_up", up);
                    fm.pset(i, "engine_distance", d);
                }
                11 => {
                    fm.pset(i, "wheel_hub_state", if f.speed_kmh.abs() > 1.0 { 1.0 } else { 0.0 });
                }
                12 => {
                    fm.pset(i, "throttle_position", f.throttle);
                    fm.pset(i, "gear", gear);
                    fm.pset(i, "gear_up", up);
                    fm.pset(i, "gear_dn", dn);
                    fm.pset(i, "rpms", f.rpm);
                    fm.pset(i, "engine_distance", d);
                }
                13 => {
                    fm.pset(i, "rpms", f.rpm);
                    fm.pset(i, "throttle_position", f.throttle);
                    fm.pset(i, "torque", f.load_pct.clamp(-200.0, 200.0));
                    fm.pset(i, "drivetrain_speed", f.speed_kmh.abs().min(300.0));
                    fm.pset(i, "gear", gear);
                    fm.pset(i, "gear_up", up);
                    fm.pset(i, "gear_dn", dn);
                    fm.pset(i, "engine_distance", d);
                }
                14 => {
                    fm.pset(i, "horn_sound", if f.horn { 1.0 } else { 0.0 });
                    fm.pset(i, "engine_distance", d);
                }
                _ => {}
            }
            fm.p3d(i, f.car_pos, f.car_vel, f.car_fwd, f.car_up);
            fm.pvol(i, master);
        }
        fm.listener(f.cam_pos, f.car_vel, f.cam_fwd, f.cam_up);
        fm.update();
    }

    /// The opponents' engines this frame (before `update`, which mixes):
    /// `cars` in the race's AI order, `ear` the listener's position.
    pub fn update_ai(&mut self, cars: &[AiCar], ear: [f32; 3], dt: f32) {
        let n_max = crate::envc::var("EVOVIEW_AI_SOUND_N").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(5);
        let r_max = crate::envc::var("EVOVIEW_AI_SOUND_M").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(300.0);
        if self.ai.voices.len() != cars.len() {
            self.ai.voices.resize_with(cars.len(), AiVoice::default);
        }
        // each model's event, its bank loaded once (the player's is in)
        for c in cars {
            let m = c.model.to_ascii_lowercase();
            if self.ai.events.iter().any(|e| e.0 == m) {
                continue;
            }
            let ev = if m == self.model {
                self.engine_event.clone()
            } else {
                let bank = car_bank(&self.pkg, &m);
                if bank.is_empty() {
                    None
                } else {
                    match self.fmod.load_extra(&bank) {
                        Ok(evs) => evs.into_iter().find(|e| e.ends_with("engine_ext")),
                        Err(e) => {
                            eprintln!("AI sound {m}: {e}");
                            None
                        }
                    }
                }
            };
            println!("AI sound {m}: {}", ev.as_deref().unwrap_or("no engine event"));
            self.ai.events.push((m, ev));
        }
        let dist = |c: &AiCar| {
            let p = c.feed.car_pos;
            ((p[0] - ear[0]).powi(2) + (p[1] - ear[1]).powi(2) + (p[2] - ear[2]).powi(2)).sqrt()
        };
        // the nearest n, a playing car counted 20% nearer (hysteresis)
        let mut rank: Vec<(f32, usize)> = cars.iter().enumerate()
            .map(|(k, c)| (dist(c) * if self.ai.voices[k].inst.is_some() { 0.8 } else { 1.0 }, k))
            .filter(|(d, _)| *d < r_max)
            .collect();
        rank.sort_by(|a, b| a.0.total_cmp(&b.0));
        rank.truncate(n_max);
        // EVOVIEW_AI_VOLUME: the opponents against the player's car (1 = the game's mix)
        let gain = crate::envc::var("EVOVIEW_AI_VOLUME").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
        let master = if self.muted { 0.0 } else { self.volume * gain };
        let fm = &self.fmod;
        for (k, c) in cars.iter().enumerate() {
            let v = &mut self.ai.voices[k];
            let want = rank.iter().any(|r| r.1 == k);
            v.out_t = if want { 0.0 } else { v.out_t + dt };
            if v.out_t > 1.0 {
                if let Some(i) = v.inst.take() {
                    fm.stop_release(i);
                    self.ai.stops += 1;
                }
            }
            if want && v.inst.is_none() {
                let m = c.model.to_ascii_lowercase();
                if let Some(e) = self.ai.events.iter().find(|e| e.0 == m).and_then(|e| e.1.as_deref()) {
                    v.inst = fm.spawn(e);
                    v.last_gear = c.feed.gear;
                    v.shift_t = 9.0;
                    self.ai.starts += 1;
                }
            }
            let Some(i) = v.inst else { continue };
            let f = &c.feed;
            if f.gear != v.last_gear {
                if v.last_gear != 0 && f.gear > 0 {
                    v.shift_up = f.gear > v.last_gear;
                    v.shift_t = 0.0;
                }
                v.last_gear = f.gear;
            }
            v.shift_t += dt;
            engine_params(fm, i, f, (v.shift_t < 0.18).then_some(v.shift_up), dist(c), master);
        }
        // EVOVIEW_AI_SND_LOG=1: the live voices twice a second
        if crate::envc::var_os("EVOVIEW_AI_SND_LOG").is_some() {
            self.ai.log_t -= dt;
            if self.ai.log_t <= 0.0 {
                self.ai.log_t = 0.5;
                let live: Vec<String> = cars.iter().zip(&self.ai.voices).filter(|(_, v)| v.inst.is_some())
                    .map(|(c, _)| format!("{} {:.0}m {:.0}rpm thr {:.2} load {:.0}% g{} {:.0}km/h",
                                          c.name, dist(c), c.feed.rpm, c.feed.throttle, c.feed.load_pct, c.feed.gear, c.feed.speed_kmh))
                    .collect();
                println!("AISND {} live (starts {} stops {}): {}", live.len(), self.ai.starts, self.ai.stops, live.join(" | "));
            }
        }
    }

    pub fn stop_all(&self) {
        for i in [self.engine_ext, self.engine_int, self.wheel_ext, self.wheel_int, self.brakes, self.wind, self.ambience]
            .into_iter().flatten().chain(self.parts.iter().map(|p| p.1)).chain(self.ai.voices.iter().filter_map(|v| v.inst)) {
            self.fmod.stop(i);
        }
        self.fmod.update();
    }
}

/// `--sound-test <model> --pkg content.kspkg [--wav out.wav]`: a scripted
/// drive rendered offline - idle, a full-throttle pull through the gears,
/// lift, brake - so the mix can be checked without speakers.
pub fn sound_test(pkg: &Path, model: &str, out: &Path) {
    let mut s = match CarSound::new(pkg, model, &[], Some(out)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };
    // one FMOD block per update (1024 frames at 48 kHz)
    let dt = s.fmod.block_seconds();
    let gears = [3.3f32, 2.1, 1.5, 1.15, 0.95];
    let (fd, r) = (3.7f32, 0.32f32);
    let mut v = 0.0f32; // m/s
    let mut gear = 1i32;
    let mut t = 0.0f32;
    let car = [0.0f32, 0.5, 0.0];
    while t < 16.0 {
        let phase_throttle = if t < 2.0 { 0.0 } else if t < 11.0 { 1.0 } else { 0.0 };
        let brake = if t > 12.0 && t < 15.0 { 1.0 } else { 0.0 };
        let ratio = gears[(gear - 1) as usize] * fd;
        let mut rpm = (v / r * 60.0 / std::f32::consts::TAU * ratio).max(900.0);
        if t >= 2.0 && t < 11.0 && v < 3.0 {
            rpm = rpm.max(3500.0);
        }
        if rpm > 6800.0 && (gear as usize) < gears.len() {
            gear += 1;
        } else if gear > 1 && rpm < 2500.0 && phase_throttle < 0.5 {
            gear -= 1;
        }
        let acc = phase_throttle * 9.0 / ratio.max(1.0) * 3.0 - brake * 9.0 - 0.02 * v;
        v = (v + acc * dt).max(0.0);
        let f = Feed {
            rpm: rpm.min(7000.0),
            throttle: phase_throttle,
            brake,
            gear,
            speed_kmh: v * 3.6,
            torque: if phase_throttle > 0.0 { 420.0 } else { -60.0 },
            load_pct: if phase_throttle > 0.0 { 100.0 } else { -25.0 },
            boost: if phase_throttle > 0.0 { (rpm / 4500.0).min(1.0) * 1.1 } else { 0.0 },
            limiter: rpm >= 6990.0,
            slip: 0.0,
            spin: 0.3,
            grip: 1.0,
            load: 0.5,
            wheel_radius: r,
            car_pos: car,
            car_vel: [0.0, 0.0, -v],
            car_fwd: [0.0, 0.0, -1.0],
            car_up: [0.0, 1.0, 0.0],
            // a chase camera 6 m behind
            cam_pos: [0.0, 1.8, 6.0],
            cam_fwd: [0.0, -0.2, -1.0],
            cam_up: [0.0, 1.0, 0.0],
            ..Default::default()
        };
        s.update(&f, dt);
        t += dt;
    }
    s.stop_all();
    s.fmod.release();
    println!("sound test: {} written", out.display());
}
