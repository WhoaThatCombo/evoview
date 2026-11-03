//! The effects half of `--check` (check.rs runs it after the animation half,
//! check_anim.rs, and writes one report): exercise a car's effects without
//! the game and flag problems.
//!
//!     evoview <pkg.kspkg> --car <id> [--base content.kspkg] --check out.json
//!         [--check-sheet sheet.png]
//!
//! Every finding is one record `{car, check, severity, detail}` (the same
//! shape look-anim's animation / suspension check writes, so the two merge
//! into one report). Severity: ok, info, warn, error.
//!
//! Data checks read what the loader resolved (fxdata.rs, material tables);
//! render checks draw the car headless at 320x180 and count pixels that
//! change between two effect states (night studio, so a lamp is visible):
//! every light function on its own, LEDs / shift lights, the dash, brake
//! glow, exhaust flames and dirt. A function the car's data lights that
//! changes no pixels is a broken binding (mask, material or class).

use crate::asset::Car;
use crate::fxdata::*;
use crate::fxstate::{FxState, Indicator};
use crate::lights::Class;
use crate::material::Blend;
use crate::shot::Still;
use serde_json::{json, Value};
use std::path::Path;

const W: u32 = 480;
const H: u32 = 270;
/// a pixel "changed" when a channel moved by more than this (0-255)
const DIFF: i32 = 10;
/// fewer changed pixels than this = nothing visible happened
const MIN_PX: usize = 6;

struct Report {
    car: String,
    recs: Vec<Value>,
    /// the animation check's measured tables
    extra: Value,
}

impl Report {
    fn add(&mut self, check: &str, severity: &str, detail: impl Into<String>) {
        let d = detail.into();
        println!("check {:<5} {:<28} {}", severity, check, d);
        self.recs.push(json!({"car": self.car, "check": check, "severity": severity, "detail": d}));
    }
}

fn diff(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4).zip(b.chunks_exact(4))
        .filter(|(p, q)| (0..3).any(|c| (p[c] as i32 - q[c] as i32).abs() > DIFF))
        .count()
}

fn cam(still: &mut Still, eye: [f32; 3], target: [f32; 3], fov: f32) {
    let spec = format!("{{\"eye\":[{},{},{}],\"target\":[{},{},{}],\"fov\":{}}}",
                       eye[0], eye[1], eye[2], target[0], target[1], target[2], fov);
    if let Ok((vp, e)) = crate::camfile::view(&spec, W as f32 / H as f32) {
        still.vp = vp;
        still.eye = e;
    }
}

fn night() -> FxState {
    FxState { night: true, blink_phase: Some(true), ..FxState::default() }
}

/// Does the data light function `f` anywhere?
fn data_has(fx: &CarFx, f: usize) -> bool {
    fx.light_sys.as_ref().map_or(false, |ls| {
        let s = crate::lights::strongest(ls, f);
        s.0 > 0.0 || s.1 > 0.0
    })
}

pub async fn run(car: Car, car_id: &str, out: &Path, sheet: Option<&Path>,
                 anim: Option<crate::check_anim::AnimCheck>) {
    let mut rep = Report { car: car_id.to_string(), recs: Vec::new(), extra: Value::Null };
    let mut anim_poses = Vec::new();
    if let Some(a) = anim {
        for r in &a.records {
            println!("check {:<5} {:<28} {}", r["severity"].as_str().unwrap_or(""),
                     r["check"].as_str().unwrap_or(""), r["detail"].as_str().unwrap_or(""));
        }
        rep.recs.extend(a.records);
        rep.extra = a.extra;
        anim_poses = a.poses;
    }
    // EVOVIEW_CHECK_ANIM_ONLY=1: the CPU half only (fleet sweeps, no GPU)
    if std::env::var_os("EVOVIEW_CHECK_ANIM_ONLY").is_some() {
        write(&rep, out);
        return;
    }
    std::env::set_var("EVOVIEW_SHOT_ALPHA", "1");
    let mut still = Still::new(car, W, H, 2.5, 0.35, 1.0, None, None, false, None).await;
    let fx: Option<CarFx> = still.car.fx.as_deref().cloned();
    let (lo, hi) = (still.lo, still.hi);
    let mut cells: Vec<(String, Vec<u8>)> = Vec::new();

    // ---------------------------------------------------------- data ----
    let Some(fx) = fx else {
        rep.add("fx_data", "error", "no actor / effect data (bare export?)");
        write(&rep, out);
        return;
    };
    for p in &fx.problems {
        rep.add("fx_data", "warn", p.clone());
    }
    match &fx.light_sys {
        Some(ls) => rep.add("light_system", "ok", format!("{} stage(s), {} shared set(s), {} blinker(s)",
                                                      ls.stages.len(), ls.shared.len(), ls.blinkers.len())),
        None => rep.add("light_system", "error", format!("no .carlightingsystem ({:?})", fx.actor.light_script)),
    }
    for (name, b) in fx.light_sys.as_ref().map(|l| l.blinkers.iter().collect::<Vec<_>>()).unwrap_or_default() {
        rep.add("blink_rate", "info", format!("{name}: on {:.2}s off {:.2}s{}", b.on, b.off,
                                              if b.start_off { " (starts off)" } else { "" }));
    }
    if fx.actor.lights.is_empty() {
        rep.add("lamp_positions", "warn", "actor has no physical lights (no light thrown on the ground)");
    } else {
        rep.add("lamp_positions", "ok", format!("{} physical light(s)", fx.actor.lights.len()));
    }
    // materials
    let mats = &still.materials;
    let mut lamps = 0;
    let mut discs = 0;
    let mut leds = 0;
    let mut glow = 0;
    let mut displays = 0;
    let mut dirt = 0;
    for (i, src) in mats.light_srcs.iter().enumerate() {
        match src.as_ref().map(|s| s.class) {
            Some(Class::Reflector) | Some(Class::Lens) => lamps += 1,
            Some(Class::Disc) => discs += 1,
            Some(Class::Led) => leds += 1,
            Some(Class::MechGlow) => glow += 1,
            Some(Class::Display(_)) => displays += 1,
            _ => {}
        }
        if mats.uniforms.get(i).map_or(false, |u| u.fx_m[1] > 0.5) {
            dirt += 1;
        }
    }
    rep.add("lamp_materials", if lamps > 0 { "ok" } else { "error" },
            format!("{lamps} lamp material(s) with function masks"));
    // fallback / placeholder / unresolved
    let fallback_draws: usize = still.meshes.iter()
        .map(|m| m.draws.iter().filter(|d| d.2 == mats.fallback).count()).sum();
    rep.add("material_fallback", if fallback_draws == 0 { "ok" } else { "warn" },
            format!("{fallback_draws} draw(s) use the fallback material (material not found)"));
    // everything the viewer had to guess (strict.rs): one record per kind,
    // naming the materials, so a converted car that only looks right here
    // shows up before it is tried in the game
    {
        let gs = crate::strict::all();
        let mut ids: Vec<&'static str> = gs.iter().map(|g| g.id).collect();
        ids.sort();
        ids.dedup();
        let shown = ids.iter().filter(|id| !crate::strict::info_only(id)).count();
        rep.add("guesses", if shown == 0 { "ok" } else { "warn" },
                if shown == 0 { "nothing had to be guessed".to_string() }
                else { format!("{shown} kind(s) of guess; run with --strict to see the parts striped") });
        for id in ids {
            let mut whats: Vec<String> = gs.iter().filter(|g| g.id == id)
                .map(|g| if g.detail.is_empty() { g.what.clone() } else { format!("{} ({})", g.what, g.detail) }).collect();
            whats.sort();
            whats.dedup();
            let n = whats.len();
            whats.truncate(10);
            rep.add(&format!("guess_{id}"), if crate::strict::info_only(id) { "info" } else { "warn" },
                    format!("{}: {n} - {}{}", crate::strict::describe(id), whats.join(", "), if n > 10 { ", ..." } else { "" }));
        }
    }
    if !fx.unresolved.is_empty() {
        let mut u = fx.unresolved.clone();
        u.sort();
        u.dedup();
        let n = u.len();
        u.truncate(12);
        rep.add("textures_unresolved", "warn", format!("{n} slot(s): {}", u.join(", ")));
    } else {
        rep.add("textures_unresolved", "ok", "every material texture slot resolved");
    }
    let mut tiny: Vec<String> = Vec::new();
    for m in &still.car.materials {
        for (slot, &ti) in &m.slots {
            if let Some(t) = still.car.textures.get(ti) {
                if t.width * t.height <= 16 && !slot.contains("Mask") {
                    tiny.push(format!("{}:{} {}x{}", m.file, slot, t.width, t.height));
                }
            }
        }
    }
    tiny.sort();
    tiny.dedup();
    rep.add("textures_placeholder", if tiny.is_empty() { "ok" } else { "info" },
            if tiny.is_empty() { "no 1x1 / 4x4 placeholder textures".to_string() }
            else { format!("{} slot(s) bind a tiny texture: {}", tiny.len(), tiny.iter().take(8).cloned().collect::<Vec<_>>().join(", ")) });
    // glass
    let mut opaque = Vec::new();
    for (name, &i) in &mats.by_name {
        if mats.blend.get(i) == Some(&Blend::Glass) {
            let op = mats.uniforms[i].gl_em[0];
            let driver = name.contains("helmet") || name.contains("visor") || name.starts_with("rt_") || name.starts_with("st_");
            if op >= 0.9 && !driver && (name.contains("window") || name.contains("windscreen") || name.contains("glass")) && !name.contains("light") {
                opaque.push(format!("{name} ({op:.2})"));
            }
        }
    }
    rep.add("glass_opacity", if opaque.is_empty() { "ok" } else { "warn" },
            if opaque.is_empty() { "windows are see-through".to_string() } else { format!("near-opaque windows: {}", opaque.join(", ")) });

    // emitters near exhaust tips
    let mut tips: Vec<[f32; 3]> = Vec::new();
    for m in &still.car.meshes {
        if !m.batches.iter().any(|b| b.material.to_lowercase().contains("exhaust")) {
            continue;
        }
        let pos = still.car.floats(m.positions);
        let idx = still.car.u32s(m.indices);
        for b in m.batches.iter().filter(|b| b.material.to_lowercase().contains("exhaust")) {
            for &k in idx.iter().skip(b.start as usize).take(b.count as usize).step_by(3) {
                let k = k as usize * 3;
                if k + 2 < pos.len() {
                    tips.push([pos[k], pos[k + 1], pos[k + 2]]);
                }
            }
        }
    }
    let exhausts: Vec<&EmitterSlot> = fx.emitters.iter().filter(|e| (2..=4).contains(&e.kind)).collect();
    if exhausts.is_empty() {
        rep.add("exhaust_emitters", "warn", "no exhaust particle emitters in the actor");
    }
    for e in &exhausts {
        let d = tips.iter().map(|t| ((t[0] - e.pos[0]).powi(2) + (t[1] - e.pos[1]).powi(2) + (t[2] - e.pos[2]).powi(2)).sqrt())
            .fold(f32::MAX, f32::min);
        let defs = e.defs.len();
        let (sev, msg) = if tips.is_empty() {
            ("info", "no exhaust-material geometry to compare with".to_string())
        } else if d <= 0.25 {
            ("ok", format!("{:.2} m from the nearest exhaust vertex", d))
        } else {
            ("warn", format!("{:.2} m from the nearest exhaust vertex (misplaced?)", d))
        };
        rep.add("exhaust_emitter_placement", sev, format!("{} at ({:.2}, {:.2}, {:.2}), {} backfire def(s): {}",
            e.name, e.pos[0], e.pos[1], e.pos[2], defs, msg));
    }
    if let Some(b) = &fx.backfire {
        rep.add("backfire_script", "ok", format!("rpm {:.0}/{:.0}/{:.0}, visual prob {:?}", b.min_rpms, b.low_rpms, b.high_rpms, b.visual_prob));
    } else {
        rep.add("backfire_script", "info", "no .carexhaustbackfire");
    }
    let tyres = fx.emitters.iter().filter(|e| e.kind == 1).count();
    rep.add("tyre_emitters", if tyres >= 4 { "ok" } else { "warn" }, format!("{tyres} tyre emitter(s)"));
    for d in &fx.emitter_defs {
        if d.material.color_tex.is_none() && d.material.opacity_tex.is_none() {
            rep.add("particle_textures", "warn", format!("{}: no texture resolved", d.path));
        }
    }
    // LEDs
    let led_rules = fx.led_sys.as_ref().map_or(0, |l| l.led_count());
    let rpm_leds: Vec<f32> = fx.led_sys.as_ref().map(|l| l.group(LED_RPM, 0).iter().map(|i| i.threshold).collect()).unwrap_or_default();
    rep.add("led_system", if led_rules == 0 || leds > 0 { "ok" } else { "error" },
            format!("{led_rules} LED rule(s), {} shift light(s) ({:.0}..{:.0} rpm), {leds} LED material(s), limiter {:.0}",
                    rpm_leds.len(), rpm_leds.iter().cloned().fold(f32::MAX, f32::min).min(1e9),
                    rpm_leds.iter().cloned().fold(0.0, f32::max),
                    fx.led_sys.as_ref().map_or(0.0, |l| l.limiter_rpm)));
    // dash
    for (pi, p) in fx.dash.iter().enumerate() {
        let a = FxState { rpm: 3000.0, speed_mps: 50.0 / 3.6, gear: 2, lights: 1, ..FxState::default() };
        let b = FxState { rpm: 6500.0, speed_mps: 180.0 / 3.6, gear: 5, lights: 2, throttle: 1.0, ..FxState::default() };
        let mut unknown = Vec::new();
        let mut changes = 0;
        for e in &p.bindings {
            let (x, y) = (crate::dash::eval_binding(e, &a), crate::dash::eval_binding(e, &b));
            match (x, y) {
                (None, _) => unknown.push(e.chars().take(60).collect::<String>()),
                (Some(x), Some(y)) if x != y => changes += 1,
                _ => {}
            }
        }
        rep.add("dash_page", if p.missing.is_empty() { "ok" } else { "warn" },
                format!("{} ({}x{}): {} font(s), {} image(s), {} binding(s), {} change with speed/rpm/gear{}",
                        crate::load::base_of(&p.html_path), p.width, p.height, p.fonts.len(), p.images.len(),
                        p.bindings.len(), changes,
                        if p.missing.is_empty() { String::new() } else { format!("; missing: {}", p.missing.join(", ")) }));
        if !unknown.is_empty() {
            unknown.sort();
            unknown.dedup();
            rep.add("dash_bindings", "info", format!("page {pi}: {} expression(s) the viewer cannot evaluate (keep authored text): {}",
                                                   unknown.len(), unknown.iter().take(6).cloned().collect::<Vec<_>>().join(" | ")));
        }
    }
    if !fx.actor.displays.is_empty() && displays == 0 {
        rep.add("dash_material", "error", "the actor names displays but no display material was found");
    }

    // -------------------------------------------------------- renders ----
    let base = still.render(&FxState::default());
    {
        let mut mask = 0usize;
        let mut magenta = 0usize;
        let mut black = 0usize;
        for p in base.chunks_exact(4) {
            if p[3] == 0 {
                continue;
            }
            mask += 1;
            if p[0] > 220 && p[1] < 40 && p[2] > 220 {
                magenta += 1;
            }
            if p[0] == 0 && p[1] == 0 && p[2] == 0 {
                black += 1;
            }
        }
        let fm = magenta as f32 / mask.max(1) as f32;
        let fb = black as f32 / mask.max(1) as f32;
        rep.add("pixels_magenta", if fm > 0.001 { "error" } else { "ok" }, format!("{magenta} magenta pixel(s) of {mask} on the car"));
        rep.add("pixels_black", if fb > 0.05 { "warn" } else { "ok" }, format!("{:.1}% pure black (NaN / missing light?)", fb * 100.0));
        cells.push(("studio".into(), base.clone()));
    }
    // ---- the animation check's key poses ----
    if !anim_poses.is_empty() {
        cam(&mut still, [hi.x + (hi.z - lo.z) * 0.55, (lo.y + hi.y) * 0.5 + 1.0, hi.z + (hi.z - lo.z) * 0.45],
            [0.0, (lo.y + hi.y) * 0.4, 0.0], 40.0);
        for (label, pose) in &anim_poses {
            still.set_pose(Some(pose));
            cells.push((label.clone(), still.render(&FxState::default())));
        }
        still.set_pose(None);
    }
    let cy = (lo.y + hi.y) * 0.5;
    let len = hi.z - lo.z;
    let front_eye = [0.9, cy + 0.35, hi.z + len * 0.85];
    let front_tgt = [0.0, cy - 0.1, hi.z - len * 0.2];
    let rear_eye = [-0.9, cy + 0.35, lo.z - len * 0.85];
    let rear_tgt = [0.0, cy - 0.1, lo.z + len * 0.2];
    let side_eye = [hi.x + len * 0.9, cy, 0.0];
    let side_tgt = [0.0, cy - 0.15, 0.0];
    // ---- lights, one function at a time ----
    let groups: [(&str, usize, bool); 15] = [
        ("daylight", FN_DAYLIGHT, true), ("lowbeam", FN_LOWBEAM, true), ("highbeam", FN_HIGHBEAM, true),
        ("tail", FN_REAR, false), ("brake", FN_BRAKE, false), ("rainfog", FN_RAINFOG, false),
        ("reverse", FN_REVERSE, false), ("indicator_lf", FN_LF_IND, true), ("indicator_rf", FN_RF_IND, true),
        ("indicator_lr", FN_LR_IND, false), ("indicator_rr", FN_RR_IND, false),
        ("special_lf", FN_LF_SPECIAL, true), ("special_rf", 12, true), ("special_lr", 13, false),
        ("special_rr", FN_RR_SPECIAL, false),
    ];
    for front in [true, false] {
        if front { cam(&mut still, front_eye, front_tgt, 40.0) } else { cam(&mut still, rear_eye, rear_tgt, 40.0) }
        let off = still.render(&night());
        for (name, f, is_front) in groups {
            if is_front != front {
                continue;
            }
            let st = FxState { force_fn: Some(f), ..night() };
            let on = still.render(&st);
            let n = diff(&off, &on);
            let has = data_has(&fx, f);
            let core = matches!(f, FN_LOWBEAM | FN_REAR | FN_BRAKE | FN_LF_IND | FN_RF_IND | FN_LR_IND | FN_RR_IND | FN_HIGHBEAM);
            // specials the data only lights in the cockpit-light set are
            // interior lamps: nothing to see from outside
            let cockpit_only = has && fx.light_sys.as_ref().map_or(false, |ls| {
                ls.stages.iter().flatten().chain(ls.shared.iter())
                    .filter(|set| set.entries.iter().any(|e| e.function as usize == f && (e.on > 0.0 || e.glass_on > 0.0)))
                    .all(|set| set.category == CAT_COCKPIT)
            });
            let (sev, msg) = match (has, n >= MIN_PX) {
                (true, true) => ("ok", format!("{n} px change")),
                (true, false) if cockpit_only => ("info", format!("a cockpit light (only in the Cockpit_Lights set): {n} px outside")),
                // DRL / reverse / fog / specials: plenty of cars carry the
                // function in the data (the default Main_Lights entry is
                // `daylight`) with no lamp for it - not a binding fault
                (true, false) if !core => ("info", format!("in the data, but no mask area lights ({n} px): this car may have no such lamp")),
                (true, false) => ("warn", format!("the data lights it but no pixels change ({n} px): mask channel empty or no lamp material")),
                (false, true) => ("info", format!("not in .carlightingsystem, but the masks carry it ({n} px)")),
                (false, false) => ("info", "not used by this car".to_string()),
            };
            rep.add(&format!("light_{name}"), sev, msg);
            if matches!(name, "lowbeam" | "brake" | "indicator_lf" | "indicator_lr") {
                cells.push((name.to_string(), on));
            }
        }
    }
    // ---- brake glow, dirt (side) ----
    cam(&mut still, side_eye, side_tgt, 40.0);
    let side = still.render(&night());
    let hot = still.render(&FxState { disc_temp_c: 950.0, ..night() });
    let n = diff(&side, &hot);
    rep.add("brake_glow", if discs == 0 { "warn" } else if n >= MIN_PX { "ok" } else { "warn" },
            format!("{discs} disc material(s); 950 C changes {n} px{}", if discs > 0 && n < MIN_PX { " (disc hidden by a closed rim, or no glow mask)" } else { "" }));
    cells.push(("disc 950C".into(), hot));
    let day_side = still.render(&FxState::default());
    let dirty = still.render(&FxState { dirt: 1.0, ..FxState::default() });
    let n = diff(&day_side, &dirty);
    rep.add("dirt", if dirt == 0 { "info" } else if n >= MIN_PX { "ok" } else { "warn" },
            format!("{dirt} material(s) with a Dirt_Map; dirt 1.0 changes {n} px"));
    cells.push(("dirt 1".into(), dirty));
    // ---- exhaust flames (rear) ----
    cam(&mut still, rear_eye, rear_tgt, 40.0);
    let quiet = still.render(&night());
    let mut best: Option<(usize, Vec<u8>)> = None;
    let mut per = Vec::new();
    for t in 1..=5u32 {
        let fire = still.render(&FxState { backfire_age: 0.02, backfire_type: Some(t), ..night() });
        let n = diff(&quiet, &fire);
        per.push(format!("type {t}: {n} px / {} particle(s)", still.fxg.particle_count()));
        if best.as_ref().map_or(true, |b| n > b.0) {
            best = Some((n, fire));
        }
    }
    let n = best.as_ref().map_or(0, |b| b.0);
    rep.add("exhaust_flames", if exhausts.is_empty() { "warn" } else if n >= MIN_PX { "ok" } else { "warn" },
            format!("backfire 20 ms in: {}", per.join(", ")));
    if let Some((_, f)) = best {
        cells.push(("backfire".into(), f));
    }
    // ---- cockpit: LEDs and dash ----
    if let Some((p, r)) = fx.actor.cockpit_cam {
        // camera looks down its -Z, turned by rot (Y 180 = car forward)
        let yaw = r[1].to_radians();
        let fwd = [-yaw.sin(), 0.0, -yaw.cos()];
        let tgt = [p[0] + fwd[0], p[1] - 0.12, p[2] + fwd[2]];
        cam(&mut still, p, tgt, 55.0);
        // the seated driver's head is where the cockpit camera sits: hide it
        // (zero instance matrix, as B does in the window)
        let z = vec![0.0f32; 20];
        for m in still.meshes.iter() {
            if still.car.meshes.get(m.src).map_or(false, |c| c.name.starts_with(crate::driver::PREFIX)) {
                still.queue.write_buffer(&m.instances, 0, bytemuck::cast_slice(&z));
            }
        }
        let dark = still.render(&FxState::default());
        let idle = FxState { rpm: 900.0, lights: 1, ..FxState::default() };
        let idle_px = still.render(&idle);
        let max_rpm = rpm_leds.iter().cloned().fold(0.0, f32::max);
        if max_rpm > 0.0 {
            let rev = still.render(&FxState { rpm: max_rpm + 10.0, ..idle });
            let n = diff(&idle_px, &rev);
            rep.add("shift_lights", if n >= MIN_PX { "ok" } else { "warn" },
                    format!("rpm 900 -> {:.0} changes {n} px in the cockpit view", max_rpm + 10.0));
            cells.push(("shift lights".into(), rev));
        }
        let n = diff(&dark, &idle_px);
        rep.add("dash_on", if fx.dash.is_empty() { "info" } else if n >= MIN_PX { "ok" } else { "warn" },
                format!("ignition on changes {n} px in the cockpit view"));
        // the dash itself, read back: does it change with the car state?
        let mut changed = 0;
        let mut pages = 0;
        for k in 0..still.materials.dash_targets.len() {
            let a = FxState { rpm: 3000.0, speed_mps: 50.0 / 3.6, gear: 2, lights: 1, ..FxState::default() };
            let b = FxState { rpm: 6500.0, speed_mps: 180.0 / 3.6, gear: 5, lights: 1, ..FxState::default() };
            let _ = still.render(&a);
            let (_, _, pa) = crate::fxgpu::read_rgba(&still.device, &still.queue, &still.materials.dash_targets[k].1);
            let _ = still.render(&b);
            let (w, h, pb) = crate::fxgpu::read_rgba(&still.device, &still.queue, &still.materials.dash_targets[k].1);
            pages += 1;
            let n = diff(&pa, &pb);
            if n > 0 {
                changed += 1;
            }
            let pi = still.materials.dash_targets[k].0;
            let bound = fx.dash.get(pi).map_or(0, |p| {
                let a = FxState { rpm: 3000.0, speed_mps: 50.0 / 3.6, gear: 2, lights: 1, ..FxState::default() };
                let b = FxState { rpm: 6500.0, speed_mps: 180.0 / 3.6, gear: 5, lights: 1, ..FxState::default() };
                p.bindings.iter().filter(|e| crate::dash::eval_binding(e, &a) != crate::dash::eval_binding(e, &b)).count()
            });
            let sev = if n > 0 { "ok" } else if bound == 0 { "info" } else { "warn" };
            rep.add("dash_values", sev, format!("screen {k} ({w}x{h}): speed/rpm/gear change {n} px{}",
                if n == 0 && bound == 0 { " (the page draws its values from script only)" }
                else if n == 0 { " (bound values hidden by the page's own CSS state?)" } else { "" }));
            if k == 0 {
                // a dash thumbnail for the sheet
                cells.push(("dash".into(), scale_to(&pb, w, h, W, H)));
            }
        }
        let _ = (changed, pages);
        cells.push(("cockpit".into(), idle_px));
    } else {
        rep.add("cockpit_camera", "warn", "no drivable camera in the actor: LED / dash render checks skipped");
    }
    let _ = Indicator::Off;

    if sheet.is_some() {
        let names: Vec<&str> = cells.iter().map(|c| c.0.as_str()).collect();
        rep.add("sheet_cells", "info", format!("{W}x{H} tiles, 4 per row: {}", names.join(" | ")));
    }
    write(&rep, out);
    if let Some(sp) = sheet {
        // a directory (look-anim's --sheet <dir>): the sheet goes inside
        let sp_buf;
        let sp = if sp.extension().map_or(true, |e| e != "png") {
            let _ = std::fs::create_dir_all(sp);
            sp_buf = sp.join("sheet.png");
            sp_buf.as_path()
        } else {
            sp
        };
        let cols = 4;
        let rows = cells.len().div_ceil(cols);
        let (sw, sh) = (W * cols as u32, H * rows as u32);
        let mut img = vec![0u8; (sw * sh * 4) as usize];
        for (i, (_, px)) in cells.iter().enumerate() {
            let (cx, cy) = ((i % cols) as u32 * W, (i / cols) as u32 * H);
            for y in 0..H {
                for x in 0..W {
                    let s = ((y * W + x) * 4) as usize;
                    let d = (((cy + y) * sw + cx + x) * 4) as usize;
                    img[d..d + 3].copy_from_slice(&px[s..s + 3]);
                    img[d + 3] = 255;
                }
            }
        }
        crate::shot::write_png(sp, sw, sh, &img);
        let names: Vec<&str> = cells.iter().map(|c| c.0.as_str()).collect();
        println!("sheet cells (row-major, {cols} per row): {}", names.join(", "));
    }
}

fn scale_to(px: &[u8], w: u32, h: u32, tw: u32, th: u32) -> Vec<u8> {
    let mut out = vec![0u8; (tw * th * 4) as usize];
    for y in 0..th {
        for x in 0..tw {
            let sx = (x * w / tw).min(w - 1);
            let sy = (y * h / th).min(h - 1);
            let s = ((sy * w + sx) * 4) as usize;
            let d = ((y * tw + x) * 4) as usize;
            out[d..d + 4].copy_from_slice(&px[s..s + 4]);
        }
    }
    out
}

fn write(rep: &Report, out: &Path) {
    if let Some(d) = out.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let mut counts = serde_json::Map::new();
    for r in &rep.recs {
        let k = r["severity"].as_str().unwrap_or("?").to_string();
        let n = counts.get(&k).and_then(|v| v.as_u64()).unwrap_or(0) + 1;
        counts.insert(k, json!(n));
    }
    println!("CHECK {} errors={} warns={} infos={} -> {}", rep.car,
             counts.get("error").and_then(|v| v.as_u64()).unwrap_or(0),
             counts.get("warn").and_then(|v| v.as_u64()).unwrap_or(0),
             counts.get("info").and_then(|v| v.as_u64()).unwrap_or(0), out.display());
    let v = json!({"car": rep.car, "records": rep.recs, "counts": counts, "anim": rep.extra});
    match std::fs::write(out, serde_json::to_string_pretty(&v).unwrap()) {
        Ok(_) => println!("wrote {}", out.display()),
        Err(e) => eprintln!("cannot write {}: {e}", out.display()),
    }
}
