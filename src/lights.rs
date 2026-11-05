//! Car lights and every other emissive surface on a car: lamps, brake-disc
//! glow, hot exhausts, dash LEDs / shift lights and display screens.
//!
//! The split mirrors the game's: a MATERIAL only says which function each
//! texel belongs to and what colour that function is; the INTENSITIES are
//! car-wide runtime state (the game's `<Fn>Intensity` material constants are
//! written by the lighting system every frame). So the material uniform is
//! static here and the intensities travel in the per-frame camera uniform
//! (`Uniforms::fx`, packed by `fx_block`), which costs nothing to change.
//!
//! LAMPS (VehicleLight reflector / VehicleGlass lens, shader 0x035e7620):
//! `FunctionMask1` front R high / G low / B daylight, rear R rain-fog /
//! G brake / B tail; `FunctionMask2` B indicator, G reverse (rear), R special
//! (in our RGBA8 upload - the textures are stored B,G,R, so the shader's .x
//! is our .b: its FunctionMask2.x indicator term reads our blue).
//! Front or rear, left or right comes from the VERTEX COLOUR in the game
//! (B > 0 rear, R left, G right); the viewer's meshes carry no vertex colour,
//! so car-space position stands in: +z front, +x left (the actor's "front
//! left" lamp sits at x = +0.72).
//! A reflector uses `intensity_when_on`, a lens `intensity_glass_when_on`
//! (`.carlightingsystem`, CarLightingSystemData).
//!
//! DISCS (VehicleDisc, 0x036acaa0): emissive = glow mask (Base_EmissiveMap,
//! Disc_Glow_Mask) * ColorFront/Rear * EmissiveFront/Rear. Colour and
//! strength over temperature are `steel_glow.curve4` (rgb, intensity) and
//! `brake_disc_emissive_mult` (car_shading_settings).
//!
//! MECHANICS GLOW: the actor's `shader_settings.mechanics_glow_batches`
//! (EXT_MECH_EXHAUST ...) light their Base_EmissiveMap with
//! `mechanics_glow_intensity` while hot.
//!
//! LEDS (VehicleLed, 0x03605230): `LedMask` red channel picks LED k when it is
//! below 0.03 + 0.04 (k - 1) (k = 1..26, background 1.0 = no LED); the LED
//! emits base emissive (EmissiveColor * Intensity * Base_EmissiveMap) *
//! Led<k> * ColorLed<k>, driven by `.carledsystem`. (The material's own
//! ColorLed<k> are white on every car checked and are not applied.)
//!
//! DISPLAYS: the dash page texture (dash.rs) on the display material.

use crate::asset::MaterialEntry;
use crate::fxdata::*;
use crate::fxstate::{FxState, Indicator};

/// Names for the light-switch positions H cycles through.
pub const STATE_NAMES: [&str; 4] = ["off", "stage 1 (DRL / parking)", "stage 2 (low beam)", "stage 3 (high beam)"];

/// Overall gain from `.carlightingsystem` intensities to our HDR scale. The
/// game multiplies by `vehicleLightEmissiveMultiplier`, which follows the
/// sun-elevation curve `lightsledsatsunelevation.curve4` (c0: 0.2 by day,
/// 1.0 at night). This is tuned so a lit DRL clips to white in its core and
/// keeps its colour at the edges under the studio rig (= the day value).
const LIGHT_GAIN: f32 = 1.2;
/// Interior backlights are seen from a metre away against near-black plastic.
const INTERIOR_GAIN: f32 = 0.6;
/// Brake disc glow: steel_glow intensity * brake_disc_emissive_mult (100)
/// is in the game's absolute units; scaled to ours by eye (a 900 C disc
/// glows orange, not white).
const DISC_GAIN: f32 = 0.025;
/// Hot exhaust: Base_EmissiveMap * mechanics_glow_intensity (0.2 on the
/// M4 GT3) * heat.
const MECH_GAIN: f32 = 12.0;
/// LEDs and display screens.
const LED_GAIN: f32 = 2.0;
const DISPLAY_GAIN: f32 = 1.4;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Class {
    None,
    Reflector,
    Lens,
    Interior,
    Disc,
    MechGlow,
    Led,
    Display(usize),
    /// a glowing race-number decal (numbers.rs)
    Number,
}

/// What the uniform needs to know about one material's emission.
#[derive(Clone)]
pub struct LightSrc {
    pub class: Class,
    /// per function slot (see `FN_*`): material colour
    colors: [[f32; 3]; 15],
    emissive: [f32; 3],
}

/// Material colour param per lighting function, in FN_* order.
const FN_COLOR: [&str; 15] = [
    "DaylightColor", "LowBeamColor", "HighBeamColor",
    "RearLightColor", "BrakeLightColor", "RainFogLightColor", "ReverseLightColor",
    "FrontIndicatorColor", "FrontIndicatorColor", "RearIndicatorColor", "RearIndicatorColor",
    "LFSpecialLightColor", "RFSpecialLightColor", "LRSpecialLightColor", "RRSpecialLightColor",
];

pub fn classify(m: &MaterialEntry, fx: Option<&CarFx>) -> Class {
    let shader = m.shader.as_deref().unwrap_or("").to_lowercase();
    let file = m.file.to_lowercase();
    let stem = file.trim_end_matches(".material");
    if let Some(fx) = fx {
        if let Some(i) = fx.dash.iter().position(|p| p.material == m.path
            || p.material.rsplit('/').next() == Some(&file)) {
            return Class::Display(i);
        }
        if fx.actor.mech_glow_intensity > 0.0
            && fx.actor.mech_glow_batches.iter().any(|b| b.to_lowercase() == stem)
            && m.slots.contains_key("Base_EmissiveMap") {
            return Class::MechGlow;
        }
    }
    if m.path.starts_with("evoview/numbers/") {
        return if m.slots.contains_key("Base_EmissiveMap") { Class::Number } else { Class::None };
    }
    let has_mask = m.slots.contains_key("FunctionMask1") || m.slots.contains_key("FunctionMask2");
    if has_mask && shader == "vehiclelight" {
        return Class::Reflector;
    }
    if has_mask && shader.contains("glass") {
        return Class::Lens;
    }
    if shader == "vehicledisc" {
        return Class::Disc;
    }
    if shader == "vehicleled" && m.slots.contains_key("LedMask") {
        return Class::Led;
    }
    // Interior backlights only. LED strips (`LedMask`) are rev lights,
    // dark at idle - Class::Led above.
    if crate::material::is_cabin(&file)
        && m.slots.contains_key("Base_EmissiveMap")
        && !m.slots.contains_key("LedMask")
        && m.param(&["Base_HasEmissiveMap"], 1.0) > 0.5
    {
        return Class::Interior;
    }
    Class::None
}

/// Which texture slot feeds `t_emis`, and whether it is colour (sRGB) data.
pub fn emissive_slot(m: &MaterialEntry, fx: Option<&CarFx>) -> Option<(usize, bool)> {
    match classify(m, fx) {
        Class::Reflector | Class::Lens => m.slot(&["FunctionMask1"]).map(|i| (i, false)),
        Class::Interior | Class::MechGlow | Class::Number => m.slot(&["Base_EmissiveMap"]).map(|i| (i, true)),
        Class::Disc => m.slot(&["Base_EmissiveMap"]).map(|i| (i, false)),
        Class::Led => m.slot(&["Base_EmissiveMap"]).map(|i| (i, true)),
        Class::Display(_) | Class::None => None,
    }
}

/// The second mask (indicators / reverse / specials) of a lamp.
pub fn mask2_slot(m: &MaterialEntry, fx: Option<&CarFx>) -> Option<usize> {
    match classify(m, fx) {
        Class::Reflector | Class::Lens => m.slot(&["FunctionMask2"]),
        // VehicleLed: the LED index mask rides in the second mask slot
        Class::Led => m.slot(&["LedMask"]),
        _ => None,
    }
}

fn fn_color(m: &MaterialEntry, name: &str, rear: bool) -> [f32; 3] {
    let dflt = if rear { [1.0, 0.0, 0.0, 1.0] } else { [1.0, 1.0, 1.0, 1.0] };
    let c = m.color(&[name], dflt);
    [c[0], c[1], c[2]]
}

pub fn source(m: &MaterialEntry, fx: Option<&CarFx>) -> LightSrc {
    let class = classify(m, fx);
    let mut colors = [[0.0; 3]; 15];
    for (i, n) in FN_COLOR.iter().enumerate() {
        // rear functions default red; indicators amber
        let rear = matches!(i, FN_REAR | FN_BRAKE | FN_RAINFOG);
        colors[i] = fn_color(m, n, rear);
        if (FN_LF_IND..=FN_RR_IND).contains(&i) && m.colors.get(*n).is_none() {
            colors[i] = [1.0, 0.305, 0.0];
        }
        if i == FN_REVERSE && m.colors.get(*n).is_none() {
            colors[i] = [1.0, 1.0, 1.0];
        }
    }
    let e = m.color(&["Base_EmissiveColor", "EmissiveColor"], [1.0, 1.0, 1.0, 1.0]);
    let k = m.param(&["Base_EmissiveIntensity", "EmissiveIntensity"], 1.0).max(0.0);
    let emissive = match class {
        // the glow mask is the whole story; intensity is the temperature's
        Class::Disc | Class::MechGlow => [e[0], e[1], e[2]],
        Class::Led => {
            let k = if k > 0.0 { k } else { 1.0 };
            [e[0] * k, e[1] * k, e[2] * k]
        }
        _ => [e[0] * k, e[1] * k, e[2] * k],
    };
    LightSrc { class, colors, emissive }
}

/// The emissive part of the material uniform:
/// (mode, emis_f[3], emis_r[3], emis_x[7], fx_m).
///
/// mode: 0 none, 1 emissive map * emis_f[0] (interior, on with the ignition),
/// 2 function mask, 3 disc glow, 4 mechanics glow, 5 LED, 6 display,
/// 7 race number (always lit).
/// For mode 2 the colours are per function, slots:
///   emis_f = [high, low, daylight]   (FunctionMask1 R/G/B at the front)
///   emis_r = [rain-fog, brake, tail] (FunctionMask1 R/G/B at the rear)
///   emis_x = [reverse, front ind, rear ind, LF, RF, LR, RR special]
/// fx_m.x = 1 lens (glass intensities), 0 reflector.
pub fn uniforms(src: &LightSrc) -> (f32, [[f32; 4]; 6], [[f32; 4]; 7], [f32; 4]) {
    let mut out = [[0.0f32; 4]; 6];
    let mut x = [[0.0f32; 4]; 7];
    let c4 = |c: [f32; 3]| [c[0], c[1], c[2], 0.0];
    match src.class {
        Class::None => (0.0, out, x, [0.0; 4]),
        Class::Interior => {
            out[0] = c4(src.emissive.map(|v| v * INTERIOR_GAIN));
            (1.0, out, x, [0.0; 4])
        }
        Class::Reflector | Class::Lens => {
            let lens = src.class == Class::Lens;
            let cl = &src.colors;
            out = [c4(cl[FN_HIGHBEAM]), c4(cl[FN_LOWBEAM]), c4(cl[FN_DAYLIGHT]),
                   c4(cl[FN_RAINFOG]), c4(cl[FN_BRAKE]), c4(cl[FN_REAR])];
            x = [c4(cl[FN_REVERSE]), c4(cl[FN_LF_IND]), c4(cl[FN_LR_IND]),
                 c4(cl[11]), c4(cl[12]), c4(cl[13]), c4(cl[14])];
            (2.0, out, x, [if lens { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0])
        }
        Class::Disc => {
            out[0] = c4(src.emissive);
            (3.0, out, x, [0.0; 4])
        }
        Class::MechGlow => {
            out[0] = c4(src.emissive);
            (4.0, out, x, [0.0; 4])
        }
        Class::Led => {
            out[0] = c4(src.emissive);
            (5.0, out, x, [0.0; 4])
        }
        Class::Display(_) => (6.0, out, x, [0.0; 4]),
        // the emissive map x colour, always lit
        Class::Number => {
            out[0] = c4(src.emissive);
            (7.0, out, x, [0.0; 4])
        }
    }
}

// ----------------------------------------------------------- runtime state --

/// Per-function intensities (reflector, glass) for the car's current state,
/// from its `.carlightingsystem`. Without one (a bare export) the BMW M4 CSL
/// numbers stand in.
pub fn function_intensities(fx: Option<&CarFx>, st: &FxState) -> ([f32; FN_COUNT], [f32; FN_COUNT]) {
    let mut refl = [0.0f32; FN_COUNT];
    let mut glass = [0.0f32; FN_COUNT];
    if let Some(f) = st.force_fn {
        let (a, b) = fx.and_then(|x| x.light_sys.as_ref()).map_or((0.0, 0.0), |ls| strongest(ls, f));
        if f < FN_COUNT {
            refl[f] = if a > 0.0 { a } else { 10.0 };
            glass[f] = if b > 0.0 { b } else { 1.0 };
        }
        return (refl, glass);
    }
    let Some(ls) = fx.and_then(|f| f.light_sys.as_ref()) else {
        // M4 CSL: stage 1 DRL 8/0.6, low 3/0.1, high 30/0.5, tail 1/1,
        // brake 30/5, fog 10/2
        let set = |r: &mut [f32; FN_COUNT], g: &mut [f32; FN_COUNT], f: usize, a: f32, b: f32| { r[f] = a; g[f] = b; };
        if st.lights >= 1 { set(&mut refl, &mut glass, FN_DAYLIGHT, 8.0, 0.6); set(&mut refl, &mut glass, FN_REAR, 1.0, 1.0); }
        if st.lights >= 2 { set(&mut refl, &mut glass, FN_LOWBEAM, 3.0, 0.1); }
        if st.lights >= 3 { set(&mut refl, &mut glass, FN_HIGHBEAM, 30.0, 0.5); }
        if st.brake > 0.05 { set(&mut refl, &mut glass, FN_BRAKE, 30.0 * st.brake, 5.0 * st.brake); }
        if st.fog { set(&mut refl, &mut glass, FN_RAINFOG, 10.0, 2.0); }
        if st.gear < 0 { set(&mut refl, &mut glass, FN_REVERSE, 10.0, 1.0); }
        return (refl, glass);
    };
    let blink_lit = |e: &LightEntry| -> bool {
        if let Some(p) = st.blink_phase {
            return p;
        }
        let (start_off, on, off) = match ls.blinkers.get(&e.blinker) {
            Some(b) => (b.start_off, b.on, b.off),
            None => match e.blink {
                Some(b) => b,
                None => return true,
            },
        };
        let period = (on + off).max(0.05);
        let t = st.time.rem_euclid(period);
        let lit = t < on;
        if start_off { !lit } else { lit }
    };
    let active = |cat: u32| -> bool {
        match cat {
            CAT_MAIN => true,
            CAT_COCKPIT => true,
            CAT_BRAKE => st.brake > 0.05,
            CAT_REVERSE => st.gear < 0,
            CAT_RAIN => st.fog,
            CAT_LEFT => st.indicator == Indicator::Left,
            CAT_RIGHT => st.indicator == Indicator::Right,
            CAT_WARNING => st.indicator == Indicator::Hazard,
            CAT_FLASHING => false,
            CAT_PIT_LIMITER => st.pit_limiter,
            _ => false,
        }
    };
    let mut apply = |set: &LightSet| {
        if !active(set.category) {
            return;
        }
        for e in &set.entries {
            let f = e.function as usize;
            if f >= FN_COUNT {
                continue;
            }
            let (mut a, mut b) = (e.on, e.glass_on);
            if (!e.blinker.is_empty() || e.blink.is_some()) && !blink_lit(e) {
                a = 0.0;
                b = 0.0;
            }
            if set.category == CAT_BRAKE {
                a *= st.brake;
                b *= st.brake;
            }
            refl[f] = a;
            glass[f] = b;
        }
    };
    if st.lights > 0 && !ls.stages.is_empty() {
        let stage = ((st.lights - 1) as usize).min(ls.stages.len() - 1);
        // Main lights first, then the modifiers (indicators turn a DRL off)
        let mut sets: Vec<&LightSet> = ls.stages[stage].iter().collect();
        sets.sort_by_key(|s| (s.category != CAT_MAIN) as u8);
        for s in sets {
            apply(s);
        }
    } else if st.lights == 0 && !ls.stages.is_empty() {
        // lights off: indicators / hazards still work, from stage 1's sets
        for s in &ls.stages[0] {
            if s.category != CAT_MAIN && s.category != CAT_COCKPIT {
                apply(s);
            }
        }
    }
    for s in &ls.shared {
        apply(s);
    }
    (refl, glass)
}

/// The strongest (reflector, glass) intensity function `f` gets anywhere in
/// the car's lighting system; (0, 0) = the car never lights it.
pub fn strongest(ls: &LightSystem, f: usize) -> (f32, f32) {
    let mut best = (0.0f32, 0.0f32);
    for set in ls.stages.iter().flatten().chain(ls.shared.iter()) {
        for e in &set.entries {
            if e.function as usize == f {
                best.0 = best.0.max(e.on);
                best.1 = best.1.max(e.glass_on);
            }
        }
    }
    best
}

/// LED values for the state: (intensity, colour) per Led_01..Led_26, from
/// the car's `.carledsystem`.
pub fn led_values(fx: Option<&CarFx>, st: &FxState) -> [(f32, [f32; 3]); 26] {
    let mut out = [(0.0f32, [0.0f32; 3]); 26];
    let Some(sys) = fx.and_then(|f| f.led_sys.as_ref()) else { return out };
    let blink = |l: &LedInst| -> bool {
        if l.time_off <= 0.0 {
            return true;
        }
        let period = (l.time_on.max(0.01) + l.time_off).max(0.02);
        let lit = st.time.rem_euclid(period) < l.time_on.max(0.01);
        if l.start_off { !lit } else { lit }
    };
    let mut set = |l: &LedInst, on: bool| {
        let i = l.led as usize;
        if i >= 26 {
            return;
        }
        if on && blink(l) {
            out[i] = (l.value_on.max(0.0), [l.color_on[0], l.color_on[1], l.color_on[2]]);
        } else if l.value_off > 0.0 {
            out[i] = (l.value_off, [l.color_off[0], l.color_off[1], l.color_off[2]]);
        }
    };
    if crate::envc::var_os("EVOVIEW_LED_LOG").is_some() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            for g in 0..48u32 {
                for stage in 0..4usize {
                    for l in sys.group(g, stage) {
                        eprintln!("LED group {g} stage {stage} -> led {} on {:.2} {:?} off {:.2} thr {:.0} blink {:.2}/{:.2}", l.led, l.value_on, l.color_on, l.value_off, l.threshold, l.time_on, l.time_off);
                    }
                }
            }
        });
    }
    let ignition = st.rpm > 0.0 || st.lights > 0;
    if ignition {
        for l in sys.group(LED_IGNITION, 0) { set(l, true); }
        if st.rpm <= 0.0 {
            for l in sys.group(LED_ENGINE_OFF, 0) { set(l, true); }
        }
    }
    if st.lights > 0 {
        for l in sys.group(LED_MAIN_STAGE, (st.lights - 1) as usize) { set(l, true); }
        for l in sys.group(LED_COCKPIT_STAGE, (st.lights - 1) as usize) { set(l, true); }
    }
    // shift lights: on above their threshold
    for l in sys.group(LED_RPM, 0) {
        set(l, st.rpm >= l.threshold && l.threshold > 0.0);
    }
    let limiter = if sys.limiter_rpm > 0.0 { sys.limiter_rpm } else { f32::MAX };
    if st.rpm >= limiter {
        for l in sys.group(LED_LIMITER, 0) { set(l, true); }
    }
    let lit = st.blink_phase.unwrap_or(st.time.rem_euclid(1.0) < 0.5);
    let (left, right) = match st.indicator {
        Indicator::Left => (true, false),
        Indicator::Right => (false, true),
        Indicator::Hazard => (true, true),
        Indicator::Off => (false, false),
    };
    if st.indicator == Indicator::Hazard {
        for l in sys.group(LED_WARNING, 0) { set(l, lit); }
    } else {
        if left { for l in sys.group(LED_LEFT, 0) { set(l, lit); } }
        if right { for l in sys.group(LED_RIGHT, 0) { set(l, lit); } }
    }
    if st.fog {
        for l in sys.group(LED_RAIN, 0) { set(l, true); }
    }
    if st.pit_limiter {
        for l in sys.group(LED_PIT, 0) { set(l, true); }
    }
    out
}

/// Brake disc temperature -> (rgb * strength) from steel_glow.curve4.
pub fn disc_glow(fx: Option<&CarFx>, temp_c: f32) -> [f32; 3] {
    let Some(sh) = fx.map(|f| &f.shading) else { return [0.0; 3] };
    let c = &sh.disc_curve;
    if c.len() < 4 {
        return [0.0; 3];
    }
    let k = c[3].eval(temp_c) * sh.disc_emissive_mult * DISC_GAIN;
    [c[0].eval(temp_c) * k, c[1].eval(temp_c) * k, c[2].eval(temp_c) * k]
}

/// Number of vec4s in the camera uniform's fx block.
pub const FX_VEC4: usize = 64;

/// Spot lights thrown by the lamps, up to this many (floor + bodywork).
pub const MAX_SPOTS: usize = 4;

/// The per-frame fx block of the camera uniform. Layout (shader.wgsl
/// `camera.fx[...]` must match):
///   0  x ambient scale (night), y time, z wet, w dirt
///   1  x mech glow, y rim velocity, z interior backlights on, w display on
///   2  disc glow rgb (front), w light gain
///   3  disc glow rgb (rear), w LED gain
///   4..8   reflector intensity per function (19 + pad)
///   9..13  glass intensity per function
///   14..20 LED intensity (26)
///   21..46 LED colour (26)
///   47..58 spots: 4 x (pos + radius, dir + cos outer, colour * I + cos inner)
///   59 dirt dust colour + roughness
///   60 dirt dust darker colour + darkening factor
///   61 dirt grime colour + roughness
///   62 dirt rubber colour + roughness
///   63 x display gain
pub fn fx_block(fx: Option<&CarFx>, st: &FxState, outdoor: bool) -> [[f32; 4]; FX_VEC4] {
    let mut b = [[0.0f32; 4]; FX_VEC4];
    let sh = fx.map(|f| &f.shading);
    // night: the studio's sky and rig go dark; the lamps take the night
    // value of the game's emissive-vs-sun curve
    let (lmul, ledmul) = match sh {
        Some(s) if st.night && s.light_mult[0] > 0.0 => (s.light_mult[1] / s.light_mult[0], s.led_mult[1] / s.led_mult[0].max(1e-3)),
        _ => (1.0, 1.0),
    };
    b[0] = [if st.night { NIGHT_AMBIENT } else { 1.0 }, st.time, st.wet, st.dirt];
    let heat = st.exhaust_heat;
    let mech = fx.map_or(0.0, |f| f.actor.mech_glow_intensity) * heat * MECH_GAIN;
    // rim velocity: 0 at tyre_min_blur_speed, 1 at tyre_max_blur_speed (km/h)
    let (lo, hi) = sh.map_or((20.0, 60.0), |s| (s.tyre_min_blur_speed, s.tyre_max_blur_speed.max(s.tyre_min_blur_speed + 1.0)));
    let rimv = ((st.speed_mps * 3.6 - lo) / (hi - lo)).clamp(0.0, 1.0);
    let ign = st.rpm > 0.0 || st.lights > 0;
    b[1] = [mech, rimv, if ign { 1.0 } else { 0.0 }, if ign { 1.0 } else { 0.0 }];
    let d = disc_glow(fx, st.disc_temp_c);
    b[2] = [d[0], d[1], d[2], LIGHT_GAIN * lmul];
    b[3] = [d[0], d[1], d[2], LED_GAIN * ledmul];
    let (refl, glass) = function_intensities(fx, st);
    for i in 0..FN_COUNT {
        b[4 + i / 4][i % 4] = refl[i];
        b[9 + i / 4][i % 4] = glass[i];
    }
    let leds = led_values(fx, st);
    for (i, (v, c)) in leds.iter().enumerate() {
        b[14 + i / 4][i % 4] = *v;
        b[21 + i] = [c[0], c[1], c[2], 1.0];
    }
    // ---- spots ----
    let mut n = 0;
    if let Some(f) = fx {
        let sh = &f.shading;
        for l in &f.actor.lights {
            if n >= MAX_SPOTS {
                break;
            }
            let front = l.location <= 1;
            let rear = l.location == 2 || l.location == 3;
            if !front && !rear {
                continue;
            }
            // which of its functions are lit, and how strongly
            let mut e = 0.0f32;
            for &fun in &l.functions {
                let fu = fun as usize;
                if fu < FN_COUNT && refl[fu] > 0.0 {
                    // emitter_intensity_on lives in the same set entries;
                    // approximate with the function's share of the lamp
                    e = e.max(emitter_share(f, st, fu));
                }
            }
            if e <= 0.0 {
                continue;
            }
            let props = if front { sh.light_front } else { sh.light_rear };
            if props.intensity <= 0.0 {
                continue;
            }
            // The lamp's axis is local -Y, rotated by the transform's Euler
            // angles (degrees, X then Y then Z). Front lamps rot.x = -80:
            // forward and 10 deg down; rear (116.5, 0, -180): backward and
            // ~26 deg down.
            let dir = lamp_dir(l.rot);
            let i = props.intensity * e * SPOT_GAIN * if st.night { 1.0 } else { 0.25 };
            let co = (props.outer.to_radians() * 0.5).cos();
            let ci = (props.inner.to_radians() * 0.5).cos();
            b[47 + n * 3] = [l.pos[0], l.pos[1], l.pos[2], props.radius.min(80.0)];
            b[48 + n * 3] = [dir[0], dir[1], dir[2], co];
            b[49 + n * 3] = [props.color[0] * i, props.color[1] * i, props.color[2] * i, ci];
            n += 1;
        }
        b[59] = [sh.dirt_dust[0], sh.dirt_dust[1], sh.dirt_dust[2], sh.dirt_dust_rough];
        b[60] = [sh.dirt_dust_darker[0], sh.dirt_dust_darker[1], sh.dirt_dust_darker[2], 0.5];
        b[61] = [sh.dirt_grime[0], sh.dirt_grime[1], sh.dirt_grime[2], sh.dirt_grime_rough];
        b[62] = [sh.dirt_rubber[0], sh.dirt_rubber[1], sh.dirt_rubber[2], sh.dirt_rubber_rough];
    }
    let _ = outdoor;
    b[63] = [DISPLAY_GAIN, n as f32, 0.0, 0.0];
    b
}

/// Rows of the camera uniform's `ai_fx` block.
pub const AI_FX_VEC4: usize = 20;

/// The AI cars' lamps (race.rs): every AI shares one light switch (`st`:
/// the session's - lit at night or in the rain) and brakes on its own, so
/// the block holds the per-function intensities with the brake off (rows
/// 0..4 reflector, 5..9 glass) and on (10..19); each AI instance carries its
/// pedal (instance facing.z = 1 + brake) and the shader blends the two.
/// The intensities are the player's car's lighting system, the colours
/// each AI material's own.
pub fn ai_fx_block(fx: Option<&CarFx>, st: &FxState) -> [[f32; 4]; AI_FX_VEC4] {
    let mut b = [[0.0f32; 4]; AI_FX_VEC4];
    for (half, brake) in [(0usize, 0.0f32), (10, 1.0)] {
        let s = FxState { brake, indicator: Indicator::Off, blink_phase: None, force_fn: None, ..*st };
        let (refl, glass) = function_intensities(fx, &s);
        for i in 0..FN_COUNT {
            b[half + i / 4][i % 4] = refl[i];
            b[half + 5 + i / 4][i % 4] = glass[i];
        }
    }
    b
}

/// Ambient scale at night (studio sky + rig). Not game data: the game's
/// night is a real sky; this just turns the studio down so lamps read.
const NIGHT_AMBIENT: f32 = 0.1;
/// Lamp spot strength on the floor, by eye.
const SPOT_GAIN: f32 = 0.35;

fn lamp_dir(rot: [f32; 3]) -> [f32; 3] {
    let (ax, ay, az) = (rot[0].to_radians(), rot[1].to_radians(), rot[2].to_radians());
    let v = [0.0f32, -1.0, 0.0];
    // about X
    let v = [v[0], v[1] * ax.cos() - v[2] * ax.sin(), v[1] * ax.sin() + v[2] * ax.cos()];
    // about Y
    let v = [v[0] * ay.cos() + v[2] * ay.sin(), v[1], -v[0] * ay.sin() + v[2] * ay.cos()];
    // about Z
    let v = [v[0] * az.cos() - v[1] * az.sin(), v[0] * az.sin() + v[1] * az.cos(), v[2]];
    normalize(v)
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / l, v[1] / l, v[2] / l]
}

/// `emitter_intensity_on` of function `f` in the active sets.
fn emitter_share(fx: &CarFx, st: &FxState, f: usize) -> f32 {
    let Some(ls) = fx.light_sys.as_ref() else { return 1.0 };
    let mut best = 0.0f32;
    let mut scan = |sets: &[LightSet]| {
        for s in sets {
            for e in &s.entries {
                if e.function as usize == f {
                    best = best.max(e.emitter_on);
                }
            }
        }
    };
    if st.lights > 0 && !ls.stages.is_empty() {
        let i = ((st.lights - 1) as usize).min(ls.stages.len() - 1);
        scan(&ls.stages[i]);
    }
    scan(&ls.shared);
    if f == FN_BRAKE {
        best *= st.brake;
    }
    best
}

/// Initial light switch: `EVOVIEW_LIGHTS=<0..3>`, else off.
pub fn initial_state() -> u32 {
    crate::fxstate::FxState::from_env().lights
}
