//! The game data behind the car's visual effects, read once at load.
//!
//! Everything here comes from the car's own files or the shared
//! `content/cars/common_assets` ones; nothing is invented. Sources:
//!
//! * `<car>.actor` (ActorData.car = CarActorData, Scene.proto):
//!   f2 physical lights (lamp positions + which functions they project),
//!   f11 displays, f21 shader settings (mechanics glow, rim velocity gammas),
//!   f25/f26 lights / LED batches, f40 particle emitters, f101/f102/f106
//!   lighting / LED / backfire scripts.
//! * `.carlightingsystem` (CarLightingSystemData): per light-switch stage,
//!   sets of (function, reflector intensity, glass intensity, blinker).
//! * `.carledsystem` (CarLedSystemData): shift lights and dash LEDs.
//! * `.carexhaustbackfire` (CarExhaustBackFireData): when backfires happen.
//! * `common_assets/car_shading_settings.carshadingglobalsetting`
//!   (CarShadingGlobalSetting): dirt colours, headlight cone, disc glow.
//! * `common_assets/emitters/*.carparticleemitterdata_emitterspack`: which
//!   particle emitter a backfire type / tyre effect uses.
//! * `content/particles/**.particleemitter` (ParticleEmitterData, Renderer.proto).
//!
//! Field numbers are the game's own (decoded with the FileDescriptorProtos
//! embedded in AssettoCorsaEVO.exe, L:/evoview_work/shaderre/protos).

use crate::proto::{fields, text, Val};
use std::collections::HashMap;

// ------------------------------------------------------------------ helpers --

fn vec_of(d: &[u8], r: (usize, usize), n: usize) -> [f32; 4] {
    let mut v = [0.0f32; 4];
    fields(d, r.0, r.1, |f, x| {
        if (1..=4).contains(&f) && (f as usize) <= n.max(1) {
            v[(f - 1) as usize] = x.f32();
        }
    });
    v
}

fn s(d: &[u8], v: Val) -> String {
    v.range().map(|r| text(d, r)).unwrap_or_default()
}

/// A transform's position and rotation (degrees).
fn transform(d: &[u8], r: (usize, usize)) -> ([f32; 3], [f32; 3]) {
    let (mut p, mut q) = ([0.0; 3], [0.0; 3]);
    fields(d, r.0, r.1, |f, v| {
        if let Some(rr) = v.range() {
            let x = vec_of(d, rr, 3);
            match f {
                1 => p = [x[0], x[1], x[2]],
                2 => q = [x[0], x[1], x[2]],
                _ => {}
            }
        }
    });
    (p, q)
}

/// Repeated enum: packed (bytes of varints) or one varint per field.
fn enums(d: &[u8], v: Val, out: &mut Vec<u32>) {
    match v {
        Val::Varint(x) => out.push(x as u32),
        Val::Bytes(a, b) => out.extend(crate::proto::varints(d, a, b)),
        _ => {}
    }
}

// ------------------------------------------------------------------- curves --

/// One piecewise-linear curve: (t, value) points, sorted by t.
#[derive(Clone, Debug, Default)]
pub struct Curve(pub Vec<(f32, f32)>);

impl Curve {
    pub fn eval(&self, t: f32) -> f32 {
        let p = &self.0;
        if p.is_empty() {
            return 0.0;
        }
        if t <= p[0].0 {
            return p[0].1;
        }
        for w in p.windows(2) {
            if t <= w[1].0 {
                let span = (w[1].0 - w[0].0).max(1e-6);
                return w[0].1 + (w[1].1 - w[0].1) * (t - w[0].0) / span;
            }
        }
        p[p.len() - 1].1
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

fn read_points(d: &[u8], a: usize, b: usize) -> Curve {
    let mut pts = Vec::new();
    fields(d, a, b, |f, v| {
        if f == 1 {
            if let Some((pa, pb)) = v.range() {
                let (mut t, mut y) = (0.0, 0.0);
                fields(d, pa, pb, |ff, vv| match ff {
                    1 => t = vv.f32(),
                    10 => y = vv.f32(),
                    _ => {}
                });
                pts.push((t, y));
            }
        }
    });
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    Curve(pts)
}

/// `.curve` (CurveDataEx: f1 points) or `.curve4` (CurvesDataEx: f1 curves,
/// each f1 points). A point is {f1 t, f10 value, f3/f4 bezier anchors}; the
/// anchors are ignored (piecewise linear).
pub fn read_curves(d: &[u8]) -> Vec<Curve> {
    // Is the first f1 entry a curve (its own f1 is a message) or a point?
    let mut nested = false;
    let mut first = true;
    fields(d, 0, d.len(), |f, v| {
        if f == 1 && first {
            first = false;
            if let Some((a, b)) = v.range() {
                fields(d, a, b, |ff, vv| {
                    if ff == 1 && vv.range().is_some() {
                        nested = true;
                    }
                });
            }
        }
    });
    if !nested {
        return vec![read_points(d, 0, d.len())];
    }
    let mut out = Vec::new();
    fields(d, 0, d.len(), |f, v| {
        if f == 1 {
            if let Some((a, b)) = v.range() {
                out.push(read_points(d, a, b));
            }
        }
    });
    out
}

// ------------------------------------------------------------------- lights --

/// CarLightingFunction (LogicScene.proto), 0..=14 used.
pub const FN_COUNT: usize = 19;
pub const FN_DAYLIGHT: usize = 0;
pub const FN_LOWBEAM: usize = 1;
pub const FN_HIGHBEAM: usize = 2;
pub const FN_REAR: usize = 3;
pub const FN_BRAKE: usize = 4;
pub const FN_RAINFOG: usize = 5;
pub const FN_REVERSE: usize = 6;
pub const FN_LF_IND: usize = 7;
pub const FN_RF_IND: usize = 8;
pub const FN_LR_IND: usize = 9;
pub const FN_RR_IND: usize = 10;
pub const FN_LF_SPECIAL: usize = 11;
pub const FN_RR_SPECIAL: usize = 14;

pub const FN_NAMES: [&str; 15] = [
    "daylight", "lowbeam", "highbeam", "rear", "brake", "rainfog", "reverse",
    "lf_indicator", "rf_indicator", "lr_indicator", "rr_indicator",
    "lf_special", "rf_special", "lr_special", "rr_special",
];

/// CarLightingCategory (LogicScene.proto).
pub const CAT_MAIN: u32 = 2;
pub const CAT_BRAKE: u32 = 4;
pub const CAT_RAIN: u32 = 5;
pub const CAT_FLASHING: u32 = 6;
pub const CAT_LEFT: u32 = 7;
pub const CAT_RIGHT: u32 = 8;
pub const CAT_REVERSE: u32 = 9;
pub const CAT_WARNING: u32 = 10;
pub const CAT_COCKPIT: u32 = 30;
/// CarLightingCategory_Pit_Limiter
pub const CAT_PIT_LIMITER: u32 = 40;

/// CarActorData.PhysicalLight: a lamp that throws light into the scene.
#[derive(Clone, Debug)]
pub struct PhysLight {
    pub name: String,
    /// PhysicalLightLocation: 0 FL 1 FR 2 RL 3 RR 4 rain 5 cockpit
    pub location: u32,
    pub pos: [f32; 3],
    /// degrees, x = pitch (front lamps -80 = pointing along +z, slightly down)
    pub rot: [f32; 3],
    pub functions: Vec<u32>,
}

/// CarLightingSystemData.Lights: one function's numbers inside a set.
#[derive(Clone, Debug, Default)]
pub struct LightEntry {
    pub function: u32,
    pub blinker: String,
    pub on: f32,
    pub glass_on: f32,
    pub emitter_on: f32,
    pub emitter_radius: f32,
    pub inner: f32,
    pub outer: f32,
    /// only `deprecated_*` blink fields set (no named blinker)
    pub blink: Option<(bool, f32, f32)>,
}

#[derive(Clone, Debug, Default)]
pub struct LightSet {
    pub category: u32,
    pub entries: Vec<LightEntry>,
}

#[derive(Clone, Debug, Default)]
pub struct Blinker {
    pub start_off: bool,
    pub on: f32,
    pub off: f32,
}

#[derive(Clone, Debug, Default)]
pub struct LightSystem {
    /// main_lights: one list of sets per light-switch stage
    pub stages: Vec<Vec<LightSet>>,
    /// shared_main_lights: brake, reverse, rain, indicators - any stage
    pub shared: Vec<LightSet>,
    pub blinkers: HashMap<String, Blinker>,
}

fn read_light_entry(d: &[u8], a: usize, b: usize) -> LightEntry {
    let mut e = LightEntry::default();
    let (mut can_blink, mut start_off, mut b_on, mut b_off) = (false, false, 0.0, 0.0);
    fields(d, a, b, |f, v| match f {
        1 => e.function = v.f32() as u32,
        27 => e.blinker = s(d, v),
        3 => e.on = v.f32(),
        20 => e.glass_on = v.f32(),
        14 => e.emitter_on = v.f32(),
        22 => e.emitter_radius = v.f32(),
        40 => e.inner = v.f32(),
        42 => e.outer = v.f32(),
        4 => can_blink = v.f32() > 0.5,
        21 => start_off = v.f32() > 0.5,
        7 => b_on = v.f32(),
        9 => b_off = v.f32(),
        _ => {}
    });
    if can_blink && e.blinker.is_empty() {
        e.blink = Some((start_off, b_on, b_off));
    }
    e
}

fn read_stage(d: &[u8], a: usize, b: usize) -> Vec<LightSet> {
    let mut sets = Vec::new();
    fields(d, a, b, |f, v| {
        if f != 1 {
            return;
        }
        let Some((sa, sb)) = v.range() else { return };
        let mut set = LightSet::default();
        fields(d, sa, sb, |ff, vv| match ff {
            1 => set.category = vv.f32() as u32,
            3 => {
                if let Some((ea, eb)) = vv.range() {
                    set.entries.push(read_light_entry(d, ea, eb));
                }
            }
            _ => {}
        });
        sets.push(set);
    });
    sets
}

pub fn read_light_system(d: &[u8]) -> LightSystem {
    let mut ls = LightSystem::default();
    fields(d, 0, d.len(), |f, v| {
        let Some((a, b)) = v.range() else { return };
        match f {
            1 => ls.stages.push(read_stage(d, a, b)),
            20 => ls.shared.extend(read_stage(d, a, b)),
            23 => {
                let (mut name, mut bl) = (String::new(), Blinker::default());
                fields(d, a, b, |ff, vv| match ff {
                    1 => name = s(d, vv),
                    2 => bl.start_off = vv.f32() > 0.5,
                    3 => bl.on = vv.f32(),
                    4 => bl.off = vv.f32(),
                    _ => {}
                });
                ls.blinkers.insert(name, bl);
            }
            _ => {}
        }
    });
    ls
}

// --------------------------------------------------------------------- LEDs --

/// CarLedSystemData.Instance: one LED's rule inside a group.
#[derive(Clone, Debug, Default)]
pub struct LedInst {
    /// LedLight enum, 0 = Led_01 .. 25 = Led_26
    pub led: u32,
    pub threshold: f32,
    pub start_off: bool,
    pub sync: Vec<u32>,
    pub time_on: f32,
    pub time_off: f32,
    pub color_on: [f32; 4],
    pub color_off: [f32; 4],
    pub value_on: f32,
    pub value_off: f32,
}

/// Group field numbers in CarLedSystemData we drive.
pub const LED_RPM: u32 = 1;
pub const LED_PIT: u32 = 2;
pub const LED_LIMITER: u32 = 3;
pub const LED_FUEL: u32 = 4;
pub const LED_ABS: u32 = 17;
pub const LED_TC: u32 = 18;
pub const LED_MAIN_STAGE: u32 = 19;
pub const LED_COCKPIT_STAGE: u32 = 20;
pub const LED_RAIN: u32 = 22;
pub const LED_LEFT: u32 = 24;
pub const LED_RIGHT: u32 = 25;
pub const LED_WARNING: u32 = 26;
pub const LED_IGNITION: u32 = 27;
pub const LED_ENGINE_OFF: u32 = 36;
pub const LED_HANDBRAKE: u32 = 41;
pub const LED_GEARBOX: u32 = 38;

#[derive(Clone, Debug, Default)]
pub struct LedSystem {
    /// group field -> list of Instances messages (repeated groups keep
    /// their order: main_light_stage[i] is stage i)
    pub groups: HashMap<u32, Vec<Vec<LedInst>>>,
    pub limiter_rpm: f32,
}

impl LedSystem {
    pub fn group(&self, f: u32, i: usize) -> &[LedInst] {
        self.groups.get(&f).and_then(|g| g.get(i)).map(|v| v.as_slice()).unwrap_or(&[])
    }
    pub fn led_count(&self) -> usize {
        self.groups.values().flatten().flatten().count()
    }
}

pub fn read_led_system(d: &[u8]) -> LedSystem {
    let mut ls = LedSystem::default();
    fields(d, 0, d.len(), |f, v| {
        if f == 33 {
            // uint32 varint; acecm-convert also leaves a fixed32 f33 behind
            // (SPEC_lights 3.5) - only the varint is the limiter
            if let Val::Varint(x) = v {
                ls.limiter_rpm = x as f32;
            }
            return;
        }
        let Some((a, b)) = v.range() else { return };
        if f == 35 {
            return;
        }
        let mut insts = Vec::new();
        fields(d, a, b, |ff, vv| {
            if ff != 1 {
                return;
            }
            let Some((ia, ib)) = vv.range() else { return };
            let mut l = LedInst { time_on: 1.0, ..Default::default() };
            let mut time_on_set = false;
            fields(d, ia, ib, |g, x| match g {
                1 => l.led = x.f32() as u32,
                2 => l.threshold = x.f32(),
                3 => l.start_off = x.f32() > 0.5,
                30 => enums(d, x, &mut l.sync),
                6 => { l.time_on = x.f32(); time_on_set = true; }
                8 => l.time_off = x.f32(),
                7 => l.value_on = x.f32(),
                9 => l.value_off = x.f32(),
                20 => if let Some(r) = x.range() { l.color_on = vec_of(d, r, 4) },
                21 => if let Some(r) = x.range() { l.color_off = vec_of(d, r, 4) },
                _ => {}
            });
            if !time_on_set {
                l.time_on = 0.0;
            }
            insts.push(l);
        });
        ls.groups.entry(f).or_default().push(insts);
    });
    ls
}

// ----------------------------------------------------------------- backfire --

#[derive(Clone, Debug, Default)]
pub struct Backfire {
    pub high_rpms: f32,
    pub low_rpms: f32,
    pub min_rpms: f32,
    pub high_rate: f32,
    pub blip_time: f32,
    pub delay: f32,
    pub variation: f32,
    pub time: f32,
    /// visual_backfire_type_1..5_probability (percent)
    pub visual_prob: [f32; 5],
}

pub fn read_backfire(d: &[u8]) -> Backfire {
    let mut b = Backfire::default();
    fields(d, 0, d.len(), |f, v| match f {
        1 => b.high_rpms = v.f32(),
        2 => b.low_rpms = v.f32(),
        3 => b.min_rpms = v.f32(),
        4 => b.high_rate = v.f32(),
        5 => b.blip_time = v.f32(),
        6 => b.delay = v.f32(),
        7 => b.variation = v.f32(),
        8 => b.time = v.f32(),
        20..=24 => b.visual_prob[(f - 20) as usize] = v.f32(),
        _ => {}
    });
    b
}

// ---------------------------------------------------------------- particles --

#[derive(Clone, Debug, Default)]
pub struct CurveParam {
    pub enable: bool,
    pub end_range: [f32; 2],
    pub curve: Vec<Curve>,
    pub absolute: bool,
}

impl CurveParam {
    /// Multiplier (or absolute value) at life fraction `t`, for a particle
    /// whose random end value is `end`.
    pub fn at(&self, t: f32, end: f32, ch: usize) -> Option<f32> {
        if !self.enable {
            return None;
        }
        if let Some(c) = self.curve.get(ch).or(self.curve.first()) {
            if !c.is_empty() {
                return Some(c.eval(t));
            }
        }
        Some(1.0 + (end - 1.0) * t)
    }
}

/// ParticleEmitterData (Renderer.proto), the parts the viewer simulates.
#[derive(Clone, Debug, Default)]
pub struct EmitterDef {
    pub path: String,
    pub gravity: [f32; 3],
    pub local: bool,
    /// EmissionSettings: f1 area_rotation (deg), f2 area_size, f3 axis_spread
    pub area_rotation: [f32; 3],
    pub area_size: f32,
    pub axis_spread: f32,
    pub lifetime: [f32; 2],
    pub speed: [f32; 2],
    pub scale: [f32; 2],
    pub alpha: [f32; 2],
    pub roll_speed: [f32; 2],
    pub tint: [f32; 3],
    pub burst: f32,
    pub spawn_rate: f32,
    pub capacity: u32,
    pub alpha_life: CurveParam,
    pub scale_life: CurveParam,
    pub speed_life: CurveParam,
    pub tint_life: CurveParam,
    pub pps_strength: Vec<Curve>,
    pub alpha_strength: Vec<Curve>,
    /// material of the particle mesh
    pub material: ParticleMat,
}

#[derive(Clone, Debug, Default)]
pub struct ParticleMat {
    pub file: String,
    pub color_tex: Option<usize>,
    pub opacity_tex: Option<usize>,
    pub cols: f32,
    pub rows: f32,
    pub flipbook: bool,
    pub emissive: f32,
    pub basecolor: [f32; 4],
    /// blendMode 1 = alpha
    pub additive: bool,
    /// Tint_TintRedChannelIsLifetime: the tint carries the life fraction
    /// (flipbook position), not a colour
    pub tint_is_life: bool,
    /// opacity in the alpha channel (packed BC3 sheets), else in red
    pub opacity_alpha: bool,
    /// a heat-haze / refraction particle (drawn faint)
    pub refracting: bool,
    /// water (spray, splash): Refraction_Transmission, the share of its
    /// look that is the bright scene bent through it
    pub water: f32,
    /// water: the flipbook NormalMap its refraction bends by, and
    /// Refraction_Distortion (how far)
    pub normal_tex: Option<usize>,
    pub distortion: f32,
    /// water: Refraction_Roughness, how blurred the scene through it is
    pub refr_rough: f32,
    /// water: Ambient_Intensity, how brightly the sky lights the mist
    pub amb_int: f32,
    /// DepthFade_Enable / DepthFade_Distance (m): the soft edge against the
    /// scene (0 = the viewer's default)
    pub depth_fade: f32,
}

fn v2(d: &[u8], v: Val) -> [f32; 2] {
    v.range().map(|r| { let x = vec_of(d, r, 2); [x[0], x[1]] }).unwrap_or([0.0; 2])
}

fn read_curve_param(d: &[u8], v: Val, load: &mut dyn FnMut(&str) -> Vec<Curve>) -> CurveParam {
    let mut c = CurveParam::default();
    let Some((a, b)) = v.range() else { return c };
    let mut path = String::new();
    fields(d, a, b, |f, x| match f {
        1 => c.enable = x.f32() > 0.5,
        2 => {
            // CurveParam: f2 end_range, f3 curve; Multiplier/ColorParam: f2 curve
            match x.range() {
                Some(r) if d[r.0..r.1].iter().all(|&ch| (0x20..0x7f).contains(&ch)) && r.1 > r.0 + 4 => {
                    path = text(d, r)
                }
                Some(r) => c.end_range = { let q = vec_of(d, r, 2); [q[0], q[1]] },
                None => {}
            }
        }
        3 => match x {
            Val::Bytes(_, _) => path = s(d, x),
            _ => c.absolute = x.f32() > 0.5,
        },
        4 => c.absolute = x.f32() > 0.5,
        _ => {}
    });
    if !path.is_empty() {
        c.curve = load(&path);
    }
    c
}

pub fn read_emitter(d: &[u8], path: &str, load: &mut dyn FnMut(&str) -> Vec<Curve>) -> (EmitterDef, String) {
    let mut e = EmitterDef { path: path.to_string(), tint: [1.0; 3], ..Default::default() };
    let mut mesh = String::new();
    fields(d, 0, d.len(), |f, v| match f {
        6 => mesh = s(d, v),
        7 => if let Some(r) = v.range() { let q = vec_of(d, r, 3); e.gravity = [q[0], q[1], q[2]] },
        8 => e.local = v.f32() > 0.5,
        10 => {
            let Some((a, b)) = v.range() else { return };
            fields(d, a, b, |g, x| match g {
                1 => if let Some(r) = x.range() { let q = vec_of(d, r, 3); e.area_rotation = [q[0], q[1], q[2]] },
                2 => e.area_size = x.f32(),
                3 => e.axis_spread = x.f32(),
                4 => e.lifetime = v2(d, x),
                5 => e.speed = v2(d, x),
                6 => e.scale = v2(d, x),
                7 => e.alpha = v2(d, x),
                10 => e.roll_speed = v2(d, x),
                12 => if let Some(r) = x.range() { let q = vec_of(d, r, 3); e.tint = [q[0], q[1], q[2]] },
                16 => e.burst = x.f32(),
                _ => {}
            });
        }
        11 => {
            let Some((a, b)) = v.range() else { return };
            fields(d, a, b, |g, x| {
                let p = read_curve_param(d, x, load);
                match g {
                    1 => e.pps_strength = p.curve,
                    4 if p.enable => e.alpha_strength = p.curve,
                    _ => {}
                }
            });
        }
        12 => {
            let Some((a, b)) = v.range() else { return };
            fields(d, a, b, |g, x| {
                let p = read_curve_param(d, x, load);
                match g {
                    1 => e.speed_life = p,
                    3 => e.alpha_life = p,
                    8 => e.tint_life = p,
                    9 => e.scale_life = p,
                    _ => {}
                }
            });
        }
        662 => {
            // quality "high"
            let Some((a, b)) = v.range() else { return };
            fields(d, a, b, |g, x| match g {
                8 => e.spawn_rate = x.f32(),
                9 => e.capacity = x.f32() as u32,
                _ => {}
            });
        }
        _ => {}
    });
    (e, mesh)
}

/// Which emitter an effect uses: ParticleEmitter packs and the car's list.
#[derive(Clone, Debug)]
pub struct CarEmitter {
    pub name: String,
    pub bone: String,
    pub pos: [f32; 3],
    pub rot: [f32; 3],
    /// CarParticleEmitterEnum: 0 car, 1 tyre, 2/3/4 exhaust 1/2/3
    pub kind: u32,
    /// (CarParticleEmitterData_EmitterType, particleemitter path) overrides
    pub overrides: Vec<(u32, String)>,
}

/// CarParticleEmitterData_EmitterType values.
pub const ET_CAR_SMOKE: u32 = 0;
pub const ET_CAR_VAPOR: u32 = 1;
pub const ET_CAR_GRASS: u32 = 2;
pub const ET_CAR_DIRT: u32 = 3;
pub const ET_BACKFIRE1: u32 = 20;
pub const ET_DEBRIS: u32 = 30;
pub const ET_TYRE_SMOKE: u32 = 40;
pub const ET_TYRE_VAPOR: u32 = 50;
pub const ET_TYRE_GRASS: u32 = 60;
pub const ET_TYRE_DIRT: u32 = 70;
/// evoview's own: the tyre throwing up a puddle (water_splash). The game
/// fires it from code, not from the tyres pack.
pub const ET_TYRE_SPLASH: u32 = 51;
pub const SPLASH_EMITTER: &str = "content\\particles\\spray\\car_tyre_splash\\water_splash.particleemitter";

pub fn read_pack(d: &[u8]) -> Vec<(u32, String)> {
    let mut out = Vec::new();
    fields(d, 0, d.len(), |f, v| {
        if f != 1 {
            return;
        }
        let Some((a, b)) = v.range() else { return };
        let (mut t, mut p) = (0u32, String::new());
        fields(d, a, b, |g, x| match g {
            1 => t = x.f32() as u32,
            2 => p = s(d, x),
            _ => {}
        });
        out.push((t, p));
    });
    out
}

// -------------------------------------------------------------------- actor --

#[derive(Clone, Debug, Default)]
pub struct DisplayDef {
    pub html: String,
    pub material: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ActorFx {
    pub lights: Vec<PhysLight>,
    pub light_script: String,
    pub led_script: String,
    pub backfire_script: String,
    pub glow_script: String,
    pub displays: Vec<DisplayDef>,
    pub lights_batches: Vec<String>,
    pub leds_batches: Vec<String>,
    pub emitters: Vec<CarEmitter>,
    pub mech_glow_batches: Vec<String>,
    pub mech_glow_intensity: f32,
    pub rim_velocity_gamma: f32,
    pub rim_velocity_gamma_blur: f32,
    pub rim_inverse_velocity_gamma: f32,
    /// base_meshes f18..f21: the motion-blur rims
    pub rim_blur_meshes: Vec<String>,
    /// drivable_cameras[0] (the cockpit camera): position, rotation (deg)
    pub cockpit_cam: Option<([f32; 3], [f32; 3])>,
    /// every camera the car brings: drivable_cameras (f10: Cockpit, Dash,
    /// Bonnet, Bumper, NearChase, FarChase) then cameras (f1: the onboard /
    /// replay views), in car mesh space
    pub cameras: Vec<CarCam>,
    /// audio_events (f9): (FMOD event path, AudioEventDataType, bank path) -
    /// the car's whole sound: engine int / ext, wheels, wind, bodywork,
    /// turbo, backfire, brakes, wheel hub, gearbox, transmission, horn,
    /// indicators
    pub audio_events: Vec<(String, u32, String)>,
    /// mirrors (f23): centre / left / right, the batch they show on, the
    /// bone they hang from, aim and field of view
    pub mirrors: Vec<MirrorDef>,
}

#[derive(Clone, Debug, Default)]
pub struct MirrorDef {
    /// 0 centre, 1 left, 2 right, 3 the rear-view camera screen
    pub kind: u32,
    pub batch: String,
    pub bone: String,
    pub offset: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
    /// the rear-view CAMERA (f24 rear_camera) feeding a dash screen: `offset`
    /// is its car-space position, not an offset from the glass
    pub camera: bool,
}

/// One of the car's own cameras (CameraModuleData).
#[derive(Clone, Debug, Default)]
pub struct CarCam {
    pub name: String,
    pub pos: [f32; 3],
    /// Euler degrees; the camera looks down its -Z (yaw 180 = car forward)
    pub rot: [f32; 3],
    pub fov: f32,
    /// CameraModeType: 1 cockpit, 2 external, 3 fixed, 4 dash, 5 bonnet
    pub mode: u32,
    pub drivable: bool,
}

fn read_cam_module(d: &[u8], a: usize, b: usize, drivable: bool) -> CarCam {
    let mut c = CarCam { fov: 55.0, drivable, ..Default::default() };
    fields(d, a, b, |g, x| match g {
        1 => if let Some(rr) = x.range() {
            let (p, q) = transform(d, rr);
            c.pos = p;
            c.rot = q;
        },
        2 => if let Some((ca, cb)) = x.range() {
            fields(d, ca, cb, |h, y| match h {
                2 => c.name = s(d, y),
                14 => if let Some((ba, bb)) = y.range() {
                    fields(d, ba, bb, |k, z| match (k, z) {
                        (2, Val::F32(f)) => c.fov = f,
                        (13, Val::Varint(n)) => c.mode = n as u32,
                        _ => {}
                    });
                },
                _ => {}
            });
        },
        _ => {}
    });
    c
}

fn read_car_actor(d: &[u8], a: usize, b: usize, out: &mut ActorFx) {
    fields(d, a, b, |f, v| {
        let r = v.range();
        match f {
            2 => {
                let Some((la, lb)) = r else { return };
                let mut l = PhysLight { name: String::new(), location: 0, pos: [0.0; 3],
                                        rot: [0.0; 3], functions: Vec::new() };
                fields(d, la, lb, |g, x| match g {
                    1 => l.name = s(d, x),
                    6 => l.location = x.f32() as u32,
                    3 => if let Some(rr) = x.range() { let (p, q) = transform(d, rr); l.pos = p; l.rot = q },
                    2 => enums(d, x, &mut l.functions),
                    _ => {}
                });
                out.lights.push(l);
            }
            10 => {
                let Some((la, lb)) = r else { return };
                if out.cockpit_cam.is_none() {
                    fields(d, la, lb, |g, x| {
                        if g == 1 {
                            if let Some(rr) = x.range() {
                                out.cockpit_cam = Some(transform(d, rr));
                            }
                        }
                    });
                }
                out.cameras.push(read_cam_module(d, la, lb, true));
            }
            1 => {
                let Some((la, lb)) = r else { return };
                let mut c = read_cam_module(d, la, lb, false);
                if c.name.is_empty() {
                    c.name = format!("Car cam {}", out.cameras.iter().filter(|c| !c.drivable).count() + 1);
                }
                out.cameras.push(c);
            }
            23 => {
                let Some((la, lb)) = r else { return };
                let mut m = MirrorDef { fov: 20.0, ..Default::default() };
                fields(d, la, lb, |g, x| match (g, x) {
                    (1, Val::Varint(n)) => m.kind = n as u32,
                    (2, _) => m.batch = s(d, x),
                    (3, _) => m.bone = s(d, x),
                    (5, _) => if let Some(rr) = x.range() { let q = vec_of(d, rr, 3); m.offset = [q[0], q[1], q[2]] },
                    (6, Val::F32(f)) => m.yaw = f,
                    (7, Val::F32(f)) => m.pitch = f,
                    (8, Val::F32(f)) => m.fov = f,
                    _ => {}
                });
                out.mirrors.push(m);
            }
            // rear_camera {1 offset, 2 yaw, 3 pitch, 4 fov, 5 batchName}: the
            // digital rear-view screen 13 race cars have (MIRROR_D) instead of
            // a centre mirror
            24 => {
                let Some((la, lb)) = r else { return };
                let mut m = MirrorDef { kind: 3, fov: 60.0, camera: true, ..Default::default() };
                fields(d, la, lb, |g, x| match (g, x) {
                    (1, _) => if let Some(rr) = x.range() { let q = vec_of(d, rr, 3); m.offset = [q[0], q[1], q[2]] },
                    (2, Val::F32(f)) => m.yaw = f,
                    (3, Val::F32(f)) => m.pitch = f,
                    (4, Val::F32(f)) => m.fov = f,
                    (5, _) => m.batch = s(d, x),
                    _ => {}
                });
                if !m.batch.is_empty() {
                    out.mirrors.push(m);
                }
            }
            9 => {
                let Some((la, lb)) = r else { return };
                let (mut ev, mut ty, mut bank) = (String::new(), 0u32, String::new());
                fields(d, la, lb, |g, x| match (g, x) {
                    (1, _) => ev = s(d, x),
                    (2, Val::Varint(n)) => ty = n as u32,
                    (6, _) => bank = s(d, x),
                    _ => {}
                });
                if !ev.is_empty() {
                    out.audio_events.push((ev, ty, bank));
                }
            }
            101 => out.light_script = s(d, v),
            102 => out.led_script = s(d, v),
            105 => out.glow_script = s(d, v),
            106 => out.backfire_script = s(d, v),
            11 => {
                let Some((la, lb)) = r else { return };
                let mut dd = DisplayDef::default();
                fields(d, la, lb, |g, x| match g {
                    1 => dd.html = s(d, x),
                    2 => dd.material = s(d, x),
                    4 => dd.width = x.f32() as u32,
                    5 => dd.height = x.f32() as u32,
                    _ => {}
                });
                out.displays.push(dd);
            }
            12 => {
                let Some((la, lb)) = r else { return };
                fields(d, la, lb, |g, x| {
                    if (18..=21).contains(&g) {
                        if let Some((ma, mb)) = x.range() {
                            fields(d, ma, mb, |h, y| {
                                if h == 1 {
                                    out.rim_blur_meshes.push(s(d, y));
                                }
                            });
                        }
                    }
                });
            }
            21 => {
                let Some((la, lb)) = r else { return };
                fields(d, la, lb, |g, x| match g {
                    2 => out.rim_velocity_gamma = x.f32(),
                    6 => out.rim_velocity_gamma_blur = x.f32(),
                    3 => out.rim_inverse_velocity_gamma = x.f32(),
                    4 => out.mech_glow_batches.push(s(d, x)),
                    5 => out.mech_glow_intensity = x.f32(),
                    _ => {}
                });
            }
            25 | 26 => {
                let Some((la, lb)) = r else { return };
                fields(d, la, lb, |g, x| {
                    if g == 1 {
                        if f == 25 { out.lights_batches.push(s(d, x)) } else { out.leds_batches.push(s(d, x)) }
                    }
                });
            }
            40 => {
                let Some((la, lb)) = r else { return };
                let mut e = CarEmitter { name: String::new(), bone: String::new(), pos: [0.0; 3],
                                         rot: [0.0; 3], kind: 0, overrides: Vec::new() };
                fields(d, la, lb, |g, x| match g {
                    1 => e.name = s(d, x),
                    2 => e.bone = s(d, x),
                    3 => if let Some(rr) = x.range() { let (p, q) = transform(d, rr); e.pos = p; e.rot = q },
                    7 => e.kind = x.f32() as u32,
                    4 => if let Some((ea, eb)) = x.range() {
                        let (mut t, mut p) = (0u32, String::new());
                        fields(d, ea, eb, |h, y| match h {
                            1 => t = y.f32() as u32,
                            2 => p = s(d, y),
                            _ => {}
                        });
                        e.overrides.push((t, p));
                    },
                    _ => {}
                });
                out.emitters.push(e);
            }
            _ => {}
        }
    });
}

/// ActorData.car (f103) of a `.actor`.
pub fn read_actor(d: &[u8]) -> ActorFx {
    let mut out = ActorFx::default();
    fields(d, 0, d.len(), |f, v| {
        if f == 103 {
            if let Some((a, b)) = v.range() {
                read_car_actor(d, a, b, &mut out);
            }
        }
    });
    out
}

// ------------------------------------------------------------ global shading --

/// CarShadingGlobalSetting.PhysicalLightProperties.
#[derive(Clone, Copy, Debug, Default)]
pub struct LampProps {
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
    pub falloff: f32,
    pub inner: f32,
    pub outer: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Shading {
    pub dirt_dust: [f32; 3],
    pub dirt_dust_darker: [f32; 3],
    pub dirt_dust_rough: f32,
    pub dirt_grime: [f32; 3],
    pub dirt_grime_rough: f32,
    pub dirt_rubber: [f32; 3],
    pub dirt_rubber_rough: f32,
    pub tyre_min_blur_speed: f32,
    pub tyre_max_blur_speed: f32,
    pub disc_emissive_mult: f32,
    pub disc_curve: Vec<Curve>,
    pub light_front: LampProps,
    pub light_rear: LampProps,
    pub light_rain: LampProps,
    /// lightsledsatsunelevation.curve4 (sun elevation + 90 deg -> multiplier):
    /// [day, night] for lights (c0) and LEDs (c3)
    pub light_mult: [f32; 2],
    pub led_mult: [f32; 2],
}

fn lamp(d: &[u8], r: (usize, usize)) -> LampProps {
    let mut l = LampProps::default();
    fields(d, r.0, r.1, |f, v| match f {
        5 => if let Some(rr) = v.range() { let q = vec_of(d, rr, 4); l.color = [q[0], q[1], q[2]] },
        1 => l.intensity = v.f32(),
        2 => l.radius = v.f32(),
        6 => l.falloff = v.f32(),
        3 => l.inner = v.f32(),
        4 => l.outer = v.f32(),
        _ => {}
    });
    l
}

pub fn read_shading(d: &[u8], load: &mut dyn FnMut(&str) -> Vec<Curve>) -> Shading {
    let mut sh = Shading {
        light_mult: [1.0, 1.0],
        led_mult: [1.0, 1.0],
        ..Default::default()
    };
    let mut led_curve = String::new();
    fields(d, 0, d.len(), |f, v| {
        let v3 = |v: Val| v.range().map(|r| { let q = vec_of(d, r, 3); [q[0], q[1], q[2]] }).unwrap_or([0.0; 3]);
        match f {
            101 => sh.dirt_dust = v3(v),
            102 => sh.dirt_dust_darker = v3(v),
            104 => sh.dirt_dust_rough = v.f32(),
            110 => sh.dirt_grime = v3(v),
            112 => sh.dirt_grime_rough = v.f32(),
            118 => sh.dirt_rubber = v3(v),
            120 => sh.dirt_rubber_rough = v.f32(),
            300 => sh.tyre_min_blur_speed = v.f32(),
            301 => sh.tyre_max_blur_speed = v.f32(),
            302 => sh.disc_emissive_mult = v.f32(),
            303 => sh.disc_curve = load(&s(d, v)),
            1100 => if let Some(r) = v.range() { sh.light_front = lamp(d, r) },
            1101 => if let Some(r) = v.range() { sh.light_rear = lamp(d, r) },
            1102 => if let Some(r) = v.range() { sh.light_rain = lamp(d, r) },
            1200 => led_curve = s(d, v),
            _ => {}
        }
    });
    if !led_curve.is_empty() {
        let c = load(&led_curve);
        // x = 90 + sun elevation: the studio's forced 83 deg = 173 (day);
        // a night sky (sun below the horizon) is < 90.
        if let (Some(a), Some(b)) = (c.first(), c.get(3)) {
            sh.light_mult = [a.eval(173.0), a.eval(45.0)];
            sh.led_mult = [b.eval(173.0), b.eval(45.0)];
        }
    }
    sh
}

// ------------------------------------------------------------------ all fx --

/// One particle effect the car can emit: where, which emitter definition.
#[derive(Clone, Debug)]
pub struct EmitterSlot {
    pub name: String,
    pub kind: u32,
    /// car space
    pub pos: [f32; 3],
    pub rot: [f32; 3],
    /// (emitter type, index into CarFx::emitter_defs)
    pub defs: Vec<(u32, usize)>,
}

#[derive(Clone, Debug, Default)]
pub struct CarFx {
    pub actor: ActorFx,
    pub light_sys: Option<LightSystem>,
    pub led_sys: Option<LedSystem>,
    pub backfire: Option<Backfire>,
    pub shading: Shading,
    pub emitter_defs: Vec<EmitterDef>,
    pub emitters: Vec<EmitterSlot>,
    /// dash pages, see dash.rs
    pub dash: Vec<crate::dash::Page>,
    /// what could not be resolved, for --check
    pub problems: Vec<String>,
    /// material texture slots the loader could not resolve ("material:slot")
    pub unresolved: Vec<String>,
}

// ------------------------------------------------------------------ loading --

use crate::asset::TextureEntry;
use crate::load::Vfs;

fn read_file(vfs: &mut Vfs, reference: &str) -> Option<Vec<u8>> {
    if reference.is_empty() {
        return None;
    }
    let k = vfs.find(reference, None)?;
    let d = vfs.read(&k);
    if d.is_empty() { None } else { Some(d) }
}

fn curves_of(vfs: &mut Vfs, reference: &str) -> Vec<Curve> {
    read_file(vfs, reference).map(|d| read_curves(&d)).unwrap_or_default()
}

/// First engine path ending in `.material` inside a mesh file.
fn mesh_material(d: &[u8]) -> Option<String> {
    let pat = b".material";
    let i = d.windows(pat.len()).position(|w| w.eq_ignore_ascii_case(pat))?;
    let mut s = i;
    while s > 0 && (0x20..0x7f).contains(&d[s - 1]) {
        s -= 1;
    }
    Some(String::from_utf8_lossy(&d[s..i + pat.len()]).into_owned())
}

fn particle_material(vfs: &mut Vfs, mesh_ref: &str, blob: &mut Vec<u8>,
                     textures: &mut Vec<TextureEntry>, tex_index: &mut HashMap<String, usize>)
                     -> ParticleMat {
    let mut pm = ParticleMat { cols: 1.0, rows: 1.0, basecolor: [1.0; 4], ..Default::default() };
    let Some(md) = read_file(vfs, mesh_ref) else { return pm };
    let Some(mref) = mesh_material(&md) else { return pm };
    let Some(mat) = read_file(vfs, &mref) else { return pm };
    let Some((_sh, slots, params, colors)) = crate::load::fx_material(&mat, &mref) else { return pm };
    pm.file = mref.clone();
    let p = |n: &str, d: f32| params.get(n).copied().unwrap_or(d);
    pm.flipbook = p("Flipbook_Enable", 0.0) > 0.5;
    pm.cols = p("Flipbook_Columns", 1.0).max(1.0);
    pm.rows = p("Flipbook_Rows", 1.0).max(1.0);
    pm.emissive = p("Emissive_Intensity", 0.0);
    // Emissive flames add light; a refracting puff (Refraction_Transmission,
    // the heat-haze / smoke of backfire_3) is drawn as faint lit smoke - the
    // viewer has no screen-space refraction.
    pm.additive = p("Emissive_UseBaseColorAsEmissive", 0.0) > 0.5 && pm.emissive > 0.0
        && p("Refraction_Transmission", 0.0) <= 0.0;
    pm.refracting = p("Refraction_Transmission", 0.0) > 0.0;
    let fl = mref.to_lowercase();
    if crate::envc::var_os("EVOVIEW_FX_PARAMS").is_some() && (fl.contains("spray") || fl.contains("splash")) {
        let mut v: Vec<_> = params.iter().collect();
        v.sort_by(|a, b| a.0.cmp(b.0));
        eprintln!("FXPARAMS {mref} {v:?} colors {colors:?} slots {slots:?}");
    }
    if pm.refracting && (fl.contains("spray") || fl.contains("splash")) {
        pm.water = p("Refraction_Transmission", 0.0).clamp(0.0, 1.0);
        pm.distortion = p("Refraction_Distortion", 0.5).max(0.0);
        pm.refr_rough = p("Refraction_Roughness", 0.0).clamp(0.0, 1.0);
        pm.amb_int = p("Ambient_Intensity", 1.0).max(0.0);
    }
    // the flipbook's NormalMap: water bends its refraction by it, smoke and
    // dust are lit through it (fx.wgsl sun_shade)
    if let Some(r) = slots.get("NormalMap").filter(|r| !r.is_empty()).cloned() {
        pm.normal_tex = crate::load::push_fx_texture(vfs, &r, blob, textures, tex_index);
    }
    if p("DepthFade_Enable", 0.0) > 0.5 {
        pm.depth_fade = p("DepthFade_Distance", 0.0).max(0.0);
    }
    pm.tint_is_life = p("Tint_TintRedChannelIsLifetime", 0.0) > 0.5;
    if let Some(c) = colors.get("Basecolor").or(colors.get("Base_Basecolor")) {
        for (i, x) in c.iter().take(4).enumerate() {
            pm.basecolor[i] = *x;
        }
    }
    // The flipbook lives in `OpacityMap` when there is one (the packed
    // 10x9 smoke sheets); `Base_OpacityMap` / `Base_BaseColorMap` beside it
    // are single-frame leftovers that do not match the flipbook grid. The
    // Smoke shader colours by Basecolor * lighting, not by a colour map.
    let smoke = _sh.as_deref().map_or(false, |x| x.eq_ignore_ascii_case("Smoke"));
    let packed = slots.get("OpacityMap").filter(|r| !r.is_empty()).cloned();
    // ⚠ Base_HasBaseColorMap 0 means the Base layer's map is a leftover:
    // car_tyre_detail_2 still names backfire's fire_tile_B there and drew
    // flame-sheet snowflakes for spray droplets. HasBaseColorMap 1 is the
    // card's own txDiffuse (its spray_1 droplets).
    let base_off = params.get("Base_HasBaseColorMap").map_or(false, |v| *v < 0.5);
    if !smoke && packed.is_none() && base_off && p("HasBaseColorMap", 0.0) > 0.5 {
        if let Some(r) = slots.get("txDiffuse").filter(|r| !r.is_empty()) {
            pm.color_tex = crate::load::push_fx_texture(vfs, r, blob, textures, tex_index);
        }
    } else if !smoke && packed.is_none() {
        if let Some(r) = slots.get("Base_BaseColorMap") {
            pm.color_tex = crate::load::push_fx_texture(vfs, r, blob, textures, tex_index);
        }
    }
    let had_packed = packed.is_some();
    if let Some(r) = packed.or_else(|| slots.get("Base_OpacityMap").cloned()) {
        pm.opacity_tex = crate::load::push_fx_texture(vfs, &r, blob, textures, tex_index);
    }
    // ⚠ The Particle shader (0x035331f0) runs only txDiffuse / OpacityMap /
    // NormalMap through the flipbook; the Base_* maps read the card's plain
    // UV. smoke_torus_detail's 3x3 sheet is txDiffuse, and cutting its
    // Base_BaseColorMap / Base_OpacityMap (single images) into 3x3 cells
    // drew angular fragments in the tyre smoke.
    if pm.flipbook && !had_packed {
        if let Some(r) = slots.get("txDiffuse").filter(|r| !r.is_empty()).cloned() {
            pm.color_tex = crate::load::push_fx_texture(vfs, &r, blob, textures, tex_index);
            pm.opacity_tex = None;
        }
    }
    // a packed BC3 sheet: RGB = the frame's shading, A = density
    if had_packed {
        if let Some(i) = pm.opacity_tex {
            if textures[i].format == "bc3" {
                pm.opacity_alpha = true;
                pm.color_tex = Some(i);
            }
        }
    }
    pm
}

/// Read every effect the car's data defines. None when there is no actor
/// (a bare .evocar export).
pub fn load(vfs: &mut Vfs, wheel_pos: &HashMap<String, [f32; 3]>, blob: &mut Vec<u8>,
            textures: &mut Vec<TextureEntry>, tex_index: &mut HashMap<String, usize>)
            -> Option<Box<CarFx>> {
    let actor_key = vfs.list("", "actor").into_iter().next()?;
    let ad = vfs.read(&actor_key);
    let mut fx = CarFx { actor: read_actor(&ad), ..Default::default() };

    let ls = fx.actor.light_script.clone();
    match read_file(vfs, &ls) {
        Some(d) => fx.light_sys = Some(read_light_system(&d)),
        None => fx.problems.push(format!("light_script not found: {ls:?}")),
    }
    let ls = fx.actor.led_script.clone();
    if let Some(d) = read_file(vfs, &ls) {
        fx.led_sys = Some(read_led_system(&d));
    } else if !ls.is_empty() {
        fx.problems.push(format!("led_script not found: {ls}"));
    }
    let bs = fx.actor.backfire_script.clone();
    if let Some(d) = read_file(vfs, &bs) {
        fx.backfire = Some(read_backfire(&d));
    }
    match read_file(vfs, "content/cars/common_assets/car_shading_settings.carshadingglobalsetting") {
        Some(d) => {
            let mut sh = None;
            {
                let mut ld = |p: &str| curves_of(vfs, p);
                sh = Some(read_shading(&d, &mut ld));
            }
            fx.shading = sh.unwrap_or_default();
        }
        None => fx.problems.push("car_shading_settings not found".into()),
    }

    // ---- particle emitters ----
    let mut def_index: HashMap<String, usize> = HashMap::new();
    let mut emitters = fx.actor.emitters.clone();
    // the body's own slot (debris / sparks, body dust, spray): a car without
    // one gets it at the middle of the floor
    if !emitters.iter().any(|e| e.kind == 0) {
        emitters.push(CarEmitter { name: "car".into(), bone: String::new(), pos: [0.0, 0.25, 0.0],
                                   rot: [0.0; 3], kind: 0, overrides: Vec::new() });
    }
    for e in &emitters {
        let pack = match e.kind {
            2..=4 => format!("content/cars/common_assets/emitters/backfire_type_{}.carparticleemitterdata_emitterspack", e.kind - 1),
            1 => "content/cars/common_assets/emitters/tyres.carparticleemitterdata_emitterspack".into(),
            // car.emitterspack: CarSmoke, CarVapor, CarGrass, CarDirt, Debris
            _ => "content/cars/common_assets/emitters/car.carparticleemitterdata_emitterspack".into(),
        };
        let mut list = read_file(vfs, &pack).map(|d| read_pack(&d)).unwrap_or_default();
        if e.kind == 1 {
            list.push((ET_TYRE_SPLASH, SPLASH_EMITTER.to_string()));
        }
        for (t, p) in &e.overrides {
            if !p.is_empty() {
                list.retain(|(tt, _)| tt != t);
                list.push((*t, p.clone()));
            }
        }
        // (every emitter of the pack: the game layers several per type -
        // smoke + torus + torus detail, stones + plant debris + dust)
        let mut defs = Vec::new();
        for (t, p) in list {
            let key = p.to_lowercase();
            let idx = if let Some(&i) = def_index.get(&key) {
                i
            } else {
                let Some(d) = read_file(vfs, &p) else {
                    fx.problems.push(format!("particle emitter not found: {p}"));
                    continue;
                };
                let (mut def, mesh) = {
                    let mut ld = |q: &str| curves_of(vfs, q);
                    read_emitter(&d, &p, &mut ld)
                };
                def.material = particle_material(vfs, &mesh, blob, textures, tex_index);
                if def.material.color_tex.is_none() && def.material.opacity_tex.is_none() {
                    fx.problems.push(format!("particle texture missing: {p}"));
                }
                if crate::envc::var_os("EVOVIEW_FX_DEFS").is_some() {
                    println!("FXDEF t{t} {p} rate {} cap {} life {:?} speed {:?} scale {:?} alpha {:?} burst {} grav {:?} pps {} additive {} tex {}",
                             def.spawn_rate, def.capacity, def.lifetime, def.speed, def.scale, def.alpha, def.burst, def.gravity,
                             def.pps_strength.first().map_or(0, |c| c.0.len()), def.material.additive, def.material.color_tex.is_some());
                    println!("   area_rot {:?} spread {} size {} local {}", def.area_rotation, def.axis_spread, def.area_size, def.local);
                    println!("   tint {:?} tint_life {} {:?} mat base {:?} emis {} tint_is_life {} pps {:?} alpha_str {:?} file {}",
                             def.tint, def.tint_life.enable, def.tint_life.curve.iter().map(|c| c.0.clone()).collect::<Vec<_>>(),
                             def.material.basecolor, def.material.emissive, def.material.tint_is_life,
                             def.pps_strength.first().map(|c| c.0.clone()), def.alpha_strength.first().map(|c| c.0.clone()), def.material.file);
                }
                fx.emitter_defs.push(def);
                def_index.insert(key, fx.emitter_defs.len() - 1);
                fx.emitter_defs.len() - 1
            };
            defs.push((t, idx));
        }
        // Tyre emitters ride on the wheel bone; exhausts are in car space.
        let mut pos = e.pos;
        if let Some(c) = e.bone.to_uppercase().strip_prefix("WHEEL_") {
            match wheel_pos.get(c) {
                Some(w) => pos = [w[0] + e.pos[0], w[1] + e.pos[1], w[2] + e.pos[2]],
                None => fx.problems.push(format!("emitter {} on unknown wheel {}", e.name, e.bone)),
            }
        }
        if crate::envc::var_os("EVOVIEW_FX_DEFS").is_some() {
            println!("FXSLOT {} kind {} bone {} pos {:?} rot {:?} defs {}", e.name, e.kind, e.bone, pos, e.rot, defs.len());
        }
        fx.emitters.push(EmitterSlot { name: e.name.clone(), kind: e.kind, pos, rot: e.rot, defs });
    }

    // ---- dash ----
    let displays = fx.actor.displays.clone();
    fx.dash = crate::dash::load(vfs, &displays, blob, textures, tex_index, &mut fx.problems);

    println!(
        "fx: {} lamp(s), {} light stage(s), {} LED rule(s), {} emitter(s) / {} def(s), {} display(s){}",
        fx.actor.lights.len(),
        fx.light_sys.as_ref().map_or(0, |l| l.stages.len()),
        fx.led_sys.as_ref().map_or(0, |l| l.led_count()),
        fx.emitters.len(), fx.emitter_defs.len(), fx.dash.len(),
        if fx.problems.is_empty() { String::new() } else { format!(", {} problem(s)", fx.problems.len()) });
    Some(Box::new(fx))
}
