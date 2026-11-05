//! Rain on the windshield: the game's static drops, cleared by the wipers.
//!
//! The game's windshield shader (VehicleGlassInterior with IsWindshield, DXIL
//! 04257970 lifted in L:/evoview_work/shaderre/x) tiles three shared maps
//! (common_assets/rain/rain_drop_mask / _alpha / _nm) at `Rain_ScaleUv`:
//!   life = saturate(mask + (mask < t ? 1 : 0) - t),  t = frc(rate * time)
//!   drop = life * alpha,  refraction = (nm.xy * 2 - 1) * Rain_ScaleRefraction
//! so every drop appears, then shrinks away, and the next one takes its
//! place. The wipers come in as up to two wipers x five swept quads
//! (Code_Rain_WiperNAreaKPointJ) from successive recent intervals; a pixel
//! in the newest swept area keeps 0.2 of its drops, older areas 0.4 / 0.6 /
//! 1.0, and glass no wiper reached 2x. The dynamic drops (a GPU simulation
//! over generated/windshield/baked*) are not ported.
//!
//! Here each wiper is tracked as a blade turning about its pivot: the
//! blade's angle in the sweep plane over the last ~1.2 s, so the shader
//! finds how long ago the blade passed a pixel and fades its drops back in.

use glam::Vec3;
use std::sync::atomic::{AtomicU32, Ordering};

static RAIN: AtomicU32 = AtomicU32::new(0);
static WET: AtomicU32 = AtomicU32::new(0);

/// This frame's weather: falling rain and how wet the ground is (0..1).
/// The sky, clouds, haze and track shading read it.
pub fn set_weather(rain: f32, wet: f32) {
    RAIN.store(rain.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    WET.store(wet.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
}

pub fn rain() -> f32 {
    f32::from_bits(RAIN.load(Ordering::Relaxed))
}

pub fn wetness() -> f32 {
    f32::from_bits(WET.load(Ordering::Relaxed))
}

/// The dynamic track's water, as the game runs it (DynamicTrack, client
/// exe 20260907: per-cell step FUN_142257fb0, weather terms FUN_142258170,
/// settings FUN_142250780 <- content/tracks/common_assets/
/// global_dynamic_track_settings.dynamictracksettings). Each cell holds
/// (rubber, wetness, marbles); per step of dt seconds with rain r:
///   wet += r dt wetness_per_full_rain_second;  rubber -= r dt rubber_wash
///   wet -= dt drain wet + (1 - 0.3 wet0) max(sun, 0.2) evaporation
///          (1 - 0.75 cloud) dt
/// cloud = (gloominess 0.25 + 0.75) cloud_coverage, sun = cos(sun zenith).
/// Cars dry their line: wetness_reduction_per_speed_per_second x speed x dt
/// under each tyre (the cell size is evoview's 1 m, a guess).
mod water {
    pub const PER_RAIN: f32 = 0.0009;
    pub const DRAIN: f32 = 0.00073;
    pub const EVAPORATION: f32 = 0.0004;
    pub const RUBBER_WASH: f32 = 0.0002;
    pub const PER_SPEED: f32 = 0.005;
    pub const CELL: f32 = 1.0;
    pub const MAX_CELLS: usize = 400_000;
    /// water flow (flow): the cells simulated around the car (radius, cells),
    /// and how often
    pub const FLOW_R: i32 = 32;
    pub const FLOW_STEP: f32 = 0.05;
    /// the game's cell (~0.366 m) over ours: water moves cell to cell at a
    /// rate set by the height step between them, so its speed in m/s goes
    /// with the cell size squared - scale the flux to keep the game's
    pub const FLOW_K: f32 = (0.366 / CELL) * (0.366 / CELL);

    pub struct Water {
        pub wet: f32,
        pub rubber: f32,
        pub cells: std::collections::HashMap<(i32, i32), f32>,
        pub started: bool,
        pub fixed: Option<f32>,
        pub speed: f32,
        pub shown: i32,
        /// the dry-line map's last upload (dry_map), and whether it was empty
        pub last_map: Option<std::time::Instant>,
        pub map_empty: bool,
        /// EVOVIEW_DRY_LINE_K: scales the cars' drying (the rate is a guess)
        pub dry_k: f32,
        /// EVOVIEW_DRY_LINE_LOOK: how strongly the map shows water a car took
        /// off (dry_map); the look only, grip still sees the cell's water
        pub dry_look: f32,
        /// water flow: the track's height per cell (NaN = no surface), the
        /// time not yet flowed, and EVOVIEW_WATER_FLOW=0 to turn it off
        pub heights: std::collections::HashMap<(i32, i32), f32>,
        pub flow_dt: f32,
        pub flow_on: bool,
    }

    pub fn step(w: f32, rain: f32, dt: f32, sun: f32, cloud: f32) -> f32 {
        let w0 = w;
        let mut w = w + rain * dt * PER_RAIN;
        w -= dt * DRAIN * w + (1.0 - 0.3 * w0) * sun.max(0.2) * EVAPORATION * (1.0 - 0.75 * cloud) * dt;
        w.clamp(0.0, 1.0)
    }
}

static WATER: std::sync::Mutex<Option<water::Water>> = std::sync::Mutex::new(None);

fn with_water<R>(f: impl FnOnce(&mut water::Water) -> R) -> R {
    let mut g = WATER.lock().unwrap();
    let w = g.get_or_insert_with(|| {
        let env = |k: &str| crate::envc::var(k).ok().and_then(|v| v.parse::<f32>().ok());
        water::Water {
            wet: 0.0,
            rubber: 1.0,
            cells: std::collections::HashMap::new(),
            started: false,
            fixed: env("EVOVIEW_GRIP_WET").map(|v| v.clamp(0.0, 1.0)),
            speed: env("EVOVIEW_WEATHER_SPEED").unwrap_or(1.0).max(0.0),
            shown: -1,
            last_map: None,
            map_empty: false,
            dry_k: env("EVOVIEW_DRY_LINE_K").unwrap_or(1.0).max(0.0),
            dry_look: env("EVOVIEW_DRY_LINE_LOOK").unwrap_or(40.0).max(0.0),
            heights: std::collections::HashMap::new(),
            flow_dt: 0.0,
            flow_on: env("EVOVIEW_WATER_FLOW").map_or(true, |v| v > 0.0),
        }
    });
    f(w)
}

/// Advance the track's water by `dt` seconds of the weather set this frame
/// (set_weather). The first call starts it at the game's preset: a session
/// that opens in rain has initial_global_wetness 0.45 (gamemodeselection-
/// weathertype_rain), a dry one 0. EVOVIEW_GRIP_WET=0..1 starts it there
/// instead; EVOVIEW_WEATHER_SPEED=k runs the weather k times real time.
pub fn step_water(dt: f32) {
    let r = rain();
    let sun = crate::sky::key_dir().y.max(0.0);
    // the game's presets: scattered (0.25 cover, 0 gloom) -> rain (1.1, 0.7)
    let k = (r / 0.65).min(1.0);
    let cloud = (0.25 + 0.85 * k) * (0.75 + 0.25 * 0.7 * k);
    with_water(|w| {
        if !w.started {
            w.started = true;
            w.wet = w.fixed.unwrap_or(if r > 0.0 { 0.45 } else { 0.0 });
        }
        let dt = dt * w.speed;
        if dt <= 0.0 {
            return;
        }
        w.wet = water::step(w.wet, r, dt, sun, cloud);
        w.rubber = (w.rubber - r * dt * water::RUBBER_WASH).clamp(0.0, 1.0);
        for c in w.cells.values_mut() {
            *c = water::step(*c, r, dt, sun, cloud);
        }
        // the dry line has met the rest of the track again: forget it
        let g = w.wet;
        w.cells.retain(|_, c| (*c - g).abs() > 1e-3);
        let pct = (w.wet * 20.0).floor() as i32;
        if pct != w.shown {
            w.shown = pct;
            println!("track wetness {:.2} (rubber {:.2}, {} dry-line cells)", w.wet, w.rubber, w.cells.len());
        }
    });
}

/// How wet the track is for grip (0..1), away from the dry line.
pub fn track_wetness() -> f32 {
    with_water(|w| if w.started { w.wet } else { w.fixed.unwrap_or(0.0) })
}

/// What the track's shading shows of that: fully wet from the rain preset's
/// 0.45 up (a look choice, not the game's).
pub fn shown_wetness() -> f32 {
    shown(track_wetness())
}

fn shown(w: f32) -> f32 {
    (w / 0.45).clamp(0.0, 1.0)
}

/// Side of the dry-line map in cells (track.wgsl dry_line).
pub const DRY_DIM: usize = 256;

/// The dry line for the track shader: a DRY_DIM^2 R8 map of how much drier
/// than the rest of the track each cell around world x, z looks (0 = as wet,
/// 1 = dry), addressed by cell index wrapped at DRY_DIM, so it needs no
/// origin. Only cells within half the map of x, z are written; the shader
/// reads no further out. None when there is nothing new to upload (at most
/// ten times a second).
pub fn dry_map(x: f32, z: f32) -> Option<Vec<u8>> {
    with_water(|w| {
        let now = std::time::Instant::now();
        if w.last_map.is_some_and(|t| now.duration_since(t).as_secs_f32() < 0.1)
            || (w.cells.is_empty() && w.map_empty) {
            return None;
        }
        w.last_map = Some(now);
        w.map_empty = w.cells.is_empty();
        // 128 = as wet as the track, down to 0 = dry, up to 255 = standing
        // water the flow collected
        let mut m = vec![128u8; DRY_DIM * DRY_DIM];
        let g = w.wet;
        if g > 0.0 {
            let (cx, cz) = cell_of(x, z);
            let h = DRY_DIM as i32 / 2;
            for (&(i, j), &c) in &w.cells {
                if (i - cx).abs() >= h || (j - cz).abs() >= h {
                    continue;
                }
                // One car's pass takes only ~0.01 off (water::PER_SPEED, both
                // axles), ~2% of a wet track, which would barely show; the
                // look eases in on what was taken off instead (~1/3 after one
                // car, nearly dry after five), and is fully dry where all of
                // it is gone.
                let gone = (g - c).max(0.0);
                let d = (1.0 - (-w.dry_look * gone).exp()).max(gone / g).clamp(0.0, 1.0);
                // water the flow piled up: full puddle at +0.3 over the track
                let more = ((c - g).max(0.0) / 0.3).min(1.0);
                let k = j.rem_euclid(DRY_DIM as i32) as usize * DRY_DIM + i.rem_euclid(DRY_DIM as i32) as usize;
                m[k] = (127.5 + 127.5 * (more - d)).round().clamp(0.0, 255.0) as u8;
            }
        }
        Some(m)
    })
}

fn cell_of(x: f32, z: f32) -> (i32, i32) {
    ((x / water::CELL).floor() as i32, (z / water::CELL).floor() as i32)
}

/// Water running downhill between cells, as the game's DynamicTrack does
/// when simulate_water is on (+0x229, set for sessions; FUN_1422582c0, run
/// per region as a job). For each cell and each neighbour with a surface:
///   flux = ((w_n 0.1 + h_n) - (w_c 0.1 + h_c)) dt 0.8
/// clamped so neither cell drops under 0.05 by it, moved n -> c in place.
/// (Across region borders the game averages the two instead; evoview has no
/// regions.) The game's cells are ~0.37 m (a 6 km box / 256 regions / 64);
/// evoview's are its 1 m dry-line cells, and only those within FLOW_R of the
/// car (x, z) flow. `height(x, z)` is the track's surface there.
pub fn flow(x: f32, z: f32, dt: f32, height: impl Fn(f32, f32) -> Option<f32>) {
    with_water(|w| {
        if !w.started || !w.flow_on || w.wet <= 0.0 {
            return;
        }
        w.flow_dt += dt * w.speed;
        if w.flow_dt < water::FLOW_STEP {
            return;
        }
        let dt = std::mem::take(&mut w.flow_dt).min(0.5);
        let (cx, cz) = cell_of(x, z);
        let r = water::FLOW_R;
        if w.heights.len() > 2_000_000 {
            w.heights.clear();
        }
        for j in cz - r - 1..=cz + r + 1 {
            for i in cx - r - 1..=cx + r + 1 {
                w.heights.entry((i, j)).or_insert_with(|| {
                    height((i as f32 + 0.5) * water::CELL, (j as f32 + 0.5) * water::CELL).unwrap_or(f32::NAN)
                });
            }
        }
        let g = w.wet;
        let full = w.cells.len() >= water::MAX_CELLS;
        for j in cz - r..=cz + r {
            for i in cx - r..=cx + r {
                let hc = w.heights[&(i, j)];
                if hc.is_nan() {
                    continue;
                }
                for (ni, nj) in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)] {
                    let hn = w.heights[&(ni, nj)];
                    if hn.is_nan() {
                        continue;
                    }
                    let wc = w.cells.get(&(i, j)).copied().unwrap_or(g);
                    let wn = w.cells.get(&(ni, nj)).copied().unwrap_or(g);
                    let f = ((wn * 0.1 + hn) - (wc * 0.1 + hc)) * dt * 0.8 * water::FLOW_K;
                    let f = f.clamp(-(wc - 0.05).min(1.0).max(0.0), (wn - 0.05).min(1.0).max(0.0));
                    if f.abs() < 1e-6 || (full && (!w.cells.contains_key(&(i, j)) || !w.cells.contains_key(&(ni, nj)))) {
                        continue;
                    }
                    w.cells.insert((i, j), wc + f);
                    w.cells.insert((ni, nj), wn - f);
                }
            }
        }
    });
}

/// A tyre rolling at `speed` m/s over world x, z for `dt` s pushes the water
/// off its cell.
pub fn dry_patch(x: f32, z: f32, speed: f32, dt: f32) {
    with_water(|w| {
        if !w.started || w.wet <= 0.0 || w.cells.len() >= water::MAX_CELLS {
            return;
        }
        let g = w.wet;
        let c = w.cells.entry(cell_of(x, z)).or_insert(g);
        *c = (*c - water::PER_SPEED * w.dry_k * speed.abs() * dt).max(0.0);
    });
}

/// The water under a tyre at world x, z (0..1), the tyre model's wet input:
/// the cell's wetness (its dry line, if cars have run there), deeper in the
/// puddles the track shader draws.
pub fn patch_wetness(x: f32, z: f32) -> f32 {
    let w = with_water(|w| {
        let g = if w.started { w.wet } else { w.fixed.unwrap_or(0.0) };
        w.cells.get(&cell_of(x, z)).copied().unwrap_or(g)
    });
    if w <= 0.0 { 0.0 } else { (w * (1.0 + puddle(x, z))).min(1.0) }
}

/// The dynamic track's grip at wetness `w`, relative to a dry rubbered line:
/// green_track_grip + rubbered_grip_increase x rubber - rubbered_grip_
/// reduction_per_wetness x rubber x wetness (tyre step; 0.96, 0.04, 0.1 from
/// the global settings). The rubber starts at 1 (evoview's dry grip of 1)
/// and the rain washes it off.
pub fn rubber_grip(w: f32) -> f32 {
    const GREEN: f32 = 0.96;
    const RUBBERED: f32 = 0.04;
    const PER_WET: f32 = 0.1;
    let r = with_water(|w| w.rubber);
    (GREEN + RUBBERED * r - PER_WET * r * w.clamp(0.0, 1.0)) / (GREEN + RUBBERED)
}

/// angle history per wiper (uniform rows 3..)
pub const HIST: usize = 20;
/// seconds between history samples
pub const STEP: f32 = 0.06;
/// rows of the camera uniform's rain block
pub const ROWS: usize = 20;

pub struct Wiper {
    bone: usize,
    pivot: Vec3,
    /// rest-pose blade tip (car space) and reach
    tip: Vec3,
    radius: f32,
    /// the sweep plane's normal, found once the blade has moved
    axis: Option<Vec3>,
    hist: [f32; HIST],
    acc: f32,
}

pub struct Rain {
    pub wipers: Vec<Wiper>,
    time: f32,
    /// the side windows' streak front (Code_Rain_RSStartPos), 0..1
    streak: f32,
    speed: f32,
}

/// Side of the packed side-window streak maps (`pack_streaks`).
pub const STREAK_SIZE: u32 = 512;

/// The side windows' rain streaks (the game's Rain_RS* maps), packed as one
/// RGBA8 array of 3 layers - vertical, diagonal, horizontal flow - at
/// STREAK_SIZE: r the flow mask's arrival time, g its alpha, b the flow
/// normal's y. `masks` are the *_MASK maps (BC7), `nms` the *_NM (BC6H).
pub fn pack_streaks(masks: &[crate::load::Tex], nms: &[crate::load::Tex]) -> Option<Vec<u8>> {
    let n = STREAK_SIZE as usize;
    let mut out = vec![0u8; n * n * 4 * 3];
    for l in 0..3 {
        let m = decode(masks.get(l)?)?;
        let nm = decode(nms.get(l)?)?;
        for y in 0..n {
            for x in 0..n {
                let o = ((l * n + y) * n + x) * 4;
                // BGRA little-endian (texture2ddecoder)
                let mp = m.2[(y * m.1 as usize / n) * m.0 as usize + x * m.0 as usize / n].to_le_bytes();
                let np = nm.2[(y * nm.1 as usize / n) * nm.0 as usize + x * nm.0 as usize / n].to_le_bytes();
                out[o] = mp[2];
                out[o + 1] = mp[3];
                out[o + 2] = np[1];
                out[o + 3] = 255;
            }
        }
    }
    Some(out)
}

fn decode(t: &crate::load::Tex) -> Option<(u32, u32, Vec<u32>)> {
    let (w, h) = (t.width as usize, t.height as usize);
    let bytes = t.levels.first().map_or(t.data.len(), |l| l.bytes as usize).min(t.data.len());
    let data = &t.data[..bytes];
    let mut px = vec![0u32; w * h];
    let r = match t.format.as_str() {
        "bc7" => texture2ddecoder::decode_bc7(data, w, h, &mut px),
        "bc6h" => texture2ddecoder::decode_bc6(data, w, h, &mut px, false),
        "bc4" => texture2ddecoder::decode_bc4(data, w, h, &mut px),
        "bc1" => texture2ddecoder::decode_bc1(data, w, h, &mut px),
        _ => return None,
    };
    r.ok()?;
    Some((w as u32, h as u32, px))
}

/// What is behind the glass, for the drops to refract: the scene so far
/// (opaque, cutout, alpha) copied out before the glass draws, at the
/// render size, top-left in a texture as large as the screen.
pub struct SceneCopy {
    pub tex: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub size: (u32, u32),
    pipe: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
}

const BLIT: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(i32(i & 1u) * 4 - 1), f32(i32(i >> 1u) * 4 - 1), 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(src, vec2<i32>(p.xy), 0);
}
"#;

impl SceneCopy {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, w: u32, h: u32) -> SceneCopy {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene copy (refraction)"),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene copy"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                count: None,
            }],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("scene copy"), source: wgpu::ShaderSource::Wgsl(BLIT.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("scene copy"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene copy"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &module, entry_point: "vs", buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &module, entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        SceneCopy { tex, view, size: (w.max(1), h.max(1)), pipe, bgl }
    }

    /// Copy `src` (the scene target, `w` x `h`) into the top-left.
    pub fn copy(&self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, src: &wgpu::TextureView, w: u32, h: u32) {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene copy"),
            layout: &self.bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) }],
        });
        let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene copy"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        p.set_viewport(0.0, 0.0, w.min(self.size.0) as f32, h.min(self.size.1) as f32, 0.0, 1.0);
        p.set_pipeline(&self.pipe);
        p.set_bind_group(0, &bg, &[]);
        p.draw(0..3, 0..1);
    }
}

fn row_vec(m: &[f32; 16], p: Vec3) -> Vec3 {
    // row-major, row-vector: p' = p * M
    Vec3::new(
        p.x * m[0] + p.y * m[4] + p.z * m[8] + m[12],
        p.x * m[1] + p.y * m[5] + p.z * m[9] + m[13],
        p.x * m[2] + p.y * m[6] + p.z * m[10] + m[14],
    )
}

impl Rain {
    /// `bones`: the skeleton bones the wiper clip animates; `bind`: their
    /// rest matrices; the car's meshes give each blade's reach.
    pub fn new(car: &crate::asset::Car, bones: &[usize], bind: &[[f32; 16]], parent: &[Option<usize>],
               names: &[String]) -> Rain {
        let mut wipers = Vec::new();
        // the clip also carries static parents; the blades are the WIPER bones
        let named: Vec<usize> = bones.iter().copied()
            .filter(|&b| names.get(b).map_or(false, |n| n.to_lowercase().contains("wiper")))
            .collect();
        let bones: &[usize] = if named.is_empty() { bones } else { &named };
        // a wiper is a root of the animated set (the arm); bones under it
        // (the blade on the arm) sweep with it and count toward its reach
        let under = |mut b: usize, root: usize| -> bool {
            for _ in 0..64 {
                if b == root {
                    return true;
                }
                match parent.get(b).copied().flatten() {
                    Some(p) => b = p,
                    None => return false,
                }
            }
            false
        };
        let roots: Vec<usize> = bones.iter().copied()
            .filter(|&b| !bones.iter().any(|&o| o != b && under(b, o)))
            .collect();
        if crate::envc::var_os("EVOVIEW_RAIN_LOG").is_some() {
            println!("rain: wiper bones {:?} roots {roots:?}", bones.iter().map(|&b| names.get(b).cloned().unwrap_or_default()).collect::<Vec<_>>());
        }
        for &b in &roots {
            let set: Vec<usize> = bones.iter().copied().filter(|&o| under(o, b)).collect();
            let Some(m) = bind.get(b) else { continue };
            let pivot = Vec3::new(m[12], m[13], m[14]);
            // the vertex skinned mostly to this bone farthest from its pivot
            let mut tip = pivot;
            let mut best = 0.0f32;
            for me in &car.meshes {
                // the car's own skinned meshes (on a track the scene's meshes
                // share this list: instanced, or carrying track streams)
                // (the car itself merges in with one identity instance)
                let instanced = me.xforms.len() > 1
                    || me.xforms.first().map_or(false, |x| *x != glam::Mat4::IDENTITY.to_cols_array());
                if instanced || me.colors.is_some() || me.uv1.is_some() {
                    continue;
                }
                let (Some(_), Some(_)) = (me.bone_idx, me.bone_wgt) else { continue };
                let p = car.floats(me.positions);
                let bi = car.floats(me.bone_idx);
                let bw = car.floats(me.bone_wgt);
                for v in 0..p.len() / 3 {
                    if bi.len() < v * 4 + 4 || bw.len() < v * 4 + 4 {
                        break;
                    }
                    let on = (0..4).any(|k| set.contains(&(bi[v * 4 + k] as usize)) && bw[v * 4 + k] > 0.5);
                    if !on {
                        continue;
                    }
                    let q = Vec3::new(p[v * 3], p[v * 3 + 1], p[v * 3 + 2]);
                    let d = q.distance(pivot);
                    if d > best {
                        best = d;
                        tip = q;
                    }
                }
            }
            if best > 0.15 && best < 2.0 {
                wipers.push(Wiper { bone: b, pivot, tip, radius: best, axis: None, hist: [0.0; HIST], acc: 0.0 });
            }
        }
        if crate::envc::var_os("EVOVIEW_RAIN_LOG").is_some() {
            for me in &car.meshes {
                let uv = car.floats(me.uvs);
                let p = car.floats(me.positions);
                let idx = car.u32s(me.indices);
                for bt in &me.batches {
                    if !bt.material.to_lowercase().contains("windshield") {
                        continue;
                    }
                    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
                    let (mut plo, mut phi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                    for &i in idx.iter().skip(bt.start as usize).take(bt.count as usize) {
                        let i = i as usize;
                        if uv.len() > i * 2 + 1 {
                            lo[0] = lo[0].min(uv[i * 2]); lo[1] = lo[1].min(uv[i * 2 + 1]);
                            hi[0] = hi[0].max(uv[i * 2]); hi[1] = hi[1].max(uv[i * 2 + 1]);
                        }
                        let q = Vec3::new(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
                        plo = plo.min(q); phi = phi.max(q);
                    }
                    println!("rain: {} {} uv {:.3?}..{:.3?} size {:.2?}", me.name, bt.material, lo, hi, phi - plo);
                }
            }
        }
        // two blades at most, the longest
        wipers.sort_by(|a, b| b.radius.total_cmp(&a.radius));
        wipers.truncate(2);
        for w in &wipers {
            println!("rain: wiper bone {} pivot {:.2?} reach {:.2} m", w.bone, w.pivot, w.radius);
        }
        Rain { wipers, time: 0.0, streak: 0.0, speed: 0.0 }
    }

    /// Advance with this frame's skinning matrices.
    pub fn update(&mut self, pose: &[[f32; 16]], dt: f32, speed_kmh: f32) {
        self.time += dt;
        // the streaks' front runs along the flow faster with speed
        // (rainstreaksatspeed's 4th curve: 1 at rest .. 12 past 250 km/h)
        self.speed = speed_kmh;
        let rate = curve(&[(0.0, 1.0), (60.0, 1.0), (124.0, 6.0), (259.0, 12.0)], speed_kmh);
        self.streak = (self.streak + dt * 0.12 * rate).fract();
        for w in self.wipers.iter_mut() {
            let Some(s) = pose.get(w.bone) else { continue };
            let now = row_vec(s, w.tip) - row_vec(s, w.pivot);
            let rest = w.tip - w.pivot;
            if w.axis.is_none() {
                let c = rest.cross(now);
                if c.length() > 0.15 * rest.length() * now.length() {
                    w.axis = Some(c.normalize());
                }
            }
            let Some(ax) = w.axis else { continue };
            let flat = |v: Vec3| v - ax * v.dot(ax);
            let (r, n) = (flat(rest), flat(now));
            let ang = r.cross(n).dot(ax).atan2(r.dot(n));
            w.acc += dt;
            if w.acc >= STEP {
                w.acc -= STEP;
                w.hist.copy_within(0..HIST - 1, 1);
            }
            w.hist[0] = ang;
        }
    }

    /// The camera uniform's rain block:
    ///   0  x intensity, y drop phase 0..1, z drop amount, w wipers
    ///   per wiper (8 rows from 1): pivot + reach, axis + tracked, rest
    ///   direction, then HIST angles (newest first), 4 per row
    /// `fwd`: the car's forward direction (world); `copy`: 1 / the size of
    /// the scene copy the drops refract (0 = no copy this frame).
    pub fn block(&self, intensity: f32, speed_kmh: f32, fwd: Vec3, copy: [f32; 2]) -> [[f32; 4]; ROWS] {
        let _ = self.speed;
        let mut b = [[0.0f32; 4]; ROWS];
        if intensity <= 0.0 {
            return b;
        }
        //   17  1 / scene-copy size, refraction on, time
        //   18  car forward (world), speed km/h
        b[17] = [copy[0], copy[1], if copy[0] > 0.0 { 1.0 } else { 0.0 }, self.time];
        b[18] = [fwd.x, fwd.y, fwd.z, speed_kmh];
        //   19  side-window streaks: vertical, diagonal, horizontal weight
        //       (rainstreaksatspeed by km/h), the streak front
        b[19] = [
            curve(&[(0.0, 1.0), (33.98, 0.58), (54.0, 0.0)], speed_kmh),
            curve(&[(0.0, 0.0), (11.0, 0.0), (34.0, 0.4), (60.0, 1.0), (102.0, 0.5), (131.0, 0.0)], speed_kmh),
            curve(&[(0.0, 0.0), (73.0, 0.0), (88.0, 0.5), (120.0, 1.0)], speed_kmh),
            self.streak,
        ];
        // common_assets/rain: rainintensitystaticdrops, rainspeedstaticdrops
        // (x the speed's share of 150 km/h), windshieldrainsticky
        let amount = curve(&[(0.0, 0.0), (0.25, 0.0), (0.5, 0.1), (0.7, 0.7), (1.0, 0.85)], intensity);
        let rate = curve(&[(0.0, 0.5), (0.3, 0.5), (1.0, 1.0)], (speed_kmh / 150.0).clamp(0.0, 1.0));
        let sticky = curve(&[(30.0, 1.0), (60.0, 1.0), (100.0, 0.5), (150.0, 0.05), (250.0, 0.01)], speed_kmh);
        // ~0.35 drop generations a second at the slow end
        let phase = (self.time * 0.35 * rate).fract();
        let dbg = match crate::envc::var("EVOVIEW_RAIN_DEBUG").as_deref() { Ok("2") => 20.0, Ok(_) => 10.0, Err(_) => 0.0 };
        b[0] = [intensity, phase, (amount * sticky).max(0.05 * intensity), self.wipers.len() as f32 + dbg];
        for (i, w) in self.wipers.iter().enumerate() {
            let r = 1 + i * 8;
            let rest = (w.tip - w.pivot).normalize_or_zero();
            let ax = w.axis.unwrap_or(Vec3::ZERO);
            b[r] = [w.pivot.x, w.pivot.y, w.pivot.z, w.radius];
            b[r + 1] = [ax.x, ax.y, ax.z, if w.axis.is_some() { 1.0 } else { 0.0 }];
            b[r + 2] = [rest.x, rest.y, rest.z, 0.0];
            for k in 0..HIST {
                b[r + 3 + k / 4][k % 4] = w.hist[k];
            }
        }
        b
    }
}

/// Piecewise-linear curve through `pts` (x ascending), held at the ends.
fn curve(pts: &[(f32, f32)], x: f32) -> f32 {
    if x <= pts[0].0 {
        return pts[0].1;
    }
    for w in pts.windows(2) {
        if x <= w[1].0 {
            let t = (x - w[0].0) / (w[1].0 - w[0].0);
            return w[0].1 + (w[1].1 - w[0].1) * t;
        }
    }
    pts[pts.len() - 1].1
}

/// The packed streak maps on the GPU (`pack_streaks`); without them a
/// 1 x 1 x 3 array whose zero alpha draws no streaks.
pub fn streak_view(device: &wgpu::Device, queue: &wgpu::Queue, data: Option<&[u8]>) -> wgpu::TextureView {
    let n = if data.is_some() { STREAK_SIZE } else { 1 };
    let zero = vec![0u8; 12];
    let bytes = data.unwrap_or(&zero);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rain streaks"),
        size: wgpu::Extent3d { width: n, height: n, depth_or_array_layers: 3 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytes,
        wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(n * 4), rows_per_image: Some(n) },
        wgpu::Extent3d { width: n, height: n, depth_or_array_layers: 3 },
    );
    tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() })
}

/// The track shader's puddles (track.wgsl wet_amount) at a world x, z, for
/// flat ground: 0..1 standing water, before the ground's wetness scales it.
pub fn puddle(x: f32, z: f32) -> f32 {
    fn hash(i: [f32; 2]) -> f32 {
        let v = (i[0] * 127.1 + i[1] * 311.7).sin() * 43758.5453;
        v - v.floor()
    }
    fn vnoise(px: f32, pz: f32) -> f32 {
        let (ix, iz) = (px.floor(), pz.floor());
        let (fx, fz) = (px - ix, pz - iz);
        let (ux, uz) = (fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz));
        let a = hash([ix, iz]);
        let b = hash([ix + 1.0, iz]);
        let c = hash([ix, iz + 1.0]);
        let d = hash([ix + 1.0, iz + 1.0]);
        let ab = a + (b - a) * ux;
        let cd = c + (d - c) * ux;
        ab + (cd - ab) * uz
    }
    let n = vnoise(x * 0.35, z * 0.35) * 0.65 + vnoise(x * 1.7, z * 1.7) * 0.35;
    let t = ((n - 0.58) / 0.1).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
