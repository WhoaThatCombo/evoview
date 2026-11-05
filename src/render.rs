//! Window, device, pipeline, draw loop.
//!
//! Scene upload and pipeline creation are shared with the headless screenshot
//! path (shot.rs) so both render exactly the same thing - a screenshot that
//! did not match the window would be worthless for checking anything.

use crate::asset::Car;
use crate::material::{Blend, Materials};
use crate::camera::{Fly, Orbit};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use std::collections::HashSet;
use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    pub eye: [f32; 4],
    /// xyz ground centre, w plane height
    pub ground: [f32; 4],
    /// xy car half-extents (contact shadow), z shadow strength, w floor radius
    pub extent: [f32; 4],
    /// x = 1 outdoor / track, 0 studio / car
    /// y = 1 lit, 0 unlit albedo
    pub mode: [f32; 4],
    /// World -> light clip, for the sun shadow map.
    pub light_vp: [[f32; 4]; 4],
    /// x = shadows on, y = depth bias, z = world metres per shadow texel,
    /// w = the normal offset in metres (track shader; uniforms_shadowed).
    pub shadow: [f32; 4],
    /// x = exposure multiplier, y = 1 use the game's grading LUT (0 = ACES),
    /// z = which LUT (0 studio, 1 outdoor), w = studio softbox rig scale.
    pub grade: [f32; 4],
    /// Effects state for the frame (lights::fx_block).
    pub fx: [[f32; 4]; crate::lights::FX_VEC4],
    /// Time of day (sky::Sky::pack): key light dir + strength, key colour +
    /// sky multiplier, sky zenith colour + night, horizon colour + night IV.
    pub sky: [[f32; 4]; 4],
    /// The far shadow cascade (layer 1 of the shadow map): world -> light
    /// clip, and x on, y depth bias, z metres per texel, w radius covered.
    pub light_vp_far: [[f32; 4]; 4],
    pub shadow_far: [f32; 4],
    /// the cloud deck (clouds.rs CloudGpu::block), for cloud shadows on cars
    pub clouds: [[f32; 4]; 6],
    /// rain on the glass and the wipers (rain.rs Rain::block)
    pub rain: [[f32; 4]; crate::rain::ROWS],
    /// The middle cascade (layer 2): world -> light clip, and x on, y depth
    /// bias, z metres per texel, w normal offset (m).
    pub light_vp_mid: [[f32; 4]; 4],
    pub shadow_mid: [f32; 4],
    /// The driven car's hero map (layer 3, a HERO_DIM corner of it): world
    /// -> light clip, and x on, y depth bias, z metres per texel, w normal
    /// offset (m). The car casts into this one only.
    pub light_vp_hero: [[f32; 4]; 4],
    pub shadow_hero: [f32; 4],
    /// The AI cars' lamps (lights::ai_fx_block): their per-function
    /// intensities brake off / brake on, scaled per instance.
    pub ai_fx: [[f32; 4]; crate::lights::AI_FX_VEC4],
}

/// The sun direction the outdoor branch shades with. Kept here so the shadow
/// pass and the shader cannot drift apart - they have to agree exactly or
/// every surface self-shadows.
pub const SUN: Vec3 = Vec3::new(0.48, 0.68, 0.32);

/// How much of the world one shadow map covers, in metres.
///
/// ⚠ A circuit is kilometres across. One map over the whole thing at 4096px is
/// about a metre per texel, which cannot resolve a kerb, a post or a tree
/// trunk. So the map follows the CAMERA TARGET and covers a neighbourhood
/// instead - 0.1 m/texel at 400 m, which is what makes the shadows read as
/// contact rather than as blur.
pub const SHADOW_SPAN: f32 = 400.0;
pub const SHADOW_DIM: u32 = 4096;

/// The far cascade: 4 km at ~1 m a texel. The near map stops 200 m from
/// the camera target; without this everything past it - the hills, the
/// grandstands across the circuit, the forest - sat in flat unshadowed light.
pub const SHADOW_FAR_SPAN: f32 = 4000.0;

/// The far cascade is snapped to this grid, so it is only redrawn when the
/// camera target has moved a long way (its texels are a metre wide, and
/// re-rendering a whole circuit's casters every frame is not cheap).
pub const SHADOW_FAR_SNAP: f32 = 250.0;

/// The middle cascade: 1.2 km at ~0.3 m a texel, between the near map's
/// 0.1 m and the far one's 1 m - past the near map's ~250 m reach the road,
/// the barriers and the grandstand roofs sat in 1 m blocks.
pub const SHADOW_MID_SPAN: f32 = 1200.0;
/// Snapped like the far cascade, so it only redraws every ~100 m of travel.
pub const SHADOW_MID_SNAP: f32 = 96.0;

/// The array layers of the sun shadow map.
pub const LAYER_MID: u32 = 2;
pub const LAYER_HERO: u32 = 3;
pub const SHADOW_LAYERS: u32 = 4;

/// Whether tracks get the middle cascade (EVOVIEW_SHADOW_MID=0 turns it off).
pub fn mid_cascade() -> bool {
    cascades() && crate::envc::var("EVOVIEW_SHADOW_MID").map_or(true, |v| v != "0")
}

/// Whether the driven car gets its own hero map (EVOVIEW_DRIVE_HERO=0: the
/// car is drawn into the near cascade over the cached scenery, as before).
pub fn drive_hero() -> bool {
    crate::envc::var("EVOVIEW_DRIVE_HERO").map_or(true, |v| v != "0")
}

/// Whether tracks get the far cascade (EVOVIEW_NO_CASCADE=1 turns it off).
pub fn cascades() -> bool {
    crate::envc::var_os("EVOVIEW_NO_CASCADE").is_none()
}

/// Orthographic world -> light clip matrix, centred on `at`.
pub fn light_matrix(at: Vec3) -> glam::Mat4 {
    light_matrix_span(at, SHADOW_SPAN)
}

/// The near cascade while driving (the scenery cache): snapped to a 32 m
/// grid, so the cache is redrawn only when the cell changes (each redraw is
/// ~3 ms of GPU; at 8 m, centred 120 m out along the view, every turn of
/// the head swept it through cells and it redrew several times a second).
pub fn light_matrix_cell(at: Vec3) -> glam::Mat4 {
    let g = 32.0;
    light_matrix(Vec3::new((at.x / g).round() * g, (at.y / g).round() * g, (at.z / g).round() * g))
}

/// The far cascade's matrix, centred on `at` snapped to SHADOW_FAR_SNAP.
pub fn light_matrix_far(at: Vec3) -> glam::Mat4 {
    let s = SHADOW_FAR_SNAP;
    let at = Vec3::new((at.x / s).round() * s, (at.y / s).round() * s, (at.z / s).round() * s);
    light_matrix_span(at, SHADOW_FAR_SPAN)
}

/// The middle cascade's matrix, centred on `at` snapped to SHADOW_MID_SNAP.
pub fn light_matrix_mid(at: Vec3) -> glam::Mat4 {
    let s = SHADOW_MID_SNAP;
    let at = Vec3::new((at.x / s).round() * s, (at.y / s).round() * s, (at.z / s).round() * s);
    light_matrix_span(at, SHADOW_MID_SPAN)
}

/// Where the middle cascade goes: on the eye. Its +-600 m already covers
/// everything past the near map's ~260 m reach that is not the far map's
/// business, and a box that does not swing with the view is not redrawn
/// when the head turns (each redraw is a ~2 ms spike).
pub fn view_focus_mid(_view_proj: glam::Mat4, eye: Vec3) -> Vec3 {
    eye - Vec3::Y * 2.0
}

/// The driven car's hero map: an orthographic box round the car's bounding
/// sphere (centre `c`, radius `r`) along the sun, deep enough behind it to
/// reach the road the shadow falls on at a low sun. Drawn into a HERO_DIM
/// corner of its layer (~6 mm a texel for a GT3), centred on the car
/// itself, so the car's own shadow does not crawl as it moves.
/// Returns the matrix (to the whole box's clip space; the shader scales its
/// uv into the corner) and the texel size.
pub fn hero_light_matrix(c: Vec3, r: f32) -> (glam::Mat4, f32) {
    let dir = crate::sky::key_dir();
    let up = if dir.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
    let eye = c + dir * (r + 2.0);
    let view = glam::Mat4::look_at_rh(eye, c, up);
    let half = r + 0.2;
    let proj = glam::Mat4::orthographic_rh(-half, half, -half, half, 0.05, r + 2.0 + r + 60.0);
    (proj * view, 2.0 * half / hero_dim() as f32)
}

/// Where a track view's near shadow map goes: ahead of the eye along the
/// view, on the level of the eye's ground, so the map covers what is seen.
/// ⚠ It used to follow the ORBIT camera's target, which the free and drive
/// cameras never move: the sharp map sat at the circuit's middle and every
/// view got only the 1 m far cascade (no small casters, blurry short shade).
pub fn view_focus(view_proj: glam::Mat4, eye: Vec3) -> Vec3 {
    let inv = view_proj.inverse();
    let a = inv * glam::Vec4::new(0.0, 0.0, 0.5, 1.0);
    let f = (a.truncate() / a.w - eye).normalize_or_zero();
    let flat = Vec3::new(f.x, 0.0, f.z).normalize_or_zero();
    // the map spans 400 m: centre it ~a third of the way out, so the ground
    // at the eye's feet is still inside
    eye + flat * (SHADOW_SPAN * 0.3) - Vec3::Y * 2.0
}

pub fn light_matrix_span(at: Vec3, span: f32) -> glam::Mat4 {
    // the time of day's key light (sky.rs); SUN until one is chosen
    let dir = crate::sky::key_dir();
    let half = span * 0.5;
    // Far enough back that nothing within the span is clipped out behind the
    // light, and deep enough that tall scenery still casts.
    let depth = span.max(SHADOW_SPAN * 2.0) * 2.0;
    let up = if dir.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
    // snap the centre to whole texels in the light's own plane: a map that
    // slides by fractions of a texel as the camera moves makes every shadow
    // edge crawl
    let texel = span / SHADOW_DIM as f32;
    let rot = glam::Mat4::look_at_rh(Vec3::ZERO, -dir, up);
    let l = rot.transform_point3(at);
    let l = Vec3::new((l.x / texel).round() * texel, (l.y / texel).round() * texel, l.z);
    let at = rot.inverse().transform_point3(l);
    let eye = at + dir * depth * 0.5;
    let view = glam::Mat4::look_at_rh(eye, at, up);
    let proj = glam::Mat4::orthographic_rh(-half, half, -half, half, 0.1, depth * 1.5);
    proj * view
}

/// The showroom sun the car's hero shadow is cast from: 83 deg elevation (the
/// game's white_showroom forces it there) on the studio key light's azimuth.
/// The shader reads the direction back out of the light matrix, so changing
/// `car_sun()` (e.g. for a sky-dome car environment) is all it takes.
pub const CAR_SUN: Vec3 = Vec3::new(0.10446, 0.99255, 0.06268);

/// The sun the car hero map is cast from, in the current car environment.
pub fn car_sun() -> Vec3 {
    // The thumbnail studio's own sun (shader TH_SUN_DIR): the map has to be
    // cast from the light that th_sun() shades with, or shadows and
    // highlights disagree.
    if crate::grade::studio() == crate::grade::Studio::Thumbnail {
        TH_SUN
    } else {
        CAR_SUN
    }
}

/// Must match TH_SUN_DIR in shader.wgsl.
pub const TH_SUN: Vec3 = Vec3::new(0.25, 0.85, 0.45);

/// Edge of the car hero map. A quarter of the track map's edge: fitted to
/// one car's light-space box that is ~5 mm a texel, whose 5x5 optimized-PCF
/// footprint (~2.5 cm) reads as a soft but attached penumbra - at 4096 every
/// edge was a razor line. Measured: 1024 costs ~half of 2048 at 1080p and is
/// visually the same (see the shadows report).
pub const HERO_DIM: u32 = 1024;

/// HERO_DIM, or EVOVIEW_HERO_DIM (512..4096) for quality/perf comparisons.
pub fn hero_dim() -> u32 {
    crate::envc::var("EVOVIEW_HERO_DIM").ok().and_then(|v| v.parse().ok())
        .map(|d: u32| d.clamp(512, 4096)).unwrap_or(HERO_DIM)
}

/// Orthographic light matrix fitted around the car - the game's separate
/// high-resolution "hero" shadow map. Returns it with the world size of one
/// texel (for the shader's normal offset).
///
/// Fitted to the car's bounding BOX in light space, not its sphere: the box
/// is ~4.6 x 2.0 m in light space against a sphere diameter of ~5.2 m, so
/// texels go where the car is, and floor pixels outside the box exit early.
pub fn car_light_matrix(lo: Vec3, hi: Vec3) -> (glam::Mat4, f32) {
    let c = (lo + hi) * 0.5;
    let r = ((hi - lo).length() * 0.5).max(0.5);
    let dir = car_sun().normalize();
    let eye = c + dir * (r * 3.0);
    let view = glam::Mat4::look_at_rh(eye, c, Vec3::Z);
    let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for i in 0..8 {
        let p = Vec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        );
        let q = view.transform_point3(p);
        mn = mn.min(q);
        mx = mx.max(q);
    }
    // Margin for the shadow thrown on the floor beside the car (at 83 deg,
    // ~0.12 m per metre of height) and for the PCF taps at the edge.
    let m = 0.25;
    let half = (mx.x - mn.x).max(mx.y - mn.y) * 0.5 + m;
    let cx = (mx.x + mn.x) * 0.5;
    let cy = (mx.y + mn.y) * 0.5;
    // the view looks down -Z, so near/far are the negated z extremes
    let near = (-mx.z - 0.5).max(0.01);
    let far = -mn.z + 1.0;
    let proj = glam::Mat4::orthographic_rh(cx - half, cx + half, cy - half, cy + half, near, far);
    (proj * view, 2.0 * half / hero_dim() as f32)
}

/// Uniforms with the car hero shadow on (shadow.x = 2 tells the shader it is
/// the car map: no edge fade, normal offset by shadow.z).
pub fn uniforms_car_shadowed(u: Uniforms, lo: Vec3, hi: Vec3) -> Uniforms {
    let (m, texel) = car_light_matrix(lo, hi);
    let mut u = u;
    u.light_vp = m.to_cols_array_2d();
    u.shadow = [2.0, 0.0004, texel, 0.0];
    u
}

/// The shadow depth target and its comparison sampler.
pub fn shadow_target(device: &wgpu::Device) -> (wgpu::TextureView, wgpu::Sampler, Vec<wgpu::TextureView>) {
    shadow_target_dim(device, SHADOW_DIM)
}

/// Same, at a chosen edge size (the car hero map is HERO_DIM).
///
/// Always a 2-layer array: layer 0 is the near map (or the car hero map),
/// layer 1 the far cascade, 2 the middle one, 3 the driven car's hero map. Returns the sampling view (2D array), the
/// sampler, and one render view per layer for the shadow passes.
pub fn shadow_target_dim(device: &wgpu::Device, dim: u32)
        -> (wgpu::TextureView, wgpu::Sampler, Vec<wgpu::TextureView>) {
    let (v, s, l, _) = shadow_target_tex(device, dim);
    (v, s, l)
}

/// shadow_target_dim and the texture itself (COPY_DST, for the cached
/// static cascade copied in while driving)
pub fn shadow_target_tex(device: &wgpu::Device, dim: u32)
        -> (wgpu::TextureView, wgpu::Sampler, Vec<wgpu::TextureView>, wgpu::Texture) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow map"),
        size: wgpu::Extent3d {
            width: dim,
            height: dim,
            depth_or_array_layers: SHADOW_LAYERS,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let layers: Vec<wgpu::TextureView> = (0..SHADOW_LAYERS).map(|l| tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: l,
        array_layer_count: Some(1),
        ..Default::default()
    })).collect();
    let samp = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("shadow sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });
    (view, samp, layers, tex)
}

/// Camera constants derived from the model bounds - the contact shadow has to
/// match the car's real footprint or it reads as a sticker under the wheels.
pub fn uniforms(view_proj: glam::Mat4, e: Vec3, lo: Vec3, hi: Vec3) -> Uniforms {
    uniforms_mode(view_proj, e, lo, hi, false)
}

/// Highest mip index of the game's environment cube, or -1 when there is none.
///
/// Set once, when the materials are built, and read by every uniform update.
/// The alternative was threading it through a dozen `uniforms_*` call sites
/// that have nothing else to do with the environment.
pub static ENV_MAX_LOD: std::sync::atomic::AtomicI32 =
    std::sync::atomic::AtomicI32::new(-1);

/// Exposure + LUT selection, decided once per process.
/// The HDR scene pipeline (tracks): the scene shaders write exposed linear
/// light into an Rgba16Float target and the post pass tonemaps once, through
/// the game's LUT, after TAA, fog, bloom and auto exposure (grade.y = 2).
pub static HDR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The view is from inside the car (cockpit or dash camera).
/// seconds since the first call (EVOVIEW_MP_TRACE)
pub fn trace_clock() -> f64 {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()
}

pub static INTERIOR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The dash screens are drawn every frame (an in-car camera, the showroom);
/// false: the camera is outside a driven / replayed car and they refresh at
/// 10 Hz (fxgpu.rs update).
pub static DASH_LIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

fn grade_uniform(outdoor: bool) -> [f32; 4] {
    static G: std::sync::OnceLock<(f32, f32, bool, f32)> = std::sync::OnceLock::new();
    let &(s, o, lut, rig) = G.get_or_init(|| {
        let (s, o) = crate::grade::exposure();
        (s, o, crate::grade::use_lut(), crate::grade::rig())
    });
    let hdr = outdoor && HDR.load(std::sync::atomic::Ordering::Relaxed);
    // from inside the car the eye adapts to the darker cabin: 3/4 stop
    // back (EVOVIEW_EV_COCKPIT), or the world outside the screen reads dim
    static COCKPIT: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    let ck = if INTERIOR.load(std::sync::atomic::Ordering::Relaxed) {
        *COCKPIT.get_or_init(|| crate::envc::var("EVOVIEW_EV_COCKPIT").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.75).exp2())
    } else { 1.0 };
    [if outdoor { o * ck * crate::sky::exposure() * crate::post::ae_mult() } else { s },
     if hdr { 2.0 } else if lut { 1.0 } else { 0.0 },
     if outdoor { 1.0 } else { 0.0 }, rig]
}

/// Set when the car brought the game's own contact-shadow decals
/// (contact.rs). The analytic ellipse under the car is then switched off -
/// stacking both double-darkens the floor.
pub static CONTACT_DECALS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn uniforms_mode(view_proj: glam::Mat4, e: Vec3, lo: Vec3, hi: Vec3, outdoor: bool) -> Uniforms {
    uniforms_lit(view_proj, e, lo, hi, outdoor, outdoor)
}

pub fn uniforms_lit(
    view_proj: glam::Mat4,
    e: Vec3,
    lo: Vec3,
    hi: Vec3,
    outdoor: bool,
    lit: bool,
) -> Uniforms {
    let centre = (lo + hi) * 0.5;
    let radius = (hi - lo).length();
    let env_lod = ENV_MAX_LOD.load(std::sync::atomic::Ordering::Relaxed) as f32;
    Uniforms {
        view_proj: view_proj.to_cols_array_2d(),
        eye: [e.x, e.y, e.z, 1.0],
        ground: [centre.x, lo.y, centre.z, lo.y],
        extent: [(hi.x - lo.x) * 0.62, (hi.z - lo.z) * 0.62,
                 if CONTACT_DECALS.load(std::sync::atomic::Ordering::Relaxed) { 0.0 } else { 0.88 },
                 radius * 6.0],
        mode: [
            if outdoor { 1.0 } else { 0.0 },
            if lit { 1.0 } else { 0.0 },
            // 2 = the thumbnail studio (analytic, see shader th_*), 1 = the
            // game's cube, 0 = the analytic fallback studio
            if !outdoor && crate::grade::studio() == crate::grade::Studio::Thumbnail {
                2.0
            } else if env_lod >= 0.0 { 1.0 } else { 0.0 },
            env_lod.max(0.0),
        ],
        light_vp: light_matrix(centre).to_cols_array_2d(),
        shadow: [0.0, 0.0, SHADOW_SPAN / SHADOW_DIM as f32, SHADOW_SPAN * 0.5],
        grade: grade_uniform(outdoor),
        fx: crate::lights::fx_block(None, &crate::fxstate::FxState::default(), outdoor),
        sky: crate::sky::block(),
        light_vp_far: light_matrix_far(centre).to_cols_array_2d(),
        shadow_far: [0.0; 4],
        clouds: [[0.0; 4]; 6],
        rain: [[0.0; 4]; crate::rain::ROWS],
        light_vp_mid: light_matrix_far(centre).to_cols_array_2d(),
        shadow_mid: [0.0; 4],
        light_vp_hero: light_matrix_far(centre).to_cols_array_2d(),
        shadow_hero: [0.0; 4],
        ai_fx: [[0.0; 4]; crate::lights::AI_FX_VEC4],
    }
}

/// A cascade's constant depth bias, in its own light-depth units: ~1.5 of its
/// texels (EVOVIEW_SHADOW_BIAS_TX), over the depth range light_matrix_span
/// gives that span. The normal offset does the rest.
/// ⚠ The far cascade used a flat 0.0006 of its 12 km deep box - ~7 m, so
/// nothing under ~8 m (cars, walls, small trees, a stand's roof edge) cast
/// past the middle cascade, and the hills lay in flat light.
/// EVOVIEW_SHADOW_OLDBIAS=1: the old fixed values, for A/B.
pub fn cascade_bias(span: f32) -> f32 {
    let depth = span.max(SHADOW_SPAN * 2.0) * 2.0 * 1.5;
    let tx = crate::envc::var("EVOVIEW_SHADOW_BIAS_TX").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5f32);
    tx * (span / SHADOW_DIM as f32) / depth
}

fn old_bias() -> bool {
    crate::envc::var_os("EVOVIEW_SHADOW_OLDBIAS").is_some()
}

/// The filter a coarse cascade's shadow vec4 asks for in its x (still > 0.5
/// = on): 1 the game's 5x5 tent, 3 a 3x3 tent, 2 one bilinear compare.
/// ⚠ The 5x5 tent is 5 texels wide whatever the texel: 1.5 m of edge blur in
/// the middle cascade and 5 m in the far one, while the near map's is 0.5 m.
/// EVOVIEW_SHADOW_KERNEL="mid,far" in texels (5, 3 or 2; default 3,3).
pub fn cascade_kernel(far: bool) -> f32 {
    let v: Vec<u32> = crate::envc::var("EVOVIEW_SHADOW_KERNEL").ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    match v.get(far as usize).copied().unwrap_or(3) {
        5 => 1.0,
        2 => 2.0,
        _ => 3.0,
    }
}

/// Texels of normal offset for the sun shadow (EVOVIEW_SHADOW_NOFS).
pub fn shadow_nofs() -> f32 {
    crate::envc::var("EVOVIEW_SHADOW_NOFS").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5)
}

/// Same, but with the shadow map enabled and aimed at `focus`.
pub fn uniforms_shadowed(
    view_proj: glam::Mat4,
    e: Vec3,
    lo: Vec3,
    hi: Vec3,
    outdoor: bool,
    lit: bool,
    focus: Vec3,
) -> Uniforms {
    let mut u = uniforms_lit(view_proj, e, lo, hi, outdoor, lit);
    u.light_vp = light_matrix(focus).to_cols_array_2d();
    u.shadow[0] = 1.0;
    // Slope-scaled bias is applied in the shader; this is the constant floor.
    // ⚠ In light-depth units of a ~2400 m deep box: 0.0015 was 3.6 m, so
    // nothing shorter than that - every car, post and kerb - ever shaded
    // the ground; only buildings and trees did. 5e-5 is ~0.12 m (0.6 m at
    // grazing with the slope term). EVOVIEW_SHADOW_BIAS overrides.
    u.shadow[1] = crate::envc::var("EVOVIEW_SHADOW_BIAS").ok().and_then(|v| v.parse().ok())
        .unwrap_or(if old_bias() { 0.00005 } else { cascade_bias(SHADOW_SPAN) });
    // w: the normal offset the track shader samples at, metres - this
    // many texels along the normal at grazing sun (the game's resolve,
    // normalOffsetScale); EVOVIEW_SHADOW_NOFS
    let nofs = shadow_nofs();
    u.shadow[3] = nofs * u.shadow[2];
    if outdoor && cascades() {
        u.light_vp_far = light_matrix_far(focus).to_cols_array_2d();
        // ~1.5 of its ~1 m texels, in its 12 km deep box's units (cascade_bias)
        let t = SHADOW_FAR_SPAN / SHADOW_DIM as f32;
        u.shadow_far = [cascade_kernel(true), if old_bias() { 0.0006 } else { cascade_bias(SHADOW_FAR_SPAN) }, t, nofs * t];
    }
    u
}

/// uniforms_shadowed with the middle cascade aimed at `focus_mid`.
pub fn uniforms_shadowed_mid(u: Uniforms, focus_mid: Vec3) -> Uniforms {
    let mut u = u;
    if u.shadow_far[0] > 0.5 && mid_cascade() {
        u.light_vp_mid = light_matrix_mid(focus_mid).to_cols_array_2d();
        // ~1.5 of its 0.3 m texels in its 3.6 km deep box's units (cascade_bias)
        let t = SHADOW_MID_SPAN / SHADOW_DIM as f32;
        u.shadow_mid = [cascade_kernel(false), if old_bias() { 0.00012 } else { cascade_bias(SHADOW_MID_SPAN) }, t, shadow_nofs() * t];
    }
    u
}

/// 5th–95th percentile of sampled vertices, so one runaway mesh cannot
/// pull the camera to the moon.
pub fn robust_bounds(car: &Car) -> Option<(Vec3, Vec3)> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    for m in &car.meshes {
        // coarser LODs repeat LOD0's placements; counting them again would
        // shift the percentiles and with them the start camera
        if m.lod.map_or(false, |l| l[0] > 0.0) {
            continue;
        }
        if !m.xforms.is_empty() {
            let step = (m.xforms.len() / 64).max(1);
            for xf in m.xforms.iter().step_by(step) {
                xs.push(xf[12]);
                ys.push(xf[13]);
                zs.push(xf[14]);
            }
            continue;
        }
        let pos: Vec<f32> = car.floats(m.positions);
        let n = pos.len() / 3;
        if n == 0 {
            continue;
        }
        let step = (n / 48).max(1);
        for i in (0..n).step_by(step) {
            xs.push(pos[i * 3]);
            ys.push(pos[i * 3 + 1]);
            zs.push(pos[i * 3 + 2]);
        }
    }
    if xs.len() < 16 {
        return None;
    }
    // ⚠ `partial_cmp().unwrap_or(Equal)` is NOT a total order once a NaN is in
    // the data: NaN compares Equal to everything, so `a < b`, `b < c` and
    // `a == c` can all hold at once. Rust's sort detects that and PANICS
    // ("user-provided comparison function does not correctly implement a total
    // order"), which made the `interns` track fail to open at all. `total_cmp`
    // is a real total order over every f32 including NaN.
    let pct = |v: &mut Vec<f32>, t: f32| -> f32 {
        v.retain(|x| x.is_finite());
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.total_cmp(b));
        v[(((v.len() - 1) as f32) * t) as usize]
    };
    Some((
        Vec3::new(pct(&mut xs, 0.12), pct(&mut ys, 0.12), pct(&mut zs, 0.12)),
        Vec3::new(pct(&mut xs, 0.88), pct(&mut ys, 0.88), pct(&mut zs, 0.88)),
    ))
}

/// A large quad under the car. Built here rather than exported: it is scenery,
/// not car data.
pub fn ground_quad(device: &wgpu::Device, lo: Vec3, hi: Vec3) -> GpuMesh {
    let c = (lo + hi) * 0.5;
    let r = (hi - lo).length() * 14.0;
    let y = lo.y;
    let mut v = [0f32; 4 * FLOATS_PER_VERTEX];
    let corners = [
        (c.x - r, c.z - r, 0.0, 0.0),
        (c.x + r, c.z - r, 1.0, 0.0),
        (c.x + r, c.z + r, 1.0, 1.0),
        (c.x - r, c.z + r, 0.0, 1.0),
    ];
    for (i, (px, pz, u, vv)) in corners.iter().enumerate() {
        let o = i * FLOATS_PER_VERTEX;
        v[o] = *px;
        v[o + 1] = y;
        v[o + 2] = *pz;
        v[o + 4] = 1.0; // normal +Y
        v[o + 6] = *u;
        v[o + 7] = *vv;
        v[o + 8] = 1.0; // tangent +X
        v[o + 11] = 1.0;
    }
    let i: [u32; 6] = [0, 1, 2, 0, 2, 3];
    let identity = pack_instances(&[], 0);
    GpuMesh {
        vertices: std::sync::Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ground"),
            contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::VERTEX,
        })),
        indices: std::sync::Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ground"),
            contents: bytemuck::cast_slice(&i),
            usage: wgpu::BufferUsages::INDEX,
        })),
        instances: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ground-inst"),
            contents: bytemuck::cast_slice(&identity),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        instance_count: 1,
        index_count: 6,
        draws: vec![],
        src: usize::MAX,
        extra: None,
        radius: 0.0,
    }
}

/// One uploaded mesh. Attributes are interleaved once at load: the game stores
/// them as separate arrays, the GPU prefers a single stream.
pub struct GpuMesh {
    /// Shared between meshes with identical contents (`car_geometry`):
    /// every AI / online look of one model is the same geometry.
    pub vertices: std::sync::Arc<wgpu::Buffer>,
    pub indices: std::sync::Arc<wgpu::Buffer>,
    pub instances: wgpu::Buffer,
    pub instance_count: u32,
    pub index_count: u32,
    /// (first_index, index_count, material bind-group index) - one draw each
    pub draws: Vec<(u32, u32, usize)>,
    /// Index of the source mesh in `Car::meshes`.
    ///
    /// ⚠ build_scene drops meshes (rim_blur, empty ones), so this is NOT the
    /// position in the returned list. Replay playback addresses cars by source
    /// index and needs the mapping back. `usize::MAX` for meshes with no
    /// source, like the studio floor.
    pub src: usize,
    /// Track meshes only: vertex colour (4) + uv1 (2) per vertex, the second
    /// vertex stream of the track pipeline. None = not track geometry.
    pub extra: Option<wgpu::Buffer>,
    /// The farthest vertex from the mesh origin (m, unskinned): carcull.rs
    /// bounds each moving car's instance with it. 0 where unused.
    pub radius: f32,
}

pub const FLOATS_PER_VERTEX: usize = 20; // pos3 nrm3 uv2 tan4 boneidx4 bonewgt4

/// A car mesh's vertex and index buffers, shared with any mesh already on
/// the GPU whose contents are the same. Each AI look and each online player
/// is its own slot of meshes, and a grid of one model held one copy of its
/// geometry per look (~30-40 MB for a GT3 each). Keyed by a 128-bit hash of
/// the final vertex and index data; held weakly, so a geometry nobody draws
/// any more is freed. Track meshes are unique and are not looked up.
fn car_geometry(device: &wgpu::Device, label: &str, verts: &[f32], idx: &[u32])
        -> (std::sync::Arc<wgpu::Buffer>, std::sync::Arc<wgpu::Buffer>) {
    use std::sync::{Arc, Mutex, Weak};
    type Geom = (Weak<wgpu::Buffer>, Weak<wgpu::Buffer>);
    static CACHE: Mutex<Option<std::collections::HashMap<(u64, u64, usize, usize), Geom>>> = Mutex::new(None);
    // two multiplicative lanes over the bytes, 8 at a time
    let mut h = (0x9e37_79b9_7f4a_7c15u64, 0xc2b2_ae3d_27d4_eb4fu64);
    let mut eat = |b: &[u8]| {
        let mut ch = b.chunks_exact(8);
        for c in &mut ch {
            let x = u64::from_le_bytes(c.try_into().unwrap());
            h.0 = (h.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
            h.1 = (h.1.rotate_left(7) ^ x).wrapping_mul(0x9e37_79b1_85eb_ca87);
        }
        for &x in ch.remainder() {
            h.0 = (h.0.rotate_left(5) ^ x as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
            h.1 = (h.1.rotate_left(7) ^ x as u64).wrapping_mul(0x9e37_79b1_85eb_ca87);
        }
    };
    eat(bytemuck::cast_slice(verts));
    eat(bytemuck::cast_slice(idx));
    let key = (h.0, h.1, verts.len(), idx.len());
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let map = cache.get_or_insert_with(Default::default);
    if let Some((v, i)) = map.get(&key).and_then(|(v, i)| Some((v.upgrade()?, i.upgrade()?))) {
        return (v, i);
    }
    let v = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(verts),
        usage: wgpu::BufferUsages::VERTEX,
    }));
    let i = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(idx),
        usage: wgpu::BufferUsages::INDEX,
    }));
    map.retain(|_, (v, _)| v.strong_count() > 0);
    map.insert(key, (Arc::downgrade(&v), Arc::downgrade(&i)));
    (v, i)
}

/// Instance buffer: 16 floats of column-major world matrix + 4-float facing.
pub fn pack_instances(xforms: &[[f32; 16]], facing: u32) -> Vec<f32> {
    let f = facing as f32;
    if xforms.is_empty() {
        let mut v = vec![0.0f32; 20];
        v[0] = 1.0;
        v[5] = 1.0;
        v[10] = 1.0;
        v[15] = 1.0;
        v[16] = f;
        return v;
    }
    let mut out = Vec::with_capacity(xforms.len() * 20);
    for xf in xforms {
        out.extend_from_slice(xf);
        out.extend_from_slice(&[f, 0.0, 0.0, 0.0]);
    }
    out
}
/// Matches MAX_BONES in shader.wgsl.
///
/// ⚠ Not "one body's bones". Every mesh carries its OWN skeleton and they are
/// merged by name into one car-wide table, so the body's ~85 plus the
/// interior, the belts and the steering wheel comfortably pass 128.
pub const MAX_BONES: usize = 256;

/// Upload every mesh, returning them plus the model bounds and triangle count.
pub fn build_scene(
    device: &wgpu::Device,
    car: &Car,
    mats: Option<&Materials>,
) -> (Vec<GpuMesh>, Vec3, Vec3, u32) {
    let mut meshes = Vec::new();
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    let mut tris = 0u32;

    for (src_i, m) in car.meshes.iter().enumerate() {
        // The rim_blur meshes are the spinning-wheel stand-ins: a smeared disc
        // the game swaps in above a speed threshold. Sitting still they overlay
        // the real rims and read as damage.
        if m.name.contains("rim_blur") || m.name.contains("_belt_off") {
            continue;
        }
        let pos: Vec<f32> = car.floats(m.positions);
        let nrm: Vec<f32> = car.floats(m.normals);
        let uvs: Vec<f32> = car.floats(m.uvs);
        let tan: Vec<f32> = car.floats(m.tangents);
        // The engine stores bone indices as f32, so both skin arrays are just
        // four floats per vertex.
        let bidx: Vec<f32> = car.floats(m.bone_idx);
        let bwgt: Vec<f32> = car.floats(m.bone_wgt);
        let idx: Vec<u32> = car.u32s(m.indices);
        let n = m.vertices as usize;
        if n == 0 || idx.is_empty() || pos.len() < n * 3 {
            continue;
        }
        if m.xforms.is_empty() {
            // Instanced props live at the origin in mesh space; their
            // translations are applied on the GPU.
        }
        let mut verts = vec![0f32; n * FLOATS_PER_VERTEX];
        for i in 0..n {
            let p = [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
            // Contact-shadow decals are wider than the car; framing on them
            // would move the camera and the floor.
            if m.xforms.is_empty() && !m.name.starts_with("contact_") {
                lo = lo.min(Vec3::from(p));
                hi = hi.max(Vec3::from(p));
            }
            let o = i * FLOATS_PER_VERTEX;
            verts[o..o + 3].copy_from_slice(&p);
            // Normals and UVs are optional; fall back rather than drop a mesh.
            if nrm.len() >= n * 3 {
                verts[o + 3] = nrm[i * 3];
                verts[o + 4] = nrm[i * 3 + 1];
                verts[o + 5] = nrm[i * 3 + 2];
            } else {
                verts[o + 4] = 1.0;
            }
            if uvs.len() >= n * 2 {
                verts[o + 6] = uvs[i * 2];
                verts[o + 7] = uvs[i * 2 + 1];
            }
            // Real tangents, when the mesh has them. Deriving a tangent frame
            // from screen-space derivatives instead falls apart wherever the
            // UV derivative is near-degenerate - dense tread blocks, headlight
            // slats - and the normal map turns to salt-and-pepper noise.
            if tan.len() >= n * 4 {
                verts[o + 8] = tan[i * 4];
                verts[o + 9] = tan[i * 4 + 1];
                verts[o + 10] = tan[i * 4 + 2];
                verts[o + 11] = tan[i * 4 + 3];
            }
            // Unskinned geometry (tyres, the studio floor) keeps zero weights,
            // which the shader reads as "leave this vertex alone".
            if bidx.len() >= n * 4 && bwgt.len() >= n * 4 {
                for k in 0..4 {
                    verts[o + 12 + k] = bidx[i * 4 + k];
                    verts[o + 16 + k] = bwgt[i * 4 + k];
                }
            }
        }
        tris += (idx.len() / 3) as u32;
        // One draw per material batch. A mesh with no batches is drawn whole
        // with the fallback material rather than dropped.
        let mut draws: Vec<(u32, u32, usize)> = m
            .batches
            .iter()
            .filter(|b| b.count > 0 && (b.start + b.count) as usize <= idx.len())
            .map(|b| {
                let mi = mats.map(|x| x.lookup(&b.material)).unwrap_or(0);
                (b.start, b.count, mi)
            })
            .collect();
        if draws.is_empty() {
            draws.push((0, idx.len() as u32, mats.map(|x| x.fallback).unwrap_or(0)));
        }
        // Track meshes carry a second vertex stream (vertex colour + uv1)
        // for the track pipeline.
        let mut extra = None;
        if m.colors.is_some() || m.uv1.is_some() {
            let col: Vec<f32> = car.floats(m.colors);
            let uv1: Vec<f32> = car.floats(m.uv1);
            let mut e = vec![0f32; n * crate::track::EXTRA_FLOATS];
            for i in 0..n {
                let o = i * crate::track::EXTRA_FLOATS;
                if col.len() >= n * 4 {
                    e[o..o + 4].copy_from_slice(&col[i * 4..i * 4 + 4]);
                } else {
                    e[o..o + 4].copy_from_slice(&[1.0 / 255.0; 4]);
                }
                if uv1.len() >= n * 2 {
                    e[o + 4] = uv1[i * 2];
                    e[o + 5] = uv1[i * 2 + 1];
                }
            }
            extra = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{}-extra", m.name)),
                contents: bytemuck::cast_slice(&e),
                usage: wgpu::BufferUsages::VERTEX,
            }));
        }
        let mut inst_data = pack_instances(&m.xforms, m.facing);
        crate::track::skin_rows(car, src_i, &mut inst_data);
        if car.track_env.is_some() {
            crate::track::wind_rows(car, src_i, &mut inst_data);
        }
        if !m.xforms.is_empty() {
            for xf in m.xforms.iter().step_by((m.xforms.len() / 32).max(1)) {
                let p = Vec3::new(xf[12], xf[13], xf[14]);
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        // car geometry is shared between identical meshes (car_geometry)
        let (vbuf, ibuf) = if extra.is_none() && crate::envc::var("EVOVIEW_SHARE_GEOMETRY").map_or(true, |v| v != "0") {
            car_geometry(device, &m.name, &verts, &idx)
        } else {
            (std::sync::Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&m.name),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            })),
             std::sync::Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&m.name),
                contents: bytemuck::cast_slice(&idx),
                usage: wgpu::BufferUsages::INDEX,
            })))
        };
        meshes.push(GpuMesh {
            draws,
            vertices: vbuf,
            indices: ibuf,
            instances: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{}-inst", m.name)),
                contents: bytemuck::cast_slice(&inst_data),
                // COPY_DST so replay playback can rewrite car transforms every
                // frame; static scenery simply never writes to it. COPY_SRC:
                // motion.rs keeps last frame's copy
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            }),
            // A coarser LOD draws nothing until a culler assigns it
            // instances; paths without one (shots) draw LOD0 everywhere.
            instance_count: if m.lod.map_or(false, |l| l[0] > 0.0) {
                0
            } else {
                m.xforms.len().max(1) as u32
            },
            index_count: idx.len() as u32,
            src: src_i,
            extra,
            radius: verts.chunks_exact(FLOATS_PER_VERTEX)
                .map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).fold(0.0f32, f32::max).sqrt(),
        });
    }
    // A machine-readable line so a sweep over every car can tell a real
    // failure from an ugly render. Batches that fell back are the honest
    // signal that a material did not bind.
    if let Some(m) = mats {
        let fb: usize = meshes
            .iter()
            .flat_map(|x| x.draws.iter())
            .filter(|d| d.2 == m.fallback)
            .count();
        println!("scene: {} mesh(es), {} triangles, {} fallback batch(es)",
                 meshes.len(), tris, fb);
        if fb > 0 {
            // name what failed to bind, not just how many: "4 fallback
            // batches" is not actionable, "EXT_GLASS.material" is
            let mut names: Vec<String> = car
                .meshes
                .iter()
                .flat_map(|mesh| mesh.batches.iter())
                .filter(|b| m.lookup(&b.material) == m.fallback)
                .map(|b| {
                    b.material
                        .rsplit(['\\', '/'])
                        .next()
                        .unwrap_or(&b.material)
                        .to_string()
                })
                .collect();
            names.sort();
            names.dedup();
            println!("unbound materials: {}", names.join(", "));
        }
    }
    (meshes, lo, hi, tris)
}

/// Shading switches both material shaders (shader.wgsl, track.wgsl) read as
/// WGSL constants, so before/after can be compared without a uniform.
/// EVOVIEW_BITANGENT=old: green along w*cross(N,T) (+dP/dv of the mesh
/// data, MikkTSpace) instead of the game's w*cross(T,N) (lifts 034e5ce0,
/// 035a3710, road 034c6530: green = -dP/dv on 98% of Imola's triangles).
/// EVOVIEW_NORMAL_COFACTOR=0: normals through the plain model matrix, as the
/// game's vertex shader does (wrong under non-uniform scale).
/// EVOVIEW_GSAA=0: no specular anti-aliasing (roughness widening + spec fade).
pub fn shading_wgsl() -> String {
    let off = |k: &str| crate::envc::var(k).map_or(false, |v| v == "0" || v == "old");
    // EVOVIEW_GSAA_CAR=var,thr,max: the frameData vehicleGsaaVariance /
    // Threshold / RoughnessMax multipliers (runtime values, not decoded; the
    // material defaults Gsaa_Variance 0.5 / Threshold 1 and a 2x cap)
    let car: Vec<f32> = crate::envc::var("EVOVIEW_GSAA_CAR").ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<f32>| v.len() == 3)
        .unwrap_or_else(|| vec![0.5, 1.0, 2.0]);
    format!("const BT_GAME: f32 = {:.1};\nconst NRM_COF: bool = {};\nconst GSAA: bool = {};\n\
             const GSAA_CAR: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n{}",
            if off("EVOVIEW_BITANGENT") { -1.0 } else { 1.0 },
            !off("EVOVIEW_NORMAL_COFACTOR"),
            !off("EVOVIEW_GSAA"),
            car[0], car[1], car[2],
            include_str!("shading_common.wgsl"))
}

/// The car shader as built: the cloud functions appended, and on a track
/// (HDR, where TAA resolves) the TAA texture LOD bias.
fn car_wgsl(src: &str) -> String {
    let s = crate::clouds::car_source(src);
    if HDR.load(std::sync::atomic::Ordering::Relaxed) {
        crate::taa::with_mip_bias(&s, crate::taa::mip_bias())
    } else {
        s
    }
}

pub fn build_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    uniform_buf: &wgpu::Buffer,
    samples: u32,
    material_layout: &wgpu::BindGroupLayout,
    bone_buf: &wgpu::Buffer,
    env_view: &wgpu::TextureView,
    env_samp: &wgpu::Sampler,
    brdf_view: &wgpu::TextureView,
    lut_view: &wgpu::TextureView,
    shadow_view: &wgpu::TextureView,
    shadow_samp: &wgpu::Sampler,
    light_buf: &wgpu::Buffer,
    clouds: &crate::clouds::CloudGpu,
    probe: Option<&wgpu::TextureView>,
    rain: [Option<&wgpu::TextureView>; 3],
    scene_copy: Option<&wgpu::TextureView>,
    streaks: &wgpu::TextureView,
) -> Pipelines {
    let copy_view = scene_copy.unwrap_or(brdf_view);
    // the windshield rain maps (rain.rs); any 2D texture stands in
    let rain_views: [&wgpu::TextureView; 3] = [rain[0].unwrap_or(brdf_view), rain[1].unwrap_or(brdf_view), rain[2].unwrap_or(brdf_view)];
    // the reflection probe (probe.rs); any cube stands in where there is none
    let probe_view = probe.unwrap_or(env_view);
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // The environment is global, not per-material, so it lives with
            // the camera rather than being rebound for every draw.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
            // The game's grading LUT, see grade.rs.
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            // the cloud deck (clouds.rs): noise slices, sampler, weather map
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            // rain_drop_mask / _alpha / _nm
            wgpu::BindGroupLayoutEntry { binding: 11, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 12, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 13, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            // the scene behind the glass, for refraction (rain.rs SceneCopy)
            wgpu::BindGroupLayoutEntry { binding: 14, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            // the side windows' streak maps (rain::pack_streaks), 3 layers
            wgpu::BindGroupLayoutEntry { binding: 15, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false }, count: None },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("camera bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(env_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(env_samp),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(brdf_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(shadow_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(shadow_samp),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(lut_view),
            },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&clouds.noise) },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&clouds.samp) },
            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&clouds.weather) },
            wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(probe_view) },
            wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(rain_views[0]) },
            wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(rain_views[1]) },
            wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(rain_views[2]) },
            wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(copy_view) },
            wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(streaks) },
        ],
    });
    // ⚠ The shadow pass reuses `vs_main` unchanged by binding a DIFFERENT
    // uniform buffer whose `view_proj` is the light matrix. That keeps skinning
    // and instancing identical between the two passes for free - a separate
    // depth-only vertex shader would be a second place for them to drift.
    //
    // ⚠ It must NOT bind the real shadow map at slot 4: that pass WRITES the
    // map as its depth attachment, and a texture cannot be sampled and
    // depth-written in the same pass ("conflicting usages ... DEPTH_STENCIL_WRITE
    // is an exclusive usage"). It is never sampled there - the pass has no
    // fragment stage - but the binding alone is enough to fail validation, so
    // it gets a 1x1 stand-in.
    let dummy_shadow = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow stand-in"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let dummy_shadow_view = dummy_shadow.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    // the far, middle and hero cascades' light cameras: same layout, their
    // own buffers
    let light_bg = |label: &str| {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(env_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(env_samp) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(brdf_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&dummy_shadow_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(shadow_samp) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(lut_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&clouds.noise) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&clouds.samp) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&clouds.weather) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(probe_view) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(rain_views[0]) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(rain_views[1]) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(rain_views[2]) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(copy_view) },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(streaks) },
            ],
        });
        (buf, bg)
    };
    let (light_buf_far, light_bind_group_far) = light_bg("light camera (far cascade)");
    let light_x = [light_bg("light camera (mid cascade)"), light_bg("light camera (hero)")];
    let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("light bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(env_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(env_samp) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(brdf_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&dummy_shadow_view) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(shadow_samp) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(lut_view) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&clouds.noise) },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&clouds.samp) },
            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&clouds.weather) },
            wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(probe_view) },
            wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(rain_views[0]) },
            wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(rain_views[1]) },
            wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(rain_views[2]) },
            wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(copy_view) },
            wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(streaks) },
        ],
    });
    let bone_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bone bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bone_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bone bg"),
        layout: &bone_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: bone_buf.as_entire_binding(),
        }],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader"),
        // EVOVIEW_SHADER=<path>: load the WGSL from disk instead of the
        // built-in copy, so shading can be tuned without a rebuild.
        source: wgpu::ShaderSource::Wgsl(
            match crate::envc::var("EVOVIEW_SHADER").ok().and_then(|p| std::fs::read_to_string(p).ok()) {
                Some(src) => car_wgsl(&src).into(),
                None => car_wgsl(include_str!("shader.wgsl")).into(),
            }),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("layout"),
        bind_group_layouts: &[&bgl, material_layout, &bone_bgl],
        push_constant_ranges: &[],
    });
    // Two pipelines off one shader. The only differences are the blend state
    // and depth write: transparent surfaces must not occlude what is behind
    // them, which for a car means the lamp reflectors inside the headlight and
    // the interior through the glass.
    // `ssr`: the main pass's SSR targets after the colour (ssr.rs): Some(true)
    // where the material writes them; see-through surfaces attenuate them
    // (ssr::Behind) - the alpha-blended by coverage, the glass by its tint
    let make_f = |label, entry: &str, blend: Option<wgpu::BlendState>, depth_write, format, cmp, ssr: Option<bool>| {
        let mut targets = vec![Some(wgpu::ColorTargetState {
            format,
            blend,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let behind = match (entry, ssr) {
            ("fs_glass_tint", Some(_)) => Some(crate::ssr::Behind::Tint),
            ("fs_main", Some(false)) if blend == Some(wgpu::BlendState::ALPHA_BLENDING) => Some(crate::ssr::Behind::Cover),
            _ => None,
        };
        match (behind, ssr) {
            (Some(b), _) => targets.extend(crate::ssr::behind_targets(b)),
            (None, Some(w)) => targets.extend(crate::ssr::extra_targets(w)),
            (None, None) => {}
        }
        // the SSR entries only when there are SSR targets to write
        let entry = match (entry, ssr == Some(true), behind) {
            _ if !crate::ssr::active() => entry,
            ("fs_main", true, _) => "fs_main_ssr",
            (_, _, Some(crate::ssr::Behind::Tint)) => "fs_glass_tint_ssr",
            (_, _, Some(crate::ssr::Behind::Cover)) => "fs_main_cover",
            _ => entry,
        };
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: (FLOATS_PER_VERTEX * 4) as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4,
                            4 => Float32x4, 5 => Float32x4],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 80,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Float32x4,
                            10 => Float32x4],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: entry,
                compilation_options: Default::default(),
                targets: &targets,
            }),
            primitive: wgpu::PrimitiveState {
                // Car meshes carry plenty of single-sided trim; culling stays
                // off until the winding convention is confirmed against the
                // game.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: depth_write,
                depth_compare: cmp,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        })
    };
    let make = |label, entry, blend, depth_write| make_f(label, entry, blend, depth_write, format, wgpu::CompareFunction::Less, Some(false));
    let pipeline = make_f("opaque", "fs_main", None, true, format, wgpu::CompareFunction::Less, Some(true));
    // Bloom source (fxgpu.rs): emission only, HDR, on the main pass's depth.
    let glow = make_f("glow", "fs_glow", Some(wgpu::BlendState {
        color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
        alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
    }), false, crate::fxgpu::GLOW_FORMAT, wgpu::CompareFunction::LessEqual, None);
    let blended = make("blended", "fs_main", Some(wgpu::BlendState::ALPHA_BLENDING), false);
    // Glass, EVO's way (SPEC_evo_shading 4.1): dst = reflection + dst * tint.
    // The game does it in one draw with dual-source blending; two draws of the
    // same few triangles do the same without needing that device feature.
    let keep_alpha = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    let glass_tint = make("glass tint", "fs_glass_tint", Some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::Src,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: keep_alpha,
    }), false);
    let glass_add = make("glass add", "fs_main", Some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: keep_alpha,
    }), false);
    // Depth-only, single-sampled, front faces culled. Culling the FRONT rather
    // than the back pushes the stored depth to the far side of a solid, which
    // is the cheapest way to keep near-flat ground from shadowing itself.
    let shadow_with = |label: &str, cull: Option<wgpu::Face>| device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            compilation_options: Default::default(),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: (FLOATS_PER_VERTEX * 4) as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4,
                        4 => Float32x4, 5 => Float32x4],
                },
                wgpu::VertexBufferLayout {
                    array_stride: 80,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Float32x4,
                        10 => Float32x4],
                },
            ],
        },
        fragment: None,
        primitive: wgpu::PrimitiveState {
            cull_mode: cull,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
        }),
        multisample: Default::default(),
        multiview: None,
        cache: None,
    });
    // ⚠ Track casters are two-sided as well: with the front faces culled a
    // single-sided panel facing the sun - a roof, a canopy, a board, a
    // bridge deck - left nothing in the map and cast no shadow. The depth
    // that put on every sunlit surface is covered by the bias (cascade_bias
    // + the normal offset + this pass's slope-scaled bias).
    // EVOVIEW_SHADOW_CULL=front: the old front-face culling, for A/B.
    let track_cull = crate::envc::var("EVOVIEW_SHADOW_CULL").map_or(false, |v| v == "front")
        .then_some(wgpu::Face::Front);
    let shadow_pipeline = shadow_with("shadow", track_cull);
    // ⚠ Cars cast two-sided: with the front faces culled only the underbody
    // was left, ~15 cm off the road - inside the bias - so cars cast nothing.
    let shadow_car = shadow_with("shadow (cars)", None);
    Pipelines { pipeline, blended, glass_tint, glass_add, shadow: shadow_pipeline, shadow_car, bind_group,
                light_bind_group, light_buf_far, light_bind_group_far, light_x, bone_bg, glow }
}

/// What `build_pipeline` hands back. Grew past what a tuple should carry once
/// the shadow pass needed its own pipeline and camera binding.
pub struct Pipelines {
    pub pipeline: wgpu::RenderPipeline,
    pub blended: wgpu::RenderPipeline,
    /// glass pass 1 (multiply by tint) and pass 2 (add reflection)
    pub glass_tint: wgpu::RenderPipeline,
    pub glass_add: wgpu::RenderPipeline,
    pub shadow: wgpu::RenderPipeline,
    pub shadow_car: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub light_bind_group: wgpu::BindGroup,
    /// the far shadow cascade's light camera
    pub light_buf_far: wgpu::Buffer,
    pub light_bind_group_far: wgpu::BindGroup,
    /// the middle cascade's and the driven car's hero map's light cameras
    pub light_x: [(wgpu::Buffer, wgpu::BindGroup); 2],
    pub bone_bg: wgpu::BindGroup,
    /// emissive-only pass into the bloom buffer (fxgpu.rs)
    pub glow: wgpu::RenderPipeline,
}

/// Everything needed to animate a replay: the motion data, and which merged
/// meshes carry which drivers' instances.
pub struct Playback {
    pub replay: crate::replay::Replay,
    pub slots: Vec<crate::replayscene::Slot>,
    pub wheels: crate::replayscene::Wheels,
    /// where playback starts, seconds (--time)
    pub start: f32,
}

pub struct ViewOpts {
    pub start_fly: bool,
    pub ground: bool,
    pub title: String,
    /// Present for a replay; the car instances are then rewritten every frame.
    pub playback: Option<Playback>,
    /// Where to put the camera on load. A replay opens on the grid, which is
    /// nowhere near the middle of the track the generic framing would pick.
    pub focus: Option<Vec3>,
    /// content.kspkg, so engine banks can be read for the car being watched.
    pub pkg: Option<std::path::PathBuf>,
    /// `--bench`: fixed orbit, uncapped, prints stats and exits.
    pub bench: Option<crate::bench::Opts>,
    /// The car kept open so V / Shift+V (and `livery ...` on stdin) can
    /// switch its livery (livery.rs).
    pub livery: Option<crate::livery::Switcher>,
    /// 0 off, 1 keyboard drive, 2 scripted demo loop, 3 replay feed (drive.rs)
    pub drive: u8,
    /// `--replay-car`: one recorded car drives the rig
    pub replay_feed: Option<crate::drive::ReplayFeed>,
    /// `--track <id> --drive-car <car>`: one car driven on the track
    /// (trackdrive.rs); its slot's instances are rewritten every frame.
    pub track_drive: Option<crate::trackdrive::TrackDrive>,
    /// the track's TV cameras for the driven car (cams.rs)
    pub tv: Option<crate::cams::TvDirector>,
    /// EVOVIEW_DRIVE_LINE=1: the racing-line driver drives instead of the keys
    pub line_driver: Option<crate::trackdrive::LineDriver>,
    /// the driven car's id (its camera settings in Saved Games)
    pub car_id: Option<String>,
    /// the driven car's pit lane, stop and garage (pits.rs)
    pub pits: Option<crate::pits::Pits>,
    /// the driven car's lap timing and the leaderboard (laps.rs)
    pub laps: Option<crate::laps::Session>,
    /// `--ai N`: the opponents (race.rs), and their render slots (the
    /// scene's slots 1..; the AI's `slot` field counts from 1)
    pub race: Option<crate::race::Race>,
    pub ai_slots: Vec<crate::replayscene::Slot>,
    /// EVOVIEW_MP: online (mp.rs); the race's cars are the other players
    pub mp: Option<crate::mp::Mp>,
}

impl Default for ViewOpts {
    fn default() -> Self {
        Self {
            start_fly: false,
            ground: true,
            title: "evoview - AC EVO".into(),
            playback: None,
            focus: None,
            pkg: None,
            bench: None,
            livery: None,
            drive: 0,
            replay_feed: None,
            track_drive: None,
            line_driver: None,
            tv: None,
            car_id: None,
            pits: None,
            laps: None,
            race: None,
            ai_slots: Vec::new(),
            mp: None,
        }
    }
}

/// Every key the window understands, printed at startup and on F1. Keep in
/// step with main.rs's header and README. No key has two meanings: drive
/// mode's keys only exist in drive mode, the effects live on the F-keys and
/// the navigation block, free-look owns WASD Q E.
pub const KEYS: &str = "\
KEYS
  view     drag orbit, wheel zoom, F free-look (WASD move, Q/E down/up, Shift fast,
           Alt slow), G floor, L track lighting, C camera (replays), M mute
  car      H light switch (off / stage 1 / 2 / 3), B driver, V / Shift+V livery,
           [ ] , . P 0 Enter animations, 1-9 doors and lids (in drive mode)
  pits     (track drive) P pit limiter, Tab pit menu (arrows / Shift / Enter in it),
           Home back to the pits (instant, setup menu up); stop on the box marker for the pit stop
  cluster  (track drive) drag it with the mouse, ` show / hide, Shift+` back to its place
  laps     (track drive) Backspace leaderboard on / off (it also shows after each lap)
  race     (--ai N) R radar automatic / on / off; Numpad 4 / 6 watch the other cars
           (online: the other players), Numpad 5 back to yours
  drive    N mode (off / keyboard / demo / replay), Up/I throttle, Down/K brake,
           Left/J Right/L steer, Space handbrake, X / Z gear up / down,
           Y wipers, T DRS
  aids     (track drive) Numpad 7/1 TC up/down, 8/2 ABS up/down, 9/3 brake
           bias front/rear; T / A switch TC / ABS off. Wheel: the game's own
           buttons, or --wheel-setup tc_up tc_down abs_up abs_down bias_up bias_down
  effects  F1 this list, F2 night, F3 bloom, F4 dirt 0 / 0.5 / 1, F5 wet,
           F6 brake-disc temperature, F7 backfire, F8 tyre smoke, F9 dust,
           F10 indicators (off / left / right / hazard), F11 rain / fog light,
           F12 hot exhaust
  manual   (drive mode off) PageUp/PageDown rpm, Shift+PageUp/PageDown gear,
           Home/End speed, Insert brake pedal, Delete reverse
  replay   Space / P play, Left / Right 5 s (Shift 30 s), Up / Down or - = speed, , . scrub,
           R restart, C camera, [ ] car, Tab car list, ` hide the overlay; click / drag the timeline
  Esc quit
";

pub async fn run(car: Car, skeleton: Option<crate::load::Skeleton>, animations: Vec<crate::load::Animation>, mut opts: ViewOpts) {
    let event_loop = EventLoop::new().unwrap();
    let mut wb = WindowBuilder::new()
        .with_title(opts.title.clone())
        .with_inner_size(winit::dpi::LogicalSize::new(1400, 850));
    if opts.bench.is_some() {
        // the bench renders offscreen at its own size; the window is a stub
        wb = wb.with_inner_size(winit::dpi::PhysicalSize::new(480, 270));
    }
    let window = Arc::new(wb.build(&event_loop).unwrap());
    // EVOVIEW_WHEEL=1: a force-feedback wheel drives the car (wheel.rs, ffb.rs)
    let mut wheel = if opts.track_drive.is_some() && crate::envc::var_os("EVOVIEW_WHEEL").is_some() {
        use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
        let hwnd = match window.window_handle().map(|h| h.as_raw()) {
            Ok(RawWindowHandle::Win32(h)) => h.hwnd.get() as *mut std::ffi::c_void,
            _ => std::ptr::null_mut(),
        };
        match crate::wheel::Wheel::open(hwnd) {
            Ok(w) => {
                println!("wheel: {} (force feedback on); devices {:?}", w.name, w.devices());
                Some(w)
            }
            Err(e) => {
                println!("wheel: {e}");
                None
            }
        }
    } else {
        None
    };
    let mut ffb = crate::ffb::Ffb::default();
    let mut kb_prev = [false; 3];
    // the last cockpit-aid press, for the HUD pop-up (aids.rs); and the
    // EVOVIEW_AIDS_DEMO=s scripted presses (captures)
    let mut aid_pop: Option<crate::aids::Pop> = None;
    let mut aid_demo = (0.0f32, 0usize);
    let mut ffb_set = crate::ffb::Settings::from_env();
    if wheel.is_some() {
        println!("ffb: gain {:.2}, wheel {:.0} deg, min damper {:.2}, damper gain {:.2}, rate damping {:.4}, trim x{:.2}{} (from {})",
            ffb_set.gain, ffb_set.wheel_range, ffb_set.min_damper, ffb_set.damper_gain, ffb_set.rate_damping, ffb_set.boost,
            if ffb_set.legacy { ", LEGACY chain" } else { "" }, ffb_set.source);
        let e = ffb_set.enh;
        println!("ffb road effects: kerbs {:.2}, bumps {:.2}, slips {:.2}, ABS {:.2}", e.curbs, e.gforce, e.slips, e.abs);
    }
    let t_device = std::time::Instant::now();

    // Vulkan explicitly: pinning the backend keeps behaviour identical rather
    // than silently falling back to DX12 on some machines.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("no Vulkan adapter - is the GPU driver installed?");
    println!("GPU: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);

    // BC1/BC7 come straight from the game; without this feature they cannot
    // be uploaded at all and every texture would need CPU decompression.
    let dev_desc = wgpu::DeviceDescriptor {
        label: Some("device"),
        required_features: wgpu::Features::TEXTURE_COMPRESSION_BC
            // SSR's R32Float Hi-Z under auto layouts (ssr.rs)
            | (adapter.features() & wgpu::Features::FLOAT32_FILTERABLE)
            | if opts.bench.is_some() {
                crate::bench::features(&adapter)
            } else {
                wgpu::Features::empty()
            }
            // FSR (fsr.rs) runs its fp16 shaders wherever the GPU has them
            // and takes the feature as enabled
            | if crate::fsr::wanted() { adapter.features() & wgpu::Features::SHADER_F16 } else { wgpu::Features::empty() },
        required_limits: crate::render::limits(&adapter),
        memory_hints: Default::default(),
    };
    // EVOVIEW_UPSCALER=dlss on a track: the device carries NGX's extensions
    // (dlss.rs); None falls back to the plain device and TAA
    let dlss_dev = if opts.start_fly { crate::dlss::request_device(&adapter, &dev_desc) } else { None };
    let dlss_device = dlss_dev.is_some();
    let (device, queue) = match dlss_dev {
        Some(dq) => dq,
        None => adapter
            .request_device(&dev_desc, None)
            .await
            .expect("device (does this GPU support BC textures?)"),
    };
    crate::perf::add("device init", t_device.elapsed().as_secs_f64());

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(caps.formats[0]);
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        // Fifo (vsync) by default - a viewer has no reason to burn a GPU
        // rendering frames nobody sees. EVOVIEW_NOVSYNC=1 uncaps it, which is
        // how you measure real headroom rather than the monitor's refresh.
        present_mode: if opts.bench.is_some() {
            crate::bench::present_mode(&caps)
        } else if crate::envc::var("EVOVIEW_NOVSYNC").is_ok() {
            wgpu::PresentMode::AutoNoVsync
        } else {
            wgpu::PresentMode::Fifo
        },
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);
    // Bench: render to an offscreen target of the requested size, so 4K can
    // be measured on a smaller monitor and the compositor stays out of the
    // numbers. From here on `config` is the RENDER size; the surface keeps
    // the stub window's size and is never presented to.
    let bench_rt = opts.bench.as_ref().map(|b| {
        config.width = b.width;
        config.height = b.height;
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bench target"),
            size: wgpu::Extent3d { width: b.width, height: b.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    });

    let mut materials = crate::perf::time("materials+tex upload", || {
        Materials::build(&device, &queue, &car)
    });
    let (mut meshes, mut lo, mut hi, total_tris) = crate::perf::time("scene upload", || {
        if crate::grid::wanted() {
            crate::grid::build(&device, &car, &materials)
        } else {
            build_scene(&device, &car, Some(&materials))
        }
    });
    // A track often has one sky/prop mesh kilometres out. Framing on the
    // raw AABB starts the camera in space. Use the bulk of the vertices.
    if opts.start_fly && !crate::grid::wanted() {
        if let Some((a, b)) = robust_bounds(&car) {
            lo = a;
            hi = b;
        }
    }
    // The floor is scenery: keep its index so it can be dropped from the draw
    // without disturbing the car meshes.
    let mut ground_mesh = None;
    if opts.ground && !crate::grid::wanted() {
        let mut g = ground_quad(&device, lo, hi);
        g.draws = vec![(0, 6, materials.ground)];
        meshes.push(g);
        ground_mesh = Some(meshes.len() - 1);
    }
    let mut ground_on = opts.ground;
    if meshes.is_empty() {
        eprintln!("nothing renderable in that file");
        return;
    }
    println!("uploaded {} mesh(es), {} triangles; bounds {:?} .. {:?}",
             meshes.len(), total_tris, lo, hi);
    let mut cam = Orbit::framing(lo, hi);

    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("camera"),
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // One uniform buffer of bone matrices, rewritten every frame while an
    // animation is playing.
    let identity: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    // ...and one more block of MAX_BONES per replay car (shader.wgsl), each
    // car's own rig posed from its recording
    let bone_blocks = 1 + opts.playback.as_ref().map_or(0, |pb| pb.replay.cars.len());
    let bone_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bones"),
        contents: bytemuck::cast_slice(&vec![identity; MAX_BONES * bone_blocks]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });
    // Tracks used to run without MSAA: every foliage card, fence and kerb edge
    // crawled and the user read it as "track quality is far below cars".
    // 4x everywhere; EVOVIEW_TRACK_MSAA=1 trades it back for speed.
    let samples = if opts.start_fly {
        // EVOVIEW_TRACK_MSAA = 1 | 4 (default 4; WebGPU only guarantees 1 and 4)
        // driving: 1 (TAA anti-aliases; 4x cost 3 ms at 3440x1440 and the
        // frame no longer fit its vsync slot)
        match crate::envc::var("EVOVIEW_TRACK_MSAA").ok().as_deref() {
            // DLSS / FSR take the aliased, jittered frame (and a single-sample depth)
            _ if dlss_device || crate::fsr::wanted() => 1,
            Some("1") => 1,
            Some("4") => 4,
            _ if opts.track_drive.is_some() => 1,
            _ => 4,
        }
    } else {
        4
    };
    // Cars get the game's hero shadow: a map fitted to the car, redrawn only
    // when the pose changes (shadow_invalid), since the car is static.
    let hero_on = !opts.start_fly && crate::envc::var("EVOVIEW_NO_SHADOW").is_err()
        && crate::envc::var("EVOVIEW_NO_HERO").is_err();
    let shadows_on = (opts.start_fly && crate::envc::var("EVOVIEW_NO_SHADOW").is_err()) || hero_on;
    let mut last_pose: Vec<[f32; 16]> = Vec::new();
    // Driving: 24 cloud steps instead of 40 (TAA integrates the per-frame
    // dither; ~2 ms at 3440x1440). EVOVIEW_CLOUD_STEPS still wins.
    if opts.track_drive.is_some() && crate::envc::var_os("EVOVIEW_CLOUD_STEPS").is_none() {
        crate::envc::set_var("EVOVIEW_CLOUD_STEPS", "24");
    }
    // and 12 fog steps / 8 AO samples instead of 24 / 12 (taa.wgsl; the
    // history averages the per-frame jitter of both)
    if opts.track_drive.is_some() {
        for (k, v) in [("EVOVIEW_FOG_STEPS", "12"), ("EVOVIEW_AO_SAMPLES", "8")] {
            if crate::envc::var_os(k).is_none() {
                crate::envc::set_var(k, v);
            }
        }
    }
    let shadow_dim = if hero_on { hero_dim() } else { SHADOW_DIM };
    let (shadow_view, shadow_samp, shadow_layers, shadow_tex) = shadow_target_tex(&device, shadow_dim);
    // Driving: the near cascade's scenery is drawn once into this cache per
    // grid cell of its centre and copied in each frame; only the car is
    // redrawn on top (the whole cascade every frame cost ~8 ms at 3440x1440).
    // EVOVIEW_SHADOW_CACHE=0 draws it all every frame again.
    let cached = opts.track_drive.is_some() && shadows_on && !hero_on
        && crate::envc::var("EVOVIEW_SHADOW_CACHE").map_or(true, |v| v != "0");
    // With the car in its own hero map (layer 3) the near cascade holds the
    // scenery alone: drawn straight into its layer when the cell changes, no
    // cache texture and no 64 MB copy a frame.
    let dhero = cached && drive_hero() && opts.playback.is_none();
    let shadow_cache = (cached && !dhero).then(|| {
        let t = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow cache"),
            size: wgpu::Extent3d { width: shadow_dim, height: shadow_dim, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let v = t.create_view(&Default::default());
        (t, v)
    });
    let mut shadow_cache_vp: Option<glam::Mat4> = None;
    // far cascade: tracks only (a car is all inside its hero map)
    let far_on = shadows_on && !hero_on && opts.start_fly && cascades();
    let mid_on = far_on && mid_cascade();
    let light_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("light camera"),
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let clouds = crate::clouds::CloudGpu::new(&device, &queue, &car);
    // HDR on tracks (EVOVIEW_HDR=0 off): every pipeline drawing into the
    // main pass targets scene_format
    let hdr = opts.start_fly && crate::envc::var("EVOVIEW_HDR").map_or(true, |v| v != "0");
    HDR.store(hdr, std::sync::atomic::Ordering::Relaxed);
    let scene_format = if hdr { HDR_FORMAT } else { format };
    // screen-space reflections (ssr.rs): the HDR scene, single-sampled (driving,
    // or EVOVIEW_TRACK_MSAA=1), through TAA; every main-pass pipeline then
    // declares its two extra targets
    crate::ssr::set_active(hdr && samples == 1 && crate::ssr::wanted()
        && device.features().contains(wgpu::Features::FLOAT32_FILTERABLE));
    // what the rain drops on the glass refract (rain.rs), single-sample only
    let no_refract = crate::envc::var_os("EVOVIEW_NO_REFRACT").is_some();
    // ⚠ sized for the monitor, not the window: it is bound into the main
    // bind groups, so it cannot follow a resize, and a window grown past it
    // (windowed -> fullscreen) refracted a stale copy on its right and
    // bottom. A frame larger than the copy goes without refraction.
    let scene_copy = (opts.track_drive.is_some() && samples == 1).then(|| {
        let mon = window.current_monitor().map(|m| m.size()).map_or((0, 0), |s| (s.width, s.height));
        crate::rain::SceneCopy::new(&device, scene_format, config.width.max(mon.0), config.height.max(mon.1))
    });
    let copy_fits = |c: &crate::rain::SceneCopy, w: u32, h: u32| w <= c.size.0 && h <= c.size.1;
    // the side windows' streak maps (rain.rs), a 1-texel stand-in without
    let streak_view = crate::rain::streak_view(&device, &queue, car.rain_streaks.as_deref().map(|v| v.as_slice()));
    // the windshield rain maps on the GPU (rain.rs)
    let rain_views: Vec<Option<std::rc::Rc<wgpu::TextureView>>> = car.rain_tex.iter()
        .map(|t| t.and_then(|i| materials.textures.get(&device, &queue, &car, i, false)))
        .collect();
    // The driven car's reflection probe (probe.rs): HDR only, since the
    // track pipelines write tonemapped light otherwise. EVOVIEW_PROBE=0 off.
    let mut probe = (opts.track_drive.is_some() && hdr
        && crate::envc::var("EVOVIEW_PROBE").map_or(true, |v| v != "0"))
        .then(|| crate::probe::Probe::new(&device, &queue, scene_format, samples));
    let pipes = crate::perf::time("pipeline creation", || {
        build_pipeline(
            &device, scene_format, &uniform_buf, samples, &materials.layout, &bone_buf,
            &materials.env_view, &materials.env_samp, &materials.brdf_view, &materials.lut_view,
            &shadow_view, &shadow_samp, &light_buf, &clouds, probe.as_ref().map(|p| &p.cube),
            [rain_views[0].as_deref(), rain_views[1].as_deref(), rain_views[2].as_deref()],
            scene_copy.as_ref().map(|c| &c.view), &streak_view)
    });
    let (pipeline, blended, bind_group, bone_bg) =
        (&pipes.pipeline, &pipes.blended, &pipes.bind_group, &pipes.bone_bg);
    // Effects (fxstate.rs / fxgpu.rs): lights, LEDs, dash, particles, bloom.
    let mut fxst = crate::fxstate::FxState::from_env();
    let wet_pinned = crate::envc::var_os("EVOVIEW_WET").is_some();
    fxst.blink_phase = crate::envc::var("EVOVIEW_BLINK").ok().map(|v| v.trim() == "1");
    let mut fxg = crate::fxgpu::FxGpu::new(&device, &queue, &car, &materials, scene_format, samples, &uniform_buf);
    // the car itself is dropped below (RAM); the effect data stays
    let car_fx: Option<Box<crate::fxdata::CarFx>> = car.fx.clone();
    let pipes_all = &pipes;
    let mut bench = opts.bench.take().map(|b| {
        let mut bench = crate::bench::Bench::new(b, &device, &queue);
        bench.info = vec![
            ("meshes", meshes.len() as f64),
            ("draws", meshes.iter().map(|m| m.draws.len()).sum::<usize>() as f64),
            ("tris", total_tris as f64),
            ("instanced_tris", meshes.iter()
                .map(|m| m.index_count as f64 / 3.0 * m.instance_count as f64).sum::<f64>()),
            ("instances", meshes.iter().map(|m| m.instance_count as f64).sum::<f64>()),
            ("textures", car.textures.len() as f64),
            ("materials", car.materials.len() as f64),
        ];
        // the heaviest meshes, so a slow scene says WHERE the triangles are
        let mut top: Vec<(f64, String)> = meshes
            .iter()
            .map(|m| {
                let name = car.meshes.get(m.src).map_or("(floor)", |x| x.name.as_str());
                (m.index_count as f64 / 3.0 * m.instance_count as f64,
                 format!("{name} x{} ({} tris each)", m.instance_count, m.index_count / 3))
            })
            .collect();
        top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        top.truncate(8);
        bench.top = top;
        bench
    });
    // Track geometry gets its own pipeline; None for a car on its own.
    let track = crate::track::TrackRender::new(
        &device, &queue, &car, scene_format, samples, &shadow_view, &shadow_samp, &materials.lut_view, &materials,
        &clouds);

    // Manual animation control. Each take has its own open/closed target and
    // its own playhead, so a door, the other door and the hood can all be open
    // together instead of one at a time.
    #[derive(Clone, Copy)]
    struct AnimState {
        open: bool,
        t: f32,
        len: f32,
    }
    let mut anim_state: Vec<AnimState> = animations
        .iter()
        .map(|a| AnimState { open: false, t: 0.0, len: a.frames as f32 / a.fps.max(1.0) })
        .collect();
    let mut selected = 0usize;
    if !animations.is_empty() {
        println!("animations ({}):", animations.len());
        for (i, a) in animations.iter().enumerate() {
            println!("  {}  {}", i + 1, a.name);
        }
        println!("  1-9 toggle open/closed  ·  0 close all  ·  [ ] select  ·  \
, . scrub the selected one  ·  Enter toggles selected");
    }
    // Drive mode (drive.rs + carrig.rs): N cycles off / keyboard / demo.
    let mut drive: Option<crate::drive::Drive> = skeleton
        .as_ref()
        .and_then(|s| s.rig.as_deref())
        .filter(|_| opts.playback.is_none())
        .map(crate::drive::Drive::new);
    // wipers run while it rains (Y still toggles them)
    if let Some(d) = drive.as_mut() {
        d.wipers_on = fxst.rain > 0.0;
    }
    // 4 = following the track drive (trackdrive.rs through Drive::follow)
    let mut drive_mode: u8 = if drive.is_some() && opts.track_drive.is_some() {
        4
    } else if drive.is_some() {
        opts.drive
    } else {
        0
    };
    let mut was_night = false;
    if drive_mode == 1 {
        if let Some(d) = drive.as_mut() { d.cs.gear = 1; }
    }
    let mut demo_t = 0.0f32;
    let mut shadow_tick = 0u32;
    let mut feed = opts.replay_feed.take();
    // replay cars with a rig: each one's Drive + recording state (replayscene
    // Slot::skel), made on first sight
    let mut rep_rigs: std::collections::HashMap<usize, (crate::drive::Drive, crate::drive::ReplayRig)> = Default::default();
    // their bone blocks (1 + car), gathered on the CPU and uploaded in one
    // write a frame (a write per car was ~50 queue calls)
    let mut rep_bones: Vec<[f32; 16]> = Vec::new();
    // replay rigs to pose once the camera is final (car, slot, its centre),
    // and which cars were posed last frame
    let mut pose_jobs: Vec<(usize, usize, Vec3)> = Vec::new();
    let mut posed_last: Vec<bool> = Vec::new();
    let mut slot_reach: Vec<f32> = Vec::new();
    let n_modes: u8 = if feed.is_some() { 4 } else { 3 };
    if drive.is_some() {
        let _ = ();
    }
    if opts.bench.is_none() {
        print!("{KEYS}");
    }
    // EVOVIEW_RENDER_SCALE (taa.rs): tracks draw the scene smaller than the
    // screen and TAA upsamples it
    let render_scale = if opts.start_fly { crate::taa::render_scale() } else { 1.0 };
    let (mut rw, mut rh) = crate::taa::scaled(config.width, config.height, render_scale);
    let mut targets = make_targets(&device, rw, rh, config.format, samples);
    // TAA on tracks (taa.rs): the main pass resolves into taa.cur and the
    // resolve pass writes the screen. Backslash toggles it.
    // falling rain after TAA (rainfall.rs), and the camera's last position
    // for its velocity
    let mut rainfall = opts.start_fly.then(|| crate::rainfall::Rainfall::new(&device, scene_format));
    let mut rain_eye: Option<Vec3> = None;
    let mut taa = opts.start_fly.then(|| crate::taa::Taa::new(&device, scene_format, config.width,
                                                               config.height, rw, rh, samples));
    if render_scale < 1.0 {
        if let Some(t) = taa.as_mut() {
            t.on = true;
        }
    }
    let mut ssr = crate::ssr::active().then(|| crate::ssr::Ssr::new(&device, scene_format, rw, rh));
    // DLSS (dlss.rs) in TAA's place, when its device opened and NGX starts
    let mut dlss = match (dlss_device, taa.as_mut()) {
        (true, Some(t)) => match crate::dlss::Dlss::new(&device, config.width, config.height, scene_format) {
            Ok(d) => {
                t.on = true;
                println!("DLSS on: render {rw}x{rh} -> {}x{}", config.width, config.height);
                Some(d)
            }
            Err(e) => {
                eprintln!("DLSS: {e} - TAA");
                None
            }
        },
        _ => None,
    };
    // AMD FSR (fsr.rs) in TAA's place: any Vulkan GPU, when its dlls load
    let mut fsr = match (crate::fsr::wanted() && dlss.is_none(), taa.as_mut()) {
        (true, Some(t)) => match crate::fsr::Fsr::new(&device, config.width, config.height, samples) {
            Ok(f) => {
                t.on = true;
                println!("FSR on: render {rw}x{rh} -> {}x{}", config.width, config.height);
                Some(f)
            }
            Err(e) => {
                eprintln!("FSR: {e} - TAA");
                None
            }
        },
        _ => None,
    };
    // per-pixel motion vectors for TAA (motion.rs); EVOVIEW_MOTION=0 off
    // (DLSS / FSR need them: always on with either, plus their RG16 copy)
    let ext_up = dlss.is_some() || fsr.is_some();
    let mut motion = taa.as_ref().filter(|_| crate::motion::wanted() || ext_up)
        .map(|_| crate::motion::Motion::new(&device, rw, rh, samples, ext_up, bone_buf.size()));
    // the driven car's wheel bones and their centres (fast spin, motion.wgsl)
    if let (Some(mo), Some(s)) = (motion.as_ref(), skeleton.as_ref()) {
        let spin: Vec<(usize, glam::Vec3)> = s.rig.as_deref().map_or(Vec::new(), |r| {
            r.corners.iter().flat_map(|c| c.spin.iter().map(|&(b, _)| b))
                .filter_map(|b| s.bind.get(b).map(|m| (b, glam::Vec3::new(m[12], m[13], m[14]))))
                .collect()
        });
        mo.set_spin(&queue, &spin);
    }
    // the game's post effects after TAA (post.rs): veil, vignette, sharpen,
    // motion blur, auto exposure. EVOVIEW_POST=0 off.
    let mut post = taa.as_ref().filter(|_| crate::envc::var("EVOVIEW_POST").map_or(true, |v| v != "0"))
        .map(|_| crate::post::Post::new(&device, scene_format, config.format, config.width, config.height, samples,
                                         &materials.lut_view));
    if hdr {
        // the HDR target has to go through TAA and the post tonemap
        if let Some(t) = taa.as_mut() {
            t.on = true;
        }
        if post.is_none() {
            eprintln!("HDR needs the post chain: EVOVIEW_POST=0 ignored");
            post = Some(crate::post::Post::new(&device, scene_format, config.format, config.width, config.height, samples,
                                               &materials.lut_view));
        }
    }
    // Driving, the camera rides the car: camera motion blur smears all the
    // scenery past the windscreen (the "ghosting" at speed). Off unless
    // EVOVIEW_MOTION_BLUR asks for it; / still toggles it.
    if opts.track_drive.is_some() && crate::envc::var_os("EVOVIEW_MOTION_BLUR").is_none() {
        if let Some(p) = post.as_mut() {
            p.motion_blur = 0.0;
        }
    }
    let mut hud = crate::hud::Hud::new(&device, config.format);
    // the driving cluster (cluster.rs) and the pedals it shows
    let mut cluster = crate::cluster::Cluster::load();
    // the opponent the camera follows instead of our car (online: another
    // player): EVOVIEW_VIEW_AI=k at start, Numpad 4 / 6 cycle, Numpad 5 ours
    let mut view_ai: Option<usize> = crate::envc::var("EVOVIEW_VIEW_AI").ok().and_then(|v| v.parse().ok());
    // online: late joiners' cars, loaded one at a time on a worker thread
    // (replayscene::load_net_car) and appended to the scene when done
    type NetCar = (u16, String, Result<(crate::asset::Car, crate::replayscene::Slot), String>);
    let (net_job_tx, net_job_rx) = std::sync::mpsc::channel::<(u16, String, Option<Vec<u8>>)>();
    let (net_done_tx, net_done_rx) = std::sync::mpsc::channel::<NetCar>();
    if let (Some(_), Some(pkg)) = (opts.mp.as_ref(), opts.pkg.clone()) {
        std::thread::Builder::new().name("mp cars".into()).spawn(move || {
            for (n, car, fs) in net_job_rx {
                let t = std::time::Instant::now();
                let r = crate::replayscene::load_net_car(&pkg, None, &car, fs.as_deref(), &format!("n{n}"));
                println!("mp: car #{n} ({car}) loaded in {:.1}s", t.elapsed().as_secs_f32());
                if net_done_tx.send((n, car, r)).is_err() {
                    break;
                }
            }
        }).ok();
    }
    // online chat being typed (Shift+Enter or Numpad / opens it)
    let mut chat_input: Option<String> = None;
    let mut pedals = (0.0f32, 0.0f32, None::<f32>, 0.0f32);
    // Engine audio, only for replays - there is nothing to sound like when
    // looking at a parked car.
    let mut engine = match (opts.playback.is_some(), opts.pkg.as_ref()) {
        (true, Some(pkg)) => match crate::audio::Engine::new(&crate::audio::game_dir(pkg)) {
            Ok(mut e) => {
                // Sound is on by default now that it is the game's own mix.
                // headless runs (--bench, --drive-test) are silent
                e.muted = std::env::args().any(|a| a == "--no-sound" || a == "--bench" || a == "--drive-test");
                println!("engine sound {} (M toggles)",
                         if e.muted { "muted" } else { "on, via the game's FMOD" });
                Some(e)
            }
            Err(err) => {
                eprintln!("no engine sound: {err}");
                None
            }
        },
        _ => None,
    };

    // The driven car's sound (carsound.rs): the game's FMOD with every bank
    // the car's events use, fed by the drive model. --no-sound / M mute.
    let mut car_snd = match (opts.track_drive.as_ref(), opts.pkg.as_ref()) {
        (Some(td), Some(pkg)) if crate::envc::var_os("EVOVIEW_NO_SOUND").is_none() => {
            let evs = car.fx.as_ref().map_or(Vec::new(), |f| f.actor.audio_events.clone());
            match crate::carsound::CarSound::new(pkg, &td.slot.model, &evs, None) {
                Ok(mut s) => {
                    s.muted = std::env::args().any(|a| a == "--no-sound" || a == "--bench" || a == "--drive-test");
                    // a quarter of full scale by default (EVOVIEW_VOLUME 0..1)
                    s.volume = crate::envc::var("EVOVIEW_VOLUME").ok().and_then(|v| v.parse().ok()).unwrap_or(0.25);
                    Some(s)
                }
                Err(e) => {
                    eprintln!("no car sound: {e}");
                    None
                }
            }
        }
        _ => None,
    };

    let mut dragging = false;
    let mut last_pos: Option<(f64, f64)> = None;
    let mut frames = 0u32;
    let mut fps_t = std::time::Instant::now();

    // Free-look mode. The orbit camera stays the default because it frames a
    // car well on load; F swaps to a fly camera for getting inside and under
    // the model, where orbiting a fixed centre stops being useful.
    let mut fly: Option<Fly> = if opts.start_fly {
        let mut f = Fly::from_orbit(&cam);
        // A circuit is kilometres across. Car-scaled fly speed crawls;
        // this is still slow enough to inspect a barrier up close.
        let span = (hi - lo).length();
        f.speed = (span * 0.08).clamp(25.0, 180.0);
        // ⚠ Start ON the track, not where the orbit camera frames it from.
        // Framing a 10x22 km road network that is only 38 m thick puts the
        // eye ~35 km away, where the whole track is a few pixels against a
        // black background - it reads as "the viewer loaded nothing" even
        // with 13 million triangles resident. Drop in just above the surface
        // at the middle of the bulk geometry and look down slightly.
        let mid = (lo + hi) * 0.5;
        let height = ((hi.y - lo.y) * 0.6).clamp(15.0, 120.0);
        f.pos = glam::Vec3::new(mid.x, hi.y + height, mid.z);
        f.pitch = -0.45;
        // A replay opens on the grid instead: 25 m back and 12 m up, which
        // frames a field of cars without being so high they turn into specks.
        if let Some(p) = opts.focus {
            f.pos = p + glam::Vec3::new(0.0, 12.0, 25.0);
            f.pitch = -0.25;
            f.speed = 40.0;
        }
        Some(f)
    } else {
        None
    };
    // Bench on a track: the load-time view looks down into whatever sits at
    // the middle of the map (often a tree). Look out at the horizon instead,
    // so a turn on the spot sweeps across most of the circuit - the
    // demanding view, and the one a frustum cull helps least.
    if let (Some(_), Some(f)) = (bench.as_ref(), fly.as_mut()) {
        f.pitch = -0.12;
    }
    // EVOVIEW_FLY="x,y,z,tx,ty,tz": start the free camera there, looking at
    // the target (debugging live-only effects: TAA, animation)
    if let (Some(f), Ok(s)) = (fly.as_mut(), crate::envc::var("EVOVIEW_FLY")) {
        let mut v: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        // EVOVIEW_TREE_CHECK=1: every foliage instance against the collision
        // ground under it - floating, buried, or standing on a road surface
        if crate::envc::var_os("EVOVIEW_TREE_CHECK").is_some() {
            if let Some(e) = car.track_env.as_deref() {
                if let Some(c) = e.collision.as_ref() {
                    let g = crate::trackdrive::Ground::new(c, &e.surface_defs);
                    let fol = |n: &str| ["tree", "fir", "spruce", "pine", "beech", "birch", "oak", "bush", "aspen", "conifer", "poplar", "cf", "bb_"]
                        .iter().any(|k| n.to_lowercase().contains(k));
                    let all = crate::envc::var("EVOVIEW_TREE_CHECK").map_or(false, |v| v == "all");
                    let (mut tn, mut tf, mut tb) = (0, 0, 0);
                    for m in car.meshes.iter().filter(|m| !m.xforms.is_empty() && (all || fol(&m.name)) && !m.name.contains("#lod")) {
                        let (mut n, mut hit, mut fl, mut bu, mut road) = (0, 0, 0, 0, 0);
                        let mut up_neg = 0;
                        let mut worst = (0.0f32, glam::Vec3::ZERO);
                        for x in &m.xforms {
                            let mx = glam::Mat4::from_cols_array(x);
                            let p = mx.w_axis.truncate();
                            n += 1;
                            if mx.y_axis.y < 0.0 { up_neg += 1; }
                            let Some(h) = g.probe(p.x, p.z, p.y + 50.0) else { continue };
                            hit += 1;
                            let d = p.y - h.y;
                            if d > 1.0 { fl += 1; }
                            if d < -1.0 { bu += 1; }
                            if e.surface_defs.get(h.surf as usize).map_or(false, |s| s.grip > 0.9) { road += 1; }
                            if d.abs() > worst.0.abs() { worst = (d, p); }
                            if d > 3.0 && d < 30.0 && fl % 400 == 1 { println!("TREEFLOAT {} {:+.1} at {:.1},{:.1},{:.1} ground {:.1} up {:.2?} x {:.2?}", m.name, d, p.x, p.y, p.z, h.y, mx.y_axis.truncate(), mx.x_axis.truncate()); }
                        }
                        tn += hit;
                        tf += fl;
                        tb += bu;
                        println!("TREECHECK {:<40} n {n:6} probed {hit:6} floating {fl:5} buried {bu:5} on-road {road:5} upside-down {up_neg:5} worst {:+.1} m at ({:.0} {:.0} {:.0})",
                                 m.name, worst.0, worst.1.x, worst.1.y, worst.1.z);
                    }
                    println!("TREECHECK TOTAL probed {tn} floating {tf} buried {tb}");
                }
            }
        }
        // EVOVIEW_FLY_GROUND=1: y and ty are heights above the ground there
        if v.len() == 6 && crate::envc::var_os("EVOVIEW_FLY_GROUND").is_some() {
            if let Some((c, defs)) = car.track_env.as_deref().and_then(|e| e.collision.as_ref().map(|c| (c, &e.surface_defs))) {
                let g = crate::trackdrive::Ground::new(c, defs);
                for (x, y, z) in [(0, 1, 2), (3, 4, 5)] {
                    if let Some(h) = g.probe(v[x], v[z], 1.0e4) {
                        v[y] += h.y;
                    }
                }
                println!("fly ground: eye y {:.2}, target y {:.2}", v[1], v[4]);
            }
        }
        if v.len() == 6 {
            let d = (glam::Vec3::new(v[3], v[4], v[5]) - glam::Vec3::new(v[0], v[1], v[2])).normalize_or_zero();
            f.pos = glam::Vec3::new(v[0], v[1], v[2]);
            f.yaw = d.z.atan2(d.x);
            f.pitch = d.y.clamp(-1.0, 1.0).asin();
        }
    }
    // the ground under replay cars' wheels, for their grass / dirt effects:
    // per collision surface (Collision.surfaces, matched to the track's
    // SurfaceDefinitionData by id) its (grass, dirt) and "id/kind" for
    // EVOVIEW_FX_LOG. SurfaceType 3 grass, 1 extra turf, 4 sand; gravel /
    // dirt by name (also when the id has no definition).
    let fx_ground: Option<(crate::trackdrive::Ground, Vec<((f32, f32), String)>)> = car.track_env.as_deref()
        .and_then(|e| e.collision.as_ref().map(|c| {
            let kinds = c.surfaces.iter().map(|s| {
                // ⚠ untagged is NOT grass here: the Karussell's concrete
                // banking is untagged, and kicked up grass on every lap
                if s.is_empty() {
                    return ((0.0, 0.0), "untagged".to_string());
                }
                let d = e.surface_defs.iter().find(|d| d.id.eq_ignore_ascii_case(s));
                let kind = d.map_or(u32::MAX, |d| d.kind);
                let id = s.to_lowercase();
                let gd = if kind == 4 || id.contains("gravel") || id.contains("dirt") || id.contains("sand") {
                    (0.0, 1.0)
                } else if kind == 3 || id.contains("grass") {
                    (1.0, 0.0)
                } else if kind == 1 {
                    (0.6, 0.2)
                } else {
                    (0.0, 0.0)
                };
                (gd, format!("{s}/{}", if kind == u32::MAX { "?".to_string() } else { kind.to_string() }))
            }).collect();
            (crate::trackdrive::Ground::new(c, &e.surface_defs), kinds)
        }));
    let surface_named = move |x: f32, z: f32, top: f32| -> ((f32, f32), String) {
        let Some((g, kinds)) = fx_ground.as_ref() else { return ((0.0, 0.0), "-".into()) };
        let Some(h) = g.probe(x, z, top) else { return ((0.0, 0.0), "none".into()) };
        kinds.get(h.surf as usize).cloned().unwrap_or(((0.0, 0.0), "?".into()))
    };
    let bench_yaw0 = (cam.yaw, fly.as_ref().map_or(0.0, |f| f.yaw));
    let mut held: HashSet<KeyCode> = HashSet::new();
    let mut looking = false;
    let mut last_frame = std::time::Instant::now();
    // Frame cap: EVOVIEW_MAX_FPS, default 60, 0 = uncapped. The viewer shares
    // the GPU with whatever else the user is doing; nothing it shows needs
    // more than the monitor refresh. Applies to --bench too: benchmarks set
    // EVOVIEW_MAX_FPS=0 explicitly (tools/bench.py does).
    // Driving paces to a whole number of refreshes (164 Hz -> every 3rd =
    // 54.7 fps) and steps the sim by exactly that: every frame on screen for
    // the same time, every move the same size. Uneven 2/3-refresh frames
    // and wall-clock dt read as a double image at speed.
    // EVOVIEW_DRIVE_FPS = the most it may run at (default 60).
    // It adapts: a frame that keeps missing its slot moves to the next
    // whole number of refreshes, one with room to spare moves back.
    // EVOVIEW_DRIVE_FPS = the most it may run at (default the display's).
    // Every mode paces this way (fly, replays and the studio too): a sleep to
    // a flat 60 fps on a 164 Hz screen showed frames for 2 or 3 refreshes in
    // turn, the judder the driving path was built to remove.
    // It starts at the fewest refreshes that keep it at or under ~90 fps
    // (164 Hz: every 2nd = 82 fps; 60 Hz: every one).
    let mut drive_pace: Option<Pace> = {
        window.current_monitor().and_then(|m| m.refresh_rate_millihertz()).map(|m| m as f32 / 1000.0).map(|hz| {
            let most = crate::envc::var("EVOVIEW_DRIVE_FPS").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(hz);
            let k_min = (hz / most.max(1.0)).ceil().max(1.0);
            let k0 = (hz / 90.0).ceil().max(1.0);
            Pace { hz, k: k_min.max(k0), k_min, miss: 0, fast: 0, n: 0, slept: 0.0 }
        })
    }.filter(|_| crate::envc::var("EVOVIEW_MAX_FPS").is_err() && crate::envc::var("EVOVIEW_NOVSYNC").is_err());
    if let Some(p) = &drive_pace {
        println!("pacing: {:.0} Hz display, starting at every {:.0} refreshes = {:.1} fps (adapts)", p.hz, p.k, p.hz / p.k);
    }
    let frame_budget = match crate::envc::var("EVOVIEW_MAX_FPS").ok().and_then(|v| v.parse::<f32>().ok()) {
        Some(f) if f <= 0.0 => None,
        Some(f) => Some(std::time::Duration::from_secs_f32(1.0 / f)),
        // Driving: vsync paces it. A 60 cap on a 164 Hz screen shows frames
        // for 2 or 3 refreshes in turn, and at speed that judder reads as a
        // double image of everything (the "ghosting")
        None if drive_pace.is_some() => None,
        None => Some(std::time::Duration::from_secs_f32(1.0 / 60.0)),
    };
    let mut next_frame = std::time::Instant::now();
    // frames counted for the rear-view screen's outside-the-car cadence
    let mut mirror_tick: u32 = 0;
    let mut fps_frames: u32 = 0;
    let mut fps_clock = std::time::Instant::now();
    let mut fps_now: f32 = 0.0;
    // Replay playhead, in seconds of captured motion.
    let mut play_t = opts.playback.as_ref().map_or(0.0f32, |pb| pb.start.max(0.0));
    let mut playing = true;
    let mut play_speed = 1.0f32;
    let mut follow = if opts.track_drive.is_some() && crate::envc::var_os("EVOVIEW_FLY").is_none() {
        crate::camera::Follow::Chase
    } else {
        crate::camera::Follow::Free
    };
    // The driven car's own cameras (actor drivable_cameras, then cameras),
    // the track's TV cameras, the viewer's chase, free: C cycles them.
    let mut car_cams: Vec<crate::fxdata::CarCam> = car.fx.as_ref().map_or(Vec::new(), |f| f.actor.cameras.clone());
    car_cams.sort_by_key(|c| !c.drivable);
    if let (Some(id), true) = (opts.car_id.as_deref(), opts.track_drive.is_some()) {
        crate::headcam::apply_user(&mut car_cams, id);
    }
    // the game's in-car camera motion (headcam.rs)
    let mut headcam = opts.track_drive.as_ref().map(|td| {
        let mut h = crate::headcam::HeadCam::new(opts.pkg.as_deref());
        if td.rig.is_some() {
            h.sprung_body();
        }
        h
    });
    let mut vlog = opts.track_drive.as_ref().and_then(|_| crate::headcam::ViewLog::from_env());
    let mut tv = opts.tv.take();
    let mut cam_sel: usize = car_cams.iter().position(|c| c.name.eq_ignore_ascii_case("NearChase"))
        .unwrap_or(car_cams.len() + 1);
    if opts.track_drive.is_none() {
        cam_sel = car_cams.len() + 1;
    }
    // EVOVIEW_CAM=<car camera name>|tv|chase|free: the starting camera
    if let (Ok(want), true) = (crate::envc::var("EVOVIEW_CAM"), opts.track_drive.is_some()) {
        let w = want.to_lowercase();
        cam_sel = match w.as_str() {
            "tv" => car_cams.len(),
            "chase" => car_cams.len() + 1,
            "free" => car_cams.len() + 2,
            _ => car_cams.iter().position(|c| c.name.to_lowercase() == w).unwrap_or(cam_sel),
        };
    }
    let n_car_cams = car_cams.len();
    let mut cam_applied: Option<usize> = None;
    let cam_names: Vec<String> = car_cams.iter().map(|c| c.name.clone()).collect();
    let cam_name = move |i: usize| -> String {
        if i < n_car_cams { cam_names[i].clone() }
        else if i == n_car_cams { "TV cameras".into() }
        else if i == n_car_cams + 1 { "viewer chase".into() }
        else { "free".into() }
    };
    if opts.track_drive.is_some() && !car_cams.is_empty() {
        println!("  C cameras: {} car view(s) ({}), TV, viewer chase, free  ·  Shift+C back",
                 car_cams.len(), car_cams.iter().map(|c| format!("{}{}:{}", if c.drivable { "" } else { "onboard " }, c.name, c.mode))
                     .collect::<Vec<_>>().join(", "));
    }
    // EVOVIEW_FOLLOW=chase|onboard|side|orbit: the starting follow camera
    if let Ok(v) = crate::envc::var("EVOVIEW_FOLLOW") {
        use crate::camera::Follow::*;
        for f in [Free, Chase, Onboard, Side, Orbit] {
            if f.name() == v {
                follow = f;
            }
        }
    }
    let mut watched = 0usize;          // index into `drivers`
    let mut title_at = 0.0f32;         // seconds since the title last changed
    // Every real driver in one flat list, so [ and ] step between cars without
    // landing on an empty session slot.
    let drivers: Vec<(usize, String)> = opts
        .playback
        .as_ref()
        .map(|pb| {
            let mut v: Vec<(usize, String)> = pb
                .slots
                .iter()
                .flat_map(|s| s.cars.iter().map(|&c| (c, s.model.clone())))
                .collect();
            v.sort_by_key(|d| d.0);
            v
        })
        .unwrap_or_default();
    // the replay overlay (replayhud.rs) and its copy of the driving cluster
    let mut rui = opts.playback.as_ref().map(|pb| crate::replayhud::ReplayUi::new(&pb.replay, &drivers));
    let mut rcluster = crate::cluster::Cluster::fixed();
    // Slots address meshes by SOURCE index; build_scene drops some meshes, so
    // translate once here rather than searching every frame.
    #[allow(clippy::type_complexity)]
    let mut slot_gpu: Vec<(Vec<usize>, [Vec<usize>; 4])> = Vec::new();
    // (AI car cabins are drawn only near the camera: EVOVIEW_AI_DETAIL_M)
    // the moving cars' instances, culled to what the camera can see (carcull.rs)
    let mut carcull = crate::carcull::CarCull::default();
    let car_detail_m = crate::envc::var("EVOVIEW_AI_DETAIL_M").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(30.0);
    if let Some(td) = opts.track_drive.as_ref() {
        let mut src_to_gpu = vec![usize::MAX; car.meshes.len()];
        for (gi, m) in meshes.iter().enumerate() {
            if m.src != usize::MAX {
                src_to_gpu[m.src] = gi;
            }
        }
        let map = |v: &Vec<usize>| -> Vec<usize> {
            v.iter().filter_map(|&si| src_to_gpu.get(si).copied()).filter(|&gi| gi != usize::MAX).collect()
        };
        let s = &td.slot;
        slot_gpu = vec![(map(&s.meshes), [map(&s.wheels[0]), map(&s.wheels[1]), map(&s.wheels[2]), map(&s.wheels[3])])];
        // the opponents' slots after it (race.rs: an AI's `slot` indexes here)
        for s in &opts.ai_slots {
            slot_gpu.push((map(&s.meshes), [map(&s.wheels[0]), map(&s.wheels[1]), map(&s.wheels[2]), map(&s.wheels[3])]));
            for &(b, d, l) in &s.lods {
                if let (Some(&gb), Some(&gl)) = (src_to_gpu.get(b), src_to_gpu.get(l)) {
                    if gb != usize::MAX && gl != usize::MAX {
                        carcull.add_lod(&mut meshes, gb, d, gl);
                    }
                }
            }
            // the cockpit, the driver and the small parts: only drawn on an
            // AI car near the camera (an M4 GT3's interior alone is more
            // triangles than its body)
            for &gi in &slot_gpu.last().unwrap().0 {
                let n = car.meshes.get(meshes[gi].src).map_or("", |m| m.name.as_str()).to_lowercase();
                if ["interior", "steer", "_net", "cable", "belt"].iter().any(|k| n.contains(k)) || n.starts_with("driver:") || n.starts_with("part_") {
                    carcull.set_detail(gi, crate::carcull::Detail::Near);
                }
            }
        }
        println!("drive: {} on the track", s.model);
        println!("  arrows drive (Up throttle, Down brake / reverse, Left / Right steer), Ctrl handbrake");
        println!("  Enter: drop the car where the camera is  ·  C camera (chase/onboard/side/orbit/free)");
    }
    // rain on the glass: the wipers the rig animates (rain.rs)
    let mut rain_fx = skeleton.as_ref().and_then(|s| s.rig.as_deref().map(|r| crate::rain::Rain::new(&car, &r.wiper_bones(), &s.bind, &s.bones.iter().map(|b| b.parent).collect::<Vec<_>>(), &s.bones.iter().map(|b| b.name.clone()).collect::<Vec<_>>())));
    // The driven car's mirrors (mirrors.rs): each a camera at its glass,
    // aimed by reflecting the driver's eye (the cockpit camera) off it.
    let mut taa_hist_off = crate::envc::var_os("EVOVIEW_TAA_HIST_OFF").is_some();
    // last frame's eye and view direction (camera-cut detection)
    let mut cut_prev: Option<(Vec3, Vec3)> = None;
    let mut mirrors = match (opts.track_drive.as_ref(), car.fx.as_ref(), slot_gpu.first()) {
        (Some(_), Some(fx), Some((body, _))) if !fx.actor.mirrors.is_empty()
                && crate::envc::var("EVOVIEW_MIRRORS").map_or(true, |v| v != "0") => {
            let eye = car_cams.iter().find(|c| c.mode == 1).or_else(|| car_cams.iter().find(|c| c.mode == 4))
                .map_or(Vec3::new(0.35, 1.1, 0.0), |c| Vec3::from(c.pos));
            crate::mirrors::Mirrors::new(&device, &car, &meshes, body, &fx.actor.mirrors, eye, &bone_buf, scene_format, samples)
        }
        _ => None,
    };
    if let Some(pb) = opts.playback.as_ref() {
        let mut src_to_gpu = vec![usize::MAX; car.meshes.len()];
        for (gi, m) in meshes.iter().enumerate() {
            if m.src != usize::MAX {
                src_to_gpu[m.src] = gi;
            }
        }
        let map = |v: &Vec<usize>| -> Vec<usize> {
            v.iter()
                .filter_map(|&si| src_to_gpu.get(si).copied())
                .filter(|&gi| gi != usize::MAX)
                .collect()
        };
        slot_gpu = pb
            .slots
            .iter()
            .map(|s| {
                let w: [Vec<usize>; 4] =
                    [map(&s.wheels[0]), map(&s.wheels[1]), map(&s.wheels[2]), map(&s.wheels[3])];
                (map(&s.meshes), w)
            })
            .collect();
        for s in &pb.slots {
            for &(b, d, l) in &s.lods {
                if let (Some(&gb), Some(&gl)) = (src_to_gpu.get(b), src_to_gpu.get(l)) {
                    if gb != usize::MAX && gl != usize::MAX {
                        carcull.add_lod(&mut meshes, gb, d, gl);
                    }
                }
            }
        }
        // the cabin, the driver and the small parts of a car too far to show
        // them (seen at a 60-degree view; a zoomed TV camera keeps them)
        for (b, _) in &slot_gpu {
            for &gi in b {
                let n = car.meshes.get(meshes[gi].src).map_or("", |m| m.name.as_str()).to_lowercase();
                if ["interior", "steer", "_net", "cable", "belt"].iter().any(|k| n.contains(k)) || n.starts_with("driver:") || n.starts_with("part_") {
                    carcull.set_detail(gi, crate::carcull::Detail::Seen);
                }
            }
        }
        println!(
            "replay: {:.0}s, {} car(s) on {}",
            pb.replay.captured(),
            pb.slots.iter().map(|s| s.cars.len()).sum::<usize>(),
            pb.replay.track.as_deref().unwrap_or("?"));
        // Not Space - that is fly-camera up, and a replay is flown around.
        println!("  P play/pause  ·  , . scrub  ·  - = speed  ·  R restart");
        println!("  C camera (free/chase/onboard/side/orbit)  ·  [ ] change car");
    }
    // Static scenery is culled per instance (tracks and replays; a lone car
    // is a handful of skinned draws and has nothing to gain).
    // EVOVIEW_NO_CULL=1 turns it off to compare.
    let mut culler = (opts.start_fly && crate::envc::var("EVOVIEW_NO_CULL").is_err()).then(|| {
        let skip: Vec<usize> = slot_gpu
            .iter()
            .flat_map(|(b, w)| b.iter().chain(w.iter().flatten()).copied())
            // the cars' coarse LODs move with them too (carcull.rs)
            .chain(carcull.lod_meshes())
            .collect();
        crate::perf::time("cull setup", || crate::cull::Culler::build(&car, &meshes, &skip))
    });
    // A lone car (no culler) sorts its opaque draws front to back.
    // EVOVIEW_NO_SORT=1 keeps load order, to compare.
    let mut draw_order = (!opts.start_fly && opts.playback.is_none()
        && crate::envc::var_os("EVOVIEW_NO_SORT").is_none())
        .then(|| crate::cull::DrawOrder::build(&car, &meshes));
    // Everything on the GPU is built; the CPU copy of the scene (its blob is
    // every mesh and texture byte, ~2 GB on a big track) would otherwise sit
    // in memory for the whole session, because event_loop.run never returns.
    // The seated driver (driver.rs): B hides it by collapsing its instance
    // matrix to zero, which every pass already handles with no branch.
    let find_driver = |car: &Car, meshes: &[GpuMesh]| -> Vec<usize> {
        meshes
            .iter()
            .enumerate()
            .filter(|(_, m)| car.meshes.get(m.src)
                .map_or(false, |c| c.name.starts_with(crate::driver::PREFIX)))
            .map(|(gi, _)| gi)
            .collect()
    };
    let mut driver_meshes: Vec<usize> = find_driver(&car, &meshes);
    let mut driver_on = true;
    if !driver_meshes.is_empty() {
        println!("B toggles the driver");
    }
    drop(car);
    // light matrix the shadow map was last drawn with
    let mut shadow_drawn: Option<glam::Mat4> = None;
    let mut far_drawn: Option<glam::Mat4> = None;
    let mut mid_drawn: Option<glam::Mat4> = None;
    // Set this to force one shadow redraw (e.g. a posed car casting into the
    // map). Cleared once the pass has run.
    let mut shadow_invalid = true;
    // Livery switching: V next, Shift+V previous, or `livery <stem|n|next|prev>`
    // on stdin (ACECM drives a running viewer that way).
    let mut livery = opts.livery.take();
    let livery_rx = livery.as_ref().map(|_| crate::livery::stdin_commands());
    let mut livery_want: Option<usize> = None;
    let livery_base_title = opts.title.clone();
    if let Some(sw) = livery.as_mut() {
        sw.mark_resident(&materials.textures.resident_srcs());
        opts.title = format!("{livery_base_title} · {}", sw.label());
        window.set_title(&opts.title);
        if sw.list.len() > 1 {
            println!("V / Shift+V: next / previous livery ({})", sw.list.len());
        }
    }
    // Tracks start lit. L flips to flat albedo so you can compare.
    let mut lit = opts.start_fly;
    let mut clock_run = false;
    if opts.start_fly {
        println!("L toggles lighting (sun + shade / flat albedo)");
        window.set_title(&format!("{} · lighting on", opts.title));
    }

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    // a bench / capture run takes no input: the window pops up
                    // under whatever the user is typing, and their keys flew
                    // the camera somewhere else
                    WindowEvent::KeyboardInput { .. } | WindowEvent::MouseInput { .. }
                    | WindowEvent::CursorMoved { .. } | WindowEvent::MouseWheel { .. } if bench.is_some() => {}
                    WindowEvent::KeyboardInput { event, .. } => {
                        let pressed = event.state == ElementState::Pressed;
                        // online chat: while the line is open every key types
                        // into it (Enter sends, Esc cancels)
                        if let (Some(line), Some(m)) = (chat_input.as_mut(), opts.mp.as_ref()) {
                            if pressed {
                                match event.physical_key {
                                    PhysicalKey::Code(KeyCode::Enter | KeyCode::NumpadEnter) => {
                                        m.say(line);
                                        chat_input = None;
                                    }
                                    PhysicalKey::Code(KeyCode::Escape) => chat_input = None,
                                    PhysicalKey::Code(KeyCode::Backspace) => {
                                        line.pop();
                                    }
                                    _ => {
                                        if let Some(t) = event.text.as_ref() {
                                            line.extend(t.chars().filter(|c| !c.is_control()));
                                        }
                                    }
                                }
                            }
                            if let PhysicalKey::Code(code) = event.physical_key {
                                held.remove(&code);
                            }
                            return;
                        }
                        let shift = held.contains(&KeyCode::ShiftLeft) || held.contains(&KeyCode::ShiftRight);
                        if pressed && opts.mp.is_some() && matches!(event.physical_key,
                            PhysicalKey::Code(KeyCode::NumpadDivide)) || (pressed && opts.mp.is_some() && shift
                            && matches!(event.physical_key, PhysicalKey::Code(KeyCode::Enter))) {
                            chat_input = Some(String::new());
                            return;
                        }
                        if let PhysicalKey::Code(code) = event.physical_key {
                            if pressed {
                                if let Some(r) = rui.as_mut() {
                                    r.wake();
                                }
                                held.insert(code);
                            } else {
                                held.remove(&code);
                            }
                            // the pit keys (pits.rs): P limiter, Tab menu, Home back to the pits;
                            // with the menu up the arrows, Enter and Backspace are its
                            // (Shift = big steps; held arrows repeat)
                            let mut pit_used = false;
                            if let (true, Some(pt), Some(td)) = (pressed, opts.pits.as_mut(), opts.track_drive.as_mut()) {
                                use crate::pits::Key;
                                let big = held.contains(&KeyCode::ShiftLeft) || held.contains(&KeyCode::ShiftRight);
                                let k = match code {
                                    KeyCode::KeyP => Some(Key::Limiter),
                                    KeyCode::Tab => Some(Key::Menu),
                                    KeyCode::Home => Some(Key::Garage),
                                    KeyCode::ArrowUp if pt.menu => Some(Key::Up),
                                    KeyCode::ArrowDown if pt.menu => Some(Key::Down),
                                    KeyCode::ArrowLeft if pt.menu => Some(Key::Left(big)),
                                    KeyCode::ArrowRight if pt.menu => Some(Key::Right(big)),
                                    KeyCode::Enter | KeyCode::NumpadEnter if pt.menu => Some(Key::Ok),
                                    _ => None,
                                };
                                if let Some(k) = k {
                                    pit_used = true;
                                    if !event.repeat || matches!(k, Key::Left(_) | Key::Right(_) | Key::Up | Key::Down) {
                                        pt.key(k, td);
                                    }
                                }
                            }
                            // ` the driving cluster on / off, Shift+` back to its place
                            if pressed && !event.repeat && code == KeyCode::Backquote && opts.track_drive.is_some() {
                                if held.contains(&KeyCode::ShiftLeft) || held.contains(&KeyCode::ShiftRight) {
                                    cluster.reset();
                                } else {
                                    cluster.toggle();
                                }
                                pit_used = true;
                            }
                            // Backspace: the leaderboard (laps.rs)
                            if pressed && !event.repeat && code == KeyCode::Backspace && opts.track_drive.is_some() {
                                if let Some(l) = opts.laps.as_mut() {
                                    l.toggle_board();
                                }
                                pit_used = true;
                            }
                            if pressed && !event.repeat && !pit_used {
                                // drive-mode keys first; they shadow the viewer's
                                // own bindings only while driving
                                let driving = (1..=3).contains(&drive_mode) && drive.is_some();
                                // the track drive has the rig too: doors and wipers
                                let rigged = drive_mode > 0 && drive.is_some();
                                let rig = skeleton.as_ref().and_then(|s| s.rig.as_deref());
                                let mut used = true;
                                match code {
                                    KeyCode::KeyN if drive.is_some() && opts.track_drive.is_none() => {
                                        drive_mode = (drive_mode + 1) % n_modes;
                                        if let (Some(d), Some(r)) = (drive.as_mut(), rig) {
                                            *d = crate::drive::Drive::new(r);
                                            if drive_mode == 1 { d.cs.gear = 1; }
                                        }
                                        demo_t = 0.0;
                                        println!("drive mode: {}", ["off", "keyboard", "demo", "replay"][drive_mode as usize]);
                                    }
                                    KeyCode::KeyX if driving => { drive.as_mut().unwrap().shift_up(); }
                                    KeyCode::KeyZ if driving => { drive.as_mut().unwrap().shift_down(); }
                                    KeyCode::KeyY if rigged => {
                                        let d = drive.as_mut().unwrap();
                                        d.wipers_on = !d.wipers_on;
                                    }
                                    KeyCode::KeyT if driving => {
                                        let d = drive.as_mut().unwrap();
                                        d.drs = !d.drs;
                                    }
                                    KeyCode::Digit1 | KeyCode::Digit2 | KeyCode::Digit3 | KeyCode::Digit4
                                    | KeyCode::Digit5 | KeyCode::Digit6 | KeyCode::Digit7 | KeyCode::Digit8
                                    | KeyCode::Digit9 if rigged && rig.is_some() => {
                                        let r = rig.unwrap();
                                        let n = match code {
                                            KeyCode::Digit1 => 0, KeyCode::Digit2 => 1, KeyCode::Digit3 => 2,
                                            KeyCode::Digit4 => 3, KeyCode::Digit5 => 4, KeyCode::Digit6 => 5,
                                            KeyCode::Digit7 => 6, KeyCode::Digit8 => 7, _ => 8,
                                        };
                                        let toggles = r.toggle_slots();
                                        if let Some(&si) = toggles.get(n) {
                                            drive.as_mut().unwrap().toggle(si);
                                            println!("{}", r.slots[si].kind.label());
                                        }
                                    }
                                    KeyCode::KeyI | KeyCode::KeyJ | KeyCode::KeyK | KeyCode::KeyL
                                    | KeyCode::ArrowUp | KeyCode::ArrowDown | KeyCode::ArrowLeft
                                    | KeyCode::ArrowRight | KeyCode::Space if driving => {}
                                    _ => used = false,
                                }
                                if !used { match code {
                                    KeyCode::KeyG => ground_on = !ground_on,
                                    KeyCode::KeyB if !driver_meshes.is_empty() => {
                                        driver_on = !driver_on;
                                        let m = if driver_on {
                                            pack_instances(&[], 0)
                                        } else {
                                            vec![0.0f32; 20]
                                        };
                                        for &gi in &driver_meshes {
                                            queue.write_buffer(&meshes[gi].instances, 0,
                                                bytemuck::cast_slice(&m));
                                        }
                                        shadow_invalid = true;
                                        println!("driver {}", if driver_on { "on" } else { "off" });
                                    }
                                    KeyCode::KeyV if livery.is_some() => {
                                        let sw = livery.as_ref().unwrap();
                                        let back = held.contains(&KeyCode::ShiftLeft)
                                            || held.contains(&KeyCode::ShiftRight);
                                        let n = sw.list.len().max(1) as isize;
                                        let i = sw.cur as isize + if back { -1 } else { 1 };
                                        livery_want = Some(i.rem_euclid(n) as usize);
                                    }
                                    KeyCode::KeyP if opts.playback.is_some() => {
                                        playing = !playing;
                                    }
                                    KeyCode::Space if opts.playback.is_some() && fly.is_none() => {
                                        playing = !playing;
                                    }
                                    KeyCode::ArrowLeft | KeyCode::ArrowRight if opts.playback.is_some() => {
                                        let span = opts.playback.as_ref().map_or(0.0, |pb| pb.replay.captured());
                                        let step = if held.contains(&KeyCode::ShiftLeft) || held.contains(&KeyCode::ShiftRight) { 30.0 } else { 5.0 };
                                        let d = if code == KeyCode::ArrowLeft { -step } else { step };
                                        play_t = (play_t + d).clamp(0.0, (span - 0.01).max(0.0));
                                    }
                                    KeyCode::ArrowUp if opts.playback.is_some() => {
                                        play_speed = (play_speed * 2.0).min(16.0);
                                    }
                                    KeyCode::ArrowDown if opts.playback.is_some() => {
                                        play_speed = (play_speed * 0.5).max(0.125);
                                    }
                                    KeyCode::Tab if rui.is_some() => {
                                        if let Some(r) = rui.as_mut() {
                                            r.list = !r.list;
                                        }
                                    }
                                    KeyCode::Backquote if rui.is_some() => {
                                        if let Some(r) = rui.as_mut() {
                                            r.show = !r.show;
                                        }
                                    }
                                    KeyCode::KeyR if opts.playback.is_some() => {
                                        play_t = 0.0;
                                    }
                                    KeyCode::Minus if opts.playback.is_some() => {
                                        play_speed = (play_speed * 0.5).max(0.125);
                                        println!("speed {play_speed}x");
                                    }
                                    KeyCode::Equal if opts.playback.is_some() => {
                                        play_speed = (play_speed * 2.0).min(16.0);
                                        println!("speed {play_speed}x");
                                    }
                                    // FFB strength on top of the game's chain,
                                    // kept in %APPDATA%/evoview/ffb.cfg
                                    KeyCode::NumpadAdd | KeyCode::NumpadSubtract if wheel.is_some() => {
                                        let up = matches!(code, KeyCode::NumpadAdd);
                                        ffb_set.boost = (ffb_set.boost * if up { 1.25 } else { 0.8 }).clamp(0.2, 20.0);
                                        crate::ffb::save_boost(ffb_set.boost);
                                        println!("ffb strength x{:.2}", ffb_set.boost);
                                    }
                                    KeyCode::KeyM if car_snd.is_some() => {
                                        if let Some(s) = car_snd.as_mut() {
                                            s.muted = !s.muted;
                                            println!("sound {}", if s.muted { "muted" } else { "on" });
                                        }
                                    }
                                    KeyCode::KeyM if engine.is_some() => {
                                        if let Some(e) = engine.as_mut() {
                                            e.muted = !e.muted;
                                            println!("sound {}",
                                                if e.muted { "muted" } else { "on" });
                                        }
                                    }
                                    KeyCode::Enter if opts.track_drive.is_some() => {
                                        // drop the car where the camera is, facing its way
                                        if let (Some(td), Some(f)) = (opts.track_drive.as_mut(), fly.as_ref()) {
                                            let d = f.forward();
                                            let at = f.pos + glam::Vec3::new(d.x, 0.0, d.z).normalize_or_zero() * 8.0;
                                            if td.place(at, d.x.atan2(d.z)) {
                                                follow = crate::camera::Follow::Chase;
                                                println!("drive: car placed at ({:.0}, {:.0}, {:.0})", td.pos.x, td.pos.y, td.pos.z);
                                            } else {
                                                println!("drive: no drivable surface under the camera");
                                            }
                                        }
                                    }
                                    KeyCode::Numpad4 | KeyCode::Numpad6 | KeyCode::Numpad5 if opts.race.is_some() => {
                                        // the cars actually on track (online: parked spares are not)
                                        let live: Vec<usize> = opts.race.as_ref().map_or(Vec::new(), |r| r.ai.iter().enumerate()
                                            .filter(|(_, a)| a.td.pos.y > -1000.0).map(|(k, _)| k).collect());
                                        view_ai = if code == KeyCode::Numpad5 || live.is_empty() {
                                            None
                                        } else {
                                            let at = view_ai.and_then(|v| live.iter().position(|&k| k == v));
                                            let n = live.len();
                                            Some(live[match (at, code == KeyCode::Numpad6) {
                                                (None, true) => 0,
                                                (None, false) => n - 1,
                                                (Some(i), true) => (i + 1) % n,
                                                (Some(i), false) => (i + n - 1) % n,
                                            }])
                                        };
                                        if follow == crate::camera::Follow::Free {
                                            follow = crate::camera::Follow::Chase;
                                        }
                                    }
                                    KeyCode::KeyR if opts.race.is_some() => {
                                        if let Some(r) = opts.race.as_mut() {
                                            r.radar_key();
                                        }
                                    }
                                    KeyCode::KeyT if opts.track_drive.is_some() => {
                                        if let Some(td) = opts.track_drive.as_mut() {
                                            // the rig car's own TC (the flag
                                            // alone only drove the old model)
                                            if let Some((rc, _)) = td.rig.as_mut() {
                                                rc.tc_on = !rc.tc_on;
                                                td.tc = rc.tc_on;
                                                println!("drive: traction control {}{}", if rc.tc_on { "on" } else { "off" },
                                                    if rc.tc.is_none() { " (this car has none)" } else { "" });
                                            } else {
                                                td.tc = !td.tc;
                                                println!("drive: traction control {}", if td.tc { "on" } else { "off" });
                                            }
                                        }
                                    }
                                    // cockpit aids, as the game's wheel buttons step them
                                    KeyCode::Numpad7 | KeyCode::Numpad1 | KeyCode::Numpad8 | KeyCode::Numpad2
                                    | KeyCode::Numpad9 | KeyCode::Numpad3
                                        if opts.track_drive.as_ref().map_or(false, |td| td.rig.is_some()) => {
                                        use crate::aids::Knob;
                                        let (k, dir) = match code {
                                            KeyCode::Numpad7 => (Knob::Tc, 1), KeyCode::Numpad1 => (Knob::Tc, -1),
                                            KeyCode::Numpad8 => (Knob::Abs, 1), KeyCode::Numpad2 => (Knob::Abs, -1),
                                            KeyCode::Numpad9 => (Knob::Bias, 1), _ => (Knob::Bias, -1),
                                        };
                                        if let Some((rc, _)) = opts.track_drive.as_mut().and_then(|td| td.rig.as_mut()) {
                                            aid_pop = crate::aids::press(rc, k, dir, true).or(aid_pop);
                                        }
                                    }
                                    KeyCode::KeyA if opts.track_drive.is_some() && follow != crate::camera::Follow::Free => {
                                        if let Some((rc, _)) = opts.track_drive.as_mut().and_then(|td| td.rig.as_mut()) {
                                            rc.abs_on = !rc.abs_on;
                                            println!("drive: ABS {}{}", if rc.abs_on { "on" } else { "off" },
                                                if rc.abs.is_none() { " (this car has none)" } else { "" });
                                        }
                                    }
                                    // the driven car: its own cameras, then the track's
                                        // TV cameras, then the viewer's chase, then free
                                    KeyCode::KeyC if opts.track_drive.is_some() => {
                                        let n = car_cams.len() + 3;
                                        cam_sel = (cam_sel + if held.contains(&KeyCode::ShiftLeft) || held.contains(&KeyCode::ShiftRight) { n - 1 } else { 1 }) % n;
                                        follow = if cam_sel == car_cams.len() + 2 {
                                            crate::camera::Follow::Free
                                        } else {
                                            crate::camera::Follow::Chase
                                        };
                                        let name = cam_name(cam_sel);
                                        println!("camera: {name}");
                                        title_at = 1e9;
                                    }
                                    KeyCode::KeyC if opts.playback.is_some() => {
                                        follow = follow.next();
                                        // Leaving a follow camera hands control
                                        // back to free-look from where it was,
                                        // rather than teleporting the view.
                                        title_at = 1e9;
                                    }
                                    KeyCode::BracketLeft if !drivers.is_empty() => {
                                        watched = (watched + drivers.len() - 1) % drivers.len();
                                        title_at = 1e9;
                                    }
                                    KeyCode::BracketRight if !drivers.is_empty() => {
                                        watched = (watched + 1) % drivers.len();
                                        title_at = 1e9;
                                    }
                                    // Time of day (sky.rs): O later, U earlier,
                                    // 15 min a press (Shift: 1 h); ; runs the
                                    // clock (1 game hour per 10 s)
                                    KeyCode::KeyO | KeyCode::KeyU => {
                                        let big = held.contains(&KeyCode::ShiftLeft)
                                            || held.contains(&KeyCode::ShiftRight);
                                        let m = if big { 60.0 } else { 15.0 };
                                        crate::sky::activate();
                                        crate::sky::step(if code == KeyCode::KeyO { m } else { -m });
                                        let s = crate::sky::now();
                                        println!("time of day: {}", s.describe());
                                        window.set_title(&format!("{} · {}", opts.title, s.clock.hhmm()));
                                    }
                                    KeyCode::Backslash if hdr => {
                                        println!("TAA stays on with the HDR pipeline (EVOVIEW_HDR=0 to compare)");
                                    }
                                    KeyCode::Backslash if render_scale < 1.0 => {
                                        println!("TAA stays on while it upsamples (EVOVIEW_RENDER_SCALE)");
                                    }
                                    KeyCode::Backslash => {
                                        if let Some(t) = taa.as_mut() {
                                            t.on = !t.on;
                                            t.reset();
                                            println!("TAA {}", if t.on { "on" } else { "off" });
                                        }
                                    }
                                    // ' : TAA history off (each frame stands alone) and
                                    // / : motion blur off - to tell which one ghosts
                                    KeyCode::Quote => {
                                        taa_hist_off = !taa_hist_off;
                                        println!("TAA history {}", if taa_hist_off { "off" } else { "on" });
                                    }
                                    KeyCode::Slash => {
                                        if let Some(p) = post.as_mut() {
                                            p.motion_blur = if p.motion_blur > 0.0 { 0.0 } else { 0.25 };
                                            println!("motion blur {}", if p.motion_blur > 0.0 { "on" } else { "off" });
                                        }
                                    }
                                    KeyCode::Semicolon => {
                                        crate::sky::activate();
                                        clock_run = !clock_run;
                                        println!("time of day: clock {}", if clock_run { "running" } else { "stopped" });
                                    }
                                    KeyCode::KeyL => {
                                        lit = !lit;
                                        println!(
                                            "lighting {}",
                                            if lit { "on (sun + shade)" } else { "off (flat albedo)" }
                                        );
                                        window.set_title(&format!(
                                            "{} · lighting {}",
                                            opts.title,
                                            if lit { "on" } else { "off" }
                                        ));
                                    }
                                    KeyCode::KeyH => {
                                        // the light switch: off / stage 1..3
                                        // of the car's .carlightingsystem
                                        let n = crate::lights::STATE_NAMES.len() as u32;
                                        fxst.lights = (fxst.lights + 1) % n;
                                        materials.light_state = fxst.lights;
                                        let name = crate::lights::STATE_NAMES[fxst.lights as usize];
                                        println!("car lights: {name}");
                                        window.set_title(&format!("{} · lights {name}", opts.title));
                                    }
                                    // ---- effects (fxstate.rs): F-keys + the nav block ----
                                    // No letter keys: drive mode owns arrows / IJKL / X Z Y T
                                    // / Space / N, fly owns WASDQE. See KEYS.
                                    KeyCode::F1 => print!("{KEYS}"),
                                    KeyCode::F2 | KeyCode::F3 | KeyCode::F4 | KeyCode::F5 | KeyCode::F6
                                    | KeyCode::F7 | KeyCode::F8 | KeyCode::F9 | KeyCode::F10 | KeyCode::F11
                                    | KeyCode::F12 | KeyCode::PageUp | KeyCode::PageDown | KeyCode::Home
                                    | KeyCode::End | KeyCode::Insert | KeyCode::Delete => {
                                        let shift = held.contains(&KeyCode::ShiftLeft)
                                            || held.contains(&KeyCode::ShiftRight);
                                        let manual = matches!(code, KeyCode::PageUp | KeyCode::PageDown
                                            | KeyCode::Home | KeyCode::End | KeyCode::Insert | KeyCode::Delete);
                                        if manual && driving {
                                            println!("fx: driving - the car state sets brake / rpm / gear / speed (N leaves drive mode)");
                                        } else {
                                            match code {
                                                KeyCode::F2 => fxst.night = !fxst.night,
                                                KeyCode::F3 => fxst.bloom = if fxst.bloom > 0.0 { 0.0 } else { 1.0 },
                                                KeyCode::F4 => fxst.dirt = if fxst.dirt >= 1.0 { 0.0 } else { fxst.dirt + 0.5 },
                                                // F5: dry -> light rain -> heavy rain -> dry
                                                KeyCode::F5 => {
                                                    fxst.rain = if fxst.rain <= 0.0 { 0.45 } else if fxst.rain < 0.9 { 1.0 } else { 0.0 };
                                                    fxst.wet = if fxst.rain > 0.0 { 1.0 } else { 0.0 };
                                                    println!("rain {:.2}", fxst.rain);
                                                    if let Some(d) = drive.as_mut() {
                                                        d.wipers_on = fxst.rain > 0.0;
                                                    }
                                                }
                                                KeyCode::F6 => {
                                                    const T: [f32; 5] = [20.0, 600.0, 750.0, 900.0, 1100.0];
                                                    let i = T.iter().position(|&t| t >= fxst.disc_temp_c - 1.0).unwrap_or(0);
                                                    fxst.disc_temp_c = T[(i + 1) % T.len()];
                                                }
                                                KeyCode::F7 => fxst.backfire_age = 0.0,
                                                KeyCode::F8 => fxst.smoke = if fxst.smoke > 0.0 { 0.0 } else { 1.0 },
                                                KeyCode::F9 => fxst.dust = if fxst.dust > 0.0 { 0.0 } else { 1.0 },
                                                KeyCode::F10 => fxst.next_indicator(),
                                                KeyCode::F11 => fxst.fog = !fxst.fog,
                                                KeyCode::F12 => fxst.exhaust_heat = if fxst.exhaust_heat > 0.0 { 0.0 } else { 1.0 },
                                                KeyCode::PageUp if shift => fxst.gear = (fxst.gear + 1).min(8),
                                                KeyCode::PageDown if shift => fxst.gear = (fxst.gear - 1).max(-1),
                                                KeyCode::PageUp => fxst.rpm = (fxst.rpm + 500.0).min(12000.0),
                                                KeyCode::PageDown => fxst.rpm = (fxst.rpm - 500.0).max(0.0),
                                                KeyCode::Home => fxst.speed_mps = (fxst.speed_mps + 10.0 / 3.6).min(100.0),
                                                KeyCode::End => fxst.speed_mps = (fxst.speed_mps - 10.0 / 3.6).max(0.0),
                                                KeyCode::Insert => fxst.brake = if fxst.brake > 0.0 { 0.0 } else { 1.0 },
                                                KeyCode::Delete => fxst.gear = if fxst.gear < 0 { 0 } else { -1 },
                                                _ => {}
                                            }
                                            let s = fxst.summary();
                                            println!("fx: {s}");
                                            window.set_title(&format!("{} · {s}", opts.title));
                                        }
                                    }
                                    KeyCode::KeyF => {
                                        // toggling back re-frames the car
                                        // rather than leaving the user
                                        // wherever they had flown to
                                        fly = match fly {
                                            Some(_) => None,
                                            None => {
                                                let mut f = Fly::from_orbit(&cam);
                                                // scale to the model so it
                                                // feels the same on a kart or
                                                // a truck
                                                f.speed = (hi - lo).length() * 0.45;
                                                Some(f)
                                            }
                                        };
                                        looking = false;
                                    }
                                    // 1-9 toggle one animation each
                                    KeyCode::Digit1
                                    | KeyCode::Digit2
                                    | KeyCode::Digit3
                                    | KeyCode::Digit4
                                    | KeyCode::Digit5
                                    | KeyCode::Digit6
                                    | KeyCode::Digit7
                                    | KeyCode::Digit8
                                    | KeyCode::Digit9 => {
                                        let n = match code {
                                            KeyCode::Digit1 => 0,
                                            KeyCode::Digit2 => 1,
                                            KeyCode::Digit3 => 2,
                                            KeyCode::Digit4 => 3,
                                            KeyCode::Digit5 => 4,
                                            KeyCode::Digit6 => 5,
                                            KeyCode::Digit7 => 6,
                                            KeyCode::Digit8 => 7,
                                            _ => 8,
                                        };
                                        if let Some(st) = anim_state.get_mut(n) {
                                            st.open = !st.open;
                                            selected = n;
                                            println!(
                                                "{} {}",
                                                animations[n].name,
                                                if st.open { "opening" } else { "closing" }
                                            );
                                        }
                                    }
                                    KeyCode::Digit0 => {
                                        for st in anim_state.iter_mut() {
                                            st.open = false;
                                        }
                                        println!("closing all");
                                    }
                                    KeyCode::BracketRight | KeyCode::BracketLeft
                                        if !animations.is_empty() =>
                                    {
                                        let n = animations.len();
                                        selected = if code == KeyCode::BracketRight {
                                            (selected + 1) % n
                                        } else {
                                            (selected + n - 1) % n
                                        };
                                        println!("selected: {}", animations[selected].name);
                                    }
                                    KeyCode::Enter if !animations.is_empty() => {
                                        if let Some(st) = anim_state.get_mut(selected) {
                                            st.open = !st.open;
                                        }
                                    }
                                    KeyCode::Escape => elwt.exit(),
                                    _ => {}
                                } }
                            }
                        }
                    }
                    WindowEvent::Resized(_) if bench_rt.is_some() => {}
                    WindowEvent::Resized(new) => {
                        config.width = new.width.max(1);
                        config.height = new.height.max(1);
                        surface.configure(&device, &config);
                        (rw, rh) = crate::taa::scaled(config.width, config.height, render_scale);
                        targets = make_targets(&device, rw, rh, config.format, samples);
                        if let Some(p) = post.as_mut() {
                            p.resize(&device, config.width, config.height);
                        }
                        if let Some(s) = ssr.as_mut() {
                            s.resize(&device, rw, rh);
                        }
                        if let Some(t) = taa.as_mut() {
                            t.resize(&device, config.width, config.height, rw, rh);
                        }
                        if let Some(m) = motion.as_mut() {
                            m.resize(&device, rw, rh);
                        }
                        if let Some(d) = dlss.as_mut() {
                            d.resize(&device, config.width, config.height);
                        }
                        if let Some(f) = fsr.as_mut() {
                            f.resize(&device, config.width, config.height);
                        }
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        let down = state == ElementState::Pressed;
                        // A press on the overlay is a button press, not the
                        // start of a camera drag - test the HUD first and
                        // swallow the event if it lands on a control.
                        // the driving cluster is dragged with the left button
                        if button == MouseButton::Left && opts.track_drive.is_some() {
                            let grab = down && last_pos.map_or(false, |p| {
                                cluster.press((p.0 as f32) / config.width as f32, (p.1 as f32) / config.height as f32)
                            });
                            if grab || (!down && cluster.release()) {
                                return;
                            }
                        }
                        if down && button == MouseButton::Left {
                            let hit = last_pos.and_then(|p| {
                                hud.hit((p.0 as f32) / config.width as f32,
                                        (p.1 as f32) / config.height as f32)
                            });
                            if let Some(a) = hit {
                                use crate::hud::Action::*;
                                let span = opts.playback.as_ref().map_or(0.0, |pb| pb.replay.captured());
                                let end = (span - 0.01).max(0.0);
                                match a {
                                    PlayPause => playing = !playing,
                                    Slower => play_speed = (play_speed * 0.5).max(0.125),
                                    Faster => play_speed = (play_speed * 2.0).min(16.0),
                                    Restart => play_t = 0.0,
                                    Back => play_t = (play_t - 5.0).clamp(0.0, end),
                                    Fwd => play_t = (play_t + 5.0).clamp(0.0, end),
                                    Camera => {
                                        follow = follow.next();
                                        title_at = 1e9;
                                    }
                                    PrevCar if !drivers.is_empty() => {
                                        watched = (watched + drivers.len() - 1) % drivers.len();
                                    }
                                    NextCar if !drivers.is_empty() => {
                                        watched = (watched + 1) % drivers.len();
                                    }
                                    Watch(i) if i < drivers.len() => watched = i,
                                    ToggleList => {
                                        if let Some(r) = rui.as_mut() {
                                            r.list = !r.list;
                                        }
                                    }
                                    // a press on the timeline jumps there and
                                    // keeps following the cursor until release
                                    Seek(f) => {
                                        play_t = (f * span).clamp(0.0, end);
                                        if let Some(r) = rui.as_mut() {
                                            r.scrub = true;
                                        }
                                    }
                                    _ => {}
                                }
                                return;
                            }
                        }
                        if !down && button == MouseButton::Left {
                            if let Some(r) = rui.as_mut().filter(|r| r.scrub) {
                                r.scrub = false;
                                return;
                            }
                        }
                        match button {
                            MouseButton::Left => {
                                if fly.is_some() {
                                    looking = down;
                                } else {
                                    dragging = down;
                                }
                            }
                            MouseButton::Right => {
                                looking = down && fly.is_some();
                            }
                            _ => {}
                        }
                        if !down {
                            last_pos = None;
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let p = (position.x, position.y);
                        if let Some(r) = rui.as_mut() {
                            r.wake();
                            if r.scrub {
                                if let Some(pb) = opts.playback.as_ref() {
                                    let span = pb.replay.captured();
                                    play_t = (r.seek_at(p.0 as f32 / config.width as f32) * span).clamp(0.0, (span - 0.01).max(0.0));
                                }
                                last_pos = Some(p);
                                return;
                            }
                        }
                        if cluster.moved((p.0 as f32) / config.width as f32, (p.1 as f32) / config.height as f32) {
                            last_pos = Some(p);
                            return;
                        }
                        if let Some(l) = last_pos {
                            let dx = (p.0 - l.0) as f32;
                            let dy = (p.1 - l.1) as f32;
                            match fly.as_mut() {
                                Some(f) if looking => f.look(dx, dy),
                                None if dragging => cam.orbit(dx, dy),
                                _ => {}
                            }
                        }
                        last_pos = Some(p);
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let y = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y,
                            MouseScrollDelta::PixelDelta(p) => p.y as f32 / 60.0,
                        };
                        // over the replay's car list the wheel scrolls it
                        let cur = last_pos.map(|p| ((p.0 as f32) / config.width as f32, (p.1 as f32) / config.height as f32));
                        if rui.as_mut().map_or(false, |r| r.wheel(cur, y, drivers.len())) {
                            return;
                        }
                        // in free-look the wheel sets how fast you fly
                        match fly.as_mut() {
                            Some(f) => {
                                f.speed = (f.speed * (1.0 + y * 0.15)).clamp(0.02, 500.0);
                            }
                            None => cam.zoom(y),
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let _frame_t = crate::bench::cpu("redraw total");
                        if let (Some(rx), Some(sw)) = (livery_rx.as_ref(), livery.as_ref()) {
                            while let Ok(cmd) = rx.try_recv() {
                                let mut it = cmd.splitn(2, ' ');
                                match (it.next().unwrap_or(""), it.next().map(str::trim)) {
                                    ("livery", Some("next")) => {
                                        livery_want = Some((sw.cur + 1) % sw.list.len().max(1));
                                    }
                                    ("livery", Some("prev")) => {
                                        let n = sw.list.len().max(1);
                                        livery_want = Some((sw.cur + n - 1) % n);
                                    }
                                    ("livery", Some(w)) => match sw.find(w) {
                                        Some(i) => livery_want = Some(i),
                                        None => println!("livery {w}: no such livery"),
                                    },
                                    ("quit", _) => elwt.exit(),
                                    _ => println!("unknown command: {cmd}"),
                                }
                            }
                        }
                        if let (Some(i), Some(sw)) = (livery_want.take(), livery.as_mut()) {
                            let t0 = std::time::Instant::now();
                            match sw.select(i as isize) {
                                Ok(l) => {
                                    let t_load = t0.elapsed().as_secs_f32();
                                    let prev = std::mem::take(&mut materials.textures);
                                    let state = materials.light_state;
                                    let mut nm = Materials::build_reusing(&device, &queue, &l.car, prev);
                                    nm.set_light_state(&queue, state);
                                    let (mut nmeshes, nlo, nhi, _) = build_scene(&device, &l.car, Some(&nm));
                                    if ground_mesh.is_some() {
                                        if let Some(mut g) = meshes.pop() {
                                            g.draws = vec![(0, 6, nm.ground)];
                                            nmeshes.push(g);
                                            ground_mesh = Some(nmeshes.len() - 1);
                                        }
                                    }
                                    if draw_order.is_some() {
                                        draw_order = Some(crate::cull::DrawOrder::build(&l.car, &nmeshes));
                                    }
                                    sw.mark_resident(&nm.textures.resident_srcs());
                                    driver_meshes = find_driver(&l.car, &nmeshes);
                                    if !driver_on {
                                        let z = vec![0.0f32; 20];
                                        for &gi in &driver_meshes {
                                            queue.write_buffer(&nmeshes[gi].instances, 0,
                                                bytemuck::cast_slice(&z));
                                        }
                                    }
                                    fxg = crate::fxgpu::FxGpu::new(&device, &queue, &l.car, &nm, scene_format, samples, &uniform_buf);
                                    meshes = nmeshes;
                                    if let Some(m) = motion.as_mut() {
                                        m.reset();
                                    }
                                    materials = nm;
                                    lo = nlo;
                                    hi = nhi;
                                    shadow_invalid = true;
                                    drop(l);
                                    opts.title = format!("{livery_base_title} · {}", sw.label());
                                    window.set_title(&opts.title);
                                    println!("{} - switched in {:.0} ms (load {:.0} ms)",
                                             sw.label(), t0.elapsed().as_secs_f32() * 1000.0,
                                             t_load * 1000.0);
                                }
                                Err(e) => eprintln!("livery switch failed: {e}"),
                            }
                        }
                        let aspect = config.width as f32 / config.height as f32;
                        // EVOVIEW_FIXED_DT=s: every frame advances exactly s (evenly
                        // spaced captures for a GIF, whatever the render speed)
                        let dt = crate::envc::var("EVOVIEW_FIXED_DT").ok().and_then(|v| v.parse::<f32>().ok())
                            .unwrap_or_else(|| {
                                let m = last_frame.elapsed().as_secs_f32().min(0.1);
                                // on cadence: the exact period, not the jittery clock
                                match drive_pace.as_mut() {
                                    Some(pc) => pc.frame(m),
                                    None => m,
                                }
                            });
                        last_frame = std::time::Instant::now();
                        fxst.time += dt;
                        crate::clouds::advance(dt);
                        if clock_run {
                            // 1 game hour per 10 s
                            crate::sky::step(dt as f64 * 6.0);
                        }
                        if fxst.backfire_age >= 0.0 {
                            fxst.backfire_age += dt;
                            if fxst.backfire_age > 3.0 {
                                fxst.backfire_age = -1.0;
                            }
                        }

                        // Rolling FPS in the title. "Is it fast?" should be a
                        // number you can read, not a feeling.
                        fps_frames += 1;
                        if fps_clock.elapsed().as_secs_f32() >= 0.5 {
                            let f = fps_frames as f32 / fps_clock.elapsed().as_secs_f32();
                            fps_now = f;
                            fps_frames = 0;
                            fps_clock = std::time::Instant::now();
                            window.set_title(&format!("{} · {:.0} fps ({:.2} ms)",
                                opts.title, f, 1000.0 / f.max(1.0)));
                        }
                        let _ = fps_now;

                        // Movement is per-second, not per-frame, so it feels
                        // identical whether the viewer runs at 60 or 4000 fps.
                        if let Some(f) = fly.as_mut() {
                            let mut dir = Vec3::ZERO;
                            let fwd = f.forward();
                            let right = f.right();
                            if held.contains(&KeyCode::KeyW) { dir += fwd; }
                            if held.contains(&KeyCode::KeyS) { dir -= fwd; }
                            if held.contains(&KeyCode::KeyD) { dir += right; }
                            if held.contains(&KeyCode::KeyA) { dir -= right; }
                            if held.contains(&KeyCode::KeyE)
                                || held.contains(&KeyCode::Space) { dir += Vec3::Y; }
                            if held.contains(&KeyCode::KeyQ)
                                || held.contains(&KeyCode::ControlLeft) { dir -= Vec3::Y; }
                            if dir != Vec3::ZERO {
                                let boost = if held.contains(&KeyCode::ShiftLeft) {
                                    4.0
                                } else if held.contains(&KeyCode::AltLeft) {
                                    0.2
                                } else {
                                    1.0
                                };
                                f.pos += dir.normalize() * f.speed * boost * dt;
                            }
                        }

                        // Advance the replay and rewrite every car's instance
                        // matrix. One write per mesh per model, not per car:
                        // all of a model's drivers live in one instance
                        // buffer, so a 24-car grid of 6 models is 6 uploads.
                        if let Some(pb) = opts.playback.as_ref() {
                            let span = pb.replay.captured();
                            let scrub = if held.contains(&KeyCode::Comma) {
                                -1.0
                            } else if held.contains(&KeyCode::Period) {
                                1.0
                            } else {
                                0.0
                            };
                            // how far the recording moved this frame (the rigs'
                            // springs and wheel spin advance by it)
                            let mut play_dt = 0.0f32;
                            if scrub != 0.0 {
                                play_t += scrub * dt * 10.0 * play_speed;
                                play_dt = dt * 10.0 * play_speed;
                            } else if playing {
                                play_t += dt * play_speed;
                                play_dt = dt * play_speed;
                            }
                            // Loop, so a replay left running keeps showing
                            // something rather than freezing on the last frame.
                            if span > 0.0 {
                                play_t = play_t.rem_euclid(span);
                            }
                            // Body and the four wheels are separate instance
                            // streams: the wheels steer, spin and follow their
                            // own suspension travel, so they cannot share the
                            // body's matrix.
                            let mut body: Vec<f32> = Vec::new();
                            // the driver meshes' copy: the watched car's driver
                            // hidden while the onboard camera sits in its head
                            let mut body_drv: Vec<f32> = Vec::new();
                            let hide_drv = drivers.get(watched).map(|&(ci, _)| ci).filter(|&ci| {
                                follow == crate::camera::Follow::Onboard
                                    && pb.slots.iter().any(|s| s.cars.contains(&ci) && s.cockpit.is_some())
                            });
                            let mut wheel: [Vec<f32>; 4] = Default::default();
                            let mut smokers: Vec<crate::particles::CarIn> = Vec::new();
                            let _rc = crate::bench::cpu("replay cars");
                            pose_jobs.clear();
                            for (si, slot) in pb.slots.iter().enumerate() {
                                body.clear();
                                body_drv.clear();
                                for w in wheel.iter_mut() {
                                    w.clear();
                                }
                                for &ci in &slot.cars {
                                    // A driver whose stream has ended collapses
                                    // to a zero matrix - every vertex lands on
                                    // one point, so the car simply is not
                                    // there, with no branch in the shader.
                                    let (b, mut wm) = crate::replayscene::car_pose(
                                        &pb.replay, &pb.wheels, slot, ci, play_t)
                                        .unwrap_or((glam::Mat4::ZERO, [glam::Mat4::ZERO; 4]));
                                    // EVOVIEW_REPLAY_MAX_CARS=n: only the first n cars (measuring)
                                    let b = if crate::envc::var("EVOVIEW_REPLAY_MAX_CARS").ok().and_then(|v| v.parse::<usize>().ok()).map_or(false, |n| ci >= n) { glam::Mat4::ZERO } else { b };
                                    // the car's own rig, posed from its recording:
                                    // everything rides the body, the bones turn,
                                    // spin and spring the wheels, lean the hull,
                                    // steer the wheel and the driver's arms
                                    let mut tail = [0.0f32; 4];
                                    let _rr = crate::bench::cpu("replay rig");
                                    if let (Some(rig), true) = (slot.skel.as_ref().and_then(|k| k.rig.as_deref()), b != glam::Mat4::ZERO) {
                                        let (d, st) = rep_rigs.entry(ci).or_insert_with(|| {
                                            (crate::drive::Drive::new(rig), crate::drive::ReplayRig::new(&pb.replay, ci, false))
                                        });
                                        // every car's rig state moves on; only the
                                        // ones on screen are posed (below, once
                                        // the camera is final)
                                        st.apply(&pb.replay, d, rig, play_t, play_dt);
                                        pose_jobs.push((ci, si, b.w_axis.truncate()));
                                        wm = [b; 4];
                                        tail[0] = -((1 + ci) as f32);
                                    }
                                    drop(_rr);
                                    let _rf = crate::bench::cpu("replay fx");
                                    // brake lights: drawn as an AI car's (shader fn_i)
                                    tail[2] = 1.0 + pb.replay.telemetry(ci, play_t).map_or(0.0, |t| t.brake).clamp(0.0, 1.0);
                                    // tyre smoke from the slide: nose off the
                                    // direction of travel, at speed
                                    if let (Some(p0), Some(p1)) = (pb.replay.sample(ci, (play_t - 0.06).max(0.0)), pb.replay.sample(ci, play_t)) {
                                        let v = (p1.pos - p0.pos) / (play_t - (play_t - 0.06).max(0.0)).max(1e-3);
                                        let flat = glam::Vec3::new(v.x, 0.0, v.z);
                                        let f = glam::Vec3::new(p1.forward.x, 0.0, p1.forward.z).normalize_or_zero();
                                        let slip = f.cross(flat.normalize_or_zero()).length();
                                        let s = smoothstep(0.2, 0.55, slip) * smoothstep(2.0, 6.0, flat.length());
                                        if b != glam::Mat4::ZERO {
                                            // every effect's inputs: the slide, the
                                            // surface under each wheel, a scrape,
                                            // the throttle (backfires on a lift)
                                            let inv = b.inverse();
                                            let mut cin = crate::particles::CarIn::at(b, s);
                                            cin.vel = v;
                                            cin.wet = crate::rain::wetness();
                                            for (k, w) in p1.wheels.iter().enumerate() {
                                                let (gr, di) = surface_named(w.x, w.z, w.y + 1.0).0;
                                                cin.wheels[k] = (inv.transform_point3(*w).into(), gr, di);
                                            }
                                            if let Some(t) = pb.replay.telemetry(ci, play_t) {
                                                cin.throttle = t.throttle;
                                                cin.rpm = t.rpm;
                                                if t.scrape > 0.01 {
                                                    let mid = (p1.wheels[0] + p1.wheels[1] + p1.wheels[2] + p1.wheels[3]) * 0.25
                                                        - p1.up * 0.34;
                                                    let at = mid + p1.forward * t.scrape_at[0] + p1.right * t.scrape_at[1];
                                                    cin.scrape = t.scrape;
                                                    cin.scrape_at = inv.transform_point3(at).into();
                                                }
                                            }
                                            if crate::envc::var_os("EVOVIEW_FX_LOG").is_some() {
                                                let ids: Vec<String> = p1.wheels.iter().map(|w| surface_named(w.x, w.z, w.y + 1.0).1).collect();
                                                eprintln!("FXLOG t {play_t:.2} car {ci} v {:.1} smoke {:.2} scrape {:.2} thr {:.2} wheels {:?}",
                                                          v.length(), s, cin.scrape, cin.throttle, ids);
                                            }
                                            smokers.push(cin);
                                        }
                                    }
                                    body.extend_from_slice(&b.to_cols_array());
                                    body.extend_from_slice(&tail);
                                    if hide_drv == Some(ci) {
                                        body_drv.extend_from_slice(&[0.0f32; 20]);
                                    } else {
                                        body_drv.extend_from_slice(&b.to_cols_array());
                                        body_drv.extend_from_slice(&tail);
                                    }
                                    for k in 0..4 {
                                        wheel[k].extend_from_slice(&wm[k].to_cols_array());
                                        wheel[k].extend_from_slice(&tail);
                                    }
                                }
                                for &gi in &slot_gpu[si].0 {
                                    let rows = if driver_meshes.contains(&gi) { &body_drv } else { &body };
                                    carcull.put(gi, rows);
                                }
                                for k in 0..4 {
                                    for &gi in &slot_gpu[si].1[k] {
                                        carcull.put(gi, &wheel[k]);
                                    }
                                }
                            }
                            // world-space emitters; an empty list would mean
                            // "one car at the origin", so park one with no smoke
                            if smokers.is_empty() {
                                smokers.push(crate::particles::CarIn::at(glam::Mat4::IDENTITY, 0.0));
                            }
                            fxg.sim.cars = smokers;
                        }

                        // The driven car (trackdrive.rs): step the physics on
                        // the arrows, then write its body and wheel instances
                        // the same way a replay does.
                        if let Some(td) = opts.track_drive.as_mut() {
                            // the pit menu has the arrows while it is up
                            let menu_keys = opts.pits.as_ref().map_or(false, |p| p.menu_takes_arrows());
                            let k = |c: KeyCode| held.contains(&c)
                                && !(menu_keys && matches!(c, KeyCode::ArrowUp | KeyCode::ArrowDown | KeyCode::ArrowLeft | KeyCode::ArrowRight));
                            let wi = wheel.as_mut().and_then(|w| w.read());
                            let mut c = crate::trackdrive::Controls {
                                // EVOVIEW_AUTO_THROTTLE: flat out (capturing a launch)
                                throttle: if k(KeyCode::ArrowUp) || crate::envc::var_os("EVOVIEW_AUTO_THROTTLE").is_some() { 1.0 } else { 0.0 },
                                brake: if k(KeyCode::ArrowDown) { 1.0 } else { 0.0 },
                                steer: (k(KeyCode::ArrowLeft) as i32 - k(KeyCode::ArrowRight) as i32) as f32,
                                handbrake: k(KeyCode::ControlRight) || k(KeyCode::ControlLeft),
                                analog: false,
                                ..Default::default()
                            };
                            // keyboard box: PageUp / PageDown shift, End starts
                            let keys = [k(KeyCode::PageUp), k(KeyCode::PageDown), k(KeyCode::End)];
                            c.shift_up = keys[0] && !kb_prev[0];
                            c.shift_down = keys[1] && !kb_prev[1];
                            c.starter = keys[2] && !kb_prev[2];
                            kb_prev = keys;
                            if let Some(w) = wi {
                                c.steer = w.steer;
                                c.throttle = c.throttle.max(w.throttle);
                                c.brake = c.brake.max(w.brake);
                                c.analog = true;
                                c.clutch = w.clutch;
                                c.shift_up |= w.shift_up;
                                c.shift_down |= w.shift_down;
                                c.h_gear = w.h_gear;
                                c.starter |= w.starter;
                                c.wheel_deg = Some(w.steer * ffb_set.wheel_range * 0.5);
                                c.handbrake |= w.handbrake.map_or(false, |h| h > 0.5);
                                c.handbrake_v = w.handbrake.map(|h| if c.handbrake && h <= 0.5 { 1.0 } else { h });
                            }
                            // cockpit aids from the wheel's buttons; the pit menu
                            // borrows the TC / ABS buttons while it is up (the
                            // game's own binds, wheel.rs GAME_ROLES)
                            if let (Some(w), Some((rc, _))) = (wi.as_ref(), td.rig.as_mut()) {
                                for (i, k) in crate::aids::Knob::ALL.iter().enumerate() {
                                    if menu_keys && matches!(k, crate::aids::Knob::Tc | crate::aids::Knob::Abs) {
                                        continue;
                                    }
                                    if w.aids[i] != 0 {
                                        aid_pop = crate::aids::press(rc, *k, w.aids[i], true).or(aid_pop);
                                    }
                                }
                            }
                            if let (Some(every), Some((rc, _))) = (crate::envc::var("EVOVIEW_AIDS_DEMO").ok().and_then(|v| v.parse::<f32>().ok()), td.rig.as_mut()) {
                                aid_demo.0 += dt;
                                if aid_demo.0 >= every {
                                    aid_demo.0 = 0.0;
                                    let k = crate::aids::Knob::ALL[aid_demo.1 % 3];
                                    aid_demo.1 += 1;
                                    aid_pop = crate::aids::press(rc, k, 1, false).or(aid_pop);
                                }
                            }
                            if let Some(p) = aid_pop.as_mut() {
                                p.age += dt;
                            }
                            if let Some(d) = opts.line_driver.as_mut() {
                                if opts.race.as_ref().map_or(false, |r| r.held()) {
                                    d.rest();
                                }
                                c = d.controls(td, dt);
                            }
                            // held on the grid until the lights go out
                            if let Some(r) = opts.race.as_ref() {
                                r.filter_player(&mut c, td.speed_kmh());
                            }
                            pedals = (c.throttle, c.brake, c.clutch, c.handbrake_v.unwrap_or(if c.handbrake { 1.0 } else { 0.0 }));
                            // the pit buttons, the crew's hold and the limiter
                            if let Some(p) = opts.pits.as_mut() {
                                if let Some(w) = wi.as_ref() {
                                    p.wheel(w.pit, td);
                                }
                                p.filter(&mut c, td);
                            }
                            let _t = crate::bench::cpu("car physics");
                            td.step(&c, dt);
                            if let Some(p) = opts.pits.as_mut() {
                                p.update(td, dt);
                            }
                            if let Some(l) = opts.laps.as_mut() {
                                let in_pit = opts.pits.as_ref().map_or(false, |p| p.in_lane);
                                l.step(td, in_pit, dt);
                            }
                            // the AI cars' tyre smoke / spray / backfires (particles.rs),
                            // after the player's in the emitter list
                            let mut ai_smokers: Vec<crate::particles::CarIn> = Vec::new();
                            // the opponents drive, then every car's contacts
                            // (the player's car can be pushed here)
                            if let Some(r) = opts.race.as_mut() {
                                // online: send ours, place the others (unused slots
                                // parked far below the track)
                                if let Some(m) = opts.mp.as_ref() {
                                    // late joiners: queue their cars, and put each
                                    // one that has loaded into the scene (its own
                                    // materials and meshes appended, a new slot)
                                    for job in m.new_cars() {
                                        net_job_tx.send(job).ok();
                                    }
                                    while let Ok((n, model, res)) = net_done_rx.try_recv() {
                                        let (mini, slot) = match res {
                                            Ok(x) => x,
                                            Err(e) => {
                                                eprintln!("mp: car #{n}: {e}");
                                                continue;
                                            }
                                        };
                                        let t0 = std::time::Instant::now();
                                        let prev = std::mem::take(&mut materials.textures);
                                        let mut mm = Materials::build_reusing(&device, &queue, &mini, prev);
                                        mm.set_light_state(&queue, materials.light_state);
                                        let (mut nmeshes, _, _, _) = build_scene(&device, &mini, Some(&mm));
                                        materials.textures = std::mem::take(&mut mm.textures);
                                        let off = materials.append(mm);
                                        let base = meshes.len();
                                        let mut to_gpu = vec![usize::MAX; mini.meshes.len()];
                                        for (k, g) in nmeshes.iter_mut().enumerate() {
                                            if let Some(t) = to_gpu.get_mut(g.src) {
                                                *t = base + k;
                                            }
                                            g.src = usize::MAX;
                                            for d in g.draws.iter_mut() {
                                                d.2 += off;
                                            }
                                        }
                                        let map = |v: &Vec<usize>| -> Vec<usize> {
                                            v.iter().filter_map(|&si| to_gpu.get(si).copied()).filter(|&gi| gi != usize::MAX).collect()
                                        };
                                        let body = map(&slot.meshes);
                                        for &si in &slot.meshes {
                                            let name = mini.meshes.get(si).map_or(String::new(), |m| m.name.to_lowercase());
                                            if let Some(&gi) = to_gpu.get(si).filter(|&&g| g != usize::MAX) {
                                                if ["interior", "steer", "_net", "cable", "belt"].iter().any(|k| name.contains(k))
                                                    || name.starts_with("driver:") || name.starts_with("part_") {
                                                    carcull.set_detail(gi, crate::carcull::Detail::Near);
                                                }
                                            }
                                        }
                                        slot_gpu.push((body, [map(&slot.wheels[0]), map(&slot.wheels[1]), map(&slot.wheels[2]), map(&slot.wheels[3])]));
                                        meshes.extend(nmeshes);
                                        for &(b, d, l) in &slot.lods {
                                            if let (Some(&gb), Some(&gl)) = (to_gpu.get(b), to_gpu.get(l)) {
                                                if gb != usize::MAX && gl != usize::MAX {
                                                    carcull.add_lod(&mut meshes, gb, d, gl);
                                                }
                                            }
                                        }
                                        opts.ai_slots.push(slot.clone());
                                        let si = opts.ai_slots.len();
                                        let car_td = crate::trackdrive::TrackDrive::with_model(td.ground.clone(), slot, None, false, false);
                                        r.add_net(car_td, si, 0);
                                        m.add_ai(&model, n);
                                        shadow_invalid = true;
                                        println!("mp: car #{n} ({model}) joined the scene in {:.0} ms", t0.elapsed().as_secs_f32() * 1000.0);
                                    }
                                    m.set_me(td.net_state());
                                    let placed = m.placed();
                                    if crate::envc::var_os("EVOVIEW_MP_LOG").is_some() {
                                        static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
                                        let mut l = LAST.lock().unwrap();
                                        if l.map_or(true, |t| t.elapsed().as_secs_f32() > 5.0) {
                                            *l = Some(std::time::Instant::now());
                                            let on: Vec<String> = placed.iter().enumerate().filter_map(|(k, x)| x.map(|x| {
                                                format!("{k}:{:?}@{:.0},{:.0}", m.who(k).map(|w| w.0), x.state.pos.x, x.state.pos.z)
                                            })).collect();
                                            println!("mp placed ({} of {}): {}", on.len(), placed.len(), on.join(" "));
                                        }
                                    }
                                    // EVOVIEW_MP_TRACE=1: the first other car as drawn, every frame
                                    if crate::envc::var_os("EVOVIEW_MP_TRACE").is_some() {
                                        if let Some(x) = placed.iter().flatten().next() {
                                            println!("MPTRACE {:.4} {:.3} {:.3} {:.3}", crate::render::trace_clock(), x.state.pos.x, x.state.pos.z, x.at.elapsed().as_secs_f32());
                                        }
                                    }
                                    for (k, a) in r.ai.iter_mut().enumerate() {
                                        match placed.get(k).copied().flatten() {
                                            Some(x) => a.td.set_net(&x.state, x.at.elapsed().as_secs_f32(), dt),
                                            None => a.td.pos = glam::Vec3::new(0.0, -5000.0, 0.0),
                                        }
                                    }
                                    // the session's time of day (its start + elapsed x its
                                    // multiplier), whenever it has moved a minute
                                    if let Some((mins, (y, mo, d))) = m.game_time() {
                                        let mut c = crate::sky::clock();
                                        if (c.minutes - mins).abs() > 1.0 || (c.year, c.month, c.day) != (y, mo, d) {
                                            c.minutes = mins;
                                            (c.year, c.month, c.day) = (y, mo, d);
                                            crate::sky::set(c);
                                            crate::sky::activate();
                                        }
                                    }
                                }
                                r.step(td, dt);
                                // their laps, and the board in race order
                                if let Some(l) = opts.laps.as_mut() {
                                    l.step_ai(&r.ai.iter().map(|a| &a.td).collect::<Vec<_>>(), dt);
                                    // a standing start: lap 1 runs from the lights
                                    if r.just_started() && r.start == crate::race::Start::Standing {
                                        l.race_start();
                                    }
                                    l.others = l.race_rows(&r.race_order(), |k| r.name(k).to_string());
                                }
                                // each AI slot's instances: body, then wheels
                                let mut rows: Vec<(Vec<f32>, [Vec<f32>; 4])> = opts.ai_slots.iter()
                                    .map(|s| (vec![0.0; s.cars.len() * 20], std::array::from_fn(|_| vec![0.0; s.cars.len() * 20])))
                                    .collect();
                                let put = |v: &mut Vec<f32>, i: usize, m: glam::Mat4| {
                                    if let Some(o) = v.get_mut(i * 20..i * 20 + 16) {
                                        o.copy_from_slice(&m.to_cols_array());
                                    }
                                };
                                for a in &r.ai {
                                    let Some((b, w)) = rows.get_mut(a.slot - 1) else { continue };
                                    let (bm, wm) = a.td.matrices();
                                    put(b, a.inst, bm);
                                    // facing.z: an AI car and its brake pedal (its brake lamps)
                                    if let Some(f) = b.get_mut(a.inst * 20 + 18) {
                                        *f = 1.0 + a.brake().clamp(0.0, 1.0);
                                    }
                                    // EVOVIEW_AI_FX_LOG=1: each AI's effect inputs a frame
                                    if crate::envc::var_os("EVOVIEW_AI_FX_LOG").is_some() {
                                        println!("AIFX t {:6.2} {:13} brk {:.2} smoke {:.2} {:5.0} km/h {:5.1} m", r.t, a.name, a.brake(), a.smoke,
                                                 a.td.speed_kmh(), a.td.pos.distance(cut_prev.map_or(td.pos, |c| c.0)));
                                    }
                                    for k in 0..4 {
                                        put(&mut w[k], a.inst, wm[k]);
                                    }
                                }
                                // the near cars' rows, packed first, for the detail meshes
                                let near_eye = cut_prev.map_or(td.pos, |c| c.0);
                                // emitters: every AI keeps its place in the list (the
                                // sim's per-car carry and backfire state), but only those
                                // within EVOVIEW_AI_SMOKE_M of the camera emit (0 = none)
                                // and at most EVOVIEW_AI_SMOKE_N of them, the nearest (4)
                                let smoke_m = crate::envc::var("EVOVIEW_AI_SMOKE_M").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(150.0);
                                let smoke_n = crate::envc::var("EVOVIEW_AI_SMOKE_N").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4);
                                let mut near: Vec<(f32, usize)> = r.ai.iter().enumerate()
                                    .map(|(k, a)| (a.td.pos.distance(near_eye), k)).filter(|d| d.0 < smoke_m).collect();
                                near.sort_by(|a, b| a.0.total_cmp(&b.0));
                                near.truncate(smoke_n);
                                for (k, a) in r.ai.iter().enumerate() {
                                    let (bm, wm) = a.td.matrices();
                                    let mut c = crate::particles::CarIn::at(bm, 0.0);
                                    if near.iter().any(|n| n.1 == k) {
                                        let f = a.td.sound_feed(false);
                                        c.smoke = a.smoke;
                                        c.throttle = f.throttle;
                                        c.rpm = f.rpm;
                                        c.wet = fxst.wet;
                                        c.vel = a.td.vel();
                                        let inv = bm.inverse();
                                        for (k, m) in wm.iter().enumerate() {
                                            c.wheels[k].0 = inv.transform_point3(m.w_axis.truncate()).into();
                                        }
                                    }
                                    ai_smokers.push(c);
                                }
                                // EVOVIEW_AI_DRAW=0: the AI drive but are not drawn (their GPU cost, A/B)
                                if crate::envc::var("EVOVIEW_AI_DRAW").map_or(false, |v| v == "0") {
                                    for (b, w) in slot_gpu.iter().skip(1) {
                                        for &gi in b.iter().chain(w.iter().flatten()) {
                                            meshes[gi].instance_count = 0;
                                        }
                                    }
                                    rows.clear();
                                }
                                // a look none of whose cars is on track within
                                // EVOVIEW_AI_DRAW_M (1500 m) of the camera is not drawn at
                                // all (online: 30 looks, most players km away round the
                                // Nordschleife, the spares parked - 58 ms of draw calls)
                                let draw_m = crate::envc::var("EVOVIEW_AI_DRAW_M").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1500.0);
                                let mut active = vec![false; rows.len()];
                                for a in &r.ai {
                                    if a.td.pos.y > -1000.0 && a.td.pos.distance(near_eye) < draw_m {
                                        if let Some(x) = active.get_mut(a.slot - 1) {
                                            *x = true;
                                        }
                                    }
                                }
                                for (si, (b, w)) in rows.iter().enumerate() {
                                    let Some((gb, gw)) = slot_gpu.get(si + 1) else { continue };
                                    if !active[si] {
                                        for &gi in gb.iter().chain(gw.iter().flatten()) {
                                            meshes[gi].instance_count = 0;
                                        }
                                        continue;
                                    }
                                    // (the cabin meshes only near the camera: carcull.rs)
                                    for &gi in gb {
                                        carcull.put(gi, b);
                                    }
                                    for k in 0..4 {
                                        for &gi in &gw[k] {
                                            carcull.put(gi, &w[k]);
                                        }
                                    }
                                }
                            }
                            if let Some(w) = wheel.as_mut() {
                                let (hub, steer, om, kmh, ffm) = td.ffb_inputs();
                                ffb.fx = td.rig.as_ref().map_or(0.0, |(rc, _)| rc.ffb_fx.offset(&ffb_set));
                                let mut o = ffb.step(&ffb_set, hub, steer, om, kmh, ffm, td.ffb_gamma(), dt);
                                // the rack's stop: the car's lock in wheel degrees
                                if let (Some((rc, _)), Some(wd)) = (td.rig.as_ref(), c.wheel_deg) {
                                    let stop = rc.max_steer.to_degrees() * rc.steer_ratio;
                                    let x = wd / stop.max(1.0);
                                    let rate = (x - ffb.lock_prev) / dt.max(1e-4);
                                    ffb.lock_prev = x;
                                    let lock = crate::ffb::soft_lock(wd, stop, rate);
                                    o.force = (o.force + lock * if ffb_set.invert { -1.0 } else { 1.0 }).clamp(-1.0, 1.0);
                                }
                                w.send(o.force, o.damper);
                            }
                            let (mut b, mut wm) = td.matrices();
                            if drive_mode == 4 {
                                // the rig leans the hull and turns / spins the
                                // wheel bones; everything rides the flat body
                                b = td.body_flat();
                                wm = [b; 4];
                            }
                            // the tyre smoke comes off this car, in the world
                            let mut cin = crate::particles::CarIn::at(b, fxst.smoke);
                            for w in cin.wheels.iter_mut() {
                                w.2 = fxst.dust;
                            }
                            cin.throttle = fxst.throttle;
                            cin.rpm = fxst.rpm;
                            cin.wet = fxst.wet;
                            // ⚠ without these the speed-gated effects (tyre and
                            // body spray, the dust kick) read a car at rest
                            cin.vel = b.transform_vector3(Vec3::Z).normalize_or_zero() * (td.speed_kmh() / 3.6);
                            let inv = b.inverse();
                            for (k, m) in wm.iter().enumerate() {
                                cin.wheels[k].0 = inv.transform_point3(m.w_axis.truncate()).into();
                            }
                            fxg.sim.cars = vec![cin];
                            fxg.sim.cars.extend(ai_smokers);
                            let row = |m: glam::Mat4| -> Vec<f32> {
                                let mut v = m.to_cols_array().to_vec();
                                v.extend_from_slice(&[0.0f32; 4]);
                                v
                            };
                            if let Some((body, wheels)) = slot_gpu.first() {
                                let br = row(b);
                                // the driver stays hidden in the cockpit views
                                let inside = car_cams.get(cam_sel).map_or(false, |c| c.mode == 1 || c.mode == 4)
                                    && follow != crate::camera::Follow::Free;
                                let zero = vec![0.0f32; 20];
                                for &gi in body {
                                    let hide = (inside || !driver_on) && driver_meshes.contains(&gi);
                                    queue.write_buffer(&meshes[gi].instances, 0,
                                        bytemuck::cast_slice(if hide { &zero } else { &br }));
                                }
                                for k in 0..4 {
                                    let wr = row(wm[k]);
                                    for &gi in &wheels[k] {
                                        queue.write_buffer(&meshes[gi].instances, 0, bytemuck::cast_slice(&wr));
                                    }
                                }
                            }
                            title_at += dt;
                            if title_at > 0.2 {
                                title_at = 0.0;
                                let g = match td.gear { 0 => "N".to_string(), g if g < 0 => "R".to_string(), g => g.to_string() };
                                let warn = td.rig.as_ref().map_or(String::new(), |(rc, _)| {
                                    if !rc.running { " · STALLED (End / starter)".into() } else if rc.missed { " · gear not in - clutch!".into() } else { String::new() }
                                });
                                window.set_title(&format!("{} · {:.0} km/h · gear {g} · {:.0} rpm · {}{warn}",
                                    opts.title, td.speed_kmh(), td.rpm, follow.name()));
                            }
                        }

                        // Drive each animation toward open or closed, and let
                        // , and . scrub the selected one by hand.
                        if let Some(skel) = skeleton.as_ref() {
                            let scrub = if held.contains(&KeyCode::Comma) {
                                -1.0
                            } else if held.contains(&KeyCode::Period) {
                                1.0
                            } else {
                                0.0
                            };
                            for (i, st) in anim_state.iter_mut().enumerate() {
                                if scrub != 0.0 && i == selected {
                                    st.t = (st.t + scrub * dt).clamp(0.0, st.len);
                                    st.open = st.t > st.len * 0.5;
                                    continue;
                                }
                                let target = if st.open { st.len } else { 0.0 };
                                let step = dt.min(0.05);
                                if st.t < target {
                                    st.t = (st.t + step).min(target);
                                } else if st.t > target {
                                    st.t = (st.t - step).max(target);
                                }
                            }
                            let playing: Vec<(&crate::load::Animation, f32)> = anim_state
                                .iter()
                                .enumerate()
                                .filter(|(_, st)| st.t > 0.0001)
                                .map(|(i, st)| (&animations[i], st.t))
                                .collect();
                            let driven = match (drive.as_mut(), skel.rig.as_deref()) {
                                (Some(d), Some(rig)) if drive_mode == 4 => {
                                    if let Some(td) = opts.track_drive.as_ref() {
                                        // headlights come on at dusk and go off
                                        // at dawn; H still overrides in between
                                        let night = crate::sky::now().night > 0.5;
                                        if night != was_night {
                                            was_night = night;
                                            fxst.lights = if night { 2 } else { 0 };
                                            materials.light_state = fxst.lights;
                                        }
                                        fxst.night = night;
                                        d.cs.lights = fxst.lights.min(3) as u8;
                                        let hb = held.contains(&KeyCode::ControlLeft) || held.contains(&KeyCode::ControlRight);
                                        d.follow(rig, &td.rig_feed(), hb, dt);
                                        if crate::envc::var_os("EVOVIEW_ATTITUDE_LOG").is_some() {
                                            let (p, r) = d.attitude_deg();
                                            eprintln!("attitude pitch {p:6.2} roll {r:6.2} a_long {:6.2} kmh {:5.1}", td.rig_feed().a_long, td.speed_kmh());
                                        }
                                    }
                                    let inp = d.inputs(rig);
                                    Some(rig.pose(&inp))
                                }
                                (Some(d), Some(rig)) if drive_mode > 0 => {
                                    if drive_mode == 3 && feed.is_some() {
                                        let f = feed.as_mut().unwrap();
                                        let span = f.length().max(1.0);
                                        demo_t = (demo_t + dt) % span;
                                        f.apply(d, rig, demo_t, dt);
                                    } else if drive_mode == 2 {
                                        crate::drive::demo_run(d, rig, demo_t, demo_t + dt);
                                        demo_t += dt;
                                    } else {
                                        d.cs.lights = fxst.lights.min(3) as u8;
                                        d.cs.indicator = match fxst.indicator {
                                            crate::fxstate::Indicator::Off => 0,
                                            crate::fxstate::Indicator::Left => -1,
                                            crate::fxstate::Indicator::Right => 1,
                                            crate::fxstate::Indicator::Hazard => 2,
                                        };
                                        let c = crate::drive::Controls {
                                            throttle: held.contains(&KeyCode::ArrowUp) || held.contains(&KeyCode::KeyI),
                                            brake: held.contains(&KeyCode::ArrowDown) || held.contains(&KeyCode::KeyK),
                                            left: held.contains(&KeyCode::ArrowLeft) || held.contains(&KeyCode::KeyJ),
                                            right: held.contains(&KeyCode::ArrowRight) || held.contains(&KeyCode::KeyL),
                                            handbrake: held.contains(&KeyCode::Space),
                                            ..crate::drive::Controls::none()
                                        };
                                        d.step(rig, &c, dt);
                                    }
                                    title_at += dt;
                                    if title_at > 0.25 {
                                        title_at = 0.0;
                                        window.set_title(&format!(
                                            "{} · {:.0} km/h · gear {} · {:.0} rpm{}",
                                            opts.title, d.cs.speed_mps * 3.6,
                                            match d.cs.gear { -1 => "R".to_string(), 0 => "N".to_string(), g => g.to_string() },
                                            d.cs.rpm, match drive_mode { 2 => " · demo", 3 => " · replay", _ => "" }));
                                    }
                                    let inp = d.inputs(rig);
                                    Some(rig.pose(&inp))
                                }
                                _ => None,
                            };
                            let mut pose = driven.unwrap_or_else(|| skel.pose_many(&playing));
                            pose.resize(MAX_BONES, identity);
                            queue.write_buffer(&bone_buf, 0, bytemuck::cast_slice(&pose));
                            if let Some(mr) = mirrors.as_mut() {
                                mr.set_pose(&pose);
                            }
                            if let Some(r) = rain_fx.as_mut() {
                                r.update(&pose, dt, opts.track_drive.as_ref().map_or(0.0, |td| td.speed_kmh()));
                            }
                            // a door opening moves its shadow too. Driving changes
                            // the pose EVERY frame (spinning wheels, needles); the
                            // hero map is then redrawn every 4th frame only - a
                            // shadow 3 frames late is invisible, a full shadow
                            // pass per frame is most of drive mode's GPU cost.
                            shadow_tick = shadow_tick.wrapping_add(1);
                            if pose != last_pose && (drive_mode == 0 || shadow_tick % 4 == 0) {
                                shadow_invalid = true;
                                last_pose = pose;
                            }
                        }

                        // Bench: a fixed orbit (cars) or a turn on the spot
                        // (tracks), the same every run.
                        if let Some(b) = bench.as_ref().filter(|_| crate::envc::var_os("EVOVIEW_BENCH_STILL").is_none()) {
                            cam.yaw = b.yaw(bench_yaw0.0);
                            if let Some(f) = fly.as_mut() {
                                f.yaw = b.yaw(bench_yaw0.1);
                            }
                        }
                        // EVOVIEW_FLY_MOVE=m: a bench free camera dollies m metres
                        // a frame along its view (LOD transition captures)
                        if let (Some(_), Some(f), Some(m)) = (bench.as_ref(), fly.as_mut(),
                            crate::envc::var("EVOVIEW_FLY_MOVE").ok().and_then(|v| v.parse::<f32>().ok())) {
                            f.pos += f.forward() * m;
                        }
                        let (mut view_proj, mut eye) = match fly.as_ref() {
                            Some(f) => (f.view_proj(aspect), f.pos),
                            None => (cam.view_proj(aspect), cam.eye()),
                        };
                        // the cockpit views hide the driver's body (whichever way
                        // the camera got there: C, EVOVIEW_CAM, the start)
                        if opts.track_drive.is_some() && Some(cam_sel) != cam_applied {
                            cam_applied = Some(cam_sel);
                            let inside = car_cams.get(cam_sel).map_or(false, |c| c.mode == 1 || c.mode == 4);
                            let m = if driver_on && !inside { pack_instances(&[], 0) } else { vec![0.0f32; 20] };
                            for &gi in &driver_meshes {
                                queue.write_buffer(&meshes[gi].instances, 0, bytemuck::cast_slice(&m));
                            }
                        }
                        if let Some(td) = opts.track_drive.as_ref() {
                            // the in-car camera's head and body filter step with
                            // the physics (once a frame, on the body as drawn)
                            let drawn = match (drive_mode, drive.as_ref()) {
                                (4, Some(d)) => td.body_flat() * d.body(),
                                _ => td.matrices().0,
                            };
                            if let Some(h) = headcam.as_mut() {
                                let m = car_cams.get(cam_sel).map_or(Default::default(), |c| h.s.mode(c.mode, c.drivable));
                                h.road_up = td.road_up();
                                h.step(dt, drawn, td.ffb_inputs().1.to_degrees(), m);
                            }
                            DASH_LIVE.store(follow != crate::camera::Follow::Free
                                && car_cams.get(cam_sel).map_or(false, |c| c.mode == 1 || c.mode == 4),
                                std::sync::atomic::Ordering::Relaxed);
                            if follow != crate::camera::Follow::Free {
                                let p = td.pose();
                                INTERIOR.store(car_cams.get(cam_sel).map_or(false, |c| {
                                    let n = c.name.to_ascii_lowercase();
                                    n == "cockpit" || n == "dash"
                                }), std::sync::atomic::Ordering::Relaxed);
                                // a chase view on the viewed opponent (online: another
                                // player) instead
                                let other = view_ai.and_then(|k| opts.race.as_ref().and_then(|r| r.ai.get(k)).filter(|a| a.td.pos.y > -1000.0).map(|a| a.td.pose()));
                                if let Some(p) = other {
                                    let (vp, e) = follow.view_proj(
                                        aspect, p.pos + p.up * 0.45, p.forward, p.up, p.right,
                                        last_frame.elapsed().as_secs_f32());
                                    view_proj = vp;
                                    eye = e;
                                } else if cam_sel < car_cams.len() {
                                    // the car's own camera: rigid on the level body
                                    // (EVOVIEW_CAM_HEAD=0), else the game's head
                                    // movement and horizon lock (headcam.rs)
                                    let (vp, e) = match headcam.as_mut().filter(|h| h.s.on) {
                                        Some(h) => h.view(&car_cams[cam_sel], drawn, td.body_flat(), aspect, td.speed_kmh()),
                                        None => crate::cams::car_view(&car_cams[cam_sel], td.body_flat(), aspect),
                                    };
                                    view_proj = vp;
                                    eye = e;
                                } else if let (true, Some(d)) = (cam_sel == car_cams.len(), tv.as_mut()) {
                                    if let Some((vp, e)) = d.view(p.pos + p.up * 0.6, aspect, dt) {
                                        view_proj = vp;
                                        eye = e;
                                    }
                                } else {
                                    // the replay cameras expect the centre of mass
                                    let (vp, e) = follow.view_proj(
                                        aspect, p.pos + p.up * 0.45, p.forward, p.up, p.right,
                                        last_frame.elapsed().as_secs_f32());
                                    view_proj = vp;
                                    eye = e;
                                }
                            }
                            // EVOVIEW_LOOK_BACK=1: a chase camera turned round (the
                            // pack behind, AI headlights in captures)
                            if crate::envc::var_os("EVOVIEW_LOOK_BACK").is_some() {
                                let p = td.pose();
                                let (vp, e) = crate::camera::Follow::Chase.view_proj(
                                    aspect, p.pos + p.up * 0.45, -p.forward, p.up, -p.right, 0.0);
                                view_proj = vp;
                                eye = e;
                            }
                            if let Some(l) = vlog.as_mut() {
                                l.log(dt, &cam_name(cam_sel), td.speed_kmh(), drawn, view_proj, eye);
                            }
                        }
                        // the driven car's sound, from this frame's physics and camera
                        if let (Some(s), Some(td)) = (car_snd.as_mut(), opts.track_drive.as_ref()) {
                            let hb = held.contains(&KeyCode::ControlLeft) || held.contains(&KeyCode::ControlRight);
                            let mut f = td.sound_feed(hb);
                            let p = td.pose();
                            let inv = view_proj.inverse();
                            let a = inv * glam::Vec4::new(0.0, 0.0, 0.5, 1.0);
                            let cam_fwd = (a.truncate() / a.w - eye).normalize_or_zero();
                            let b = inv * glam::Vec4::new(0.0, 1.0, 0.5, 1.0);
                            let cam_up = ((b.truncate() / b.w - eye) - cam_fwd * (b.truncate() / b.w - eye).dot(cam_fwd))
                                .normalize_or_zero();
                            f.car_pos = (p.pos + p.up * 0.5).to_array();
                            f.car_vel = (p.forward * td.vx).to_array();
                            f.car_fwd = p.forward.to_array();
                            f.car_up = p.up.to_array();
                            f.cam_pos = eye.to_array();
                            f.cam_fwd = cam_fwd.to_array();
                            f.cam_up = if cam_up == glam::Vec3::ZERO { [0.0, 1.0, 0.0] } else { cam_up.to_array() };
                            f.horn = held.contains(&KeyCode::KeyK);
                            f.onboard = follow == crate::camera::Follow::Onboard
                                || car_cams.get(cam_sel).map_or(false, |c| c.mode == 1 || c.mode == 4);
                            // the opponents' engines, the nearest few
                            if let Some(r) = opts.race.as_ref() {
                                s.update_ai(&r.sound_cars(), f.cam_pos, dt);
                            }
                            s.update(&f, dt);
                        }
                        // A follow camera overrides the fly camera rather than
                        // replacing it, so C can hand control back without the
                        // free view having moved in the meantime.
                        if let Some(pb) = opts.playback.as_ref() {
                            DASH_LIVE.store(follow == crate::camera::Follow::Onboard, std::sync::atomic::Ordering::Relaxed);
                            if follow != crate::camera::Follow::Free {
                                if let Some(&(ci, _)) = drivers.get(watched) {
                                    if let Some(p) = pb.replay.sample(ci, play_t) {
                                        // onboard: the car's own cockpit camera, where
                                        // the model has one (the wheel, the hands,
                                        // the dash), else the generic eye point
                                        let slot = pb.slots.iter().find(|s| s.cars.contains(&ci));
                                        let cockpit = slot.and_then(|s| s.cockpit.as_ref().map(|c| (c, s.offset)))
                                            .filter(|_| follow == crate::camera::Follow::Onboard);
                                        INTERIOR.store(cockpit.is_some(), std::sync::atomic::Ordering::Relaxed);
                                        let (vp, e) = match cockpit {
                                            Some((c, off)) => crate::cams::car_view(c, crate::replayscene::car_matrix(&p, off), aspect),
                                            None => follow.view_proj(
                                                aspect, p.pos, p.forward, p.up, p.right,
                                                last_frame.elapsed().as_secs_f32() + play_t),
                                        };
                                        view_proj = vp;
                                        eye = e;
                                    }
                                }
                            }
                            // Cheap HUD: evoview has no text renderer, so the
                            // playhead, car and camera go in the window title.
                            // Throttled - retitling every frame is wasted work.
                            title_at += dt;
                            if title_at > 0.2 {
                                title_at = 0.0;
                                let who = drivers.get(watched)
                                    .map(|d| format!("car {} {}", d.0, d.1))
                                    .unwrap_or_else(|| "no car".into());
                                window.set_title(&format!(
                                    "{} · {:.1}/{:.0}s · {} · {} · {}x{}",
                                    opts.title, play_t, pb.replay.captured(), who,
                                    follow.name(), play_speed,
                                    if playing { "" } else { " · paused" }));
                            }
                        }
                        if let Some(c) = culler.as_mut() {
                            c.update(&queue, &meshes, view_proj, eye, rh);
                        }
                        carcull.flush(&device, &queue, &mut meshes, motion.as_mut(), view_proj, eye,
                                      crate::sky::key_dir(), car_detail_m);
                        // the replay rigs of the cars that can show, posed into
                        // their bone blocks and uploaded in one write
                        if let (Some(pb), false) = (opts.playback.as_ref(), pose_jobs.is_empty()) {
                            let _t = crate::bench::cpu("replay pose");
                            let (mut lo, mut hi) = (usize::MAX, 0usize);
                            let mut posed = vec![false; posed_last.len().max(pb.replay.cars.len())];
                            // each slot's reach: at least carcull's own radius for
                            // any of its meshes, so a drawn car is always posed
                            if slot_reach.len() != slot_gpu.len() {
                                slot_reach = slot_gpu.iter().map(|(b, w)| {
                                    b.iter().chain(w.iter().flatten())
                                        .map(|&gi| meshes.get(gi).map_or(0.0, |m| m.radius))
                                        .fold(0.0f32, f32::max) + crate::carcull::REACH_MARGIN
                                }).map(|r| r.max(crate::carcull::MIN_REACH)).collect();
                            }
                            for &(ci, si, c) in &pose_jobs {
                                if !carcull.visible(c, slot_reach.get(si).copied().unwrap_or(f32::MAX)) {
                                    continue;
                                }
                                let Some(rig) = pb.slots.get(si).and_then(|s| s.skel.as_ref()).and_then(|k| k.rig.as_deref()) else { continue };
                                let Some((d, _)) = rep_rigs.get(&ci) else { continue };
                                let mut pose = rig.pose(&d.inputs(rig));
                                pose.resize(MAX_BONES, identity);
                                let blk = 1 + ci;
                                if rep_bones.len() < (blk + 1) * MAX_BONES {
                                    rep_bones.resize((blk + 1) * MAX_BONES, identity);
                                }
                                rep_bones[blk * MAX_BONES..(blk + 1) * MAX_BONES].copy_from_slice(&pose);
                                lo = lo.min(blk);
                                hi = hi.max(blk);
                                if let Some(p) = posed.get_mut(ci) {
                                    *p = true;
                                }
                                // first frame posed again: no history
                                if !posed_last.get(ci).copied().unwrap_or(false) {
                                    if let Some(mo) = motion.as_ref() {
                                        mo.set_prev_bones(&queue, blk, &pose);
                                    }
                                }
                            }
                            posed_last = posed;
                            // blocks between two posed ones keep what they held
                            let fits = (bone_buf.size() as usize / 64).min(rep_bones.len());
                            if lo <= hi {
                                let (a, b) = (lo * MAX_BONES, ((hi + 1) * MAX_BONES).min(fits));
                                if a < b {
                                    queue.write_buffer(&bone_buf, (a * 64) as u64, bytemuck::cast_slice(&rep_bones[a..b]));
                                }
                            }
                        }
                        // the drawn projection carries TAA's sub-pixel jitter;
                        // culling and the resolve use the clean one
                        let taa_on = taa.as_ref().map_or(false, |t| t.on);
                        // no history = no jitter either (each frame stands alone;
                        // supersampled captures of replays, whose cars TAA cannot follow)
                        let vp_draw = taa.as_ref().filter(|_| !taa_hist_off).map_or(view_proj, |t| t.jittered(view_proj));
                        // A camera cut (C, a TV camera switching, a teleport,
                        // a replay seek): the eye jumps or the view turns
                        // further than any real frame does. Every history
                        // restarts (TAA, motion vectors, SSR, DLSS/FSR, motion
                        // blur); kept, the old view ghosted ~20 frames.
                        let fwd = {
                            let f = view_proj.inverse() * glam::Vec4::new(0.0, 0.0, 0.5, 1.0);
                            (f.truncate() / f.w - eye).normalize_or_zero()
                        };
                        let cut = cut_prev.map_or(false, |(pe, pf): (Vec3, Vec3)| {
                            pe.distance(eye) > 12.0 || pf.dot(fwd) < 0.9
                        });
                        cut_prev = Some((eye, fwd));
                        if cut {
                            if let Some(p) = post.as_mut() {
                                p.cut();
                            }
                        }
                        let hist_reset = taa_hist_off || cut;
                        // The shadow map follows the camera target, so both
                        // buffers are rewritten every frame. On a track that
                        // is the view itself (view_focus), not the orbit
                        // camera nobody is using.
                        let mut focus = if opts.start_fly { view_focus(view_proj, eye) } else { cam.target };
                        // driving (the cached cascade): only 60 m out, so turning
                        // moves it a little and the map still reaches ~260 m ahead
                        if cached {
                            focus = eye + (focus - eye) * 0.5;
                        }
                        let focus_mid = if opts.start_fly { view_focus_mid(view_proj, eye) } else { cam.target };
                        let mut cam_u = if hero_on {
                            uniforms_car_shadowed(
                                uniforms_lit(vp_draw, eye, lo, hi, opts.start_fly, lit), lo, hi)
                        } else if shadows_on {
                            uniforms_shadowed(vp_draw, eye, lo, hi,
                                              opts.start_fly, lit, focus)
                        } else {
                            uniforms_lit(vp_draw, eye, lo, hi, opts.start_fly, lit)
                        };
                        // ⚠ the cached cascade is DRAWN on the cell's matrix: sampled
                        // through the unsnapped one every shadow sat up to 4 m off
                        if shadows_on && !hero_on && cached {
                            cam_u.light_vp = light_matrix_cell(focus).to_cols_array_2d();
                        }
                        if shadows_on && !hero_on {
                            cam_u = uniforms_shadowed_mid(cam_u, focus_mid);
                        }
                        // the driven car's hero map round the body as drawn
                        // (the same box the TAA object test uses)
                        let hero_vp = opts.track_drive.as_ref().filter(|_| dhero).map(|td| {
                            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                            for v in td.slot.centres {
                                lo = lo.min(v);
                                hi = hi.max(v);
                            }
                            let r = td.slot.radius.max(0.2);
                            let lo = Vec3::new(lo.x - 0.3, lo.y - r, lo.z - 1.1);
                            let hi = Vec3::new(hi.x + 0.3, hi.y - r + 1.5, hi.z + 1.1);
                            let b = match (drive_mode, drive.as_ref()) {
                                (4, Some(d)) => td.body_flat() * d.body(),
                                _ => td.matrices().0,
                            };
                            let c = b.transform_point3((lo + hi) * 0.5);
                            let (m, texel) = hero_light_matrix(c, (hi - lo).length() * 0.5);
                            cam_u.light_vp_hero = m.to_cols_array_2d();
                            // y: ~1 cm of depth bias in its ~70 m box
                            cam_u.shadow_hero = [hero_dim() as f32 / shadow_dim as f32, 0.00015, texel,
                                                 2.0 * texel];
                            m
                        });
                        if crate::envc::var_os("EVOVIEW_FX_SUNCHECK").is_some() {
                            let m = cam_u.light_vp;
                            let l = -Vec3::new(m[0][2], m[1][2], m[2][2]).normalize_or_zero();
                            eprintln!("SUNCHECK fx sun {l:?} . sky key {:?} = {:.3}", crate::sky::key_dir(), l.dot(crate::sky::key_dir()));
                        }
                        let smoke_was = fxst.smoke;
                        if let (Some(d), true) = (drive.as_ref(), drive_mode > 0) {
                            fxst.follow_car(&d.cs, dt);
                        }
                        // the rig car's driver aids and tyres on the dash
                        if let Some((rc, _)) = opts.track_drive.as_ref().and_then(|td| td.rig.as_ref()) {
                            fxst.tc_level = Some(match rc.tc.as_ref() { Some(t) if rc.tc_on => t.level as f32, _ => 0.0 });
                            fxst.abs_level = Some(match rc.abs.as_ref() { Some(a) if rc.abs_on => a.level as f32, _ => 0.0 });
                            fxst.brake_bias = Some(rc.brake_bias());
                            fxst.tyre_bar = rc.tyre_psi().map(|p| p.map(|x| x / 14.5038));
                        }
                        fxst.pit_limiter = opts.pits.as_ref().map_or(false, |p| p.limiter);
                        fxst.lap = opts.laps.as_ref().map(|l| l.dash());
                        // the rig knows its tyres: rear smoke from how fast the
                        // rear contact patches really slide (spin and drift
                        // angle), at any speed, not the launch / handbrake guess
                        if let Some((rc, _)) = opts.track_drive.as_ref().and_then(|td| td.rig.as_ref()) {
                            let v = rc.v.length();
                            let slide = rc.c[2..].iter().filter(|c| c.on_ground)
                                .map(|c| v * (c.sr * c.sr + c.alpha.tan().powi(2)).sqrt())
                                .fold(0.0f32, f32::max);
                            let want = ((slide - 2.0) / 5.0).clamp(0.0, 1.0);
                            fxst.smoke = smoke_was + (want - smoke_was) * (dt * 6.0).min(1.0);
                        }
                        cam_u.fx = crate::lights::fx_block(car_fx.as_deref(), &fxst, opts.start_fly);
                        if opts.race.is_some() || opts.playback.is_some() {
                            // the AI's (and replay cars') lamps: lit at night and in the rain (the
                            // rain light too), EVOVIEW_AI_LIGHTS=0..3 forces a stage
                            let night = fxst.night || crate::sky::now().night > 0.5;
                            let wet = fxst.rain > 0.0;
                            let st = crate::fxstate::FxState {
                                lights: crate::envc::var("EVOVIEW_AI_LIGHTS").ok().and_then(|v| v.parse::<u32>().ok())
                                    .unwrap_or(if night || wet { 2 } else { 0 }).min(3),
                                fog: wet,
                                night,
                                ..fxst
                            };
                            cam_u.ai_fx = crate::lights::ai_fx_block(car_fx.as_deref(), &st);
                        }
                        if opts.start_fly {
                            // smoke outdoors: lit by sun + sky, in the scene's units
                            let b = crate::sky::block();
                            let k = crate::envc::var("EVOVIEW_SMOKE_LIGHT").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
                            // under rain the sun is behind the deck (sky::block
                            // dims it) but a translucent cloud of spray or smoke
                            // is lit by the bright overcast all round: undo the dimming
                            let sun = b[0][3].max(0.0) / (1.0 - 0.85 * crate::rain::rain()).max(0.15);
                            cam_u.fx[63][2] = (sun.min(1.0) * 0.6 + 0.4) * 9.0 * k / crate::sky::exposure().max(1e-3);
                            // the car's lamps, like the track's (strict SKY3):
                            // divided back by most of the night exposure, or the
                            // tail lights bloom into a red blob the size of the car
                            if crate::sky::active() {
                                let a = crate::envc::var("EVOVIEW_EMISSIVE_ADAPT").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.8);
                                cam_u.fx[63][3] = crate::sky::exposure().max(1e-3).powf(-a);
                            }
                        }
                        // the weather everything reads this frame (rain.rs); the
                        // track's water builds and dries over time, and the
                        // shading follows it unless EVOVIEW_WET pins it
                        crate::rain::set_weather(fxst.rain, fxst.wet);
                        crate::rain::step_water(dt);
                        if !wet_pinned {
                            fxst.wet = crate::rain::shown_wetness();
                            crate::rain::set_weather(fxst.rain, fxst.wet);
                        }
                        // rain on the glass: the wipers' sweep and the drop phase
                        if let Some(r) = rain_fx.as_ref() {
                            let kmh = opts.track_drive.as_ref().map_or(0.0, |td| td.speed_kmh());
                            let fwd = opts.track_drive.as_ref().map_or(Vec3::Z, |td| td.matrices().0.transform_vector3(Vec3::Z).normalize_or_zero());
                            // refraction needs the scene in a texture: the TAA target
                            let copy = match (scene_copy.as_ref().filter(|c| !no_refract && copy_fits(c, rw, rh)), taa.as_ref().map_or(false, |t| t.on)) {
                                (Some(c), true) => [1.0 / c.size.0 as f32, 1.0 / c.size.1 as f32],
                                _ => [0.0, 0.0],
                            };
                            cam_u.rain = r.block(fxst.rain, kmh, fwd, copy);
                        }
                        if opts.start_fly {
                            cam_u.clouds = clouds.block();
                            // the probe's last mip, once all six faces exist
                            if let Some(p) = probe.as_ref().filter(|p| p.ready()) {
                                cam_u.clouds[5][2] = (p.mips - 1) as f32;
                            }
                        }
                        // w: frame number mod 16, the car cutouts' dither shift
                        // (shader.wgsl cutout_drop; ditherTemporalOffset)
                        {
                            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                            cam_u.clouds[5][3] = (N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 16) as f32;
                        }
                        let fx_ambient = cam_u.fx[0][0];
                        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&cam_u));
                        // the particles drawn after TAA: no jitter (cam_u's
                        // view_proj is the drawn, jittered one)
                        {
                            let mut lu = cam_u;
                            lu.view_proj = view_proj.to_cols_array_2d();
                            fxg.set_late_camera(&queue, bytemuck::bytes_of(&lu));
                        }
                        if let Some(t) = track.as_ref() {
                            {
                                let _t = crate::bench::cpu("track update");
                                t.update(&queue, vp_draw, eye, lit, &cam_u);
                            }
                            let _t = crate::bench::cpu("track anim");
                            t.animate(&queue, crate::clouds::time());
                        }
                        // The sun and the scenery are static, so the shadow map
                        // only needs redrawing when its box moves - or every
                        // frame during a replay, where the cars cast.
                        let light_vp = if hero_on {
                            car_light_matrix(lo, hi).0
                        } else if cached {
                            // the cache is redrawn when this cell changes
                            light_matrix_cell(focus)
                        } else {
                            light_matrix(focus)
                        };
                        // ⚠ not on shadow_invalid: that is the CAR changing (its
                        // pose every 4th frame while driving, the driver, a
                        // livery), and the car is drawn over the cache every
                        // frame anyway - it redrew the scenery (~3 ms) every
                        // 4th frame. The matrix carries the cell and the sun.
                        let cache_dirty = cached && shadow_cache_vp != Some(light_vp);
                        // with the hero map the near layer IS the cache: only
                        // a new cell redraws it, the car changing does not
                        let every = crate::envc::var_os("EVOVIEW_SHADOW_EVERY_FRAME").is_some();
                        let shadow_dirty = shadows_on && if dhero { cache_dirty || every } else {
                            shadow_invalid || shadow_drawn != Some(light_vp) || shadow_cache.is_some()
                                || opts.playback.is_some() || every
                        };
                        // The far cascade is snapped to a coarse grid, so it
                        // redraws only when the target has moved far (or the
                        // sun has: its matrix carries the direction).
                        let far_vp = light_matrix_far(focus);
                        // driving, the far map is scenery only (the car is always
                        // inside the near one): the car changing is not a redraw
                        let far_dirty = far_on
                            && ((shadow_invalid && !cached) || far_drawn != Some(far_vp)
                                || crate::envc::var_os("EVOVIEW_SHADOW_EVERY_FRAME").is_some());
                        if shadow_dirty {
                            let mut lu = cam_u;
                            lu.view_proj = light_vp.to_cols_array_2d();
                            queue.write_buffer(&light_buf, 0, bytemuck::bytes_of(&lu));
                            // with the cache only its redraw needs the culled
                            // set; the car on top is drawn whole
                            if let Some(c) = culler.as_mut().filter(|_| cache_dirty || !cached) {
                                // a caster under ~2 px on screen (EVOVIEW_SHADOW_PX)
                                // casts nothing visible
                                let px = crate::envc::var("EVOVIEW_SHADOW_PX").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(2.0);
                                let focal = view_proj.row(1).truncate().length() * rh as f32 * 0.5;
                                let tiny = (px > 0.0 && focal > 0.0).then(|| (eye, px / focal));
                                c.update_shadow(&device, &queue, 0, light_vp, focus, 0.0, tiny);
                            }
                            shadow_drawn = Some(light_vp);
                        }
                        if far_dirty {
                            let mut lu = cam_u;
                            lu.view_proj = far_vp.to_cols_array_2d();
                            queue.write_buffer(&pipes_all.light_buf_far, 0, bytemuck::bytes_of(&lu));
                            if let Some(c) = culler.as_mut() {
                                // a far texel is ~1 m: casters under ~1.5 m
                                // across cannot show and are most of the count
                                c.update_shadow(&device, &queue, 1, far_vp, focus, 0.75, None);
                            }
                            far_drawn = Some(far_vp);
                        }
                        // the middle cascade: snapped like the far one, scenery
                        // only while driving
                        let mid_vp = light_matrix_mid(focus_mid);
                        let mid_dirty = mid_on
                            && ((shadow_invalid && !cached) || mid_drawn != Some(mid_vp) || every);
                        if mid_dirty {
                            let mut lu = cam_u;
                            lu.view_proj = mid_vp.to_cols_array_2d();
                            queue.write_buffer(&pipes_all.light_x[0].0, 0, bytemuck::bytes_of(&lu));
                            if let Some(c) = culler.as_mut() {
                                // a mid texel is ~0.3 m, and its casters are far:
                                // drop those under EVOVIEW_SHADOW_MID_PX on screen
                                // (each redraw is a frame's spike)
                                let px = crate::envc::var("EVOVIEW_SHADOW_MID_PX").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(4.0);
                                let focal = view_proj.row(1).truncate().length() * rh as f32 * 0.5;
                                let tiny = (px > 0.0 && focal > 0.0).then(|| (eye, px / focal));
                                c.update_shadow(&device, &queue, 5, mid_vp, focus_mid, 0.3, tiny);
                            }
                            mid_drawn = Some(mid_vp);
                        }
                        // the car's hero map: every frame, it is the car alone
                        if let Some(m) = hero_vp {
                            let mut lu = cam_u;
                            lu.view_proj = m.to_cols_array_2d();
                            queue.write_buffer(&pipes_all.light_x[1].0, 0, bytemuck::bytes_of(&lu));
                        }
                        if shadow_dirty || far_dirty || mid_dirty {
                            shadow_invalid = false;
                        }
                        // Build the overlay before the encoder exists, so the
                        // vertex upload is finished before the pass borrows it.
                        hud.begin(config.width, config.height);
                        // EVOVIEW_NO_HUD=1: clean frames (captures, GIFs)
                        if let Some(pb) = opts.playback.as_ref().filter(|_| crate::envc::var_os("EVOVIEW_NO_HUD").is_none()) {
                            let cur = last_pos.map(|p| {
                                ((p.0 as f32) / config.width as f32,
                                 (p.1 as f32) / config.height as f32)
                            });
                            // EVOVIEW_HUD_CURSOR=x,y (0..1): a pretend cursor, to
                            // capture hover states in bench shots
                            let cur = cur.or_else(|| crate::envc::var("EVOVIEW_HUD_CURSOR").ok().and_then(|v| {
                                let (x, y) = v.split_once(',')?;
                                Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
                            }));
                            if let Some(r) = rui.as_mut() {
                                r.draw(&mut hud, &mut rcluster, &pb.replay, &drivers, watched, play_t, playing,
                                       play_speed, follow.name(), cur, aspect, dt);
                            }
                            if let Some((ci, model)) = drivers.get(watched).cloned() {
                                // Swap the engine bank when the watched car
                                // changes; the read happens off-thread.
                                if let (Some(e), Some(pkg)) =
                                    (engine.as_mut(), opts.pkg.as_ref()) {
                                    e.load_car(pkg, &model);
                                    e.poll();
                                    e.set_inside(
                                        follow == crate::camera::Follow::Onboard);
                                }
                                if let Some(t) = pb.replay.telemetry(ci, play_t) {
                                    // Feed the engine note. Quieter from
                                    // outside the car than from the cockpit,
                                    // and silent while paused.
                                    if let Some(e) = engine.as_ref() {
                                        let gain = if !playing { 0.0 }
                                            else if follow == crate::camera::Follow::Onboard { 1.0 }
                                            else { 0.6 };
                                        // The event wants road speed and gear,
                                        // not rpm. Limiter is derived: the bank
                                        // has no rpm input to notice it itself.
                                        let kmh = pb.replay.speed(ci, play_t) * 3.6;
                                        let lim = if t.rpm > 7600.0 { 1.0 } else { 0.0 };
                                        e.update(kmh, t.gear, lim, gain);
                                    }
                                }
                            }
                        }
                        if let Some(td) = opts.track_drive.as_ref().filter(|_| crate::envc::var_os("EVOVIEW_NO_HUD").is_none()) {
                            let rig = td.rig.as_ref().map(|(rc, _)| rc);
                            // watching another car: its own speed, rpm, gear and pedals
                            let viewed = view_ai.and_then(|k| opts.race.as_ref().and_then(|r| r.ai.get(k))).filter(|a| a.td.pos.y > -1000.0);
                            let r = if let Some(a) = viewed {
                                let f = a.td.rig_feed();
                                crate::cluster::Reading {
                                    kmh: a.td.speed_kmh(),
                                    rpm: a.td.rpm,
                                    limiter: a.td.rpm_limit(),
                                    gear: a.td.gear,
                                    throttle: f.throttle,
                                    brake: f.brake,
                                    clutch: 0.0,
                                    handbrake: 0.0,
                                    stalled: false,
                                    on_limiter: false,
                                    engine: true,
                                    aux: false,
                                }
                            } else { crate::cluster::Reading {
                                kmh: td.speed_kmh(),
                                rpm: rig.map_or(td.rpm, |rc| rc.rpm),
                                limiter: td.rpm_limit(),
                                gear: rig.map_or(td.gear, |rc| rc.gear),
                                throttle: pedals.0,
                                brake: pedals.1,
                                clutch: pedals.2.unwrap_or_else(|| rig.map_or(0.0, |rc| 1.0 - rc.clutch_e)),
                                handbrake: rig.map_or(pedals.3, |rc| rc.hb.max(pedals.3)),
                                stalled: rig.map_or(false, |rc| !rc.running),
                                on_limiter: rig.map_or(false, |rc| rc.limiter_hit),
                                engine: true,
                                aux: true,
                            } };
                            cluster.draw(&mut hud, &r, aspect);
                            // online: the server's chat, bottom left - each line for 20 s,
                            // all of them while typing
                            if let Some(m) = opts.mp.as_ref() {
                                use crate::hud::{theme as t, Weight};
                                let typing = chat_input.is_some();
                                let lines: Vec<_> = m.chat().into_iter().filter(|c| typing || c.0 < 20.0).collect();
                                let sz = 0.022;
                                let lh = sz * 1.45;
                                let mut y = 0.80 - lh * lines.len() as f32;
                                for (age, who, msg, kind) in &lines {
                                    let a = if typing { 1.0 } else { (20.0 - age).clamp(0.0, 1.0) };
                                    let mut x = 0.02;
                                    if !who.is_empty() {
                                        let c = if *kind == 3 || *kind == 4 { t::gold(a) } else { t::accent(a) };
                                        x += hud.label(&format!("{who}: "), x, y, sz, Weight::Semibold, c);
                                    } else if *kind == 4 {
                                        x += hud.label("server: ", x, y, sz, Weight::Semibold, t::gold(a));
                                    }
                                    hud.label(msg, x, y, sz, Weight::Regular, t::fg(a));
                                    y += lh;
                                }
                                if let Some(line) = chat_input.as_ref() {
                                    let s = format!("say: {line}_");
                                    let w = hud.label_w(&s, sz, Weight::Regular).max(0.25);
                                    hud.panel(0.012, 0.805, w + 0.016, lh + 0.008, 6.0, t::panel(0.7), t::line(0.6));
                                    hud.label(&s, 0.02, 0.81, sz, Weight::Regular, t::fg(1.0));
                                }
                            }
                            // online and the connection is gone: say so, plainly -
                            // evoview drives on, and that looked like single player
                            if let Some(why) = opts.mp.as_ref().and_then(|m| m.lost()) {
                                use crate::hud::{theme as t, Weight};
                                let s = format!("Disconnected from the server ({why}) - driving offline");
                                let sz = 0.024;
                                let w = hud.label_w(&s, sz, Weight::Semibold);
                                hud.panel(0.5 - w * 0.5 - 0.014, 0.095, w + 0.028, sz * 2.0, 8.0, t::red(0.85), t::line(0.6));
                                hud.label(&s, 0.5 - w * 0.5, 0.095 + sz * 0.5, sz, Weight::Semibold, t::fg(1.0));
                            }
                            // online: whose car the camera follows
                            if let (Some(k), Some(m), true) = (view_ai, opts.mp.as_ref(), viewed.is_some()) {
                                if let Some((n, car, name)) = m.who(k) {
                                    use crate::hud::{theme as t, Weight};
                                    let s = [format!("#{n}"), name, car].into_iter().filter(|x| !x.is_empty())
                                        .collect::<Vec<_>>().join("  ·  ");
                                    let sz = 0.026;
                                    let w = hud.label_w(&s, sz, Weight::Semibold);
                                    hud.panel(0.5 - w * 0.5 - 0.012, 0.035, w + 0.024, sz * 1.9, 8.0, t::panel(0.6), t::line(0.5));
                                    hud.label(&s, 0.5 - w * 0.5, 0.035 + sz * 0.45, sz, Weight::Semibold, t::fg(1.0));
                                }
                            }
                            // the start lights and the radar (race.rs)
                            if let Some(rc) = opts.race.as_ref() {
                                rc.draw_hud(&mut hud, td, view_ai.filter(|_| viewed.is_some()), aspect);
                            }
                            // brake bias / TC / ABS beside it, in the same style
                            if let Some(rc) = rig {
                                crate::aids::draw(&mut hud, rc, aid_pop, aspect, cluster.rect(aspect));
                            }
                        }
                        if let (Some(l), true) = (opts.laps.as_ref(), opts.track_drive.is_some() && crate::envc::var_os("EVOVIEW_NO_HUD").is_none()) {
                            l.draw(&mut hud, aspect);
                            if let Some(p) = opts.pits.as_mut() {
                                p.sign_right = crate::laps::panel_rect(aspect).0 - 0.012 / aspect;
                            }
                        }
                        if let (Some(p), Some(td)) = (opts.pits.as_ref(), opts.track_drive.as_ref()) {
                            if crate::envc::var_os("EVOVIEW_NO_HUD").is_none() {
                                p.draw(&mut hud, td, view_proj, aspect);
                            }
                        }
                        let hud_n = hud.upload(&device, &queue);
                        let frame = match bench_rt.as_ref() {
                            Some(_) => None,
                            None => match surface.get_current_texture() {
                                Ok(f) => Some(f),
                                Err(_) => {
                                    surface.configure(&device, &config);
                                    return;
                                }
                            },
                        };
                        let target = match (frame.as_ref(), bench_rt.as_ref()) {
                            (Some(f), _) => &f.texture,
                            (None, Some(t)) => t,
                            (None, None) => unreachable!(),
                        };
                        let view = target.create_view(&Default::default());
                        let mut enc = device.create_command_encoder(&Default::default());
                        let encode_t = crate::bench::cpu("encode");
                        if let Some(b) = bench.as_mut() {
                            b.begin(&mut enc);
                        }
                        fxg.update(&device, &queue, &mut enc, car_fx.as_deref(), &fxst, dt, [eye.x, eye.y, eye.z]);
                        if let Some(b) = bench.as_mut() { b.mark(&mut enc, "fx"); }
                        if let Some(t) = track.as_ref() {
                            t.cloud_map_pass(&mut enc);
                            crate::bench::mark(&mut enc, "cloudmap");
                            t.sky_pano_pass(&mut enc);
                            crate::bench::mark(&mut enc, "skypano");
                        }
                        {
                            let clear = if opts.start_fly {
                                wgpu::Color { r: 0.55, g: 0.68, b: 0.88, a: 1.0 }
                            } else {
                                { let c = crate::grade::studio_clear_scaled(fx_ambient); wgpu::Color { r: c[0], g: c[1], b: c[2], a: 1.0 } }
                            };
                            let main_out = match taa.as_ref() {
                                Some(t) if taa_on => &t.cur,
                                _ => &view,
                            };
                            let color = if samples > 1 {
                                wgpu::RenderPassColorAttachment {
                                    view: &targets.msaa,
                                    resolve_target: Some(main_out),
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(clear),
                                        store: wgpu::StoreOp::Store,
                                    },
                                }
                            } else {
                                wgpu::RenderPassColorAttachment {
                                    view: main_out,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(clear),
                                        store: wgpu::StoreOp::Store,
                                    },
                                }
                            };
                            // jobs: (layer, target, clear, what) - what 0 all,
                            // 1 the scenery only (the cache), 2 the car only
                            let mut jobs: Vec<(usize, &wgpu::TextureView, bool, u8)> = Vec::new();
                            if shadow_dirty {
                                match shadow_cache.as_ref() {
                                    Some((_, cv)) => {
                                        if cache_dirty {
                                            jobs.push((0, cv, true, 1));
                                        }
                                        jobs.push((0, &shadow_layers[0], false, 2));
                                    }
                                    None => jobs.push((0, &shadow_layers[0], true, if dhero { 1 } else { 0 })),
                                }
                            }
                            if far_dirty {
                                jobs.push((1, &shadow_layers[1], true, if cached { 1 } else { 0 }));
                            }
                            if mid_dirty {
                                jobs.push((LAYER_MID as usize, &shadow_layers[LAYER_MID as usize], true, if cached { 1 } else { 0 }));
                            }
                            if hero_vp.is_some() {
                                jobs.push((LAYER_HERO as usize, &shadow_layers[LAYER_HERO as usize], true, 2));
                            }
                            if cache_dirty && shadow_dirty {
                                shadow_cache_vp = Some(light_vp);
                            }
                            let mut enc_sec = crate::bench::cpu("enc shadow");
                            for (layer, target, clear_it, what) in jobs {
                                if what == 2 {
                                    // the cached scenery into the live cascade
                                    if let Some((ct, _)) = shadow_cache.as_ref() {
                                        enc.copy_texture_to_texture(
                                            ct.as_image_copy(),
                                            wgpu::ImageCopyTexture { texture: &shadow_tex, mip_level: 0,
                                                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 }, aspect: wgpu::TextureAspect::All },
                                            wgpu::Extent3d { width: shadow_dim, height: shadow_dim, depth_or_array_layers: 1 });
                                    }
                                }
                                let mut sp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some(["shadow", "shadow (far)", "shadow (mid)", "shadow (hero)"][layer.min(3)]),
                                    color_attachments: &[],
                                    depth_stencil_attachment: Some(
                                        wgpu::RenderPassDepthStencilAttachment {
                                            view: target,
                                            depth_ops: Some(wgpu::Operations {
                                                load: if clear_it { wgpu::LoadOp::Clear(1.0) } else { wgpu::LoadOp::Load },
                                                store: wgpu::StoreOp::Store,
                                            }),
                                            stencil_ops: None,
                                        },
                                    ),
                                    ..Default::default()
                                });
                                sp.set_pipeline(&pipes_all.shadow);
                                sp.set_bind_group(0, match layer {
                                    0 => &pipes_all.light_bind_group,
                                    1 => &pipes_all.light_bind_group_far,
                                    2 => &pipes_all.light_x[0].1,
                                    _ => &pipes_all.light_x[1].1,
                                }, &[]);
                                // the hero map is a HERO_DIM corner of its layer
                                if layer == LAYER_HERO as usize {
                                    let d = hero_dim().min(shadow_dim) as f32;
                                    sp.set_viewport(0.0, 0.0, d, d, 0.0, 1.0);
                                }
                                // the culler's slot for this cascade (5: the middle one)
                                let cslot = if layer == LAYER_MID as usize { 5 } else { layer };
                                // this map's light box: a moving car's mesh whose
                                // cars all lie outside it casts nothing here
                                let layer_vp = match layer {
                                    0 => Some(light_vp),
                                    1 => Some(far_vp),
                                    l if l == LAYER_MID as usize => Some(mid_vp),
                                    _ => hero_vp,
                                };
                                sp.set_bind_group(2, bone_bg, &[]);
                                for (mi_mesh, m) in meshes.iter().enumerate() {
                                    if Some(mi_mesh) == ground_mesh {
                                        continue;
                                    }
                                    // track geometry carries its extra stream, the car not
                                    if (what == 1 && m.extra.is_none()) || (what == 2 && m.extra.is_some()) {
                                        continue;
                                    }
                                    if m.extra.is_none() && m.instance_count > 0
                                        && layer_vp.map_or(false, |lv| !carcull.casts_into(mi_mesh, lv)) {
                                        continue;
                                    }
                                    // culled to the light's box; casters
                                    // off camera still cast
                                    let (inst_buf, n) = match culler.as_ref().filter(|_| layer != LAYER_HERO as usize)
                                        .and_then(|c| c.shadow(cslot, mi_mesh)) {
                                        Some((b, n)) => (b.unwrap_or(&m.instances), n),
                                        None => (&m.instances, m.instance_count),
                                    };
                                    if n == 0 {
                                        continue;
                                    }
                                    // track geometry carries its extra stream; cars do not
                                    sp.set_pipeline(if m.extra.is_some() { &pipes_all.shadow } else { &pipes_all.shadow_car });
                                    sp.set_vertex_buffer(0, m.vertices.slice(..));
                                    sp.set_vertex_buffer(1, inst_buf.slice(..));
                                    sp.set_index_buffer(m.indices.slice(..),
                                                        wgpu::IndexFormat::Uint32);
                                    for &(start, count, mi) in &m.draws {
                                        if materials.blend.get(mi).copied()
                                            .unwrap_or(Blend::Opaque).transparent()
                                            || materials.casts_no_shadow(mi)
                                            // track cutouts: alpha-tested below
                                            || track.as_ref().map_or(false, |t| t.casts_cutout(mi)) {
                                            continue;
                                        }
                                        sp.set_bind_group(1, &materials.groups[mi], &[]);
                                        sp.draw_indexed(start..start + count, 0, 0..n);
                                    }
                                }
                                if let Some(t) = track.as_ref().filter(|_| what != 2) {
                                    let cul = culler.as_ref();
                                    t.draw_shadow_cutouts(&mut sp, layer, &meshes, &|gi| {
                                        let m = &meshes[gi];
                                        match cul.and_then(|c| c.shadow(cslot, gi)) {
                                            Some((b, n)) => (b.unwrap_or(&m.instances), n),
                                            None => (&m.instances, m.instance_count),
                                        }
                                    });
                                }
                                drop(sp);
                                crate::bench::mark(&mut enc, ["shadow near", "shadow far", "shadow mid", "shadow hero"][layer.min(3)]);
                            }
                            if let Some(b) = bench.as_mut() { b.mark(&mut enc, "shadow"); }
                            drop(enc_sec);
                            enc_sec = crate::bench::cpu("enc mirror");
                            // the main camera's focal length in pixels (mirror / probe LODs)
                            let main_focal = view_proj.row(1).truncate().length() * rh as f32 * 0.5;
                            // one glass mirror a frame in turn, and the rear-view
                            // camera screen every frame: the track behind, through
                            // the track pipelines with each one's camera (its own
                            // track view and cull set, as both draw this frame)
                            if let (Some(mr), Some(t), Some(td)) = (mirrors.as_mut(), track.as_ref(), opts.track_drive.as_ref()) {
                                let body = if drive_mode == 4 { td.body_flat() } else { td.matrices().0 };
                                // Only what the main camera can see: a mirror facing away
                                // or off screen is skipped (each one's camera, its cull and
                                // the whole track again: ~1 ms each at 3440x1440). Seen
                                // from outside the car, the centre mirror and the rear-view
                                // screen sit behind the roof and tinted glass: every 4th
                                // frame. EVOVIEW_MIRROR_SEEN=0: always drawn.
                                let inside = follow != crate::camera::Follow::Free
                                    && car_cams.get(cam_sel).map_or(false, |c| c.mode == 1 || c.mode == 4);
                                let all = crate::envc::var("EVOVIEW_MIRROR_SEEN").map_or(false, |v| v == "0");
                                let seen = |mr: &crate::mirrors::Mirrors, i: usize| all || mr.seen(i, body, vp_draw, eye);
                                let glass = mr.due().filter(|&i| seen(mr, i)
                                    && (all || inside || mr.list[i].def.kind != 0 || mirror_tick % 4 == 0));
                                let cam = mr.camera().filter(|&i| seen(mr, i) && (all || inside || mirror_tick % 4 == 0));
                                mirror_tick = mirror_tick.wrapping_add(1);
                                let due = [glass.map(|i| (i, 2usize, 1usize)), cam.map(|i| (i, 4, 3))];
                                for (i, slot, view) in due.into_iter().flatten() {
                                    // sized to the glass on screen; off screen, not drawn
                                    if !mr.fit(&device, i, body, view_proj, config.height) {
                                        continue;
                                    }
                                    let (mvp, meye) = mr.view(&queue, i, body);
                                    if let Some(c) = culler.as_mut() {
                                        // a mirror is ~200 px tall: whatever covers under
                                        // ~1.5 of its pixels is only draw calls
                                        let focal = mvp.row(1).truncate().length() * mr.list[i].height as f32 * 0.5;
                                        let tiny = (focal > 0.0).then(|| (meye, 1.5 / focal));
                                        c.update_view(&device, &queue, slot, mvp, meye, 0.0, tiny, view_lod_scale(main_focal, focal));
                                    }
                                    t.update_mirror(&queue, mvp, meye, view == 3);
                                    let m = &mr.list[i];
                                    let mut mp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: Some("mirror"),
                                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                            view: if mr.samples() > 1 { &m.msaa } else { &m.resolve },
                                            resolve_target: (mr.samples() > 1).then_some(&m.resolve),
                                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
                                        })],
                                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                            view: &m.depth,
                                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                                            stencil_ops: None,
                                        }),
                                        ..Default::default()
                                    });
                                    let cul = culler.as_ref();
                                    for phase in [Blend::Opaque, Blend::Cutout, Blend::Alpha] {
                                        if phase == Blend::Alpha {
                                            t.draw_sky_view(&mut mp, view);
                                        }
                                        t.draw_phase_view(&mut mp, &meshes, phase,
                                            &|gi, n| cul.and_then(|c| c.shadow(slot, gi)).map_or(n, |s| s.1),
                                            &|gi| cul.and_then(|c| c.shadow(slot, gi)).and_then(|s| s.0),
                                            view);
                                    }
                                    drop(mp);
                                    mr.mips(&mut enc, i);
                                    crate::bench::mark(&mut enc, if view == 3 { "mirror cam" } else { "mirror glass" });
                                }
                            }
                            if let Some(b) = bench.as_mut() { b.mark(&mut enc, "mirror"); }
                            drop(enc_sec);
                            enc_sec = crate::bench::cpu("enc probe");
                            // one reflection-probe face a frame, from the middle
                            // of the car, through the track pipelines again
                            if let (Some(pr), Some(t), Some(td)) = (probe.as_mut(), track.as_ref(), opts.track_drive.as_ref()) {
                                let face = pr.due();
                                let body = if drive_mode == 4 { td.body_flat() } else { td.matrices().0 };
                                // above the roof: from 0.7 m a kerb under the car
                                // filled the whole lower face and the windows
                                // flashed red and white over every kerb
                                let pos = body.w_axis.truncate() + Vec3::Y * 1.6;
                                let (pvp, peye) = crate::probe::Probe::view(face, pos);
                                if let Some(c) = culler.as_mut() {
                                    // whatever covers under ~2 face pixels is only draw calls
                                    let focal = pr.size as f32 * 0.5;
                                    c.update_view(&device, &queue, 3, pvp, peye, 0.0, Some((peye, 2.0 / focal)), view_lod_scale(main_focal, focal));
                                }
                                t.update_probe(&queue, pvp, peye);
                                {
                                    let mut pp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: Some("reflection probe"),
                                        color_attachments: &[Some(pr.attachment(face, clear))],
                                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                            view: pr.depth(),
                                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                                            stencil_ops: None,
                                        }),
                                        ..Default::default()
                                    });
                                    let cul = culler.as_ref();
                                    for phase in [Blend::Opaque, Blend::Cutout, Blend::Alpha] {
                                        if phase == Blend::Alpha {
                                            t.draw_sky_view(&mut pp, 2);
                                        }
                                        t.draw_phase_view(&mut pp, &meshes, phase,
                                            &|gi, n| cul.and_then(|c| c.shadow(3, gi)).map_or(n, |s| s.1),
                                            &|gi| cul.and_then(|c| c.shadow(3, gi)).and_then(|s| s.0),
                                            2);
                                    }
                                }
                                pr.filter(&mut enc, face);
                                if let Some(b) = bench.as_mut() { b.mark(&mut enc, "probe"); }
                            }
                            drop(enc_sec);
                            let _enc_main = crate::bench::cpu("enc main");
                            // + the SSR targets (ssr.rs), cleared: nothing reflects
                            let mut atts = vec![Some(color)];
                            if let Some(s) = ssr.as_ref() {
                                atts.extend(s.attachments(true));
                            }
                            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("main"),
                                color_attachments: &atts,
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachment {
                                        view: &targets.depth,
                                        depth_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(1.0),
                                            store: wgpu::StoreOp::Store,
                                        }),
                                        stencil_ops: None,
                                    },
                                ),
                                ..Default::default()
                            });
                            pass.set_bind_group(0, bind_group, &[]);
                            pass.set_bind_group(2, bone_bg, &[]);
                    // Opaque first, then everything transparent. Within the transparent
                    // pass depth writes are off, so glass never hides the lamp internals
                    // or the interior behind it.
                    // rain: the glass refracts what is behind it, so the scene so
                    // far is copied out first (rain.rs SceneCopy)
                    let refract = fxst.rain > 0.0 && samples == 1 && scene_copy.as_ref().map_or(false, |c| copy_fits(c, rw, rh)) && !no_refract
                        && taa.as_ref().map_or(false, |t| t.on && taa_on);
                    // EVOVIEW_BENCH_SPLIT=1 (--bench, no MSAA): the main pass
                    // broken at each phase so each gets its own GPU time
                    let split = bench.is_some() && samples == 1 && crate::envc::var_os("EVOVIEW_BENCH_SPLIT").is_some();
                    macro_rules! resume_main {
                        ($label:expr) => {{
                            drop(pass);
                            crate::bench::mark(&mut enc, $label);
                            let mut atts = vec![Some(wgpu::RenderPassColorAttachment {
                                view: main_out,
                                resolve_target: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                            })];
                            if let Some(s) = ssr.as_ref() {
                                atts.extend(s.attachments(false));
                            }
                            pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("main (split)"),
                                color_attachments: &atts,
                                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                    view: &targets.depth,
                                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                                    stencil_ops: None,
                                }),
                                ..Default::default()
                            });
                            pass.set_bind_group(0, bind_group, &[]);
                            pass.set_bind_group(2, bone_bg, &[]);
                        }};
                    }
                    let car_first_on = crate::envc::var("EVOVIEW_CAR_FIRST").map_or(true, |v| v != "0");
                    macro_rules! cars_ordered {
                        ($order:expr, $phase:expr) => {{
                            let mut bound = usize::MAX;
                            for (mi_mesh, k) in $order.sorted(eye) {
                                if Some(mi_mesh) == ground_mesh && !ground_on {
                                    continue;
                                }
                                let m = &meshes[mi_mesh];
                                let (start, count, mi) = m.draws[k];
                                if materials.blend.get(mi).copied().unwrap_or(Blend::Opaque) != $phase {
                                    continue;
                                }
                                if mirrors.as_ref().map_or(false, |mr| mr.is_glass(mi_mesh, start)) {
                                    continue;
                                }
                                if bound != mi_mesh {
                                    bound = mi_mesh;
                                    pass.set_vertex_buffer(0, m.vertices.slice(..));
                                    pass.set_vertex_buffer(1, m.instances.slice(..));
                                    pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                                }
                                pass.set_bind_group(1, &materials.groups[mi], &[]);
                                pass.draw_indexed(start..start + count, 0, 0..m.instance_count);
                            }
                        }};
                    }
                    // every non-track mesh in load order (the cars on a track)
                    macro_rules! generic_draws {
                        ($phase:expr) => {{
                        for (mi_mesh, m) in meshes.iter().enumerate() {
                            if Some(mi_mesh) == ground_mesh && !ground_on {
                                continue;
                            }
                            let instances = culler
                                .as_ref()
                                .map_or(m.instance_count, |c| c.count(mi_mesh, m.instance_count));
                            if instances == 0 {
                                continue;
                            }
                            if track.as_ref().map_or(false, |t| t.owns_mesh(m)) {
                                continue;
                            }
                            pass.set_vertex_buffer(0, m.vertices.slice(..));
                            pass.set_vertex_buffer(1, m.instances.slice(..));
                            pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                            for &(start, count, mi) in &m.draws {
                                if m.extra.is_some() && track.as_ref().map_or(false, |t| t.owns(mi)) {
                                    continue;
                                }
                                if materials.blend.get(mi).copied().unwrap_or(Blend::Opaque) != $phase {
                                    continue;
                                }
                                if mirrors.as_ref().map_or(false, |mr| mr.is_glass(mi_mesh, start)) {
                                    continue;
                                }
                                pass.set_bind_group(1, &materials.groups[mi], &[]);
                                if $phase == Blend::Glass {
                                    pass.set_pipeline(&pipes_all.glass_tint);
                                    pass.draw_indexed(start..start + count, 0, 0..instances);
                                    pass.set_pipeline(&pipes_all.glass_add);
                                }
                                pass.draw_indexed(start..start + count, 0, 0..instances);
                            }
                        }
                        }};
                    }
                    for (idx, phase) in [Blend::Opaque, Blend::Cutout, Blend::Alpha, Blend::Glass].iter().enumerate() {
                        if split && idx > 0 {
                            resume_main!(["main opaque", "main cutout", "main alpha"][idx - 1]);
                        }
                        if *phase == Blend::Glass && refract {
                            drop(pass);
                            if let Some(c) = scene_copy.as_ref() {
                                c.copy(&device, &mut enc, main_out, rw, rh);
                            }
                            let mut atts = vec![Some(wgpu::RenderPassColorAttachment {
                                view: main_out,
                                resolve_target: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                            })];
                            if let Some(s) = ssr.as_ref() {
                                atts.extend(s.attachments(false));
                            }
                            pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("main (glass)"),
                                color_attachments: &atts,
                                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                    view: &targets.depth,
                                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                                    stencil_ops: None,
                                }),
                                ..Default::default()
                            });
                            pass.set_bind_group(0, bind_group, &[]);
                            pass.set_bind_group(2, bone_bg, &[]);
                        }
                        // The cars' opaque parts BEFORE the track's: in the cockpit
                        // the car covers half the screen, and drawn second it
                        // shaded the track behind the dash only to cover it
                        // (2-3 ms at 3440x1440). Depth-tested opaque draws: the
                        // same picture. EVOVIEW_CAR_FIRST=0: the old order.
                        let car_first = *phase == Blend::Opaque && track.is_some() && car_first_on;
                        if car_first {
                            pass.set_pipeline(pipeline);
                            match draw_order.as_mut() {
                                Some(order) => cars_ordered!(order, *phase),
                                None => generic_draws!(*phase),
                            }
                            if split {
                                resume_main!("car opaque");
                            }
                        }
                        // Track geometry, through its own pipeline; the
                        // sky fills what is left once everything opaque is in.
                        if let Some(t) = track.as_ref() {
                            if *phase == Blend::Alpha {
                                t.draw_sky(&mut pass);
                                if split {
                                    resume_main!("sky");
                                }
                            }
                            t.draw_phase(&mut pass, &meshes, *phase, &|i, n| {
                                culler.as_ref().map_or(n, |c| c.count(i, n))
                            });
                            pass.set_bind_group(0, bind_group, &[]);
                            pass.set_bind_group(2, bone_bg, &[]);
                        }
                        if split && *phase == Blend::Opaque {
                            resume_main!(if car_first { "track opaque" } else { "track opaque (first)" });
                        }
                        pass.set_pipeline(if *phase == Blend::Alpha { blended } else { pipeline });
                        let _ = idx;
                        // Cars: opaque + cutout front to back (see DrawOrder).
                        if let (Some(order), Blend::Opaque | Blend::Cutout) =
                            (draw_order.as_mut(), *phase)
                        {
                            if !car_first {
                                cars_ordered!(order, *phase);
                            }
                            continue;
                        }
                        if !car_first {
                            generic_draws!(*phase);
                        }
                    }
                            if let Some(mr) = mirrors.as_ref() {
                                mr.draw_glass(&queue, &mut pass, &meshes, vp_draw);
                            }
                            drop(pass);
                            // particles after the main pass, soft against its depth
                            // (after TAA when it can: smoke has no motion vectors
                            // and its history turned it into trailing blocks)
                            let late_parts = samples == 1 && post.is_some() && taa.as_ref().map_or(false, |t| t.on);
                            if fxg.has_particles() && !late_parts {
                                let pdepth = fxg.depth_group(&device, &targets.depth, None);
                                let mut pp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("particles"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: if samples > 1 { &targets.msaa } else { main_out },
                                        resolve_target: (samples > 1).then_some(main_out),
                                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                                    })],
                                    ..Default::default()
                                });
                                fxg.draw_particles_soft(&mut pp, &pdepth, false);
                            }
                        }
                        if let Some(b) = bench.as_mut() { b.mark(&mut enc, "main"); }
                        if let Some(t) = taa.as_mut().filter(|t| t.on) {
                            if hist_reset {
                                t.reset();
                            }
                            // the driven car moves through a static scene
                            // a box from the wheel layout: track + 0.3 m a side,
                            // wheelbase + 1.1 m overhang each end, 8 cm to 1.5 m
                            if let Some(td) = opts.track_drive.as_ref() {
                                let c = td.slot.centres;
                                let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                                for v in c {
                                    lo = lo.min(v);
                                    hi = hi.max(v);
                                }
                                let r = td.slot.radius.max(0.2);
                                // ⚠ the bottom stops 8 cm ABOVE the road: a box down
                                // into the ground dragged the road under the car
                                // along with it (TAA and motion blur ghosting)
                                let lo = Vec3::new(lo.x - 0.3, lo.y - r + 0.08, lo.z - 1.1);
                                let hi = Vec3::new(hi.x + 0.3, hi.y - r + 1.5, hi.z + 1.1);
                                // ⚠ the body as DRAWN: with the rig the hull's pitch /
                                // roll / heave rides the bones on top of the flat
                                // body - a box on matrices() missed it and the car's
                                // history slid a pixel as it bounced (the car ghost)
                                let b = match (drive_mode, drive.as_ref()) {
                                    (4, Some(d)) => td.body_flat() * d.body(),
                                    _ => td.matrices().0,
                                };
                                let m = b * glam::Mat4::from_translation((lo + hi) * 0.5);
                                t.set_object(Some(m), (hi - lo) * 0.5);
                                if let Some(s) = ssr.as_mut() {
                                    s.set_object(Some(m));
                                }
                                t.set_wheels(td.rig.as_ref().map(|(rc, _)| ([0, 1, 2, 3].map(|k| rc.hub(k)), r + 0.05)));
                                if let Some(p) = post.as_mut() {
                                    p.set_object(Some(m), (hi - lo) * 0.5);
                                }
                            } else {
                                t.set_object(None, Vec3::ZERO);
                                if let Some(s) = ssr.as_mut() {
                                    s.set_object(None);
                                }
                                t.set_wheels(None);
                                if let Some(p) = post.as_mut() {
                                    p.set_object(None, Vec3::ZERO);
                                }
                            }
                            t.set_eye(eye);
                            // reflections off what is on screen, before TAA resolves it
                            if let Some(s) = ssr.as_mut() {
                                if hist_reset {
                                    s.reset();
                                }
                                s.run(&device, &queue, &mut enc, &targets.depth, &t.cur, vp_draw, eye);
                                if let Some(b) = bench.as_mut() { b.mark(&mut enc, "ssr"); }
                            }
                            // motion vectors: the scenery through the camera,
                            // the cars (everything not track geometry) drawn again
                            let _enc_post = crate::bench::cpu("enc post");
                            let mv_on = match motion.as_mut() {
                                Some(mo) => {
                                    if hist_reset {
                                        mo.reset();
                                    }
                                    // the cars: the driven one's / each replay slot's
                                    // body and wheel meshes
                                    let mut dyn_mesh = vec![false; meshes.len()];
                                    for gi in carcull.lod_meshes() {
                                        if let Some(d) = dyn_mesh.get_mut(gi) {
                                            *d = true;
                                        }
                                    }
                                    for (b, w) in &slot_gpu {
                                        for &gi in b.iter().chain(w.iter().flatten()) {
                                            if let Some(d) = dyn_mesh.get_mut(gi) {
                                                *d = true;
                                            }
                                        }
                                    }
                                    let mats = &materials;
                                    // (the camera it reprojects through, before run moves it on)
                                    let pvp = mo.prev_vp().unwrap_or(view_proj);
                                    mo.run(&device, &queue, &mut enc, &targets.depth, &bone_buf, &meshes,
                                           &|i| dyn_mesh[i],
                                           &|mi| !mats.blend.get(mi).copied().unwrap_or(Blend::Opaque).transparent(),
                                           vp_draw, view_proj, eye);
                                    // the track's animated props and swaying vegetation
                                    if let Some(tr) = track.as_ref() {
                                        tr.draw_motion(&device, &queue, &mut enc, (&mo.view, mo.rg.as_ref().map(|r| &r.1)),
                                                       &targets.depth, mo.size(), view_proj, pvp, hist_reset, &meshes,
                                                       &|i, n| culler.as_ref().map_or(n, |c| c.count(i, n)));
                                    }
                                    if let Some(b) = bench.as_mut() { b.mark(&mut enc, "motion"); }
                                    true
                                }
                                None => false,
                            };
                            let mv = motion.as_ref().filter(|_| mv_on).map(|m| &m.view);
                            // DLSS: the jittered scene + depth + motion vectors ->
                            // its output, which the resolve then only finishes
                            // (occlusion, fog); a failure drops back to TAA
                            let mut ext_ok = false;
                            if let (Some(d), Some(mo)) = (dlss.as_mut(), motion.as_ref()) {
                                if hist_reset {
                                    d.reset();
                                }
                                let (jx, jy) = t.jitter();
                                let r = match mo.rg.as_ref() {
                                    Some((rt, rv)) => d.run(&device, &mut enc, (&t.cur_tex, &t.cur),
                                                            (&targets.depth_tex, &targets.depth), (rt, rv), (jx, -jy)),
                                    None => Err("no RG16 motion vectors".into()),
                                };
                                match r {
                                    Ok(()) => ext_ok = true,
                                    Err(e) => eprintln!("DLSS: {e} - back to TAA"),
                                }
                                if let Some(b) = bench.as_mut() { b.mark(&mut enc, "dlss"); }
                            }
                            if !ext_ok && dlss.is_some() {
                                dlss = None;
                                t.reset();
                            }
                            // FSR, the same way
                            let mut fsr_ok = false;
                            if let (Some(f), Some(mo)) = (fsr.as_mut(), motion.as_ref()) {
                                if hist_reset {
                                    f.reset();
                                }
                                let (jx, jy) = t.jitter();
                                let r = match mo.rg.as_ref() {
                                    Some((rt, rv)) => f.run(&device, &mut enc, (&t.cur_tex, &t.cur), &targets.depth,
                                                            (rt, rv), (jx, -jy), view_proj, dt),
                                    None => Err("no RG16 motion vectors".into()),
                                };
                                match r {
                                    Ok(()) => fsr_ok = true,
                                    Err(e) => eprintln!("FSR: {e} - back to TAA"),
                                }
                                if let Some(b) = bench.as_mut() { b.mark(&mut enc, "fsr"); }
                            }
                            if !fsr_ok && fsr.is_some() {
                                if let Some(f) = fsr.take() {
                                    f.shutdown(&device);
                                }
                                t.reset();
                            }
                            let ext = dlss.as_ref().filter(|_| ext_ok).map(|d| &d.out)
                                .or(fsr.as_ref().filter(|_| fsr_ok).map(|f| &f.out));
                            // volumetric fog / sun shafts (taa.wgsl vol_fog) on tracks
                            t.set_volumetrics((opts.start_fly && shadows_on && !hero_on).then(|| {
                                let b = crate::sky::block();
                                crate::taa::Volumetrics {
                                    lvp_near: light_vp,
                                    lvp_far: far_vp,
                                    far: far_on,
                                    lvp_mid: (cam_u.shadow_mid[0] > 0.5)
                                        .then(|| glam::Mat4::from_cols_array_2d(&cam_u.light_vp_mid)),
                                    bias: [cam_u.shadow[1], cam_u.shadow_far[1], cam_u.shadow_mid[1]],
                                    nofs: [cam_u.shadow[3], cam_u.shadow_far[3], cam_u.shadow_mid[3]],
                                    sun_dir: crate::sky::key_dir(),
                                    sun_strength: b[0][3],
                                    sun_col: [b[1][0], b[1][1], b[1][2]],
                                    amb: [b[2][0], b[2][1], b[2][2]],
                                }
                            }));
                            // the post chain (post.rs) after TAA, when there is one
                            match post.as_mut() {
                                Some(p) => {
                                    p.set_eye(eye);
                                    // depth of field: only the TV cameras' overrides set one
                                    p.dof = (opts.track_drive.is_some() && cam_sel == n_car_cams
                                             && follow != crate::camera::Follow::Free)
                                        .then(|| tv.as_ref().and_then(|d| d.dof())).flatten();
                                    // auto focus on the car (the game's path without an
                                    // explicit distance: the depth at the focus point, here
                                    // the car on screen). Our director picks cameras by
                                    // coverage, not the game's zones, so a camera's fixed
                                    // focus (141 m on Imola's pcot set) can land on a car
                                    // 60 m away. EVOVIEW_DOF_FOCUS=game keeps the file's.
                                    if let (Some(d), Some(td)) = (p.dof.as_mut(), opts.track_drive.as_ref()) {
                                        let pc = view_proj * (td.pose().pos + td.pose().up * 0.6).extend(1.0);
                                        if pc.w > 0.0 && crate::envc::var("EVOVIEW_DOF_FOCUS").map_or(true, |v| v != "game") {
                                            d.focus = None;
                                            d.center = Some([(pc.x / pc.w * 0.5 + 0.5).clamp(0.0, 1.0),
                                                             (0.5 - pc.y / pc.w * 0.5).clamp(0.0, 1.0)]);
                                        }
                                    }
                                    t.run(&device, &queue, &mut enc, &targets.depth, &p.input, view_proj, (&shadow_view, &shadow_samp), mv, ext);
                                    if let Some(b) = bench.as_mut() { b.mark(&mut enc, "taa"); }
                                    // half resolution (EVOVIEW_PARTICLES_FULL=1: full)
                                    if samples == 1 && fxg.has_particles() && crate::envc::var_os("EVOVIEW_PARTICLES_FULL").is_none() {
                                        let pdepth = fxg.depth_group(&device, &targets.depth, Some(&p.input));
                                        // half the DEPTH's size: the shader reads its 2x2
                                        fxg.draw_particles_half(&device, &mut enc, &pdepth, &p.input, rw, rh);
                                    } else if samples == 1 && fxg.has_particles() {
                                        let pdepth = fxg.depth_group(&device, &targets.depth, None);
                                        let mut pp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                            label: Some("particles after taa"),
                                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                                view: &p.input,
                                                resolve_target: None,
                                                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                                            })],
                                            ..Default::default()
                                        });
                                        fxg.draw_particles_soft(&mut pp, &pdepth, true);
                                    }
                                    if let Some(b) = bench.as_mut() { b.mark(&mut enc, "particles"); }
                                    // falling rain, over everything but the post chain
                                    let cam_vel = rain_eye.map_or(Vec3::ZERO, |p| (eye - p) / dt.max(1e-3));
                                    rain_eye = Some(eye);
                                    if let (Some(rf), true) = (rainfall.as_mut(), samples == 1 && fxst.rain > 0.0) {
                                        rf.draw(&device, &queue, &mut enc, &p.input, (config.width, config.height),
                                                &targets.depth, (rw, rh), view_proj, eye, cam_vel, fxst.rain, cam_u.grade[0], dt);
                                        if let Some(b) = bench.as_mut() { b.mark(&mut enc, "rain"); }
                                    }
                                    p.run(&device, &queue, &mut enc, &targets.depth, &view, view_proj, &materials.lut_view, mv);
                                    if let Some(b) = bench.as_mut() { b.mark(&mut enc, "post"); }
                                }
                                None => t.run(&device, &queue, &mut enc, &targets.depth, &view, view_proj, (&shadow_view, &shadow_samp), mv, ext),
                            }
                        }
                        if fxst.bloom > 0.0 && !hdr {
                            // the glow pass tests against the scene's depth: its size
                            fxg.bloom(&device, &queue, &mut enc, fxst.bloom, &meshes, &materials, pipes_all,
                                      &targets.depth, &view, rw, rh);
                        }
                        // the replay HUD over the finished frame: drawn in the
                        // main pass it was jittered, fogged and sharpened
                        hud.overlay(&mut enc, &view, hud_n);
                        if let Some(b) = bench.as_mut() {
                            b.end(&mut enc, &device, target);
                        }
                        drop(encode_t);
                        let submitted = {
                            let _t = crate::bench::cpu("submit");
                            queue.submit([enc.finish()])
                        };
                        if let Some(p) = post.as_ref() {
                            p.after_submit();
                            device.poll(wgpu::Maintain::Poll);
                        }
                        if let Some(f) = frame {
                            f.present();
                        }
                        if let Some(b) = bench.as_mut() {
                            b.after_present(&device);
                            // EVOVIEW_DASH_DUMP=<dir> while driving: the dash screens
                            // as drawn at EVOVIEW_BENCH_SHOT_FRAME (the --shot path
                            // only sees a still state)
                            static DASH_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                            let n = DASH_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if let Ok(dir) = crate::envc::var("EVOVIEW_DASH_DUMP") {
                                let at = crate::envc::var("EVOVIEW_BENCH_SHOT_FRAME").ok().and_then(|v| v.parse().ok()).unwrap_or(0u64);
                                if n == at {
                                    for (k, (pi, t)) in materials.dash_targets.iter().enumerate() {
                                        let (w, h, px) = crate::fxgpu::read_rgba(&device, &queue, t);
                                        let p = std::path::Path::new(&dir).join(format!("drive_dash_{k}_page{pi}.png"));
                                        crate::shot::write_png(&p, w, h, &px);
                                        println!("dash dump: {}", p.display());
                                    }
                                }
                            }
                            if b.after_submit(&device, submitted) {
                                b.report();
                                crate::bench::release();
                                elwt.exit();
                                return;
                            }
                        }

                        frames += 1;
                        if fps_t.elapsed().as_secs_f32() >= 1.0 {
                            window.set_title(&format!(
                                "evoview - {} tris - {:.0} fps · lighting {}",
                                total_tris,
                                frames as f32 / fps_t.elapsed().as_secs_f32(),
                                if lit { "on" } else { "off" }
                            ));
                            frames = 0;
                            fps_t = std::time::Instant::now();
                        }
                    }
                    _ => {}
                },
                // NGX's feature goes before the device it was made on
                Event::LoopExiting => {
                    if let Some(d) = dlss.take() {
                        d.shutdown(&device);
                    }
                    if let Some(f) = fsr.take() {
                        f.shutdown(&device);
                    }
                }
                Event::AboutToWait => {
                    if let Some(budget) = frame_budget.or_else(|| drive_pace.as_ref().map(|p| std::time::Duration::from_secs_f32(p.period() * 0.97))) {
                        let slept_from = std::time::Instant::now();
                        // A plain sleep, not ControlFlow::WaitUntil: winit's
                        // wait rounds to the 15.6 ms Windows timer tick and a
                        // 60 fps cap came out near 20 fps. std's sleep uses a
                        // high-resolution timer. Scheduled from the ideal
                        // time so the average rate is the cap; resyncs after
                        // a stall.
                        let now = std::time::Instant::now();
                        if now < next_frame {
                            std::thread::sleep(next_frame - now);
                        }
                        next_frame = (next_frame + budget).max(std::time::Instant::now());
                        if let Some(p) = drive_pace.as_mut() {
                            p.slept = slept_from.elapsed().as_secs_f32();
                        }
                    }
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .unwrap();
}

struct Targets {
    depth: wgpu::TextureView,
    depth_tex: wgpu::Texture,
    msaa: wgpu::TextureView,
}

fn make_targets(device: &wgpu::Device, w: u32, h: u32, format: wgpu::TextureFormat, samples: u32) -> Targets {
    let scene = if HDR.load(std::sync::atomic::Ordering::Relaxed) { HDR_FORMAT } else { format };
    let size = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    let tex = |fmt| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
    };
    // sampled by the TAA resolve (taa.rs) as well as drawn into, and read
    // by DLSS through its texture
    let depth_tex = tex(wgpu::TextureFormat::Depth32Float);
    Targets {
        depth: depth_tex.create_view(&Default::default()),
        depth_tex,
        msaa: tex(scene).create_view(&Default::default()),
    }
}

/// Device limits: wgpu's defaults, but with as many sampled textures per
/// stage as the adapter offers (up to 32). The car shader binds 18, one more
/// than the default 16 allows on a strict adapter.
fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let have = adapter.limits().max_sampled_textures_per_shader_stage;
    if have < 21 {
        eprintln!("warning: this GPU allows {have} textures per shader stage; the car shader needs 21 (18 + the effects' FunctionMask2, Dirt_Map)");
    }
    wgpu::Limits {
        max_sampled_textures_per_shader_stage: have.min(32).max(16),
        ..wgpu::Limits::default()
    }
}

/// Driving's frame pacing: every frame on screen for k whole refreshes.
/// LOD distance scale for a small view of focal length `focal` px next to
/// the main camera's `main` (cull.rs update_view); EVOVIEW_VIEW_LOD=0: off.
fn view_lod_scale(main: f32, focal: f32) -> f32 {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| crate::envc::var("EVOVIEW_VIEW_LOD").map_or(false, |v| v == "0")) || focal <= 0.0 {
        return 1.0;
    }
    (main / focal).max(1.0)
}

struct Pace {
    hz: f32,
    k: f32,
    k_min: f32,
    miss: u32,
    fast: u32,
    n: u32,
    /// the pacing sleep inside the last frame (not work)
    slept: f32,
}

impl Pace {
    fn period(&self) -> f32 {
        self.k / self.hz
    }

    /// Take the measured frame time; give the dt to step by. Misses (a frame
    /// that ran a refresh long) push k up, a long run with room to spare
    /// for k - 1 brings it down.
    fn frame(&mut self, m: f32) -> f32 {
        let p = self.period();
        self.n += 1;
        // the work: the frame minus the pacing sleep (vsync's own wait is
        // still in it, so "fast" is judged against k - 1 refreshes)
        let work = m - self.slept;
        if work > p * 1.02 {
            self.miss += 1;
        }
        if work < (self.k - 1.0) / self.hz * 0.8 {
            self.fast += 1;
        }
        if self.n >= 90 {
            if self.miss > 20 && self.k < 3.0 {
                self.k += 1.0;
                println!("drive pacing: every {:.0} refreshes = {:.1} fps", self.k, self.hz / self.k);
            } else if crate::envc::var_os("EVOVIEW_PACE_DOWN").is_some() && self.fast > 85 && self.k > self.k_min {
                self.k -= 1.0;
                println!("drive pacing: every {:.0} refreshes = {:.1} fps", self.k, self.hz / self.k);
            }
            self.n = 0;
            self.miss = 0;
            self.fast = 0;
        }
        if (m - p).abs() < 0.3 * p { p } else { m }
    }
}
