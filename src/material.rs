//! Textures and materials on the GPU.
//!
//! BC1/BC7 blocks are uploaded exactly as they came out of the game - the
//! exporter de-tiles them, nothing decompresses them. That keeps VRAM use at
//! the game's own footprint and skips a slow CPU step entirely.

use crate::asset::{Car, MaterialEntry};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Per-material constants. Kept in sync with `Material` in shader.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MaterialUniform {
    pub base_color: [f32; 4],
    /// x roughness, y metallic, z has_albedo, w has_normal
    pub factors: [f32; 4],
    /// x has_ao, y opacity, z clearcoat, w is_ground
    pub flags: [f32; 4],
    /// x is_glass, y cutout, z reflectance, w unused
    pub misc: [f32; 4],
    /// splat layers: rgb colour + w roughness, one per channel of the splat map
    pub layer_r: [f32; 4],
    pub layer_g: [f32; 4],
    pub layer_b: [f32; 4],
    /// x/y/z metalness per layer, w has_splat
    pub layer_metal: [f32; 4],
    /// x/y/z clear coat per layer, w unused
    pub layer_coat: [f32; 4],
    /// x = triplanar world scale (0 = mesh UVs)
    pub uv_scale: [f32; 4],
    /// Which of the new data maps are real, so the shader can skip the white
    /// stand-ins: x base roughness, y red-layer roughness, z global-AO strength
    /// (0 = no map), w clear-coat mask.
    pub maps: [f32; 4],
    /// Paint detail layers (bindings 10-13). EVO's per-layer maps are DETAIL
    /// maps that tile at `<Layer>_UVscale` over the unwrap.
    /// x/y/z = Red/Green/Blue detail tile, w = clear-coat normal tile.
    pub layer_uv: [f32; 4],
    /// x/y/z = Red/Green/Blue detail-normal strength (`<Layer>_NormalScale`,
    /// 0 = that layer has no detail normal bound), w = clear-coat normal strength.
    pub layer_nscale: [f32; 4],
    /// x/y/z = Red/Green/Blue clear-coat roughness, w = detail-AO strength:
    /// > 0 the AO map is the Red layer's, < 0 the Green layer's, 0 none.
    pub layer_ccr: [f32; 4],
    // ---- glass + lights (see lights.rs). Paint's layer_uv / layer_nscale /
    // layer_aniso go BEFORE this block when they land.
    /// x glass opacity (Basecolor.w or OpacityOverride), y EdgeThickness,
    /// z emissive mode (0 none, 1 emissive map * emis_f[0], 2 function mask),
    /// w unused
    pub gl_em: [f32; 4],
    /// function-mask colours (rgb = colour * intensity) for the R, G, B
    /// channels at the FRONT of the car (high, low, daylight)
    pub emis_f: [[f32; 4]; 3],
    /// the same at the REAR (fog, brake, tail)
    pub emis_r: [[f32; 4]; 3],
    // ---- effects (lights.rs / fxstate.rs) ----
    /// FunctionMask2 colours: reverse, front indicator, rear indicator,
    /// LF / RF / LR / RR special
    pub emis_x: [[f32; 4]; 7],
    /// x 1 = lamp lens (glass intensities), y 1 = Dirt_Map bound (binding 17),
    /// z rim blur mode (1 blur rim, 2 static rim), w VelocityGamma
    pub fx_m: [f32; 4],
    // ---- splat-layer maps (layers.rs; see `layer_maps`) ----
    /// Per layer (Red, Green, Blue): the layer's own UV, as the game builds
    /// it: xy = <Layer>_UVscale (applied only when either is non-zero),
    /// z = <Layer>_UVRotationAngle in radians (0 unless <Layer>_RotateUV),
    /// w = array layer of its colour map in binding 15 (-1 none).
    pub layer_xf: [[f32; 4]; 3],
    /// Per layer: xy = <Layer>_UVoffset, z = array layer of its roughness map
    /// in binding 7 (-1 none), w = 1 when the roughness map has its own UV
    /// (<Layer>_CustomPbrUV: uv * <Layer>_UVscalePbr, no rotation).
    pub layer_xo: [[f32; 4]; 3],
    /// Per layer: xy = <Layer>_UVscalePbr; z, w unused.
    /// [3]: x = Splatmap_Tighten threshold (weights under it are 0; -1 off).
    pub layer_pbr: [[f32; 4]; 4],
    /// The Uber vehicle shaders' own decode (UberVehicleMaterial DXIL
    /// 0x035a3710 / UberVehicleInteriorMaterial 0x03734cc0; SPEC_uber.md):
    /// [0] x Base_Reflectance, y base clear coat, z Base_ClearCoatRoughness,
    ///     w 1 = this path is on
    /// [1] x Base_Metalness, y Base_Roughness, z Base_AmbientOcclusion
    ///     (AO-map strength), w Global_AmbientOcclusionLevel
    /// [2..5] per layer R/G/B: override mode of x roughness (0 none,
    ///     1 replace, 2 multiply, 3 additive), y metalness, z clear coat,
    ///     w clear-coat roughness (0 none, 1 replace, 2 multiply)
    /// [5..8] per layer: x L_Metalness, y L_ClearCoat, z
    ///     L_ClearCoatRoughness, w L_Roughness
    /// [8..11] per layer: x reflectance mode, y L_Reflectance,
    ///     z Modulation_<L>Roughness, w unused
    /// A layer changes an attribute ONLY when its override flag for that
    /// attribute is set; absent params are 0, as the game reads them.
    pub uber: [[f32; 4]; 11],
    /// x 1 = something about this material had to be guessed and strict
    /// mode is on (strict.rs): drawn with warning stripes
    pub strict: [f32; 4],
}

/// How a material is composited. The engine stores this as `blendMode`:
/// 0 opaque, 1 alpha-blended (glass), 2 alpha-tested (decals, stitching).
#[derive(Clone, Copy, PartialEq)]
pub enum Blend {
    Opaque,
    Alpha,
    Cutout,
    /// Glass: EVO's tinted transmission. Drawn twice - once multiplying what
    /// is behind by the tint, once adding the reflection - instead of
    /// alpha-over, which is what made windows read as mirrors.
    Glass,
}

impl Blend {
    /// Not drawn into depth or the shadow map.
    pub fn transparent(self) -> bool {
        matches!(self, Blend::Alpha | Blend::Glass)
    }
}

pub struct Materials {
    pub layout: wgpu::BindGroupLayout,
    pub groups: Vec<wgpu::BindGroup>,
    /// material file name (lowercase, no extension) -> index into `groups`
    pub by_name: HashMap<String, usize>,
    /// Full path -> index. Tried FIRST, because basenames collide.
    pub by_path: HashMap<String, usize>,
    pub fallback: usize,
    /// the studio floor - not a car material, built here so it shares the layout
    pub ground: usize,
    /// composite mode per bind group, parallel to `groups`
    pub blend: Vec<Blend>,
    /// The environment the shader reflects. The game's own studio cubemap
    /// when the package could supply it, otherwise a 1x1 black cube and the
    /// shader falls back to its analytic studio.
    pub env_view: wgpu::TextureView,
    pub env_samp: wgpu::Sampler,
    pub has_env: bool,
    pub brdf_view: wgpu::TextureView,
    /// The game's grading LUT (studio + outdoor stacked in z), see grade.rs.
    pub lut_view: wgpu::TextureView,
    /// Per bind group: the uniform buffer, its current contents and where the
    /// emissive part comes from, so the light state (H) can be rewritten in
    /// place without rebuilding anything.
    pub buffers: Vec<wgpu::Buffer>,
    pub uniforms: Vec<MaterialUniform>,
    pub light_srcs: Vec<Option<crate::lights::LightSrc>>,
    pub light_state: u32,
    /// (dash page index, render target) per display material (fxgpu.rs)
    pub dash_targets: Vec<(usize, wgpu::Texture)>,
    /// Every texture uploaded for these materials; reuse it rather than
    /// uploading a second copy (see TexCache).
    pub textures: TexCache,
}

fn wgpu_format(kind: &str, srgb: bool) -> wgpu::TextureFormat {
    match (kind, srgb) {
        ("bc1", true) => wgpu::TextureFormat::Bc1RgbaUnormSrgb,
        ("bc1", false) => wgpu::TextureFormat::Bc1RgbaUnorm,
        // single-channel masks (the contact-shadow decals): same 8-byte
        // blocks as BC1, which is why the format enum cannot tell them apart
        ("bc4", _) => wgpu::TextureFormat::Bc4RUnorm,
        ("bc3", true) => wgpu::TextureFormat::Bc3RgbaUnormSrgb,
        ("bc3", false) => wgpu::TextureFormat::Bc3RgbaUnorm,
        ("bc7", true) => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
        ("bc7", false) => wgpu::TextureFormat::Bc7RgbaUnorm,
        // BC6H is an HDR float format: no sRGB variant, and never one.
        ("bc6h", _) => wgpu::TextureFormat::Bc6hRgbUfloat,
        ("rgba8", true) => wgpu::TextureFormat::Rgba8UnormSrgb,
        // ⚠ format 9: 8 bytes a pixel. Uploaded as Rgba8 with a 4-byte row
        // it came out as garbage (half-width, smeared halves)
        ("rgba16f", _) | ("rgba16u", _) => wgpu::TextureFormat::Rgba16Float,
        _ => wgpu::TextureFormat::Rgba8Unorm,
    }
}

/// Upload one texture with its whole mip chain.
///
/// ⚠ `srgb` must be false for anything that is DATA rather than colour -
/// normal maps, roughness, masks. Decoding those through the sRGB curve bends
/// the values and the lighting goes subtly wrong in a way that is hard to
/// attribute later.
/// Name of a texture slot, for diagnostics.
fn alb_name_of(car: &Car, i: Option<usize>) -> String {
    i.and_then(|j| car.textures.get(j))
        .map(|t| t.name.clone())
        .unwrap_or_else(|| "?".into())
}

/// Fill the emissive part of a material uniform. Static: the light state
/// travels in the camera uniform's fx block (lights::fx_block).
fn apply_lights(u: &mut MaterialUniform, src: &crate::lights::LightSrc, _s: u32) {
    let (mode, e, x, fm) = crate::lights::uniforms(src);
    u.gl_em[2] = mode;
    u.emis_f = [e[0], e[1], e[2]];
    u.emis_r = [e[3], e[4], e[5]];
    u.emis_x = x;
    u.fx_m[0] = fm[0];
}

/// Scale on the glass surface's own lighting: the game's car-wide
/// `CarShadingGlobalSetting.glass_reflection_strength` (0.17, SPEC 4.1).
/// EVOVIEW_GLASS_REFL overrides it for tuning.
fn glass_reflection_strength() -> f32 {
    // ⚠ was 0.17: the glass shaders scale nothing - RT0 = a * lit + E
    // (SPEC_small §2), and the extra factor made reflections ~6x too weak
    crate::envc::var("EVOVIEW_GLASS_REFL").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0)
}

/// Glass coverage head-on, EVO's way: `OpacityOverride` when the material
/// asks for it, else the alpha of its `Basecolor` (0.235 on most windows).
/// NOT `ksTransparancy`: that is a legacy AC1 field, 0.8 on every window.
fn glass_opacity(m: &MaterialEntry, has_map: bool) -> f32 {
    let sh = m.shader.as_deref().unwrap_or("").to_lowercase();
    // a txDiffuse without Modulation_BaseColor REPLACES Basecolor, alpha too:
    // the coverage is the map's
    let replaced = has_map && m.param(&["Modulation_BaseColor"], 0.0) < 0.5
        && matches!(sh.as_str(), "vehicleglass" | "vehicleglassinterior" | "vehiclewindow");
    // VehicleWindow has no opacity override: the file's UseOpacityOverride is dead
    if sh != "vehiclewindow" && m.param(&["UseOpacityOverride"], 0.0) > 0.5 {
        return m.param(&["OpacityOverride"], 1.0).clamp(0.0, 1.0);
    }
    // The loader stores a vec4 whose w was omitted as 3 components, so only a
    // real 4th component is an opacity.
    if replaced {
        return 1.0;
    }
    match m.colors.get("Basecolor") {
        Some(v) if v.len() >= 4 => v[3].clamp(0.0, 1.0),
        _ => 0.25,
    }
}

/// Does this material name a car cabin surface?
///
/// EVO names every interior material `int_*`. Kept in one place because the
/// grid legend has to agree with what the shader actually flagged - a legend
/// that disagrees with the picture is worse than no legend.
pub fn is_cabin(fname_lower: &str) -> bool {
    fname_lower.starts_with("int_")
        || fname_lower.contains("_int_")
        || fname_lower.contains("interior")
        || fname_lower.contains("cockpit")
}

/// Is this one of the seated driver's materials (driver.rs)? The parts live
/// under content/characters, per-livery ones under the car's skins/driver/,
/// and an AC1-converted car's under its materials as `ac1_m_drv_*`.
/// ⚠ `skins/driver/` WITH the slash: the F2004 keeps its per-driver BODY
/// liveries in `skins/driver_1/`, `skins/driver_def/`.
pub fn is_driver(path_lower: &str) -> bool {
    path_lower.starts_with("content/characters/")
        || path_lower.contains("/skins/driver/")
        || path_lower.contains("/ac1_m_drv_")
}

/// GPU copies of `Car::textures`, uploaded at most once per (texture index,
/// sRGB). Many materials share a map - on a track the same albedo or normal
/// map backs dozens of materials - and each used to get its own copy. A view
/// per binding is free; the copy is not.
///
/// `Materials::build` fills one and keeps it as `Materials::textures`, so
/// any later pipeline (the track renderer) can bind the same copies.
#[derive(Default)]
pub struct TexCache {
    #[allow(clippy::type_complexity)]
    map: std::cell::RefCell<
        HashMap<(TexKey, bool), Option<(std::rc::Rc<wgpu::Texture>, std::rc::Rc<wgpu::TextureView>)>>,
    >,
}

/// What identifies a texture's CONTENT, so a cache survives a reload of the
/// same car (livery switching builds a new `Car` with new indices; see
/// livery.rs): its vfs key when known ("s:"), else name + byte length ("n:"),
/// plus format and size.
/// ⚠ Never the bare name: every SF-25 livery folder has an ext_skin_d.texture
/// of the same size, and a name key shows the first livery's skin on all six.
type TexKey = (String, String, u32, u32, u64);

fn tex_key(car: &Car, idx: usize) -> TexKey {
    let src = car.tex_src.get(idx).filter(|s| !s.is_empty());
    match (car.textures.get(idx), src) {
        (Some(t), Some(s)) => (format!("s:{s}"), t.format.clone(), t.width, t.height, 0),
        (Some(t), None) => (format!("n:{}", t.name), t.format.clone(), t.width, t.height, t.data[1]),
        (None, _) => (format!("#{idx}"), String::new(), 0, 0, 0),
    }
}

impl TexCache {
    /// Vfs keys of every texture uploaded (see `Vfs::mark_resident`).
    pub fn resident_srcs(&self) -> Vec<String> {
        self.map
            .borrow()
            .iter()
            .filter(|(_, v)| v.is_some())
            .filter_map(|((k, _), _)| k.0.strip_prefix("s:").map(str::to_string))
            .collect()
    }

    /// The uploaded view of texture `idx`, uploading it on first use.
    /// None when the texture is missing or cannot be uploaded (unaligned BC).
    pub fn get(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        car: &Car,
        idx: usize,
        srgb: bool,
    ) -> Option<std::rc::Rc<wgpu::TextureView>> {
        self.entry(device, queue, car, idx, srgb).map(|e| e.1)
    }

    /// The texture itself, for callers that want their own view of it.
    pub fn texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        car: &Car,
        idx: usize,
        srgb: bool,
    ) -> Option<std::rc::Rc<wgpu::Texture>> {
        self.entry(device, queue, car, idx, srgb).map(|e| e.0)
    }

    fn entry(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        car: &Car,
        idx: usize,
        srgb: bool,
    ) -> Option<(std::rc::Rc<wgpu::Texture>, std::rc::Rc<wgpu::TextureView>)> {
        let key = (tex_key(car, idx), srgb);
        if let Some(v) = self.map.borrow().get(&key) {
            crate::perf::add("tex upload: shared", 0.0);
            return v.clone();
        }
        let t0 = std::time::Instant::now();
        let e = upload_texture(device, queue, car, idx, srgb).map(|t| {
            let v = std::rc::Rc::new(t.create_view(&Default::default()));
            (std::rc::Rc::new(t), v)
        });
        crate::perf::add("tex upload: unique", t0.elapsed().as_secs_f64());
        self.map.borrow_mut().insert(key, e.clone());
        e
    }
}

thread_local! {
    /// The cache `upload` fills while a `Materials::build` runs, so the many
    /// call sites inside it need no extra parameter.
    static BUILDING: std::cell::RefCell<Option<TexCache>> = const { std::cell::RefCell::new(None) };
}

pub(crate) fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    car: &Car,
    idx: usize,
    srgb: bool,
) -> Option<wgpu::TextureView> {
    BUILDING.with(|c| {
        let c = c.borrow();
        match c.as_ref() {
            // a fresh view of the shared texture; bind groups hold their own
            Some(cache) => cache
                .texture(device, queue, car, idx, srgb)
                .map(|t| t.create_view(&Default::default())),
            None => upload_texture(device, queue, car, idx, srgb)
                .map(|t| t.create_view(&Default::default())),
        }
    })
}

/// `upload`, but the texture itself (shared through the build's cache), for
/// callers that make their own views - the layer arrays (layers.rs).
pub(crate) fn upload_rc(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    car: &Car,
    idx: usize,
    srgb: bool,
) -> Option<std::rc::Rc<wgpu::Texture>> {
    BUILDING.with(|c| match c.borrow().as_ref() {
        Some(cache) => cache.texture(device, queue, car, idx, srgb),
        None => upload_texture(device, queue, car, idx, srgb).map(std::rc::Rc::new),
    })
}

/// `write_texture` copies into a staging buffer that lives until the next
/// submit completes. Materials::build uploads every texture before anything
/// is submitted, so a track briefly held a second, staged copy of all its
/// textures (~1.3 GB) in system memory. Flushing every 256 MB caps that.
fn flush_staging(device: &wgpu::Device, queue: &wgpu::Queue, bytes: u64) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static PENDING: AtomicU64 = AtomicU64::new(0);
    if PENDING.fetch_add(bytes, Ordering::Relaxed) + bytes >= 256 << 20 {
        PENDING.store(0, Ordering::Relaxed);
        let t = std::time::Instant::now();
        queue.submit(std::iter::empty());
        device.poll(wgpu::Maintain::Wait);
        crate::perf::add("tex upload: staging flush", t.elapsed().as_secs_f64());
    }
}

fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    car: &Car,
    idx: usize,
    srgb: bool,
) -> Option<wgpu::Texture> {
    let t = car.textures.get(idx)?;
    let data = car.slice(Some(t.data));
    if data.is_empty() || t.levels.is_empty() {
        return None;
    }
    // RGBA16 UNORM (load.rs format 25) as half floats; a colour slot gets its
    // sRGB decode here, since there is no sRGB 16-bit format to do it
    let conv;
    let data = if t.format == "rgba16u" {
        conv = data.chunks_exact(2).enumerate().flat_map(|(i, c)| {
            let v = u16::from_le_bytes([c[0], c[1]]) as f32 / 65535.0;
            // (alpha, every fourth value, stays linear)
            let v = if srgb && i % 4 != 3 { if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } } else { v };
            crate::load::unorm_to_half(v).to_le_bytes()
        }).collect::<Vec<u8>>();
        &conv[..]
    } else {
        data
    };
    let format = wgpu_format(&t.format, srgb);
    let block = if matches!(t.format.as_str(), "bc1" | "bc4" | "bc7" | "bc6h" | "bc3") { 4 } else { 1 };
    let block_bytes = match t.format.as_str() {
        "bc1" | "bc4" => 8,
        "bc7" | "bc6h" | "bc3" => 16,
        "rgba16f" | "rgba16u" => 8,
        _ => 4,
    };
    // BC formats require both edges of the texture to be a multiple of 4
    // (COTA railing_glass is 50 px tall, Watkins runoff 1082 wide: created at
    // those sizes the GPU texture panics). The texture is created at the
    // block-rounded size so EVERY level fits: the old rule (keep the run of
    // levels whose own size is a multiple of 4) dropped the top level or cut
    // the chain at the first odd mip (1082 -> 541), leaving distant surfaces
    // to alias on the last level kept. Below 4 px a level's physical size is
    // one block, which wgpu accepts as the copy extent.
    let up = |v: u32| (v + block - 1) / block * block;
    let blocks = |v: u32| (v + block - 1) / block;
    let (fw, fh) = (up(t.levels[0].width.max(1)), up(t.levels[0].height.max(1)));
    let max_levels = 32 - fw.max(fh).leading_zeros();
    let mut have = 0usize;
    let mut end = 0usize;
    for l in &t.levels {
        end += l.bytes as usize;
        if end > data.len() {
            break;
        }
        have += 1;
    }
    let count = have.min(max_levels as usize);
    if count == 0 {
        return None;
    }
    // EVOVIEW_MIP_LOG=1: what the old rule would have kept
    if crate::envc::var_os("EVOVIEW_MIP_LOG").is_some() {
        let aligned = |l: &crate::asset::Level| {
            l.width >= block && l.height >= block && l.width % block == 0 && l.height % block == 0
        };
        let first = t.levels[..have].iter().position(|l| aligned(l));
        let kept = first.map_or(0, |f| t.levels[f..have].iter().take_while(|l| aligned(l)).count());
        // levels the old rule lost that are not just the sub-block tail
        let lost = (0..have).filter(|&i| {
            let l = &t.levels[i];
            let inside = first.map_or(false, |f| i >= f && i < f + kept);
            !inside && l.width >= block && l.height >= block
        }).count();
        static OLD: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        static NEW: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        static ALL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        use std::sync::atomic::Ordering::Relaxed;
        let all = ALL.fetch_add(1, Relaxed) + 1;
        let old = if lost > 0 { OLD.fetch_add(1, Relaxed) + 1 } else { OLD.load(Relaxed) };
        let new = if count < have && t.levels[count].width >= block && t.levels[count].height >= block {
            NEW.fetch_add(1, Relaxed) + 1
        } else {
            NEW.load(Relaxed)
        };
        if lost > 0 {
            println!("mips: {} {}x{} {}: old rule kept {kept} of {have} levels from {:?} ({lost} real levels lost), now {count} [textures affected: old {old}, now {new}, of {all}]",
                     t.name, t.levels[0].width, t.levels[0].height, t.format, first);
        } else if all % 500 == 0 {
            println!("mips: [textures affected: old {old}, now {new}, of {all}]");
        }
    }

    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&t.name),
        size: wgpu::Extent3d {
            width: fw,
            height: fh,
            depth_or_array_layers: 1,
        },
        mip_level_count: count as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let mut off = 0usize;
    for (i, l) in t.levels.iter().take(count).enumerate() {
        let n = l.bytes as usize;
        let src = &data[off..off + n];
        off += n;
        // the level as stored, and as the (block-rounded) texture holds it
        let (dbx, dby) = (blocks(l.width.max(1)), blocks(l.height.max(1)));
        let (tw, th) = (up((fw >> i).max(1)), up((fh >> i).max(1)));
        let (tbx, tby) = (tw / block, th / block);
        let row = dbx * block_bytes;
        if (dbx as usize) * (dby as usize) * (block_bytes as usize) > src.len() {
            break;
        }
        // a halving that rounds differently (1081 -> 540 vs the texture's
        // 542) can leave the texture one block wider or taller than the
        // data: repeat the edge blocks into it
        let repacked;
        let (bytes, row) = if (dbx, dby) == (tbx, tby) {
            (src, row)
        } else {
            let bb = block_bytes as usize;
            let mut v = vec![0u8; (tbx * tby) as usize * bb];
            for y in 0..tby as usize {
                let sy = y.min(dby as usize - 1);
                for x in 0..tbx as usize {
                    let sx = x.min(dbx as usize - 1);
                    let s = sy * row as usize + sx * bb;
                    let d = (y * tbx as usize + x) * bb;
                    v[d..d + bb].copy_from_slice(&src[s..s + bb]);
                }
            }
            repacked = v;
            (&repacked[..], tbx * block_bytes)
        };
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: i as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(tby),
            },
            wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
        );
    }
    flush_staging(device, queue, data.len() as u64);
    // a chain that stops early (no small levels) sparkles wherever it is
    // minified: completed on the GPU (mipfill.rs)
    if crate::mipfill::wanted() {
        let have = (0..count).take_while(|&i| {
            let l = &t.levels[i];
            (blocks(l.width.max(1)) as usize) * (blocks(l.height.max(1)) as usize) * (block_bytes as usize) <= l.bytes as usize
        }).count() as u32;
        let t0 = std::time::Instant::now();
        let filled = crate::mipfill::complete(device, queue, &tex, have);
        if let Some(full) = filled {
            crate::perf::add("tex: mip fill", t0.elapsed().as_secs_f64());
            if crate::envc::var_os("EVOVIEW_MIPFILL_LOG").is_some() {
                println!("mipfill: {} {}x{} {} had {have} of {} levels", t.name, fw, fh, t.format, full.mip_level_count());
            }
            return Some(full);
        }
    }
    Some(tex)
}

fn paint_tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// The Red splat layer's own colour map, when the game would draw it.
///
/// EVO's Uber shaders (UberVehicleInteriorMaterial DXIL 0x03734cc0) compose
/// base colour per layer:
///   Red_v = Red_HasBaseColorMap && map bound
///           ? (Modulation_RedBaseColor ? map * Red_Basecolor : map)
///           : Red_Basecolor
///   v = lerp(Base_v, Replace ? Red_v : Red_v * Base_v, w.r)
/// and only when RedOverrideReplace_ or RedOverrideMultiply_BaseColor is set.
/// The map is sampled at Red_UVscale (rotated by Red_UVRotationAngle when
/// Red_RotateUV), not at the base unwrap.
///
/// ⚠ We used to fold this map into the BASE albedo and then let the layer
/// colour replace it, so the map vanished. GR86 `int_occ_fabric_perf` has
/// Red_Basecolor 1.0 + Replace, so the seat centres drew pure white instead of
/// the dark `detail_leather_dotted_c`; AC1-converted suits/gloves put the real
/// colours on Red_BaseColorMap (Multiply) and drew only the grey fabric base.
///
/// ⚠ `Red_HasBaseColorMap` is honoured: the GR86 alcantara and silver fabric
/// bind a map with the flag off and the game draws the flat Red_Basecolor.
/// Only where the shader's layer path runs (a SpatMap, or Splatmap_UseConstant
/// with the 1x1 weight texture). Body paint (`ext_skin*`) is left to the
/// livery code.
///
/// The Green and Blue layers follow the SAME rule (UberVehicleMaterial DXIL
/// 0x035a3710, flags named by cbuffer offset with layers/flagname.py):
///   skip unless <L>_Enable && w > 0, and unless <L>OverrideReplace_ or
///   <L>OverrideMultiply_BaseColor; map only if <L>_HasBaseColorMap && bound;
///   Modulation_<L>BaseColor multiplies it by <L>_Basecolor.
/// Stock uses Green for oil / dirt detail (detail_engine_oil_Colour, multiply)
/// and Blue for noise; the AC1 converter puts its tiled AC1 detail textures
/// (leather, carpet, carbon) on Blue in multiply mode, which drew WHITE while
/// only the Red map was implemented.
/// Body paint (`ext_skin*`) keeps its Red layer to the livery code.
fn layer_map(m: &MaterialEntry, layer: &str, shader_l: &str, fname: &str,
             splat_path: bool) -> Option<usize> {
    if !shader_l.contains("ubervehicle") || !splat_path
        || (layer == "Red" && fname.starts_with("ext_skin"))
    {
        return None;
    }
    let on = |k: String| m.param(&[&k], 0.0) > 0.5;
    if !on(format!("{layer}_Enable")) || !on(format!("{layer}_HasBaseColorMap"))
        || !(on(format!("{layer}OverrideReplace_BaseColor"))
            || on(format!("{layer}OverrideMultiply_BaseColor")))
    {
        return None;
    }
    m.slot(&[&format!("{layer}_BaseColorMap")])
}

/// A layer's roughness map, when the game samples it: <L>_HasRoughnessMap
/// and bound (the Red one keeps evoview's older rule - bound is enough - so
/// stock paint does not change).
fn layer_rough_map(m: &MaterialEntry, layer: &str) -> Option<usize> {
    if m.param(&[&format!("{layer}_Enable")], 0.0) < 0.5 {
        return if layer == "Red" { m.slot(&["Red_RoughnessMap"]) } else { None };
    }
    if layer != "Red" && m.param(&[&format!("{layer}_HasRoughnessMap")], 0.0) < 0.5 {
        return None;
    }
    m.slot(&[&format!("{layer}_RoughnessMap")])
}

/// A layer's UV transform as the game applies it (see MaterialUniform.layer_xf).
fn layer_uv_xf(m: &MaterialEntry, layer: &str) -> ([f32; 4], [f32; 2], [f32; 2], bool) {
    let s = uv_scale2(m, &format!("{layer}_UVscale")).unwrap_or([0.0, 0.0]);
    let rot = if m.param(&[&format!("{layer}_RotateUV")], 0.0) > 0.5 {
        m.param(&[&format!("{layer}_UVRotationAngle")], 0.0).to_radians()
    } else {
        0.0
    };
    let off = uv_scale2(m, &format!("{layer}_UVoffset")).unwrap_or([0.0, 0.0]);
    let pbr = uv_scale2(m, &format!("{layer}_UVscalePbr")).unwrap_or([0.0, 0.0]);
    let custom = m.param(&[&format!("{layer}_CustomPbrUV")], 0.0) > 0.5;
    ([s[0], s[1], rot, -1.0], off, pbr, custom)
}

/// A 2-vector param such as `Red_UVscale` / `Base_UVscale`, which EVO stores
/// as [u, v]. ⚠ `MaterialEntry::color` only accepts 3+ components, so every
/// UV scale read through it silently came back as the default.
fn vec2_param(m: &MaterialEntry, key: &str) -> Option<[f32; 2]> {
    let v = m.colors.get(key)?;
    let u = *v.first()?;
    Some([u, *v.get(1).unwrap_or(&u)])
}

/// A UV scale / offset however it was stored: a 2-vector, a wider vector, or
/// a scalar (uniform). None when absent (an empty value = 0 is dropped by the
/// parser, so absent also means 0).
fn uv_scale2(m: &MaterialEntry, key: &str) -> Option<[f32; 2]> {
    vec2_param(m, key).or_else(|| m.params.get(key).map(|&s| [s, s]))
}

fn layer_array_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

/// Which paint detail maps a material binds, and the constants that go with
/// them. EVO's UberVehicleMaterial gives every splat layer its own DETAIL
/// maps tiled at `<Layer>_UVscale`: `Red_NormalMap` is flakes_NM on paint,
/// Green/Blue carry carbon twill, `Red_ClearCoatNormalMap` is orange peel.
/// There are four slots for up to seven maps, so Green and Blue share one
/// normal slot (they are usually the same texture) and Red/Green share AO.
pub struct PaintDetail {
    pub n_r: Option<usize>,
    pub n_gb: Option<usize>,
    pub cc_n: Option<usize>,
    pub ao: Option<usize>,
    pub layer_uv: [f32; 4],
    pub layer_nscale: [f32; 4],
    pub layer_ccr: [f32; 4],
}

/// One detail tile per layer for the shared detail-normal / AO slots. EVO
/// stores the scale as a 2-vector [u, v]; ⚠ it was read through the 3+
/// component colour accessor, so every one fell back to 1 and flakes /
/// carbon twill / AC1 leather grain were stretched across the whole part.
/// The slots take one scalar: u (every shipped detail tiles square).
fn layer_tile(m: &MaterialEntry, key: &str) -> f32 {
    let s = uv_scale2(m, key).map_or(0.0, |v| v[0]);
    if s > 0.0 { s } else { 1.0 }
}

/// The tile a layer's detail NORMAL uses: <L>_UVscaleNormal when the layer
/// gives normals their own UV (<L>_CustomNormalUV), else the layer's UV.
fn layer_normal_tile(m: &MaterialEntry, layer: &str) -> f32 {
    if m.param(&[&format!("{layer}_CustomNormalUV")], 0.0) > 0.5 {
        layer_tile(m, &format!("{layer}_UVscaleNormal"))
    } else {
        layer_tile(m, &format!("{layer}_UVscale"))
    }
}

pub fn paint_detail_maps(m: &MaterialEntry) -> PaintDetail {
    let on = |l: &str| m.param(&[&format!("{l}_Enable")], 0.0) > 0.5;
    let slot = |l: &str, what: &str| -> Option<usize> {
        if on(l) { m.slot(&[&format!("{l}_{what}")]) } else { None }
    };
    let n_r = slot("Red", "NormalMap");
    let g_n = slot("Green", "NormalMap");
    let b_n = slot("Blue", "NormalMap");
    let n_gb = g_n.or(b_n);
    let scale = |l: &str, idx: Option<usize>| -> f32 {
        match idx {
            Some(_) => m.param(&[&format!("{l}_NormalScale")], 0.0).clamp(0.0, 4.0),
            None => 0.0,
        }
    };
    // The shared slot only serves the layers whose map IS that texture.
    let g_use = if g_n.is_some() && g_n == n_gb { g_n } else { None };
    let b_use = if b_n.is_some() && b_n == n_gb { b_n } else { None };
    // Clear-coat normal: the paint layer's orange peel first, then the base's.
    // `Base_ClearCoatNormalMap` is often the part's own normal map (calipers,
    // lights), which then carries straight through to the coat.
    let (cc_n, cc_scale, cc_tile) = match slot("Red", "ClearCoatNormalMap") {
        Some(i) => (Some(i),
                    m.param(&["Red_ClearCoatNormalScale"], 0.0).clamp(0.0, 4.0),
                    layer_tile(m, "Red_UVscaleClearCoatNormal")),
        None => match m.slot(&["Base_ClearCoatNormalMap"]) {
            Some(i) => (Some(i),
                        m.param(&["Base_ClearCoatNormalScale"], 0.0).clamp(0.0, 4.0),
                        layer_tile(m, "Base_UVscaleClearCoatNormal")),
            None => (None, 0.0, 1.0),
        },
    };
    // Detail AO: strength is `<Layer>_AmbientOcclusion` (0 switches it off).
    let (ao, ao_s) = match slot("Red", "AmbientOcclusionMap") {
        Some(i) => (Some(i), m.param(&["Red_AmbientOcclusion"], 0.0).clamp(0.0, 1.0)),
        None => match slot("Green", "AmbientOcclusionMap") {
            Some(i) => (Some(i), -m.param(&["Green_AmbientOcclusion"], 0.0).clamp(0.0, 1.0)),
            None => (None, 0.0),
        },
    };
    let ao = if ao_s == 0.0 { None } else { ao };
    let cc_n = if cc_scale > 0.0 { cc_n } else { None };
    let ccr = |l: &str| m.param(&[&format!("{l}_ClearCoatRoughness")], 0.0).clamp(0.0, 1.0);
    PaintDetail {
        n_r: if scale("Red", n_r) > 0.0 { n_r } else { None },
        n_gb: if scale("Green", g_use) + scale("Blue", b_use) > 0.0 { n_gb } else { None },
        cc_n,
        ao,
        layer_uv: [layer_normal_tile(m, "Red"), layer_normal_tile(m, "Green"),
                   layer_normal_tile(m, "Blue"), cc_tile],
        layer_nscale: [scale("Red", n_r), scale("Green", g_use), scale("Blue", b_use),
                       if cc_n.is_some() { cc_scale } else { 0.0 }],
        layer_ccr: [ccr("Red"), ccr("Green"), ccr("Blue"), if ao.is_some() { ao_s } else { 0.0 }],
    }
}

/// Fixed splat weights for an UberVehicleMaterial with `Splatmap_UseConstant`
/// and no SpatMap, as an RGBA8 texel. None when the material has a real splat
/// (or anything else in the splat slot), or is not a vehicle material - track
/// UberBiplanar ground also carries a Splatmap_Constant and has its own path.
fn paint_splat_constant(m: &MaterialEntry, slot_taken: bool) -> Option<[u8; 4]> {
    if slot_taken || m.param(&["Splatmap_UseConstant"], 0.0) < 0.5 {
        return None;
    }
    let shader = m.shader.as_deref().unwrap_or("").to_lowercase();
    if !shader.contains("ubervehicle") {
        return None;
    }
    let c = match m.colors.get("Splatmap_Constant") {
        Some(v) => [v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0),
                    v.get(2).copied().unwrap_or(0.0)],
        None => [m.param(&["Splatmap_Constant"], 0.0), 0.0, 0.0],
    };
    let q = |x: f32| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    Some([q(c[0]), q(c[1]), q(c[2]), 255])
}

pub(crate) fn solid(device: &wgpu::Device, queue: &wgpu::Queue, rgba: [u8; 4]) -> wgpu::TextureView {
    let tex = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("solid"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        Default::default(),
        &rgba,
    );
    tex.create_view(&Default::default())
}

impl Materials {
    pub fn build(device: &wgpu::Device, queue: &wgpu::Queue, car: &Car) -> Self {
        Self::build_reusing(device, queue, car, TexCache::default())
    }

    /// Take `other`'s material groups on after ours (online: a late joiner's
    /// car, built on its own); returns the index its first group now has.
    /// Its layout matches ours field for field, which wgpu deduplicates, so
    /// the pipelines accept its groups.
    pub fn append(&mut self, other: Materials) -> usize {
        let off = self.groups.len();
        self.groups.extend(other.groups);
        self.blend.extend(other.blend);
        self.buffers.extend(other.buffers);
        self.uniforms.extend(other.uniforms);
        self.light_srcs.extend(other.light_srcs);
        for (k, v) in other.by_name {
            self.by_name.entry(k).or_insert(v + off);
        }
        for (k, v) in other.by_path {
            self.by_path.entry(k).or_insert(v + off);
        }
        self.dash_targets.extend(other.dash_targets.into_iter().map(|(i, t)| (i + off, t)));
        off
    }

    /// `build`, starting from textures an earlier build uploaded (a livery
    /// switch: most of the car's maps are the same files).
    pub fn build_reusing(device: &wgpu::Device, queue: &wgpu::Queue, car: &Car, prev: TexCache) -> Self {
        BUILDING.with(|c| *c.borrow_mut() = Some(prev));
        let mut m = Self::build_inner(device, queue, car);
        m.textures = BUILDING.with(|c| c.borrow_mut().take()).unwrap_or_default();
        m
    }

    fn build_inner(device: &wgpu::Device, queue: &wgpu::Queue, car: &Car) -> Self {
        let layout_entries = vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // albedo, normal, ao
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // splat map: which paint layer owns each panel
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 6..9 - the data maps EVO ships that we were throwing away.
                // Roughness used to come from ONE constant per layer, which is why
                // paint and carbon read flat: EVO's materials bind Base + Red
                // roughness, a global AO and a clear-coat map.
                //   6 base roughness   7 red-layer roughness
                //   8 global AO        9 clear-coat mask
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 7: the splat layers' roughness maps, one array (layers.rs)
                layer_array_entry(7),
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
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
                // 10..13 - paint detail layers (see `paint_detail_maps`):
                //   10 Red detail normal (flakes_NM)   11 Green/Blue detail normal
                //   12 clear-coat normal (orange peel) 13 Red/Green detail AO
                paint_tex_entry(10),
                paint_tex_entry(11),
                paint_tex_entry(12),
                paint_tex_entry(13),
                // 14: t_emis - FunctionMask1 for lamps, Base_EmissiveMap for
                // interior backlights (lights.rs)
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 15: t_layer_c - the splat layers' own colour maps, one array
                // (layer_maps / layers.rs)
                layer_array_entry(15),
                // 16: FunctionMask2 (indicators / reverse / specials),
                // 17: Dirt_Map (dust / grime / rubber) - effects, lights.rs
                paint_tex_entry(16),
                paint_tex_entry(17),
            ];
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material bgl"),
            entries: &layout_entries,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("aniso"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            // 16x anisotropy: car paint is seen at grazing angles constantly,
            // and this is where most of the perceived sharpness comes from.
            anisotropy_clamp: 16,
            ..Default::default()
        });

        let white = solid(device, queue, [255, 255, 255, 255]);
        let flat_normal = solid(device, queue, [128, 128, 255, 255]);
        let black = solid(device, queue, [0, 0, 0, 255]);
        // splat-layer colour / roughness maps, one array each (layers.rs)
        let mut arrays = crate::layers::LayerArrays::new(device, queue);
        let white_arr = arrays.white_view();
        let mut n_layer_maps = [0u32; 3];
        let mut n_base_off = 0u32;

        let mut groups = Vec::new();
        let mut by_name = HashMap::new();
        let (mut n_rough, mut n_redrough, mut n_gao, mut n_coat) = (0u32, 0u32, 0u32, 0u32);
        let mut n_int = 0u32;
        let mut missing_albedo: Vec<String> = Vec::new();
        // Every texture index this track binds as somebody's base colour.
        let used_albedo: std::collections::HashSet<usize> = car
            .materials
            .iter()
            .filter_map(|m| {
                m.slot(&[
                    "Base_BaseColorMap", "BaseColorMap", "txDiffuse",
                    "DiffuseMap", "AlbedoMap", "ksDiffuse", "Red_BaseColorMap",
                    "Green_BaseColorMap", "Blue_BaseColorMap",
                ])
            })
            .collect();
        let mut by_path: HashMap<String, usize> = HashMap::new();
        let mut blend: Vec<Blend> = Vec::new();
        let mut buffers: Vec<wgpu::Buffer> = Vec::new();
        let mut uniforms: Vec<MaterialUniform> = Vec::new();
        let mut light_srcs: Vec<Option<crate::lights::LightSrc>> = Vec::new();
        let light_state = crate::lights::initial_state();
        let car_fx = car.fx.as_deref();
        // dash screens render into these (fxgpu.rs), bound as t_emis
        let mut dash_targets: Vec<(usize, wgpu::Texture)> = Vec::new();
        // Paint detail maps are SHARED (flakes_NM sits on 60+ materials of one
        // car), so upload each once rather than once per material.
        let mut detail_cache: HashMap<usize, wgpu::TextureView> = HashMap::new();
        let (mut n_pd_n, mut n_pd_cc, mut n_pd_ao) = (0u32, 0u32, 0u32);

        // Track materials are shaded by the track pipeline (track.rs), which
        // uploads their textures itself; here they only need a group for the
        // shadow pass (depth only, no fragment), so skip their textures.
        let track_owned = crate::track::owned_material_count(car);
        let placeholder_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("track placeholder"),
            // never shaded (depth-only shadow pass), so all-zero is fine
            contents: bytemuck::bytes_of(&<MaterialUniform as bytemuck::Zeroable>::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        for (mat_i, m) in car.materials.iter().enumerate() {
            crate::strict::mute(is_driver(&m.path.to_lowercase())
                || m.shader.as_deref() == Some(crate::contact::SHADER));
            if mat_i < track_owned {
                let alpha = matches!(m.param(&["blendMode"], 0.0) as i32, 1 | 3)
                    || m.shader.as_deref().map_or(false, |s| s.to_lowercase().contains("glass"));
                blend.push(if alpha { Blend::Alpha } else { Blend::Opaque });
                // One entry per layout slot, whatever the layout grows to:
                // buffers get the zero uniform, samplers the shared sampler,
                // textures white (their contents never matter here).
                let entries: Vec<wgpu::BindGroupEntry> = layout_entries
                    .iter()
                    .map(|e| wgpu::BindGroupEntry {
                        binding: e.binding,
                        resource: match e.ty {
                            wgpu::BindingType::Buffer { .. } => placeholder_buf.as_entire_binding(),
                            wgpu::BindingType::Sampler(_) => wgpu::BindingResource::Sampler(&sampler),
                            wgpu::BindingType::Texture {
                                view_dimension: wgpu::TextureViewDimension::D2Array, ..
                            } => wgpu::BindingResource::TextureView(&white_arr),
                            _ => wgpu::BindingResource::TextureView(&white),
                        },
                    })
                    .collect();
                groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&m.file),
                    layout: &layout,
                    entries: &entries,
                }));
                // keep the per-group arrays parallel to `groups`
                buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("track placeholder"),
                    size: std::mem::size_of::<MaterialUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: false,
                }));
                uniforms.push(bytemuck::Zeroable::zeroed());
                light_srcs.push(None);
                by_name.insert(key_of(&m.file), groups.len() - 1);
                if !m.path.is_empty() {
                    by_path.insert(m.path.clone(), groups.len() - 1);
                }
                continue;
            }
            let mut albedo_i = m.slot(&[
                "Base_BaseColorMap", "BaseColorMap", "txDiffuse",
                "DiffuseMap", "AlbedoMap", "ksDiffuse",
            ]);
            // ⚠ The Uber vehicle shaders sample Base_BaseColorMap only when
            // Base_HasBaseColorMap is set (UberVehicleMaterial DXIL 0x035a3710:
            // `icmp eq Base_HasBaseColorMap, 0` skips the sample; cbuffer
            // offset 316). A bound map with the flag off is the flat
            // Base_Basecolor. Third-party converters lean on it: the AZ-1
            // LB40 kit binds the AC1 diffuse with the flag off and a blue
            // Base_Basecolor - the game draws it blue, we drew the AC1 skin.
            // (An empty value is 0 and the parser drops it, so absent = off.)
            if albedo_i.is_some()
                && albedo_i == m.slot(&["Base_BaseColorMap"])
                && m.shader.as_deref().map_or(false, |s| s.to_lowercase().starts_with("ubervehicle"))
                && m.param(&["Base_HasBaseColorMap"], 0.0) < 0.5
            {
                albedo_i = None;
                n_base_off += 1;
                if crate::envc::var_os("EVOVIEW_MAT_LOG").is_some() { eprintln!("base map off: {}", m.file); }
            }
            let normal_i = m.slot(&[
                "Base_NormalMap", "NormalMap", "txNormal", "Normal",
            ]);
            let rough_i = m.slot(&[
                "RoughnessMap", "Base_RoughnessMap", "txRoughness",
                "HasRoughnessMap",
            ]);
            // ⚠ The glass shaders (VehicleGlass / VehicleGlassInterior /
            // VehicleWindow) sample txDiffuse, NormalMap and RoughnessMap, each
            // only behind its Has flag; stock glass also carries Base_* slots
            // the shader never reads (SPEC_small §2).
            let sh_l = m.shader.as_deref().unwrap_or("").to_lowercase();
            let glass_tpl = matches!(sh_l.as_str(), "vehicleglass" | "vehicleglassinterior" | "vehiclewindow");
            let gate = |flag: &str, slot: &str| if m.param(&[flag], 0.0) > 0.5 { m.slot(&[slot]) } else { None };
            let (mut albedo_i, normal_i, rough_i) = if glass_tpl {
                (gate("HasBaseColorMap", "txDiffuse"), gate("HasNormalMap", "NormalMap"), gate("HasRoughnessMap", "RoughnessMap"))
            } else {
                (albedo_i, normal_i, rough_i)
            };
            let ao_i = m.slot(&["Base_AmbientOcclusionMap", "AmbientOcclusionMap", "txAO"]);
            // EVO ships per-layer data maps; we did not bind them, so every
            // layer shared one scalar roughness.
            // the layers' roughness maps (binding 7, an array): layer_maps below
            let rough_layer_i = [layer_rough_map(m, "Red"), layer_rough_map(m, "Green"),
                                 layer_rough_map(m, "Blue")];
            let gao_i = m.slot(&["GlobalAo_AmbientOcclusionMap"]);
            let coat_map_i = m.slot(&["Base_ClearCoatMap"]);
            let opacity_i = m.slot(&["Base_OpacityMap", "OpacityMap", "txOpacity"]);
            let detail_i = m.slot(&["txDetail", "Detail_BaseColorMap", "DetailMap"]);
            // Tracks prefer a roughness map over AO: asphalt without it is clay.
            // But a car ships BOTH, and this slot only holds one - so roughness won
            // and the AO map was dropped on the floor. Now roughness has its own
            // binding (6), so when both exist this slot can carry the AO and
            // nothing is lost. Rough-only materials (tracks) are unchanged.
            let data_i = if rough_i.is_some() && ao_i.is_some() { ao_i } else { rough_i.or(ao_i) };
            let data_is_rough = data_i.is_some() && ao_i.is_none();

            // colour goes through sRGB, data maps do not
            // ⚠ A texture the material names but that is not in the archive
            // used to fall through to a WHITE 1x1 with `factors[2]` still 1.0,
            // so the material rendered flat white with no shading cue that
            // anything was wrong. Roll-bar padding, seat fabric and steering
            // rims all share `t_fabric_archliner_colour`, so one missing file
            // turned several parts of an interior pure white at once.
            let albedo_tex = albedo_i.and_then(|i| upload(device, queue, car, i, true));
            let has_albedo_tex = albedo_tex.is_some();
            let albedo_missing = albedo_i.is_some() && albedo_tex.is_none();
            if albedo_missing {
                missing_albedo.push(format!("{} ({})", m.file, alb_name_of(car, albedo_i)));
                crate::strict::note("T11", &m.file, alb_name_of(car, albedo_i));
            }
            let albedo = albedo_tex
                .unwrap_or_else(|| solid(device, queue, [255, 255, 255, 255]));
            let normal_tex = normal_i.and_then(|i| upload(device, queue, car, i, false));
            if normal_i.is_some() && normal_tex.is_none() {
                crate::strict::note("T12", &m.file, format!("normal map {}", alb_name_of(car, normal_i)));
            }
            let normal = normal_tex.unwrap_or_else(|| solid(device, queue, [128, 128, 255, 255]));
            let ao_tex = data_i.and_then(|i| upload(device, queue, car, i, false));
            if data_i.is_some() && ao_tex.is_none() {
                crate::strict::note("T12", &m.file, format!("AO / data map {}", alb_name_of(car, data_i)));
            }
            let ao = ao_tex.unwrap_or_else(|| solid(device, queue, [255, 255, 255, 255]));
            // A dedicated roughness map, separate from `data`. `data_i` collapses
            // roughness and AO into one slot (fine for tracks, wrong for cars,
            // which ship both).
            let rough_tex = rough_i.and_then(|i| upload(device, queue, car, i, false));
            let gao_tex = gao_i.and_then(|i| upload(device, queue, car, i, false));
            let coat_tex = coat_map_i.and_then(|i| upload(device, queue, car, i, false));
            for (i, t, what) in [(rough_i, rough_tex.is_some(), "roughness"), (gao_i, gao_tex.is_some(), "global AO"),
                                 (coat_map_i, coat_tex.is_some(), "clear coat")] {
                if i.is_some() && !t {
                    crate::strict::note("T13", &m.file, format!("{what} map {}", alb_name_of(car, i)));
                }
            }
            let has_rough_map = rough_tex.is_some();
            let maps = [
                if has_rough_map { 1.0 } else { 0.0 },
                // (was the Red layer's roughness map; the layer maps moved to
                // the binding-7 array, see layer_xo)
                0.0,
                if gao_tex.is_some() {
                    m.param(&["GlobalAo_Strength", "GlobalAo_Intensity"], 1.0).clamp(0.0, 4.0)
                } else {
                    0.0
                },
                if coat_tex.is_some() { 1.0 } else { 0.0 },
            ];
            let low_name = m.file.to_lowercase();
            if is_cabin(&low_name) {
                n_int += 1;
                if n_int <= 12 { println!("  cabin material: {}", m.file); }
            }
            if maps[0] > 0.0 { n_rough += 1; }
            if rough_layer_i[0].is_some() { n_redrough += 1; }
            if maps[2] > 0.0 { n_gao += 1; }
            if maps[3] > 0.0 { n_coat += 1; }
            let t_rough = rough_tex.unwrap_or_else(|| solid(device, queue, [255, 255, 255, 255]));
            let t_gao = gao_tex.unwrap_or_else(|| solid(device, queue, [255, 255, 255, 255]));
            let t_coat = coat_tex.unwrap_or_else(|| solid(device, queue, [255, 255, 255, 255]));
            let pd = paint_detail_maps(m);
            for i in [pd.n_r, pd.n_gb, pd.cc_n, pd.ao].into_iter().flatten() {
                if !detail_cache.contains_key(&i) {
                    if let Some(v) = upload(device, queue, car, i, false) {
                        detail_cache.insert(i, v);
                    }
                }
            }
            if pd.n_r.is_some() || pd.n_gb.is_some() { n_pd_n += 1; }
            if pd.cc_n.is_some() { n_pd_cc += 1; }
            if pd.ao.is_some() { n_pd_ao += 1; }
            let shader_l = m.shader.clone().unwrap_or_default().to_lowercase();
            let fname = m.file.to_lowercase();
            // ⚠ Terrain is a TRACK idea. On a car the same name keywords hit
            // real parts (`fari` = lamps, `ext_mech_mount`), which then got a
            // tiled grass photo or a green base.
            let is_car = car.track_env.is_none() && car.fx.is_some();
            let terrain = is_terrain_ground(&fname, &shader_l);
            if terrain && is_car {
                crate::strict::note("S4", &m.file, "name looks like terrain");
            }
            let terrain = terrain && !is_car;
            if is_car && !shader_l.is_empty() && !CAR_SHADERS.contains(&shader_l.as_str()) {
                crate::strict::note("S1", &m.file, m.shader.clone().unwrap_or_default());
            }
            // Only world-project when the material asks. Forcing it on every
            // UberBiplanar surface smeared UV-authored asphalt and hills.
            let triplanar = m.param(&["Base_Triplanar"], 0.0) > 0.5;
            let mut uv_s = if triplanar {
                m.color(&["Base_TriplanarScale"], [0.08, 0.08, 1.0, 0.0])[0]
                    .abs()
                    .max(0.01)
            } else {
                0.0
            };
            // DynamicTrack's real layers are runtime-only. The fallback
            // asphalt tile is a repeating photo — world-project it so the
            // ribbon is not a UV-smeared white unwrap.
            if shader_l.contains("dynamictrack") && uv_s == 0.0 {
                uv_s = 0.055;
            }
            let mut uv_tiling = m.color(&["Base_UVscale", "UVscale"], [1.0, 1.0, 0.0, 0.0]);
            // Base_UVscale is a 2-vector, which `color` drops. Honour it where
            // one tiling is right for everything sample2 tiles: a base colour
            // map with no AO / roughness map (those ride their own
            // Base_UVscalePbr in the game) and a normal map, if any, at the
            // same Base_UVscaleNormal. That is the AC1 converter's detail
            // layer - the RX-8 suit's fabric at 20x, drawn 1x as big grey
            // blotches. The three stock materials with a tiled base all bind
            // baked AO, so stock cars are untouched.
            // ⚠ An opacity-only part tiles too: the MX5 ND Cup's front grille
            // net (ext_grid_2) has no colour map, Grid_Net_3_AT as its opacity
            // at Base_UVscale [13, 36] and the matching normal. Sampled 1x, one
            // stretched mesh cell covered the opening, mostly hole - you saw
            // straight through the bumper. The game samples base colour,
            // opacity and (without Base_CustomNormalUV) the normal in the
            // same base UV.
            if shader_l.contains("ubervehicle") && (albedo_i.is_some() || opacity_i.is_some())
                && ao_i.is_none() && rough_i.is_none()
            {
                if let Some(b) = vec2_param(m, "Base_UVscale") {
                    let normal_same = m.param(&["Base_CustomNormalUV"], 0.0) < 0.5
                        || vec2_param(m, "Base_UVscaleNormal") == Some(b);
                    if normal_i.is_none() || normal_same {
                        uv_tiling[0] = b[0];
                        uv_tiling[1] = b[1];
                    }
                }
            }
            if !m.has(&["Roughness", "Base_Roughness", "ksRoughness"]) && !shader_l.contains("glass") {
                crate::strict::note("P5", &m.file, "no Roughness");
            }
            let default_rough = if shader_l.contains("biplanar")
                || shader_l.contains("defaultmaterial")
                || shader_l.contains("fence")
            {
                0.78
            } else {
                0.45
            };

            // A bound albedo already carries the colour. Fence materials store
            // a near-black Base_Basecolor used as a subsurface tint - using it
            // as the albedo turns every plant black.
            // ⚠ Shared *neutral* detail maps must be tinted by the material.
            // `detail_heatshield_colour`, `detail_styrofoam_colour`,
            // `detail_noise_mechanical_m` and `t_fabric_archliner_colour` live
            // in common_assets and are authored grey on purpose - each
            // material multiplies them by its own `Base_Basecolor`. Sampling
            // them untinted is why roll-bar padding, wheel-arch liners and
            // underbody rendered flat white.
            //
            // It has to stay narrow. `Base_Basecolor` is present on nearly
            // every material, usually at EVO's unset default (0.41698843), and
            // the car's OWN authored atlases (`ext_skin_c`, `int_elec_c`,
            // `detail_wood_2_c` - the `_c` suffix means colour) already carry
            // the colour. Tinting those would darken the whole car by half.
            let alb_is_neutral_detail = {
                let n = albedo_i
                    .and_then(|i| car.textures.get(i))
                    .map(|t| t.name.to_lowercase())
                    .unwrap_or_default();
                (n.starts_with("detail_") || n.starts_with("t_")) && !n.ends_with("_c")
            };
            if albedo_i.is_some() && !albedo_missing && alb_is_neutral_detail && !shader_l.contains("fence") {
                crate::strict::note("P4", &m.file, alb_name_of(car, albedo_i));
            }
            if (albedo_i.is_none() || albedo_missing) && !shader_l.contains("glass") && !shader_l.contains("windshield")
                && !m.has(&["Base_Basecolor", "ksBaseColor", "BaseColor"]) {
                crate::strict::note("P3", &m.file, "no base colour");
            }
            if (shader_l.contains("glass") || shader_l.contains("windshield")) && (albedo_i.is_none() || albedo_missing)
                && !m.has(&["Basecolor", "Base_Basecolor", "ksBaseColor"]) {
                crate::strict::note("A5", &m.file, "no tint colour");
            }
            let mut base = if albedo_i.is_some() && !albedo_missing {
                let tint = if alb_is_neutral_detail && !shader_l.contains("fence") {
                    m.color(&["Basecolor", "Base_Basecolor"], [1.0, 1.0, 1.0, 1.0])
                } else {
                    m.color(&["Basecolor"], [1.0, 1.0, 1.0, 1.0])
                };
                tint
            } else if shader_l.contains("glass") || shader_l.contains("windshield") {
                // Glass shaders (VehicleGlass*, VehicleWindshield) keep their
                // tint in `Basecolor` - (0.45, 0.56, 0.58, a 0.235) on the M4's
                // windows. Falling through to ksBaseColor gave near-black
                // (0.03) glass that hid the cabin.
                m.color(&["Basecolor", "Base_Basecolor", "ksBaseColor"],
                       [0.5, 0.5, 0.5, 0.25])
            } else {
                m.color(&["Base_Basecolor", "ksBaseColor", "BaseColor"],
                       [0.62, 0.63, 0.66, 1.0])
            };
            // "SpatMap" is the engine's own spelling.
            // A LED decal sheet (VehicleLed: gauge faces, switch legends) ships a
            // SpatMap AND a Base_OpacityMap; the one extra binding took the
            // splat, so the opacity was dropped and the whole sheet drew opaque
            // over the gauges in its layer tints (the MX-5 NA's orange dials).
            // Its cut-out matters more than the tint: keep the opacity.
            let splat_i = m.slot(&["SpatMap", "SplatMap"])
                .filter(|_| !(shader_l == "vehicleled" && opacity_i.is_some()));
            let mut alb_name = albedo_i
                .and_then(|i| car.textures.get(i))
                .map(|t| t.name.to_lowercase())
                .unwrap_or_default();
            let mut alb_is_placeholder = albedo_missing
                || alb_name.contains("physics")
                || alb_name.contains("1color")
                || is_white_placeholder(&alb_name)
                || alb_name == "null"
                || alb_name.is_empty();
            // UberBiplanar ground (Spa grass.material, grass_brd, …) has no
            // SpatMap. The authored look is the Red layer, often triplanar —
            // Splatmap_Constant is (1, ·) so the whole surface is that tile.
            // Sampling only Base_BaseColorMap with the mesh unwrap stretches
            // a 512px grass photo over 500 m and the hillside reads as white.
            let red_i = m.slot(&["Red_BaseColorMap"]);
            let red_on = m.param(&["Red_Enable"], 0.0) > 0.5;
            let red_trip = m.param(&["Red_Triplanar"], 0.0) > 0.5;
            // Same test paint_splat_constant makes below (nothing else in the
            // splat slot), so the shader's layer path is sure to run.
            let splat_path = splat_i.is_some()
                || (m.param(&["Splatmap_UseConstant"], 0.0) > 0.5
                    && opacity_i.is_none() && detail_i.is_none());
            // Each splat layer's own colour and roughness maps (layer_map /
            // layer_rough_map), bound as two arrays (layers.rs).
            let col_tex: Vec<Option<std::rc::Rc<wgpu::Texture>>> = ["Red", "Green", "Blue"]
                .iter()
                .map(|l| layer_map(m, l, &shader_l, &fname, splat_path)
                    .and_then(|i| upload_rc(device, queue, car, i, true)))
                .collect();
            let rough_tex: Vec<Option<std::rc::Rc<wgpu::Texture>>> = rough_layer_i
                .iter()
                .map(|i| if splat_path || splat_i.is_some() {
                    i.and_then(|i| upload_rc(device, queue, car, i, false))
                } else {
                    None
                })
                .collect();
            let red_layer_tex = col_tex[0].clone();
            if splat_i.is_none()
                && red_layer_tex.is_none()
                && red_on
                && red_i.is_some()
                && (alb_is_placeholder || red_trip || terrain)
            {
                albedo_i = red_i;
                alb_name = albedo_i
                    .and_then(|i| car.textures.get(i))
                    .map(|t| t.name.to_lowercase())
                    .unwrap_or_default();
                alb_is_placeholder = false;
                // ⚠ Forcing white here is right for a tiled ground photo and
                // WRONG for a car. Cars reach this path too: roll-bar padding,
                // wheel-arch liner, underbody and steering rims bind a shared
                // NEUTRAL detail map (`detail_heatshield_colour`,
                // `t_fabric_archliner_colour`, ...) on `Red_BaseColorMap` with
                // no base albedo at all. Those maps are authored grey on
                // purpose - the material supplies the colour - so sampling
                // them at full white is what made padding and liners render as
                // flat white patches inside otherwise-black interiors.
                //
                // With no SpatMap there is nothing to blend the Red layer in
                // WITH, so what survives is the base layer: `Base_Basecolor`
                // is the colour, and the Red map only supplies the detail.
                // `ext_mech_foam_black` settles it - the material is named
                // black, its `Base_Basecolor` is 0.133 and its `Red_Basecolor`
                // is 0.355. Fall back to the Red tint, then to white, for
                // materials that leave the base colour unset.
                base = if terrain {
                    [1.0, 1.0, 1.0, 1.0]
                } else {
                    let mut b = m.color(&["Base_Basecolor", "Red_Basecolor", "Red_BaseColor"],
                                        [1.0, 1.0, 1.0, 1.0]);
                    // With `Splatmap_UseConstant` the whole part sits on the
                    // Red layer, and the Red colour then applies through its
                    // override mode. The M4 CSL's seats are the case:
                    // Base_Basecolor is 1.34 (a multiplier, not a colour) and
                    // Red_Basecolor is 0.061 with RedOverrideMultiply_BaseColor
                    // = 1, i.e. black alcantara. Taking the base alone
                    // rendered them white. Red-stitched variants use the same
                    // pattern with a red Red_Basecolor.
                    let w = if m.param(&["Splatmap_UseConstant"], 0.0) > 0.5 {
                        m.color(&["Splatmap_Constant"], [0.0; 4])[0]
                            .max(m.param(&["Splatmap_Constant"], 0.0))
                            .clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    if w > 0.0 && m.color(&["Red_Basecolor"], [-1.0; 4])[0] >= 0.0 {
                        let r = m.color(&["Red_Basecolor"], [1.0; 4]);
                        let mul = m.param(&["RedOverrideMultiply_BaseColor"], 0.0) > 0.5;
                        let rep = m.param(&["RedOverrideReplace_BaseColor"], 0.0) > 0.5;
                        for c in 0..3 {
                            if mul {
                                b[c] *= 1.0 + (r[c] - 1.0) * w;
                            } else if rep {
                                b[c] += (r[c] - b[c]) * w;
                            }
                        }
                    }
                    b
                };
                if red_trip && uv_s == 0.0 {
                    uv_s = m.color(&["Red_TriplanarScale"], [0.1, 0.1, 0.1, 0.0])[0]
                        .abs()
                        .max(0.02);
                }
            }
            // A baked unique unwrap (grass_far, map3, outerland) needs a
            // tiled grass photo from THIS track. A material that already
            // points at grass_cut / grass_base / grass.texture is left alone
            // so Nürburg and Miami keep their authored look.
            // Unique unwraps (terrain_4k, top_far, ext-top, map3) are one
            // photo stretched over kilometres. If we sample them with mesh
            // UVs they read as a white/snow skirt (Laguna far, Watkins pad).
            // Tile any terrain albedo that is not already a repeating ground
            // photo.
            let need_tile = terrain
                && splat_i.is_none()
                && (alb_is_placeholder
                    || !albedo_looks_tiled_ground(&alb_name)
                    || albedo_needs_world_tile(&alb_name, &fname)
                    || fname.contains("top-ext")
                    || fname.contains("far"));
            // Track UberBiplanar splat layers are TEXTURES (grass_2, gravel,
            // forest floor). The car splat path only blends the layer
            // *colours* — Red/Green Basecolor is often (1,1,1) or (2,2,2) —
            // so new_top / sbanc / top-ext paint the whole hillside white.
            let ground_layer = {
                let looks = |i: Option<usize>| {
                    i.and_then(|j| car.textures.get(j))
                        .map(|t| t.name.to_lowercase())
                        .filter(|n| {
                            n.contains("grass")
                                || n.contains("forest")
                                || n.contains("erba")
                                || n.contains("dirt")
                                || n.contains("soil")
                                || n.contains("gravel")
                                || n.contains("sand")
                                || n.contains("ground")
                        })
                        .is_some()
                };
                let green_i = m.slot(&["Green_BaseColorMap"]);
                let blue_i = m.slot(&["Blue_BaseColorMap"]);
                if m.param(&["Green_Enable"], 0.0) > 0.5 && looks(green_i) {
                    green_i
                } else if red_on && looks(red_i) {
                    red_i
                } else if m.param(&["Blue_Enable"], 0.0) > 0.5 && looks(blue_i) {
                    blue_i
                } else {
                    None
                }
            };
            // The "white pad" signature: every splat layer left at the unset
            // white, so blending them produces nothing but white.
            let splat_layers_degenerate = {
                let l = |n: &str| {
                    m.color(&[&format!("{n}_Basecolor"), &format!("{n}_BaseColor")],
                            [1.0, 1.0, 1.0, 1.0])
                };
                [l("Red"), l("Green"), l("Blue")]
                    .iter()
                    .all(|c| c[0].min(c[1]).min(c[2]) > 0.95)
            };
            let track_splat_ground = splat_i.is_some() && ground_layer.is_some() && terrain;
            let grass_tile = if track_splat_ground {
                ground_layer
            } else if need_tile {
                pick_ground_tile(car, albedo_i, &fname, &used_albedo)
            } else if terrain
                && splat_i.is_some()
                && splat_layers_degenerate
                && !albedo_looks_tiled_ground(&alb_name)
            {
                // Baked unwrap + splat colours of (1,1,1) is the white pad
                // on Laguna / COTA / Watkins. Tile a real ground photo.
                //
                // ⚠ Both guards matter. Without them this fired on EVERY splat
                // terrain and threw a library tile over surfaces that already
                // had a perfectly good authored one - which is what painted
                // Brands Hatch yellow: `grass_ext` has a SpatMap and a real
                // `grass_ext_base_b2` albedo, and got a dry grass photo
                // stamped over the whole circuit.
                pick_ground_tile(car, albedo_i, &fname, &used_albedo)
            } else {
                None
            };
            if grass_tile.is_some() {
                if !track_splat_ground {
                    base = [0.42, 0.58, 0.28, 1.0];
                }
                println!(
                    "terrain tile {}: albedo={} -> {} (#{})",
                    m.file,
                    alb_name,
                    grass_tile.and_then(|i| car.textures.get(i))
                        .map(|t| t.name.clone()).unwrap_or_default(),
                    grass_tile.unwrap()
                );
            } else if terrain && alb_is_placeholder {
                base = [0.38, 0.52, 0.24, 1.0];
            }
            if terrain || track_splat_ground {
                base[0] = base[0].min(1.0);
                base[1] = base[1].min(1.0);
                base[2] = base[2].min(1.0);
            }
            let extra_i = grass_tile.or(if track_splat_ground { None } else { splat_i })
                .or(opacity_i)
                .or(detail_i);
            // colour (sRGB) only when it IS the grass tile or the detail
            // map: with both an opacity and a detail map the opacity map
            // wins the slot and must stay linear
            let extra_srgb = extra_i.is_some()
                && (grass_tile.is_some()
                    || (splat_i.is_none() && opacity_i.is_none() && detail_i.is_some()));
            // `Splatmap_UseConstant`: a car part with no SpatMap still blends
            // its layers, with fixed weights from `Splatmap_Constant`. Every
            // ext_mech_* material on the M2 works this way (chrome, copper,
            // brushed, rubber), so ignoring it rendered them as the Base
            // fallback only. A 1x1 texture of the weights keeps the shader's
            // splat path unchanged.
            let splat_const = paint_splat_constant(m, splat_i.is_some() || extra_i.is_some());
            let splat = match splat_const {
                Some(w) => solid(device, queue, w),
                None => extra_i
                    .and_then(|i| upload(device, queue, car, i, extra_srgb))
                    .unwrap_or_else(|| solid(device, queue, [0, 0, 0, 255])),
            };
            let _has_opacity_map = splat_i.is_none() && opacity_i.is_some();
            // misc.w: 0 none, 1 opacity-in-splat, 2 world-space terrain tile,
            // 3 mesh-UV detail multiply (roads).
            let extra_mode = if grass_tile.is_some() {
                2.0
            } else if splat_i.is_none() && opacity_i.is_some() {
                1.0
            } else if splat_i.is_none() && detail_i.is_some() && uv_s == 0.0 {
                3.0
            } else {
                0.0
            };
            let detail_scale = if extra_mode > 1.5 && extra_mode < 2.5 {
                // TerrainRenderSettings.world_uv_scale: values > 1 are
                // metres-per-tile, values in (0,1] are already UV-per-metre.
                if car.world_uv_scale > 1.0 {
                    1.0 / car.world_uv_scale
                } else if car.world_uv_scale > 0.0 {
                    car.world_uv_scale
                } else {
                    0.085
                }
            } else if extra_mode > 2.5 {
                let d = m.color(&["detailMultiplier"], [8.0, 8.0, 0.0, 0.0]);
                d[0].abs().max(1.0)
            } else {
                0.0
            };
            let layer = |name: &str, fallback: [f32; 4]| -> [f32; 4] {
                if m.param(&[&format!("{name}_Enable")], 0.0) < 0.5 {
                    return [fallback[0], fallback[1], fallback[2], -1.0];
                }
                let c = m.color(&[&format!("{name}_Basecolor"),
                                  &format!("{name}_BaseColor")], fallback);
                // How the layer's colour combines with what is under it
                // (UberVehicle): <Layer>OverrideReplace_BaseColor replaces it,
                // <Layer>OverrideMultiply_BaseColor multiplies it, and with
                // NEITHER set the layer leaves the colour alone. That last case
                // is what liveries rely on: the Huracan ST EVO2 and M4 GT3 skins
                // put the whole livery in Base_BaseColorMap with a dark
                // Red_Basecolor and neither flag; treating it as replace (or
                // multiply) painted the livery over in flat green/grey.
                // Encoded for the shader (paint_layer_col): positive = replace,
                // negative = multiply by |c|, (-1,-1,-1) = no effect.
                // VehicleRim layers the same way: the SF-25's Monza rim has
                // its gold in Base_BaseColorMap and a grey Green layer that
                // only replaces METALNESS - replacing the colour too turned
                // the game's gold rims grey.
                let vehicle = shader_l.contains("ubervehicle") || shader_l == "vehiclerim";
                let rep = m.param(&[&format!("{name}OverrideReplace_BaseColor")], 0.0) > 0.5;
                let mul = m.param(&[&format!("{name}OverrideMultiply_BaseColor")], 0.0) > 0.5;
                let (c, k) = if !vehicle || rep {
                    (c, 1.0)
                } else if mul {
                    (c, -1.0)
                } else {
                    ([1.0, 1.0, 1.0, 1.0], -1.0)
                };
                [c[0].max(0.0) * k, c[1].max(0.0) * k, c[2].max(0.0) * k,
                 m.param(&[&format!("{name}_Roughness")], 0.4)]
            };
            // ⚠ Strict: the SHADER decides. A converted `ext_glass_black` on
            // UberVehicleMaterial is a solid black part in the game; the name
            // test drew it as see-through glass here.
            let strict_on = crate::strict::on();
            let is_glass = shader_l.contains("glass") || shader_l.contains("windshield") || shader_l == "vehiclewindow"
                || (!strict_on && (fname.contains("glass") || fname.contains("transp")));
            let is_fence = shader_l.contains("fence");
            // ⚠ EVO uses more blend modes than 0/1/2, and everything else was
            // falling through to OPAQUE. Mode 4 is the alpha-card mode - the
            // background forest, billboards, `*_alpha` props - 43 materials
            // across four tracks. Drawn opaque, every tree card rendered its
            // transparent region as a dark polygon, which is the hard-edged
            // halo around each canopy. Mode 3 is translucent (simpleglasslod).
            let bm = m.param(&["blendMode"], 0.0) as i32;
            if !(1..=4).contains(&bm) && (is_glass || is_fence || opacity_i.is_some()) {
                crate::strict::note("A1", &m.file, format!("blendMode {bm} drawn transparent"));
            }
            if is_glass && !(shader_l.contains("glass") || shader_l.contains("windshield")) {
                crate::strict::note("A2", &m.file, m.shader.clone().unwrap_or_default());
            }
            let mode = match bm {
                1 => Blend::Alpha,
                2 => Blend::Cutout,
                3 => Blend::Alpha,
                4 => Blend::Cutout,
                // strict: blendMode 0 is opaque, whatever maps it carries
                _ if !strict_on && (is_glass || is_fence || opacity_i.is_some()) => Blend::Alpha,
                _ => Blend::Opaque,
            };
            // EVO carries its own alpha-test threshold. Ours was hardcoded at
            // 0.5, which eats the thin edges of a leaf card that was authored
            // against 0.15. A cutoff of 0 means the material is not actually
            // alpha-to-coverage, so fall back to the old default there.
            let a2c = m.param(&["a2cCutoff"], 0.0);
            // The engine's cutout (SPEC_small §1): coverage =
            // saturate(scale * (alpha - cutoff)) from the material's top-level
            // fields (2.0 / 0.2 on every car checked). One hard threshold
            // stands in for that ramp at its 50% point.
            let (sc, co) = (m.param(&["__a2c_scale"], 0.0), m.param(&["__a2c_cutoff"], -1.0));
            let cutoff = if co >= 0.0 && sc > 0.0 {
                (co + 0.5 / sc).clamp(0.02, 0.98)
            } else if a2c > 0.01 {
                a2c.clamp(0.02, 0.9)
            } else {
                0.5
            };
            if mode == Blend::Cutout && co < 0.0 && a2c <= 0.01 {
                crate::strict::note("A3", &m.file, "no cutout fields");
            }
            // Blended glass is EVO's tinted transmission, not alpha-over. An
            // OPAQUE glass material (blendMode 0, e.g. a black lens surround)
            // stays opaque and is shaded like any other surface.
            let glass_blend = is_glass && mode == Blend::Alpha;
            let mode = if glass_blend { Blend::Glass } else { mode };
            blend.push(mode);
            // Two diagnostics that between them identify which material owns a
            // pixel, with no guessing. MAT_HL paints matching materials
            // magenta - probe by ASSIGNMENT, not addition, so "no visible
            // change" cannot be mistaken for "the flag never set".
            let mut hl_override = false;
            if let Ok(hl) = crate::envc::var("EVOVIEW_MAT_HL") {
                // Comma-separated, so a bisect can highlight half the material
                // list at once and find which one owns a pixel in log2(n)
                // renders instead of n.
                // A token ending in `.material` is matched EXACTLY. Substring
                // matching is convenient ("int_") but useless for a bisect:
                // "grass" also selects grass_ext, grass2_b and grass_block, so
                // the search converges on a name that is not the real owner.
                if hl.to_lowercase().split(',').any(|w| {
                    let w = w.trim();
                    if w.is_empty() {
                        false
                    } else if let Some(exact) = w.strip_suffix(".material") {
                        fname.trim_end_matches(".material") == exact
                    } else {
                        fname.contains(w)
                    }
                }) {
                    base = [1.0, 0.0, 1.0, base[3]];
                    // ⚠ Also switch the texture OFF. Tinting base alone gets
                    // multiplied by the albedo, so a yellow terrain highlighted
                    // magenta comes out red and a bisect looking for magenta
                    // never finds it.
                    hl_override = true;
                }
            }
            if crate::envc::var("EVOVIEW_MAT_DUMP").is_ok() {
                let mut slots: Vec<String> = m.slots.iter().map(|(k, v)| format!("{k}={}", alb_name_of(car, Some(*v)))).collect();
                slots.sort();
                let mut cols: Vec<(&String, &Vec<f32>)> = m.colors.iter().collect();
                cols.sort_by_key(|x| x.0);
                let mut prm: Vec<(&String, &f32)> = m.params.iter().collect();
                prm.sort_by_key(|x| x.0);
                eprintln!(
                    "DUMP	{}	shader={}	textured={}	slots={:?}	colors={:?}	params={:?}",
                    m.file,
                    m.shader.clone().unwrap_or_default(),
                    albedo_i.is_some() && !alb_is_placeholder,
                    slots,
                    cols.iter().map(|(k, v)| format!("{k}={v:.3?}")).collect::<Vec<_>>(),
                    prm.iter().map(|(k, v)| format!("{k}={v:.3}")).collect::<Vec<_>>(),
                );
            }
            if crate::envc::var("EVOVIEW_MAT_LOG").is_ok() {
                eprintln!("MAT {:<34} shader={:<26} albedo={:<26} base={:.3},{:.3},{:.3}                            textured={} Base_Basecolor={:?}",
                    m.file, m.shader.clone().unwrap_or_default(), alb_name,
                    base[0], base[1], base[2],
                    albedo_i.is_some() && !alb_is_placeholder,
                    m.color(&["Base_Basecolor"], [-1.0; 4]));
            }
            let mut _hl = hl_override;
            let uniform = MaterialUniform {
                base_color: base,
                factors: [
                    m.param(&["Roughness", "Base_Roughness", "ksRoughness"], default_rough),
                    // ⚠ "Has a metalness map" with no map bound samples the
                    // engine's default (black): the scalar multiplies zero.
                    // The 992 Cup Black Falcon livery ships Base_Metalness 1
                    // that way and came out olive chrome instead of yellow.
                    if m.param(&["Base_HasMetalnessMap"], 0.0) > 0.5
                        && m.slot(&["Base_MetalnessMap"]).is_none()
                    {
                        0.0
                    } else {
                        m.param(&["Metalness", "Base_Metalness", "ksMetallic"], 0.0)
                    },
                    if hl_override { 0.0 }
                    else if albedo_i.is_some() && !alb_is_placeholder { 1.0 } else { 0.0 },
                    if hl_override { 0.0 } else if normal_i.is_some() { 1.0 } else { 0.0 },
                ],
                flags: [
                    if data_is_rough { 2.0 } else if ao_i.is_some() { 1.0 } else { 0.0 },
                    m.param(&["ksTransparancy", "Opacity"], 1.0),
                    // Clear coat comes from the paint file when there is one.
                    // Glass carries its own smooth lobe already; rubber, cloth
                    // and unpainted trim must not get one or they turn plastic.
                    m.param(&["ClearCoat", "Detail_ClearCoatStrenght",
                              "ClearCoatStrength"], 0.0)
                        .clamp(0.0, 1.0),
                    cutoff,
                ],
                layer_r: layer("Red", base),
                layer_g: layer("Green", base),
                layer_b: layer("Blue", base),
                layer_metal: [
                    m.param(&["Red_Metalness"], 0.0),
                    m.param(&["Green_Metalness"], 0.0),
                    m.param(&["Blue_Metalness"], 0.0),
                    if (splat_i.is_some() && grass_tile.is_none()) || splat_const.is_some() { 1.0 } else { 0.0 },
                ],
                layer_coat: [
                    m.param(&["Red_ClearCoat"], 0.0),
                    m.param(&["Green_ClearCoat"], 0.0),
                    m.param(&["Blue_ClearCoat"], 0.0),
                    // w: cabin surface. EVO names every interior material
                    // `int_*`, and a cabin is a closed box the studio rig
                    // cannot reach - without its own fill it renders black and
                    // you cannot see into the car at all.
                    // The seated driver sits in that box too: lit like the
                    // seat under it, not by the outside rig through the roof.
                    if is_cabin(&fname) || is_driver(&m.path) { 1.0 } else { 0.0 },
                ],
                misc: [
                    if glass_blend { 1.0 } else { 0.0 },
                    // y: 1 = hard cutout at flags.w; >= 2 = the game's
                    // dithered coverage at that opacity scale (shader.wgsl
                    // cutout_drop; EVOVIEW_CUTOUT_DITHER=0 the old threshold)
                    if mode != Blend::Cutout { 0.0 }
                    else if crate::envc::var("EVOVIEW_CUTOUT_DITHER").map_or(false, |v| v == "0") { 1.0 }
                    else {
                        let s = if co >= 0.0 && sc > 0.0 { sc } else { 2.0 };
                        s.max(1.5).max(0.5 / cutoff.max(0.01)).max(0.5 / (1.0 - cutoff).max(0.01))
                    },
                    m.param(&["Base_Reflectance", "Reflectance", "ksF0Percentage"], 0.5),
                    extra_mode,
                ],
                uv_scale: [uv_s, uv_tiling[0], uv_tiling[1], detail_scale],
                maps,
                // A slot whose upload failed must read as "absent", or the
                // shader would tile a flat stand-in and change nothing - but
                // the AO strength would still darken through a white texture.
                layer_uv: pd.layer_uv,
                layer_nscale: [
                    if pd.n_r.map_or(false, |i| detail_cache.contains_key(&i)) { pd.layer_nscale[0] } else { 0.0 },
                    if pd.n_gb.map_or(false, |i| detail_cache.contains_key(&i)) { pd.layer_nscale[1] } else { 0.0 },
                    if pd.n_gb.map_or(false, |i| detail_cache.contains_key(&i)) { pd.layer_nscale[2] } else { 0.0 },
                    if pd.cc_n.map_or(false, |i| detail_cache.contains_key(&i)) { pd.layer_nscale[3] } else { 0.0 },
                ],
                layer_ccr: [
                    pd.layer_ccr[0], pd.layer_ccr[1], pd.layer_ccr[2],
                    if pd.ao.map_or(false, |i| detail_cache.contains_key(&i)) { pd.layer_ccr[3] } else { 0.0 },
                ],
                gl_em: [glass_opacity(m, has_albedo_tex),
                        // VehicleGlassInterior declares EdgeThickness but never reads it
                        if shader_l == "vehicleglassinterior" { 0.0 } else { m.param(&["EdgeThickness"], 0.0).max(0.0) },
                        0.0, glass_reflection_strength()],
                ..bytemuck::Zeroable::zeroed()
            };
            let light_src = crate::lights::source(m, car_fx);
            let mut uniform = uniform;
            // The splat layers' own maps (layer_map / layer_rough_map). A layer
            // colour keeps its sign (+ replace, - multiply); the shader
            // multiplies it by the map, so it carries only what the modulation
            // flag lets <Layer>_Basecolor contribute.
            let mut col_maps: Vec<std::rc::Rc<wgpu::Texture>> = Vec::new();
            let mut rough_maps: Vec<std::rc::Rc<wgpu::Texture>> = Vec::new();
            for (li, l) in ["Red", "Green", "Blue"].iter().enumerate() {
                let (mut xf, off, pbr, custom) = layer_uv_xf(m, l);
                if let Some(t) = &col_tex[li] {
                    let k = if m.param(&[&format!("{l}OverrideReplace_BaseColor")], 0.0) > 0.5 { 1.0 } else { -1.0 };
                    let c = if m.param(&[&format!("Modulation_{l}BaseColor")], 0.0) > 0.5 {
                        m.color(&[&format!("{l}_Basecolor"), &format!("{l}_BaseColor")], [1.0; 4])
                    } else {
                        [1.0; 4]
                    };
                    let lc = match li { 0 => &mut uniform.layer_r, 1 => &mut uniform.layer_g, _ => &mut uniform.layer_b };
                    for i in 0..3 {
                        lc[i] = c[i].max(0.0) * k;
                    }
                    xf[3] = col_maps.len() as f32;
                    col_maps.push(t.clone());
                    n_layer_maps[li] += 1;
                }
                let mut xo = [off[0], off[1], -1.0, if custom { 1.0 } else { 0.0 }];
                if let Some(t) = &rough_tex[li] {
                    xo[2] = rough_maps.len() as f32;
                    rough_maps.push(t.clone());
                }
                uniform.layer_xf[li] = xf;
                uniform.layer_xo[li] = xo;
                uniform.layer_pbr[li] = [pbr[0], pbr[1], 0.0, 0.0];
            }
            // Splatmap_Tighten: weights under TightenAmount / 2 drop to 0 (SPEC 2.6)
            uniform.layer_pbr[3][0] = if m.param(&["Splatmap_Tighten"], 0.0) > 0.5 {
                m.param(&["Splatmap_TightenAmount"], 0.0) * 0.5
            } else {
                -1.0
            };
            let layer_c_view = arrays.build(device, queue, &col_maps, true);
            let layer_r_view = arrays.build(device, queue, &rough_maps, false);
            apply_lights(&mut uniform, &light_src, light_state);
            let emis_tex = if let crate::lights::Class::Display(pi) = light_src.class {
                let pg = &car_fx.unwrap().dash[pi];
                let t = crate::fxgpu::dash_target(device, pg.width, pg.height);
                let v = t.create_view(&Default::default());
                dash_targets.push((pi, t));
                v
            } else {
                crate::lights::emissive_slot(m, car_fx)
                    .and_then(|(i, srgb)| upload(device, queue, car, i, srgb))
                    .unwrap_or_else(|| solid(device, queue, [0, 0, 0, 255]))
            };
            let mask2_tex = crate::lights::mask2_slot(m, car_fx)
                .and_then(|i| upload(device, queue, car, i, false))
                .unwrap_or_else(|| solid(device, queue, [0, 0, 0, 255]));
            // Rim motion blur (VehicleRim): z 1 = shows with speed (blur
            // rim), 2 = hides with speed (the static rim); w = VelocityGamma
            if m.shader.as_deref().map_or(false, |sh| sh.eq_ignore_ascii_case("VehicleRim")) {
                // ⚠ The static rim (InverseVelocity) dithers OUT with speed
                // for the blur rim to take over - but the blur rims are only
                // drawn with EVOVIEW_RIM_BLUR=1 (load.rs). Without them the
                // rim vanished from ~40 km/h (fully past tyre_max_blur_speed),
                // leaving the far sidewall showing through: "rims disappear
                // when they spin fast". It stays solid unless its blur rim is
                // there to replace it.
                let blur_drawn = crate::envc::var("EVOVIEW_RIM_BLUR").map_or(false, |v| v == "1");
                uniform.fx_m[2] = if m.param(&["InverseVelocity"], 0.0) > 0.5 {
                    if blur_drawn { 2.0 } else { 0.0 }
                } else { 1.0 };
                uniform.fx_m[3] = m.param(&["VelocityGamma"], 1.0).max(0.01);
            }
            let dirt_tex = match m.slot(&["Dirt_Map"]).and_then(|i| upload(device, queue, car, i, false)) {
                Some(v) => { uniform.fx_m[1] = 1.0; v }
                None => solid(device, queue, [0, 0, 0, 255]),
            };
            // the cabin stays dry in the rain (shader wetness)
            if is_cabin(&m.file.to_lowercase()) {
                uniform.layer_pbr[3][1] = 1.0;
            }
            // the windshield (IsWindshield): rain drops at Rain_ScaleUv (rain.rs)
            let glass = matches!(shader_l.as_str(), "vehicleglass" | "vehicleglassinterior" | "vehiclewindow");
            if m.param(&["IsWindshield"], 0.0) > 0.5 {
                uniform.layer_pbr[3][2] = m.param(&["Rain_ScaleUv"], 15.0).max(1.0);
            } else if glass && m.slot(&["Rain_RSVertMask"]).is_some() {
                // a side window (Rain_RS*): its own static drops, and the
                // streaks the airflow pushes over it. x streak uv scale, y
                // the visible run, z refraction
                uniform.layer_pbr[3][2] = m.param(&["Rain_ScaleUv"], 20.0).max(1.0);
                uniform.layer_pbr[0][2] = m.param(&["Rain_RSScaleUv"], 7.0).max(0.1);
                uniform.layer_pbr[0][3] = m.param(&["Rain_RSRange"], 0.5);
                uniform.layer_pbr[1][2] = m.param(&["Rain_ScaleRefraction"], 0.2);
            }
            // bodywork drops (Rain_RaindropsEnabled): the shared drop maps
            // on the upward faces, tiled at Rain_ScaleUv. Unset (most skins),
            // a fine 24 x 24: the 6 x 1.5 some skins author drew palm-sized
            // smears across a livery's uv sheet that read as holes
            if !glass && m.param(&["Rain_RaindropsEnabled"], 0.0) > 0.5 && crate::envc::var_os("EVOVIEW_NO_BODY_RAIN").is_none() {
                let s = m.colors.get("Rain_ScaleUv").filter(|v| v.len() >= 2 && v[0] > 0.0)
                    .map_or([24.0, 24.0], |v| [v[0], v[1]]);
                uniform.layer_pbr[2][2] = s[0];
                uniform.layer_pbr[2][3] = s[1];
            }
            // race-number decals (numbers.rs): coverage x the depth fade
            // carried in |tangent.w|
            if m.path.starts_with("evoview/numbers/") {
                uniform.layer_pbr[3][3] = 1.0;
            }
            // The game's contact-shadow decal (contact.rs): the shader writes
            // black with the texture as coverage, nothing else.
            if m.shader.as_deref() == Some(crate::contact::SHADER) {
                uniform.misc[3] = 4.0;
                crate::render::CONTACT_DECALS
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
            // ---- glass shaders (SPEC_small §2): F0 from Reflectance, a txDiffuse
            // without Modulation_BaseColor replaces the colour, absent = 0 ----
            if matches!(shader_l.as_str(), "vehicleglass" | "vehicleglassinterior" | "vehiclewindow") {
                uniform.uber[0] = [m.param(&["Reflectance"], 0.0).clamp(0.0, 1.0), 0.0, 0.0, 2.0];
                uniform.factors[0] = m.param(&["Roughness"], 0.0);
                uniform.factors[1] = m.param(&["Metalness"], 0.0).clamp(0.0, 1.0);
                if has_albedo_tex && m.param(&["Modulation_BaseColor"], 0.0) < 0.5 {
                    uniform.base_color = [1.0, 1.0, 1.0, uniform.base_color[3]];
                }
            }
            // ---- the Uber vehicle shaders' own decode (SPEC_uber §4-§7) ----
            if shader_l.starts_with("ubervehicle") {
                let p = |n: &str| m.param(&[n], 0.0);
                let f = |n: &str| m.param(&[n], 0.0) > 0.5;
                // a paint file's clear coat (load.rs) wins over the material's
                let cc = if m.has(&["ClearCoat", "Detail_ClearCoatStrenght", "ClearCoatStrength"]) {
                    uniform.flags[2]
                } else {
                    p("Base_ClearCoat").clamp(0.0, 1.0)
                };
                uniform.uber[0] = [p("Base_Reflectance").clamp(0.0, 1.0), cc, p("Base_ClearCoatRoughness"), 1.0];
                uniform.uber[1] = [uniform.factors[1], p("Base_Roughness"), p("Base_AmbientOcclusion").max(0.0),
                                   p("Global_AmbientOcclusionLevel")];
                // no Roughness default: the game reads an absent value as 0
                uniform.factors[0] = p("Base_Roughness");
                for (li, l) in ["Red", "Green", "Blue"].iter().enumerate() {
                    let on = f(&format!("{l}_Enable"));
                    let mode = |x: &str| -> f32 {
                        if !on { 0.0 }
                        else if f(&format!("{l}OverrideReplace_{x}")) { 1.0 }
                        else if f(&format!("{l}OverrideMultiply_{x}")) { 2.0 }
                        else { 0.0 }
                    };
                    let rmode = if !on { 0.0 } else if f(&format!("{l}OverrideReplace_Roughness")) { 1.0 }
                        else if f(&format!("{l}OverrideMultiply_Roughness")) { 2.0 }
                        else if f(&format!("{l}OverrideCombine_AdditiveRoughness")) { 3.0 } else { 0.0 };
                    uniform.uber[2 + li] = [rmode, mode("Metalness"), mode("ClearCoat"), mode("ClearCoatRoughness")];
                    uniform.uber[5 + li] = [p(&format!("{l}_Metalness")).clamp(0.0, 1.0), p(&format!("{l}_ClearCoat")),
                                            p(&format!("{l}_ClearCoatRoughness")), p(&format!("{l}_Roughness")).clamp(0.0, 1.0)];
                    uniform.uber[8 + li] = [mode("Reflectance"), p(&format!("{l}_Reflectance")),
                                            if f(&format!("Modulation_{l}Roughness")) { 1.0 } else { 0.0 }, 0.0];
                }
            }
            if crate::strict::on() && crate::strict::touches(&m.file) {
                uniform.strict[0] = 1.0;
            }
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&m.file),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&m.file),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&albedo) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&normal) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&ao) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&splat) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&t_rough) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&layer_r_view) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&t_gao) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&t_coat) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(
                        pd.n_r.and_then(|i| detail_cache.get(&i)).unwrap_or(&flat_normal)) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(
                        pd.n_gb.and_then(|i| detail_cache.get(&i)).unwrap_or(&flat_normal)) },
                    wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(
                        pd.cc_n.and_then(|i| detail_cache.get(&i)).unwrap_or(&flat_normal)) },
                    wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(
                        pd.ao.and_then(|i| detail_cache.get(&i)).unwrap_or(&white)) },
                    wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(&emis_tex) },
                    wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&layer_c_view) },
                    wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(&mask2_tex) },
                    wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::TextureView(&dirt_tex) },
                ],
            }));
            buffers.push(buf);
            uniforms.push(uniform);
            light_srcs.push(Some(light_src));
            by_name.insert(key_of(&m.file), groups.len() - 1);
            if !m.path.is_empty() {
                by_path.insert(m.path.clone(), groups.len() - 1);
            }
        }
        crate::strict::mute(false);

        if !missing_albedo.is_empty() {
            eprintln!("{} material(s) name an albedo texture that is not in the                        archive; they fall back to their base colour:",
                      missing_albedo.len());
            for m in missing_albedo.iter().take(8) { eprintln!("  {m}"); }
        }
        println!("paint detail: {} detail-normal, {} clear-coat normal, {} detail-AO materials ({} shared textures)",
                 n_pd_n, n_pd_cc, n_pd_ao, detail_cache.len());
        println!("data maps bound: {} roughness, {} red-layer roughness, {} global AO, {} clear-coat; {} cabin materials",
                 n_rough, n_redrough, n_gao, n_coat, n_int);
        println!("layer colour maps: {} red, {} green, {} blue ({} resampled arrays, {:.1} MB); {} base map(s) off (Base_HasBaseColorMap 0)",
                 n_layer_maps[0], n_layer_maps[1], n_layer_maps[2], arrays.resampled,
                 arrays.bytes as f64 / 1048576.0, n_base_off);

        // A batch may reference a material we do not have; give it something
        // neutral rather than skipping the geometry.
        let uniform = MaterialUniform {
            base_color: [0.55, 0.56, 0.58, 1.0],
            factors: [0.5, 0.0, 0.0, 0.0],
            flags: [0.0, 1.0, 0.0, 0.0],
            misc: [0.0, 0.0, 0.5, 0.0],
            layer_r: [0.0; 4],
            layer_g: [0.0; 4],
            layer_b: [0.0; 4],
            layer_metal: [0.0; 4],
            layer_coat: [0.0; 4],
            uv_scale: [0.0; 4],
            maps: [0.0; 4],
            strict: [if crate::strict::on() { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
            ..bytemuck::Zeroable::zeroed()
        };
        blend.push(Blend::Opaque);
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fallback"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fallback"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&white) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&flat_normal) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&white) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&black) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&white) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&white_arr) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&white) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&white) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&flat_normal) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&flat_normal) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&flat_normal) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(&white) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(&black) },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&white_arr) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(&black) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::TextureView(&black) },
            ],
        }));
        buffers.push(buf);
        uniforms.push(uniform);
        light_srcs.push(None);
        let fallback = groups.len() - 1;

        // The studio floor. Dark and fairly smooth so the car gets a soft
        // reflection under it - that reflection is most of what sells the
        // shot as a photograph rather than a model floating in space.
        let ground_uniform = MaterialUniform {
            base_color: [0.045, 0.046, 0.05, 1.0],
            factors: [0.42, 0.0, 0.0, 0.0],
            flags: [0.0, 1.0, 0.0, 1.0],
            misc: [0.0, 0.0, 0.5, 0.0],
            layer_r: [0.0; 4],
            layer_g: [0.0; 4],
            layer_b: [0.0; 4],
            layer_metal: [0.0; 4],
            layer_coat: [0.0; 4],
            uv_scale: [0.0; 4],
            maps: [0.0; 4],
            ..bytemuck::Zeroable::zeroed()
        };
        blend.push(Blend::Opaque);
        let gbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ground"),
            contents: bytemuck::bytes_of(&ground_uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ground"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&white) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&flat_normal) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&white) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&black) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&white) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&white_arr) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&white) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&white) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&flat_normal) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&flat_normal) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&flat_normal) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(&white) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(&black) },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&white_arr) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(&black) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::TextureView(&black) },
            ],
        }));
        buffers.push(gbuf);
        uniforms.push(ground_uniform);
        light_srcs.push(None);
        let ground = groups.len() - 1;

        println!("materials: {} bound, fallback for the rest", by_name.len());
        let env = car.env.as_ref().and_then(|c| upload_cube(device, queue, c));
        let has_env = env.is_some();
        // how many mips actually reached the GPU decides the roughest lookup
        let lod = if has_env {
            car.env
                .as_ref()
                .map(|c| {
                    c.levels
                        .iter()
                        .take_while(|(w, _)| (w.div_ceil(4) * 16) % 256 == 0)
                        .count()
                        .saturating_sub(1) as i32
                })
                .unwrap_or(0)
        } else {
            -1
        };
        crate::render::ENV_MAX_LOD.store(lod, std::sync::atomic::Ordering::Relaxed);
        let env_view = env.unwrap_or_else(|| dummy_cube(device, queue));
        // Trilinear across the mip chain: the roughness lookup picks a level,
        // so filtering BETWEEN levels is what stops a rough surface banding as
        // it turns.
        let env_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("env sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let brdf_view = upload_brdf(device, queue, car.brdf.as_ref());
        Materials { layout, groups, by_name, by_path, fallback, ground, blend,
                    env_view, env_samp, has_env, brdf_view,
                    lut_view: crate::grade::upload(device, queue),
                    buffers, uniforms, light_srcs, light_state, dash_targets,
                    textures: TexCache::default() }
    }

    /// Switch the car's lights to state `s` (lights::STATE_NAMES) by
    /// rewriting only the uniforms whose emissive part changes.
    pub fn set_light_state(&mut self, queue: &wgpu::Queue, s: u32) {
        self.light_state = s;
        for i in 0..self.uniforms.len() {
            let Some(src) = &self.light_srcs[i] else { continue };
            let mut u = self.uniforms[i];
            apply_lights(&mut u, src, s);
            if bytemuck::bytes_of(&u) != bytemuck::bytes_of(&self.uniforms[i]) {
                self.uniforms[i] = u;
                queue.write_buffer(&self.buffers[i], 0, bytemuck::bytes_of(&u));
            }
        }
    }

    /// The motion-blur rim: invisible at rest (rim_visible in the shader),
    /// so it must not stamp the shadow map either (the depth-only pass has
    /// no fragment stage to discard it).
    pub fn casts_no_shadow(&self, mi: usize) -> bool {
        self.uniforms.get(mi).map_or(false, |u| u.fx_m[2] > 0.5 && u.fx_m[2] < 1.5)
    }

    /// Resolve a batch's material reference (an engine path) to a bind group.
    /// ⚠ Resolve by FULL PATH first. Material basenames are not unique - a
    /// track ships several `bark_mat.material` under different vegetation
    /// folders, and resolving on the basename alone handed every mesh whichever
    /// copy happened to load last. That is why bark rendered correctly on Fuji
    /// and Spa (which also ship a textured copy) and flat white on Brands
    /// Hatch, Cota and Donington (which ship only the untextured one).
    /// Mesh batches reference materials by full path, so this is decidable.
    pub fn lookup(&self, reference: &str) -> usize {
        if let Some(&i) = self.by_path.get(&crate::load::norm_mat_path(reference)) {
            return i;
        }
        let base = reference
            .rsplit(|c| c == '\\' || c == '/')
            .next()
            .unwrap_or(reference);
        self.by_name
            .get(&key_of(base))
            .copied()
            .unwrap_or(self.fallback)
    }
}

/// Materials are referenced with inconsistent case (EXT_SKIN_2.material vs
/// ext_skin_2.material), so everything is keyed lowercase without extension.
/// Is this a 1x1 white stand-in, or a real texture that merely has "white" in
/// its name?
///
/// ⚠ A bare `contains("white")` throws away real textures. On Spa alone it was
/// discarding `whiteline` (the track's own line markings),
/// `painted_scraped_sheet_white_albedo` and `aspen_white_custom_1_albedo`, and
/// on cars `tyres_albedo_white` - each rendered flat white with its texture
/// switched off. The placeholder is named `white`, possibly with a size or
/// extension after it; a real texture continues with more letters.
fn is_white_placeholder(n: &str) -> bool {
    match n.strip_prefix("white") {
        None => false,
        Some(rest) => !rest.starts_with(|c: char| c.is_ascii_alphabetic()),
    }
}

pub fn key_of(name: &str) -> String {
    name.trim_end_matches(".material").to_lowercase()
}

/// Ground / lawn / dirt — not fence cards, not buildings that happen to
/// have "ext" in the name.
/// The car shader templates the viewer models (what stock and converted
/// cars use). Anything else on a car is drawn as generic PBR - a guess.
pub const CAR_SHADERS: &[&str] = &[
    "ubervehiclematerial", "ubervehicleinteriormaterial", "vehicleglass", "vehicleglassinterior",
    "vehiclerim", "vehicleled", "vehiclelight", "vehicledisc", "vehicletyre", "vehiclewindshield", "vehiclewindow",
    "evoviewcontactshadow",
];

fn is_terrain_ground(file: &str, shader: &str) -> bool {
    if shader.contains("fence") || shader.contains("glass") {
        return false;
    }
    if file.contains("_bb") || file.contains("billboard") || file.contains("grass_bb") {
        return false;
    }
    // ⚠ Foliage is not ground, however it is named. "background" is a terrain
    // keyword below, and Spa's tree line is `background_trees.material` - so
    // it was classified as terrain and had its `treesline` albedo swapped for
    // a tiled ground photo. Those meshes are 1200-2000 m long strips ringing
    // the circuit, which is why the result was long white streaks radiating
    // across the hillsides rather than a wrong-looking field.
    if file.contains("tree") || file.contains("bush") || file.contains("hedge") {
        return false;
    }
    const KEYS: &[&str] = &[
        "grass", "terrain", "ground", "outerland", "dirt", "sand", "soil",
        "erba", "lawn", "field", "gravel", "stone", "rock", "quarry",
        "bank", "hill", "mount", "sbanc", "top-ext", "new_top", "top_ext",
        "far", "background", "sat_far", "satfar",
    ];
    KEYS.iter().any(|k| file.contains(k))
        || file.contains("splat-map")
        || file.contains("splat_map")
        || file.contains("splatmap")
        || file.contains("top_near")
        || file.contains("top_far")
        || file.starts_with("top_")
}

/// The albedo is already a repeating ground photo — leave mesh UVs alone.
fn albedo_looks_tiled_ground(name: &str) -> bool {
    if name.contains("far") {
        return false;
    }
    name.contains("grass")
        || name.contains("cut")
        || name.contains("turf")
        || name.contains("erba")
        || name.contains("sand")
        || name.contains("dirt")
        || name.contains("gravel")
        || name.contains("soil")
        || name.contains("asph")
        || name.contains("tarmac")
}

/// Baked unique unwraps (grass_far, map3, outerland) need a world-space tile
/// from the same track. Named after no particular circuit.
fn albedo_needs_world_tile(alb: &str, file: &str) -> bool {
    if alb.is_empty() || albedo_looks_tiled_ground(alb) {
        return false;
    }
    alb.contains("far")
        || alb.contains("mount_d")
        || (alb.contains("map") && !alb.contains("normal") && !alb.contains("splat"))
        || (file.contains("grass")
            && !file.contains("outer")
            && !alb.contains("grass")
            && !alb.contains("erba")
            && !alb.contains("cut"))
}

/// Pick a tileable grass photo from whatever this track shipped.
/// Which texture to tile over a baked-unwrap terrain surface.
///
/// ⚠ Name scoring alone is not enough, and Brands Hatch is the proof: it picked
/// `grass_wild_1_albedo` - a dry straw texture out of common_assets that the
/// track never references - and painted the entire circuit yellow. The same
/// track ships `grass_2_general_albedo`, the green tile Spa and Monza both use
/// happily.
///
/// Two signals fix it, and both are about intent rather than filename:
///   * a texture THIS TRACK actually uses as somebody's albedo is the right
///     look for this track; an unused one from the shared library is a guess.
///   * the tile should match the MATERIAL's family. Monza's `sand.material`
///     was being handed a grass tile.
fn pick_ground_tile(
    car: &Car,
    skip: Option<usize>,
    fname: &str,
    used: &std::collections::HashSet<usize>,
) -> Option<usize> {
    // EVOVIEW_NO_TILE=1: show materials exactly as authored (conversion previews)
    if crate::envc::var("EVOVIEW_NO_TILE").is_ok() {
        return None;
    }
    // What kind of ground is this material meant to be?
    let want: &[&str] = if fname.contains("sand") {
        &["sand"]
    } else if fname.contains("gravel") {
        &["gravel"]
    } else if fname.contains("dirt") || fname.contains("soil") || fname.contains("mud") {
        &["dirt", "soil", "ground"]
    } else if fname.contains("grass") || fname.contains("erba")
        || fname.contains("turf") || fname.contains("lawn")
    {
        // ⚠ Symmetry matters. Without this, Donington's `grass.material` was
        // handed `gravelblack2_4x4_albedo` - the candidate filter accepts
        // gravel, so a grass surface could be tiled with grey stones.
        &["grass", "erba", "turf", "lawn"]
    } else {
        &[]
    };
    let mut best: Option<(i32, usize)> = None;
    for (i, t) in car.textures.iter().enumerate() {
        if skip == Some(i) {
            continue;
        }
        let n = t.name.to_lowercase();
        if !(n.contains("grass")
            || n.contains("erba")
            || n.contains("turf")
            || n.contains("lawn")
            || n.contains("dirt")
            || n.contains("sand")
            || n.contains("soil")
            || n.contains("gravel"))
        {
            continue;
        }
        if [
            "_nm", "normal", "_nrm", "_far", "mask", "alpha", "_bb", "billboard",
        ]
        .iter()
        .any(|x| n.contains(x))
            // UE-style suffixes (AC Rally conversions): _n normal, _l packed, _m mask
            || [".texture", ""].iter().any(|ext| ["_n", "_l", "_m"].iter().any(|sfx| n.ends_with(&format!("{sfx}{ext}"))))
        {
            continue;
        }
        let mut score = (t.width.saturating_mul(t.height) / 1024) as i32;
        if n.contains("cut")
            || n.contains("detail")
            || n.contains("base")
            || n.contains("albedo")
            || n.contains("top")
        {
            score += 8_000;
        }
        // Numbered grass photos (grass15, Grass0118, grass3_d) beat a pale
        // generic grass_d / grass_far leftover.
        if n.chars().any(|c| c.is_ascii_digit()) {
            score += 6_000;
        }
        if n == "grass" || n.starts_with("grass_base") {
            score += 4_000;
        }
        // A texture the track itself references beats an unused one out of the
        // shared library - this is the Brands Hatch fix.
        if used.contains(&i) {
            score += 30_000;
        }
        // Dry, wild and dead variants are the ones that read as "the whole
        // circuit is yellow".
        if ["wild", "dry", "dead", "straw", "burnt", "autumn", "yellow"]
            .iter()
            .any(|x| n.contains(x))
        {
            score -= 20_000;
        }
        if !want.is_empty() {
            if want.iter().any(|w| n.contains(w)) {
                score += 50_000;
            } else {
                // Still allowed, but only if nothing in the right family exists.
                score -= 25_000;
            }
        }
        if best.map(|(s, _)| score > s).unwrap_or(true) {
            best = Some((score, i));
        }
    }
    best.map(|(_, i)| i)
}

pub fn material_entry_name(m: &MaterialEntry) -> String {
    key_of(&m.file)
}

/// Upload a decoded cubemap as a GPU cube texture.
///
/// ⚠ Mips stop once a row is under 256 bytes. `write_texture` requires
/// `bytes_per_row` to be a multiple of 256, and for BC7 that means a face at
/// least 64 px wide (16 blocks x 16 B). The tail of the chain is 32 px and
/// below, which is past the point where a reflection carries any shape, so
/// dropping it costs nothing and avoids padding every small level by hand.
pub fn upload_cube(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cube: &crate::load::Cube,
) -> Option<wgpu::TextureView> {
    let format = match cube.format.as_str() {
        "bc7" => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
        "bc6h" => wgpu::TextureFormat::Bc6hRgbUfloat,
        "bc1" => wgpu::TextureFormat::Bc1RgbaUnormSrgb,
        "rgba16f" => wgpu::TextureFormat::Rgba16Float,
        _ => wgpu::TextureFormat::Rgba8UnormSrgb,
    };
    let block = if matches!(cube.format.as_str(), "bc1" | "bc7" | "bc6h") { 4u32 } else { 1 };
    let keep: Vec<usize> = cube
        .levels
        .iter()
        .enumerate()
        .take_while(|(_, (w, _))| {
            let row = match cube.format.as_str() {
                "bc1" => (w.div_ceil(4)) * 8,
                "bc7" | "bc6h" => (w.div_ceil(4)) * 16,
                _ => w * 4,
            };
            row % 256 == 0
        })
        .map(|(i, _)| i)
        .collect();
    if keep.is_empty() {
        return None;
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("env cube"),
        size: wgpu::Extent3d {
            width: cube.size,
            height: cube.size,
            depth_or_array_layers: 6,
        },
        mip_level_count: keep.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let per_face: usize = cube.levels.iter().map(|(_, n)| *n).sum();
    for face in 0..6usize {
        let mut off = face * per_face;
        for (i, &(w, n)) in cube.levels.iter().enumerate() {
            if off + n > cube.data.len() {
                break;
            }
            if keep.contains(&i) {
                let row = match cube.format.as_str() {
                    "bc1" => (w.div_ceil(4)) * 8,
                    "bc7" | "bc6h" => (w.div_ceil(4)) * 16,
                    _ => w * 4,
                };
                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &tex,
                        mip_level: i as u32,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: face as u32 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &cube.data[off..off + n],
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(w.div_ceil(block)),
                    },
                    wgpu::Extent3d { width: w, height: w, depth_or_array_layers: 1 },
                );
            }
            off += n;
        }
    }
    Some(tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    }))
}

/// A 1x1 black cube, so the pipeline has something to bind when the game's
/// own environment is not available (a bare car folder, no package).
pub fn dummy_cube(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("env cube (none)"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for face in 0..6u32 {
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: face },
                aspect: wgpu::TextureAspect::All,
            },
            &[0u8, 0, 0, 255],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
    }
    tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    })
}

/// The split-sum BRDF table as a 2D texture, or a 1x1 stand-in.
///
/// ⚠ Falls back rather than failing: without the package there is no table,
/// and the shader keeps its analytic fit. mode.z already says whether the
/// game's environment is present, and the two arrive together.
pub fn upload_brdf(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lut: Option<&(u32, Vec<u8>)>,
) -> wgpu::TextureView {
    let (edge, data): (u32, &[u8]) = match lut {
        Some((e, d)) => (*e, d),
        None => (1, &[0u8, 0, 0, 0]),
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("brdf lut"),
        size: wgpu::Extent3d { width: edge, height: edge, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(edge * 4),
            rows_per_image: Some(edge),
        },
        wgpu::Extent3d { width: edge, height: edge, depth_or_array_layers: 1 },
    );
    tex.create_view(&Default::default())
}
