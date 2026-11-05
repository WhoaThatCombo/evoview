//! AMD FSR upscaling (FSR 3.1 on any Vulkan GPU; the SDK's FSR 4 provider
//! where AMD supports it) through the FidelityFX "ffx_api", on wgpu 22.
//!
//! EVOVIEW_UPSCALER=fsr swaps the TAA resolve for FSR (default: taa).
//! EVOVIEW_FSR_QUALITY = native | quality | balanced | performance |
//! ultraperformance picks the render scale (main.rs sets EVOVIEW_RENDER_SCALE
//! from it; default quality = 1/1.5). native = FSR's anti-aliasing mode.
//! EVOVIEW_FSR_VERSION=<substring> picks a provider by name (the list is
//! printed at start), EVOVIEW_FSR_LOG=1 shows FSR's messages.
//!
//! ffx_api is plain C (five functions, struct-chained descriptors) in AMD's
//! signed amd_fidelityfx_loader_vk.dll, which loads
//! amd_fidelityfx_upscaler_vk.dll from beside the exe: both are loaded at
//! runtime here, no shim, nothing in git. No dlls or any failure: one line,
//! and TAA.
//!
//! Same plumbing as dlss.rs: FSR records into wgpu's command buffer
//! (as_hal_mut) after a no-op dispatch that puts its inputs in
//! SHADER_READ_ONLY and the output in GENERAL. The depth goes in as an
//! R32Float colour copy: wgpu keeps a sampled depth buffer in
//! DEPTH_STENCIL_READ_ONLY, while FSR's descriptors say SHADER_READ_ONLY
//! (harmless on NVIDIA, not on every GPU). The device needs no extension;
//! render.rs asks wgpu for SHADER_F16 when FSR is wanted, because FSR picks
//! its fp16 shaders from what the GPU supports, assuming it is enabled.

use std::ffi::{c_char, c_void};

/// EVOVIEW_UPSCALER=fsr asks for FSR.
pub fn wanted() -> bool {
    crate::envc::var("EVOVIEW_UPSCALER").map_or(false, |v| v.eq_ignore_ascii_case("fsr"))
}

/// (FfxApiUpscaleQualityMode, render scale) for EVOVIEW_FSR_QUALITY.
pub fn quality() -> (u32, f32) {
    match crate::envc::var("EVOVIEW_FSR_QUALITY").unwrap_or_default().to_ascii_lowercase().as_str() {
        "native" | "nativeaa" | "aa" => (0, 1.0),
        "balanced" => (2, 1.0 / 1.7),
        "performance" | "perf" => (3, 0.5),
        "ultraperformance" | "ultra_performance" => (4, 1.0 / 3.0),
        _ => (1, 1.0 / 1.5),
    }
}

/// Startup: the render scale from the quality mode unless EVOVIEW_RENDER_SCALE
/// is given (as dlss::apply_render_scale).
pub fn apply_render_scale() {
    if wanted() && crate::envc::var_os("EVOVIEW_RENDER_SCALE").is_none() {
        crate::envc::set_var("EVOVIEW_RENDER_SCALE", format!("{}", quality().1));
    }
}

// ---- ffx_api (Kits/FidelityFX/api/include/ffx_api*.h, upscalers/include/ffx_upscale.h)
type Ctx = *mut c_void;
#[repr(C)]
struct Header {
    ty: u64,
    next: *mut Header,
}
const DESC_OVERRIDE_VERSION: u64 = 5;
const QUERY_GET_PROVIDER_VERSION: u64 = 6;
const QUERY_GET_VERSIONS: u64 = 4;
const CREATE_BACKEND_VK: u64 = 0x0100_0003;
const CREATE_UPSCALE: u64 = 0x0001_0000;
const DISPATCH_UPSCALE: u64 = 0x0001_0001;
const CREATE_UPSCALE_VERSION: u64 = 0x0001_000b;
/// FFX_UPSCALER_VERSION 4.1.2 (the headers this was written against)
const UPSCALER_VERSION: u32 = (4 << 22) | (1 << 12) | 2;
const FLAG_HDR: u32 = 1 << 0;
const FLAG_DEPTH_INFINITE: u32 = 1 << 4;
const FLAG_AUTO_EXPOSURE: u32 = 1 << 5;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Dim {
    w: u32,
    h: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct F2 {
    x: f32,
    y: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct ResDesc {
    ty: u32,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    mips: u32,
    flags: u32,
    usage: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Res {
    resource: *mut c_void,
    desc: ResDesc,
    state: u32,
}
impl Res {
    fn none() -> Res {
        Res { resource: std::ptr::null_mut(), desc: ResDesc { ty: 0, format: 0, width: 0, height: 0, depth: 0, mips: 0, flags: 0, usage: 0 }, state: 0 }
    }
}
type MsgFn = unsafe extern "C" fn(u32, *const u16);
#[repr(C)]
struct CreateUpscale {
    header: Header,
    flags: u32,
    max_render: Dim,
    max_upscale: Dim,
    msg: Option<MsgFn>,
}
#[repr(C)]
struct CreateUpscaleVersion {
    header: Header,
    version: u32,
}
#[repr(C)]
struct CreateBackendVk {
    header: Header,
    device: ash::vk::Device,
    phys: ash::vk::PhysicalDevice,
    gdpa: ash::vk::PFN_vkGetDeviceProcAddr,
}
#[repr(C)]
struct OverrideVersion {
    header: Header,
    id: u64,
}
#[repr(C)]
struct QueryVersions {
    header: Header,
    create_desc_type: u64,
    device: *mut c_void,
    count: *mut u64,
    ids: *mut u64,
    names: *mut *const c_char,
}
#[repr(C)]
struct QueryProvider {
    header: Header,
    id: u64,
    name: *const c_char,
}
#[repr(C)]
struct DispatchUpscale {
    header: Header,
    cmd: *mut c_void,
    color: Res,
    depth: Res,
    mv: Res,
    exposure: Res,
    reactive: Res,
    tac: Res,
    output: Res,
    jitter: F2,
    mv_scale: F2,
    render_size: Dim,
    upscale_size: Dim,
    sharpen: bool,
    sharpness: f32,
    frame_ms: f32,
    pre_exposure: f32,
    reset: bool,
    near: f32,
    far: f32,
    fov_y: f32,
    meters: f32,
    flags: u32,
}
/// the VkDeviceContext ffx's Vulkan queries take
#[repr(C)]
struct VkDeviceContext {
    device: ash::vk::Device,
    phys: ash::vk::PhysicalDevice,
    gdpa: ash::vk::PFN_vkGetDeviceProcAddr,
}

struct Api {
    _lib: libloading::Library,
    create: unsafe extern "C" fn(*mut Ctx, *mut Header, *const c_void) -> u32,
    destroy: unsafe extern "C" fn(*mut Ctx, *const c_void) -> u32,
    dispatch: unsafe extern "C" fn(*mut Ctx, *const Header) -> u32,
    query: unsafe extern "C" fn(*mut Ctx, *mut Header) -> u32,
}

fn dirs() -> Vec<std::path::PathBuf> {
    crate::envc::var_os("EVOVIEW_FSR_DIR").map(Into::into).into_iter()
        .chain(std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())))
        .collect()
}

fn load() -> Result<Api, String> {
    let ds = dirs();
    let dir = ds.iter().find(|d| d.join("amd_fidelityfx_loader_vk.dll").exists())
        .ok_or_else(|| format!("amd_fidelityfx_loader_vk.dll not found in {ds:?} (FidelityFX SDK Kits/FidelityFX/signedbin)"))?;
    if !dir.join("amd_fidelityfx_upscaler_vk.dll").exists() && !dirs().iter().any(|d| d.join("amd_fidelityfx_upscaler_vk.dll").exists()) {
        return Err("amd_fidelityfx_upscaler_vk.dll not found beside the loader".into());
    }
    unsafe {
        let lib = libloading::Library::new(dir.join("amd_fidelityfx_loader_vk.dll")).map_err(|e| format!("amd_fidelityfx_loader_vk.dll: {e}"))?;
        macro_rules! f {
            ($n:literal) => {
                *lib.get($n).map_err(|e| format!("amd_fidelityfx_loader_vk.dll: {e}"))?
            };
        }
        Ok(Api { create: f!(b"ffxCreateContext"), destroy: f!(b"ffxDestroyContext"), dispatch: f!(b"ffxDispatch"),
                 query: f!(b"ffxQuery"), _lib: lib })
    }
}

unsafe extern "C" fn ffx_msg(ty: u32, msg: *const u16) {
    if msg.is_null() {
        return;
    }
    let mut n = 0;
    while *msg.add(n) != 0 && n < 4096 {
        n += 1;
    }
    let s = String::from_utf16_lossy(std::slice::from_raw_parts(msg, n));
    eprintln!("FSR {}: {}", if ty == 0 { "error" } else { "warning" }, s.trim_end());
}

pub struct Fsr {
    api: Api,
    ctx: Ctx,
    /// (render size, output size) the context was made for
    made: Option<((u32, u32), (u32, u32))>,
    dc: (ash::vk::Device, ash::vk::PhysicalDevice, ash::vk::PFN_vkGetDeviceProcAddr),
    /// FSR's output (screen size, storage)
    pub out: wgpu::TextureView,
    out_tex: wgpu::Texture,
    /// the depth as R32Float colour (render size)
    zcopy: Option<(wgpu::Texture, wgpu::TextureView)>,
    z_pipe: wgpu::RenderPipeline,
    z_bgl: wgpu::BindGroupLayout,
    tr_pipe: wgpu::ComputePipeline,
    tr_bgl: wgpu::BindGroupLayout,
    size: (u32, u32),
    reset: bool,
    version: Option<u64>,
}

pub const OUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const Z_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

impl Fsr {
    /// Load ffx_api and list the upscaler providers this GPU gets. The
    /// context itself is made on the first frame (it needs the render size).
    pub fn new(device: &wgpu::Device, w: u32, h: u32, samples: u32) -> Result<Fsr, String> {
        if samples > 1 {
            return Err("needs single-sample depth (EVOVIEW_TRACK_MSAA=1)".into());
        }
        let api = load()?;
        let dc = unsafe {
            use wgpu::hal::api::Vulkan;
            device.as_hal::<Vulkan, _, _>(|d| d.map(|d| {
                (d.raw_device().handle(), d.raw_physical_device(), d.shared_instance().raw_instance().fp_v1_0().get_device_proc_addr)
            })).flatten().ok_or("not a Vulkan device")?
        };
        // the providers on offer (FSR 4 only on the AMD GPUs it supports)
        let mut version = None;
        unsafe {
            let mut vdc = VkDeviceContext { device: dc.0, phys: dc.1, gdpa: dc.2 };
            let mut count = 0u64;
            let mut q = QueryVersions { header: Header { ty: QUERY_GET_VERSIONS, next: std::ptr::null_mut() }, create_desc_type: CREATE_UPSCALE,
                                        device: &mut vdc as *mut _ as *mut c_void, count: &mut count, ids: std::ptr::null_mut(), names: std::ptr::null_mut() };
            if (api.query)(std::ptr::null_mut(), &mut q.header) == 0 && count > 0 {
                let mut ids = vec![0u64; count as usize];
                let mut names = vec![std::ptr::null::<c_char>(); count as usize];
                q.ids = ids.as_mut_ptr();
                q.names = names.as_mut_ptr();
                if (api.query)(std::ptr::null_mut(), &mut q.header) == 0 {
                    let list: Vec<(u64, String)> = ids.iter().zip(&names).take(count as usize)
                        .map(|(&i, &n)| (i, if n.is_null() { "?".into() } else { std::ffi::CStr::from_ptr(n).to_string_lossy().into_owned() }))
                        .collect();
                    println!("FSR providers: {}", list.iter().map(|l| l.1.as_str()).collect::<Vec<_>>().join(", "));
                    if let Ok(want) = crate::envc::var("EVOVIEW_FSR_VERSION") {
                        version = list.iter().find(|l| l.1.contains(&want)).map(|l| l.0);
                        if version.is_none() {
                            eprintln!("FSR: no provider matching EVOVIEW_FSR_VERSION={want}");
                        }
                    }
                }
            }
        }
        let (out_tex, out) = Self::output(device, w, h);
        let (tr_pipe, tr_bgl) = Self::transition_pipe(device);
        let (z_pipe, z_bgl) = Self::depth_pipe(device);
        Ok(Fsr { api, ctx: std::ptr::null_mut(), made: None, dc, out, out_tex, zcopy: None, z_pipe, z_bgl, tr_pipe, tr_bgl,
                 size: (w, h), reset: true, version })
    }

    fn output(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let t = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fsr output"),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OUT_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let v = t.create_view(&Default::default());
        (t, v)
    }

    fn depth_pipe(device: &wgpu::Device) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fsr depth copy"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fsr depth copy"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fsr depth copy"),
            source: wgpu::ShaderSource::Wgsl("
@group(0) @binding(0) var t: texture_depth_2d;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(textureLoad(t, vec2<i32>(p.xy), 0), 0.0, 0.0, 1.0);
}".into()),
        });
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fsr depth copy"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs", compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs", compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: Z_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        (pipe, bgl)
    }

    fn transition_pipe(device: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let tex = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                             view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fsr transitions"),
            entries: &[
                tex(0), tex(1), tex(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: OUT_FORMAT,
                                                            view_dimension: wgpu::TextureViewDimension::D2 },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fsr transitions"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fsr transitions"),
            source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
        });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fsr transitions"), layout: Some(&layout), module: &shader, entry_point: "main",
            compilation_options: Default::default(), cache: None });
        (pipe, bgl)
    }

    pub fn resize(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if (w, h) != self.size {
            let (t, v) = Self::output(device, w, h);
            self.out_tex = t;
            self.out = v;
            self.size = (w, h);
            self.reset = true;
        }
    }

    pub fn reset(&mut self) {
        self.reset = true;
    }

    fn destroy_ctx(&mut self) {
        if !self.ctx.is_null() {
            unsafe { (self.api.destroy)(&mut self.ctx, std::ptr::null()) };
            self.ctx = std::ptr::null_mut();
        }
        self.made = None;
    }

    /// Upscale `color` (render size, jittered by `jitter` render px, x right
    /// y down) into `self.out`. `vp` is the unjittered view-projection (FSR
    /// reads the near plane and vertical fov off it), `dt` seconds.
    #[allow(clippy::too_many_arguments)]
    pub fn run(&mut self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder,
               color: (&wgpu::Texture, &wgpu::TextureView), depth: &wgpu::TextureView,
               mv: (&wgpu::Texture, &wgpu::TextureView), jitter: (f32, f32), vp: glam::Mat4, dt: f32) -> Result<(), String> {
        use wgpu::hal::api::Vulkan;
        let rsize = (color.0.width(), color.0.height());
        let hdr = color.0.format() == wgpu::TextureFormat::Rgba16Float;
        if self.made != Some((rsize, self.size)) {
            self.destroy_ctx();
            let (dev, phys, gdpa) = self.dc;
            let mut backend = CreateBackendVk { header: Header { ty: CREATE_BACKEND_VK, next: std::ptr::null_mut() }, device: dev, phys, gdpa };
            let mut ver = CreateUpscaleVersion { header: Header { ty: CREATE_UPSCALE_VERSION, next: &mut backend.header }, version: UPSCALER_VERSION };
            let mut ovr = OverrideVersion { header: Header { ty: DESC_OVERRIDE_VERSION, next: &mut ver.header }, id: self.version.unwrap_or(0) };
            let log = crate::envc::var_os("EVOVIEW_FSR_LOG").is_some();
            let mut cd = CreateUpscale {
                header: Header { ty: CREATE_UPSCALE, next: if self.version.is_some() { &mut ovr.header } else { &mut ver.header } },
                flags: FLAG_DEPTH_INFINITE | if hdr { FLAG_HDR | FLAG_AUTO_EXPOSURE } else { 0 },
                max_render: Dim { w: rsize.0, h: rsize.1 },
                max_upscale: Dim { w: self.size.0, h: self.size.1 },
                msg: if log { Some(ffx_msg) } else { None },
            };
            let mut ctx: Ctx = std::ptr::null_mut();
            let r = unsafe { (self.api.create)(&mut ctx, &mut cd.header, std::ptr::null()) };
            if r != 0 || ctx.is_null() {
                return Err(format!("ffxCreateContext failed ({r})"));
            }
            self.ctx = ctx;
            self.made = Some((rsize, self.size));
            self.reset = true;
            let mut q = QueryProvider { header: Header { ty: QUERY_GET_PROVIDER_VERSION, next: std::ptr::null_mut() }, id: 0, name: std::ptr::null() };
            let name = unsafe {
                if (self.api.query)(&mut self.ctx, &mut q.header) == 0 && !q.name.is_null() {
                    std::ffi::CStr::from_ptr(q.name).to_string_lossy().into_owned()
                } else {
                    "?".into()
                }
            };
            println!("FSR: {} {}x{} -> {}x{}{}", name, rsize.0, rsize.1, self.size.0, self.size.1, if hdr { ", HDR" } else { "" });
        }
        // the depth as colour, so FSR finds it in the layout it binds it in
        if self.zcopy.as_ref().map_or(true, |(t, _)| (t.width(), t.height()) != rsize) {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fsr depth"),
                size: wgpu::Extent3d { width: rsize.0, height: rsize.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Z_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            self.zcopy = Some((t, v));
        }
        let (ztex, zview) = self.zcopy.as_ref().unwrap();
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fsr depth copy"),
                layout: &self.z_bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth) }],
            });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fsr depth copy"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: zview,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.z_pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fsr transitions"),
            layout: &self.tr_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(color.1) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(zview) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(mv.1) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.out) },
            ],
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("fsr transitions"), timestamp_writes: None });
            cp.set_pipeline(&self.tr_pipe);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }
        let image = |t: &wgpu::Texture| unsafe { t.as_hal::<Vulkan, _, _>(|t| t.map(|t| t.raw_handle())) };
        let fmt = |f: wgpu::TextureFormat| match f {
            wgpu::TextureFormat::Rgba16Float => 4,
            wgpu::TextureFormat::Rg16Float => 18,
            wgpu::TextureFormat::R32Float => 28,
            wgpu::TextureFormat::Rgba8UnormSrgb => 12,
            wgpu::TextureFormat::Bgra8UnormSrgb => 15,
            _ => 0,
        };
        // FfxApiResource: TEXTURE2D, COMPUTE_READ (4) / UNORDERED_ACCESS (2)
        let res = |t: &wgpu::Texture, uav: bool| -> Option<Res> {
            use ash::vk::Handle;
            let img = image(t)?;
            Some(Res {
                resource: img.as_raw() as *mut c_void,
                desc: ResDesc { ty: 2, format: fmt(t.format()), width: t.width(), height: t.height(), depth: 1, mips: 1, flags: 0,
                                usage: if uav { 2 } else { 0 } },
                state: if uav { 2 } else { 4 },
            })
        };
        let near = vp.row(3).w - vp.row(2).w;
        let fy = vp.row(1).truncate().length();
        let mut d = DispatchUpscale {
            header: Header { ty: DISPATCH_UPSCALE, next: std::ptr::null_mut() },
            cmd: std::ptr::null_mut(),
            color: res(color.0, false).ok_or("colour")?,
            depth: res(ztex, false).ok_or("depth")?,
            mv: res(mv.0, false).ok_or("motion")?,
            exposure: Res::none(),
            reactive: Res::none(),
            tac: Res::none(),
            output: res(&self.out_tex, true).ok_or("output")?,
            jitter: F2 { x: jitter.0, y: jitter.1 },
            mv_scale: F2 { x: 1.0, y: 1.0 },
            render_size: Dim { w: rsize.0, h: rsize.1 },
            upscale_size: Dim { w: self.size.0, h: self.size.1 },
            // post.rs sharpens (CAS) already
            sharpen: false,
            sharpness: 0.0,
            frame_ms: (dt * 1000.0).clamp(0.1, 200.0),
            pre_exposure: 1.0,
            reset: std::mem::replace(&mut self.reset, false),
            near: if near > 0.0 { near } else { 0.1 },
            far: 1.0e6,
            fov_y: if fy > 0.0 { 2.0 * (1.0 / fy).atan() } else { 1.0 },
            meters: 1.0,
            flags: 0,
        };
        let ctx = &mut self.ctx;
        let dispatch = self.api.dispatch;
        let r = unsafe {
            enc.as_hal_mut::<Vulkan, _, _>(|e| -> Result<(), String> {
                use ash::vk::Handle;
                let cmd = e.ok_or("no Vulkan encoder")?.raw_handle();
                d.cmd = cmd.as_raw() as *mut c_void;
                let r = dispatch(ctx, &d.header);
                if r != 0 {
                    return Err(format!("ffxDispatch failed ({r})"));
                }
                Ok(())
            })
        };
        r.ok_or("as_hal_mut")?
    }

    /// Destroy the context once the GPU is idle, before the device goes.
    pub fn shutdown(mut self, device: &wgpu::Device) {
        device.poll(wgpu::Maintain::Wait);
        self.destroy_ctx();
    }
}
