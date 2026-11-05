//! NVIDIA DLSS Super Resolution / DLAA through NGX, on wgpu 22's Vulkan.
//!
//! EVOVIEW_UPSCALER=dlss swaps the TAA resolve for DLSS (default: taa).
//! EVOVIEW_DLSS_QUALITY = dlaa | quality | balanced | performance |
//! ultraperformance picks the render scale (main.rs sets EVOVIEW_RENDER_SCALE
//! from it before anything reads it; default quality = 0.667).
//!
//! How it is wired without a wgpu upgrade (dlss_wgpu needs wgpu 26+):
//! - the device is opened by hand through wgpu-hal with the extensions NGX
//!   asks for added to wgpu's own (`request_device`), then wrapped back with
//!   `create_device_from_hal`;
//! - the NGX calls are recorded straight into wgpu's command buffer through
//!   `CommandEncoder::as_hal_mut`; a no-op compute dispatch that binds the
//!   inputs as sampled images and the output as a storage image first makes
//!   wgpu put them in the layouts DLSS expects (SHADER_READ_ONLY / GENERAL);
//! - NGX is called in the NVIDIA driver's own runtime (`_nvngx.dll`, found
//!   through the registry; see ffi): evoview links and ships no NVIDIA code;
//! - nvngx_dlss.dll is NVIDIA's DLSS model, which the player downloads from
//!   NVIDIA and puts beside the exe (or EVOVIEW_DLSS_DIR); NGX loads it.
//!   No dll, a non-NVIDIA GPU, or any NGX failure: a message, and TAA as
//!   before.
//!
//! Inputs: the jittered scene colour (taa.cur, render size, HDR linear),
//! the depth buffer, motion.rs's vectors (an Rg16Float copy written by the
//! same pass), the jitter offset; exposure is DLSS's own (AutoExposure).
//! The output (screen size) goes through taa.rs's fs_ext, which adds the
//! ambient occlusion and volumetric fog TAA's resolve would have, then post.

/// EVOVIEW_UPSCALER=dlss asks for DLSS.
pub fn wanted() -> bool {
    crate::envc::var("EVOVIEW_UPSCALER").map_or(false, |v| v.eq_ignore_ascii_case("dlss"))
}

/// (perf-quality enum value, render scale) for EVOVIEW_DLSS_QUALITY.
pub fn quality() -> (i32, f32) {
    match crate::envc::var("EVOVIEW_DLSS_QUALITY").unwrap_or_default().to_ascii_lowercase().as_str() {
        "dlaa" => (5, 1.0),
        "balanced" => (1, 0.58),
        "performance" | "perf" => (0, 0.5),
        "ultraperformance" | "ultra_performance" => (3, 0.333),
        "ultraquality" => (4, 0.77),
        _ => (2, 0.667),
    }
}

/// Startup: when DLSS is asked for and no EVOVIEW_RENDER_SCALE is given,
/// derive it from the quality mode, so the render size, the texture LOD bias
/// and taa.rs all agree. (taa::render_scale clamps to 0.5..1.)
pub fn apply_render_scale() {
    if wanted() && crate::envc::var_os("EVOVIEW_RENDER_SCALE").is_none() {
        crate::envc::set_var("EVOVIEW_RENDER_SCALE", format!("{}", quality().1));
    }
}

/// Where nvngx_dlss.dll is looked for: EVOVIEW_DLSS_DIR,
/// then the exe's folder.
fn dirs() -> Vec<std::path::PathBuf> {
    crate::envc::var_os("EVOVIEW_DLSS_DIR").map(Into::into).into_iter()
        .chain(std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())))
        .collect()
}

mod ffi {
    #![allow(non_snake_case, dead_code)]
    use ash::vk;
    use std::ffi::{c_char, c_void};

    pub type Res = u32;
    pub fn ok(r: Res) -> bool {
        r & 0xFFF0_0000 != 0xBAD0_0000
    }
    #[repr(C)]
    pub struct Param {
        _p: [u8; 0],
    }
    #[repr(C)]
    pub struct Handle {
        _p: [u8; 0],
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct ImageViewInfo {
        pub view: vk::ImageView,
        pub image: vk::Image,
        pub range: vk::ImageSubresourceRange,
        pub format: vk::Format,
        pub width: u32,
        pub height: u32,
    }
    #[repr(C)]
    pub struct ResourceVk {
        pub info: ImageViewInfo,
        /// 0 = image view
        pub ty: i32,
        pub read_write: bool,
    }
    #[repr(C)]
    pub struct PathListInfo {
        pub path: *const *const u16,
        pub length: u32,
    }
    pub type LogCb = unsafe extern "C" fn(*const c_char, i32, i32);
    #[repr(C)]
    pub struct LoggingInfo {
        pub cb: Option<LogCb>,
        pub level: i32,
        pub disable_other: bool,
    }
    #[repr(C)]
    pub struct FeatureCommonInfo {
        pub paths: PathListInfo,
        pub internal: *mut c_void,
        pub logging: LoggingInfo,
    }
    pub const VERSION_API: i32 = 0x15;
    pub const FEATURE_SUPER_SAMPLING: i32 = 1;

    type P = *mut c_void;

    /// The NGX runtime NVIDIA's driver installs (`_nvngx.dll`), called
    /// directly: no NVIDIA SDK code is linked into or shipped with evoview.
    /// Its exports are the SDK's Vulkan entry points; the SDK's static
    /// library is only a loader in front of them. What it adds, done here:
    /// - Init_with_ProjectID -> the driver's Init_ProjectID_Ext (or
    ///   Init_ProjectID without the loader pointers), whose last two
    ///   arguments are the SDK version THEN the feature info (swapped from
    ///   the public header);
    /// - CreateFeature1 first fills the DLSS render-preset hints the app
    ///   left unset with the SDK's "default" values (populate_defaults);
    /// - the NVSDK_NGX_Parameter_Set*/Get* helpers are calls into the
    ///   parameter object's vtable. MSVC lays out overloaded virtuals in
    ///   reverse order of declaration, which the driver's own tables
    ///   confirm: Set(void*) 0, Set(int) 3, Set(unsigned) 4, Set(double) 5,
    ///   Set(float) 6, Set(u64) 7; Get(void**) 8, Get(int*) 11,
    ///   Get(unsigned*) 12, Get(double*) 13, Get(float*) 14, Get(u64*) 15;
    ///   Reset 16.
    pub struct Api {
        _lib: libloading::Library,
        required_extensions: Option<unsafe extern "C" fn(*mut u32, *mut *const *const c_char, *mut u32, *mut *const *const c_char) -> Res>,
        init_ext: Option<unsafe extern "C" fn(*const c_char, i32, *const c_char, *const u16, vk::Instance, vk::PhysicalDevice,
                                              vk::Device, P, P, i32, *const FeatureCommonInfo) -> Res>,
        init: unsafe extern "C" fn(*const c_char, i32, *const c_char, *const u16, vk::Instance, vk::PhysicalDevice,
                                   vk::Device, i32, *const FeatureCommonInfo) -> Res,
        get_capability_parameters: unsafe extern "C" fn(*mut *mut Param) -> Res,
        create_feature1: unsafe extern "C" fn(vk::Device, vk::CommandBuffer, i32, *mut Param, *mut *mut Handle) -> Res,
        evaluate_feature: unsafe extern "C" fn(vk::CommandBuffer, *const Handle, *const Param, P) -> Res,
        release_feature: unsafe extern "C" fn(*mut Handle) -> Res,
        destroy_parameters: unsafe extern "C" fn(*mut Param) -> Res,
        shutdown1: unsafe extern "C" fn(vk::Device) -> Res,
    }
    unsafe impl Send for Api {}
    unsafe impl Sync for Api {}

    static API: std::sync::OnceLock<Result<Api, String>> = std::sync::OnceLock::new();

    #[link(name = "advapi32")]
    extern "system" {
        fn RegGetValueW(key: isize, sub: *const u16, value: *const u16, flags: u32, ty: *mut u32, data: *mut c_void, len: *mut u32) -> i32;
    }

    /// The driver's NGX folder: EVOVIEW_NGX_CORE, else
    /// HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore\FullPath.
    fn core_dir() -> Option<std::path::PathBuf> {
        if let Some(d) = crate::envc::var_os("EVOVIEW_NGX_CORE") {
            return Some(d.into());
        }
        let w = |s: &str| s.encode_utf16().chain(Some(0)).collect::<Vec<u16>>();
        let (sub, val) = (w(r"SOFTWARE\NVIDIA Corporation\Global\NGXCore"), w("FullPath"));
        let mut buf = vec![0u16; 1024];
        let mut len = (buf.len() * 2) as u32;
        // HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ
        let rc = unsafe { RegGetValueW(0x8000_0002u32 as i32 as isize, sub.as_ptr(), val.as_ptr(), 0x2, std::ptr::null_mut(), buf.as_mut_ptr().cast(), &mut len) };
        if rc != 0 {
            return None;
        }
        let n = (len as usize / 2).saturating_sub(1);
        Some(String::from_utf16_lossy(&buf[..n]).into())
    }

    pub fn load() -> Result<&'static Api, String> {
        API.get_or_init(|| unsafe {
            let dir = core_dir().ok_or("NVIDIA's NGX runtime is not installed (no NGXCore in the registry: an NVIDIA driver is needed)")?;
            let path = dir.join("_nvngx.dll");
            let lib = libloading::Library::new(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            macro_rules! f {
                ($n:literal) => {
                    *lib.get($n).map_err(|e| format!("_nvngx.dll: {e}"))?
                };
            }
            macro_rules! opt {
                ($n:literal) => {
                    lib.get($n).ok().map(|s| *s)
                };
            }
            Ok(Api {
                required_extensions: opt!(b"NVSDK_NGX_VULKAN_RequiredExtensions"),
                init_ext: opt!(b"NVSDK_NGX_VULKAN_Init_ProjectID_Ext"),
                init: f!(b"NVSDK_NGX_VULKAN_Init_ProjectID"),
                get_capability_parameters: f!(b"NVSDK_NGX_VULKAN_GetCapabilityParameters"),
                create_feature1: f!(b"NVSDK_NGX_VULKAN_CreateFeature1"),
                evaluate_feature: f!(b"NVSDK_NGX_VULKAN_EvaluateFeature"),
                release_feature: f!(b"NVSDK_NGX_VULKAN_ReleaseFeature"),
                destroy_parameters: f!(b"NVSDK_NGX_VULKAN_DestroyParameters"),
                shutdown1: f!(b"NVSDK_NGX_VULKAN_Shutdown1"),
                _lib: lib,
            })
        }).as_ref().map_err(|e| e.clone())
    }

    fn api() -> &'static Api {
        load().expect("NGX used before load()")
    }

    /// What NGX needs on top of wgpu's Vulkan: the driver's list, or the
    /// fixed one the SDK reports (its own RequiredExtensions is a constant).
    pub unsafe fn NVSDK_NGX_VULKAN_RequiredExtensions(ic: *mut u32, ie: *mut *const *const c_char, dc: *mut u32, de: *mut *const *const c_char) -> Res {
        if let Some(f) = api().required_extensions {
            let r = f(ic, ie, dc, de);
            if ok(r) && *dc > 0 {
                return r;
            }
        }
        struct Lists([*const c_char; 1], [*const c_char; 4]);
        unsafe impl Sync for Lists {}
        static L: Lists = Lists(
            [c"VK_KHR_get_physical_device_properties2".as_ptr()],
            [c"VK_NVX_binary_import".as_ptr(), c"VK_NVX_image_view_handle".as_ptr(),
             c"VK_EXT_buffer_device_address".as_ptr(), c"VK_KHR_push_descriptor".as_ptr()],
        );
        *ic = 1;
        *ie = L.0.as_ptr();
        *dc = 4;
        *de = L.1.as_ptr();
        1
    }
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn NVSDK_NGX_VULKAN_Init_with_ProjectID(project: *const c_char, engine: i32, ver: *const c_char, data: *const u16,
                                                       inst: vk::Instance, pd: vk::PhysicalDevice, dev: vk::Device,
                                                       gipa: Option<vk::PFN_vkGetInstanceProcAddr>, gdpa: Option<vk::PFN_vkGetDeviceProcAddr>,
                                                       info: *const FeatureCommonInfo, version: i32) -> Res {
        let a = api();
        match (a.init_ext, gipa, gdpa) {
            (Some(f), Some(gi), Some(gd)) => f(project, engine, ver, data, inst, pd, dev, gi as P, gd as P, version, info),
            _ => (a.init)(project, engine, ver, data, inst, pd, dev, version, info),
        }
    }
    pub unsafe fn NVSDK_NGX_VULKAN_GetCapabilityParameters(out: *mut *mut Param) -> Res {
        (api().get_capability_parameters)(out)
    }
    pub unsafe fn NVSDK_NGX_VULKAN_CreateFeature1(dev: vk::Device, cmd: vk::CommandBuffer, f: i32, p: *mut Param, out: *mut *mut Handle) -> Res {
        populate_defaults(p);
        (api().create_feature1)(dev, cmd, f, p, out)
    }
    pub unsafe fn NVSDK_NGX_VULKAN_EvaluateFeature_C(cmd: vk::CommandBuffer, h: *const Handle, p: *const Param, _cb: *const c_void) -> Res {
        (api().evaluate_feature)(cmd, h, p, std::ptr::null_mut())
    }
    pub unsafe fn NVSDK_NGX_VULKAN_ReleaseFeature(h: *mut Handle) -> Res {
        (api().release_feature)(h)
    }
    pub unsafe fn NVSDK_NGX_VULKAN_DestroyParameters(p: *mut Param) -> Res {
        (api().destroy_parameters)(p)
    }
    pub unsafe fn NVSDK_NGX_VULKAN_Shutdown1(dev: vk::Device) -> Res {
        (api().shutdown1)(dev)
    }

    /// Slot `k` of the parameter object's vtable.
    unsafe fn slot(p: *const Param, k: usize) -> *const c_void {
        let vt = *(p as *const *const *const c_void);
        *vt.add(k)
    }
    pub unsafe fn NVSDK_NGX_Parameter_SetI(p: *mut Param, n: *const c_char, v: i32) {
        let f: unsafe extern "C" fn(*mut Param, *const c_char, i32) = std::mem::transmute(slot(p, 3));
        f(p, n, v)
    }
    pub unsafe fn NVSDK_NGX_Parameter_SetUI(p: *mut Param, n: *const c_char, v: u32) {
        let f: unsafe extern "C" fn(*mut Param, *const c_char, u32) = std::mem::transmute(slot(p, 4));
        f(p, n, v)
    }
    pub unsafe fn NVSDK_NGX_Parameter_SetF(p: *mut Param, n: *const c_char, v: f32) {
        let f: unsafe extern "C" fn(*mut Param, *const c_char, f32) = std::mem::transmute(slot(p, 6));
        f(p, n, v)
    }
    pub unsafe fn NVSDK_NGX_Parameter_SetVoidPointer(p: *mut Param, n: *const c_char, v: *mut c_void) {
        let f: unsafe extern "C" fn(*mut Param, *const c_char, *mut c_void) = std::mem::transmute(slot(p, 0));
        f(p, n, v)
    }
    pub unsafe fn NVSDK_NGX_Parameter_GetI(p: *mut Param, n: *const c_char, v: *mut i32) -> Res {
        let f: unsafe extern "C" fn(*mut Param, *const c_char, *mut i32) -> Res = std::mem::transmute(slot(p, 11));
        f(p, n, v)
    }
    unsafe fn get_ui(p: *mut Param, n: *const c_char, v: *mut u32) -> Res {
        let f: unsafe extern "C" fn(*mut Param, *const c_char, *mut u32) -> Res = std::mem::transmute(slot(p, 12));
        f(p, n, v)
    }

    /// The SDK's defaults for render-preset hints the app did not set.
    unsafe fn populate_defaults(p: *mut Param) {
        for (n, v) in [(c"DLSS.Hint.Render.Preset.DLAA", 0x8000_000bu32), (c"DLSS.Hint.Render.Preset.Quality", 0x8000_000b),
                       (c"DLSS.Hint.Render.Preset.Balanced", 0x8000_000b), (c"DLSS.Hint.Render.Preset.Performance", 0x8000_000d),
                       (c"DLSS.Hint.Render.Preset.UltraQuality", 0x8000_000b), (c"DLSS.Hint.Render.Preset.UltraPerformance", 0x8000_000c)] {
            let mut have = 0u32;
            if !ok(get_ui(p, n.as_ptr(), &mut have)) {
                NVSDK_NGX_Parameter_SetUI(p, n.as_ptr(), v);
            }
        }
    }
}

unsafe extern "C" fn ngx_log(msg: *const std::ffi::c_char, _level: i32, _src: i32) {
    if !msg.is_null() {
        eprint!("NGX: {}", std::ffi::CStr::from_ptr(msg).to_string_lossy());
    }
}

/// Open the device with NGX's Vulkan extensions on top of wgpu's. None when
/// DLSS is not wanted / not compiled / not NVIDIA / an extension is missing:
/// the caller opens the device the usual way.
pub fn request_device(adapter: &wgpu::Adapter, desc: &wgpu::DeviceDescriptor) -> Option<(wgpu::Device, wgpu::Queue)> {
    if !wanted() {
        return None;
    }
    if let Err(e) = ffi::load() {
        eprintln!("DLSS: {e} - TAA");
        return None;
    }
    if adapter.get_info().vendor != 0x10de {
        eprintln!("DLSS: not an NVIDIA GPU ({}) - TAA", adapter.get_info().name);
        return None;
    }
    open_device(adapter, desc)
}

fn open_device(adapter: &wgpu::Adapter, desc: &wgpu::DeviceDescriptor) -> Option<(wgpu::Device, wgpu::Queue)> {
    use ash::vk;
    use std::ffi::CStr;
    use wgpu::hal::api::Vulkan;
    // NGX's instance + device extensions
    let (mut ic, mut dc) = (0u32, 0u32);
    let (mut ie, mut de): (*const *const std::ffi::c_char, *const *const std::ffi::c_char) = (std::ptr::null(), std::ptr::null());
    let r = unsafe { ffi::NVSDK_NGX_VULKAN_RequiredExtensions(&mut ic, &mut ie, &mut dc, &mut de) };
    if !ffi::ok(r) {
        eprintln!("DLSS: NVSDK_NGX_VULKAN_RequiredExtensions failed {r:#x} - TAA");
        return None;
    }
    let list = |n: u32, p: *const *const std::ffi::c_char| -> Vec<&'static CStr> {
        (0..n as usize).map(|i| unsafe { CStr::from_ptr(*p.add(i)) }).collect()
    };
    let inst_exts = list(ic, ie);
    let dev_exts = list(dc, de);
    println!("DLSS: NGX wants instance {:?}, device {:?}", inst_exts, dev_exts);
    let opened = unsafe {
        adapter.as_hal::<Vulkan, _, _>(|a| {
            let a = a?;
            let have_inst = a.shared_instance().extensions();
            for e in &inst_exts {
                // promoted to core by Vulkan 1.1, which wgpu asks for
                if !have_inst.contains(e) && a.shared_instance().instance_api_version() < vk::API_VERSION_1_1 {
                    eprintln!("DLSS: instance extension {e:?} not enabled by wgpu - TAA");
                    return None;
                }
            }
            let mut exts = a.required_device_extensions(desc.required_features);
            let raw_inst = a.shared_instance().raw_instance();
            let avail = raw_inst.enumerate_device_extension_properties(a.raw_physical_device()).ok()?;
            for e in &dev_exts {
                if exts.contains(e) {
                    continue;
                }
                if !avail.iter().any(|p| p.extension_name_as_c_str().ok() == Some(*e)) {
                    eprintln!("DLSS: device extension {e:?} not available - TAA");
                    return None;
                }
                exts.push(e);
            }
            let mut feats = a.physical_device_features(&exts, desc.required_features);
            let family_index = 0;
            let prio = [1.0f32];
            let fam = [vk::DeviceQueueCreateInfo::default().queue_family_index(family_index).queue_priorities(&prio)];
            let names: Vec<*const std::ffi::c_char> = exts.iter().map(|s| s.as_ptr()).collect();
            // NGX may ask for buffer device address; its feature bit has to be on
            let mut bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
            let want_bda = exts.iter().any(|e| e.to_bytes().ends_with(b"buffer_device_address"));
            let mut info = feats.add_to_device_create(
                vk::DeviceCreateInfo::default().queue_create_infos(&fam).enabled_extension_names(&names));
            if want_bda {
                info = info.push_next(&mut bda);
            }
            let raw = match raw_inst.create_device(a.raw_physical_device(), &info, None) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("DLSS: vkCreateDevice with NGX's extensions failed: {e:?} - TAA");
                    return None;
                }
            };
            a.device_from_raw(raw, true, &exts, desc.required_features, &desc.memory_hints, family_index, 0).ok()
        })
    }?;
    match unsafe { adapter.create_device_from_hal::<Vulkan>(opened, desc, None) } {
        Ok(dq) => Some(dq),
        Err(e) => {
            eprintln!("DLSS: create_device_from_hal: {e} - TAA");
            None
        }
    }
}

struct Feature {
    handle: *mut ffi::Handle,
    rsize: (u32, u32),
    size: (u32, u32),
}

pub struct Dlss {
        params: *mut ffi::Param,
        feature: Option<Feature>,
        raw_device: ash::vk::Device,
    /// DLSS's output (screen size, storage)
    pub out: wgpu::TextureView,
    out_tex: wgpu::Texture,
    /// the no-op dispatch that puts the inputs / output in DLSS's layouts
    tr_pipe: wgpu::ComputePipeline,
    tr_bgl: wgpu::BindGroupLayout,
    size: (u32, u32),
    quality: i32,
    /// the next evaluate drops DLSS's history
    reset: bool,
}

pub const OUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

impl Dlss {
    /// Initialise NGX on `device` (opened by `request_device`) and check DLSS
    /// is available. Err: why not (the caller stays on TAA).
    pub fn new(device: &wgpu::Device, w: u32, h: u32, color_format: wgpu::TextureFormat) -> Result<Dlss, String> {
                unsafe {
            use wgpu::hal::api::Vulkan;
            let dirs = dirs();
            if !dirs.iter().any(|d| d.join("nvngx_dlss.dll").exists()) {
                return Err(format!("nvngx_dlss.dll not found in {:?} (download it from NVIDIA: github.com/NVIDIA/DLSS, lib/Windows_x86_64/rel)", dirs));
            }
            let wide = |s: &std::ffi::OsStr| -> Vec<u16> {
                use std::os::windows::ffi::OsStrExt;
                s.encode_wide().chain(std::iter::once(0)).collect()
            };
            let dir_w: Vec<Vec<u16>> = dirs.iter().map(|d| wide(d.as_os_str())).collect();
            let dir_p: Vec<*const u16> = dir_w.iter().map(|v| v.as_ptr()).collect();
            let data = std::env::temp_dir().join("evoview_ngx");
            let _ = std::fs::create_dir_all(&data);
            let data_w = wide(data.as_os_str());
            let log = crate::envc::var_os("EVOVIEW_DLSS_LOG").is_some();
            let info = ffi::FeatureCommonInfo {
                paths: ffi::PathListInfo { path: dir_p.as_ptr(), length: dir_p.len() as u32 },
                internal: std::ptr::null_mut(),
                logging: ffi::LoggingInfo { cb: if log { Some(ngx_log) } else { None }, level: if log { 1 } else { 0 }, disable_other: false },
            };
            let handles = device.as_hal::<Vulkan, _, _>(|d| d.map(|d| {
                let inst = d.shared_instance();
                (inst.raw_instance().handle(), d.raw_physical_device(), d.raw_device().handle(),
                 inst.entry().static_fn().get_instance_proc_addr, inst.raw_instance().fp_v1_0().get_device_proc_addr)
            })).flatten().ok_or("not a Vulkan device")?;
            let (inst, pd, dev, gipa, gdpa) = handles;
            let r = ffi::NVSDK_NGX_VULKAN_Init_with_ProjectID(
                c"8a9c0e2b-6f1d-4c3e-9b7a-0e7e01e7f00d".as_ptr(), 0, c"evoview".as_ptr(), data_w.as_ptr(),
                inst, pd, dev, Some(gipa), Some(gdpa), &info, ffi::VERSION_API);
            if !ffi::ok(r) {
                return Err(format!("NVSDK_NGX_VULKAN_Init failed {r:#x} (EVOVIEW_DLSS_LOG=1 for NGX's log)"));
            }
            let mut params = std::ptr::null_mut();
            let r = ffi::NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut params);
            if !ffi::ok(r) || params.is_null() {
                ffi::NVSDK_NGX_VULKAN_Shutdown1(dev);
                return Err(format!("GetCapabilityParameters failed {r:#x}"));
            }
            let mut avail = 0i32;
            ffi::NVSDK_NGX_Parameter_GetI(params, c"SuperSampling.Available".as_ptr(), &mut avail);
            if avail == 0 {
                let mut why = 0i32;
                ffi::NVSDK_NGX_Parameter_GetI(params, c"SuperSampling.FeatureInitResult".as_ptr(), &mut why);
                let mut drv = 0i32;
                ffi::NVSDK_NGX_Parameter_GetI(params, c"SuperSampling.NeedsUpdatedDriver".as_ptr(), &mut drv);
                ffi::NVSDK_NGX_VULKAN_DestroyParameters(params);
                ffi::NVSDK_NGX_VULKAN_Shutdown1(dev);
                return Err(format!("DLSS not available (init result {why:#x}, needs newer driver {drv})"));
            }
            let _ = color_format;
            let (out_tex, out) = Self::output(device, w, h);
            let (tr_pipe, tr_bgl) = Self::transition_pipe(device);
            Ok(Dlss { params, feature: None, raw_device: dev, out, out_tex, tr_pipe, tr_bgl, size: (w, h), quality: quality().0, reset: true })
        }
    }

    fn output(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let t = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dlss output"),
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

    #[allow(dead_code)]
    fn transition_pipe(device: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let tex = |b: u32, st: wgpu::TextureSampleType| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture { sample_type: st, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dlss transitions"),
            entries: &[
                tex(0, wgpu::TextureSampleType::Float { filterable: false }),
                tex(1, wgpu::TextureSampleType::Depth),
                tex(2, wgpu::TextureSampleType::Float { filterable: false }),
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
            label: Some("dlss transitions"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dlss transitions"),
            source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
        });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dlss transitions"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });
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

    /// Drop DLSS's history on the next evaluate (camera cut).
    pub fn reset(&mut self) {
        self.reset = true;
    }

    /// Upscale `color` (render size, jittered by `jitter` render pixels in
    /// DLSS's convention: x right, y down) into `self.out`.
    #[allow(clippy::too_many_arguments)]
    pub fn run(&mut self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder,
               color: (&wgpu::Texture, &wgpu::TextureView), depth: (&wgpu::Texture, &wgpu::TextureView),
               mv: (&wgpu::Texture, &wgpu::TextureView), jitter: (f32, f32)) -> Result<(), String> {
        // the layouts DLSS reads / writes them in
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dlss transitions"),
            layout: &self.tr_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(color.1) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth.1) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(mv.1) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.out) },
            ],
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("dlss transitions"), timestamp_writes: None });
            cp.set_pipeline(&self.tr_pipe);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }
                unsafe {
            use ash::vk;
            use wgpu::hal::api::Vulkan;
            let rsize = (color.0.width(), color.0.height());
            let res = |t: &wgpu::Texture, v: &wgpu::TextureView, rw: bool| -> Option<ffi::ResourceVk> {
                let image = t.as_hal::<Vulkan, _, _>(|t| t.map(|t| t.raw_handle()))?;
                let view = v.as_hal::<Vulkan, _, _>(|v| v.map(|v| v.raw_handle()))?;
                let (format, aspect) = match t.format() {
                    wgpu::TextureFormat::Rgba16Float => (vk::Format::R16G16B16A16_SFLOAT, vk::ImageAspectFlags::COLOR),
                    wgpu::TextureFormat::Rg16Float => (vk::Format::R16G16_SFLOAT, vk::ImageAspectFlags::COLOR),
                    wgpu::TextureFormat::Depth32Float => (vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH),
                    wgpu::TextureFormat::Rgba8UnormSrgb => (vk::Format::R8G8B8A8_SRGB, vk::ImageAspectFlags::COLOR),
                    wgpu::TextureFormat::Bgra8UnormSrgb => (vk::Format::B8G8R8A8_SRGB, vk::ImageAspectFlags::COLOR),
                    _ => return None,
                };
                Some(ffi::ResourceVk {
                    info: ffi::ImageViewInfo {
                        view, image,
                        range: vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 },
                        format, width: t.width(), height: t.height(),
                    },
                    ty: 0,
                    read_write: rw,
                })
            };
            let mut r_color = res(color.0, color.1, false).ok_or("colour format")?;
            let mut r_depth = res(depth.0, depth.1, false).ok_or("depth format")?;
            let mut r_mv = res(mv.0, mv.1, false).ok_or("motion format")?;
            let mut r_out = res(&self.out_tex, &self.out, true).ok_or("output format")?;
            let hdr = color.0.format() == wgpu::TextureFormat::Rgba16Float;
            let p = self.params;
            let dev = self.raw_device;
            let size = self.size;
            let quality = self.quality;
            let need_new = self.feature.as_ref().map_or(true, |f| f.rsize != rsize || f.size != size);
            let old = if need_new { self.feature.take() } else { None };
            let reset = std::mem::replace(&mut self.reset, false) || need_new;
            let out = enc.as_hal_mut::<Vulkan, _, _>(|e| -> Result<Option<Feature>, String> {
                let cmd = e.ok_or("no Vulkan encoder")?.raw_handle();
                let mut created = None;
                if need_new {
                    if let Some(f) = old {
                        ffi::NVSDK_NGX_VULKAN_ReleaseFeature(f.handle);
                    }
                    ffi::NVSDK_NGX_Parameter_SetUI(p, c"CreationNodeMask".as_ptr(), 1);
                    ffi::NVSDK_NGX_Parameter_SetUI(p, c"VisibilityNodeMask".as_ptr(), 1);
                    ffi::NVSDK_NGX_Parameter_SetUI(p, c"Width".as_ptr(), rsize.0);
                    ffi::NVSDK_NGX_Parameter_SetUI(p, c"Height".as_ptr(), rsize.1);
                    ffi::NVSDK_NGX_Parameter_SetUI(p, c"OutWidth".as_ptr(), size.0);
                    ffi::NVSDK_NGX_Parameter_SetUI(p, c"OutHeight".as_ptr(), size.1);
                    ffi::NVSDK_NGX_Parameter_SetI(p, c"PerfQualityValue".as_ptr(), quality);
                    // IsHDR | MVLowRes | AutoExposure
                    let flags = (if hdr { 1 } else { 0 }) | (1 << 1) | (if hdr { 1 << 6 } else { 0 });
                    ffi::NVSDK_NGX_Parameter_SetI(p, c"DLSS.Feature.Create.Flags".as_ptr(), flags);
                    ffi::NVSDK_NGX_Parameter_SetI(p, c"DLSS.Enable.Output.Subrects".as_ptr(), 0);
                    // EVOVIEW_DLSS_PRESET = e | f (the older CNN model, deprecated, cheaper)
                    // | j | k | l | m (transformer); default: NVIDIA's per mode
                    if let Some(pr) = crate::envc::var("EVOVIEW_DLSS_PRESET").ok().and_then(|v| match v.to_ascii_lowercase().as_str() {
                        "e" => Some(5u32), "f" => Some(6), "j" => Some(10), "k" => Some(11), "l" => Some(12), "m" => Some(13), _ => None }) {
                        for n in [c"DLSS.Hint.Render.Preset.DLAA", c"DLSS.Hint.Render.Preset.Quality", c"DLSS.Hint.Render.Preset.Balanced",
                                  c"DLSS.Hint.Render.Preset.Performance", c"DLSS.Hint.Render.Preset.UltraPerformance",
                                  c"DLSS.Hint.Render.Preset.UltraQuality"] {
                            ffi::NVSDK_NGX_Parameter_SetUI(p, n.as_ptr(), pr);
                        }
                    }
                    let mut h = std::ptr::null_mut();
                    let r = ffi::NVSDK_NGX_VULKAN_CreateFeature1(dev, cmd, ffi::FEATURE_SUPER_SAMPLING, p, &mut h);
                    if !ffi::ok(r) || h.is_null() {
                        return Err(format!("CreateFeature failed {r:#x}"));
                    }
                    println!("DLSS: {}x{} -> {}x{}, mode {quality}{}", rsize.0, rsize.1, size.0, size.1, if hdr { ", HDR" } else { "" });
                    created = Some(Feature { handle: h, rsize, size });
                }
                Ok(created)
            }).ok_or("as_hal_mut")??;
            if let Some(f) = out {
                self.feature = Some(f);
            }
            let handle = self.feature.as_ref().ok_or("no feature")?.handle;
            let sx = crate::envc::var("EVOVIEW_DLSS_JX").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
            let sy = crate::envc::var("EVOVIEW_DLSS_JY").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
            enc.as_hal_mut::<Vulkan, _, _>(|e| -> Result<(), String> {
                let cmd = e.ok_or("no Vulkan encoder")?.raw_handle();
                let vp = |r: &mut ffi::ResourceVk| r as *mut ffi::ResourceVk as *mut std::ffi::c_void;
                ffi::NVSDK_NGX_Parameter_SetVoidPointer(p, c"Color".as_ptr(), vp(&mut r_color));
                ffi::NVSDK_NGX_Parameter_SetVoidPointer(p, c"Output".as_ptr(), vp(&mut r_out));
                ffi::NVSDK_NGX_Parameter_SetVoidPointer(p, c"Depth".as_ptr(), vp(&mut r_depth));
                ffi::NVSDK_NGX_Parameter_SetVoidPointer(p, c"MotionVectors".as_ptr(), vp(&mut r_mv));
                ffi::NVSDK_NGX_Parameter_SetF(p, c"Jitter.Offset.X".as_ptr(), jitter.0 * sx);
                ffi::NVSDK_NGX_Parameter_SetF(p, c"Jitter.Offset.Y".as_ptr(), jitter.1 * sy);
                ffi::NVSDK_NGX_Parameter_SetF(p, c"Sharpness".as_ptr(), 0.0);
                ffi::NVSDK_NGX_Parameter_SetI(p, c"Reset".as_ptr(), reset as i32);
                ffi::NVSDK_NGX_Parameter_SetF(p, c"MV.Scale.X".as_ptr(), 1.0);
                ffi::NVSDK_NGX_Parameter_SetF(p, c"MV.Scale.Y".as_ptr(), 1.0);
                ffi::NVSDK_NGX_Parameter_SetUI(p, c"DLSS.Render.Subrect.Dimensions.Width".as_ptr(), rsize.0);
                ffi::NVSDK_NGX_Parameter_SetUI(p, c"DLSS.Render.Subrect.Dimensions.Height".as_ptr(), rsize.1);
                ffi::NVSDK_NGX_Parameter_SetF(p, c"DLSS.Pre.Exposure".as_ptr(), 1.0);
                ffi::NVSDK_NGX_Parameter_SetF(p, c"DLSS.Exposure.Scale".as_ptr(), 1.0);
                ffi::NVSDK_NGX_Parameter_SetVoidPointer(p, c"ExposureTexture".as_ptr(), std::ptr::null_mut());
                let r = ffi::NVSDK_NGX_VULKAN_EvaluateFeature_C(cmd, handle, p, std::ptr::null());
                if !ffi::ok(r) {
                    return Err(format!("EvaluateFeature failed {r:#x}"));
                }
                Ok(())
            }).ok_or("as_hal_mut")??;
            Ok(())
        }
    }
}

impl Dlss {
    /// Release the feature and NGX once the GPU is idle; before the device
    /// is dropped (left to the process exit, the driver crashed on the way out).
    pub fn shutdown(mut self, device: &wgpu::Device) {
        device.poll(wgpu::Maintain::Wait);
        unsafe {
            if let Some(f) = self.feature.take() {
                ffi::NVSDK_NGX_VULKAN_ReleaseFeature(f.handle);
            }
            ffi::NVSDK_NGX_VULKAN_DestroyParameters(self.params);
            ffi::NVSDK_NGX_VULKAN_Shutdown1(self.raw_device);
        }
    }
}
