//! Load-phase timers and process memory probes for `--bench`.
//!
//! Timers are always on: one mutex lock per call, a few thousand calls on the
//! biggest track, which is noise next to the work being timed. Phases that run
//! on worker threads ACCUMULATE thread time, so a parallel phase can exceed
//! the wall-clock load time - that is the point, it shows the work done.

use std::sync::Mutex;
use std::time::Instant;

static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// Call first thing in main; `since_start` measures from here.
pub fn mark_start() {
    let _ = START.set(Instant::now());
}

pub fn since_start() -> f64 {
    START.get().map_or(0.0, |s| s.elapsed().as_secs_f64())
}

static PHASES: Mutex<Vec<(&'static str, f64, u32)>> = Mutex::new(Vec::new());

pub fn add(name: &'static str, secs: f64) {
    let mut p = PHASES.lock().unwrap();
    match p.iter_mut().find(|e| e.0 == name) {
        Some(e) => {
            e.1 += secs;
            e.2 += 1;
        }
        None => p.push((name, secs, 1)),
    }
}

/// EVOVIEW_LOAD_LOG=1: the start-up as a timeline - `stamp`s, every timed
/// phase's end and the VRAM checkpoints, in seconds since process start.
fn load_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::envc::var_os("EVOVIEW_LOAD_LOG").is_some())
}

/// A start-up checkpoint (EVOVIEW_LOAD_LOG=1).
pub fn stamp(what: &str) {
    if load_log() {
        println!("load @ {:7.2} s: {what}", since_start());
    }
}

/// Time `f` under `name`.
pub fn time<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    add(name, t.elapsed().as_secs_f64());
    if load_log() && t.elapsed().as_secs_f64() > 0.1 {
        stamp(&format!("{name} ({:.2} s)", t.elapsed().as_secs_f64()));
    }
    r
}

/// EVOVIEW_VRAM_LOG=1: this process's dedicated VRAM after a step (a PDH
/// query over every process: ~ms, startup checkpoints only).
pub fn vram_note(what: &str) {
    stamp(what);
    if crate::envc::var_os("EVOVIEW_VRAM_LOG").is_some() {
        if let Some(v) = vram_bytes() {
            println!("vram: {:.0} MB after {what}", v as f64 / 1048576.0);
        }
    }
}

static BYTES: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());

/// Count bytes moved under `name` (reported in MB beside the timings).
pub fn add_bytes(name: &'static str, n: u64) {
    let mut b = BYTES.lock().unwrap();
    match b.iter_mut().find(|e| e.0 == name) {
        Some(e) => e.1 += n,
        None => b.push((name, n)),
    }
}

pub fn bytes() -> Vec<(&'static str, u64)> {
    BYTES.lock().unwrap().clone()
}

/// (name, seconds, calls) in first-seen order.
pub fn phases() -> Vec<(&'static str, f64, u32)> {
    PHASES.lock().unwrap().clone()
}

/// Process memory: (peak working set, peak commit) in bytes.
pub fn peak_ram() -> (u64, u64) {
    let m = ram();
    (m.0, m.1)
}

/// Process memory: (peak working set, peak commit, current working set).
#[cfg(windows)]
pub fn ram() -> (u64, u64, u64) {
    #[repr(C)]
    #[derive(Default)]
    struct Pmc {
        cb: u32,
        faults: u32,
        peak_ws: usize,
        ws: usize,
        qpp: usize,
        qp: usize,
        qpnp: usize,
        qnp: usize,
        pagefile: usize,
        peak_pagefile: usize,
    }
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(p: isize, c: *mut Pmc, cb: u32) -> i32;
    }
    let mut c = Pmc { cb: std::mem::size_of::<Pmc>() as u32, ..Default::default() };
    // SAFETY: plain Win32 call filling a correctly sized struct.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    if ok == 0 {
        return (0, 0, 0);
    }
    (c.peak_ws as u64, c.peak_pagefile as u64, c.ws as u64)
}
#[cfg(not(windows))]
pub fn ram() -> (u64, u64, u64) {
    (0, 0, 0)
}

/// Dedicated VRAM this process holds, from the OS GPU counters
/// (`\GPU Process Memory(pid_N_*)\Dedicated Usage`). wgpu 22 has no memory
/// report on Vulkan, and these are what Task Manager shows. Covers every
/// adapter the process touched.
pub fn vram_bytes() -> Option<u64> {
    let prefix = format!("pid_{}_", std::process::id());
    pdh_counter("\\GPU Process Memory(*)\\Dedicated Usage")
        .map(|v| v.iter().filter(|(n, _)| n.starts_with(&prefix)).map(|(_, b)| b).sum())
}

/// Dedicated VRAM the busiest GPU has in use by EVERY process
/// (`\GPU Adapter Memory(*)\Dedicated Usage`): with `vram_bytes`, what the
/// rest of the desktop (browsers, overlays, wallpaper) holds.
#[allow(dead_code)]
pub fn vram_adapter_used() -> Option<u64> {
    pdh_counter("\\GPU Adapter Memory(*)\\Dedicated Usage").and_then(|v| v.iter().map(|(_, b)| *b).max())
}

/// Dedicated VRAM other processes hold, and the largest GPU's size: how much
/// is really free. ⚠ Not DXGI's QueryVideoMemoryInfo budget - that stays at
/// ~94% of the card however much the desktop holds (11.2 of 12 GB with 2 GB
/// used elsewhere), and going past what is physically free pages every pass
/// over PCIe: a 50-player lobby went from 14 to 340 ms a frame at +70 MB.
pub fn vram_free_for_us() -> Option<(u64, u64)> {
    let (total, luid) = largest_adapter()?;
    // all three numbers off the one adapter (its LUID is in the counters'
    // instance names): the busiest adapter's use, minus our use on every
    // adapter, against the largest one's size mixed three GPUs on a
    // dual-GPU laptop. Counters without the LUID: as before.
    let on = |n: &String| n.to_ascii_lowercase().contains(&luid);
    let adapter = pdh_counter("\\GPU Adapter Memory(*)\\Dedicated Usage")?;
    let ours_all = pdh_counter("\\GPU Process Memory(*)\\Dedicated Usage").unwrap_or_default();
    let prefix = format!("pid_{}_", std::process::id());
    let ours = |f: &dyn Fn(&String) -> bool| ours_all.iter().filter(|(n, _)| n.starts_with(&prefix) && f(n)).map(|(_, b)| b).sum::<u64>();
    let others = match adapter.iter().filter(|(n, _)| on(n)).map(|(_, b)| *b).max() {
        Some(used) => used.saturating_sub(ours(&on)),
        None => adapter.iter().map(|(_, b)| *b).max()?.saturating_sub(ours(&|_| true)),
    };
    Some((total.saturating_sub(others), total))
}

/// One PDH counter's instances: (instance name, value).
#[cfg(windows)]
fn pdh_counter(path: &str) -> Option<Vec<(String, u64)>> {
    #[repr(C)]
    struct Item {
        name: *const u16,
        status: u32,
        value: i64, // union; PDH_FMT_LARGE reads the i64 arm
    }
    static PDH: std::sync::OnceLock<Option<libloading::Library>> = std::sync::OnceLock::new();
    let lib = system32(&PDH, "pdh.dll")?;
    // SAFETY: the documented PDH signatures
    #[allow(non_snake_case)]
    let (PdhOpenQueryW, PdhAddEnglishCounterW, PdhCollectQueryData, PdhGetFormattedCounterArrayW, PdhCloseQuery) = unsafe {
        (
            *lib.get::<unsafe extern "system" fn(*const u16, usize, *mut isize) -> u32>(b"PdhOpenQueryW").ok()?,
            *lib.get::<unsafe extern "system" fn(isize, *const u16, usize, *mut isize) -> u32>(b"PdhAddEnglishCounterW").ok()?,
            *lib.get::<unsafe extern "system" fn(isize) -> u32>(b"PdhCollectQueryData").ok()?,
            *lib.get::<unsafe extern "system" fn(isize, u32, *mut u32, *mut u32, *mut Item) -> u32>(b"PdhGetFormattedCounterArrayW").ok()?,
            *lib.get::<unsafe extern "system" fn(isize) -> u32>(b"PdhCloseQuery").ok()?,
        )
    };
    const PDH_FMT_LARGE: u32 = 0x400;
    const PDH_MORE_DATA: u32 = 0x8000_07D2;
    let path: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
    // SAFETY: standard PDH query sequence; the item buffer is sized by PDH's
    // own first call and kept alive while the names it points into are read.
    unsafe {
        let mut q = 0isize;
        if PdhOpenQueryW(std::ptr::null(), 0, &mut q) != 0 {
            return None;
        }
        let mut c = 0isize;
        let mut out = None;
        if PdhAddEnglishCounterW(q, path.as_ptr(), 0, &mut c) == 0
            && PdhCollectQueryData(q) == 0
        {
            let (mut size, mut count) = (0u32, 0u32);
            let r = PdhGetFormattedCounterArrayW(
                c, PDH_FMT_LARGE, &mut size, &mut count, std::ptr::null_mut());
            if r == PDH_MORE_DATA && size > 0 {
                let mut buf = vec![0u64; (size as usize).div_ceil(8)];
                let items = buf.as_mut_ptr() as *mut Item;
                if PdhGetFormattedCounterArrayW(c, PDH_FMT_LARGE, &mut size, &mut count, items)
                    == 0
                {
                    let mut v = Vec::with_capacity(count as usize);
                    for i in 0..count as usize {
                        let it = &*items.add(i);
                        let mut n = 0;
                        while *it.name.add(n) != 0 {
                            n += 1;
                        }
                        // PDH_CSTATUS_VALID_DATA 0 / NEW_DATA 1; anything
                        // else is no reading
                        if it.status > 1 {
                            continue;
                        }
                        let name = String::from_utf16_lossy(std::slice::from_raw_parts(it.name, n));
                        v.push((name, it.value.max(0) as u64));
                    }
                    out = Some(v);
                }
            }
        }
        PdhCloseQuery(q);
        out
    }
}
#[cfg(not(windows))]
fn pdh_counter(_: &str) -> Option<Vec<(String, u64)>> {
    None
}

/// ⚠ pdh.dll / dxgi.dll were static imports, resolved from the exe's folder
/// first (neither is a KnownDLL): a stray ReShade-style dxgi.dll there was
/// loaded into every run. System32 only, on first use.
#[cfg(windows)]
fn system32(cell: &'static std::sync::OnceLock<Option<libloading::Library>>, name: &str) -> Option<&'static libloading::Library> {
    const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x800;
    cell.get_or_init(|| {
        // SAFETY: a system DLL with no special init requirements
        unsafe { libloading::os::windows::Library::load_with_flags(name, LOAD_LIBRARY_SEARCH_SYSTEM32) }.ok().map(Into::into)
    }).as_ref()
}

/// The largest GPU's dedicated memory (DXGI adapter descriptions).
#[allow(dead_code)]
pub fn vram_total() -> Option<u64> {
    largest_adapter().map(|a| a.0)
}

/// The largest GPU: its dedicated memory, and its LUID as the GPU counters
/// name it (`luid_0x<high>_0x<low>`, lower case).
#[cfg(windows)]
fn largest_adapter() -> Option<(u64, String)> {
    #[repr(C)]
    struct Guid(u32, u16, u16, [u8; 8]);
    #[repr(C)]
    struct Desc1 {
        description: [u16; 128],
        vendor: u32,
        device: u32,
        subsys: u32,
        revision: u32,
        dedicated_video: usize,
        dedicated_system: usize,
        shared_system: usize,
        luid: u64,
        flags: u32,
    }
    static DXGI: std::sync::OnceLock<Option<libloading::Library>> = std::sync::OnceLock::new();
    let lib = system32(&DXGI, "dxgi.dll")?;
    // SAFETY: the documented signature
    #[allow(non_snake_case)]
    let CreateDXGIFactory1 = unsafe {
        *lib.get::<unsafe extern "system" fn(*const Guid, *mut *mut std::ffi::c_void) -> i32>(b"CreateDXGIFactory1").ok()?
    };
    type EnumAdapters1 = unsafe extern "system" fn(*mut std::ffi::c_void, u32, *mut *mut std::ffi::c_void) -> i32;
    type GetDesc1 = unsafe extern "system" fn(*mut std::ffi::c_void, *mut Desc1) -> i32;
    type Release = unsafe extern "system" fn(*mut std::ffi::c_void) -> u32;
    // IDXGIFactory1
    let iid = Guid(0x770aae78, 0xf26f, 0x4dba, [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87]);
    // SAFETY: COM calls through the documented vtable slots (IDXGIFactory1
    // EnumAdapters1 = 12, IDXGIAdapter1 GetDesc1 = 10, Release = 2); every
    // object obtained is released.
    unsafe {
        let mut f = std::ptr::null_mut();
        if CreateDXGIFactory1(&iid, &mut f) != 0 || f.is_null() {
            return None;
        }
        let vt = |o: *mut std::ffi::c_void, i: usize| *(*(o as *const *const usize)).add(i);
        let mut best = (0u64, String::new());
        for i in 0.. {
            let mut a = std::ptr::null_mut();
            let e: EnumAdapters1 = std::mem::transmute(vt(f, 12));
            if e(f, i, &mut a) != 0 || a.is_null() {
                break;
            }
            let mut d: Desc1 = std::mem::zeroed();
            let g: GetDesc1 = std::mem::transmute(vt(a, 10));
            if g(a, &mut d) == 0 && d.dedicated_video as u64 > best.0 {
                best = (d.dedicated_video as u64, format!("luid_0x{:08x}_0x{:08x}", (d.luid >> 32) as u32, d.luid as u32));
            }
            let r: Release = std::mem::transmute(vt(a, 2));
            r(a);
        }
        let r: Release = std::mem::transmute(vt(f, 2));
        r(f);
        (best.0 > 0).then_some(best)
    }
}
#[cfg(not(windows))]
fn largest_adapter() -> Option<(u64, String)> {
    None
}
