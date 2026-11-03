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

/// Time `f` under `name`.
pub fn time<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    add(name, t.elapsed().as_secs_f64());
    r
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
#[cfg(windows)]
pub fn vram_bytes() -> Option<u64> {
    #[repr(C)]
    struct Item {
        name: *const u16,
        status: u32,
        value: i64, // union; PDH_FMT_LARGE reads the i64 arm
    }
    #[link(name = "pdh")]
    extern "system" {
        fn PdhOpenQueryW(src: *const u16, user: usize, q: *mut isize) -> u32;
        fn PdhAddEnglishCounterW(q: isize, path: *const u16, user: usize, c: *mut isize) -> u32;
        fn PdhCollectQueryData(q: isize) -> u32;
        fn PdhGetFormattedCounterArrayW(
            c: isize, fmt: u32, size: *mut u32, count: *mut u32, items: *mut Item,
        ) -> u32;
        fn PdhCloseQuery(q: isize) -> u32;
    }
    const PDH_FMT_LARGE: u32 = 0x400;
    const PDH_MORE_DATA: u32 = 0x8000_07D2;
    let path: Vec<u16> = "\\GPU Process Memory(*)\\Dedicated Usage\0".encode_utf16().collect();
    let prefix = format!("pid_{}_", std::process::id());
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
                    let mut total = 0u64;
                    for i in 0..count as usize {
                        let it = &*items.add(i);
                        let mut n = 0;
                        while *it.name.add(n) != 0 {
                            n += 1;
                        }
                        let name = String::from_utf16_lossy(std::slice::from_raw_parts(it.name, n));
                        if name.starts_with(&prefix) {
                            total += it.value.max(0) as u64;
                        }
                    }
                    out = Some(total);
                }
            }
        }
        PdhCloseQuery(q);
        out
    }
}
#[cfg(not(windows))]
pub fn vram_bytes() -> Option<u64> {
    None
}
