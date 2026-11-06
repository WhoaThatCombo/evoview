//! Keeping a session inside the VRAM the GPU really has free.
//!
//! Going past it is a cliff, not a slope: Windows pages every pass's
//! resources over PCIe and a frame goes from ~14 ms to ~340 ms (a 44-player
//! Nordschleife Touristenfahrten lobby at 3440x1440 on a 12 GB card, with
//! ~2 GB held by the desktop - browsers, overlays, a live wallpaper - was
//! 70 MB over). The other players' cars are the part a session can give up:
//! their textures drop whole top mip levels before upload until the session
//! fits. Only when it would not fit otherwise; the player's own car and the
//! track are untouched.
//!
//! The session's size is estimated before anything is on the GPU (the
//! opponents upload before the track renderer), from the data loaded:
//!   base      1.85 x (track + player texture and mesh bytes)
//!             + 84 B per output pixel - 300 MB
//!             (fitted on the Nordschleife and Imola at 3440x1440, the
//!             pixel term from 1720x720: within ~20 MB of the OS counter)
//!   opponents 1.25 x their texture and mesh bytes (materials, LOD meshes,
//!             bones, per-look buffers on top)
//! and checked against total VRAM minus what other processes hold, less
//! EVOVIEW_VRAM_MARGIN_MB (800: paging starts before the last byte, and late
//! joiners need room). EVOVIEW_VRAM_FREE_MB pretends that much is free (to
//! test); EVOVIEW_VRAM_PLAN=0 turns the planning off.

use crate::asset::TextureEntry;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

const MB: f64 = 1048576.0;

/// Mip levels the other cars' textures were dropped by at load: late joiners
/// get the same (or more, `late_drop`).
pub static OPP_TEX_DROP: AtomicU32 = AtomicU32::new(0);

/// The output size in pixels (main.rs: the bench size, else the screen).
pub static TARGET_PX: AtomicU64 = AtomicU64::new(0);

/// The primary screen's pixels (a window session is played full screen).
#[cfg(windows)]
pub fn screen_px() -> u64 {
    extern "system" {
        fn GetSystemMetrics(i: i32) -> i32;
    }
    // SAFETY: plain Win32 query (SM_CXSCREEN, SM_CYSCREEN)
    let (w, h) = unsafe { (GetSystemMetrics(0), GetSystemMetrics(1)) };
    (w.max(0) as u64 * h.max(0) as u64).max(1280 * 720)
}
#[cfg(not(windows))]
pub fn screen_px() -> u64 {
    1920 * 1080
}

fn on() -> bool {
    crate::envc::var("EVOVIEW_VRAM_PLAN").map_or(true, |v| v != "0")
}

fn margin() -> f64 {
    crate::envc::var("EVOVIEW_VRAM_MARGIN_MB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(800.0) * MB
}

/// Free VRAM for this process, total (bytes).
fn free_total() -> Option<(f64, f64)> {
    let (free, total) = crate::perf::vram_free_for_us()?;
    let free = crate::envc::var("EVOVIEW_VRAM_FREE_MB").ok().and_then(|v| v.parse::<f64>().ok()).map_or(free as f64, |m| m * MB);
    Some((free, total as f64))
}

fn bc(t: &TextureEntry) -> bool {
    matches!(t.format.as_str(), "bc1" | "bc4" | "bc7" | "bc6h" | "bc3")
}

/// Levels `drop_top_mips` would really drop from `t` (the same rule as the
/// loader's own mip skip: block-compressed, 8 px or more, one level kept).
fn droppable(t: &TextureEntry, k: u32) -> usize {
    if !bc(t) || t.width < 8 || t.height < 8 {
        return 0;
    }
    (k as usize).min(t.levels.len().saturating_sub(1))
}

/// Bytes per texel of an uncompressed texture `shrink` may box-filter: RGBA8
/// and RGBA16 colour/mask maps. The cars carry several, 20-40 MB each with
/// no mip chain to drop (lamp FunctionMasks ext_lights_f_*, interior decals,
/// dash icon atlases): two thirds of a lobby's car texture memory. Not an LED
/// mask - its red channel is an LED NUMBER, and averaging would make new
/// ones along every edge.
fn filterable(t: &TextureEntry) -> Option<u64> {
    let bpp = match t.format.as_str() {
        "rgba8" => 4,
        "rgba16u" => 8,
        _ => return None,
    };
    let l = t.levels.first()?;
    (!t.name.to_ascii_lowercase().contains("led") && l.bytes == l.width as u64 * l.height as u64 * bpp).then_some(bpp)
}

/// Halvings `shrink` makes of an uncompressed `t` for `k` (down to 8 px).
fn halvings(t: &TextureEntry, k: u32) -> u32 {
    if filterable(t).is_none() {
        return 0;
    }
    let l = &t.levels[0];
    (0..k).take_while(|i| l.width >> (i + 1) >= 8 && l.height >> (i + 1) >= 8).count() as u32
}

/// Bytes of `t` once `k` levels are dropped / halvings made.
fn bytes_after(t: &TextureEntry, k: u32) -> u64 {
    let h = halvings(t, k);
    if h > 0 {
        return t.levels[0].bytes >> (2 * h);
    }
    let n = droppable(t, k);
    t.data[1].saturating_sub(t.levels[..n].iter().map(|l| l.bytes).sum::<u64>())
}

/// `k` levels off every texture of a car: block-compressed ones drop their
/// top mips (`drop_top_mips`), uncompressed colour maps are box-filtered
/// down on the CPU into the start of their own bytes (one level; no blob
/// growth - the merged scene blob is GBs, and growing it can copy it all).
/// ⚠ `tex` must own its bytes: Scene merging gives every opponent entry its
/// own copy (a look sharing the player's map points at the player's entry,
/// outside `tex`). Entries sharing bytes within `tex` are filtered once.
pub fn shrink(tex: &mut [TextureEntry], blob: &mut [u8], k: u32) {
    if k == 0 {
        return;
    }
    let mut done: std::collections::HashMap<[u64; 2], ([u64; 2], u32, u32)> = Default::default();
    for t in tex.iter_mut() {
        let h = halvings(t, k);
        if h == 0 {
            drop_top_mips(t, k);
            continue;
        }
        let bpp = filterable(t).unwrap_or(4) as usize;
        let key = t.data;
        let (data, w, hh) = match done.get(&key) {
            Some(&d) => d,
            None => {
                let (mut w, mut hh) = (t.levels[0].width as usize, t.levels[0].height as usize);
                let mut img = blob[t.data[0] as usize..t.data[0] as usize + w * hh * bpp].to_vec();
                for _ in 0..h {
                    img = halve(&img, w, hh, bpp);
                    w /= 2;
                    hh /= 2;
                }
                let d = ([key[0], img.len() as u64], w as u32, hh as u32);
                blob[key[0] as usize..key[0] as usize + img.len()].copy_from_slice(&img);
                done.insert(key, d);
                d
            }
        };
        t.data = data;
        t.width = w;
        t.height = hh;
        t.levels = vec![crate::asset::Level { width: w, height: hh, bytes: data[1] }];
    }
}

/// 2x2 box filter of a `w` x `h` image of RGBA8 (`bpp` 4) or RGBA16 LE (8).
fn halve(src: &[u8], w: usize, h: usize, bpp: usize) -> Vec<u8> {
    let (nw, nh) = (w / 2, h / 2);
    let mut out = vec![0u8; nw * nh * bpp];
    let row = w * bpp;
    for y in 0..nh {
        for x in 0..nw {
            let a = (2 * y) * row + 2 * x * bpp;
            let o = (y * nw + x) * bpp;
            if bpp == 4 {
                for c in 0..4 {
                    let s = src[a + c] as u32 + src[a + 4 + c] as u32 + src[a + row + c] as u32 + src[a + row + 4 + c] as u32;
                    out[o + c] = ((s + 2) / 4) as u8;
                }
            } else {
                for c in 0..4 {
                    let g = |i: usize| u16::from_le_bytes([src[i], src[i + 1]]) as u32;
                    let s = g(a + 2 * c) + g(a + 8 + 2 * c) + g(a + row + 2 * c) + g(a + row + 8 + 2 * c);
                    out[o + 2 * c..o + 2 * c + 2].copy_from_slice(&(((s + 2) / 4) as u16).to_le_bytes());
                }
            }
        }
    }
    out
}

/// Drops the top `k` mip levels of a loaded texture without copying: the
/// levels sit top-first in its blob range, so the range just starts later.
pub fn drop_top_mips(t: &mut TextureEntry, k: u32) {
    let n = droppable(t, k);
    if n == 0 {
        return;
    }
    let skip: u64 = t.levels[..n].iter().map(|l| l.bytes).sum();
    if skip >= t.data[1] {
        return;
    }
    t.data[0] += skip;
    t.data[1] -= skip;
    t.levels.drain(..n);
    t.width = t.levels[0].width;
    t.height = t.levels[0].height;
}

/// Unique textures (shared looks point at the same bytes) and their bytes
/// after dropping `k` levels.
pub fn tex_bytes(tex: &[TextureEntry], k: u32) -> u64 {
    let mut seen = std::collections::HashSet::new();
    tex.iter().filter(|t| seen.insert(t.data)).map(|t| bytes_after(t, k)).sum()
}

pub fn mesh_bytes(meshes: &[crate::asset::MeshEntry]) -> u64 {
    meshes.iter().map(|m| m.vertices as u64 * 80 + m.indices.map_or(0, |i| i[1])).sum()
}

/// Load time (Scene::build_race): how many levels the opponents' textures
/// drop so the session fits, applied to `opp`. `base_bytes`: the track and
/// player car's texture + mesh bytes.
pub fn plan(base_bytes: u64, opp: &mut [TextureEntry], blob: &mut [u8], opp_mesh: u64) {
    if !on() || opp.is_empty() {
        return;
    }
    let Some((free, total)) = free_total() else { return };
    let px = TARGET_PX.load(Relaxed) as f64;
    let base = (1.85 * base_bytes as f64 + 84.0 * px - 300.0 * MB).max(base_bytes as f64);
    let need = |k: u32| base + 1.25 * (tex_bytes(opp, k) + opp_mesh) as f64;
    let room = free - margin();
    if crate::envc::var_os("EVOVIEW_VRAM_LOG").is_some() {
        let mut by: std::collections::BTreeMap<String, (u64, u64)> = Default::default();
        let mut seen = std::collections::HashSet::new();
        for t in opp.iter().filter(|t| seen.insert(t.data)) {
            let e = by.entry(format!("{} {}", t.format, if droppable(t, 1) + halvings(t, 1) as usize > 0 { "shrinks" } else { "kept" })).or_default();
            e.0 += 1;
            e.1 += t.data[1];
        }
        for (k, (n, b)) in by {
            println!("vram: opponents' {k}: {n} texture(s), {:.0} MB", b as f64 / MB);
        }
    }
    let k = (0..=3).find(|&k| need(k) <= room).unwrap_or(3);
    println!("vram: {:.0} of {:.0} MB free; this session ~{:.0} MB{}",
             free / MB, total / MB, need(k) / MB,
             if k > 0 { format!(" with the other cars' textures {k} more mip(s) down (~{:.0} MB at full)", need(0) / MB) } else { String::new() });
    shrink(opp, blob, k);
    OPP_TEX_DROP.store(k, Relaxed);
}

/// A late joiner's car (worker thread, before upload): drop its textures
/// like the others', and one level more while it would not fit beside what
/// this process holds now.
pub fn late_drop(tex: &mut [TextureEntry], blob: &mut [u8], mesh: u64) {
    let mut k = OPP_TEX_DROP.load(Relaxed);
    if on() {
        let t = std::time::Instant::now();
        let probe = (free_total(), crate::perf::vram_bytes());
        if crate::envc::var_os("EVOVIEW_VRAM_LOG").is_some() {
            println!("vram: late joiner check {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        if let (Some((free, _)), Some(ours)) = probe {
            // the margin held back at load is for exactly this; keep a
            // smaller one at run time
            let room = free - ours as f64 - 0.25 * margin();
            while k < 3 && 1.25 * (tex_bytes(tex, k) + mesh) as f64 > room {
                k += 1;
            }
        }
    }
    shrink(tex, blob, k);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::Level;

    fn tex(fmt: &str, w: u32, levels: u32) -> TextureEntry {
        let lv: Vec<Level> = (0..levels).map(|i| {
            let s = (w >> i).max(1);
            Level { width: s, height: s, bytes: (s.div_ceil(4) * s.div_ceil(4) * 16) as u64 }
        }).collect();
        let n = lv.iter().map(|l| l.bytes).sum();
        TextureEntry { name: "t".into(), format: fmt.into(), width: w, height: w, levels: lv, data: [1000, n] }
    }

    #[test]
    fn drops_top_levels_in_place() {
        let mut t = tex("bc7", 1024, 11);
        let top = t.levels[0].bytes + t.levels[1].bytes;
        let n = t.data[1];
        drop_top_mips(&mut t, 2);
        assert_eq!((t.width, t.height, t.levels.len()), (256, 256, 9));
        assert_eq!(t.data, [1000 + top, n - top]);
        assert_eq!(t.levels.iter().map(|l| l.bytes).sum::<u64>(), t.data[1]);
    }

    #[test]
    fn shrinks_uncompressed_once_per_shared_texture() {
        // 16x16 RGBA8, left half 0, right half 200; two looks share it
        let mut blob = vec![9u8; 1000];
        for y in 0..16 {
            for x in 0..16 {
                blob.extend_from_slice(&[if x < 8 { 0 } else { 200 }; 4]);
            }
            let _ = y;
        }
        let t = TextureEntry { name: "ext_lights_f_1".into(), format: "rgba8".into(), width: 16, height: 16,
                               levels: vec![Level { width: 16, height: 16, bytes: 1024 }], data: [1000, 1024] };
        let mut tex = vec![t.clone(), t.clone(), TextureEntry { name: "int_leds_f".into(), ..t.clone() }];
        tex[2].data = [1000, 1024];
        tex[2].name = "int_leds_f".into();
        shrink(&mut tex, &mut blob, 1);
        assert_eq!((tex[0].width, tex[0].height, tex[0].data), (8, 8, [1000, 256]));
        assert_eq!(tex[1].data, tex[0].data);
        assert_eq!(blob.len(), 2024);
        assert_eq!(&blob[1000..1004], &[0, 0, 0, 0]);
        assert_eq!(&blob[1000 + 4 * 4..1000 + 4 * 4 + 4], &[200, 200, 200, 200]);
        // the LED mask: its own bytes (here the same - a test shortcut) untouched as an entry
        assert_eq!((tex[2].width, tex[2].data), (16, [1000, 1024]));
    }

    #[test]
    fn keeps_last_level_and_skips_small_or_uncompressed() {
        let mut t = tex("bc1", 16, 2);
        drop_top_mips(&mut t, 3);
        assert_eq!(t.levels.len(), 1);
        let mut s = tex("bc1", 4, 3);
        drop_top_mips(&mut s, 1);
        assert_eq!(s.width, 4);
        let mut r = tex("rgba8", 256, 9);
        drop_top_mips(&mut r, 1);
        assert_eq!(r.width, 256);
    }
}
