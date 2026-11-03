//! Reading a Kunos `.kspkg` archive directly.
//!
//! The game ships every car inside `content.kspkg` (~74 GB). Extracting a car
//! folder just to look at it costs 200-800 MB on disk per car, so the viewer
//! reads the archive in place instead: index once, then pull the handful of
//! entries a car actually needs.
//!
//! LAYOUT
//!     [data region: file bytes, XOR'd with the engine key; each file's key
//!      phase RESETS TO 0 at its own start]
//!     [fixed 64 MiB table: one 256-byte bucket per path, same phase reset,
//!      phase taken relative to the TABLE'S OWN start]
//!
//! Bucket: [0..n] NUL-terminated path, [240..244] u32 size,
//!         [248..256] u64 offset. ⚠ The high dword is load-bearing - this
//!         archive's offsets exceed 2^32.
//!
//! ⚠ `.texturemips` are stored UNENCRYPTED - the engine streams mip pages
//! straight off disk. Decrypting them like everything else yields noise that
//! looks exactly like a de-tiling bug.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const KEY: [u8; 8] = [0xc1, 0x35, 0x11, 0x7d, 0xa9, 0x21, 0x97, 0x9f];
const BUCKET: usize = 256;
const TABLE: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy)]
pub struct Entry {
    pub offset: u64,
    pub size: u32,
    /// stored as-is rather than XOR'd (`.texturemips`)
    pub plain: bool,
}

pub struct Package {
    file: File,
    /// full lowercase entry path -> entry
    pub by_path: HashMap<String, Entry>,
}

/// Fill `buf` from `offset` without moving a shared file cursor.
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0usize;
        while done < buf.len() {
            match f.seek_read(&mut buf[done..], offset + done as u64) {
                Ok(0) | Err(_) => return false,
                Ok(n) => done += n,
            }
        }
        true
    }
    #[cfg(not(windows))]
    {
        use std::os::unix::fs::FileExt;
        f.read_exact_at(buf, offset).is_ok()
    }
}

fn dexor(buf: &mut [u8], phase: usize) {
    for (i, b) in buf.iter_mut().enumerate() {
        *b ^= KEY[(i + phase) % 8];
    }
}

impl Package {
    pub fn open(path: &Path) -> Result<Self, String> {
        Self::open_filtered(path, &[])
    }

    /// Open, keeping only entries whose path starts with one of `prefixes`
    /// (case-insensitive, backslash form). An empty list keeps everything.
    ///
    /// The table is ~80k entries and the old loop allocated twice per entry -
    /// a `to_vec()` to decrypt into and a `to_lowercase()` for the key - which
    /// is 160k allocations to reach the few hundred shared files a car needs.
    /// Filtering before either allocation is most of the cost gone.
    pub fn open_filtered(path: &Path, prefixes: &[&str]) -> Result<Self, String> {
        crate::perf::time("archive index", || Self::open_inner(path, prefixes))
    }

    fn open_inner(path: &Path, prefixes: &[&str]) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let total = file
            .metadata()
            .map_err(|e| e.to_string())?
            .len();
        if total <= TABLE {
            return Err("file is too small to be a .kspkg".into());
        }
        let table_start = total - TABLE;
        file.seek(SeekFrom::Start(table_start)).map_err(|e| e.to_string())?;

        // Empty buckets are one repeating pattern; skipping them before
        // decrypting is what keeps indexing to a couple of seconds.
        let fillers: Vec<[u8; 8]> = (0..8)
            .map(|ph| {
                let mut t = [0u8; 8];
                for (i, b) in t.iter_mut().enumerate() {
                    *b = KEY[(i + ph) % 8];
                }
                t
            })
            .collect();

        let mut by_path = HashMap::new();
        let mut buf = vec![0u8; BUCKET * 4096];
        // one scratch bucket, reused for every entry
        let mut dec = [0u8; BUCKET];
        let mut rel = 0usize;
        loop {
            let n = file.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            for i in (0..n.saturating_sub(BUCKET - 1)).step_by(BUCKET) {
                let pos = rel + i;
                let raw = &buf[i..i + BUCKET];
                let head: [u8; 8] = raw[0..8].try_into().unwrap();
                if fillers.contains(&head) && raw[8..64].iter().all(|_| true) {
                    // cheap check on the first 8 bytes, then confirm
                    let ph = pos % 8;
                    let mut t = [0u8; 8];
                    for (k, b) in t.iter_mut().enumerate() {
                        *b = KEY[(k + ph) % 8];
                    }
                    if head == t {
                        continue;
                    }
                }
                dec.copy_from_slice(raw);
                dexor(&mut dec, pos % 8);
                let end = match dec.iter().position(|&c| c == 0) {
                    Some(0) | None => continue,
                    Some(e) => e,
                };
                let Ok(p) = std::str::from_utf8(&dec[..end]) else { continue };
                if !p.contains('\\') {
                    continue;
                }
                let size = u32::from_le_bytes(dec[240..244].try_into().unwrap());
                if size == 0 {
                    continue;               // directory entry
                }
                if !prefixes.is_empty()
                    && !prefixes.iter().any(|q| p.len() >= q.len()
                        && p[..q.len()].eq_ignore_ascii_case(q))
                {
                    continue;               // not a shared asset - skip before allocating
                }
                let offset = u64::from_le_bytes(dec[248..256].try_into().unwrap());
                let low = p.to_lowercase();
                let plain = low.ends_with(".texturemips");
                by_path.insert(low, Entry { offset, size, plain });
            }
            rel += n;
        }
        Ok(Package { file, by_path })
    }

    /// Read one entry. Positional (no shared seek cursor), so it takes
    /// `&self` and several threads can read the same archive at once.
    pub fn read(&self, e: Entry) -> Vec<u8> {
        let mut v = vec![0u8; e.size as usize];
        let t = std::time::Instant::now();
        if !read_at(&self.file, &mut v, e.offset) {
            return Vec::new();
        }
        crate::perf::add("pkg io", t.elapsed().as_secs_f64());
        crate::perf::add_bytes("pkg io", v.len() as u64);
        if !e.plain {
            let t = std::time::Instant::now();
            dexor(&mut v, 0);
            crate::perf::add("pkg dexor", t.elapsed().as_secs_f64());
        }
        v
    }

    /// Every track folder that has static meshes. `common_assets` is shared
    /// kit, not a circuit.
    pub fn tracks(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .by_path
            .keys()
            .filter(|p| {
                let low = p.to_lowercase();
                low.starts_with("content\\tracks\\")
                    && low.ends_with(".mesh")
                    && !low.contains("\\common_assets\\")
            })
            .filter_map(|p| p.split('\\').nth(2).map(|s| s.to_string()))
            .filter(|c| !c.eq_ignore_ascii_case("common_assets"))
            .collect();
        out.sort();
        out.dedup();
        out
    }

    /// Every car id in the archive that actually has geometry.
    pub fn cars(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .by_path
            .keys()
            .filter(|p| p.starts_with("content\\cars\\") && p.contains("\\meshes\\")
                && p.ends_with(".mesh"))
            .filter_map(|p| p.split('\\').nth(2).map(|s| s.to_string()))
            .filter(|c| !c.starts_with("common"))
            .collect();
        out.sort();
        out.dedup();
        out
    }
}
