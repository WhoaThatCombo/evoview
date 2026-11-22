//! Where the simulation's time goes, and proof that a change left it alone.
//!
//! `EVOVIEW_SIM_PROF=1`: scoped timers over the per-car work (race.rs,
//! trackdrive.rs, the per-opponent loops of the frame), summed over every
//! thread and printed by `report` - at the end of `--drive-test` (headless,
//! no GPU) and of `--bench`. Off, a scope is one relaxed load of a bool.
//! "wall" rows are time on the calling thread; "sum" rows add up what the
//! AI worker threads spent, so they can exceed the wall time of the step.
//!
//! `EVOVIEW_SIM_DUMP=<file>`: every tick of `--drive-test`, the player's
//! whole car state as raw bits and a hash of every opponent's - two runs
//! (before / after a change) must give the same file byte for byte.
//!
//! Built with `--features alloc-count`, the report also counts heap
//! allocations per step (a counting global allocator; not in normal builds).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

macro_rules! ids {
    ($($name:ident = $label:expr,)*) => {
        #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
        #[derive(Clone, Copy)]
        #[repr(usize)]
        pub enum Id { $($name,)* N }
        const LABELS: [&str; Id::N as usize] = [$($label,)*];
    };
}

ids! {
    Player = "player car step (wall)",
    Step = "race step (wall)",
    Traffic = "  traffic (wall)",
    Drive = "  drivers + physics (wall)",
    Controls = "    driver controls (sum)",
    Physics = "    car physics (sum)",
    Probe = "      ground probes (sum)",
    Walls = "      wall queries (sum)",
    Contacts = "  contacts (wall)",
    Progress = "  progress (wall)",
    Unstick = "  unstick (wall)",
}

static NS: [AtomicU64; Id::N as usize] = [const { AtomicU64::new(0) }; Id::N as usize];
static CALLS: [AtomicU64; Id::N as usize] = [const { AtomicU64::new(0) }; Id::N as usize];

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::envc::var_os("EVOVIEW_SIM_PROF").is_some())
}

pub struct Scope(Id, Option<Instant>);

#[inline]
pub fn scope(id: Id) -> Scope {
    Scope(id, on().then(Instant::now))
}

impl Drop for Scope {
    #[inline]
    fn drop(&mut self) {
        if let Some(t) = self.1 {
            NS[self.0 as usize].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            CALLS[self.0 as usize].fetch_add(1, Relaxed);
        }
    }
}

/// Forget what has been timed so far (a one-off first step).
pub fn reset() {
    for k in 0..Id::N as usize {
        NS[k].store(0, Relaxed);
        CALLS[k].store(0, Relaxed);
    }
}

/// Allocations so far (count, bytes); zeros without `alloc-count`.
pub fn allocs() -> (u64, u64) {
    #[cfg(feature = "alloc-count")]
    {
        (counting::COUNT.load(Relaxed), counting::BYTES.load(Relaxed))
    }
    #[cfg(not(feature = "alloc-count"))]
    {
        (0, 0)
    }
}

#[cfg(feature = "alloc-count")]
mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static COUNT: AtomicU64 = AtomicU64::new(0);
    pub static BYTES: AtomicU64 = AtomicU64::new(0);
    struct Counting;
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            COUNT.fetch_add(1, Relaxed);
            BYTES.fetch_add(l.size() as u64, Relaxed);
            System.alloc(l)
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            System.dealloc(p, l)
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            COUNT.fetch_add(1, Relaxed);
            BYTES.fetch_add(n as u64, Relaxed);
            System.realloc(p, l, n)
        }
    }
    #[global_allocator]
    static A: Counting = Counting;
}

/// How many cars the race steps (set by race.rs), for the per-car column.
pub static CARS: AtomicU64 = AtomicU64::new(1);

/// The table: ms per step and us per car per step, over `steps` steps. `allocs0`: `allocs()` taken when the steps began.
pub fn report(steps: u64, allocs0: (u64, u64)) {
    if !on() || steps == 0 {
        return;
    }
    let cars = CARS.load(Relaxed) as usize;
    println!("SIMPROF {steps} steps, {cars} cars");
    for (k, l) in LABELS.iter().enumerate() {
        let (ns, calls) = (NS[k].load(Relaxed), CALLS[k].load(Relaxed));
        if calls == 0 {
            continue;
        }
        let ms = ns as f64 / 1e6 / steps as f64;
        println!("SIMPROF {l:42} {ms:8.3} ms/step {:8.2} us/car {:9.1} calls/step", ms * 1e3 / cars.max(1) as f64, calls as f64 / steps as f64);
    }
    let a = allocs();
    if a.0 > 0 && allocs0.0 > 0 {
        println!("SIMPROF allocations {:.1} / step ({:.1} per car), {:.0} bytes / step",
                 (a.0 - allocs0.0) as f64 / steps as f64, (a.0 - allocs0.0) as f64 / steps as f64 / cars.max(1) as f64,
                 (a.1 - allocs0.1) as f64 / steps as f64);
    }
}

/// `EVOVIEW_SIM_DUMP=<file>`: one record a tick (see the module text).
pub struct Dump(Option<std::io::BufWriter<std::fs::File>>, Vec<u8>);

impl Dump {
    pub fn from_env() -> Dump {
        let f = crate::envc::var("EVOVIEW_SIM_DUMP").ok().filter(|p| !p.is_empty())
            .and_then(|p| std::fs::File::create(p).ok()).map(std::io::BufWriter::new);
        Dump(f, Vec::new())
    }

    /// The player's state in full, then one FNV-1a hash over the others'.
    pub fn tick<'a>(&mut self, player: &crate::trackdrive::TrackDrive, others: impl Iterator<Item = &'a crate::trackdrive::TrackDrive>) {
        use std::io::Write;
        let Some(f) = self.0.as_mut() else { return };
        self.1.clear();
        player.state_bits(&mut self.1);
        let _ = f.write_all(&self.1);
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for o in others {
            self.1.clear();
            o.state_bits(&mut self.1);
            for &b in &self.1 {
                h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        let _ = f.write_all(&h.to_le_bytes());
    }
}
