//! A small persistent thread pool for the per-car simulation (race.rs).
//!
//! race.rs used to spawn its worker threads anew every step
//! (`std::thread::scope`): three OS threads created and joined 60-120 times
//! a second, the cars dealt out in fixed thirds. Here the workers live as
//! long as the pool and sleep between steps; each step they (and the calling
//! thread) take the next car off a shared counter, so a car that costs more
//! (lost, backing out of a wall) does not hold up a whole third of the field.
//!
//! Every car is stepped exactly once and alone, by whichever thread takes
//! it: the result is the same as stepping them one after the other.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};

/// One `for_each_mut` call, as the workers see it.
#[derive(Clone, Copy)]
struct Job {
    /// `&(dyn Fn(usize) + Sync)` with its lifetime erased: alive until the
    /// caller has seen every worker leave the job (`busy == 0`)
    run: *const (dyn Fn(usize) + Sync),
    n: usize,
}

unsafe impl Send for Job {}

struct State {
    job: Option<Job>,
    /// bumped per job, so a worker takes each job once
    gen: u64,
    /// workers inside the current job
    busy: usize,
    /// a worker's `f` panicked in the current job
    panicked: bool,
    quit: bool,
}

struct Shared {
    state: Mutex<State>,
    /// a job is up (or quit)
    wake: Condvar,
    /// a worker left the job
    done: Condvar,
    next: AtomicUsize,
}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    /// `threads` in all, the caller's included (1 = no workers).
    pub fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared {
            state: Mutex::new(State { job: None, gen: 0, busy: 0, panicked: false, quit: false }),
            wake: Condvar::new(),
            done: Condvar::new(),
            next: AtomicUsize::new(0),
        });
        let workers = (1..threads.max(1)).filter_map(|k| {
            let sh = shared.clone();
            std::thread::Builder::new().name(format!("sim {k}")).spawn(move || worker(&sh)).ok()
        }).collect();
        Pool { shared, workers }
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// `f` on every item, each exactly once, spread over the pool's threads
    /// and this one; returns when all are done.
    pub fn for_each_mut<T: Send>(&self, items: &mut [T], f: impl Fn(usize, &mut T) + Sync) {
        let n = items.len();
        if self.workers.is_empty() || n < 2 {
            items.iter_mut().enumerate().for_each(|(i, x)| f(i, x));
            return;
        }
        struct Ptr<T>(*mut T);
        unsafe impl<T: Send> Sync for Ptr<T> {}
        impl<T> Ptr<T> {
            fn at(&self, i: usize) -> *mut T {
                unsafe { self.0.add(i) }
            }
        }
        let base = Ptr(items.as_mut_ptr());
        // each index is handed out once (the shared counter), so the &mut
        // made from it is the only one to that item
        let run = |i: usize| f(i, unsafe { &mut *base.at(i) });
        let run: &(dyn Fn(usize) + Sync) = &run;
        // the lifetime erased: this call does not return (or unwind) before
        // every worker has left the job - see Leave below
        let job = Job { run: unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), *const (dyn Fn(usize) + Sync)>(run) }, n };
        {
            let mut s = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
            self.shared.next.store(0, Relaxed);
            s.job = Some(job);
            s.gen += 1;
        }
        self.shared.wake.notify_all();
        // waits for the workers even if `f` panics on this thread
        struct Leave<'a>(&'a Shared);
        impl Drop for Leave<'_> {
            fn drop(&mut self) {
                let mut s = self.0.state.lock().unwrap_or_else(|p| p.into_inner());
                // no worker may start on it from here
                s.job = None;
                while s.busy > 0 {
                    s = self.0.done.wait(s).unwrap_or_else(|p| p.into_inner());
                }
            }
        }
        let leave = Leave(&self.shared);
        take(&self.shared.next, job);
        drop(leave);
        // as std::thread::scope did: a worker's panic is the caller's
        let mut s = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
        if std::mem::take(&mut s.panicked) {
            drop(s);
            panic!("sim pool: a car's step panicked on a worker thread");
        }
    }
}

/// Items off the counter until there are none left.
fn take(next: &AtomicUsize, job: Job) {
    loop {
        let i = next.fetch_add(1, Relaxed);
        if i >= job.n {
            return;
        }
        unsafe { (*job.run)(i) };
    }
}

fn worker(sh: &Shared) {
    let mut seen = 0u64;
    loop {
        let job = {
            let mut s = sh.state.lock().unwrap_or_else(|p| p.into_inner());
            loop {
                if s.quit {
                    return;
                }
                if s.gen != seen {
                    seen = s.gen;
                    if let Some(j) = s.job {
                        s.busy += 1;
                        break j;
                    }
                }
                s = sh.wake.wait(s).unwrap_or_else(|p| p.into_inner());
            }
        };
        // a panic in `f` must still count the worker out, or the caller
        // would wait for ever
        struct Out<'a>(&'a Shared);
        impl Drop for Out<'_> {
            fn drop(&mut self) {
                let mut s = self.0.state.lock().unwrap_or_else(|p| p.into_inner());
                s.busy -= 1;
                drop(s);
                self.0.done.notify_all();
            }
        }
        let out = Out(sh);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| take(&sh.next, job)));
        if r.is_err() {
            sh.state.lock().unwrap_or_else(|p| p.into_inner()).panicked = true;
        }
        drop(out);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut s = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
            s.quit = true;
        }
        self.shared.wake.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_item_once() {
        let pool = Pool::new(4);
        for round in 0..200 {
            let mut v: Vec<u64> = (0..(round % 37) as u64).collect();
            pool.for_each_mut(&mut v, |i, x| *x = i as u64 * 2 + 1);
            assert!(v.iter().enumerate().all(|(i, &x)| x == i as u64 * 2 + 1), "round {round}");
        }
    }

    #[test]
    fn one_thread_runs_inline() {
        let pool = Pool::new(1);
        let mut v = vec![1, 2, 3];
        pool.for_each_mut(&mut v, |_, x| *x += 1);
        assert_eq!(v, [2, 3, 4]);
        assert_eq!(pool.threads(), 1);
    }
}
