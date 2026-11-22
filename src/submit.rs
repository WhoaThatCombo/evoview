//! The frame's command buffers, submitted by a thread of their own as they
//! are finished.
//!
//! wgpu 22's `Queue::submit` is not a hand-over. Per command buffer it walks
//! every buffer, texture and bind group the buffer used - and every texture
//! view of every bind group - to stamp them with the submission, merges
//! their states into the device's, and then frees what the finished
//! submissions held (their trackers: a reference dropped per resource; their
//! Vulkan command pools reset). Timed inside wgpu-core with 200 cars of 69
//! looks, 10 command buffers a frame, 5.0 ms on the calling thread:
//!   1.7 ms  2,600 bind groups stamped, with their 41,000 texture views
//!   0.5 ms  17,000 texture entries, 0.3 ms 6,700 buffer entries
//!   0.3 ms  state merge + barriers, 0.1 ms the transition encoders
//!   2.0 ms  freeing the submissions the GPU finished (125 Vulkan command
//!           buffers reset, their trackers dropped)
//!   0.2 ms  vkQueueSubmit, the pending writes (130 staging buffers)
//! It is the amount of resources a frame touches, not the number of command
//! buffers: one buffer or ten walk the same bind groups.
//!
//! None of it has to wait for the frame's last pass. The passes are recorded
//! on worker threads into command buffers of their own (render.rs), and a
//! command buffer can go to the queue as soon as every one before it in the
//! frame has. So each is sent here when it is finished, with its place in
//! the frame's order, and this thread submits them in that order while the
//! render thread is still recording; the render thread only waits for the
//! frame's last one.
//!
//! It needs the wgpu-core in vendor/ (vendor/wgpu-core/PATCH.md). Upstream
//! 22.1 holds its command buffer registry for a submit's whole walk, and
//! beginning a pass, making or finishing an encoder and writing a timestamp
//! all need that registry: the post chain's dozens of short passes stalled
//! behind every walk and most of the gain came back as encode time (23 cars:
//! encode + submit 4.8 -> 5.3 ms). The vendored copy releases the registry
//! before the walk. Render thread encode + submit, one submit with upstream's
//! locking -> this thread (the frames are GPU bound and do not change):
//!   200 cars   7.6 -> 6.6 ms (submit 1.8 -> 0.5, encode 5.8 -> 6.1)
//!   100 cars   6.0 -> 5.1    (1.2 -> 0.1, 4.8 -> 5.0)
//!    23 cars   4.0 -> 3.6    (0.8 -> 0.2, 3.2 -> 3.4)
//!   replay     2.4 -> 2.4
//! EVOVIEW_SUBMIT_THREAD=0: one submit of everything by the render thread at
//! the end of the frame.
//!
//! ⚠ `queue.write_buffer` data goes out with the NEXT submit, whichever
//! thread makes it. With one submit a frame every write of a frame precedes
//! every pass of it. With this thread a write only precedes the command
//! buffers submitted after it: a buffer written while the frame is recorded
//! must not be read by a command buffer EARLIER in the frame's order than
//! the one recorded by whoever writes it. (Held today: such writes are the fx
//! job's particles and dash glyphs, the motion job's uniforms, the mirror
//! glass's camera and the post chain's uniforms, each read by its own or a
//! later command buffer; the track's animation, which the passes do read,
//! opens the gate below.)

use std::sync::mpsc::{channel, Receiver, Sender};

enum Msg {
    /// (frame, place in the frame's order from 0, the command buffer)
    Cmd(u64, u32, wgpu::CommandBuffer),
    /// (frame) the writes the frame's passes read are all queued (the
    /// track's animation runs beside the recording): nothing goes out before
    Open(u64),
    /// (frame) every command buffer of it was sent: reply with the last
    /// submission
    End(u64),
}

pub struct Submitter {
    tx: Sender<Msg>,
    done: Receiver<Option<wgpu::SubmissionIndex>>,
    frame: u64,
}

/// Whether the frame's command buffers go through the submit thread.
/// On unless EVOVIEW_SUBMIT_THREAD=0.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::envc::var("EVOVIEW_SUBMIT_THREAD").map_or(true, |v| v != "0"))
}

impl Submitter {
    pub fn new(queue: std::sync::Arc<wgpu::Queue>) -> Option<Submitter> {
        let (tx, rx) = channel::<Msg>();
        let (done_tx, done) = channel();
        std::thread::Builder::new()
            .name("submit".into())
            .spawn(move || run(&queue, rx, done_tx))
            .ok()?;
        Some(Submitter { tx, done, frame: 0 })
    }

    /// A new frame: what its recorders send with.
    pub fn begin(&mut self) -> FrameTx {
        self.frame += 1;
        FrameTx { tx: self.tx.clone(), frame: self.frame }
    }

    /// The frame's last command buffer, `seq` in its order: back once the
    /// whole frame is on the queue, with its last submission. None: the
    /// submit thread is gone (it panicked - a wgpu validation error).
    pub fn end(&self, seq: u32, cb: wgpu::CommandBuffer) -> Option<wgpu::SubmissionIndex> {
        let _ = self.tx.send(Msg::Cmd(self.frame, seq, cb));
        let _ = self.tx.send(Msg::End(self.frame));
        self.done.recv().ok().flatten()
    }
}

/// A frame's way to the submit thread, shared by its recording threads.
pub struct FrameTx {
    tx: Sender<Msg>,
    frame: u64,
}

impl FrameTx {
    /// A finished command buffer, `seq` in the frame's order (from 0, no gaps).
    pub fn cmd(&self, seq: u32, cb: wgpu::CommandBuffer) {
        let _ = self.tx.send(Msg::Cmd(self.frame, seq, cb));
    }

    /// Every write the frame's passes read is queued: they may go out.
    pub fn open(&self) {
        let _ = self.tx.send(Msg::Open(self.frame));
    }
}

/// A frame's command buffers waiting for their turn: they arrive in any
/// order (whichever worker finishes first) and leave in the frame's.
struct Line<T> {
    held: Vec<(u32, T)>,
    /// the place whose turn it is
    next: u32,
    /// the gate (`Msg::Open`)
    open: bool,
}

impl<T> Line<T> {
    fn new() -> Self {
        Line { held: Vec::new(), next: 0, open: false }
    }

    fn push(&mut self, seq: u32, t: T) {
        self.held.push((seq, t));
    }

    /// Everything that is next in line and here, in order; empty while the
    /// gate is shut or the next place has not arrived.
    fn ready(&mut self) -> Vec<T> {
        let mut out = Vec::new();
        while self.open {
            let Some(i) = self.held.iter().position(|h| h.0 == self.next) else { break };
            out.push(self.held.swap_remove(i).1);
            self.next += 1;
        }
        out
    }

    /// Whatever is left, in order (the end of a frame: nothing may stay
    /// waiting for a place nobody sent).
    fn rest(&mut self) -> Vec<T> {
        self.held.sort_by_key(|h| h.0);
        self.held.drain(..).map(|h| h.1).collect()
    }
}

fn run(queue: &wgpu::Queue, rx: Receiver<Msg>, done: Sender<Option<wgpu::SubmissionIndex>>) {
    let mut frame = 0u64;
    let mut line: Line<wgpu::CommandBuffer> = Line::new();
    let mut last: Option<wgpu::SubmissionIndex> = None;
    for msg in rx {
        let f = match &msg {
            Msg::Cmd(f, ..) | Msg::Open(f) | Msg::End(f) => *f,
        };
        if f != frame {
            // (a frame given up half way leaves nothing behind)
            frame = f;
            line = Line::new();
            last = None;
        }
        let end = matches!(msg, Msg::End(_));
        match msg {
            Msg::Cmd(_, seq, cb) => line.push(seq, cb),
            // (at the end the render thread has joined every recorder)
            Msg::Open(_) | Msg::End(_) => line.open = true,
        }
        let mut go = line.ready();
        if end {
            go.extend(line.rest());
        }
        if !go.is_empty() {
            let _t = crate::bench::cpu("submit thread");
            last = Some(queue.submit(go));
        }
        if end && done.send(last.take()).is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Line;

    #[test]
    fn line_keeps_the_frames_order() {
        let mut l: Line<u32> = Line::new();
        // nothing leaves before the gate opens
        l.push(0, 0);
        l.push(2, 2);
        assert!(l.ready().is_empty());
        l.open = true;
        // 0 is here, 1 is not: 2 waits
        assert_eq!(l.ready(), vec![0]);
        assert!(l.ready().is_empty());
        l.push(3, 3);
        assert!(l.ready().is_empty());
        l.push(1, 1);
        assert_eq!(l.ready(), vec![1, 2, 3]);
        // a place nobody sent: the rest still goes, in order
        l.push(6, 6);
        l.push(5, 5);
        assert!(l.ready().is_empty());
        assert_eq!(l.rest(), vec![5, 6]);
        assert!(l.rest().is_empty());
    }
}
