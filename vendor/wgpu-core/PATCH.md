# wgpu-core 22.1.0, patched for evoview

This directory is the published `wgpu-core` 22.1.0 crate (crates.io, MIT OR Apache-2.0, upstream licence files
kept) with ONE change, in `src/device/queue.rs`, marked `evoview:` in the source. The change alone is
`evoview-submit-registry.diff` beside this file. `Cargo.toml` wires the copy in:

```toml
[patch.crates-io]
wgpu-core = { path = "vendor/wgpu-core" }
```

22.1.0 is the last 22.x release (checked 2026-11-21); there is no newer 22.x to bump to instead.

## What changed

`Global::queue_submit` takes the write lock of the command buffer registry (`hub.command_buffers.write()`) and
upstream keeps it for the whole "prepare" loop: per command buffer, the walk over every buffer, texture, view
and bind group it used, memory initialisation, the merge into the device tracker and the barrier encoder -
milliseconds in a frame with many resources.

Every other thread needs that registry to begin a render or compute pass, to create or finish a command encoder,
to write a timestamp and for the encoder-level copies and clears. evoview records its passes on several threads
and submits finished command buffers from a thread of its own while the others are still recording
(`src/submit.rs`), so those threads stalled behind every walk: the time taken off the render thread's submit came
back as time waiting to begin passes.

The patch takes all the submitted command buffers out of the registry first (the same `replace_with_error` call,
in the same order), releases the registry, and then runs the loop unchanged on the taken values.

`EVOVIEW_WGPU_SUBMIT_LOCK=1` keeps the guard for the whole loop, as upstream (for A/B runs).

## Why releasing the registry early is safe

The registry guard protects the registry's storage - the id -> `Arc<CommandBuffer>` table - and nothing else.
Inside the loop upstream used it for exactly one thing, the `replace_with_error(id)` call. Everything after that
call works on the `Arc<CommandBuffer>` it returned and on locks of its own (the command buffer's `data` mutex,
`device.trackers`, `device.pending_writes`, the life tracker), all still taken exactly where they were.

- **The submitted ids cannot be reached meanwhile.** Their entries are `Element::Error` from the moment they are
  taken, exactly as they are after an upstream submit, so a lookup gives `InvalidId` either way; upstream only
  made such a lookup wait first. And nothing can hold those ids: `wgpu::Queue::submit` consumes the
  `wgpu::CommandBuffer`s it is given.
- **The ids cannot be reused meanwhile.** An id returns to the allocator in `Registry::unregister`, which runs
  from `command_buffer_drop`; wgpu calls that for each submitted buffer only after `queue_submit` has returned
  (`wgpu/src/backend/wgpu_core.rs`, `queue_submit`). An encoder created on another thread during the walk gets
  a different id.
- **No second owner of a command buffer appears.** `from_arc_into_baked` needs the `Arc` to be unique. Clones
  come only from registry lookups by id (a pass holds one while it is open); a pass cannot be open on a finished
  encoder, and after the take a lookup fails. The lock never covered clones taken before it, so nothing is lost.
- **Submits stay serialised.** `device.fence.write()` is taken before the registry and held to the end of
  `queue_submit`, unchanged; two submits still cannot interleave, and the device tracker is still updated in
  submission order.
- **No new lock order.** The patch only holds one lock for less time; it acquires nothing new and nothing in a
  different order, so it cannot add a deadlock (the debug-build lock ranks are satisfied as before).
- **Error paths.** Upstream, an error in the middle of the loop left the later ids in the registry until wgpu
  dropped them; patched, they are already taken and are dropped with the local vector (`CommandBuffer::drop`
  resets and destroys their encoder, as it does for any unsubmitted command buffer). wgpu 22 treats a submit
  error as fatal (`handle_error_fatal`) in both cases.

What does change is timing: other threads now run wgpu calls on OTHER command encoders during the walk. Those
calls were already legal at any moment outside a submit and touch per-encoder state, or device state behind its
own locks.

Checked in evoview on the Nordschleife with 23, 100 and 200 cars, the 53-car replay and players joining
mid-session, submit thread on and off: no panic, no validation error, frame captures identical within the
scene's run-to-run noise (`L:/evoview_work/stress/OPT_SUBMIT.md`, section 7).

## Re-applying after a wgpu upgrade

- wgpu 23 and later pass command buffers to the queue as `Arc`s and no longer hold the registry across this
  loop; check `queue_submit` in the new version - most likely this directory and the `[patch]` section can
  simply be deleted (and THIRD-PARTY-NOTICES.md's "Vendored source" entry with them).
- For another 22.x: copy the published crate here, apply `evoview-submit-registry.diff` (`patch -p1`), keep this
  file and the diff.
