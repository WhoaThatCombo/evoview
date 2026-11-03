# evoview performance

How fast evoview is, how to measure it, and how much room each new feature
has. The numbers are from `tools/bench.py`. Measure your change the same way
before you merge it.

## Running the benchmark

```
cargo build --release
python tools/bench.py --compare tools/bench_baseline.json
python tools/bench.py --only ks_bmw_m4_gt3 --sizes 1920x1080 --compare tools/bench_baseline.json
```

- **The set:** a GT3 (`ks_bmw_m4_gt3`), a road car (`ks_toyota_gr86`), an
  open-wheeler (`ks_ferrari_sf_25`) and two circuits (`brands_hatch`,
  `laguna_seca`). Each runs at 1920x1080 and 3840x2160, which is the
  official pair.
- **How each run works:** every item gets an untimed warm-up load (OS file
  cache), then 3 timed runs. Each metric is the minimum over the runs. The PC
  is shared, so interference only ever adds time. A sub-ms car frame swings
  +-50% run to run; the minimum of 3 is stable to a few %.
- **Output:** JSON goes to `L:\evoview_work\bench\<rev>.json`, or `--out`. The
  table shows the percent change against `--compare`.
- **Executable:** `--exe` or `EVOVIEW_EXE` picks it. The default is
  `L:\build\evoview-perf\release\evoview.exe`.
- **Be considerate:** run ONE benchmark at a time, and not continuously. It
  takes the GPU away from whatever else the user is doing.

Under the hood, `evoview <content.kspkg> --car|--track <id> --bench <s>
--size WxH [--bench-shot f.png]` drives the **real interactive renderer**
(`render::run`), so any pass you add there is measured automatically. Bench
mode changes only these things:

- it renders to an offscreen target of the requested size, so 4K works on any
  monitor and the compositor stays out of the numbers;
- it keeps at most 2 frames in flight;
- it uses a fixed camera: cars orbit once every 8 s, and tracks turn on the
  spot looking at the horizon;
- it asks for timestamp queries.

It prints:

- load phases;
- CPU frame time and GPU frame time (timestamps around the whole frame
  encoder, so every pass counts), each as mean / p50 / p99;
- VRAM, from the OS GPU counters, the same figure Task Manager shows;
- peak RAM, plus the working set while rendering;
- the heaviest meshes.

The last line is `BENCH_JSON {...}`. `--bench-shot` saves frame 0 at the
start angle. Use it to prove a speed-up did not change the picture: diff it
against a run with the feature switched off.

`EVOVIEW_MAX_FPS` caps every window at 60 by default, and bench mode obeys
it too. `bench.py` sets `EVOVIEW_MAX_FPS=0` for its runs. If you run
`--bench` by hand, set it yourself.

## Baseline vs now

- **Baseline:** `tools/bench_baseline.json` = 1e5506d (before the game-look
  work) with this harness backported (the `L:/evoview_work/perf/basewt`
  worktree).
- **Now:** `tools/bench_current.json` = look-perf 36857df on look-base7.
- **Units:** GPU and CPU in ms, load in s (process start to first frame), VRAM
  and peak working set in MB.

| item | GPU mean | GPU p99 | CPU p50 | load | VRAM | peak RAM |
|---|---|---|---|---|---|---|
| ks_bmw_m4_gt3 @ 1920x1080 | 0.36 -> 0.42 (+15%) | 0.90 -> 1.20 (+33%) | 0.35 -> 0.40 (+13%) | 1.69 -> 1.42 (-16%) | 854 -> 599 (-30%) | 1525 -> 1078 (-29%) |
| ks_bmw_m4_gt3 @ 3840x2160 | 0.87 -> 1.07 (+23%) | 1.42 -> 1.83 (+29%) | 0.84 -> 1.03 (+23%) | 1.97 -> 1.57 (-20%) | 1044 -> 853 (-18%) | 1525 -> 1081 (-29%) |
| ks_toyota_gr86 @ 1920x1080 | 0.28 -> 0.34 (+22%) | 0.75 -> 0.90 (+19%) | 0.28 -> 0.33 (+18%) | 1.91 -> 1.46 (-23%) | 758 -> 534 (-30%) | 1410 -> 894 (-37%) |
| ks_toyota_gr86 @ 3840x2160 | 0.71 -> 0.91 (+28%) | 1.26 -> 1.66 (+32%) | 0.69 -> 0.88 (+28%) | 1.56 -> 1.40 (-10%) | 980 -> 725 (-26%) | 1410 -> 894 (-37%) |
| ks_ferrari_sf_25 @ 1920x1080 | 0.32 -> 0.28 (-11%) | 0.92 -> 0.83 (-10%) | 0.31 -> 0.28 (-7%) | 1.26 -> 1.23 (-2%) | 597 -> 407 (-32%) | 730 -> 733 (+0%) |
| ks_ferrari_sf_25 @ 3840x2160 | 0.78 -> 0.80 (+4%) | 1.31 -> 1.56 (+20%) | 0.75 -> 0.77 (+3%) | 1.15 -> 1.28 (+12%) | 787 -> 661 (-16%) | 730 -> 733 (+0%) |
| brands_hatch @ 1920x1080 | 12.82 -> 0.78 (-94%) | 13.94 -> 1.33 (-90%) | 12.81 -> 1.12 (-91%) | 2.02 -> 3.21 (+59%) | 629 -> 1337 (+113%) | 851 -> 1890 (+122%) |
| brands_hatch @ 3840x2160 | 13.49 -> 2.55 (-81%) | 14.75 -> 4.37 (-70%) | 13.45 -> 2.44 (-82%) | 1.99 -> 3.02 (+52%) | 660 -> 1367 (+107%) | 849 -> 1890 (+123%) |
| laguna_seca @ 1920x1080 | 6.93 -> 0.47 (-93%) | 8.05 -> 0.80 (-90%) | 6.89 -> 0.97 (-86%) | 1.99 -> 3.56 (+79%) | 790 -> 1529 (+94%) | 1424 -> 2044 (+44%) |
| laguna_seca @ 3840x2160 | 7.34 -> 1.36 (-81%) | 8.51 -> 2.66 (-69%) | 7.30 -> 1.39 (-81%) | 2.35 -> 2.66 (+13%) | 853 -> 1623 (+90%) | 1424 -> 2043 (+44%) |

Reading it:

- **Tracks** draw 6-17x faster than before the game-look work, despite the
  port's shadow map, LUT and extra bindings. They were vertex bound:
  Brands Hatch submitted 78 M instanced triangles a frame, and its frame time
  did not move between 1600x900 and 400x225.
- **Cars** are still +15-28% GPU on the M4 and GR86. All of it is fragment
  shading from the look work (4K costs ~2.3x 1080p): PBR port, IBL model,
  paint layers, two-draw glass. The SF-25 is within the gate. In absolute
  terms a car frame is 0.3 ms at 1080p and 0.8-1.1 ms at 4K.
- **Track VRAM and RAM** roughly doubled. The port uploads full-resolution
  textures below 4K. Visible track quality is a priority, so we deliberately
  do NOT cap it.
- **Track working set while rendering** is now ~0.75 GB (Laguna 2.6 GB
  before), because the CPU copy is freed after upload. The peak during load
  is what's in the table.
- **Track load:** part of the rise is a measurement shift. Texture uploads now
  complete inside "load" (staging is flushed every 256 MB to cap RAM) instead
  of inside frame 0.

## What changed (look-perf)

| commit | change |
|---|---|
| 3d790e5 | `--bench` mode + `tools/bench.py` |
| 758c963 | per-instance frustum + half-pixel culling of static scenery (visible rows compacted into the existing instance buffer, no layout change); the sun shadow pass culled to the light box via its own buffers and **re-recorded only when the light box moves** (or in replays, or when `shadow_invalid` is set) |
| c2ab458 | MeshLodData LOD1+ for instanced props as sibling meshes, each instance in one distance band; texture uploads shared across materials; offscreen 1080p/4K bench |
| e8e8308 | `material::TexCache` (pub, kept as `Materials::textures`); `instanced_entry()` builds every LOD row |
| 2cfaf35 | car opaque / cutout draws sorted front to back |
| 5f980b0 | `EVOVIEW_MAX_FPS` frame cap, default 60 |
| 1866377 | textures read + de-tiled on up to 6 threads before the material pass; positional kspkg reads |
| a27d2d3 | skip the four studio softboxes when their scale (`grade.w`) is 0 - the thumbnail default |
| a8491ec | the CPU scene copy is dropped after upload; texture staging flushed every 256 MB |
| 36857df | LOD switch distances in metres, as the game stores them |

Each has an off switch for A/B checks: `EVOVIEW_NO_CULL`, `EVOVIEW_NO_LOD`,
`EVOVIEW_LOD_SCALE`, `EVOVIEW_NO_SORT`, `EVOVIEW_SHADOW_EVERY_FRAME`,
`EVOVIEW_RIG`.

Every change was checked with `--bench-shot` against the feature switched
off:

- culling, the shadow cache and parallel decode are pixel-identical;
- sorting differs by 4-8 px of equal-depth ties;
- LODs differ only in distant leaf density.

## Rules for new passes and features

- **Instance buffers are compacted.** On tracks, `m.instances` holds only the
  instances visible to the camera. Draw with
  `culler.count(i, m.instance_count)`, never `m.instance_count`. A pass with a
  different view, like the shadow pass, needs its own culled set: copy
  `Culler::update_shadow` / `Culler::shadow`.
- **LOD rows.** Coarser LOD rows have `instance_count == 0` until a culler
  hands them instances. Paths without a culler (shots) draw LOD0 only.
- **Shadow map.** It is cached. If your feature changes what casts, set
  `shadow_invalid = true`.
- **`car` is gone after setup.** It is dropped in `render::run` after the
  culler / DrawOrder are built. Anything that needs CPU scene data must take
  it before that point.
- **Textures.** Bind textures through `materials.textures`
  (`TexCache::get`) instead of uploading a second copy.

## Budgets

The ship gate is GPU frame time within +10% of 1e5506d at both sizes. Track
features have a lot of headroom; car features have none.

| | 1080p | 4K | headroom now |
|---|---|---|---|
| car, gate (1.1 x 1e5506d, M4 / GR86 / SF-25) | 0.40 / 0.31 / 0.35 ms | 0.96 / 0.78 / 0.86 ms | **already over** on M4 / GR86 |
| track, gate (Brands / Laguna) | 14.1 / 7.6 ms | 14.8 / 8.1 ms | ~12 / ~6 ms |

What that means for each feature:

- **Car shading (paint, glass, lighting, hero shadow, decals):** budget is
  zero or negative. Every car feature must pay for itself, or come with an
  equal saving. Measure the M4 at 4K; it is the most sensitive. Rough sizes:
  - one extra full-screen texture fetch per car pixel: ~0.02-0.04 ms at 4K;
  - one area light: ~0.1 ms at 4K (the four softboxes cost ~0.4 ms when on).

  Cheap levers left:
  - an opaque depth pre-pass (needs `@invariant` on the clip position and a
    LessEqual main pipeline, so it's a structural change; ask first);
  - CPU-side per-light constants, instead of `make_light` per pixel.
- **Track features:** the gate leaves ~12 ms. Spend no more than this, so
  the 60 fps default cap is never at risk and the GPU stays usable for the
  user's other work:
  - total track frame <= 6 ms at 4K;
  - <= 1.5 ms per new pass;
  - shadow redraw <= 3 ms, since it's cached and only paid when the light box
    moves.

  Anything per-instance must go through the culler, or it reintroduces the
  78 M-triangle problem. The grass carpet (track.rs) is outside the culler:
  keep it inside its 90 m radius.
- **Load:** cars <= 1e5506d (1.3-2.0 s; currently 10-23% under). Tracks are
  over (+13..+79%) because of full-res textures. Don't add serial per-texture
  work; put it in the parallel prefetch (`Vfs::prefetch_textures`).
- **VRAM:**
  - cars <= 1e5506d (currently 16-32% under);
  - tracks: no new full-size render targets without measuring (one 4K
    RGBA16F target is 64 MB; the 4096^2 shadow map is 64 MB).
- **RAM:** stay under 3 GB peak for any track. Laguna peaks at ~2.0 GB
  working set / 3.5 GB commit during load. Do not hold a second CPU copy of
  textures.
