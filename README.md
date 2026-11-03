# evoview

A fast car and track viewer for Assetto Corsa EVO. Vulkan via wgpu, no engine,
no editor — point it at a car folder and it opens.

```
evoview <car-folder>                    interactive
evoview <car-folder> --shot out.png     one frame, headless
evoview <content.kspkg> --car <id>      straight out of the archive
evoview <track-folder> --track          a whole circuit
```

```
evoview <src> --materials               every material as a sphere, on a grid
```

`--help` lists every flag. `--version` prints the version.

## Liveries

A car's liveries are its visual presets. `V` / `Shift+V` step through them in
the window (the title shows the game's name for the livery, its preset stem
and n/N); a switch reloads only the car's protobuf and geometry and reuses
every texture already decoded or on the GPU, so it takes ~150-300 ms.
`--list-liveries` prints `stem<TAB>name<TAB>race number` per preset and exits,
and a running viewer reads `livery <stem | n | next | prev>` lines on stdin -
that is how ACECM switches it. `EVOVIEW_VISUAL=<stem>` picks one at launch.

## Driving it

`N` cycles drive mode: off / keyboard / scripted demo (and replay, under
`--replay-car`). The car runs on a treadmill: Up/I throttle, Down/K brake,
Left/J Right/L steer, Space handbrake, X/Z gear up/down, Y wipers, T DRS,
1-9 doors, hood, boot and roof, H light switch (pop-ups follow low beam). Every
animation is bound the way the game binds it - the `.actor`'s animation slots
say which clip answers to which input - and the wheels, hubs, steering wheel,
suspension arms and dash needles move procedurally from the car's own physics
data (steer lock and ratio, suspension hard points, travel, springs, gears).
Pitch and roll come from a spring model using the real spring rates.

    evoview content.kspkg --car ks_toyota_gr86 --demo 7.5 --shot out.png
    evoview content.kspkg --car ks_toyota_gr86 --demo 1 --frames 240 --fps 15 --shot f.png
    evoview --replay-car lap.rpy --pkg content.kspkg [--car N] [--time T --shot out.png]

`--replay-car` drives one recorded car's steering, pedals, gears, wheel spin
and suspension heave / twist from the .rpy.

### Racing AI opponents

    evoview content.kspkg --track imola --drive-car ks_bmw_m4_gt3 --ai 7

`--ai N` (or `EVOVIEW_AI=N`) puts N opponents on the track's grid, each with
its own physics (the rig car when `EVOVIEW_RIGCAR` is on, else the simple
model; `EVOVIEW_AI_RIG=0/1` picks) and a racing-line driver with its own pace,
line and braking point. They follow or pass a car ahead, keep room beside a
car alongside, and cars push apart on contact. Five red lights, then out
(`EVOVIEW_AI_START=rolling`: a formation capped at 90 km/h until the leader
takes the line). `R` cycles the proximity radar automatic / on / off.
Knobs: `EVOVIEW_AI_CARS=id1,id2` (models, default the player's),
`EVOVIEW_AI_PACE=lo,hi` (default 0.9,1.0), `EVOVIEW_AI_GRID_POS=n` (the
player's slot, 1 = pole), `EVOVIEW_RADAR_POS=x,y`, `EVOVIEW_AI_LOG=1`.
With `--drive-test S` the player's car drives the racing line too and the
run logs order, laps, contacts and stuck cars.

## Lights and effects

Lamps, LEDs, the dash and the rest come from the car's own data
(`.carlightingsystem`, `.carledsystem`, the display pages, the particle
emitters, `car_shading_settings`); see `src/lights.rs` and `src/fxdata.rs`.
While driving (any `N` mode) the effects follow the car: brake lamps from the
pedal, reverse lamps in reverse, the live dash and shift lights from speed /
rpm / gear, brake discs heating under braking, a backfire on lift-off, rear
tyre smoke on a launch or a handbrake slide. Off by default, so a still looks
like the game's showroom.

| key | effect | key | effect |
|---|---|---|---|
| H | light switch: off / stage 1 / 2 / 3 | F7 | backfire |
| F1 | print every key | F8 | rear tyre smoke |
| F2 | night | F9 | dust |
| F3 | bloom | F10 | indicators: off / left / right / hazard |
| F4 | dirt 0 / 0.5 / 1 | F11 | rain / fog light |
| F5 | wet | F12 | hot exhaust |
| F6 | brake-disc temperature | | |

Not driving, the car state is set by hand: PageUp / PageDown rpm
(Shift: gear), Home / End speed, Insert brake pedal, Delete reverse. No key
has two meanings: drive mode owns the arrows, I J K L, X Z Y T, Space and N;
free-look owns W A S D Q E. Every effect has an env var for `--shot`
(`EVOVIEW_LIGHTS`, `_BRAKE`, `_INDICATOR`, `_RPM`, `_SPEED`, `_GEAR`,
`_NIGHT`, `_BLOOM`, `_DIRT`, `_WET`, `_DISC_TEMP`, `_BACKFIRE`, `_SMOKE` ...,
listed in `src/fxstate.rs`), and `--demo T --frames N` sequences carry the
effects along with the car.

## Checking a car

`--check report.json [--sheet dir | sheet.png]` exercises a car without the
game and writes ONE report, `{car, records: [{car, check, severity, detail}],
counts, anim}`:

- animation (`anim_*` records, check_anim.rs): steer lock to lock, every
  corner from droop to bump (and bump + lock), pedals, gears, every bound clip
  and every dash needle - unbound or missing clips, clips that move nothing,
  NaN poses, bones that fly off, tyres inside the bodywork, wheels / discs /
  calipers / steering wheel / needles not bound, suspension data out of range,
  driver hands leaving the rim;
- effects (check_fx.rs): every light function lit on its own and diffed at
  night (DRL, low, high, tail, brake, rain / fog, reverse, the four
  indicators, the four specials), shift lights, the dash (ignition on, values
  changing with speed / rpm / gear), brake-disc glow, all five backfire types,
  dirt, exhaust emitters vs the exhaust geometry, unresolved / placeholder
  textures, fallback materials, near-opaque windows, magenta / black pixels.

The sheet tiles the animation key poses and the effect renders.
`tools/carcheck.py --stock` / `--mods <folder>` runs it over many cars at idle
priority, one at a time, and writes `<tag>.txt` (findings per check, then
every warning / error) and `<tag>.json` (every record). The thresholds were
set on the stock fleet - the known-good baseline - so a finding on a converted
car means something.

## Finding the one material that is wrong

A car is ninety materials and a circuit is nearly three hundred, and when one
of them is wrong the symptom is a single white patch buried in a mesh.
`--materials` lays them all out as spheres, lit by the same shader path the
real content uses, and prints a legend whose row and column match the picture.
`EVOVIEW_MAT_HL=<substr>` paints matching materials magenta when you need to
confirm which one owns a pixel, and `EVOVIEW_MAT_LOG=1` dumps what the loader
decided about every one of them.

## It ships no game content

Tyres, paints, the studio environment and shared track textures all come from
**your own install**. evoview finds `content.kspkg` through Steam's library
folders; if your copy lives somewhere unusual:

```
set EVOVIEW_KSPKG=D:\Games\Assetto Corsa EVO\content.kspkg
```

Nothing from the game is redistributed with the viewer, and nothing is
extracted to disk — the archive is read in place.

## Performance

A 542k-triangle GT3 car, 1080p, RTX 4070:

| | |
|---|---|
| load + first frame | **1.4 s** |
| steady state | **0.5-1.0 ms/frame (1000-2100 fps uncapped)** |
| resident memory | ~1.1 GB |

Vsync is on by default because a viewer has no reason to burn a GPU on frames
nobody sees. `EVOVIEW_NOVSYNC=1` uncaps it, which is how those numbers were
measured. The window title shows live fps and frame time.

## What it renders

Physically-based, built around the way EVO actually stores a car:

- **SpatMap paint layers** — red/green/blue channels each carry their own
  colour, roughness, metalness and clear coat, including the per-layer
  roughness *maps*, not just constants.
- **Clear coat** as a second lobe with its own mask, which is what makes paint
  go glassy at grazing angles.
- **Rectangular area lights.** A studio is lit by softboxes a metre across, and
  that size is the look: a punctual light puts a round glint on a wing, a
  softbox lays a long soft streak down the whole shoulder line. Four of them,
  scaled to the car's own bounding box so a hatchback and a prototype are lit
  the same way.
- **A cabin light**, because an interior is a closed box the outside rig cannot
  reach. Dashboards are black gloss plastic: a diffuse fill multiplied by a
  near-black albedo stays black, so what makes them readable is specular.
- **Glass that transmits.** Fresnel drives coverage, so a windscreen is a few
  percent reflective head-on and a mirror at grazing angles — and you can see
  the interior through it.
- **The driver in the seat**, dressed and posed the way the game does it: the
  car's own `.driverfinalstate` when a livery dresses its driver (SF-25,
  F2004, AC1-converted cars), else the game's default racing or street outfit,
  posed by the car's `driver_position` animation with the arms on a centred
  wheel. On by default; `B` hides it, `EVOVIEW_DRIVER=0` skips it
  (`=racing` / `=street` force the outfit). `tools/lookcmp.py` keeps it off:
  the game's thumbnails are shot with an empty cockpit.
- Multi-bounce AO, multiple-scattering energy compensation, Burley diffuse,
  BC1/BC6H/BC7 uploaded as-is with full mip chains, 16x anisotropy.

See `THIRD-PARTY-NOTICES.md` for the crates it links, and `CHANGELOG.md` for
what changed and why.

## Upscalers (optional)

TAA is the default. AMD FSR 3.1 and NVIDIA DLSS / DLAA work when their
vendor's files sit beside `evoview.exe` (ACECM adds them by drag and drop):

- FSR: `amd_fidelityfx_loader_vk.dll` and `amd_fidelityfx_upscaler_vk.dll`
  from [AMD's FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/tree/main/Kits/FidelityFX/signedbin).
- DLSS: `nvngx_dlss.dll` from [NVIDIA's DLSS SDK](https://github.com/NVIDIA/DLSS/tree/main/lib/Windows_x86_64/rel).
  evoview talks to the NGX runtime in NVIDIA's own driver; no NVIDIA SDK code
  is linked into it.

`EVOVIEW_UPSCALER=fsr|dlss` picks one.

## Building

```
cargo build --release
```

Needs a Vulkan-capable GPU and Rust. Nothing else: no game files, SDKs or
vendor libraries are needed to build it.

## Licence

GPL-3.0-or-later (`LICENSE`). evoview is not affiliated with Kunos Simulazioni
or 505 Games; Assetto Corsa EVO is their trademark. It reads the game from the
player's own install and redistributes none of it.
