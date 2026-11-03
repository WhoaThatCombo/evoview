# Changelog

## Unreleased — accuracy and provenance pass

### Every splat layer draws its own maps; mods open on their own preset

`src/material.rs`, `src/layers.rs`, `src/shader.wgsl`. Green and Blue layers
now draw their colour and roughness maps exactly as UberVehicleMaterial does
(DXIL 0x035a3710, flags named by cbuffer offset): Enable + weight, Replace /
Multiply_BaseColor, map only with <L>_HasBaseColorMap, Modulation_<L>BaseColor,
the layer UV (UVscale applied when either component is non-zero, RotateUV
angle, UVoffset) and CustomPbrUV / UVscalePbr for roughness. The AC1
converter's tiled detail (leather, carpet, carbon on Blue, multiply) drew
WHITE before. Stays at 20 fragment textures: the layer maps are one
`texture_2d_array` per kind (1 map = a view, 2-3 = resampled, RGBA8 / R8).
Also: every 2-vector `<Layer>_UVscale` is read as a 2-vector (flakes, twill,
roughness tiles had fallen back to 1); `Splatmap_Tighten`; Base_BaseColorMap
is ignored when Base_HasBaseColorMap is 0 (the shader's own test); an
opacity-only part tiles at Base_UVscale (MX-5 ND Cup grille net).

`src/load.rs`, `src/parts.rs`, `src/livery.rs`: a mod's `<pkg>.json` sets the
default mech preset, a chosen mech preset wears its first visual preset, and
the pair's `.carfinalstate` supplies the wheel set, rim design and - when the
mech preset fits nothing itself - the parts (third-party converted cars keep
their body kit only there).

lookcmp 66.13 -> 66.09 (E30 / F40 -0.5..0.7: their seat pattern and oil
detail now render; GR86 / Huracan up). Bench 1080p: GPU +0.0 ms, VRAM +0-32 MB.

### The driver sits in the car

`src/driver.rs`. Every car the game ships (71/71) and AC1-converted mods now
have their driver: helmet, head, suit, gloves and shoes from
`content/characters`, with per-livery helmets and suits where the visual preset
names its own `.driverfinalstate`. Placement needs nothing from the actor's
nodes: the `driver_position` animation's bone locals are already in car space.
Each part is skinned against its OWN bind (the gloves' A_Head rest is 1.5 m off
the suit's), head and helmet hang on A_Head through the rig's axis swap, and
the pose is baked once on the CPU. Cost on the bench (1080p, min of 3): GPU
+0.03 ms, VRAM +30-60 MB (driver maps load one mip down), load +0.1 s.

The goal was showroom quality at Vulkan speed, and a tree that is
entirely ours to ship.

### Ships no game content any more

`assets/` used to carry 85 MB of Kunos files — tyre meshes and materials lifted
out of `content.kspkg`. That is not ours to redistribute, so it is gone.
`src/install.rs` now finds the player's own install through Steam's
`libraryfolders.vdf` and layers their `content.kspkg` under the car folder.

This turned out to be a **correctness fix, not just a licensing one**: the
bundled copy was stale. Reading the real install:

| | bundled assets | player's install |
|---|---|---|
| materials bound (M4 GT3) | 80 | **85** |
| meshes | 10 | **14** (tyres appear) |
| fallback batches | 4 (mirrors unbound) | **0** |
| per-layer roughness maps (M2) | 3 | **11 + 12 red-layer** |

### Lighting: rectangular area lights

Four softboxes, sized and placed from the car's own bounding box, using the
representative-point approximation (Karis, *Real Shading in Unreal Engine 4*,
SIGGRAPH 2013), plus a fifth inside the cabin. Punctual lights were the reason
highlights read as glints instead of the long streaks a real studio gives.

Edges are feathered by how far outside the rectangle the reflection landed —
a hard-clamped rect stamps a crisp-edged bar on the paint that reads as a decal
rather than a light.

### Materials

- Roughness got its own texture binding. `data_i = rough_i.or(ao_i)` gave
  roughness and AO one slot; tracks ship roughness, cars ship **both**, so on
  every car the AO map was silently discarded.
- Bound the per-layer roughness, global AO and clear-coat maps EVO ships.
- Multi-bounce AO (Jimenez et al., SIGGRAPH 2016) instead of `ao * colour`,
  which treats occluded pixels as if light never bounces off the albedo.
- Multiple-scattering energy compensation (Fdez-Aguera, JCGT 2019) so rough
  metal stops going grey.
- Burley diffuse (Disney, SIGGRAPH 2012) in place of Lambert.

### Foliage: EVO has more blend modes than we handled

Every tree had a hard-edged dark polygon around its canopy, and the background
forest was flat slabs. The cause: `blendMode` was mapped for 1 (alpha) and 2
(cutout) only, and **everything else fell through to OPAQUE**.

Mode 4 is the alpha-card mode - `background_trees`, `billboard`, `bb_*`,
`truck2_alpha` - 43 materials across four tracks, all drawn opaque, so each card
rendered its transparent region as a dark polygon. Mode 3 is translucent
(`simpleglasslod`). Both are now mapped.

The alpha-test threshold was also hardcoded at 0.5 while EVO ships its own
`a2cCutoff` - 0.15 on the beech branches, which is a lot more leaf than 0.5
keeps. The material's own cutoff is used when it has one.

### Track textures were loading at half resolution

`read_texture_lod(hdr, mips, 1)` - a flat "drop the top mip" on **every** track
texture. Cars loaded at mip 0, tracks did not, which is most of why a track
render looked soft next to a car render: every map, from a 256px sign to a 4K
asphalt sheet, was rendered at half resolution and a quarter of the texels.

The reason was real - the Ring ships hundreds of 4K maps and keeping every texel
OOM'd it - but the cost was being paid by every texture on every track,
including the ones that were never the problem. The skip is now chosen from the
texture's own size: only maps of 4096px or more give up their top level.

| track | old | size-aware | full mip 0 |
|---|---|---|---|
| nurburgring | 4.78 s | **6.69 s** | 8.27 s |
| spa | 3.29 s | **3.25 s** | 3.35 s |
| brands_hatch | 2.41 s | **2.36 s** | 2.79 s |

Free on a normal circuit, ~2 s on the Ring, and nothing ran out of memory at any
setting. `EVOVIEW_TEX_LOD=0` forces full resolution everywhere, `=1` restores
the old behaviour.

### Terrain tile selection

Which library texture gets tiled over a baked-unwrap terrain surface was scored
on filename alone. Three fixes, all about intent rather than spelling:

- **A texture the track itself references beats an unused one from the shared
  library.** Brands Hatch was picking `grass_wild_1_albedo` - dry straw out of
  common_assets that the track never mentions - while shipping
  `grass_2_general_albedo`, the green tile Spa and Monza both use.
- **The tile has to match the material's family, in both directions.** Monza's
  `sand.material` was being handed a grass tile; Donington's `grass.material`
  was handed `gravelblack2_4x4_albedo`, grey stones on a lawn. Red Bull Ring's
  sand now correctly resolves to `sand_base`.
- Dry/wild/dead/straw/burnt variants are penalised.

The "white pad" rule that tiles a splat terrain is now guarded on the condition
it was written for - every splat layer left at the unset white AND a baked
albedo. It had been firing on every splat terrain and stamping a library tile
over surfaces that already had a good authored one.

### ⚠ Brands Hatch yellow is NOT a tile-selection bug

Worth writing down because two rounds of tile fixes did not move it. Its
`grass_ext` is a full three-layer splat material:

| layer | UV scale | tint |
|---|---|---|
| Base | 1 | 1.373 (a brightener, above 1) |
| Red | 850 | 0.543 |
| Green | 0.1 | (0.381, 0.354, 0.855) |
| Blue | 350 | 0.454 |

...blended through a SpatMap. We collapse all of that into "pick one layer
texture and stamp it at one scale" - and the UV scales differ by four orders of
magnitude, so no single choice can be right. The same limitation is the leading
explanation for the blown-white terrain on Donington, Oulton Park and Red Bull
Ring. Implementing the real per-layer blend (three layer albedos, each with its
own scale and tint, mixed by the SpatMap) is the fix; it is not done.

### Sun shadows on tracks

The biggest thing missing outdoors. Nothing cast, so every tree, grandstand and
barrier sat in its own ambient light and a circuit read as a painted model.

A 4096px orthographic depth map from the sun, 3x3 PCF, slope-scaled bias.

- **The map follows the camera target, not the track.** A circuit is kilometres
  across; one map over the whole thing is about a metre per texel, which cannot
  resolve a kerb or a tree trunk. Covering a 400 m neighbourhood instead gives
  0.1 m/texel, which is what makes the shadows read as contact.
- Faded out over the outer 15% of the map, or its edge draws a hard rectangular
  border across the scenery.
- Alpha-BLENDED surfaces are skipped when filling the map: glass would
  otherwise stamp a solid black rectangle on the ground.
- Front-face culling in the depth pass, so near-flat tarmac does not shadow
  itself.
- Off for cars - the studio is area lights, which this cannot represent, and
  the car already has its contact shadow. Car renders are pixel-identical.

Costs about 105 ms on a full headless track render. `EVOVIEW_NO_SHADOW=1`
disables it.

⚠ The shadow pass reuses `vs_main` by binding a second uniform buffer holding
the light matrix, so skinning and instancing cannot drift between the two
passes. It must bind a 1x1 stand-in at the shadow-map slot, though: a texture
cannot be sampled and depth-written in the same pass.

### Materials resolve by path, and can adopt sibling textures

- **Lookup was keyed on the bare filename, and filenames are not unique.** A
  track ships several `bark_mat.material` in different vegetation folders and
  they overwrote each other - last one loaded won, so a mesh could silently get
  another folder's material. Nurburgring has 26 names that collide with a
  materially different entry, interns 10, Spa and Fuji 3. Mesh batches
  reference materials by full path, so this is decidable; lookup now tries the
  full path first and falls back to the basename.
- **Some materials bind no textures at all and rely on sibling files.**
  `bark_mat.material` sits next to `bark_albedo.texture`,
  `bark_normal.texture`, `bark_roughness.texture` and names none of them. Tree
  trunks rendered flat white on every track shipping only that copy (Brands
  Hatch, Cota, Donington). Generalised from the `companion_alpha_ref` trick
  already used for fence cutouts, and kept narrow: the candidate name must be
  derived from the material's own name, so a glass material cannot adopt a
  neighbour's albedo just by sharing a folder.

### Crash: the `interns` track could not open at all

`partial_cmp().unwrap_or(Equal)` is not a total order once a NaN is in the
data - NaN compares Equal to everything, so `a < b`, `b < c` and `a == c` can
all hold. Rust's sort detects that and panics. Now `total_cmp`, with
non-finite values filtered first.

### `--materials`: a debug world for materials

Minecraft ships a debug world that lays out every block state in a grid,
because the fastest way to find the one that is wrong is to see all of them at
once. Finding the roll-bar padding bug took four `EVOVIEW_MAT_HL` passes just to
work out which material owned the white pixels.

`evoview <src> --materials` replaces the scene with one sphere per material,
alphabetical, lit by **the same shader path the real content uses** - so a car
sheet is lit by the studio and a track sheet by the sun. It prints a legend
whose row/column match the picture, flagging fallback (F), alpha (A), cutout
(C), cabin (I), unused (U) and splat/triplanar (S).

Two flags exist to stop the sheet lying to you:

- **U** - the material file exists but no mesh draws it. Without it you go
  hunting a white sphere that never appears on screen.
- **S** - the colour comes from a SpatMap and world position, and a sphere at
  the origin has neither. 72 of Spa's materials are in this class; they have to
  be judged on the real mesh, not here.

`--grid-radius` sets the sphere size, which matters because world-tiled
materials scale by world size - a 1 m ball shows a fraction of one tile.

**It found two bugs on its first run**, which is the point:

1. **`contains("white")` was throwing away real textures.** The placeholder
   test meant to catch a 1x1 stand-in was also matching `whiteline` (Spa's own
   line markings), `painted_scraped_sheet_white_albedo`,
   `aspen_white_custom_1_albedo` and `tyres_albedo_white` - each rendered flat
   white with its texture switched off. The test now requires the name to BE
   `white`, not merely contain it.
2. A visible lead, not yet a fix: 30 of Spa's 49 untextured materials are
   `UberBiplanarMaterial`, i.e. they get their colour from splat layers we fall
   back out of. Flagged S, and worth chasing next.

### The same material model on tracks

Everything learned from EVO's own showroom renderer lived in the car branch,
because that is where it was written. A material model does not care what is
lighting it, so the outdoor branch now uses the same one:

- Burley diffuse in place of Lambert - retroreflection at grazing angles is
  most of why a hillside reads as a surface rather than a flat photo.
- Multi-bounce AO in place of `ao * albedo`, which treats an occluded pixel as
  if light never bounces off the albedo at all.
- Split-sum ambient specular with multiscatter compensation, in place of a
  flat scalar.
- The sun widened by its own angular radius. It is a half-degree disc, not a
  point, and that is what stops polished metal and wet asphalt showing a
  one-pixel aliasing sparkle instead of a small round highlight.

**The light rig does not transfer.** A circuit is lit by the sun and the sky,
not by four softboxes, so the lighting is unchanged - only the BRDF is. This is
the same split as before: take EVO's material model, keep our own lighting.

Effect is real but modest, and honest about where: ~6-9% of pixels move by more
than 5/255, concentrated on grazing hillsides, the armco line and gravel
run-off. Overall exposure moves about 1%. Track surfaces are mostly diffuse,
and without a shadow map there is only so much for an AO model to work with.

The per-layer roughness maps needed no work - the material binding is shared,
so tracks were already getting them (Spa binds 165 roughness + 81 red-layer).

⚠ **Cabin fill is now car-only.** The `int_*` test is a car naming convention,
and on a track it also catches pit-garage interiors (`box_int_2`,
`box_old_int_a` at Spa). Those were being lit by a cabin source positioned from
a car's bounding box - a phantom light in the middle of a circuit.

### White interiors

Once the glass transmitted, the cabins it revealed had white patches in them -
roll-bar padding, wheel-arch liner, underbody, steering rims. Two causes:

1. **A car reaching the terrain code path.** Those parts bind a shared
   *neutral* detail map (`detail_heatshield_colour`,
   `t_fabric_archliner_colour`, `detail_styrofoam_colour`,
   `detail_noise_mechanical_m`) on `Red_BaseColorMap` and have no base albedo
   at all. That combination promotes the Red map to the albedo - written for
   tracks, where a tiled grass photo should be sampled at full white - and
   forced `base = (1,1,1)`. But those maps are authored grey *on purpose*: the
   material supplies the colour in `Base_Basecolor`. `ext_mech_foam_black`
   settles which one is right - the material is named black, its
   `Base_Basecolor` is 0.133 and its `Red_Basecolor` is 0.355. The promotion
   now keeps `Base_Basecolor` for cars and is unchanged for terrain.
   Roll-bar padding went from 233 to 132 sRGB.

2. **Double-lit cabins.** Interior materials were getting the full outside
   softbox rig *and* the cabin fill on top. A cabin is a closed box; only what
   comes through the glass should reach a dashboard. Invisible on the
   near-black plastic this was tuned against, but it blew light leather and
   alcantara straight to white. The outside rig is now attenuated to 20% on
   cabin materials, and the cabin block's ambient specular and wrap were
   brought down to match.

A missing albedo texture also used to fall through to a white 1x1 with the
"use the texture" flag still set, so it rendered flat white with no cue that
anything was wrong. It now falls back to the material's base colour and the
viewer says which materials are affected.

### Two glass bugs, both of which hid the interior

1. **Mine.** The new rig highlight was added straight into alpha. Reflections
   are HDR and routinely exceed 1, so alpha saturated to fully opaque across
   the whole windscreen. Compressed with `x/(1+x)` before use.
2. **Pre-existing.** `f = 0.04 + 0.96 * fresnel(n_dot_v, 0.04)` double-counted
   F0, because `fresnel()` already returns `F0 + (1-F0)(1-cos)^5`. Every window
   was far more mirror than glass.

With both fixed the cabin — seats, wheel, dash, console — is visible through
the screen, with the softbox reflection still on it.

### Load time

Reading the player's archive instead of a bundled copy costs something: the
table is ~80k entries and all of it has to be scanned. Two fixes took the
regression from +620 ms down to +300 ms:

- `Package::open_filtered` skips non-shared entries *before* allocating. The
  old loop allocated twice per entry - a `to_vec()` to decrypt into and a
  `to_lowercase()` for the key - 160k allocations to reach a few hundred files.
- The archive opens on a worker thread while the car folder is walked.

| | load + first frame |
|---|---|
| original (stale bundled assets) | 1.10 s |
| **this build** (player's install, correct assets) | **1.40 s** |

Steady state is 0.5-1.0 ms/frame (1000-2100 fps uncapped) on a 542k-triangle
car at 1080p, RTX 4070. The area lights and the extra maps cost nothing
measurable - they are a handful of ALU behind uniform branches.

### Housekeeping

- `--version`, `--help`, live fps/frame time in the window title.
- `EVOVIEW_MAT_LOG=1` dumps every material (albedo, resolved base colour,
  params) and `EVOVIEW_MAT_HL=<substr>` paints matching materials magenta.
  Between them they identify which material owns a pixel without guessing.
- `EVOVIEW_NOVSYNC=1` to uncap for measurement.
- An invalid `EVOVIEW_KSPKG` now warns instead of silently using another install.
- `THIRD-PARTY-NOTICES.md` generated from the real dependency tree (219 crates;
  the Apache-2.0 ones carry an attribution obligation into the binary).
- Compiler warnings 22 → 16; the rest are pre-existing dead code.

### Not done

- No shadow map. The car has a contact shadow under it, nothing cast.
- Sheen and anisotropy are still missing; alcantara and brushed metal do not
  get their own lobes.
- No track-side reflection captures — a car on a circuit still reflects the
  analytic sky rather than its surroundings.
