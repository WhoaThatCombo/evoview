// Car shading: layered automotive paint in an analytic studio.
//
// The single biggest thing that makes a car read as a car is REFLECTION. A
// perfect BRDF with no environment still looks like plastic, because a real
// body panel is mostly a mirror of its surroundings.
//
// The game's OWN studio cubemap is used when it can be read: the paint scene
// at content	racks\common_assets	extures\cubemaps\studio (BC7, six faces,
// 1024 square, 11 mips). That is the same environment the game lights a
// showroom car with, so highlights land where they land in game rather than
// where a hand-built studio puts them.
//
// ⚠ The earlier note here said those were "texture format 30". They are not -
// format 30 is the irradiance volumes. The cube is format 33.
//
// Without the package (a bare car folder) there is no cube, and the analytic
// studio below stands in: a sky/ground gradient plus soft overhead light
// cards, the setup a studio would use to put rolling highlights down a
// shoulder line.
//
// Paint is two lobes, not one: a coloured base coat, and a thin clear coat
// with its own roughness and Fresnel on top. That second lobe is what gives
// the deep wet look; without it metallics read as grey plastic.

struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    // xyz = ground plane centre, w = plane height
    ground: vec4<f32>,
    // xz = car half-extents for the contact shadow, z = shadow strength
    extent: vec4<f32>,
    // x = 1 outdoor (track), 0 studio (car)
    // y = 1 lit (sun + shade), 0 unlit albedo
    // z = 1 the game's own environment cube is bound, 0 use the analytic one
    // w = highest mip index of that cube, for the roughness lookup
    mode: vec4<f32>,
    // world -> light clip, for the sun shadow map
    light_vp: mat4x4<f32>,
    // x = shadows on, y = depth bias, z = world metres per texel,
    // w = radius the map covers (the shadow fades out at its edge)
    shadow: vec4<f32>,
    // x = exposure multiplier, y = 1 use the game's grading LUT (0 = ACES),
    // z = which LUT (0 studio, 1 outdoor), w = studio rig (softbox) scale
    grade: vec4<f32>,
    // Effects state for this frame (lights::fx_block has the layout):
    // ambient scale, lamp / LED intensities, disc glow, dirt, spots ...
    fx: array<vec4<f32>, 64>,
    // Time of day (sky.rs Sky::pack):
    // 0 xyz toward the key light (sun, or moon at night), w its strength
    // 1 rgb key colour multiplier, w sky light multiplier
    // 2 rgb sky zenith colour, w night 0..1
    // 3 rgb sky horizon colour, w night irradiance multiplier
    sky: array<vec4<f32>, 4>,
    // the far shadow cascade (layer 1): world -> light clip, and
    // x on, y depth bias, z metres per texel, w radius covered
    light_vp_far: mat4x4<f32>,
    shadow_far: vec4<f32>,
    // the cloud deck (clouds.rs CloudGpu::block); row 5 x = ground height
    clouds: array<vec4<f32>, 6>,
    // rain on the glass (rain.rs Rain::block): 0 = intensity, drop phase,
    // drop amount, wipers; then per wiper pivot+reach, axis+tracked, rest
    // direction, 20 recent blade angles
    // 17 1 / scene-copy size, refraction on, time; 18 car forward, km/h
    rain: array<vec4<f32>, 20>,
    // the middle cascade (layer 2): x on, y bias, z m/texel, w normal offset
    light_vp_mid: mat4x4<f32>,
    shadow_mid: vec4<f32>,
    // the driven car's hero map (layer 3): x uv scale into its corner
    // (0 = off), y bias, z m/texel, w normal offset m
    light_vp_hero: mat4x4<f32>,
    shadow_hero: vec4<f32>,
    // the AI cars' lamps (lights::ai_fx_block): 0..4 reflector and 5..9
    // glass intensity per function with the brake off, 10..19 brake on
    ai_fx: array<vec4<f32>, 20>,
};

/// Toward the sun (or the moon at night).
fn key_dir() -> vec3<f32> {
    return normalize(camera.sky[0].xyz);
}

/// Key light radiance relative to the calibrated noon sun.
fn key_rgb() -> vec3<f32> {
    return camera.sky[1].rgb * camera.sky[0].w;
}

struct Material {
    base_color: vec4<f32>,
    // x roughness, y metallic, z has_albedo, w has_normal
    factors: vec4<f32>,
    // x has_ao, y opacity, z clearcoat, w is_ground
    flags: vec4<f32>,
    // x is_glass, y cutout, z reflectance
    // w extra: 0 none, 1 opacity-in-splat, 2 world terrain tile, 3 UV detail
    misc: vec4<f32>,
    // splat layers: rgb colour + w roughness
    layer_r: vec4<f32>,
    layer_g: vec4<f32>,
    layer_b: vec4<f32>,
    // xyz metalness per layer, w has_splat
    layer_metal: vec4<f32>,
    // xyz clear coat per layer
    layer_coat: vec4<f32>,
    // x = world-space triplanar (0 = mesh UV), yz = UV tile, w = detail tile
    uv_scale: vec4<f32>,
    // x base roughness map, y red-layer roughness map (= Red_UVscale), z global-AO strength,
    // w clear-coat mask. 0 means "not shipped, use the constant".
    maps: vec4<f32>,
    // Paint detail layers, see paint_detail(). xyz = Red/Green/Blue detail
    // tile (<Layer>_UVscale), w = clear-coat normal tile.
    layer_uv: vec4<f32>,
    // xyz = Red/Green/Blue detail-normal strength (0 = none), w = clear-coat
    // normal strength (0 = none).
    layer_nscale: vec4<f32>,
    // xyz = Red/Green/Blue clear-coat roughness, w = detail-AO strength
    // (> 0 Red layer's map, < 0 Green layer's, 0 none).
    layer_ccr: vec4<f32>,
    // ---- glass + lights (lights.rs). Paint's layer_uv / layer_nscale /
    // layer_aniso go BEFORE this block when they land.
    // x glass opacity, y EdgeThickness, z emissive mode (0 none, 1 map, 2 mask),
    // w glass reflection strength
    gl_em: vec4<f32>,
    // function-mask colours, R/G/B channel: front (high, low, DRL) and rear
    // (fog, brake, tail)
    emis_f: array<vec4<f32>, 3>,
    emis_r: array<vec4<f32>, 3>,
    // FunctionMask2 colours: reverse, front ind, rear ind, LF RF LR RR special
    emis_x: array<vec4<f32>, 7>,
    // x 1 = lamp lens, y 1 = Dirt_Map bound
    fx_m: vec4<f32>,
    // Splat-layer maps (material.rs MaterialUniform has the full layout).
    // Per layer R/G/B: xy UV scale, z rotation, w colour-map array layer (-1 none)
    layer_xf: array<vec4<f32>, 3>,
    // Per layer: xy UV offset, z roughness-map array layer (-1 none),
    // w 1 = the roughness map uses uv * UVscalePbr
    layer_xo: array<vec4<f32>, 3>,
    // Per layer: xy UVscalePbr. [3].x = Splatmap_Tighten threshold (-1 off)
    layer_pbr: array<vec4<f32>, 4>,
    // The Uber vehicle shaders' own decode (material.rs MaterialUniform::uber)
    uber: array<vec4<f32>, 11>,
    // x 1 = strict mode and this material relied on a guess (strict.rs)
    strict: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
// The game's own environment. mode.z says whether it is real; when it is not,
// this is a 1x1 black cube and the analytic studio below is used instead.
@group(0) @binding(1) var t_env: texture_cube<f32>;
@group(0) @binding(2) var env_samp: sampler;
// The game's split-sum table, indexed (n_dot_v, roughness).
@group(0) @binding(3) var t_brdf: texture_2d<f32>;
// layer 0 the near map (or the car hero map), layer 1 the far cascade
@group(0) @binding(4) var t_shadow: texture_depth_2d_array;
// the cloud deck (clouds.rs), for cloud shadows on cars: noise slices,
// sampler, weather map; clouds_common.wgsl is appended to this file
@group(0) @binding(7) var t_cloud: texture_2d_array<f32>;
@group(0) @binding(8) var s_cloud: sampler;
@group(0) @binding(9) var t_weather: texture_2d<f32>;
// the driven car's reflection probe (probe.rs): the track around the car,
// stored with cube_dir's z negation; live when probe_on()
@group(0) @binding(10) var t_probe: texture_cube<f32>;
// the shared windshield rain maps (common_assets/rain): drop mask, alpha, normal
@group(0) @binding(11) var t_rain_mask: texture_2d<f32>;
@group(0) @binding(12) var t_rain_alpha: texture_2d<f32>;
@group(0) @binding(13) var t_rain_nm: texture_2d<f32>;
// the scene behind the glass, copied before the glass draws (refraction)
@group(0) @binding(14) var t_scene: texture_2d<f32>;
// the side windows' rain streak maps (rain.rs pack_streaks): layers vertical,
// diagonal, horizontal flow; r arrival time, g alpha, b flow normal y
@group(0) @binding(15) var t_streaks: texture_2d_array<f32>;
@group(0) @binding(5) var shadow_samp: sampler_comparison;
// The game's grading LUT: 32x32x64, studio in z 0..32, outdoor in 32..64.
@group(0) @binding(6) var t_lut: texture_3d<f32>;

@group(1) @binding(0) var<uniform> mat: Material;
@group(1) @binding(1) var t_albedo: texture_2d<f32>;
@group(1) @binding(2) var t_normal: texture_2d<f32>;
@group(1) @binding(3) var t_ao: texture_2d<f32>;
@group(1) @binding(4) var samp: sampler;
@group(1) @binding(5) var t_splat: texture_2d<f32>;
@group(1) @binding(6) var t_rough: texture_2d<f32>;
// the splat layers' roughness maps, one array (layers.rs); layer_xo[i].z picks
@group(1) @binding(7) var t_rough_r: texture_2d_array<f32>;
@group(1) @binding(8) var t_gao: texture_2d<f32>;
@group(1) @binding(9) var t_coat: texture_2d<f32>;
// Paint detail layers (material.rs `paint_detail_maps`).
@group(1) @binding(10) var t_layer_n_r: texture_2d<f32>;
@group(1) @binding(11) var t_layer_n_gb: texture_2d<f32>;
@group(1) @binding(12) var t_cc_n: texture_2d<f32>;
@group(1) @binding(13) var t_layer_ao: texture_2d<f32>;
// FunctionMask1 (lamps) or Base_EmissiveMap (interior backlights)
@group(1) @binding(14) var t_emis: texture_2d<f32>;
// The Red splat layer's own colour map - fabric/leather detail on interior
// seats, the real colours of AC1-converted suits (material.rs layer_map).
// (now every splat layer's colour map, one array; layer_xf[i].w picks)
@group(1) @binding(15) var t_layer_c: texture_2d_array<f32>;
// FunctionMask2 (indicators / reverse / specials) and Dirt_Map
@group(1) @binding(16) var t_mask2: texture_2d<f32>;
@group(1) @binding(17) var t_dirt: texture_2d<f32>;

// Skinning matrices, one per bone: anim_world * inverse(bind_world).
//
// ⚠ The engine writes matrices ROW-major with the translation in the last row,
// while WGSL reads a mat4x4 column-major. Uploading the floats untouched is
// therefore exactly the transpose we need, and `bone * v` is correct. Do not
// "fix" the order.
const MAX_BONES: u32 = 256u;
// one block of MAX_BONES per car: block 0 the driven / studio car, block
// 1 + i replay car i (its instance's facing.x = -(1 + i))
@group(2) @binding(0) var<storage, read> bones: array<mat4x4<f32>>;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) bone_idx: vec4<f32>,
    @location(5) bone_wgt: vec4<f32>,
    @location(6) inst0: vec4<f32>,
    @location(7) inst1: vec4<f32>,
    @location(8) inst2: vec4<f32>,
    @location(9) inst3: vec4<f32>,
    // x = CameraFacingMode (0 none, 1/3 Y, 2/4 XAndY)
    @location(10) facing: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    // car-space position (skinned, before the instance transform): the front
    // and rear lamps share one function-mask atlas, and z says which end
    @location(4) obj: vec3<f32>,
    // an AI car (race.rs): x = 1 + its brake pedal, 0 = not an AI
    @location(5) @interpolate(flat) ai: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    var pos = in.position;
    var nrm = in.normal;
    var tan = in.tangent.xyz;
    var tw = in.tangent.w;

    // Linear blend skinning. Geometry with no skin data - tyres, the studio
    // floor - carries zero weights and is left exactly as authored, so one
    // pipeline serves both.
    let total = in.bone_wgt.x + in.bone_wgt.y + in.bone_wgt.z + in.bone_wgt.w;
    // facing.y > 0: an animated TRACK prop, whose bone indices point into the
    // track's own bone buffer, not this car table (drawn at rest here: the
    // shadow pass)
    if (total > 0.0001 && in.facing.y < 0.5) {
        let bone_base = u32(max(-in.facing.x, 0.0) + 0.5) * MAX_BONES;
        var p = vec3<f32>(0.0);
        var n = vec3<f32>(0.0);
        var t = vec3<f32>(0.0);
        for (var k = 0u; k < 4u; k = k + 1u) {
            let w = in.bone_wgt[k];
            if (w <= 0.0) { continue; }
            let bi = u32(max(in.bone_idx[k], 0.0));
            if (bi >= MAX_BONES) { continue; }
            let m = bones[bone_base + bi];
            p = p + (m * vec4<f32>(in.position, 1.0)).xyz * w;
            n = n + (m * vec4<f32>(in.normal, 0.0)).xyz * w;
            t = t + (m * vec4<f32>(in.tangent.xyz, 0.0)).xyz * w;
        }
        let inv = 1.0 / total;
        pos = p * inv;
        nrm = n * inv;
        tan = t * inv;
    }

    out.obj = pos;
    out.ai = in.facing.zw;
    // Instance matrix is uploaded column-major (see inst_matrix in load.rs).
    let world = mat4x4<f32>(in.inst0, in.inst1, in.inst2, in.inst3);
    let face = in.facing.x;
    if (face > 0.5) {
        // CameraFacingMode: keep instance translation + scale, yaw (and
        // optionally pitch) the card toward the camera. That is why forest
        // LOD cards look like striped walls from above when this is ignored.
        let origin = in.inst3.xyz;
        let sx = length(in.inst0.xyz);
        let sy = length(in.inst1.xyz);
        let sz = length(in.inst2.xyz);
        var lp = pos * vec3<f32>(max(sx, 0.0001), max(sy, 0.0001), max(sz, 0.0001));
        var ln = nrm;
        let to = camera.eye.xyz - origin;
        if (face < 1.5 || (face > 2.5 && face < 3.5)) {
            let hz = vec2<f32>(to.x, to.z);
            if (dot(hz, hz) > 1e-8) {
                let fdir = normalize(hz);
                let c = fdir.y;
                let s = fdir.x;
                lp = vec3<f32>(c * lp.x + s * lp.z, lp.y, -s * lp.x + c * lp.z);
                ln = vec3<f32>(c * ln.x + s * ln.z, ln.y, -s * ln.x + c * ln.z);
            }
        } else {
            let fdir = normalize(to);
            var right = cross(vec3<f32>(0.0, 1.0, 0.0), fdir);
            if (dot(right, right) < 1e-8) {
                right = vec3<f32>(1.0, 0.0, 0.0);
            } else {
                right = normalize(right);
            }
            let up = cross(fdir, right);
            lp = right * lp.x + up * lp.y + fdir * lp.z;
            ln = right * ln.x + up * ln.y + fdir * ln.z;
        }
        pos = lp + origin;
        nrm = ln;
        tan = tan;
    } else {
        pos = (world * vec4<f32>(pos, 1.0)).xyz;
        let m3 = mat3x3<f32>(world[0].xyz, world[1].xyz, world[2].xyz);
        nrm = normal_to_world(m3, nrm);
        tan = m3 * tan;
        tw = tangent_w_to_world(m3, tw);
    }

    out.world = pos;
    out.normal = nrm;
    out.uv = in.uv;
    out.tangent = vec4<f32>(tan, tw);
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    return out;
}

const PI: f32 = 3.14159265359;

fn ggx(n_dot_h: f32, rough: f32) -> f32 {
    let a = rough * rough;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

fn smith(n_dot_v: f32, n_dot_l: f32, rough: f32) -> f32 {
    let r = rough + 1.0;
    let k = (r * r) / 8.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

fn fresnel(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

/// A soft rectangular "light card", as used in a photo studio. Returns how
/// much of the card a ray in direction `d` sees.
fn card(d: vec3<f32>, dir: vec3<f32>, width: f32, softness: f32) -> f32 {
    let a = max(dot(normalize(d), normalize(dir)), 0.0);
    return pow(a, width) * softness;
}

/// World-space triplanar sample. Track shaders (UberBiplanarMaterial) do not
/// use mesh UVs — they project from the two strongest axes. Sampling with
/// mesh UVs makes asphalt and grass look like smeared plastic.
fn triplanar(tex: texture_2d<f32>, samp: sampler, p: vec3<f32>, n: vec3<f32>, scale: f32) -> vec4<f32> {
    let an = abs(n);
    let w = pow(max(an, vec3<f32>(0.001)), vec3<f32>(4.0));
    let s = w / max(w.x + w.y + w.z, 1e-4);
    let yz = textureSample(tex, samp, p.yz * scale);
    let xz = textureSample(tex, samp, p.xz * scale);
    let xy = textureSample(tex, samp, p.xy * scale);
    return yz * s.x + xz * s.y + xy * s.z;
}

fn sample2(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>, p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
    if (mat.uv_scale.x > 0.0) {
        return triplanar(tex, samp, p, n, mat.uv_scale.x);
    }
    let tile = vec2<f32>(
        select(1.0, mat.uv_scale.y, abs(mat.uv_scale.y) > 0.0001),
        select(1.0, mat.uv_scale.z, abs(mat.uv_scale.z) > 0.0001),
    );
    return textureSample(tex, samp, uv * tile);
}

/// Daylight sky. A track is outdoors; the studio cards used for cars make
/// asphalt read as grey clay.
fn env_sky(d: vec3<f32>) -> vec3<f32> {
    let y = clamp(d.y, -1.0, 1.0);
    // colours and brightness follow the time of day (sky.rs)
    let k = camera.sky[1].w;
    let zenith = camera.sky[2].rgb * k;
    let horizon = camera.sky[3].rgb * k;
    let ground = vec3<f32>(0.22, 0.20, 0.16) * k;
    if (y >= 0.0) {
        return mix(horizon, zenith, pow(y, 0.45));
    }
    return mix(horizon * 0.85, ground, pow(-y, 0.55));
}

/// A car on the track with its reflection probe drawn: clouds[5].z is the
/// probe's last mip.
fn probe_on() -> bool {
    return camera.clouds[5].y > 0.5 && camera.clouds[5].z > 0.5;
}

/// What the probe sees along `d` for a surface of `rough`ness: the blurred
/// levels are wider cones (probe.rs), picked like the game's cube_lod.
fn probe_env(d: vec3<f32>, rough: f32) -> vec3<f32> {
    let r = clamp(rough, 0.0, 1.0);
    return textureSampleLevel(t_probe, env_samp, cube_dir(d), r * (2.0 - r) * camera.clouds[5].z).rgb;
}

/// The environment: what a mirror pointing along `d` would see.
fn env(d: vec3<f32>) -> vec3<f32> {
    if (probe_on()) {
        return probe_env(d, 0.0);
    }
    // ⚠ Outdoor FIRST. The cube here is the paint studio, and a studio is the
    // wrong environment for a track - letting it win outdoors replaced the sky
    // with a black room, which is not what a car sitting on a circuit reflects.
    // A track's own reflection captures are a separate asset and not wired up.
    if (th_on()) {
        return th_env(d);
    }
    if (camera.mode.x > 0.5) {
        // ⚠ The disc used to sit at (0.38, 0.78, 0.42), not where the sun
        // that lights and shadows the scene is. Same direction now.
        let sun = key_dir();
        var c = env_sky(d);
        c += vec3<f32>(12.0, 11.2, 10.0) * key_rgb() * pow(max(dot(d, sun), 0.0), 220.0);
        return c;
    }
    if (camera.mode.z > 0.5) {
        return textureSampleLevel(t_env, env_samp, cube_dir(d), 0.0).rgb;
    }
    let up = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);

    // Sky: cool and bright overhead, softening to a warm horizon.
    // ⚠ These are deliberately DARK. A photo studio is a black room with a few
    // very bright softboxes; the contrast between the two is what makes paint
    // look like paint. An evenly bright environment desaturates every colour
    // toward grey no matter how correct the BRDF is.
    let zenith = vec3<f32>(0.170, 0.205, 0.290);
    let horizon = vec3<f32>(0.225, 0.225, 0.235);
    let floor_col = vec3<f32>(0.045, 0.045, 0.043);

    var c: vec3<f32>;
    if (d.y >= 0.0) {
        // tighter falloff near the horizon line looks far more like a real sky
        c = mix(horizon, zenith, pow(clamp(d.y, 0.0, 1.0), 0.55));
    } else {
        c = mix(horizon * 0.55, floor_col, pow(clamp(-d.y, 0.0, 1.0), 0.35));
    }

    // Three overhead softboxes. These are what become the long highlights
    // travelling down the roof and shoulder line as the car rotates.
    c += vec3<f32>(1.00, 0.98, 0.94) * card(d, vec3<f32>(0.30, 1.0, 0.15), 26.0, 5.0);
    c += vec3<f32>(0.80, 0.86, 1.00) * card(d, vec3<f32>(-0.85, 0.55, -0.4), 34.0, 2.4);
    c += vec3<f32>(1.00, 0.92, 0.82) * card(d, vec3<f32>(0.15, 0.35, -1.0), 30.0, 1.8);
    return c;
}

// ---- the thumbnail studio ---------------------------------------------------
//
// The game's car thumbnails are NOT shot in the paint studio cube. They are
// rendered inside a dark sphere (content/tracks/interns/thumbnail, walls =
// DefaultMaterial Basecolor 0.0212, roughness 0.6) under the time-of-day sun
// and sky, with thumbnails.graphicsoverride exposure. What a car sees there is
// a bright overhead dome fading to a dark warm wall at the horizon, plus a
// high sun. This is that environment, analytically - the constants are FITTED
// to the game's own thumbnails (L:/evoview_work/scripts/cal.py), not guessed.
// camera.mode.z == 2 selects it.

const TH_ZENITH: vec3<f32> = vec3<f32>(0.32, 0.44, 0.70);
const TH_HORIZON: vec3<f32> = vec3<f32>(0.03, 0.035, 0.045);
const TH_FLOOR: vec3<f32> = vec3<f32>(0.020, 0.019, 0.017);
const TH_K: f32 = 1.0;
// The sphere wall where the sun lights it: a brighter warm band around the
// horizon (the game's backdrop is ~52/255 there vs ~45 elsewhere), which is
// what a car's flanks and a silver car's shoulder line reflect.
const TH_BAND: vec3<f32> = vec3<f32>(0.30, 0.26, 0.20);
const TH_BAND_W: f32 = 0.18;
const TH_SUN_DIR: vec3<f32> = vec3<f32>(0.25, 0.85, 0.45);
const TH_SUN: vec3<f32> = vec3<f32>(4.4, 4.15, 3.8);
const TH_WALL_ALBEDO: f32 = 0.0212;
const TH_SHADOW_POW: f32 = 3.0;
// what the camera sees of the wall/floor; = grade.rs TH_BACKDROP
const TH_BACKDROP: vec3<f32> = vec3<f32>(0.0405, 0.0374, 0.0340);

fn th_on() -> bool {
    return camera.mode.z > 1.5 && camera.mode.x < 0.5;
}

/// Radiance seen along d.
fn th_env(d: vec3<f32>) -> vec3<f32> {
    let y = clamp(d.y, -1.0, 1.0);
    let band = TH_BAND * exp(-(y - 0.05) * (y - 0.05) / (TH_BAND_W * TH_BAND_W));
    if (y >= 0.0) {
        return mix(TH_HORIZON, TH_ZENITH, pow(y, TH_K)) + band;
    }
    return mix(TH_HORIZON, TH_FLOOR, clamp(-y * 5.0, 0.0, 1.0)) + band;
}

/// Cosine-weighted average radiance around n (what a Lambert surface sees).
/// A vertical-gradient environment's irradiance depends on n.y only; this
/// quadratic in u = n.y matches the numerical integral of th_env closely
/// enough at the fitted constants (top, side, bottom).
fn th_irr(n: vec3<f32>) -> vec3<f32> {
    let up = mix(TH_HORIZON, TH_ZENITH, 0.62);
    let side = 0.5 * mix(TH_HORIZON, TH_ZENITH, 0.38) + 0.5 * mix(TH_HORIZON, TH_FLOOR, 0.6);
    let down = mix(TH_HORIZON, TH_FLOOR, 0.75);
    let y = clamp(n.y, -1.0, 1.0);
    if (y >= 0.0) {
        return mix(side, up, y);
    }
    return mix(side, down, -y);
}

/// Blurred environment for a rough reflection: slide from the sharp dome to
/// the irradiance by the same r(2-r) the game uses for its mip choice.
fn th_env_rough(d: vec3<f32>, rough: f32) -> vec3<f32> {
    let r = clamp(rough, 0.0, 1.0);
    return mix(th_env(d), th_irr(d), r * (2.0 - r));
}

// ---- the game's IBL conventions (SPEC_evo_shading.md section 3) -----------

/// ⚠ The game samples EVERY cube with z negated: (R.x, R.y, -R.z). Without it
/// the studio is mirrored front-to-back - the softbox that should sit ahead of
/// the car lands behind it.
fn cube_dir(d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(d.x, d.y, -d.z);
}

/// Perceptual roughness -> mip: `6 r (2 - r) * RoughnessToLodScale`. The scale
/// is CPU-side and undecoded; (mips - 1) / 6 puts r = 1 on the last mip.
fn cube_lod(rough: f32) -> f32 {
    let r = clamp(rough, 0.0, 1.0);
    return r * (2.0 - r) * camera.mode.w;
}

/// Car diffuse in the game is NOT an irradiance map or SH: it is the same cube
/// at a fixed coarse mip along N, times 1.5. The mip level is CPU-side and
/// undecoded; 4 below the last (16 px faces on the 1024 studio) is a guess.
fn car_diffuse_env(n: vec3<f32>) -> vec3<f32> {
    if (th_on()) {
        return th_irr(n);
    }
    let mip = max(camera.mode.w - 4.0, 0.0);
    return textureSampleLevel(t_env, env_samp, cube_dir(n), mip).rgb * 1.5;
}

/// The game's split-sum ambient for a car: specular and diffuse IBL, with
/// F90 = saturate(50 F0.g), Fdez-Aguera multiscatter taken from DIFFUSE only
/// (it is never added back to specular), Jimenez multi-bounce AO on diffuse and
/// Lagarde specular occlusion (multi-bounced by F0) on specular.
fn car_ibl(n: vec3<f32>, r: vec3<f32>, n_dot_v: f32, albedo: vec3<f32>,
           metallic: f32, rough: f32, f0: vec3<f32>, ao: f32) -> array<vec3<f32>, 2> {
    let lut = env_brdf_ab(rough, n_dot_v);
    let f90 = clamp(50.0 * f0.g, 0.0, 1.0);
    let fss = f0 * lut.x + vec3<f32>(f90 * lut.y);
    let ems = 1.0 - (lut.x + lut.y);
    let favg = f0 + (vec3<f32>(1.0) - f0) / 21.0;
    let fms = ems * fss * favg / (vec3<f32>(1.0) - ems * favg);
    let diff_col = albedo * (1.0 - metallic);
    let diffuse = car_diffuse_env(n) * diff_col
                * max(vec3<f32>(1.0) - fss - fms, vec3<f32>(0.0))
                * multi_bounce(ao, diff_col);
    let so = clamp(pow(n_dot_v + ao, exp2(-16.0 * rough * rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
    let spec = env_rough(r, rough) * fss * multi_bounce(so, f0);
    return array<vec3<f32>, 2>(diffuse, spec);
}

/// The thumbnail sun: Frostbite-normalised Burley + GGX/height-correlated
/// Smith, as the game's direct loop (SPEC 2.2). No cast shadow (the studio
/// floor has its own contact shadow); a cabin gets a fraction of it.
fn th_sun(n: vec3<f32>, v: vec3<f32>, albedo: vec3<f32>, metallic: f32,
          rough: f32, f0: vec3<f32>) -> vec3<f32> {
    let l = normalize(TH_SUN_DIR);
    let ndl = max(dot(n, l), 1e-4);
    let ndv = max(dot(n, v), 1e-4);
    let h = normalize(l + v);
    let ldh = max(dot(l, h), 0.0);
    let ndh = max(dot(n, h), 0.0);
    let r = clamp(rough, 0.045, 1.0);
    let fd90 = 0.5 * r + 2.0 * r * ldh * ldh;
    let fd = (1.0 / PI - 0.1075086 * r) * (1.0 + (fd90 - 1.0) * pow(1.0 - ndl, 5.0))
           * (1.0 + (fd90 - 1.0) * pow(1.0 - ndv, 5.0));
    let a2 = r * r * r * r;
    let dd = ndh * ndh * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * dd * dd);
    let vis = 0.5 / (ndl * sqrt(ndv * ndv * (1.0 - a2) + a2) + ndv * sqrt(ndl * ndl * (1.0 - a2) + a2));
    let f = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
    let lit = select(0.0, 1.0, dot(n, l) > 0.0);
    return (albedo * (1.0 - metallic) * fd + d * vis * f) * TH_SUN * PI * ndl * lit;
}

/// Roughness-aware environment lookup. A rough surface sees a blurred version
/// of the world; approximate that by widening the cards and pulling toward the
/// average sky rather than sampling a mip chain we do not have.
fn env_rough(d: vec3<f32>, rough: f32) -> vec3<f32> {
    if (th_on()) {
        return th_env_rough(d, rough);
    }
    if (probe_on()) {
        return probe_env(d, rough);
    }
    if (camera.mode.z > 0.5 && camera.mode.x < 0.5) {
        // A rough surface sees a blurred world, and the mip chain IS that
        // blur - so pick a level instead of lerping toward a made-up average.
        // sqrt spreads the sharp end over more levels, which is where the eye
        // actually notices the difference on a car body.
        return textureSampleLevel(t_env, env_samp, cube_dir(d), cube_lod(rough)).rgb;
    }
    let sharp = env(d);
    let broad = mix(vec3<f32>(0.205, 0.235, 0.300), vec3<f32>(0.068, 0.068, 0.070),
                    clamp(-d.y * 0.5 + 0.5, 0.0, 1.0))
              * select(1.0, camera.sky[1].w, camera.mode.x > 0.5);
    return mix(sharp, broad, clamp(rough * 1.35, 0.0, 1.0));
}

/// Analytic split-sum: the environment BRDF, so Fresnel stays energy-correct
/// at grazing angles instead of blowing out.
fn env_brdf_ab(rough: f32, n_dot_v: f32) -> vec2<f32> {
    if (camera.mode.z > 0.5) {
        return textureSampleLevel(
            t_brdf, env_samp,
            vec2<f32>(clamp(n_dot_v, 0.0, 1.0), clamp(rough, 0.0, 1.0)), 0.0).rg;
    }
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = vec4<f32>(rough, rough, rough, rough) * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + vec2<f32>(r.z, r.w);
}

fn env_brdf(f0: vec3<f32>, rough: f32, n_dot_v: f32) -> vec3<f32> {
    if (camera.mode.z > 0.5) {
        // The game ships this curve as a texture; use it rather than fitting
        // it. Same inputs, no polynomial in the middle.
        let ab = textureSampleLevel(
            t_brdf, env_samp,
            vec2<f32>(clamp(n_dot_v, 0.0, 1.0), clamp(rough, 0.0, 1.0)), 0.0).rg;
        return f0 * ab.x + vec3<f32>(ab.y);
    }
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = vec4<f32>(rough, rough, rough, rough) * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + vec2<f32>(r.z, r.w);
    return f0 * ab.x + vec3<f32>(ab.y);
}

fn direct(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>,
          albedo: vec3<f32>, metallic: f32, rough: f32) -> vec3<f32> {
    let n_dot_l = max(dot(n, l), 0.0);
    if (n_dot_l <= 0.0) { return vec3<f32>(0.0); }
    let h = normalize(v + l);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f = fresnel(max(dot(h, v), 0.0), f0);
    let spec = (ggx(n_dot_h, rough) * smith(n_dot_v, n_dot_l, rough) * f)
             / max(4.0 * n_dot_v * n_dot_l, 1e-4);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    return (kd * albedo / PI + spec) * radiance * n_dot_l;
}

/// Specular only, no diffuse. A clear coat is a transparent film: it reflects,
/// it does not scatter. Giving it a diffuse term washes the base coat out to
/// pastel, which is exactly what it looks like.
fn direct_spec(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>,
               rough: f32) -> vec3<f32> {
    let n_dot_l = max(dot(n, l), 0.0);
    if (n_dot_l <= 0.0) { return vec3<f32>(0.0); }
    let h = normalize(v + l);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);
    let f = fresnel(max(dot(h, v), 0.0), vec3<f32>(0.04));
    let spec = (ggx(n_dot_h, rough) * smith(n_dot_v, n_dot_l, rough) * f)
             / max(4.0 * n_dot_v * n_dot_l, 1e-4);
    return spec * radiance * n_dot_l;
}

/// Soft elliptical contact shadow under the car. Cheap, and it is what stops
/// the car reading as floating - far more important than its accuracy.
fn contact_shadow(p: vec3<f32>) -> f32 {
    let d = vec2<f32>(p.x - camera.ground.x, p.z - camera.ground.z)
          / max(camera.extent.xy, vec2<f32>(0.05));
    let r = length(d);
    return 1.0 - camera.extent.z * (1.0 - smoothstep(0.35, 1.05, r));
}

/// The game's own contact shadow, stamped flat on the ground: black, with
/// the authored texture as coverage. Blended over whatever is below it, so it
/// works on the studio floor and on a track alike.
fn contact_decal(uv: vec2<f32>) -> vec4<f32> {
    let s = textureSample(t_albedo, samp, uv);
    // The masks are BC4: white = full shadow, black = none. flags.y carries
    // the actor's opacity (f7).
    return vec4<f32>(0.0, 0.0, 0.0, clamp(s.r * mat.flags.y, 0.0, 1.0));
}

// ACES filmic curve. Reinhard washes out saturated paint; this keeps red red
// while still rolling the highlights off cleanly.
fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

/// The ONLY exit for final colour: exposure, then the game's grading LUT.
///
/// AC EVO's last pass is `c * exposure / adapted`, then a 32^3 grading LUT
/// (grade.rs: colour correction, GT curve, path to white) read through a log2 shaper.
/// The LUT holds DISPLAY-LINEAR values and the game writes them to an sRGB
/// target - which is exactly what our *UnormSrgb target does, so no gamma
/// here. Per fragment rather than a post pass: same maths, no extra pass.
fn tonemap(c: vec3<f32>) -> vec3<f32> {
    // HDR pipeline (render.rs HDR): exposed linear light, tonemapped once in
    // the post pass
    if (camera.grade.y > 1.5) {
        return hdr_out(c * camera.grade.x);
    }
    if (camera.grade.y < 0.5) {
        return aces(c);
    }
    let x = max(c * camera.grade.x, vec3<f32>(0.0));
    var uvw = clamp((log2(x + vec3<f32>(0.0009765625)) + vec3<f32>(10.0)) / 16.0,
                    vec3<f32>(0.0), vec3<f32>(1.0));
    uvw = uvw * (31.0 / 32.0) + vec3<f32>(0.5 / 32.0);
    // two LUTs stacked along z
    uvw.z = (uvw.z + camera.grade.z) * 0.5;
    return textureSampleLevel(t_lut, env_samp, uvw, 0.0).rgb;
}

/// Showroom sun for the hero shadow: 83 deg elevation (white_showroom), on
/// the key light's azimuth. Must match CAR_SUN in render.rs.
const CAR_SUN: vec3<f32> = vec3<f32>(0.10446, 0.99255, 0.06268);

// How hard the hero shadow bites. IblDiffuse/SpecularShadowFloor are runtime
// frameData in the game (not in any content file), so these are tuned by eye
// against the reference shots.
const HERO_DIRECT: f32 = 0.75;
const HERO_IBL_DIFF_FLOOR: f32 = 0.55;
const HERO_IBL_SPEC_FLOOR: f32 = 0.75;
const HERO_FLOOR: f32 = 0.55;

const KEY: vec3<f32> = vec3<f32>(0.30, 0.92, 0.18);


/// Multi-bounce ambient occlusion.
///
/// Plain `ao * colour` treats every occluded pixel as if light never bounces
/// back off the albedo, which reads sooty on light paint. This puts that energy
/// back, tinted by the surface's own colour.
///
/// Cubic fit from Jimenez et al., "Practical Realtime Strategies for Accurate
/// Indirect Occlusion" (SIGGRAPH 2016 course notes) - the published constants.
fn multi_bounce(x: f32, k: vec3<f32>) -> vec3<f32> {
    let a = 2.0404 * k - 0.3324;
    let b = 0.6417 - 4.7951 * k;
    let c = 2.7552 * k + 0.6903;
    return max(vec3<f32>(x), ((x * a + b) * x + c) * x);
}

/// Multiple-scattering energy compensation.
///
/// Single-scattering GGX loses energy as roughness climbs, so rough metal goes
/// grey and dead. Fdez-Aguera, "A Multiple-Scattering Microfacet Model for
/// Real-Time Image-Based Lighting" (JCGT 2019) recovers it from the same
/// split-sum terms we already sample.
fn ms_compensate(f0: vec3<f32>, lut: vec2<f32>) -> vec3<f32> {
    let e = lut.x + lut.y;
    return vec3<f32>(1.0) + f0 * (1.0 / max(e, 1e-3) - 1.0);
}


// ---------------------------------------------------------------------------
// Rectangular area lights.
//
// A photo studio is lit by softboxes - big rectangles a metre or two across -
// and that size IS the look. A punctual light puts a small round glint on a
// wing; a 1.5 m softbox lays a long soft streak down the whole shoulder line.
// Reflected cards alone cannot do it: they show up in the mirror direction but
// contribute no diffuse and no correctly-shaped specular lobe.
//
// Representative-point approximation, after Karis, "Real Shading in Unreal
// Engine 4" (SIGGRAPH 2013 course notes): find the point on the rectangle
// closest to the reflection ray, treat that as a punctual light, then widen the
// specular lobe by the light's solid angle and renormalise so total energy is
// preserved. Cheap - no LUTs, ~20 ALU per light - and stable under motion.
// ---------------------------------------------------------------------------

struct AreaLight {
    centre: vec3<f32>,
    ax: vec3<f32>,      // unit, half-width along this axis is hx
    ay: vec3<f32>,      // unit, half-height along this axis is hy
    hx: f32,
    hy: f32,
    colour: vec3<f32>,  // already multiplied by intensity
    ln: vec3<f32>,      // unit normal, ALWAYS pointing at the subject
};

/// Build a rectangle from where it is and what it points at. Deriving the
/// basis from the aim direction means the normal always faces the subject -
/// getting that backwards silently kills the light, since a softbox only
/// emits from its front face.
fn make_light(pos: vec3<f32>, aim: vec3<f32>, hx: f32, hy: f32,
              colour: vec3<f32>) -> AreaLight {
    var L: AreaLight;
    let n = normalize(aim - pos);
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(n.y) > 0.95) { up = vec3<f32>(0.0, 0.0, 1.0); }
    L.ax = normalize(cross(up, n));
    L.ay = cross(n, L.ax);
    L.ln = n;
    L.centre = pos;
    L.hx = hx;
    L.hy = hy;
    L.colour = colour;
    return L;
}

/// Where the studio lights sit, in car-relative units, so a hatchback and a
/// prototype get the same rig rather than one of them sitting outside it.
/// The rig, with the four outside softboxes scaled by Camera.grade.w (the
/// game lights its thumbnails by IBL alone; see grade.rs RIG). The cabin
/// source (i = 4) is not scaled.
fn studio_light(i: i32, r: f32, centre: vec3<f32>) -> AreaLight {
    var L = studio_light_raw(i, r, centre);
    if (i < 4) {
        L.colour = L.colour * camera.grade.w;
    }
    return L;
}

fn studio_light_raw(i: i32, r: f32, centre: vec3<f32>) -> AreaLight {
    if (i == 0) {
        // Key: high, front-left, large, a touch warm.
        return make_light(centre + vec3<f32>(-1.05 * r, 2.35 * r, 0.95 * r), centre,
                          1.35 * r, 0.85 * r, vec3<f32>(3.35, 3.24, 3.02));
    } else if (i == 1) {
        // Fill: opposite side, wide and cool, weak - opens the shadows without
        // flattening the form.
        return make_light(centre + vec3<f32>(1.45 * r, 1.25 * r, -0.55 * r), centre,
                          1.60 * r, 0.95 * r, vec3<f32>(0.60, 0.68, 0.86));
    } else if (i == 2) {
        // Top strip along the car: this is the highlight that travels down the
        // roof and bonnet as it turns.
        return make_light(centre + vec3<f32>(0.0, 2.90 * r, 0.0), centre,
                          2.30 * r, 0.42 * r, vec3<f32>(1.55, 1.57, 1.62));
    }
    if (i == 3) {
        // Rim: low and behind, separates the tail from a dark backdrop.
        return make_light(centre + vec3<f32>(0.35 * r, 0.85 * r, -2.20 * r), centre,
                          1.15 * r, 0.55 * r, vec3<f32>(0.95, 0.93, 1.02));
    }
    // Cabin light. A dashboard is black gloss plastic: a diffuse fill
    // multiplied by a near-black albedo stays black, and what actually makes
    // it readable is the SPECULAR - the sheen of a light source on the gloss.
    // So the interior gets its own small source rather than a lift in exposure.
    return make_light(centre + vec3<f32>(0.0, 0.70 * r, -0.05 * r),
                      centre + vec3<f32>(0.0, -0.40 * r, -0.25 * r),
                      0.62 * r, 0.62 * r, vec3<f32>(2.60, 2.62, 2.75));
}

/// Burley (Disney) diffuse - Lambert is too flat at grazing angles and loses
/// the soft edge falloff that makes paint look like paint.
/// Burley et al., "Physically-Based Shading at Disney" (SIGGRAPH 2012).
fn burley(n_dot_l: f32, n_dot_v: f32, l_dot_h: f32, rough: f32) -> f32 {
    let f90 = 0.5 + 2.0 * rough * l_dot_h * l_dot_h;
    let fl = pow(1.0 - n_dot_l, 5.0);
    let fv = pow(1.0 - n_dot_v, 5.0);
    return (1.0 + (f90 - 1.0) * fl) * (1.0 + (f90 - 1.0) * fv) * (1.0 / 3.14159265);
}


/// Just the specular half of a rectangle, for layers that mirror the rig
/// without a diffuse term of their own - glass and the clear coat. A
/// windscreen that only sees the blurred environment never shows the crisp
/// streak of a softbox, which is most of what reads as "glass".
fn area_spec(L: AreaLight, p: vec3<f32>, n: vec3<f32>, v: vec3<f32>,
             rough: f32, f0: vec3<f32>) -> vec3<f32> {
    let refl = reflect(-v, n);
    let denom = dot(refl, L.ln);
    var q = L.centre;
    if (abs(denom) > 1e-4) {
        let t = dot(L.centre - p, L.ln) / denom;
        if (t > 0.0) { q = p + refl * t; }
    }
    let d = q - L.centre;
    let px = dot(d, L.ax);
    let py = dot(d, L.ay);
    let closest = L.centre
        + L.ax * clamp(px, -L.hx, L.hx)
        + L.ay * clamp(py, -L.hy, L.hy);
    let to = closest - p;
    let dist = max(length(to), 1e-4);
    let l = to / dist;
    // Real softboxes have diffusion at the edge. Clamping to a hard rectangle
    // stamps a crisp-edged bar on the paint, which reads as a decal rather
    // than a light, so fade by how far outside the rect the reflection landed.
    let over = length(vec2<f32>(max(abs(px) - L.hx, 0.0), max(abs(py) - L.hy, 0.0)));
    let feather = 1.0 / (1.0 + over * over * 5.0 / max(L.hx * L.hy, 1e-3));

    let radius = 0.5 * (L.hx + L.hy);
    let a = max(rough * rough, 1e-4);
    let a_p = clamp(a + radius / (3.0 * dist), 0.0, 1.0);
    let norm = (a / a_p) * (a / a_p);

    let facing = max(dot(-l, L.ln), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 1e-4);
    let h = normalize(l + v);
    let spec = ggx(max(dot(n, h), 0.0), sqrt(a_p)) * smith(n_dot_v, n_dot_l, rough)
             * fresnel(max(dot(h, v), 0.0), f0)
             / (4.0 * n_dot_v * max(n_dot_l, 1e-4) + 1e-4);
    let area = 4.0 * L.hx * L.hy;
    return spec * norm * n_dot_l * facing / (dist * dist) * area * feather * L.colour;
}

/// Diffuse + specular from one rectangle.
fn area_light(L: AreaLight, p: vec3<f32>, n: vec3<f32>, v: vec3<f32>,
              albedo: vec3<f32>, metallic: f32, rough: f32, f0: vec3<f32>) -> vec3<f32> {
    let ln = L.ln;

    // --- specular: the point on the rect closest to the reflection ray ------
    let refl = reflect(-v, n);
    let denom = dot(refl, ln);
    var q = L.centre;
    if (abs(denom) > 1e-4) {
        let t = dot(L.centre - p, ln) / denom;
        if (t > 0.0) { q = p + refl * t; }
    }
    let d = q - L.centre;
    let px = dot(d, L.ax);
    let py = dot(d, L.ay);
    let closest = L.centre
        + L.ax * clamp(px, -L.hx, L.hx)
        + L.ay * clamp(py, -L.hy, L.hy);
    let over = length(vec2<f32>(max(abs(px) - L.hx, 0.0), max(abs(py) - L.hy, 0.0)));
    let feather = 1.0 / (1.0 + over * over * 5.0 / max(L.hx * L.hy, 1e-3));

    let to_spec = closest - p;
    let dist_s = max(length(to_spec), 1e-4);
    let ls = to_spec / dist_s;

    // Widen the lobe by the solid angle the rect covers, then renormalise:
    // a big light spreads the same energy over a broader highlight.
    let radius = 0.5 * (L.hx + L.hy);
    let a = max(rough * rough, 1e-3);
    let a_p = clamp(a + radius / (3.0 * dist_s), 0.0, 1.0);
    let norm = (a / a_p) * (a / a_p);

    // Light only emits from its front face, and only reaches front faces.
    let facing = max(dot(-ls, ln), 0.0);
    let n_dot_ls = max(dot(n, ls), 0.0);
    let atten_s = facing / (dist_s * dist_s);

    let h = normalize(ls + v);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);
    let spec = ggx(n_dot_h, sqrt(a_p)) * smith(n_dot_v, n_dot_ls, rough)
             * fresnel(max(dot(h, v), 0.0), f0)
             / (4.0 * n_dot_v * max(n_dot_ls, 1e-4) + 1e-4);

    // --- diffuse: the whole rect, sampled at its centre ---------------------
    let to_diff = L.centre - p;
    let dist_d = max(length(to_diff), 1e-4);
    let ld = to_diff / dist_d;
    let n_dot_ld = max(dot(n, ld), 0.0);
    let atten_d = max(dot(-ld, ln), 0.0) / (dist_d * dist_d);
    let area = 4.0 * L.hx * L.hy;

    let kd = (vec3<f32>(1.0) - fresnel(n_dot_v, f0)) * (1.0 - metallic);
    let diffuse = kd * albedo * burley(n_dot_ld, n_dot_v, max(dot(h, ld), 0.0), rough)
                * n_dot_ld * atten_d * area;
    let specular = spec * norm * n_dot_ls * atten_s * area * feather;

    return (diffuse + specular) * L.colour;
}


/// 5x5 "optimized PCF" (Castano, The Witness): nine hardware-bilinear
/// comparison taps whose positions and weights reproduce a 5x5 tent. The
/// game's sun and hero shadows both use this filter (SPEC_evo_shading 4.3).
/// `uv` in [0,1], `z` the reference depth, `dim` the map edge in texels.
/// 3x3 tent from four hardware-bilinear compare taps (the FilterSize-3 case
/// of the same Castano scheme). Used for the car hero map instead of pcf5:
/// at its ~5 mm texels the penumbra still reads soft, and the 5 taps saved
/// are most of the hero map's frame cost (+0.04 -> +0.02 ms at 1080p).
fn pcf3(uv: vec2<f32>, z: f32, dim: f32, layer: i32) -> f32 {
    let inv = 1.0 / dim;
    let p = uv * dim;
    var base = floor(p + 0.5);
    let s = p.x + 0.5 - base.x;
    let t = p.y + 0.5 - base.y;
    base = (base - 0.5) * inv;
    let uw0 = 3.0 - 2.0 * s;
    let uw1 = 1.0 + 2.0 * s;
    let u0 = (2.0 - s) / uw0 - 1.0;
    let u1 = s / uw1 + 1.0;
    let vw0 = 3.0 - 2.0 * t;
    let vw1 = 1.0 + 2.0 * t;
    let v0 = (2.0 - t) / vw0 - 1.0;
    let v1 = t / vw1 + 1.0;
    var sum = uw0 * vw0 * textureSampleCompareLevel(t_shadow, shadow_samp, base + vec2<f32>(u0, v0) * inv, layer, z);
    sum += uw1 * vw0 * textureSampleCompareLevel(t_shadow, shadow_samp, base + vec2<f32>(u1, v0) * inv, layer, z);
    sum += uw0 * vw1 * textureSampleCompareLevel(t_shadow, shadow_samp, base + vec2<f32>(u0, v1) * inv, layer, z);
    sum += uw1 * vw1 * textureSampleCompareLevel(t_shadow, shadow_samp, base + vec2<f32>(u1, v1) * inv, layer, z);
    return sum / 16.0;
}

fn pcf5(uv: vec2<f32>, z: f32, dim: f32, layer: i32) -> f32 {
    let inv = 1.0 / dim;
    let p = uv * dim;
    var base = floor(p + 0.5);
    let s = p.x + 0.5 - base.x;
    let t = p.y + 0.5 - base.y;
    base = (base - 0.5) * inv;

    let uw0 = 4.0 - 3.0 * s;
    let uw1 = 7.0;
    let uw2 = 1.0 + 3.0 * s;
    let u0 = (3.0 - 2.0 * s) / uw0 - 2.0;
    let u1 = (3.0 + s) / uw1;
    let u2 = s / uw2 + 2.0;

    let vw0 = 4.0 - 3.0 * t;
    let vw1 = 7.0;
    let vw2 = 1.0 + 3.0 * t;
    let v0 = (3.0 - 2.0 * t) / vw0 - 2.0;
    let v1 = (3.0 + t) / vw1;
    let v2 = t / vw2 + 2.0;

    var sum = 0.0;
    sum += uw0 * vw0 * textureSampleCompareLevel(t_shadow, shadow_samp, base + vec2<f32>(u0, v0) * inv, layer, z);
    sum += uw1 * vw0 * textureSampleCompareLevel(t_shadow, shadow_samp, base + vec2<f32>(u1, v0) * inv, layer, z);
    sum += uw2 * vw0 * textureSampleCompareLevel(t_shadow, shadow_samp, base + vec2<f32>(u2, v0) * inv, layer, z);
    sum += uw0 * vw1 * textureSampleCompareLevel(t_shadow, shadow_samp, base + vec2<f32>(u0, v1) * inv, layer, z);
    sum += uw1 * vw1 * textureSampleCompareLevel(t_shadow, shadow_samp, base + vec2<f32>(u1, v1) * inv, layer, z);
    sum += uw2 * vw1 * textureSampleCompareLevel(t_shadow, shadow_samp, base + vec2<f32>(u2, v1) * inv, layer, z);
    sum += uw0 * vw2 * textureSampleCompareLevel(t_shadow, shadow_samp, base + vec2<f32>(u0, v2) * inv, layer, z);
    sum += uw1 * vw2 * textureSampleCompareLevel(t_shadow, shadow_samp, base + vec2<f32>(u1, v2) * inv, layer, z);
    sum += uw2 * vw2 * textureSampleCompareLevel(t_shadow, shadow_samp, base + vec2<f32>(u2, v2) * inv, layer, z);
    return sum / 144.0;
}

/// Sun visibility from the shadow map: 0 fully shadowed, 1 fully lit.
///
/// ⚠ Slope-scaled bias, not a constant. A flat constant big enough to stop
/// acne on a surface facing the sun edge-on is big enough to make everything
/// else float above its own shadow ("peter-panning"), and a circuit is mostly
/// near-flat tarmac where that reads as a detached grey smear.
///
/// The map only covers a neighbourhood of the camera target, so the result is
/// faded back to "lit" at the edge - otherwise the far half of a circuit sits
/// under a hard rectangular shadow border.
fn sun_shadow(world: vec3<f32>, n_dot_l: f32) -> f32 {
    if (camera.shadow.x < 0.5) {
        return 1.0;
    }
    let lc = camera.light_vp * vec4<f32>(world, 1.0);
    if (lc.w <= 0.0) {
        return far_shadow(world, n_dot_l);
    }
    let ndc = lc.xyz / lc.w;
    if (ndc.z <= 0.0 || ndc.z >= 1.0) {
        return far_shadow(world, n_dot_l);
    }
    var uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return far_shadow(world, n_dot_l);
    }
    let slope = clamp(1.0 - n_dot_l, 0.0, 1.0);
    let bias = camera.shadow.y * (1.0 + 4.0 * slope);
    let dim = f32(textureDimensions(t_shadow).x);
    if (camera.shadow.x > 1.5) {
        return pcf3(uv, ndc.z - bias, dim, 0);
    }
    let lit = pcf5(uv, ndc.z - bias, dim, 0);
    // (The car hero map, shadow.x == 2, returned above: it is fitted around
    // the car, so there is no edge to hide.)
    // Hand over to the far cascade over the outer 15% of the near map, or
    // fade to lit there when there is none.
    let d = max(abs(uv.x - 0.5), abs(uv.y - 0.5)) * 2.0;
    let t = smoothstep(0.7, 1.0, d);
    if (t <= 0.0) {
        return lit;
    }
    return mix(lit, far_shadow(world, n_dot_l), t);
}

/// The far cascade (layer 1): 4 km at ~1 m a texel, faded out at its edge;
/// the middle one (layer 2) first where it covers.
fn far_shadow(world: vec3<f32>, n_dot_l: f32) -> f32 {
    if (camera.shadow_far.x < 0.5) {
        return 1.0;
    }
    if (camera.shadow_mid.x > 0.5) {
        let mc = camera.light_vp_mid * vec4<f32>(world, 1.0);
        let mn = mc.xyz / mc.w;
        let muv = mn.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
        let md = max(abs(muv.x - 0.5), abs(muv.y - 0.5)) * 2.0;
        if (mn.z > 0.0 && mn.z < 1.0 && md < 0.8) {
            let slope = clamp(1.0 - n_dot_l, 0.0, 1.0);
            return pcf5(muv, mn.z - camera.shadow_mid.y * (1.0 + 4.0 * slope),
                        f32(textureDimensions(t_shadow).x), 2);
        }
    }
    let lc = camera.light_vp_far * vec4<f32>(world, 1.0);
    if (lc.w <= 0.0) {
        return 1.0;
    }
    let ndc = lc.xyz / lc.w;
    if (ndc.z <= 0.0 || ndc.z >= 1.0) {
        return 1.0;
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }
    let slope = clamp(1.0 - n_dot_l, 0.0, 1.0);
    let bias = camera.shadow_far.y * (1.0 + 4.0 * slope);
    let lit = pcf3(uv, ndc.z - bias, f32(textureDimensions(t_shadow).x), 1);
    let d = max(abs(uv.x - 0.5), abs(uv.y - 0.5)) * 2.0;
    return mix(lit, 1.0, smoothstep(0.8, 1.0, d));
}

/// The sun shadow on a car outdoors: the cascades, and the driven car's own
/// hero map (render.rs hero_light_matrix) for its self-shadowing - offset
/// along the normal by its texels, as the game's resolve does.
fn car_sun_shadow(world: vec3<f32>, n: vec3<f32>, ndl: f32) -> f32 {
    let s = sun_shadow(world, ndl);
    let sc = camera.shadow_hero;
    if (sc.x <= 0.0) {
        return s;
    }
    let slope = clamp(1.0 - ndl, 0.0, 1.0);
    let lc = camera.light_vp_hero * vec4<f32>(world + n * (sc.w * slope), 1.0);
    let ndc = lc.xyz / lc.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (ndc.z <= 0.0 || ndc.z >= 1.0 || any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return s;
    }
    let h = pcf5(uv * sc.x, ndc.z - sc.y * (1.0 + 2.0 * slope), f32(textureDimensions(t_shadow).x), 3);
    return min(s, h);
}

/// Lamp intensity of lighting function `f` this frame (lights::fx_block):
/// reflector or lens column.
fn fn_i(f: i32, ai: vec2<f32>) -> f32 {
    let lens = mat.fx_m.x > 0.5;
    if (ai.x > 0.5) {
        // an AI car: the session's lamps, its own brake (lights::ai_fx_block)
        let b = select(0, 5, lens) + f / 4;
        return mix(camera.ai_fx[b][f % 4], camera.ai_fx[10 + b][f % 4], clamp(ai.x - 1.0, 0.0, 1.0));
    }
    let base = select(4, 9, lens);
    return camera.fx[base + f / 4][f % 4];
}

/// Emitted radiance (HDR, before tonemap). Lamps are driven by function
/// masks; the colour of each function is the material's and its intensity is
/// this frame's lighting state (lights.rs). Car-space position stands in for
/// the game's vertex colour: +z front, +x left.
fn car_emissive(uv: vec2<f32>, obj: vec3<f32>, ai: vec2<f32>) -> vec3<f32> {
    let mode = mat.gl_em.z;
    // an AI car's discs, exhaust, LEDs and screens are not the player's
    if (mode < 0.5 || (ai.x > 0.5 && mode > 2.5 && mode < 6.5)) {
        return vec3<f32>(0.0);
    }
    let e = textureSample(t_emis, samp, uv);
    if (mode > 6.5) {
        // a glowing race number (numbers.rs): always lit
        return e.rgb * mat.emis_f[0].rgb;
    }
    if (mode < 1.5) {
        // interior backlight: on with the ignition
        return e.rgb * mat.emis_f[0].rgb * camera.fx[1].z;
    }
    if (mode < 2.5) {
        // ⚠ The masks are stored B,G,R: the shader's .x (daylight / tail,
        // and FunctionMask2's indicator) is our .b, its .z (high / fog,
        // special) our .r. Hence indicator = m2.b, special = m2.r.
        let m2 = textureSample(t_mask2, samp, uv);
        let left = obj.x >= 0.0;
        var c: vec3<f32>;
        if (obj.z >= 0.0) {
            c = e.r * mat.emis_f[0].rgb * fn_i(2, ai) + e.g * mat.emis_f[1].rgb * fn_i(1, ai)
              + e.b * mat.emis_f[2].rgb * fn_i(0, ai);
            c += m2.b * mat.emis_x[1].rgb * select(fn_i(8, ai), fn_i(7, ai), left);
            c += m2.r * select(mat.emis_x[4].rgb * fn_i(12, ai), mat.emis_x[3].rgb * fn_i(11, ai), left);
        } else {
            c = e.r * mat.emis_r[0].rgb * fn_i(5, ai) + e.g * mat.emis_r[1].rgb * fn_i(4, ai)
              + e.b * mat.emis_r[2].rgb * fn_i(3, ai);
            c += m2.g * mat.emis_x[0].rgb * fn_i(6, ai);
            c += m2.b * mat.emis_x[2].rgb * select(fn_i(10, ai), fn_i(9, ai), left);
            c += m2.r * select(mat.emis_x[6].rgb * fn_i(14, ai), mat.emis_x[5].rgb * fn_i(13, ai), left);
        }
        return c * camera.fx[2].w;
    }
    if (mode < 3.5) {
        // brake disc: glow mask * steel_glow(temperature)
        let g = select(camera.fx[3].rgb, camera.fx[2].rgb, obj.z >= 0.0);
        return e.r * g * mat.emis_f[0].rgb;
    }
    if (mode < 4.5) {
        // hot exhaust (mechanics glow batches)
        return e.rgb * mat.emis_f[0].rgb * camera.fx[1].x;
    }
    if (mode < 5.5) {
        // LED: LedMask (t_mask2) picks Led_k below 0.03 + 0.04 (k - 1);
        // the base emissive map (t_emis) shapes it
        let m = textureSample(t_mask2, samp, uv).r;
        var i = 0;
        if (m >= 0.03) {
            i = i32(floor((m - 0.03) / 0.04)) + 1;
        }
        if (i > 25) {
            return vec3<f32>(0.0);
        }
        let v = camera.fx[14 + i / 4][i % 4];
        return e.rgb * mat.emis_f[0].rgb * v * camera.fx[21 + i].rgb * camera.fx[3].w;
    }
    // display: the dash page (dash.rs / fxgpu.rs)
    return e.rgb * camera.fx[63].x * camera.fx[1].w;
}

/// Lamp light thrown into the scene (car space = world in the studio):
/// spot cones from the actor's physical lights, the game's
/// "falloff exponent" attenuation (SPEC 4.2 b) and cone term.
fn lamp_spots(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var c = vec3<f32>(0.0);
    let count = i32(camera.fx[63].y);
    for (var i = 0; i < count; i = i + 1) {
        let a = camera.fx[47 + i * 3];
        let b = camera.fx[48 + i * 3];
        let k = camera.fx[49 + i * 3];
        let dv = a.xyz - p;
        let d = max(length(dv), 0.05);
        let l = dv / d;
        let atten = smoothstep(a.w, a.w * 0.9, d) / pow(d, 1.2);
        let cone = clamp((dot(-l, b.xyz) - b.w) / max(k.w - b.w, 1e-3), 0.0, 1.0);
        c += k.rgb * atten * cone * cone * max(dot(n, l), 0.0);
    }
    return c;
}

/// The game's dirt layers (body shader 0x035a3710): Dirt_Map R/G/B weight
/// dust / grime / rubber, each pulling base colour, roughness and clear coat
/// toward the CarShadingGlobalSetting values.
struct Dirty {
    albedo: vec3<f32>,
    rough: f32,
    coat: f32,
};

fn apply_dirt(uv: vec2<f32>, albedo: vec3<f32>, rough: f32, coat: f32) -> Dirty {
    var o: Dirty;
    o.albedo = albedo;
    o.rough = rough;
    o.coat = coat;
    let amount = camera.fx[0].w;
    if (mat.fx_m.y < 0.5 || amount <= 0.0) {
        return o;
    }
    let dm = clamp(textureSample(t_dirt, samp, uv).rgb * amount, vec3<f32>(0.0), vec3<f32>(1.0));
    let dust = mix(camera.fx[59].rgb, camera.fx[60].rgb, camera.fx[60].w);
    o.albedo = mix(o.albedo, dust, dm.x);
    o.rough = mix(o.rough, camera.fx[59].w, dm.x);
    o.albedo = mix(o.albedo, camera.fx[61].rgb, dm.y);
    o.rough = mix(o.rough, camera.fx[61].w, dm.y);
    o.albedo = mix(o.albedo, camera.fx[62].rgb, dm.z);
    o.rough = mix(o.rough, camera.fx[62].w, dm.z);
    o.coat = o.coat * (1.0 - dm.x) * (1.0 - dm.y) * (1.0 - dm.z);
    return o;
}

/// How much of its drops the glass at `obj` keeps after the wipers: 0.2
/// just swept, back up to the unwiped 2x over ~1.2 s (the game keeps 0.2 /
/// 0.4 / 0.6 / 1.0 over its last swept areas and 2.0 where no wiper went).
fn rain_wiped(obj: vec3<f32>) -> f32 {
    var f = 2.0;
    let n = i32(camera.rain[0].w) % 10;
    for (var w = 0; w < 2; w = w + 1) {
        if (w >= n) {
            break;
        }
        let r = 1 + w * 8;
        let pivot = camera.rain[r].xyz;
        let reach = camera.rain[r].w;
        let ax = camera.rain[r + 1].xyz;
        if (camera.rain[r + 1].w < 0.5) {
            continue;
        }
        var d = obj - pivot;
        d = d - ax * dot(d, ax);
        let len = length(d);
        if (len > reach * 1.03 || len < 0.03) {
            continue;
        }
        var rest = camera.rain[r + 2].xyz;
        rest = normalize(rest - ax * dot(rest, ax));
        let a = atan2(dot(cross(rest, d), ax), dot(rest, d));
        // the blade's width as an angle at this radius
        let pad = 0.025 / len;
        for (var k = 0; k < 19; k = k + 1) {
            let a0 = camera.rain[r + 3 + k / 4][k % 4];
            let a1 = camera.rain[r + 3 + (k + 1) / 4][(k + 1) % 4];
            if (a >= min(a0, a1) - pad && a <= max(a0, a1) + pad) {
                f = min(f, 0.2 + 1.8 * f32(k) / 19.0);
                break;
            }
        }
    }
    return f;
}

fn rhash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

/// The moving water (the game's dynamic drops, a GPU simulation over its
/// baked windshield, approximated): drops sliding along `flow` (uv per
/// second) in columns, each with the trail it leaves. x coverage, yz slope.
fn rain_slide(uv: vec2<f32>, flow: vec2<f32>, t: f32, seed: f32) -> vec3<f32> {
    let sp = length(flow);
    if (sp < 1e-5) {
        return vec3<f32>(0.0);
    }
    let f = flow / sp;
    let side = vec2<f32>(-f.y, f.x);
    // cells: ~6 cm across the flow, ~16 cm along it
    let cells = vec2<f32>(22.0, 8.0);
    let q = vec2<f32>(dot(uv, side), dot(uv, f)) * cells;
    let col = floor(q.x) + seed * 17.0;
    let rc = rhash(vec2<f32>(col, seed));
    // stick-slip: each column's drop moves in bursts
    let ph = t * sp * cells.y * (0.55 + rc * 0.9);
    let y = q.y - (ph + 0.35 * sin(ph * 1.7 + rc * 6.28));
    let cell = floor(y);
    let r1 = rhash(vec2<f32>(col, cell));
    if (r1 > 0.55) {
        return vec3<f32>(0.0);
    }
    let fy = fract(y);
    let fx = fract(q.x) - 0.5 - (rhash(vec2<f32>(cell, col)) - 0.5) * 0.5;
    // to physical-ish units (uv), a 8-15 mm drop at the cell's head
    let d = vec2<f32>(fx / cells.x, (fy - 0.82) / cells.y);
    let rad = 0.006 + 0.005 * r1;
    let cov = smoothstep(rad, rad * 0.55, length(d));
    // the trail upstream of it: a thin wet line, beaded
    let w = rad * 0.4 * smoothstep(0.0, 0.82, fy);
    let bead = 0.6 + 0.4 * sin(fy * 60.0 + col);
    let trail = smoothstep(w, w * 0.25, abs(d.x)) * step(fy, 0.8) * smoothstep(0.0, 0.5, fy) * 0.55 * bead;
    let c = max(cov, trail);
    // the slope: a lens over the drop, a ridge across the trail
    let sl = select(vec2<f32>(d.x / w * 0.5, 0.0) * trail, d / rad * cov, cov > trail);
    return vec3<f32>(c, sl * vec2<f32>(dot(side, vec2<f32>(1.0, 0.0)), dot(side, vec2<f32>(0.0, 1.0))));
}

/// The windshield's water (the game's VehicleGlassInterior rain): its
/// static drops plus the moving water, both cleared by the wipers.
/// x coverage 0..1, yz the water's tangent-space slope.
/// The side windows' rain streaks (the game's VehicleGlass Rain_RS*): three
/// flow maps hold, per texel, when the water front reaches it (r); a run of
/// RSRange behind the moving front shows, weighted by speed - running down
/// the glass at rest, diagonal by ~60 km/h, flat back past ~120. x coverage,
/// yz slope (the game bends only v, by -ScaleRefraction x the normal's y).
fn rain_streaks(uv: vec2<f32>) -> vec3<f32> {
    if (mat.layer_pbr[0].z <= 0.0 || camera.rain[0].x <= 0.0) {
        return vec3<f32>(0.0);
    }
    let q = uv * mat.layer_pbr[0].z;
    let start = camera.rain[19].w;
    var cov = 0.0;
    var sl = 0.0;
    for (var i = 0; i < 3; i = i + 1) {
        let len = mat.layer_pbr[0].w * camera.rain[19][i] * camera.rain[0].x;
        if (len <= 0.0) {
            continue;
        }
        let s = textureSample(t_streaks, samp, q, i);
        let end = start + len;
        let on = (s.r > start && s.r < end) || s.r < end - 1.0;
        let a = select(0.0, camera.rain[0].x * s.g, on);
        cov = cov + a;
        sl = mix(sl, -(s.b * 2.0 - 1.0) * mat.layer_pbr[1].z * 5.0, a);
    }
    return vec3<f32>(clamp(cov, 0.0, 1.0), 0.0, sl);
}

fn rain_drops_at(uv: vec2<f32>, obj: vec3<f32>, flow: vec2<f32>) -> vec3<f32> {
    if (camera.rain[0].x <= 0.0 || mat.layer_pbr[3].z <= 0.0) {
        return vec3<f32>(0.0);
    }
    // a side window: its drops and streaks, no wiper, no sliding water
    if (mat.layer_pbr[0].z > 0.0) {
        let q = uv * mat.layer_pbr[3].z;
        let t = camera.rain[0].y;
        let m = textureSample(t_rain_mask, samp, q).r;
        let life = clamp(m + select(0.0, 1.0, m < t) - t, 0.0, 1.0);
        let a = life * textureSample(t_rain_alpha, samp, q).r;
        let nm = textureSample(t_rain_nm, samp, q).xy * 2.0 - 1.0;
        let c = clamp(a * camera.rain[0].z, 0.0, 1.0);
        let st = rain_streaks(uv);
        // side glass drops are small and flat: a lighter bend than the windshield's
        return vec3<f32>(max(c, st.x), nm * c * 0.4 + st.yz);
    }
    let q = uv * mat.layer_pbr[3].z;
    let t = camera.rain[0].y;
    let m = textureSample(t_rain_mask, samp, q).r;
    let life = clamp(m + select(0.0, 1.0, m < t) - t, 0.0, 1.0);
    let a = life * textureSample(t_rain_alpha, samp, q).r;
    let nm = textureSample(t_rain_nm, samp, q).xy * 2.0 - 1.0;
    let wiped = rain_wiped(obj);
    let c = clamp(a * camera.rain[0].z * wiped, 0.0, 1.0);
    // moving water: two layers at different scales; wiped glass carries less
    let tm = camera.rain[17].w;
    let s1 = rain_slide(uv, flow, tm, 0.0);
    let s2 = rain_slide(uv * 1.7 + vec2<f32>(0.37, 0.11), flow * 1.3, tm, 1.0);
    let moving = clamp(wiped, 0.35, 1.0) * (0.4 + 0.6 * camera.rain[0].x);
    let mc = max(s1.x, s2.x) * moving;
    let ms = (s1.yz + s2.yz) * moving;
    let cov = max(c, mc);
    return vec3<f32>(cov, nm * c + ms);
}

/// How the water runs over this glass, in uv per second: gravity down the
/// pane at a crawl, the airflow back over it once the car is moving.
fn rain_flow(in: VsOut) -> vec2<f32> {
    let n = normalize(in.normal);
    if (dot(in.tangent.xyz, in.tangent.xyz) < 1e-8) {
        return vec2<f32>(0.0, 0.05);
    }
    let t = normalize(in.tangent.xyz - n * dot(n, in.tangent.xyz));
    let b = cross(n, t) * select(-1.0, 1.0, in.tangent.w >= 0.0);
    let g = vec3<f32>(0.0, -1.0, 0.0);
    let fwd = camera.rain[18].xyz;
    let kmh = camera.rain[18].w;
    let air = -fwd * smoothstep(35.0, 110.0, kmh) * 3.0;
    var dir = g + air;
    dir = dir - n * dot(dir, n);
    // ~6 cm/s creeping down, up to ~1 m/s blown back at speed
    let speed = mix(0.06, 0.9, smoothstep(35.0, 160.0, kmh));
    let f = vec2<f32>(dot(dir, t), dot(dir, b));
    let l = length(f);
    if (l < 1e-4) {
        return vec2<f32>(0.0);
    }
    // uv per metre: the windshield's uv spans its ~1.4 m width
    return f / l * speed * 0.7;
}

fn rain_drops(in: VsOut) -> vec3<f32> {
    return rain_drops_at(in.uv, in.obj, rain_flow(in));
}

/// What the water shows of the scene behind it: the copy before the glass,
/// pushed by the water's slope (exposed already - added after tonemap()).
fn rain_refract(in: VsOut, drops: vec3<f32>) -> vec3<f32> {
    if (camera.rain[17].z < 0.5 || drops.x <= 0.001) {
        return vec3<f32>(0.0);
    }
    let off = drops.yz * vec2<f32>(12.0, -12.0);
    let uv = (in.clip.xy + off) * camera.rain[17].xy;
    let s = textureSampleLevel(t_scene, env_samp, uv, 0.0).rgb;
    // water darkens and blurs a little of what it bends
    let s2 = textureSampleLevel(t_scene, env_samp, uv + vec2<f32>(3.0, 2.0) * camera.rain[17].xy, 0.0).rgb;
    return mix(s, s2, 0.5) * 0.82;
}

/// EVO glass coverage: the material's opacity head-on, going opaque toward
/// grazing angles at a rate set by EdgeThickness (SPEC_evo_shading 4.1).
fn glass_coverage(n_dot_v: f32, map_a: f32) -> f32 {
    return glass_coverage_side(n_dot_v, map_a, false);
}

/// `inside`: seen from the cabin (the glass's outward normal faces away).
/// A windscreen from the driver's seat is seen at a grazing angle, and the
/// full edge ramp turned it near-opaque - tinted dark and milky.
fn glass_coverage_side(n_dot_v: f32, map_a: f32, inside: bool) -> f32 {
    let base_a = clamp(mat.gl_em.x * map_a, 0.0, 1.0);
    let edge = clamp((1.0 - clamp(n_dot_v + 0.01, 0.0, 1.0)) * mat.gl_em.y, 0.0, 1.0);
    return mix(base_a, 1.0, edge * select(1.0, 0.25, inside));
}

fn glass_inside(in: VsOut) -> bool {
    return dot(normalize(in.normal), camera.eye.xyz - in.world) < 0.0;
}

/// Glass pass 1 of 2: multiply what is behind by the transmittance,
/// lerp(baseColour, 1, 1 - a). Blend state is dst * src. Pass 2 (fs_main) then
/// ADDS the reflection. This is the game's dual-source blend split in two
/// draws, so it needs no device feature.
// SSR (ssr.rs): the glass darkens what is behind it, its reflection change too
@fragment
fn fs_glass_tint_ssr(in: VsOut) -> SsrOut {
    let t = glass_tint(in);
    return SsrOut(t, vec4<f32>(t.rgb, t.r), vec4<f32>(0.0));
}

@fragment
fn fs_glass_tint(in: VsOut) -> @location(0) vec4<f32> {
    return glass_tint(in);
}

fn glass_tint(in: VsOut) -> vec4<f32> {
    var tint = mat.base_color.rgb;
    var map_a = 1.0;
    if (mat.factors.z > 0.5) {
        let s = textureSample(t_albedo, samp, in.uv);
        tint = s.rgb * mat.base_color.rgb;
        map_a = s.a;
    }
    let v = normalize(camera.eye.xyz - in.world);
    let n_dot_v = abs(dot(normalize(in.normal), v));
    let a = glass_coverage_side(n_dot_v, map_a, glass_inside(in));
    // ⚠ Car glass is blendMode 1: dst = rgb + dst * (1 - a). The background
    // is darkened, NOT tinted - the tinted transmission is mode 3, which no
    // car material uses (SPEC_small §1-§3).
    _ = tint;
    let drops = rain_drops(in);
    // refracting, the water hides what is straight behind it (the add pass
    // puts back what it bends in); without the copy it only scatters
    let hide = select(0.5, 1.0, camera.rain[17].z > 0.5);
    return vec4<f32>(vec3<f32>((1.0 - a) * (1.0 - hide * drops.x)), 1.0);
}

// ===========================================================================
// Paint detail layers.
//
// EVO's UberVehicleMaterial gives every splat layer its own DETAIL maps,
// tiled at <Layer>_UVscale over the unwrap (SPEC_evo_shading 2.6):
//   Red_NormalMap           flakes_NM on paint, strength Red_NormalScale,
//                           which the .oemmultilayercolor sets (normal_intensity)
//   Green/Blue_NormalMap    carbon twill, or flakes again
//   Red_ClearCoatNormalMap  orange_peel_NM at Red_UVscaleClearCoatNormal
//   <Layer>_AmbientOcclusionMap  detail AO, strength <Layer>_AmbientOcclusion
// There is no flake BRDF in the game (SPEC 2.5): "metallic flake" is just
// the detail normal under the clear coat. The coat has its OWN normal
// (orange peel over the panel), so the flakes break up the base reflection
// while the coat stays a smooth, glassy mirror on top.
// ===========================================================================

struct PaintDetail {
    n: vec3<f32>,       // shading normal for the base coat (with flakes)
    n_cc: vec3<f32>,    // clear-coat normal (panel + orange peel, no flakes)
    ao: f32,            // detail-AO multiplier
};

/// Splat weights as EVO applies them: sequential lerps Base -> Red -> Green
/// -> Blue (SPEC 2.6), not a normalised sum. Returns the effective weight of
/// (Red, Green, Blue, Base); they always sum to 1. Magenta (red + blue)
/// therefore means "blue wins", which is what the game draws.
fn paint_splat_weights(w: vec3<f32>) -> vec4<f32> {
    let s = clamp(w, vec3<f32>(0.0), vec3<f32>(1.0));
    let b = s.z;
    let g = s.y * (1.0 - s.z);
    let r = s.x * (1.0 - s.y) * (1.0 - s.z);
    return vec4<f32>(r, g, b, (1.0 - s.x) * (1.0 - s.y) * (1.0 - s.z));
}

/// A splat layer's own UV, built the way UberVehicleMaterial builds it
/// (DXIL 0x035a3710): uv * <L>_UVscale when either component is non-zero,
/// rotated by <L>_UVRotationAngle when <L>_RotateUV, plus <L>_UVoffset.
fn layer_uv(i: i32, uv: vec2<f32>) -> vec2<f32> {
    let x = mat.layer_xf[i];
    var t = uv;
    if (x.x != 0.0 || x.y != 0.0) {
        t = uv * x.xy;
    }
    let a = x.z;
    let r = vec2<f32>(cos(a) * t.x + sin(a) * t.y, cos(a) * t.y - sin(a) * t.x);
    return r + mat.layer_xo[i].xy;
}

/// Layer colour times the layer's own colour map, when it has one.
fn layer_col(i: i32, c: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let k = mat.layer_xf[i].w;
    if (k < 0.0) {
        return c;
    }
    return c * textureSample(t_layer_c, samp, layer_uv(i, uv), i32(k)).rgb;
}

/// Layer roughness times the layer's roughness map, when it has one: in the
/// layer UV, or uv * <L>_UVscalePbr with <L>_CustomPbrUV.
fn layer_rough(i: i32, r: f32, uv: vec2<f32>) -> f32 {
    let o = mat.layer_xo[i];
    if (o.z < 0.0) {
        return r;
    }
    var t = layer_uv(i, uv);
    if (o.w > 0.5) {
        t = uv * mat.layer_pbr[i].xy;
    }
    return clamp(textureSample(t_rough_r, samp, t, i32(o.z)).r * r, 0.045, 1.0);
}

/// A layer's base colour. Negative = <Layer>OverrideReplace_BaseColor off:
/// the layer multiplies the colour beneath it instead of replacing it.
fn paint_layer_col(c: vec3<f32>, under: vec3<f32>) -> vec3<f32> {
    if (c.x < 0.0 || c.y < 0.0 || c.z < 0.0) {
        return -c * under;
    }
    return c;
}

/// Unpack a [0,1] tangent-space normal and scale its slope by `scale`.
fn paint_unpack_normal(t: vec3<f32>, scale: f32) -> vec3<f32> {
    let n = t * 2.0 - 1.0;
    return normalize(vec3<f32>(n.xy * scale, max(n.z, 1e-3)));
}

/// Reoriented Normal Mapping: Barre-Brisebois and Hill, "Blending in Detail"
/// (2012). Rotates the detail normal into the base normal's frame instead of
/// adding slopes, so flakes stay correct on a strongly curved base map.
fn paint_rnm(base: vec3<f32>, detail: vec3<f32>) -> vec3<f32> {
    let t = base + vec3<f32>(0.0, 0.0, 1.0);
    let u = detail * vec3<f32>(-1.0, -1.0, 1.0);
    return normalize(t * dot(t, u) / t.z - u);
}

/// Detail normals + detail AO for a splat material. `tbn` is the tangent frame
/// and `tn_base` the base normal-map sample in tangent space; `ew` the
/// effective splat weights from paint_splat_weights().
fn paint_detail(uv: vec2<f32>, tbn: mat3x3<f32>, tn_base: vec3<f32>,
                ew: vec4<f32>) -> PaintDetail {
    var out: PaintDetail;
    let flat_n = vec3<f32>(0.0, 0.0, 1.0);
    let ns = mat.layer_nscale;
    // Weighted mix of the layer detail normals, flat where a layer has none.
    var d = flat_n * ew.w;
    if (ns.x > 0.0) {
        d += paint_unpack_normal(textureSample(t_layer_n_r, samp, uv * mat.layer_uv.x).xyz, ns.x) * ew.x;
    } else {
        d += flat_n * ew.x;
    }
    if (ns.y > 0.0 || ns.z > 0.0) {
        // Green and Blue share the slot (same texture); the dominant layer's
        // tile is used for the one sample.
        let tile = select(mat.layer_uv.z, mat.layer_uv.y, ew.y * ns.y >= ew.z * ns.z);
        let raw = textureSample(t_layer_n_gb, samp, uv * tile).xyz;
        d += paint_unpack_normal(raw, ns.y) * ew.y + paint_unpack_normal(raw, ns.z) * ew.z;
    } else {
        d += flat_n * (ew.y + ew.z);
    }
    out.n = normalize(tbn * paint_rnm(tn_base, normalize(d)));
    // Clear coat: the panel's own normal plus orange peel, never the flakes.
    var tn_cc = tn_base;
    if (ns.w > 0.0) {
        let peel = paint_unpack_normal(textureSample(t_cc_n, samp, uv * mat.layer_uv.w).xyz, ns.w);
        tn_cc = paint_rnm(tn_base, peel);
    }
    out.n_cc = normalize(tbn * tn_cc);
    // Detail AO, faded by layer weight and strength (0 = off).
    out.ao = 1.0;
    let aos = mat.layer_ccr.w;
    if (aos != 0.0) {
        let w = select(ew.y, ew.x, aos > 0.0);
        let tile = select(mat.layer_uv.y, mat.layer_uv.x, aos > 0.0);
        let a = textureSample(t_layer_ao, samp, uv * tile).r;
        out.ao = mix(1.0, a, abs(aos) * w);
    }
    return out;
}

/// Clear-coat roughness from the per-layer constants (Base has none on any
/// shipped car, so it reads 0). Clamped like every other roughness in EVO.
fn paint_cc_rough(ew: vec4<f32>) -> f32 {
    let r = mat.layer_ccr.x * ew.x + mat.layer_ccr.y * ew.y + mat.layer_ccr.z * ew.z;
    return clamp(r, 0.045, 1.0);
}

/// Clear-coat Fresnel (SPEC 2.4): F0 = 0.16 * 0.85^2 = 0.1156, not 0.04.
fn paint_coat_fresnel(n_cc: vec3<f32>, v: vec3<f32>, coat: f32) -> f32 {
    let cc_f0 = 0.1156006;
    let nv = max(dot(n_cc, v), 1e-4);
    return (cc_f0 + (1.0 - cc_f0) * pow(1.0 - nv, 5.0)) * coat;
}

/// The clear coat as the game applies it, on car_ibl's (diffuse, specular)
/// pair. It exists ONLY in the environment term - no sun, softbox or
/// punctual coat lobe in the car body shader (SPEC 2.4):
///   spec = ccRefl + sqrt(1 - Fcc) * spec_base,  diffuse = (1 - Fcc) * diffuse
/// The coat is sampled along its own normal (panel + orange peel) at its own
/// roughness, so flakes in the base never break up the coat's reflection.
fn paint_coat_ibl(ibl: array<vec3<f32>, 2>, n_cc: vec3<f32>, v: vec3<f32>,
                  coat: f32, rough: f32) -> array<vec3<f32>, 2> {
    let fcc = paint_coat_fresnel(n_cc, v, coat);
    let cc_refl = env_rough(reflect(-v, n_cc), rough) * fcc;
    return array<vec3<f32>, 2>(ibl[0] * (1.0 - fcc),
                               cc_refl + ibl[1] * sqrt(1.0 - fcc));
}

/// Fallback for paths that do not go through car_ibl (outdoor, or no game
/// cube): same coat Fresnel and normal, applied to the finished colour.
fn paint_clear_coat(color: vec3<f32>, n_cc: vec3<f32>, v: vec3<f32>,
                    coat: f32, rough: f32) -> vec3<f32> {
    let fcc = paint_coat_fresnel(n_cc, v, coat);
    return color * (1.0 - fcc) + env_rough(reflect(-v, n_cc), rough) * fcc;
}

/// The car's own "hero" shadow in the studio (render.rs car_light_matrix):
/// the game lights its showroom with a sun forced to 83 deg elevation and
/// gives the player car a dedicated high-resolution map. Returns 1 when the
/// map is off, so the outdoor and unlit paths are untouched.
///
/// Normal offset rather than a big depth bias: at ~1.5 mm per texel a few
/// texels along the normal kills acne on the curved body without detaching
/// the shadow from the wheel arch it starts in.
fn hero_shadow(world: vec3<f32>, n: vec3<f32>) -> f32 {
    if (camera.shadow.x < 1.5) {
        return 1.0;
    }
    let p = world + n * camera.shadow.z * 2.0;
    return sun_shadow(p, abs(dot(n, CAR_SUN)));
}

/// Rim motion blur (rim shader 0x03693a40): v = saturate(Velocity^gamma);
/// the blur rim covers a v share of its pixels, the static rim 1 - v, by a
/// 4x4 Bayer dither. Velocity is this frame's (fx[1].y), not the material's.
fn rim_visible(frag: vec2<f32>) -> bool {
    if (mat.fx_m.z < 0.5) {
        return true;
    }
    let v = clamp(pow(camera.fx[1].y, mat.fx_m.w), 0.0, 1.0);
    let cover = select(v, 1.0 - v, mat.fx_m.z > 1.5);
    let ix = u32(frag.x) & 3u;
    let iy = u32(frag.y) & 3u;
    var bayer = array<f32, 16>(0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0,
                               3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
    return cover > (bayer[iy * 4u + ix] + 0.5) / 16.0;
}

/// The game's cutout (a2c prepass 0x03350b70, SPEC_small 7b / D14): keep a
/// pixel where sat(opacityScale (alpha - cutoff)) beats a 4x4 Bayer shifted
/// each frame (clouds[5].w, render.rs), which TAA turns into partial
/// coverage. flags.w is the 50 % point, misc.y the scale (material.rs);
/// misc.y 1 = the old hard threshold. True = drop the pixel.
fn cutout_drop(p: vec2<f32>, alpha: f32) -> bool {
    if (mat.misc.y < 1.25) {
        return alpha < mat.flags.w;
    }
    let cov = clamp(mat.misc.y * (alpha - mat.flags.w) + 0.5, 0.0, 1.0);
    // BayerMatrix4 (k + 1) / 17, and the per-frame shift in ordered-dither
    // order, as track.wgsl's
    var bayer = array<f32, 16>(1.0, 13.0, 4.0, 16.0, 9.0, 5.0, 12.0, 8.0,
                               3.0, 15.0, 2.0, 14.0, 11.0, 7.0, 10.0, 6.0);
    var shifts = array<vec2<u32>, 16>(
        vec2<u32>(0u, 0u), vec2<u32>(2u, 2u), vec2<u32>(2u, 0u), vec2<u32>(0u, 2u),
        vec2<u32>(1u, 1u), vec2<u32>(3u, 3u), vec2<u32>(3u, 1u), vec2<u32>(1u, 3u),
        vec2<u32>(1u, 0u), vec2<u32>(3u, 2u), vec2<u32>(3u, 0u), vec2<u32>(1u, 2u),
        vec2<u32>(0u, 1u), vec2<u32>(2u, 3u), vec2<u32>(2u, 1u), vec2<u32>(0u, 3u));
    let q = (vec2<u32>(p) + shifts[u32(camera.clouds[5].w) & 15u]) & vec2<u32>(3u);
    return !(cov > bayer[q.x + q.y * 4u] / 17.0);
}

// Screen-space reflections (ssr.rs): the environment reflection this pixel
// added (output units) + the weight it had, and the octahedral normal,
// roughness and kind (2/3: the driven car, which moves), for the SSR pass
var<private> g_env: vec4<f32>;
var<private> g_geo: vec4<f32>;

struct SsrOut {
    @location(0) color: vec4<f32>,
    @location(1) env: vec4<f32>,
    @location(2) geo: vec4<f32>,
}

fn ssr_oct(n: vec3<f32>) -> vec2<f32> {
    let a = n / (abs(n.x) + abs(n.y) + abs(n.z));
    var e = a.xy;
    if (a.z < 0.0) {
        e = (vec2<f32>(1.0) - abs(a.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), a.xy >= vec2<f32>(0.0));
    }
    return e * 0.5 + 0.5;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return shade(in);
}

// SSR, alpha-blended: covers what is behind by its alpha (ssr.rs cover)
@fragment
fn fs_main_cover(in: VsOut) -> SsrOut {
    let c = shade(in);
    return SsrOut(c, vec4<f32>(c.a), vec4<f32>(0.0));
}

@fragment
fn fs_main_ssr(in: VsOut) -> SsrOut {
    let c = shade(in);
    return SsrOut(c, vec4<f32>(tonemap(g_env.rgb), g_env.a), g_geo);
}

fn shade(in: VsOut) -> vec4<f32> {
    if (!rim_visible(in.clip.xy)) {
        discard;
    }
    let v = normalize(camera.eye.xyz - in.world);

    // ---- ground plane ----------------------------------------------------
    if (mat.flags.w > 0.5) {
        let n = vec3<f32>(0.0, 1.0, 0.0);
        let shade = contact_shadow(in.world);
        // Subtle radial falloff so the floor fades out rather than ending at a
        // hard edge, plus a wide soft reflection of the sky.
        let dist = length(in.world.xz - camera.ground.xz);
        let fade = 1.0 - smoothstep(camera.extent.w * 0.12, camera.extent.w * 0.75, dist);
        // The floor is a dark seamless sweep. Its brightness has to come from
        // the grazing reflection, not from ambient, or the whole frame lifts to
        // mid-grey and the car loses its contrast against it.
        if (th_on()) {
            // The sphere's floor: its own dark albedo under the same light as
            // the car, so the backdrop/car contrast comes out of the lighting.
            // The sphere floor reads as one flat dark tone that melts into the
            // background (studio_clear() is TH_BACKDROP through the same LUT),
            // darkened only by the car's contact shadow.
            // / grade.x: the backdrop is fitted AFTER exposure, so re-fitting
            // the car's exposure does not move it
            let floor_c = TH_BACKDROP * pow(shade, TH_SHADOW_POW) / max(camera.grade.x, 1e-4) * camera.fx[0].x
                        + lamp_spots(in.world, n) * TH_WALL_ALBEDO * 4.0;
            return vec4<f32>(tonemap(floor_c), 1.0);
        }
        let base = mat.base_color.rgb * 1.6;
        let fres = fresnel(max(dot(n, v), 1e-4), vec3<f32>(0.03)).x;
        let refl = env_rough(reflect(-v, n), 0.30) * fres * 0.20;
        let hero = mix(1.0, hero_shadow(in.world, n), HERO_FLOOR);
        var col = (base * 0.5 + refl) * shade * hero;
        col = mix(vec3<f32>(0.012, 0.013, 0.015), col, fade) * camera.fx[0].x;
        col += lamp_spots(in.world, n) * base * 0.5;
        return vec4<f32>(tonemap(col), 1.0);
    }

    // ---- contact-shadow decal (contact.rs) --------------------------------
    if (mat.misc.w > 3.5) {
        return contact_decal(in.uv);
    }

    // ---- car -------------------------------------------------------------
    // The studio rig, in car-relative units. Defined once here because the
    // paint, the clear coat and the glass all have to mirror the SAME lights -
    // if they disagree the highlights slide apart and nothing looks solid.
    let glass_r = max(max(camera.extent.x, camera.extent.z), 0.6);
    let glass_c = vec3<f32>(camera.ground.x,
                            camera.ground.w + glass_r * 0.55,
                            camera.ground.z);

    var n = normalize(in.normal);
    // Tangent frame + base normal sample, kept for the paint detail layers.
    var tbn_ok = false;
    var tbn = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), n);
    var tn_base = vec3<f32>(0.0, 0.0, 1.0);

    var albedo = mat.base_color.rgb;
    var alpha = mat.flags.y;
    if (mat.factors.z > 0.5) {
        let s = sample2(t_albedo, samp, in.uv, in.world, n);
        albedo = s.rgb * mat.base_color.rgb;
        alpha = alpha * s.a;
    }
    // race-number decal (numbers.rs): soft edges, the depth fade in |w|
    if (mat.layer_pbr[3].w > 0.5) {
        alpha = alpha * abs(in.tangent.w);
    }
    if (mat.misc.w > 0.5 && mat.misc.w < 1.5) {
        // the opacity map shares the base colour's UV (Base_UVscale)
        alpha = alpha * sample2(t_splat, samp, in.uv, in.world, n).r;
    } else if (mat.misc.w > 1.5 && mat.misc.w < 2.5) {
        // World-space grass tile. Used when the albedo is a baked unwrap
        // (any track's *far / mapN / outerland), never on authored grass UVs.
        let d = textureSample(t_splat, samp, in.world.xz * max(mat.uv_scale.w, 0.04)).rgb;
        let baked = albedo;
        albedo = mix(d, d * baked, 0.22);
    } else if (mat.misc.w > 2.5 && mat.uv_scale.w > 0.0) {
        let d = textureSample(t_splat, samp, in.uv * mat.uv_scale.w).rgb;
        albedo = albedo * mix(vec3<f32>(1.0), d, 0.45);
    }

    // Normal mapping from screen-space derivatives - works without a tangent
    // attribute, so every mesh benefits regardless of what the exporter had.
    if (mat.factors.w > 0.5) {
        // Prefer the mesh's own tangent (xyz + handedness in w). Screen-space
        // derivatives are only a fallback: where the UV derivative is nearly
        // degenerate - tyre tread blocks, headlight slats, grille mesh - the
        // derived tangent explodes and the normal map reads as pixel noise.
        var t: vec3<f32>;
        var sign_w = 1.0;
        var have = false;
        if (dot(in.tangent.xyz, in.tangent.xyz) > 1e-8) {
            t = in.tangent.xyz;
            sign_w = select(-1.0, 1.0, in.tangent.w >= 0.0);
            have = true;
        } else {
            let dp1 = dpdx(in.world);
            let dp2 = dpdy(in.world);
            let duv1 = dpdx(in.uv);
            let duv2 = dpdy(in.uv);
            let det = duv1.x * duv2.y - duv2.x * duv1.y;
            if (abs(det) > 1e-8) {
                t = (dp1 * duv2.y - dp2 * duv1.y) / det;
                have = dot(t, t) > 1e-12;
                // the w the mesh data would carry: w * cross(N, T) = +dP/dv
                let bv = (dp2 * duv1.x - dp1 * duv2.x) / det;
                sign_w = select(-1.0, 1.0, dot(cross(n, t), bv) >= 0.0);
            }
        }
        // Gram-Schmidt: the interpolated tangent is rarely still
        // perpendicular to the interpolated normal (and one along the normal
        // leaves nothing: normalize(0) is NaN)
        let tp = t - n * dot(n, t);
        if (have && dot(tp, tp) > 1e-10) {
            t = normalize(tp);
            // the game's frame (SPEC_uber 2.1): B = cross(T, N) * w
            let b = normalize(cross(t, n)) * sign_w * BT_GAME;
            let tn = sample2(t_normal, samp, in.uv, in.world, n).xyz * 2.0 - 1.0;
            tbn = mat3x3<f32>(t, b, n);
            tbn_ok = true;
            tn_base = normalize(tn);
            n = normalize(tbn * tn);
        }
    }

    // Alpha-tested materials (decals, stitching, defroster lines) carry their
    // shape in the albedo alpha; without the discard they render as opaque
    // rectangles sitting on the bodywork.
    if (mat.misc.y > 0.5 && cutout_drop(in.clip.xy, alpha)) {
        discard;
    }

    var ao = 1.0;
    var rough = clamp(mat.factors.x, 0.045, 1.0);
    if (mat.flags.x > 1.5) {
        // t_ao is bound as a roughness map (track DefaultMaterial)
        rough = clamp(sample2(t_ao, samp, in.uv, in.world, n).r, 0.045, 1.0);
    } else if (mat.flags.x > 0.5) {
        ao = sample2(t_ao, samp, in.uv, in.world, n).r;
    }
    // A car ships a real roughness map; it used to fight the AO map for one
    // slot. Now it has its own, so both survive.
    if (mat.maps.x > 0.5) {
        rough = clamp(sample2(t_rough, samp, in.uv, in.world, n).r, 0.045, 1.0);
    }
    // Global AO layered on top of the per-material AO, strength-scaled as a
    // plain lerp from 1: g = (map - 1) * strength + 1, so strength 0 is a no-op.
    if (mat.maps.z > 0.0) {
        let g = (textureSample(t_gao, samp, in.uv).r - 1.0) * mat.maps.z + 1.0;
        ao = ao * clamp(g, 0.0, 1.0);
    }
    var metallic = clamp(mat.factors.y, 0.0, 1.0);
    var clearcoat = clamp(mat.flags.z, 0.0, 1.0);
    var n_cc = n;
    var cc_rough = 0.045;
    var coat_done = false;
    // Uber vehicle shaders (SPEC_uber §4): AO map at Base_AmbientOcclusion
    // strength, the material's own clear coat, and a coat normal that is the
    // SMOOTH geometric normal - the base normal map never reaches the coat.
    // uber[0].w: 1 = the Uber decode, 2 = only the game's F0 (glass shaders)
    let uber = mat.uber[0].w > 0.5 && mat.uber[0].w < 1.5;
    let game_f0 = mat.uber[0].w > 0.5;
    var refl = clamp(mat.misc.z, 0.0, 1.0);
    if (game_f0) {
        refl = mat.uber[0].x;
    }
    if (uber) {
        if (mat.flags.x > 0.5 && mat.flags.x < 1.5) {
            ao = 1.0 + (ao - 1.0) * mat.uber[1].z;
        }
        clearcoat = mat.uber[0].y;
        if (mat.maps.w > 0.5) {
            clearcoat = textureSample(t_coat, samp, in.uv).r;
        }
        cc_rough = mat.uber[0].z;
        n_cc = normalize(in.normal);
    }

    // ---- splat layers ----------------------------------------------------
    // A body material is not one surface. The splat map assigns every panel to
    // a layer - red is the paint, green a secondary colour, blue the exposed
    // carbon - and each layer carries its own colour, roughness, metalness and
    // clear coat. Render only the base and an F1 car comes out black, because
    // the base is just the fallback for wherever the splat map is empty.
    // A disabled layer is flagged with w < 0 and folded back into the base.
    if (mat.layer_metal.w > 0.5) {
        var sm = textureSample(t_splat, samp, in.uv).rgb;
        // Splatmap_Tighten: weights under the threshold drop to 0 (SPEC 2.6)
        let tight = mat.layer_pbr[3].x;
        if (tight >= 0.0) {
            sm = select(sm, vec3<f32>(0.0), sm < vec3<f32>(tight));
        }
        let ew = paint_splat_weights(vec3<f32>(
            select(0.0, sm.r, mat.layer_r.w >= 0.0),
            select(0.0, sm.g, mat.layer_g.w >= 0.0),
            select(0.0, sm.b, mat.layer_b.w >= 0.0)));
        let wr = ew.x;
        let wg = ew.y;
        let wb = ew.z;
        let wbase = ew.w;
        // Detail normals need the tangent frame; detail AO does not.
        let want_n = tbn_ok && dot(mat.layer_nscale, mat.layer_nscale) > 0.0;
        if (want_n || mat.layer_ccr.w != 0.0) {
            let pd = paint_detail(in.uv, tbn, tn_base, ew);
            if (want_n) {
                n = pd.n;
                n_cc = pd.n_cc;
            }
            ao = ao * pd.ao;
        }
        if (uber) {
            // ⚠ SPEC_uber §5: a layer changes roughness / metalness / clear
            // coat / reflectance ONLY when its override flag for that
            // attribute is set. Converted carbon (a Blue layer with every
            // value at 1 and no flags) is the plain base in the game; drawn
            // with the values it came out fully metallic and white.
            let sw = clamp(vec3<f32>(
                select(0.0, sm.r, mat.layer_r.w >= 0.0),
                select(0.0, sm.g, mat.layer_g.w >= 0.0),
                select(0.0, sm.b, mat.layer_b.w >= 0.0)), vec3<f32>(0.0), vec3<f32>(1.0));
            albedo = mix(albedo, paint_layer_col(layer_col(0, mat.layer_r.rgb, in.uv), albedo), sw.x);
            albedo = mix(albedo, paint_layer_col(layer_col(1, mat.layer_g.rgb, in.uv), albedo), sw.y);
            albedo = mix(albedo, paint_layer_col(layer_col(2, mat.layer_b.rgb, in.uv), albedo), sw.z);
            for (var i = 0; i < 3; i++) {
                let w = sw[i];
                if (w <= 0.0) { continue; }
                let md = mat.uber[2 + i];
                let lv = mat.uber[5 + i];
                let lx = mat.uber[8 + i];
                if (md.x > 0.5) {
                    var r = lv.w;
                    let o = mat.layer_xo[i];
                    if (o.z >= 0.0) {
                        var t = layer_uv(i, in.uv);
                        if (o.w > 0.5) { t = in.uv * mat.layer_pbr[i].xy; }
                        let tm = textureSample(t_rough_r, samp, t, i32(o.z)).r;
                        r = select(tm, tm * r, lx.z > 0.5);
                    }
                    let v2 = select(select(r * rough, r + rough, md.x > 2.5), r, md.x < 1.5);
                    rough = mix(rough, v2, w);
                }
                if (md.y > 0.5) {
                    metallic = mix(metallic, select(lv.x * metallic, lv.x, md.y < 1.5), w);
                }
                if (md.z > 0.5) {
                    clearcoat = mix(clearcoat, select(lv.y * clearcoat, lv.y, md.z < 1.5), w);
                }
                if (md.w > 0.5) {
                    cc_rough = mix(cc_rough, select(lv.z * cc_rough, lv.z, md.w < 1.5), w);
                }
                if (lx.x > 0.5) {
                    refl = mix(refl, select(lx.y * refl, lx.y, lx.x < 1.5), w);
                }
            }
            rough = clamp(rough, 0.045, 1.0);
            metallic = clamp(metallic, 0.0, 1.0);
            clearcoat = clamp(clearcoat, 0.0, 1.0);
            // the coat normal: orange peel only (a clear-coat normal map)
            if (tbn_ok && mat.layer_nscale.w > 0.0) {
                let peel = paint_unpack_normal(textureSample(t_cc_n, samp, in.uv * mat.layer_uv.w).xyz, mat.layer_nscale.w);
                n_cc = normalize(tbn * peel);
            } else {
                n_cc = normalize(in.normal);
            }
        } else {
        cc_rough = paint_cc_rough(ew);
        if (wr + wg + wb > 0.001) {
            // Sequential, so a multiply layer tints whatever is under it.
            let sw = clamp(vec3<f32>(
                select(0.0, sm.r, mat.layer_r.w >= 0.0),
                select(0.0, sm.g, mat.layer_g.w >= 0.0),
                select(0.0, sm.b, mat.layer_b.w >= 0.0)), vec3<f32>(0.0), vec3<f32>(1.0));
            // <L>_v = layer colour (sign = replace/multiply) * its own map,
            // sampled in the layer's own UV (layer_uv), for all three layers.
            albedo = mix(albedo, paint_layer_col(layer_col(0, mat.layer_r.rgb, in.uv), albedo), sw.x);
            albedo = mix(albedo, paint_layer_col(layer_col(1, mat.layer_g.rgb, in.uv), albedo), sw.y);
            albedo = mix(albedo, paint_layer_col(layer_col(2, mat.layer_b.rgb, in.uv), albedo), sw.z);
            // The red layer carries its own roughness MAP, not just a scalar -
            // that spatial variation is most of what makes paint and carbon
            // read as real rather than a uniform sheet.
            let rr = layer_rough(0, max(mat.layer_r.w, 0.0), in.uv);
            let rg = layer_rough(1, max(mat.layer_g.w, 0.0), in.uv);
            let rb = layer_rough(2, max(mat.layer_b.w, 0.0), in.uv);
            rough = clamp(rough * wbase + rr * wr + rg * wg + rb * wb, 0.045, 1.0);
            metallic = clamp(metallic * wbase
                        + mat.layer_metal.x * wr
                        + mat.layer_metal.y * wg
                        + mat.layer_metal.z * wb, 0.0, 1.0);
            clearcoat = clamp(clearcoat * wbase
                        + mat.layer_coat.x * wr
                        + mat.layer_coat.y * wg
                        + mat.layer_coat.z * wb, 0.0, 1.0);
        }
        if (mat.maps.w > 0.5) {
            clearcoat = clearcoat * textureSample(t_coat, samp, in.uv).r;
        }
        }
    }
    cc_rough = max(cc_rough, 0.045);
    // ---- dirt and wetness (effects; off at their defaults) ----
    {
        let dd = apply_dirt(in.uv, albedo, rough, clearcoat);
        albedo = dd.albedo;
        rough = dd.rough;
        clearcoat = dd.coat;
        let wet = camera.fx[0].z;
        // drops beading on the bodywork (the game's Rain_RaindropsEnabled
        // paint): the windshield's drop maps at Rain_ScaleUv, alpha x 2 x
        // life, on faces turned up (saturate(normal.y)), bending the coat
        if (camera.rain[0].x > 0.0 && mat.layer_pbr[2].z > 0.0) {
            let q = in.uv * mat.layer_pbr[2].zw;
            let t = camera.rain[0].y;
            let m = textureSample(t_rain_mask, samp, q).r;
            let life = clamp(m + select(0.0, 1.0, m < t) - t, 0.0, 1.0);
            let k = clamp(textureSample(t_rain_alpha, samp, q).r * 2.0 * life, 0.0, 1.0) * 0.7
                * clamp(in.normal.y, 0.0, 1.0) * min(camera.rain[0].x * 2.0, 1.0);
            let nm = textureSample(t_rain_nm, samp, q).xyz * 2.0 - 1.0;
            if (k > 0.001 && dot(in.tangent.xyz, in.tangent.xyz) > 1e-8) {
                let nn = normalize(in.normal);
                let tt = normalize(in.tangent.xyz - nn * dot(nn, in.tangent.xyz));
                let bb = cross(tt, nn) * select(-1.0, 1.0, in.tangent.w >= 0.0) * BT_GAME;
                // a bead is a shallow lens on the coat: tilt it, never turn
                // it away from the sky (a full swap to the drop normal
                // reflected the ground and punched black holes in liveries)
                let dn = normalize(n_cc + (tt * nm.x + bb * nm.y) * 0.45 * k);
                n_cc = select(dn, n_cc, dn.y < 0.0 && n_cc.y >= 0.0);
                clearcoat = max(clearcoat, 0.8 * k);
            }
        }
        if (wet > 0.0 && mat.layer_coat.w < 0.5 && mat.misc.x < 0.5 && mat.layer_pbr[3].y < 0.5) {
            // a water film: rough, porous surfaces darken; everything
            // gets a smooth top layer
            albedo = albedo * mix(1.0, 0.72, wet * rough);
            rough = mix(rough, 0.06, wet * 0.85);
            clearcoat = max(clearcoat, wet * 0.6);
        }
    }

    // GSAA, UberVehicleMaterial's runtime path (SPEC_uber 7), non-metals:
    // the geometric normal's screen variance, fading out by 25 m; the
    // frameData multipliers are not decoded (GSAA_CAR, EVOVIEW_GSAA_CAR)
    let ng_var = sq_deriv(normalize(in.normal));
    var ks = 1.0;
    if (metallic < 0.5) {
        let fd = 1.0 - clamp(length(in.world - camera.eye.xyz) * 0.04, 0.0, 1.0);
        let gv = fd * GSAA_CAR.x * ng_var;
        rough = min(gsaa_rough(rough, gv, fd * GSAA_CAR.y), rough * GSAA_CAR.z);
        cc_rough = min(gsaa_rough(cc_rough, gv, fd * GSAA_CAR.y), cc_rough * GSAA_CAR.z);
        ks = spec_aa(ng_var);
    }
    let ks_ibl = clamp(ks + 0.75, 0.0, 1.0);

    // Track meshes mix winding. Light the side facing the camera or
    // every other triangle is a black hole.
    if (camera.mode.x > 0.5 && dot(n, v) < 0.0) {
        n = -n;
        n_cc = -n_cc;
    }
    let n_dot_v = max(dot(n, v), 1e-4);
    // the driven car on a track is kind 2/3 (it moves), anything else static
    g_geo = vec4<f32>(ssr_oct(n), rough, select(1.0 / 3.0, 2.0 / 3.0, camera.clouds[5].y > 0.5));
    let f0 = select(mix(vec3<f32>(0.04), albedo, metallic),
                    mix(vec3<f32>(0.16 * refl * refl), albedo, metallic), game_f0);

    let r = reflect(-v, n);
    var color: vec3<f32>;
    if (camera.mode.x > 0.5) {
        let dist = length(in.world - camera.eye.xyz);
        let fog_near = max(camera.extent.w * 0.45, 800.0);
        let fog = smoothstep(fog_near, fog_near * 3.5, dist);
        let sun = key_dir();
        let sun_k = key_rgb();
        let sky_k = camera.sky[1].w;
        if (camera.mode.y < 0.5) {
            // Unlit: authored albedo, no sun. Toggle with L.
            color = albedo;
        } else if (camera.clouds[5].y > 0.5) {
            // ---- a car on the track ----
            // The material model the studio branch uses (Burley + GGX sun,
            // split-sum sky reflections with multiscatter, multi-bounce AO,
            // and the clear coat applied after this branch), lit by what is
            // really there: the time-of-day sun through the shadow maps and
            // the cloud deck, and the sky (env_sky / env_rough read the sky
            // block). The track branch below was tuned for tarmac and grass -
            // its flat fill and forced shadow-side darkening made paint dull
            // and pushed red toward pink.
            let ndl = max(dot(n, sun), 0.0);
            let shade = car_sun_shadow(in.world, n, ndl) * cloud_shadow(in.world);
            let sun_rough = clamp(rough + 0.0023, 0.02, 1.0);
            let h = normalize(v + sun);
            let l_dot_h = max(dot(sun, h), 0.0);
            let n_dot_h = max(dot(n, h), 0.0);
            let f = fresnel(l_dot_h, f0);
            let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
            let spec = (ggx(n_dot_h, sun_rough) * smith(n_dot_v, ndl, sun_rough) * f)
                     / max(4.0 * n_dot_v * ndl, 1e-4);
            color = (kd * albedo * burley(ndl, n_dot_v, l_dot_h, rough) + spec * ks)
                  * vec3<f32>(4.2, 3.85, 3.35) * sun_k * ndl * shade;
            // sky: diffuse irradiance from the blurred sky around the normal,
            // reflections from the sky along r (both carry sky_k already)
            let mb = multi_bounce(ao, albedo);
            let kd_amb = (vec3<f32>(1.0) - fresnel(n_dot_v, f0)) * (1.0 - metallic);
            // x1.1 (was 0.55): the track's shade fill went up 3.3x to the
            // game's sun / shade ratio (track.rs ambient 9), the exposure came
            // down with it, and cars left at the old fill read dull and dark
            color += kd_amb * albedo * env_rough(n, 0.95) * 1.1 * mb;
            let ab = env_brdf_ab(rough, n_dot_v);
            // shadowed by the car's own body and the scenery, with a floor
            // so the sky still reaches a shaded flank
            let so = mix(0.35, 1.0, shade);
            let sw = (f0 * ab.x + vec3<f32>(ab.y)) * ao * ms_compensate(f0, ab) * mix(0.6, 1.0, so) * ks_ibl;
            color += env_rough(r, rough) * sw;
            g_env = vec4<f32>(env_rough(r, rough) * sw, dot(sw, vec3<f32>(0.2126, 0.7152, 0.0722)));
        } else {
            // Outdoor sun. The previous path had a 0.48 hemisphere that
            // filled every face equally, so hills and buildings read as
            // a flat texture with no light. Drop the fill, keep a hard
            // sun and a cool sky bounce so slopes actually shade.
            //
            // What the showroom branch learned from EVO's own renderer is a
            // MATERIAL model, and a material model does not care what is
            // lighting it - Burley diffuse, multi-bounce AO, split-sum ambient
            // specular and multiscatter compensation were only ever in that
            // branch because that is where they were written. The light RIG
            // does not transfer: a circuit is lit by the sun and the sky, not
            // by four softboxes, so the lighting here is unchanged.
            let ndl = max(dot(n, sun), 0.0);
            let wrap = clamp(dot(n, sun) * 0.5 + 0.5, 0.0, 1.0);
            let sun_vis = mix(wrap * wrap, ndl, 0.65);
            let sky_vis = clamp(n.y * 0.55 + 0.45, 0.12, 1.0);
            let bounce = max(-n.y, 0.0);

            // The sun is a half-degree DISC, not a point. Widening the lobe by
            // its angular radius is what stops polished metal and wet asphalt
            // showing a one-pixel aliasing sparkle where they should show a
            // small round highlight - the same reason the studio uses
            // rectangles rather than points, at the one size the sky gives us.
            // Cast shadows. This is the single biggest thing missing outdoors:
            // without it every tree, grandstand and barrier sits in its own
            // ambient light and the scene reads as a painted model.
            // the shadow map's, and the cloud deck's (clouds_common.wgsl)
            let shade = car_sun_shadow(in.world, n, ndl) * cloud_shadow(in.world);
            let sun_rough = clamp(rough + 0.0023, 0.02, 1.0);
            let h = normalize(v + sun);
            let l_dot_h = max(dot(sun, h), 0.0);
            let n_dot_h = max(dot(n, h), 0.0);
            let f = fresnel(l_dot_h, f0);
            let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
            let spec = (ggx(n_dot_h, sun_rough) * smith(n_dot_v, ndl, sun_rough) * f)
                     / max(4.0 * n_dot_v * ndl, 1e-4);
            // Burley in place of Lambert: retroreflection at grazing angles is
            // most of why asphalt and dry grass read as a surface rather than
            // as a flat photo.
            // Every term is scaled by the time of day: `sun_k` the key light
            // (1 = the calibrated noon sun, ~0.003 moonlight), `sky_k` the
            // sky. At noon both are ~1 and this is the calibrated look.
            var direct = (kd * albedo * burley(ndl, n_dot_v, l_dot_h, rough) + spec * ks)
                  * vec3<f32>(4.2, 3.85, 3.35) * ndl * shade;

            // Multi-bounce AO. `ao * albedo` treats an occluded pixel as if
            // light never bounces off the albedo at all, so creases under
            // grandstands and inside tree canopies went muddy grey.
            let mb = multi_bounce(ao, albedo);
            let amb = albedo * camera.sky[2].rgb * vec3<f32>(0.58, 0.54, 0.52) * sky_vis * 0.22 * mb
                    * sky_k;
            direct += albedo * vec3<f32>(0.18, 0.15, 0.10) * bounce * 0.12 * mb;
            // Extra lambert so even missing-normal meshes pick up a side.
            direct += albedo * vec3<f32>(1.15, 1.05, 0.88) * sun_vis * 0.55 * mb
                   * mix(0.35, 1.0, shade);
            // Split-sum ambient specular with multiscatter compensation,
            // instead of a flat scalar - the same two terms the showroom uses.
            // (env_rough reads env_sky, which already carries sky_k.)
            let ab = env_brdf_ab(rough, n_dot_v);
            let amb_spec = env_rough(r, rough) * (f0 * ab.x + vec3<f32>(ab.y))
                   * ao * ms_compensate(f0, ab) * 0.10 * ks_ibl;
            color = direct * sun_k + amb + amb_spec;
            // Shadow side: faces away from the sun go down, not grey.
            color *= mix(0.38, 1.0, clamp(sun_vis * 1.15, 0.0, 1.0));
            // Shadowed ground keeps the sky term but loses the warm bounce.
            color *= mix(0.62, 1.0, shade);
        }
        color = mix(color, env_sky(normalize(in.world - camera.eye.xyz)), fog * 0.55);
    } else {
        // Four softboxes instead of two punctual lights. Size is the whole
        // point: the highlight down the shoulder is the shape of the light.
        color = vec3<f32>(0.0);
        // A cabin is a closed box. The softboxes are OUTSIDE it, so only what
        // gets through the glass should reach a dashboard - the rest arrives
        // from the cabin source below. Letting the full outside rig in as well
        // is double-lighting: invisible on near-black plastic, but it blows a
        // light-coloured leather or alcantara interior straight to white.
        let cabin = mat.layer_coat.w > 0.5;
        let thru = select(1.0, 0.20, cabin);
        // Hero shadow: the rig is occluded by the car itself - arches, sills,
        // the cabin under the roof. The game applies the sun shadow to IBL
        // too, but with a floor so shadowed areas keep some sky.
        let hs = hero_shadow(in.world, normalize(in.normal));
        let hs_direct = mix(1.0, hs, HERO_DIRECT);
        // grade.w scales these four softboxes, and the default thumbnail
        // studio sets it to 0 (the game lights thumbnails by IBL alone).
        // Uniform branch: skip four area lights that would add exactly zero.
        if (camera.grade.w > 0.0) {
            for (var i = 0; i < 4; i = i + 1) {
                color += area_light(studio_light(i, glass_r, glass_c),
                                    in.world, n, v, albedo, metallic, rough, f0) * thru * hs_direct;
            }
        }
        // The game applies the sun shadow to IBL as well, through a floor
        // (IblDiffuse/SpecularShadowFloor), so shadowed areas keep some sky.
        let hs_diff = mix(HERO_IBL_DIFF_FLOOR, 1.0, hs);
        let hs_spec = mix(HERO_IBL_SPEC_FLOOR, 1.0, hs);
        if (camera.mode.z > 0.5) {
            // The game's own cube: its IBL model, see car_ibl.
            var ibl = car_ibl(n, r, n_dot_v, albedo, metallic, rough, f0, ao);
            if (clearcoat > 0.0) {
                ibl = paint_coat_ibl(ibl, n_cc, v, clearcoat, cc_rough);
                coat_done = true;
            }
            color += ibl[0] * select(1.0, 0.35, cabin) * hs_diff
                   + ibl[1] * select(1.0, 0.30, cabin) * hs_spec;
            if (th_on()) {
                // the thumbnail sun IS the light the hero map is cast from
                color += th_sun(n, v, albedo, metallic, rough, f0) * select(1.0, 0.15, cabin) * hs;
            }
        } else {
            let spec_env = env_rough(r, rough);
            let diff_env = env_rough(n, 0.85);
            let kd = (vec3<f32>(1.0) - fresnel(n_dot_v, f0)) * (1.0 - metallic);
            let mb = multi_bounce(ao, albedo);
            let ab = env_brdf_ab(rough, n_dot_v);
            color += kd * albedo * diff_env * 0.45 * mb * select(1.0, 0.35, cabin) * hs_diff;
            color += spec_env * (f0 * ab.x + vec3<f32>(ab.y)) * ao * ms_compensate(f0, ab)
                   * select(1.0, 0.30, cabin) * hs_spec;
        }
    }

    // ---- cabin fill ------------------------------------------------------
    // A car interior is a closed box: the studio rig lights the outside, and
    // nothing gets in. Real showroom renders put a soft source in the cabin
    // rather than leaving it black. Applied only to `int_*` materials so it
    // cannot leak onto the bodywork and flatten the exterior shading.
    // ⚠ Cars only. The `int_*` name test is a CAR convention, and on a track it
    // also catches pit-garage interiors (`box_int_2`, `box_old_int_a` at Spa).
    // The cabin source is placed from the car's bounding box, so on a circuit
    // it is a phantom light in the middle of nowhere.
    if (mat.layer_coat.w > 0.5 && camera.mode.x < 0.5) {
        let cab = studio_light(4, glass_r, glass_c);
        // Diffuse AND specular: on black gloss the specular is the whole story.
        color += area_light(cab, in.world, n, v, albedo, metallic, rough, f0) * ao;
        // Black plastic reflects its surroundings; without an ambient specular
        // term a dashboard is a void no matter how bright the fill.
        let amb_ab = env_brdf_ab(max(rough, 0.08), n_dot_v);
        color += env_rough(r, max(rough, 0.08)) * (f0 * amb_ab.x + vec3<f32>(amb_ab.y))
               * ao * 1.1;
        // and a gentle wrap so deep footwells read as dark, not as holes
        let up_wrap = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
        color += albedo * vec3<f32>(0.34, 0.35, 0.39) * up_wrap * multi_bounce(ao, albedo);
    }

    // ---- clear coat ------------------------------------------------------
    // A second, much smoother lobe over the top. Its Fresnel is what makes the
    // paint go bright and glassy at grazing angles.
    if (clearcoat > 0.0 && !coat_done) {
        color = paint_clear_coat(color, n_cc, v, clearcoat, cc_rough);
        // SSR takes the coat's reflection when it outweighs the base's
        let fcc = paint_coat_fresnel(n_cc, v, clearcoat);
        g_env = g_env * (1.0 - fcc);
        if (fcc > g_env.a) {
            g_env = vec4<f32>(env_rough(reflect(-v, n_cc), cc_rough) * fcc, fcc);
            g_geo = vec4<f32>(ssr_oct(n_cc), cc_rough, g_geo.w);
        }
    }

    // ⚠ No manual gamma here. The render target is *UnormSrgb, so the hardware
    // encodes on write; doing it again in the shader lifts every dark value and
    // washes the whole frame out to pastel - which looks like "the lighting is
    // too bright" and sends you tuning the environment instead of the bug.
    // ---- emissive --------------------------------------------------------
    // ---- night + lamp light on the bodywork (effects) ----
    if (camera.mode.x < 0.5) {
        color = color * camera.fx[0].x;
        if (camera.fx[63].y > 0.5) {
            color += lamp_spots(in.world, n) * albedo * (1.0 - metallic) * 0.3183;
        }
    }
    // fx[63].w > 0: outdoors on a timed track, the night exposure taken back
    // out of the lamps (render.rs)
    let emis = car_emissive(in.uv, in.obj, in.ai) * select(1.0, camera.fx[63].w, camera.fx[63].w > 0.0);

    // ---- glass: pass 2 of 2 ----------------------------------------------
    // The glass surface's own lighting, premultiplied by its coverage, plus
    // what it emits; the background was already multiplied by the tint in
    // fs_glass_tint. Blend state is ONE + ONE.
    if (mat.misc.x > 0.5) {
        let inside = glass_inside(in);
        let a = glass_coverage_side(abs(dot(n, v)), alpha / max(mat.flags.y, 1e-3), inside);
        // from the cabin the glass mirrors the dark interior, not the sky
        let refl = select(1.0, 0.12, inside);
        // rain: each drop a small lens - the sky / track along its own
        // normal, a sun glint, at the drop's coverage
        let drops = rain_drops(in);
        let refr = rain_refract(in, drops) * drops.x * (1.0 - a);
        var wet = vec3<f32>(0.0);
        if (drops.x > 0.001) {
            let t = normalize(in.tangent.xyz);
            let bt = cross(n, t) * select(-1.0, 1.0, in.tangent.w >= 0.0);
            let dn = normalize(n + (t * drops.y + bt * drops.z) * 1.6);
            let rv = reflect(-v, dn);
            let f = 0.06 + 0.94 * pow(1.0 - abs(dot(dn, v)), 5.0);
            let glint = pow(max(dot(rv, key_dir()), 0.0), 300.0) * 4.0 * key_rgb();
            wet = (env_rough(rv, 0.15) * f * 0.6 + glint) * drops.x;
        }
        // debug (rain[0].w >= 20, EVOVIEW_RAIN_DEBUG=2): water coverage, |slope|
        if (camera.rain[0].w >= 19.5 && mat.layer_pbr[3].z > 0.0) {
            return vec4<f32>(drops.x, length(drops.yz) * 4.0, 0.0, 0.0);
        }
        // debug (rain[0].w >= 10): the wipers' recency as colour
        if (camera.rain[0].w >= 9.5 && mat.layer_pbr[3].z > 0.0) {
            let f = rain_wiped(in.obj);
            return vec4<f32>(tonemap(vec3<f32>(f * 0.5, 1.0 - f * 0.5, 0.0) * 0.3), 0.0);
        }
        return vec4<f32>(tonemap(color * a * mat.gl_em.w * refl + emis + wet) + refr, 0.0);
    }
    color += emis;

    // strict mode: diagonal warning stripes over whatever had to be guessed,
    // the guessed look still visible between them
    if (mat.strict.x > 0.5) {
        let band = fract((in.clip.x + in.clip.y) / 28.0);
        if (band < 0.5) {
            return vec4<f32>(mix(tonemap(color), vec3<f32>(1.0, 0.0, 0.85), 0.65), max(alpha, 0.85));
        }
    }
    return vec4<f32>(tonemap(color), alpha);
}


/// Bloom source (fxgpu.rs): this surface's emission only, exposed, HDR.
/// Drawn with the main pass's depth (LessEqual, no write), additively.
@fragment
fn fs_glow(in: VsOut) -> @location(0) vec4<f32> {
    if (mat.misc.y > 0.5 && mat.factors.z > 0.5) {
        let a = textureSample(t_albedo, samp, in.uv).a * mat.flags.y;
        if (cutout_drop(in.clip.xy, a)) {
            discard;
        }
    }
    let e = car_emissive(in.uv, in.obj, in.ai) * camera.grade.x * select(1.0, camera.fx[63].w, camera.fx[63].w > 0.0);
    if (max(e.r, max(e.g, e.b)) <= 1e-4) {
        discard;
    }
    return vec4<f32>(e, 1.0);
}
