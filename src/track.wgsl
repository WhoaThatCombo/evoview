// Track shading - the track's own pipeline, separate from the car path.
//
// What the game does for a circuit (from its DXIL, L:/evoview_work/shaderre):
//   * ambient comes from IRRADIANCE VOLUMES: 72 m columns of 16x3x16 probes,
//     each probe four SH-ish coefficients (R11G11B10), evaluated as
//       E = 0.886 c0 + 1.0233 (c1 max(-n.y,0) + c2 max(n.z,0) + c3 max(-n.x,0))
//     and divided by pi. That is the term that makes the ground under trees
//     and grandstands go dark and the open grass glow.
//   * terrain is a splat of Base + Red/Green/Blue layers, each with its own
//     tile scale (4 orders of magnitude apart), world-aligned triplanar for
//     some layers, weights from a splat texture, the vertex colour, or a
//     constant.
//   * the vertex colour also carries AO (channel 3, intensity x (a - offset)).
//   * distance haze is "omni haze": density ~0.002 /m by day.
//
// Group 0: frame (camera, sun, haze, irradiance atlas).
// Group 1: per-material constants + textures (track only).

struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    // xyz direction TO the sun, w illuminance
    sun: vec4<f32>,
    // rgb sun colour, w exposure
    sun_color: vec4<f32>,
    // x haze density /m, y haze height falloff m, z haze base height, w sky scale
    haze: vec4<f32>,
    // xy atlas origin (min x, min z), z 1/(atlas width m), w 1/(atlas depth m)
    iv_origin: vec4<f32>,
    // xyz probe layer heights above ground, w 1 = atlas present
    iv_elev: vec4<f32>,
    // x ambient scale, y lit (0 = unlit albedo), z grass fade start m, w grass fade end m
    misc: vec4<f32>,
    // the base's sun shadow map (see render.rs light_matrix)
    light_vp: mat4x4<f32>,
    // x on, y depth bias, z metres per texel, w normal offset (m)
    shadow: vec4<f32>,
    // the lead's grade: x exposure, y 1 = use the LUT, z which LUT
    grade: vec4<f32>,
    // pigment map: x min x, y min z, z 1/size x, w 1/size z (0 = none)
    pigment: vec4<f32>,
    // x debug view: 0 off, 1 albedo, 2 irradiance, 3 AO, 4 normal, 5 sun shadow
    // y 1 = cutouts use alpha-to-coverage (MSAA), z 1 = honour camera facing
    debug: vec4<f32>,
    // time of day (sky.rs Sky::pack): 0 key dir + strength, 1 key colour +
    // sky multiplier, 2 zenith colour + night, 3 horizon colour + night IV
    sky: array<vec4<f32>, 4>,
    // lamp grid: x min x, y min z, z 1/cell, w cells across
    lamp_grid: vec4<f32>,
    // x cells deep, y lamp count, z lamp strength, w night irradiance scale
    lamps: vec4<f32>,
    // far shadow cascade: world -> light clip; x on, y bias, z m/texel, w normal offset m
    light_vp_far: mat4x4<f32>,
    shadow_far: vec4<f32>,
    // x haze start distance, metres (the track's haze_start_distance)
    haze2: vec4<f32>,
    // clouds: x base above the ground, y thickness, z coverage, w density
    clouds: vec4<f32>,
    // x noise scale, y erosion detail scale, z erosion width, w hardness
    clouds2: vec4<f32>,
    // x time s, yz wind m/s, w 1 = noise present
    clouds3: vec4<f32>,
    // x cirrus amount, y cirrus height, z 1 = cirrus present, w cloud
    // shadow strength on the ground
    clouds4: vec4<f32>,
    // weather map: x present, y length m, z width m, w angle rad
    clouds5: vec4<f32>,
    // cloud shadow map: x min x, y min z, z span m, w 1 = drawn
    cmap: vec4<f32>,
    // x ground wetness, y falling rain, z time (rain.rs)
    wet: vec4<f32>,
    // middle cascade (layer 2): x on, y bias, z m/texel, w normal offset m
    light_vp_mid: mat4x4<f32>,
    shadow_mid: vec4<f32>,
    // the driven car's hero map (layer 3): x uv scale into its corner
    // (0 = off), y bias, z m/texel, w normal offset m
    light_vp_hero: mat4x4<f32>,
    shadow_hero: vec4<f32>,
    // x texture LOD bias: TAA's (taa.rs mip_bias) for the main view, 0 for
    // the mirrors, the probe and the rear camera, which have no TAA
    lod: vec4<f32>,
    // physical sky (track.rs FrameUniform.atm): xyz true sun direction, w on
    atm: vec4<f32>,
    // x cloud panorama history blend, y radiance scale, z panorama march
    // steps, w falling rain
    atm2: vec4<f32>,
    // cloud light: x sun gain, y ambient gain, z extinction scale toward
    // the sun, w powder strength
    atm3: vec4<f32>,
    // x the deck's extinction as seen over its shadow's
    atm4: vec4<f32>,
};

struct Lamp {
    // xyz position, w radius
    pos: vec4<f32>,
    // rgb colour x intensity, w falloff exponent (0 = inverse square)
    col: vec4<f32>,
    // xyz spot direction, w cos(outer half-angle)
    dir: vec4<f32>,
    // x cos(inner), y 1 = spot, z 1 = on by day too
    ext: vec4<f32>,
};

struct TrackMat {
    // rgb base tint, w kind (0 uber, 1 road, 2 road multilayer, 3 foliage, 4 glass, 5 sky-ish)
    base: vec4<f32>,
    // x roughness, y reflectance, z normal scale, w alpha cutoff (0 = opaque)
    surf: vec4<f32>,
    // xy base uv scale, z base uv channel, w has base albedo
    base_uv: vec4<f32>,
    // per layer: rgb tint, w enabled
    red: vec4<f32>,
    green: vec4<f32>,
    blue: vec4<f32>,
    // per layer: xy uv scale, z triplanar world scale (0 = mesh uv), w uv channel
    red_uv: vec4<f32>,
    green_uv: vec4<f32>,
    blue_uv: vec4<f32>,
    // per layer roughness (xyz), w modulate-by-base flags bitmask (1 r, 2 g, 4 b)
    layer_rough: vec4<f32>,
    // splat: x mode (0 none, 1 texture, 2 vertex, 3 constant), y uv channel,
    // z has layer normal maps bitmask, w blend contrast
    splat: vec4<f32>,
    // constant weights (Splatmap_Constant) rgb, w vertex colour scale
    splat_const: vec4<f32>,
    // x detail enable, y detail multiplier, z detail uv scale, w detail strength
    detail: vec4<f32>,
    // x vertex AO on, y AO intensity, z AO offset, w AO channel
    vao: vec4<f32>,
    // rgb emissive colour * intensity, w has emissive map
    emissive: vec4<f32>,
    // x has normal map, y has rough map, z has opacity map, w grass (distance fade)
    flags: vec4<f32>,
    // x pigment tint, y light wrap, z specular scale
    extra: vec4<f32>,
    // vegetation wind (track.rs TrackMatUniform.wind)
    wind0: vec4<f32>,
    wind1: vec4<f32>,
    wind2: vec4<f32>,
    // material AO map: x strength, y mode (track.rs TrackMatUniform.ao)
    ao: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var t_iv: texture_3d<f32>;
@group(0) @binding(2) var s_clamp: sampler;
// layer 0 near map, 1 far cascade, 2 middle cascade, 3 the driven car's
// hero map (render.rs shadow_target_tex)
@group(0) @binding(3) var t_shadow: texture_depth_2d_array;
@group(0) @binding(4) var s_shadow: sampler_comparison;
@group(0) @binding(5) var t_lut: texture_3d<f32>;
@group(0) @binding(6) var t_pigment: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> lamps: array<Lamp>;
@group(0) @binding(8) var<storage, read> lamp_cells: array<vec2<u32>>;
@group(0) @binding(9) var<storage, read> lamp_index: array<u32>;
// the game's cloud noise volume, 512x512x64, as 64 array slices
@group(0) @binding(10) var t_cloud: texture_2d_array<f32>;
@group(0) @binding(11) var s_cloud: sampler;
// content/weather/clouds/cirrus_layer: R streaky cirrus, G cirrocumulus, B veil
@group(0) @binding(12) var t_cirrus: texture_2d<f32>;
// content/weather/weather_map6: R cloud cover, B rain (clouds.rs)
@group(0) @binding(13) var t_weather: texture_2d<f32>;
// animated props' bone sets (track.rs TrackAnim); instance facing.y = slot + 1
@group(0) @binding(14) var<storage, read> track_bones: array<mat4x4<f32>>;
// the clouds' shadow on the ground around the eye (fs_cloud_map)
@group(0) @binding(15) var t_cmap: texture_2d<f32>;
// the sky panorama (fs_sky_pano, read copy): clear sky radiance, and the
// cloud deck as (in-scattered rgb, transmittance)
@group(0) @binding(17) var t_pano_sky: texture_2d<f32>;
@group(0) @binding(18) var t_pano_cloud: texture_2d<f32>;
// the dry line cars leave in the wet (rain.rs dry_map): per 1 m cell, how
// much drier than the rest of the track, wrapped at 256 cells
@group(0) @binding(16) var t_dry: texture_2d<f32>;

@group(1) @binding(0) var<uniform> mat: TrackMat;
@group(1) @binding(1) var s_wrap: sampler;
@group(1) @binding(2) var t_base: texture_2d<f32>;
@group(1) @binding(3) var t_base_n: texture_2d<f32>;
@group(1) @binding(4) var t_base_r: texture_2d<f32>;
@group(1) @binding(5) var t_splat: texture_2d<f32>;
@group(1) @binding(6) var t_red: texture_2d<f32>;
@group(1) @binding(7) var t_red_n: texture_2d<f32>;
@group(1) @binding(8) var t_green: texture_2d<f32>;
@group(1) @binding(9) var t_green_n: texture_2d<f32>;
@group(1) @binding(10) var t_blue: texture_2d<f32>;
@group(1) @binding(11) var t_blue_n: texture_2d<f32>;
@group(1) @binding(12) var t_detail: texture_2d<f32>;
@group(1) @binding(13) var t_opacity: texture_2d<f32>;
@group(1) @binding(14) var t_ao: texture_2d<f32>;

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
    @location(10) facing: vec4<f32>,
    // raw vertex colour (the engine stores colour / 255) and the second UV set
    @location(11) color: vec4<f32>,
    @location(12) uv1: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
    // x = distance of the instance origin from the eye (grass fade)
    @location(6) inst: vec4<f32>,
};

fn wind_ch(c: vec4<f32>, ch: f32, k: f32) -> f32 {
    let i = i32(ch + 0.5);
    if (i == 1) { return c.g * k; }
    if (i == 2) { return c.b * k; }
    if (i == 3) { return c.a * k; }
    if (i >= 4) { return k; }
    return c.r * k;
}

fn wind_offset(in: VsIn, pos: vec3<f32>, t: f32) -> vec3<f32> {
    // the engine stores colour / 255; the masks are 0 / 1 bytes
    let col = clamp(in.color * 255.0, vec4<f32>(0.0), vec4<f32>(1.0));
    let weight = wind_ch(col, mat.wind0.x, mat.wind0.y);
    let phase = wind_ch(col, mat.wind0.z, mat.wind0.w);
    let flut = wind_ch(col, mat.wind1.x, mat.wind1.y);
    // height inside the mesh's bounds, optionally remapped
    var hh = clamp((in.position.y - in.facing.z) / max(in.facing.w - in.facing.z, 0.001), 0.0, 1.0);
    if (mat.wind2.w > mat.wind2.z) {
        hh = clamp((hh - mat.wind2.z) / (mat.wind2.w - mat.wind2.z), 0.0, 1.0);
    }
    let hk = mat.wind2.x * (hh - 1.0) + 1.0;
    let w_main = hk * weight;
    let w_flut = hk * flut;
    let base = in.inst3.xyz;
    // global wind (the weather's): direction from the cloud wind, strength
    // scaled so the materials' intensities (80-300 for trees) give tens of cm
    var wv = frame.clouds3.yz;
    if (length(wv) < 0.5) {
        wv = vec2<f32>(2.5, 1.0);
    }
    let wdir = normalize(vec3<f32>(wv.x, 0.0, wv.y));
    // ⚠ windStrength is not decoded (a weather input). 0.0025 swung a tree
    // crown ~1.5 m: with the formula's per-vertex phase (stiffness) its two
    // sides moved apart and the canopy read as stretching, not swaying.
    let wind_strength = 0.0008 * clamp(length(wv) / 4.0, 0.5, 2.0);
    let gust_strength = 0.5;
    let gust_freq = 0.35;
    let dir_var = 0.35;
    let pi2 = 6.283185;
    let ph0 = phase * pi2;
    var sp = dot(base.xz, vec2<f32>(0.379, 0.523));
    sp = sp - round(sp / pi2) * pi2;
    let d = pos - base;
    let stiff = (1.0 - mat.wind1.z) * (d.y * 0.13 + dot(d.xz, vec2<f32>(0.31, 0.27)));
    // the wind's direction wanders
    let a = dir_var * (sin(sp * 0.11 + t * 0.149) * 0.6 + sin(t * 0.317 + 1.9) * 0.4);
    let sa = sin(a);
    let ca = cos(a);
    // gusts travel along the wind
    let g0 = dot(base.xz, wdir.xz) * 0.045;
    let gs = gust_freq * t - (g0 - round(g0 / pi2) * pi2);
    let gust = clamp((sin(gs) * 0.5 + sin(gs * 2.317 + 1.3) * 0.3 + sin(gs * 0.531 + 4.1) * 0.2) * 0.5 + 0.5, 0.0, 1.0);
    let g3 = gust * gust * gust;
    let amp = mat.wind1.w * wind_strength * (gust_strength * (g3 * 2.6 - 0.6) + 1.0);
    let p = ((sp + (ph0 + mat.wind2.y * t)) + stiff) + gust_strength * g3 * 2.0;
    let main = w_main * (sin(p) * 0.55 + sin(p * 0.731 + 1.3) * 0.45 + 0.35);
    let side = w_main * sin(p * 1.279 + 2.1) * 0.6;
    let fl = w_flut * (sin(p * 4.271) * 0.6 + sin(p * 6.913 + 1.7) * 0.4) * 0.25;
    // rotated wind direction and its side vector, as the engine builds them
    let dx = wdir.z * sa + wdir.x * ca;
    let dy = wdir.y * ca;
    let dz = -wdir.x * sa + wdir.z * ca;
    let px = wdir.z * ca - wdir.x * sa;
    let py = -wdir.y * sa;
    let pz = -wdir.x * ca - wdir.z * sa;
    let ox = amp * (px * side + dx * main + px * 0.5 * fl);
    let oz = amp * (pz * side + dz * main + pz * 0.5 * fl);
    let oy = amp * (py * side + dy * main + (py * 0.5 + 0.8) * fl) - (ox * ox + oz * oz) * 0.5;
    return vec3<f32>(ox, oy, oz);
}

/// A camera-facing card's (MeshLodData f20) vertex turned to the camera:
/// world position and normal.
fn facing_vertex(in: VsIn, face: f32) -> array<vec3<f32>, 2> {
    // CameraFacingMode: keep the instance translation and scale, turn the
    // card toward the camera (yaw only for 1/3, yaw + pitch for 2/4).
    let origin = in.inst3.xyz;
    let sc = vec3<f32>(length(in.inst0.xyz), length(in.inst1.xyz), length(in.inst2.xyz));
    var lp = in.position * max(sc, vec3<f32>(0.0001));
    var ln = in.normal;
    var to = frame.eye.xyz - origin;
    if (frame.debug.z < 1.5) {
        // the view direction, from the clip -> world matrix
        let a = frame.inv_view_proj * vec4<f32>(0.0, 0.0, 0.5, 1.0);
        to = frame.eye.xyz - a.xyz / a.w;
    }
    if (face < 1.5 || (face > 2.5 && face < 3.5)) {
        let hz = vec2<f32>(to.x, to.z);
        if (dot(hz, hz) > 1e-8) {
            let f = normalize(hz);
            lp = vec3<f32>(f.y * lp.x + f.x * lp.z, lp.y, -f.x * lp.x + f.y * lp.z);
            ln = vec3<f32>(f.y * ln.x + f.x * ln.z, ln.y, -f.x * ln.x + f.y * ln.z);
        }
    } else {
        let f = normalize(to);
        var right = cross(vec3<f32>(0.0, 1.0, 0.0), f);
        right = select(normalize(right), vec3<f32>(1.0, 0.0, 0.0), dot(right, right) < 1e-8);
        let up = cross(f, right);
        lp = right * lp.x + up * lp.y + f * lp.z;
        ln = right * ln.x + up * ln.y + f * ln.z;
    }
    return array<vec3<f32>, 2>(lp + origin, ln);
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world = mat4x4<f32>(in.inst0, in.inst1, in.inst2, in.inst3);
    var pos: vec3<f32>;
    var nrm: vec3<f32>;
    var tan: vec3<f32>;
    var tw = in.tangent.w;
    // Camera-facing cards (MeshLodData f20 = 1 on every card tree: cf1,
    // cf2, bb_3 ...). Left in their authored orientation they read as two
    // crossed flat photos - "split images". Turned toward the eye POSITION
    // they visibly spin as a free camera passes them, so by default they
    // are aligned to the view DIRECTION instead: sideways moves leave them
    // still, only turning the camera turns them (strict TR50: the game's
    // own facing rule is not decoded). EVOVIEW_TRACK_FACING = 0 off,
    // 2 toward the eye.
    let face = select(0.0, in.facing.x, frame.debug.z > 0.5);
    if (face > 0.5) {
        let fv = facing_vertex(in, face);
        pos = fv[0];
        nrm = fv[1];
        tan = in.tangent.xyz;
    } else {
        var lp = in.position;
        var ln = in.normal;
        var lt = in.tangent.xyz;
        // an animated prop: linear blend skinning with its instance's bone set
        let total = dot(in.bone_wgt, vec4<f32>(1.0));
        if (in.facing.y > 0.5 && total > 0.0001) {
            // facing.y = slot + 1, so slot = floor(facing.y - 0.5)
            let base = u32(in.facing.y - 0.5);
            let nb = arrayLength(&track_bones);
            var p = vec3<f32>(0.0);
            var n = vec3<f32>(0.0);
            var t = vec3<f32>(0.0);
            for (var k = 0u; k < 4u; k = k + 1u) {
                let w = in.bone_wgt[k];
                if (w <= 0.0) { continue; }
                let bi = base + u32(max(in.bone_idx[k], 0.0));
                if (bi >= nb) { continue; }
                let m = track_bones[bi];
                p = p + (m * vec4<f32>(in.position, 1.0)).xyz * w;
                n = n + (m * vec4<f32>(in.normal, 0.0)).xyz * w;
                t = t + (m * vec4<f32>(in.tangent.xyz, 0.0)).xyz * w;
            }
            lp = p / total;
            ln = n / total;
            lt = t / total;
        }
        pos = (world * vec4<f32>(lp, 1.0)).xyz;
        let m3 = mat3x3<f32>(world[0].xyz, world[1].xyz, world[2].xyz);
        nrm = normal_to_world(m3, ln);
        tan = m3 * lt;
        tw = tangent_w_to_world(m3, tw);
    }
    // vegetation wind (MaterialData.windable, extra.w bit 3): the game's
    // MaterialVS wind (exe vertex shader 0x034a96e0, lifted). Per material:
    // weight / phase / flutter from a vertex-colour channel (0-3 r g b a,
    // 4 constant only) x constant, stiffness, intensity, frequency, height
    // falloff over the mesh's own height (instance row 18/19). Globals the
    // weather would set (strength, gusts, direction wander) are fixed here.
    if ((u32(mat.extra.w + 0.5) & 8u) != 0u) {
        pos = pos + wind_offset(in, pos, frame.clouds3.x);
    }
    out.world = pos;
    out.normal = nrm;
    out.uv = in.uv;
    out.tangent = vec4<f32>(tan, tw);
    out.color = in.color;
    out.uv1 = in.uv1;
    // y: the instance's LOD dither-out amount (cull.rs, matrix [0][3])
    // z: the placement's handedness (a mirrored instance turns its winding,
    // and with it front_facing, inside out)
    let hand = select(-1.0, 1.0, dot(cross(in.inst0.xyz, in.inst1.xyz), in.inst2.xyz) >= 0.0);
    out.inst = vec4<f32>(length(in.inst3.xyz - frame.eye.xyz), in.inst0.w, hand, 0.0);
    out.clip = frame.view_proj * vec4<f32>(pos, 1.0);
    return out;
}

// ---------------------------------------------------- motion vectors ----
// track.rs draw_motion: the animated props (TrackAnim bones) and the
// wind-swayed vegetation drawn again into motion.rs's target, from this and
// last frame's bone sets and wind clock. Everything else on the track is
// static: motion.wgsl's camera pass covers it.

struct MvU {
    // this frame's and last frame's unjittered world -> clip
    vp: mat4x4<f32>,
    prev_vp: mat4x4<f32>,
    // render w, h, 1/w, 1/h
    rsize: vec4<f32>,
    // x last frame's clock (the wind's frame.clouds3.x), y how far from the
    // eye swaying vegetation gets vectors (m)
    misc: vec4<f32>,
};
@group(2) @binding(0) var<uniform> mvu: MvU;
@group(2) @binding(1) var t_mv_depth: MV_DEPTH;
@group(2) @binding(2) var<storage, read> prev_track_bones: array<mat4x4<f32>>;

struct MvVs {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) prev_world: vec3<f32>,
};

// vs_main's skinning, position only, with this or last frame's bone sets
fn skin_pos(in: VsIn, prev: bool) -> vec3<f32> {
    let total = dot(in.bone_wgt, vec4<f32>(1.0));
    if (in.facing.y < 0.5 || total <= 0.0001) {
        return in.position;
    }
    let base = u32(in.facing.y - 0.5);
    let nb = arrayLength(&track_bones);
    var p = vec3<f32>(0.0);
    for (var k = 0u; k < 4u; k = k + 1u) {
        let w = in.bone_wgt[k];
        if (w <= 0.0) { continue; }
        let bi = base + u32(max(in.bone_idx[k], 0.0));
        if (bi >= nb) { continue; }
        var m = track_bones[bi];
        if (prev) {
            m = prev_track_bones[bi];
        }
        p = p + (m * vec4<f32>(in.position, 1.0)).xyz * w;
    }
    return p / total;
}

@vertex
fn vs_motion(in: VsIn) -> MvVs {
    var o: MvVs;
    // vegetation past the range sways well under a pixel a frame: dropped
    // before any of the work (a degenerate, clipped triangle)
    if (in.facing.y < 0.5 && length(in.inst3.xyz - frame.eye.xyz) > mvu.misc.y) {
        o.clip = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        return o;
    }
    let world = mat4x4<f32>(in.inst0, in.inst1, in.inst2, in.inst3);
    var pos: vec3<f32>;
    var prev: vec3<f32>;
    let face = select(0.0, in.facing.x, frame.debug.z > 0.5);
    if (face > 0.5) {
        // (turned with the camera: treated as static, as the camera pass does)
        pos = facing_vertex(in, face)[0];
        prev = pos;
    } else {
        pos = (world * vec4<f32>(skin_pos(in, false), 1.0)).xyz;
        prev = (world * vec4<f32>(skin_pos(in, true), 1.0)).xyz;
    }
    if ((u32(mat.extra.w + 0.5) & 8u) != 0u) {
        pos = pos + wind_offset(in, pos, frame.clouds3.x);
        prev = prev + wind_offset(in, prev, mvu.misc.x);
    }
    o.world = pos;
    o.prev_world = prev;
    o.clip = frame.view_proj * vec4<f32>(pos, 1.0);
    return o;
}

fn mv_px(m: mat4x4<f32>, w: vec3<f32>) -> vec3<f32> {
    let c = m * vec4<f32>(w, 1.0);
    let ndc = c.xy / max(c.w, 1e-6);
    return vec3<f32>((ndc * vec2<f32>(0.5, -0.5) + 0.5) * mvu.rsize.xy, c.w);
}

fn mv_value(in: MvVs) -> vec4<f32> {
    // only where this surface is what the main pass kept (as motion.wgsl
    // fs_obj: the depth buffer's point lies on it; leaf holes fail)
    let p = vec2<i32>(in.clip.xy);
    let mx = vec2<i32>(i32(mvu.rsize.x) - 1, i32(mvu.rsize.y) - 1);
    let d = textureLoad(t_mv_depth, clamp(p, vec2<i32>(0), mx), 0);
    let uv = (vec2<f32>(p) + 0.5) * mvu.rsize.zw;
    let wd4 = frame.inv_view_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, min(d, 0.9999999), 1.0);
    let dist = length(in.world - frame.eye.xyz);
    if (abs(length(wd4.xyz / wd4.w - frame.eye.xyz) - dist) > 0.01 * dist + 0.02) {
        discard;
    }
    let now = mv_px(mvu.vp, in.world);
    let was = mv_px(mvu.prev_vp, in.prev_world);
    // w 0: scenery with exact vectors, not a car (TAA's car handling)
    return vec4<f32>(was.xy - now.xy, min(max(was.z, 0.0), 60000.0), 0.0);
}

@fragment
fn fs_motion(in: MvVs) -> @location(0) vec4<f32> {
    return mv_value(in);
}

struct MvRg {
    @location(0) mv: vec4<f32>,
    @location(1) rg: vec2<f32>,
};

// with DLSS / FSR's RG16F copy
@fragment
fn fs_motion_rg(in: MvVs) -> MvRg {
    let v = mv_value(in);
    return MvRg(v, v.xy);
}

const PI: f32 = 3.14159265359;

// ------------------------------------------------------------------ sky ----

fn hg(mu: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));
}

/// Clear daytime sky radiance along `d`, without the sun disc. Deliberately
/// cheap: a Rayleigh-ish zenith-to-horizon gradient whose horizon whitens with
/// the haze, plus a Mie glow around the sun. Scaled by haze.w.
fn sky_radiance_old(d: vec3<f32>) -> vec3<f32> {
    let sun = frame.sun.xyz;
    let y = clamp(d.y, -1.0, 1.0);
    let up = max(y, 0.0);
    // ⚠ Was (0.16, 0.30, 0.62) reaching full strength a few degrees up, so
    // the whole dome read as one pale band. The game's own track previews
    // run from a pale horizon (~185, 210, 230 sRGB) to a much deeper blue
    // overhead (Spa / Monza ~110, 150, 190). Deeper zenith, slower gradient.
    let zenith = vec3<f32>(0.035, 0.17, 0.50) * frame.sky[2].rgb / vec3<f32>(0.38, 0.56, 0.92);
    let horizon = vec3<f32>(0.46, 0.60, 0.74) * frame.sky[3].rgb / vec3<f32>(0.78, 0.82, 0.88);
    var c = mix(horizon, zenith, pow(up, 0.5));
    // below the horizon the sky fades to the haze colour, never to black
    c = select(c, horizon * mix(1.0, 0.82, clamp(-y * 3.0, 0.0, 1.0)), y < 0.0);
    let mu = dot(d, sun);
    // sun-side brightening (Mie forward scatter) and a mild anti-sun Rayleigh lobe
    // (the broad lobe was 0.35: it whitened the whole dome, paler than the
    // previews' deep blue overhead)
    let mie = hg(mu, 0.78) * 0.55 + hg(mu, 0.2) * 0.12;
    c += frame.sun_color.rgb * frame.sky[0].w * mie * (0.35 + 0.65 * (1.0 - up));
    // the low sun warms the horizon band
    let low = 1.0 - clamp(sun.y * 2.5, 0.0, 1.0);
    c *= mix(vec3<f32>(1.0), vec3<f32>(1.08, 0.96, 0.82), low * (1.0 - up));
    return c * frame.haze.w;
}

// --------------------------------------------------- physical sky ----
//
// Single scattering through a spherical Rayleigh + Mie + ozone atmosphere
// (Hillaire 2020's Earth constants), sun transmittance marched per sample,
// and a flat multiple-scattering term (half the local scattering, isotropic)
// for the twilight blues. Radiance for a sun of illuminance 1, scaled by
// frame.atm2.y so a 60-degree sun's zenith matches the gradient sky's.

const ATM_R0: f32 = 6360000.0;
const ATM_R1: f32 = 6460000.0;
const ATM_RAY: vec3<f32> = vec3<f32>(5.802e-6, 13.558e-6, 33.1e-6);
const ATM_OZ: vec3<f32> = vec3<f32>(0.650e-6, 1.881e-6, 0.085e-6);
const ATM_MIE_S: f32 = 3.996e-6;
const ATM_MIE_E: f32 = 4.40e-6;
// eye altitude above the planet's surface: the track's own height is not
// known, and a couple of hundred metres is where most circuits sit
const ATM_EYE: f32 = 200.0;

fn atm_ext(h: f32) -> vec3<f32> {
    return ATM_RAY * exp(-h / 8000.0) + vec3<f32>(ATM_MIE_E * exp(-h / 1200.0))
         + ATM_OZ * max(0.0, 1.0 - abs(h - 25000.0) / 15000.0);
}

/// (near, far) distances from radius `r`, cosine of the zenith angle `mu`,
/// to the sphere of radius `rad`; both negative when the ray misses.
fn atm_hit(r: f32, mu: f32, rad: f32) -> vec2<f32> {
    let b = r * mu;
    let c = (r - rad) * (r + rad);
    let d = b * b - c;
    if (d < 0.0) {
        return vec2<f32>(-1.0);
    }
    let q = sqrt(d);
    return vec2<f32>(-b - q, -b + q);
}

/// Sunlight left at radius `r` for a sun at cosine-zenith `mus`.
fn atm_sun_tr(r: f32, mus: f32) -> vec3<f32> {
    if (atm_hit(r, mus, ATM_R0).x > 0.0) {
        return vec3<f32>(0.0);
    }
    let far = atm_hit(r, mus, ATM_R1).y;
    var od = vec3<f32>(0.0);
    for (var i = 0; i < 8; i = i + 1) {
        let f0 = f32(i) / 8.0;
        let f1 = f32(i + 1) / 8.0;
        let t = far * (f0 * f0 + f1 * f1) * 0.5;
        let rr = sqrt(r * r + t * t + 2.0 * r * t * mus);
        od += atm_ext(rr - ATM_R0) * (far * (f1 * f1 - f0 * f0));
    }
    return exp(-od);
}

/// Sky radiance along `d` for the sun toward `sun` (illuminance 1).
fn atmosphere(d: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let r0 = ATM_R0 + ATM_EYE;
    var tmax = atm_hit(r0, d.y, ATM_R1).y;
    let g = atm_hit(r0, d.y, ATM_R0);
    if (g.x > 0.0) {
        tmax = g.x;
    }
    let nu = dot(d, sun);
    let ph_r = 3.0 / (16.0 * PI) * (1.0 + nu * nu);
    let ph_m = hg(nu, 0.8);
    var l = vec3<f32>(0.0);
    var tr = vec3<f32>(1.0);
    let n = 24;
    for (var i = 0; i < n; i = i + 1) {
        let f0 = f32(i) / f32(n);
        let f1 = f32(i + 1) / f32(n);
        let t = tmax * (f0 * f0 + f1 * f1) * 0.5;
        let dt = tmax * (f1 * f1 - f0 * f0);
        let p = vec3<f32>(0.0, r0, 0.0) + d * t;
        let rr = length(p);
        let h = rr - ATM_R0;
        let sr = ATM_RAY * exp(-h / 8000.0);
        let sm = ATM_MIE_S * exp(-h / 1200.0);
        let e = atm_ext(h);
        let ts = atm_sun_tr(rr, dot(p / rr, sun));
        let src = ts * (sr * ph_r + vec3<f32>(sm * ph_m) + 0.5 * (sr + vec3<f32>(sm)) / (4.0 * PI));
        let st = exp(-e * dt);
        l += tr * src * (vec3<f32>(1.0) - st) / max(e, vec3<f32>(1e-12));
        tr *= st;
    }
    return l;
}

/// The physical clear sky in evoview's units: the atmosphere for the true
/// sun, the old night sky (stars glow, moonlight) faded in below the
/// horizon, and rain's flat grey.
fn sky_physical(d: vec3<f32>) -> vec3<f32> {
    let sun = frame.atm.xyz;
    var c = atmosphere(d, sun) * frame.atm2.y;
    // the gradient sky still carries the night (moon key, night floor)
    let night = 1.0 - smoothstep(-0.14, -0.02, sun.y);
    c += sky_radiance_old(d) * night;
    let r = frame.atm2.w;
    if (r > 0.0) {
        let l = dot(c, vec3<f32>(0.3, 0.59, 0.11));
        c = mix(c, vec3<f32>(l * 0.85), 0.8 * r) * (1.0 - 0.25 * r);
    }
    return c;
}

const PANO_LO: f32 = -0.1;
const PANO_SPAN: f32 = 1.6707963;

/// Panorama uv of direction `d` (track.rs SkyPano) in a panorama `hgt`
/// texels high (kept off the top and bottom rows' outer halves: the
/// sampler repeats).
fn pano_uv(d: vec3<f32>, hgt: f32) -> vec2<f32> {
    let el = asin(clamp(d.y, -1.0, 1.0));
    let v = sqrt(clamp((el - PANO_LO) / PANO_SPAN, 0.0, 1.0));
    let u = atan2(d.x, d.z) / (2.0 * PI) + 0.5;
    return vec2<f32>(u, clamp(1.0 - v, 0.5 / hgt, 1.0 - 0.5 / hgt));
}

fn pano_dir(uv: vec2<f32>) -> vec3<f32> {
    let v = 1.0 - uv.y;
    let el = v * v * PANO_SPAN + PANO_LO;
    let az = (uv.x - 0.5) * 2.0 * PI;
    return vec3<f32>(cos(el) * sin(az), sin(el), cos(el) * cos(az));
}

fn sky_radiance(d: vec3<f32>) -> vec3<f32> {
    if (frame.atm.w > 0.5) {
        return textureSampleLevel(t_pano_sky, s_cloud, pano_uv(d, f32(textureDimensions(t_pano_sky).y)), 0.0).rgb;
    }
    return sky_radiance_old(d);
}

/// The cloud panorama at `uv`, Catmull-Rom filtered (five bilinear taps,
/// the corners dropped): the panorama is coarser than the screen, and
/// bilinear magnification softens every cloud edge.
fn pano_cloud_cr(uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(t_pano_cloud));
    let sp = uv * size;
    let tc = floor(sp - 0.5) + 0.5;
    let f = sp - tc;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let lo = 0.5 / size.y;
    let t0 = vec2<f32>((tc.x - 1.0) / size.x, clamp((tc.y - 1.0) / size.y, lo, 1.0 - lo));
    let t3 = vec2<f32>((tc.x + 2.0) / size.x, clamp((tc.y + 2.0) / size.y, lo, 1.0 - lo));
    let t12 = vec2<f32>((tc.x + w2.x / w12.x) / size.x, clamp((tc.y + w2.y / w12.y) / size.y, lo, 1.0 - lo));
    let r = textureSampleLevel(t_pano_cloud, s_cloud, vec2<f32>(t12.x, t0.y), 0.0) * (w12.x * w0.y)
          + textureSampleLevel(t_pano_cloud, s_cloud, vec2<f32>(t0.x, t12.y), 0.0) * (w0.x * w12.y)
          + textureSampleLevel(t_pano_cloud, s_cloud, t12, 0.0) * (w12.x * w12.y)
          + textureSampleLevel(t_pano_cloud, s_cloud, vec2<f32>(t3.x, t12.y), 0.0) * (w3.x * w12.y)
          + textureSampleLevel(t_pano_cloud, s_cloud, vec2<f32>(t12.x, t3.y), 0.0) * (w12.x * w3.y);
    let wsum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    let c = r / wsum;
    return vec4<f32>(max(c.rgb, vec3<f32>(0.0)), clamp(c.a, 0.0, 1.0));
}

/// The cloud deck along `d`: the panorama, or marched here without one.
fn clouds_at(d: vec3<f32>, px: vec2<f32>) -> vec4<f32> {
    if (frame.atm.w > 0.5) {
        return pano_cloud_cr(pano_uv(d, f32(textureDimensions(t_pano_cloud).y)));
    }
    return clouds(d, px, 0);
}

/// One texel of the clear-sky panorama.
@fragment
fn fs_sky_atm(in: SkyOut) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(in.ndc.x * 0.5 + 0.5, 0.5 - in.ndc.y * 0.5);
    return vec4<f32>(sky_physical(pano_dir(uv)), 1.0);
}

/// One texel of the cloud panorama: the deck marched with this frame's
/// jitter and averaged into what the texel held. A quarter of the texels a
/// frame (one of each 2x2), all of them on a fresh start.
@fragment
fn fs_sky_pano(in: SkyOut) -> @location(0) vec4<f32> {
    let q = vec2<u32>(in.clip.xy) & vec2<u32>(1u);
    let fresh = frame.atm2.x >= 1.0;
    let prev = textureLoad(t_pano_cloud, vec2<i32>(in.clip.xy), 0);
    // ⚠ not discard: naga's discard demotes to a helper that still runs
    // the whole march (2.75 ms instead of ~0.4)
    if (!fresh && q.x + 2u * q.y != (u32(frame.haze2.w) & 3u)) {
        return prev;
    }
    let uv = vec2<f32>(in.ndc.x * 0.5 + 0.5, 0.5 - in.ndc.y * 0.5);
    let c = clouds(pano_dir(uv), in.clip.xy * 0.5, i32(frame.atm2.z));
    return mix(prev, c, frame.atm2.x);
}

/// Haze between the eye and `p`: (transmittance, inscattered radiance).
/// Height-falloff exponential fog, integrated analytically along the ray.
fn haze(p: vec3<f32>) -> vec4<f32> {
    let e = frame.eye.xyz;
    let dv = p - e;
    let dist = length(dv);
    let dir = dv / max(dist, 1e-3);
    let k = 1.0 / max(frame.haze.y, 1.0);
    let h0 = e.y - frame.haze.z;
    let dh = dv.y;
    // integral of exp(-k h) along the segment, divided by its length
    var avg = exp(-k * h0);
    if (abs(dh) > 0.01) {
        avg = (exp(-k * h0) - exp(-k * (h0 + dh))) / (k * dh);
    }
    // no haze nearer than the track's start distance
    let od = frame.haze.x * min(avg, 4.0) * max(dist - frame.haze2.x, 0.0);
    let t = exp(-od);
    if (t > 0.995) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // inscatter colour: what the sky looks like just above the horizon in this
    // direction, so distant hills melt into the sky rather than into grey
    let hd = normalize(vec3<f32>(dir.x, max(dir.y, 0.0) * 0.35 + 0.04, dir.z));
    // the haze itself is whiter than the sky behind it (Mie, not Rayleigh)
    var ins = mix(sky_radiance(hd), vec3<f32>(0.78, 0.80, 0.82) * frame.haze.w, 0.35);
    if (frame.atm.w > 0.5) {
        // the physical sky's own horizon, a little whiter (aerosol): warm
        // toward a low sun, blue away from it, dim at dusk - the constant
        // grey stayed noon-grey at every hour
        let hz = sky_radiance(hd);
        ins = mix(hz, vec3<f32>(dot(hz, vec3<f32>(0.2126, 0.7152, 0.0722))) * 1.12, 0.3);
    }
    return vec4<f32>(ins, t);
}

/// The one place colour leaves the HDR domain; nothing else in this file
/// calls aces().
// ⚠ Kept identical to tonemap() in shader.wgsl (the lead owns it): the two
// pipelines must grade the same or cars and track disagree on one screen.
fn tonemap(c: vec3<f32>) -> vec3<f32> {
    // HDR pipeline (render.rs HDR): exposed linear light, tonemapped once in
    // the post pass
    if (frame.grade.y > 1.5) {
        return hdr_out(c * frame.grade.x);
    }
    if (frame.grade.y < 0.5) {
        return aces(c);
    }
    let x = max(c * frame.grade.x, vec3<f32>(0.0));
    var uvw = clamp((log2(x + vec3<f32>(0.0009765625)) + vec3<f32>(10.0)) / 16.0,
                    vec3<f32>(0.0), vec3<f32>(1.0));
    uvw = uvw * (31.0 / 32.0) + vec3<f32>(0.5 / 32.0);
    uvw.z = (uvw.z + frame.grade.z) * 0.5;
    return textureSampleLevel(t_lut, s_clamp, uvw, 0.0).rgb;
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ------------------------------------------------------------- shadow ----

/// The game's sun-shadow filter (shadow resolve 033e0fd0): a 5x5 tent as 9
/// bilinear compare taps (Castano's optimised PCF), weights over 144.
fn pcf5(uv: vec2<f32>, z: f32, layer: i32) -> f32 {
    let dim = f32(textureDimensions(t_shadow).x);
    let inv = 1.0 / dim;
    let p = uv * dim + 0.5;
    let b = floor(p);
    let s = p.x - b.x;
    let t = p.y - b.y;
    let uw = vec3<f32>(4.0 - 3.0 * s, 7.0, 1.0 + 3.0 * s);
    let uo = vec3<f32>((3.0 - 2.0 * s) / uw.x - 2.0, (3.0 + s) / 7.0, s / uw.z + 2.0);
    let vw = vec3<f32>(4.0 - 3.0 * t, 7.0, 1.0 + 3.0 * t);
    let vo = vec3<f32>((3.0 - 2.0 * t) / vw.x - 2.0, (3.0 + t) / 7.0, t / vw.z + 2.0);
    let base = (b - 0.5) * inv;
    var sum = 0.0;
    for (var j = 0; j < 3; j = j + 1) {
        for (var i = 0; i < 3; i = i + 1) {
            sum += uw[i] * vw[j] * textureSampleCompareLevel(t_shadow, s_shadow,
                base + vec2<f32>(uo[i], vo[j]) * inv, layer, z);
        }
    }
    return sum / 144.0;
}

/// The 3x3 tent as 4 bilinear compare taps (Castano), weights over 16: for
/// the coarse cascades, whose texels make the 5x5 blur metres wide.
fn pcf3(uv: vec2<f32>, z: f32, layer: i32) -> f32 {
    let dim = f32(textureDimensions(t_shadow).x);
    let inv = 1.0 / dim;
    let p = uv * dim + 0.5;
    let b = floor(p);
    let s = p.x - b.x;
    let t = p.y - b.y;
    let uw = vec2<f32>(3.0 - 2.0 * s, 1.0 + 2.0 * s);
    let uo = vec2<f32>((2.0 - s) / uw.x - 1.0, s / uw.y + 1.0);
    let vw = vec2<f32>(3.0 - 2.0 * t, 1.0 + 2.0 * t);
    let vo = vec2<f32>((2.0 - t) / vw.x - 1.0, t / vw.y + 1.0);
    let base = (b - 0.5) * inv;
    var sum = 0.0;
    for (var j = 0; j < 2; j = j + 1) {
        for (var i = 0; i < 2; i = i + 1) {
            sum += uw[i] * vw[j] * textureSampleCompareLevel(t_shadow, s_shadow,
                base + vec2<f32>(uo[i], vo[j]) * inv, layer, z);
        }
    }
    return sum / 16.0;
}

/// A cascade's filter by its shadow vec4's x (render.rs cascade_kernel).
fn pcf_k(uv: vec2<f32>, z: f32, layer: i32, k: f32) -> f32 {
    if (k > 2.5) {
        return pcf3(uv, z, layer);
    }
    if (k > 1.5) {
        return textureSampleCompareLevel(t_shadow, s_shadow, uv, layer, z);
    }
    return pcf5(uv, z, layer);
}

/// Sun visibility from one cascade, and how far into its edge band the
/// point is (0 inside .. 1 outside).
/// Normal offset as the game's resolve does it (normalOffsetScale x
/// (1 - |N.L|) texels along the normal) instead of a big slope-scaled depth
/// bias: the depth bias that stops acne on a grazing wall pushed every
/// shadow up to ~0.4 m off its caster (peter-panning under kerbs, posts,
/// the car). sc: the cascade's shadow vec4 (y depth bias, w normal offset m).
fn shadow_layer(world: vec3<f32>, n: vec3<f32>, n_dot_l: f32, vp: mat4x4<f32>, sc: vec4<f32>,
                layer: i32, fade0: f32) -> vec2<f32> {
    let slope = clamp(1.0 - n_dot_l, 0.0, 1.0);
    let off = n * (sc.w * slope);
    let lc = vp * vec4<f32>(world + off, 1.0);
    if (lc.w <= 0.0) {
        return vec2<f32>(1.0, 1.0);
    }
    let ndc = lc.xyz / lc.w;
    if (ndc.z <= 0.0 || ndc.z >= 1.0) {
        return vec2<f32>(1.0, 1.0);
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec2<f32>(1.0, 1.0);
    }
    let d = max(abs(uv.x - 0.5), abs(uv.y - 0.5)) * 2.0;
    let edge = smoothstep(fade0, 1.0, d);
    if (edge >= 1.0) {
        return vec2<f32>(1.0, 1.0);
    }
    // sc.y is ~1.5 texels of depth (render.rs cascade_bias); the normal
    // offset above and the pass's slope-scaled bias cover grazing receivers,
    // so this only grows by half again (it was 3x: 4.4 m in the far map)
    let bias = sc.y * (1.0 + slope);
    return vec2<f32>(pcf_k(uv, ndc.z - bias, layer, sc.x), edge);
}

/// The driven car's hero map: the car's own shadow, at ~6 mm a texel, in
/// a corner of layer 3 (render.rs hero_light_matrix). 1 outside it.
fn hero_shadow(world: vec3<f32>, nl: vec3<f32>, ndl: f32) -> f32 {
    let sc = frame.shadow_hero;
    if (sc.x <= 0.0) {
        return 1.0;
    }
    let slope = clamp(1.0 - ndl, 0.0, 1.0);
    let lc = frame.light_vp_hero * vec4<f32>(world + nl * (sc.w * slope), 1.0);
    let ndc = lc.xyz / lc.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (ndc.z <= 0.0 || ndc.z >= 1.0 || any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return 1.0;
    }
    return pcf5(uv * sc.x, ndc.z - sc.y * (1.0 + 2.0 * slope), 3);
}

/// Sun shadow from the near map, handing over to the middle cascade at its
/// edge, that to the far one, and that to lit at the far edge; with the
/// driven car's own hero map on top. `n` the surface normal, for the
/// normal offset.
fn sun_shadow(world: vec3<f32>, n: vec3<f32>) -> f32 {
    if (frame.shadow.x < 0.5) {
        return 1.0;
    }
    // the side of the surface the sun is on (cards are lit from both)
    let nl = n * select(-1.0, 1.0, dot(n, frame.sun.xyz) >= 0.0);
    let ndl = dot(nl, frame.sun.xyz);
    let hero = hero_shadow(world, nl, ndl);
    let near = shadow_layer(world, nl, ndl, frame.light_vp, frame.shadow, 0, 0.7);
    if (near.y <= 0.0) {
        return min(near.x, hero);
    }
    var far = 1.0;
    var far_t = 1.0;
    if (frame.shadow_mid.x > 0.5) {
        let m = shadow_layer(world, nl, ndl, frame.light_vp_mid, frame.shadow_mid, 2, 0.8);
        far = m.x;
        far_t = m.y;
    }
    if (far_t > 0.0 && frame.shadow_far.x > 0.5) {
        let f = shadow_layer(world, nl, ndl, frame.light_vp_far, frame.shadow_far, 1, 0.8);
        far = mix(far, mix(f.x, 1.0, f.y), far_t);
    } else if (far_t > 0.0) {
        far = mix(far, 1.0, far_t);
    }
    return min(mix(near.x, far, near.y), hero);
}

// ------------------------------------------------------------ ambient ----

/// Irradiance from the volumes at `p` for normal `n`, already divided by pi.
/// Returns -1 in x when there is no atlas or `p` lies outside it.
fn iv_irradiance(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return iv_rows(p, n, 0.0);
}

/// The night irradiance baked into the same volumes (rows 12..24).
fn iv_night(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return iv_rows(p, n, 12.0);
}

fn iv_rows(p: vec3<f32>, n: vec3<f32>, row0: f32) -> vec3<f32> {
    if (frame.iv_elev.w < 0.5) {
        return vec3<f32>(-1.0);
    }
    let u = (p.x - frame.iv_origin.x) * frame.iv_origin.z;
    let w = (p.z - frame.iv_origin.y) * frame.iv_origin.w;
    if (u < 0.0 || u > 1.0 || w < 0.0 || w > 1.0) {
        return vec3<f32>(-1.0);
    }
    // Ground height under this column rides in the alpha of every texel.
    let rows = 24.0;
    let g = textureSampleLevel(t_iv, s_clamp, vec3<f32>(u, 0.5 / rows, w), 0.0).a;
    let h = p.y - g;
    // probe layers sit at iv_elev.xyz above the ground
    let e = frame.iv_elev.xyz;
    var t = 0.0;
    if (h > e.x) {
        t = select(1.0 + clamp((h - e.y) / max(e.z - e.y, 0.01), 0.0, 1.0),
                   clamp((h - e.x) / max(e.y - e.x, 0.01), 0.0, 1.0), h < e.y);
    }
    var c: array<vec3<f32>, 4>;
    for (var k = 0; k < 4; k = k + 1) {
        let v = (row0 + f32(k) * 3.0 + 0.5 + t) / rows;
        c[k] = textureSampleLevel(t_iv, s_clamp, vec3<f32>(u, v, w), 0.0).rgb;
    }
    // Rain (frame.wet.y): the volumes are baked under a clear sky, their
    // directional terms mostly the sun's bounce off the ground and walls.
    // Under the deck that bounce is gone and the light comes evenly from a
    // grey dome: keep the average (L0), flatten the direction and the
    // colour toward neutral. Day volumes only - the night's are the lamps.
    let flat_k = select(0.0, frame.wet.y, row0 < 0.5);
    let dir_k = 1.0 - 0.75 * flat_k;
    var e_ = c[0] * 0.886227;
    e_ += c[1] * max(-n.y, 0.0) * 1.0233 * dir_k;
    e_ += c[2] * max(n.z, 0.0) * 1.0233 * dir_k;
    e_ += c[3] * max(-n.x, 0.0) * 1.0233 * dir_k;
    let lum = dot(e_, vec3<f32>(0.3, 0.59, 0.11));
    e_ = mix(e_, vec3<f32>(lum) * vec3<f32>(0.96, 0.99, 1.04), 0.6 * flat_k);
    return max(e_, vec3<f32>(0.0)) * (1.0 / PI);
}

// ------------------------------------------------------------- lamps ----

/// The track's own lamps near `p` (LightData: point / spot, UE-style falloff).
fn lamp_light(p: vec3<f32>, n: vec3<f32>, v: vec3<f32>, albedo: vec3<f32>, f0: vec3<f32>, r: f32)
        -> vec3<f32> {
    if (frame.lamps.z <= 0.0 || frame.lamps.y < 0.5) {
        return vec3<f32>(0.0);
    }
    let cx = i32(floor((p.x - frame.lamp_grid.x) * frame.lamp_grid.z));
    let cz = i32(floor((p.z - frame.lamp_grid.y) * frame.lamp_grid.z));
    let nx = i32(frame.lamp_grid.w);
    if (cx < 0 || cz < 0 || cx >= nx || cz >= i32(frame.lamps.x)) {
        return vec3<f32>(0.0);
    }
    let cell = lamp_cells[cz * nx + cx];
    let night = frame.sky[2].w;
    var out = vec3<f32>(0.0);
    for (var i = 0u; i < min(cell.y, 64u); i = i + 1u) {
        let L = lamps[lamp_index[cell.x + i]];
        let on = max(night, L.ext.z);
        if (on <= 0.0) {
            continue;
        }
        let d = L.pos.xyz - p;
        let d2 = dot(d, d);
        let rad = L.pos.w;
        if (d2 > rad * rad) {
            continue;
        }
        let l = d * inverseSqrt(max(d2, 1e-6));
        let nol = dot(n, l);
        if (nol <= 0.0) {
            continue;
        }
        // UE: inverse square with a smooth window, or (1 - (d/r)^2)^exp
        let x2 = d2 / (rad * rad);
        var att: f32;
        if (L.col.w > 0.0) {
            att = pow(clamp(1.0 - x2, 0.0, 1.0), L.col.w);
        } else {
            let win = clamp(1.0 - x2 * x2, 0.0, 1.0);
            att = win * win / max(d2, 0.01);
        }
        if (L.ext.y > 0.5) {
            let cd = dot(-l, normalize(L.dir.xyz));
            att *= smoothstep(L.dir.w, L.ext.x, cd);
        }
        if (att <= 0.0) {
            continue;
        }
        let h = normalize(v + l);
        let noh = max(dot(n, h), 0.0);
        let a2 = max(r * r * r * r, 1e-4);
        let dd = noh * noh * (a2 - 1.0) + 1.0;
        let spec = f0 * a2 / max(PI * dd * dd, 1e-6) * 0.25;
        out += (albedo / PI + spec) * L.col.rgb * (att * nol * on);
    }
    return out * frame.lamps.z;
}

// ---------------------------------------------------------- materials ----

fn uv_for(channel: f32, in: VsOut) -> vec2<f32> {
    return select(in.uv, in.uv1, channel > 0.5);
}

fn triplanar(tex: texture_2d<f32>, p: vec3<f32>, n: vec3<f32>, scale: f32) -> vec4<f32> {
    let an = abs(n);
    var w = pow(max(an, vec3<f32>(0.0001)), vec3<f32>(5.0));
    w = w / (w.x + w.y + w.z);
    var r = vec4<f32>(0.0);
    // skip the weak planes: most terrain is nearly flat and only needs one tap
    if (w.y > 0.02) { r += textureSample(tex, s_wrap, p.xz * scale) * w.y; }
    if (w.x > 0.02) { r += textureSample(tex, s_wrap, p.zy * scale) * w.x; }
    if (w.z > 0.02) { r += textureSample(tex, s_wrap, p.xy * scale) * w.z; }
    return r / max(select(0.0, w.y, w.y > 0.02) + select(0.0, w.x, w.x > 0.02)
                   + select(0.0, w.z, w.z > 0.02), 1e-4);
}

fn layer_sample(tex: texture_2d<f32>, cfg: vec4<f32>, in: VsOut, n: vec3<f32>) -> vec4<f32> {
    if (cfg.z > 0.0) {
        return triplanar(tex, in.world, n, cfg.z);
    }
    let s = select(vec2<f32>(1.0), cfg.xy, abs(cfg.x) > 1e-5);
    return textureSample(tex, s_wrap, uv_for(cfg.w, in) * s);
}

fn unpack_normal(s: vec4<f32>) -> vec3<f32> {
    // BC5 / BC7 tangent-space maps: xy carries the vector, z is rebuilt
    let xy = s.xy * 2.0 - 1.0;
    return vec3<f32>(xy, sqrt(max(1.0 - dot(xy, xy), 0.0)));
}

// The game's 4x4 screen-door matrix for LOD transitions (BayerMatrix4 in
// its prepass/material pixel shaders, e.g. 0331fa40): values (k + 1) / 17.
const BAYER4 = array<f32, 16>(
    1.0, 13.0, 4.0, 16.0, 9.0, 5.0, 12.0, 8.0,
    3.0, 15.0, 2.0, 14.0, 11.0, 7.0, 10.0, 6.0);
// per-frame shift of that matrix (the game's ditherTemporalOffset), in
// ordered-dither order so successive frames land far apart and TAA's short
// history sees every threshold
const DITHER_SHIFT = array<vec2<u32>, 16>(
    vec2<u32>(0u, 0u), vec2<u32>(2u, 2u), vec2<u32>(2u, 0u), vec2<u32>(0u, 2u),
    vec2<u32>(1u, 1u), vec2<u32>(3u, 3u), vec2<u32>(3u, 1u), vec2<u32>(1u, 3u),
    vec2<u32>(1u, 0u), vec2<u32>(3u, 2u), vec2<u32>(3u, 0u), vec2<u32>(1u, 2u),
    vec2<u32>(0u, 1u), vec2<u32>(2u, 3u), vec2<u32>(2u, 1u), vec2<u32>(0u, 3u));

/// LOD crossfade: the game keeps a pixel when uint(bayer + coverage) >= 1,
/// i.e. drops it when bayer < 1 - coverage (`out` here).
fn lod_dither(p: vec2<f32>, out: f32) -> bool {
    // naga indexes only a variable array with a runtime index
    var shifts = DITHER_SHIFT;
    var bayer = BAYER4;
    let q = (vec2<u32>(p) + shifts[u32(frame.haze2.w) & 15u]) & vec2<u32>(3u);
    return bayer[q.x + q.y * 4u] / 17.0 < out;
}

/// The game's cutout (a2c prepass 0x03350b70, msaaSampleCount 1): coverage
/// sat(opacityScale (alpha - cutoff)) kept where it beats the temporally
/// shifted Bayer, so a thin cable whose mips fade its alpha below the
/// threshold stays as a sparse dither TAA turns into a faint line, not
/// dashes. surf.w is the 50 % point, cutoff + 0.5 / scale (track.rs); ao.z
/// the scale, 0 = the old hard threshold. True = drop the pixel.
fn cutout_drop(p: vec2<f32>, alpha: f32) -> bool {
    if (mat.ao.z <= 0.0) {
        return alpha < mat.surf.w;
    }
    let cov = clamp(mat.ao.z * (alpha - mat.surf.w) + 0.5, 0.0, 1.0);
    var shifts = DITHER_SHIFT;
    var bayer = BAYER4;
    let q = (vec2<u32>(p) + shifts[u32(frame.haze2.w) & 15u]) & vec2<u32>(3u);
    return !(cov > bayer[q.x + q.y * 4u] / 17.0);
}


/// This material's texture LOD bias (track.rs turns every plain
/// textureSample into a textureSampleBias by it): TAA's sharpening bias
/// (taa.rs mip_bias, -1), except on alpha-blended materials (extra.w bit
/// 6: fences, nets, screens; not the cutout foliage), which take frame.atm4.y of
/// it (EVOVIEW_SEETHROUGH_MIP_K, 0). ⚠ A chain-link mesh sampled sharper
/// than the pixel grid beats against it: the fences crawled and shimmered
/// in motion (1 cm/frame dolly past Monza's: frame-to-frame fine change
/// 2.69 at -1, 1.33 at 0), and the blended pass gives TAA nothing to
/// clamp them with.
fn lod_bias() -> f32 {
    if ((u32(mat.extra.w + 0.5) & 64u) != 0u) {
        return frame.lod.x * frame.atm4.y;
    }
    return frame.lod.x;
}

/// Dither for alpha-tested cards and the grass fade, stable per pixel.
fn dither(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

struct Surface {
    albedo: vec3<f32>,
    // the base map colour before the splat layers (debug view 6)
    base_col: vec3<f32>,
    alpha: f32,
    n_ts: vec3<f32>,
    rough: f32,
    ao: f32,
    // added to the shading normal in world space (DynamicTrack detail
    // normal, sampled in world XZ); zero elsewhere
    n_world: vec3<f32>,
    // Grass subsurface colour (SubsurfaceColorMap x Subsurface_Color x the
    // ground pigment); zero elsewhere
    sss: vec3<f32>,
};

/// DynamicTrack, the game's road shader (lift road/lift_036c4720): the
/// base map is packed (x y normal, z albedo, w roughness), the detail map
/// (a r a o) is sampled in world XZ at Detail_BiplanarScale + 0.01, the
/// variation map on its UV channel, the detail normal in world XZ too.
fn road_surface(in: VsOut, ng: vec3<f32>) -> Surface {
    var s: Surface;
    let buv = uv_for(0.0, in) * mat.base_uv.xy;
    let bm = textureSample(t_base, s_wrap, buv);
    let wxz = in.world.xz * mat.green_uv.xy;
    let dm = textureSample(t_detail, s_wrap, wxz);
    // lift: the map alone unless Base_ModulateBaseColor (base_uv.z)
    var c = vec3<f32>(bm.z) * dm.r;
    if (mat.base_uv.z > 0.5) {
        c = c * mat.base.rgb;
    }
    if (mat.green.w > 0.5) {
        c = c * mat.green.rgb;
    }
    if (mat.red.w > 0.5) {
        let vuv = uv_for(mat.red_uv.z, in) * mat.red_uv.xy;
        let vr = textureSample(t_red, s_wrap, vuv).r;
        let tint = select(vec3<f32>(1.0), mat.red.rgb, mat.red_uv.w > 0.5);
        c = c * vr * tint;
    }
    s.albedo = c;
    s.base_col = c;
    s.alpha = 1.0;
    let nxy = (bm.xy * 2.0 - 1.0) * mat.blue.z;
    s.n_ts = vec3<f32>(nxy, sqrt(max(1.0 - dot(nxy, nxy), 0.0)));
    s.rough = clamp(mat.blue.x * (0.5 + bm.w) * mix(1.0, dm.g * mat.blue.y, mat.blue.w), 0.2, 1.0);
    s.ao = 1.0;
    if (mat.green_uv.w > 0.5) {
        let dn = textureSample(t_red_n, s_wrap, wxz).xyz * 2.0 - 1.0;
        // the lift adds (x, -y) of the detail normal to the geometric X / Z
        s.n_world = vec3<f32>(dn.x, 0.0, -dn.y) * mat.green_uv.z * 0.5;
    }
    return s;
}

fn surface(in: VsOut, ng: vec3<f32>) -> Surface {
    if (mat.splat.x > 8.5) {
        return road_surface(in, ng);
    }
    var s: Surface;
    s.n_ts = vec3<f32>(0.0, 0.0, 1.0);
    s.rough = mat.surf.x;
    s.ao = 1.0;
    let buv = uv_for(mat.base_uv.z, in) * select(vec2<f32>(1.0), mat.base_uv.xy, abs(mat.base_uv.x) > 1e-5);
    var base = vec4<f32>(1.0);
    if (mat.base_uv.w > 0.5) {
        base = textureSample(t_base, s_wrap, buv);
    }
    s.albedo = base.rgb * mat.base.rgb;
    s.alpha = base.a;
    if (mat.extra.x > 0.5) {
        // grass cards take the colour of the ground they stand on
        var pig = vec3<f32>(0.5);
        if (frame.pigment.z > 0.0) {
            let puv = vec2<f32>((in.world.x - frame.pigment.x) * frame.pigment.z,
                                (in.world.z - frame.pigment.y) * frame.pigment.w);
            pig = textureSampleLevel(t_pigment, s_clamp, puv, 0.0).rgb;
        }
        s.albedo = s.albedo * pig;
    }
    // the subsurface map rides in the blue layer slot (track.rs): grass
    // tinted by the ground pigment, foliage as it is
    if (mat.splat.x < 0.5 && mat.blue.w > 0.5 && mat.base.w > 2.5 && mat.base.w < 3.5) {
        var tint = vec3<f32>(1.0);
        if (mat.extra.x > 0.5) {
            var pig = vec3<f32>(0.5);
            if (frame.pigment.z > 0.0) {
                let puv = vec2<f32>((in.world.x - frame.pigment.x) * frame.pigment.z,
                                    (in.world.z - frame.pigment.y) * frame.pigment.w);
                pig = textureSampleLevel(t_pigment, s_clamp, puv, 0.0).rgb;
            }
            tint = pig * 2.0;
        }
        s.sss = textureSample(t_blue, s_wrap, buv).rgb * mat.blue.rgb * tint;
    }
    if (mat.flags.x > 0.5) {
        s.n_ts = unpack_normal(textureSample(t_base_n, s_wrap, buv));
    }
    if (mat.flags.y > 0.5) {
        s.rough = clamp(textureSample(t_base_r, s_wrap, buv).r * max(mat.surf.x, 0.05) * 1.25, 0.04, 1.0);
    }
    if (mat.flags.z > 0.5) {
        s.alpha = textureSample(t_opacity, s_wrap, buv).r;
    }
    // the cutout's alpha at the material's own mip bias (f11), as the
    // game's prepass reads it
    if (mat.surf.w > 0.0 && mat.ao.w != 0.0) {
        if (mat.flags.z > 0.5) {
            s.alpha = textureSampleBias(t_opacity, s_wrap, buv, lod_bias() + mat.ao.w).r;
        } else if (mat.base_uv.w > 0.5) {
            s.alpha = textureSampleBias(t_base, s_wrap, buv, lod_bias() + mat.ao.w).a;
        }
    }

    s.base_col = s.albedo;
    // ---- splat layers ----
    let mode = mat.splat.x;
    if (mode > 0.5) {
        var wts = vec3<f32>(0.0);
        if (mode < 1.5) {
            wts = textureSample(t_splat, s_wrap, uv_for(mat.splat.y, in)).rgb;
        } else if (mode < 2.5) {
            // the vertex colour is stored / 255
            wts = clamp(in.color.rgb * 255.0 * mat.splat_const.w, vec3<f32>(0.0), vec3<f32>(1.0));
        } else {
            wts = mat.splat_const.rgb;
        }
        // As the game's UberBiplanar does it (lift of shader 0x034e5ce0):
        // unless Splatmap_DisableRestrictBlendToTwoLayers, the weakest of
        // the three is dropped; base takes saturate(1 - r - g - b); then
        // either a normalised weighted SUM, or (Splatmap_UseMultiply) a
        // PRODUCT of lerp(1, layer, w) terms.
        let flags = u32(mat.extra.w + 0.5);
        if ((flags & 2u) == 0u) {
            let mn = min(min(wts.x, wts.y), wts.z);
            wts = wts * select(vec3<f32>(1.0), vec3<f32>(0.0), wts == vec3<f32>(mn));
        }
        wts = clamp(wts, vec3<f32>(0.0), vec3<f32>(1.0)) * vec3<f32>(mat.red.w, mat.green.w, mat.blue.w);
        // contrast: push towards the dominant layer (cheap height-blend stand-in)
        let c = max(mat.splat.w, 1.0);
        if (c > 1.0) {
            wts = pow(wts, vec3<f32>(c));
        }
        let wbase = clamp(1.0 - wts.x - wts.y - wts.z, 0.0, 1.0);
        let wsum = max(wbase + wts.x + wts.y + wts.z, 1e-4);
        let multiply = (flags & 1u) != 0u;
        let nmask = u32(mat.splat.z + 0.5);
        var lr = vec3<f32>(1.0);
        var lg = vec3<f32>(1.0);
        var lb = vec3<f32>(1.0);
        var rgh = s.rough * wbase;
        var nts = s.n_ts * wbase;
        if (wts.x > 0.003) {
            lr = layer_sample(t_red, mat.red_uv, in, ng).rgb * mat.red.rgb;
            rgh += mat.layer_rough.x * wts.x;
            nts += select(vec3<f32>(0.0, 0.0, 1.0), unpack_normal(layer_sample(t_red_n, mat.red_uv, in, ng)), (nmask & 1u) != 0u) * wts.x;
        }
        if (wts.y > 0.003) {
            lg = layer_sample(t_green, mat.green_uv, in, ng).rgb * mat.green.rgb;
            rgh += mat.layer_rough.y * wts.y;
            nts += select(vec3<f32>(0.0, 0.0, 1.0), unpack_normal(layer_sample(t_green_n, mat.green_uv, in, ng)), (nmask & 2u) != 0u) * wts.y;
        }
        if (wts.z > 0.003) {
            lb = layer_sample(t_blue, mat.blue_uv, in, ng).rgb * mat.blue.rgb;
            rgh += mat.layer_rough.z * wts.z;
            nts += select(vec3<f32>(0.0, 0.0, 1.0), unpack_normal(layer_sample(t_blue_n, mat.blue_uv, in, ng)), (nmask & 4u) != 0u) * wts.z;
        }
        var alb: vec3<f32>;
        let road = mat.base.w > 0.5 && mat.base.w < 1.5;
        if (road) {
            // DynamicTrack is its own shader, not the lifted UberBiplanar: its
            // weights are the runtime rubber/dirt map. Treat every layer as a
            // grain over the base map, normalised by the layer's own mean
            // (its smallest mip), so the base keeps the road's brightness.
            let nr = lr / max(dot(textureSampleLevel(t_red, s_wrap, vec2<f32>(0.5), 16.0).rgb, vec3<f32>(0.333)), 0.03);
            let nb = lb / max(dot(textureSampleLevel(t_blue, s_wrap, vec2<f32>(0.5), 16.0).rgb, vec3<f32>(0.333)), 0.03);
            alb = s.albedo * mix(vec3<f32>(1.0), clamp(nr, vec3<f32>(0.0), vec3<f32>(2.5)), wts.x * 0.6)
                           * mix(vec3<f32>(1.0), clamp(nb, vec3<f32>(0.0), vec3<f32>(2.5)), wts.z * 0.6);
        } else if (multiply) {
            alb = mix(vec3<f32>(1.0), s.albedo, wbase / wsum)
                * mix(vec3<f32>(1.0), lr, wts.x)
                * mix(vec3<f32>(1.0), lg, wts.y)
                * mix(vec3<f32>(1.0), lb, wts.z);
        } else {
            alb = (s.albedo * wbase + lr * wts.x + lg * wts.y + lb * wts.z) / wsum;
        }
        // ---- Detail (lift of 0x034e5ce0, end of the material block) ----
        // NOT a detail texture: the BASE map (the big baked colour, or the
        // kerb paint) re-tints whatever the layers made, scaled by
        // Detail_BaseColorMultiplier, faded out where the base itself shows:
        //   out = saturate(layers * lerp(M * base, 1, base_weight))
        // This is what turns a brown grass tile green and a concrete grain
        // into blue/yellow kerbs.
        if (mat.detail.x > 0.5 && !road) {
            let k = mat.detail.y * s.albedo;
            let mk = wbase / wsum;
            alb = clamp(alb * (k + (vec3<f32>(1.0) - k) * mk), vec3<f32>(0.0), vec3<f32>(1.0));
        }
        rgh = rgh / wsum;
        nts = normalize(nts);
        s.albedo = clamp(alb, vec3<f32>(0.0), vec3<f32>(1.0));
        s.rough = clamp(rgh, 0.045, 1.0);
        s.n_ts = nts;
    }

    // ---- material AO map (Uber: lerp by strength; Subsurface: x strength) ----
    if (mat.ao.y > 0.5) {
        let a = clamp(textureSample(t_ao, s_wrap, buv).r, 0.0, 1.0);
        s.ao = select(clamp(a * mat.ao.x, 0.0, 1.0), clamp(1.0 + (a - 1.0) * mat.ao.x, 0.0, 1.0), mat.ao.y < 1.5);
    }
    // ---- vertex AO (Vertex_AmbientOcclusionChannel / Intensity / Offset) ----
    if (mat.vao.x > 0.5) {
        let ch = i32(mat.vao.w + 0.5);
        let raw = select(select(select(in.color.a, in.color.b, ch == 2), in.color.g, ch == 1), in.color.r, ch == 0);
        // on top of the map's AO (the Uber's sat(COLOR x ao x intensity))
        s.ao = s.ao * clamp((raw - mat.vao.z) * mat.vao.y, 0.0, 1.0);
    }
    return s;
}

// ------------------------------------------------------------- shading ----

fn ggx(n_dot_h: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

fn env_brdf(f0: vec3<f32>, rough: f32, n_dot_v: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = vec4<f32>(rough) * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + vec2<f32>(r.z, r.w);
    return f0 * ab.x + vec3<f32>(ab.y);
}

/// Shadow casting for cutout materials: depth only, with the alpha test, so
/// a tree card casts its leaves and not its rectangle.
@fragment
fn fs_shadow(in: VsOut, @builtin(front_facing) front: bool) {
    var ng = normalize(in.normal);
    if (!front) {
        ng = -ng;
    }
    let s = surface(in, ng);
    // the same dithered coverage, in shadow-map texels: the 5x5 PCF then
    // reads a thin or fading card as part shadow instead of all or nothing
    if (mat.surf.w > 0.0 && cutout_drop(in.clip.xy, s.alpha)) {
        discard;
    }
}

/// 2D value noise, for puddles.
fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = fract(sin(dot(i, vec2<f32>(127.1, 311.7))) * 43758.5453);
    let b = fract(sin(dot(i + vec2<f32>(1.0, 0.0), vec2<f32>(127.1, 311.7))) * 43758.5453);
    let c = fract(sin(dot(i + vec2<f32>(0.0, 1.0), vec2<f32>(127.1, 311.7))) * 43758.5453);
    let d = fract(sin(dot(i + vec2<f32>(1.0, 1.0), vec2<f32>(127.1, 311.7))) * 43758.5453);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/// How the water at `world` differs from the track's (rain.rs dry_map): 1 dry
/// (the dry line, or drained by the flow), 0 as wet, -1 standing water the
/// flow collected. The map's 1 m cells, filtered between their centres,
/// within the half of the map around the eye the CPU writes.
fn dry_line(world: vec3<f32>) -> f32 {
    if (frame.wet.w < 0.5) {
        return 0.0;
    }
    let p = world.xz - vec2<f32>(0.5);
    let f = floor(p);
    let d = abs(f - floor(frame.eye.xz));
    if (max(d.x, d.y) > 118.0) {
        return 0.0;
    }
    let t = p - f;
    let i = vec2<i32>(f) & vec2<i32>(255);
    let j = (i + vec2<i32>(1)) & vec2<i32>(255);
    let a = textureLoad(t_dry, i, 0).r;
    let b = textureLoad(t_dry, vec2<i32>(j.x, i.y), 0).r;
    let c = textureLoad(t_dry, vec2<i32>(i.x, j.y), 0).r;
    let e = textureLoad(t_dry, j, 0).r;
    return 1.0 - 2.0 * mix(mix(a, b, t.x), mix(c, e, t.x), t.y);
}

/// Rain on the ground: porous surfaces darken, the upward faces go glossy
/// (water's F0 0.02), and flat ground collects puddles. x the surface's
/// wetness 0..1, y how much of it is standing water.
fn wet_amount(world: vec3<f32>, ng: vec3<f32>, kind: f32) -> vec2<f32> {
    // squared: a line that has lost half its water reads as mostly dry
    // film, the way a wet racing line goes matte well before it is dry
    let dw = dry_line(world);
    let dl = 1.0 - max(dw, 0.0);
    let pool = max(-dw, 0.0);
    let w = min(frame.wet.x * dl * dl + pool * frame.wet.x, 1.0);
    if (w <= 0.0 || (kind > 2.5 && kind < 3.5)) {
        return vec2<f32>(0.0);
    }
    let up = clamp((ng.y - 0.2) / 0.6, 0.0, 1.0);
    let n = vnoise(world.xz * 0.35) * 0.65 + vnoise(world.xz * 1.7) * 0.35;
    let flat = clamp((ng.y - 0.985) / 0.012, 0.0, 1.0);
    // the noise's puddles, and where the flow piled the water up
    let puddle = max(smoothstep(0.58, 0.68, n) * w, smoothstep(0.2, 0.7, pool) * frame.wet.x) * flat;
    return vec2<f32>(w * mix(0.35, 1.0, up), puddle);
}

/// Raindrops landing on standing water: in each 22 cm cell a drop hits at a
/// random spot and time and a ring runs out from it, fading as it spreads.
/// xy the water surface's slope (world x, z), z the splash at impact.
fn rain_ripples(p: vec2<f32>, t: f32) -> vec3<f32> {
    var out = vec3<f32>(0.0);
    for (var layer = 0; layer < 2; layer = layer + 1) {
        let fl = f32(layer);
        let q = p / 0.22 + vec2<f32>(fl * 0.37, fl * 0.71);
        let cell = floor(q);
        for (var j = 0; j < 4; j = j + 1) {
            // this cell and its neighbours towards the point: rings cross cells
            let c = cell + vec2<f32>(f32(j & 1), f32(j >> 1)) - vec2<f32>(select(0.0, 1.0, fract(q.x) < 0.5), select(0.0, 1.0, fract(q.y) < 0.5));
            let h1 = fract(sin(dot(c, vec2<f32>(127.1, 311.7)) + fl * 17.0) * 43758.5453);
            let h2 = fract(sin(dot(c, vec2<f32>(269.5, 183.3)) + fl * 31.0) * 43758.5453);
            let h3 = fract(sin(dot(c, vec2<f32>(419.2, 371.9)) + fl * 7.0) * 43758.5453);
            let centre = c + vec2<f32>(0.15 + 0.7 * h1, 0.15 + 0.7 * h2);
            // about one drop a second per cell, each ring living 0.8 s
            let life = fract(t * (0.9 + 0.4 * h3) + h3);
            let d = q - centre;
            let r = length(d) * 0.22;
            let radius = life * 0.11;
            let x = (r - radius) / 0.012;
            let fade = (1.0 - life) * (1.0 - life);
            // one crest and trough, travelling out
            let wave = -x * exp(-x * x) * fade;
            if (r > 1e-4) {
                out = vec3<f32>(out.xy + d / max(length(d), 1e-4) * wave, out.z);
            }
            out.z = max(out.z, smoothstep(0.012, 0.0, r) * smoothstep(0.12, 0.0, life));
        }
    }
    return out;
}

// Screen-space reflections (ssr.rs): the environment reflection this pixel
// added (output units) + the weight it had, and the octahedral normal,
// roughness and kind, for the SSR pass to swap the sky for what is on screen
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
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return shade(in, front);
}

// SSR, alpha-blended: covers what is behind by its alpha (ssr.rs cover)
@fragment
fn fs_main_cover(in: VsOut, @builtin(front_facing) front: bool) -> SsrOut {
    let c = shade(in, front);
    return SsrOut(c, vec4<f32>(c.a), vec4<f32>(0.0));
}

@fragment
fn fs_main_ssr(in: VsOut, @builtin(front_facing) front: bool) -> SsrOut {
    let c = shade(in, front);
    return SsrOut(c, g_env, g_geo);
}

fn shade(in: VsOut, front: bool) -> vec4<f32> {
    if (in.inst.y > 0.0 && lod_dither(in.clip.xy, in.inst.y)) {
        discard;
    }
    let kind = mat.base.w;
    // Grass fades out with distance, the way the game thins its carpet. A
    // dithered discard keeps it in the opaque pass with depth writes.
    if (mat.flags.w > 0.5) {
        let f = clamp((in.inst.x - frame.misc.z) / max(frame.misc.w - frame.misc.z, 1.0), 0.0, 1.0);
        if (f > 0.0 && dither(in.clip.xy) < f) {
            discard;
        }
    }
    // MaterialData.cullMode (extra.w bits 4 / 5): a single-sided surface
    // seen from the side the game culls is not there
    let cull = u32(mat.extra.w + 0.5) & 48u;
    if (cull != 0u) {
        let outer = front == (in.inst.z >= 0.0);
        if ((cull == 16u && !outer) || (cull == 32u && outer)) {
            discard;
        }
    }
    var ng = normalize(in.normal);
    // doNotInvertNormalOnBackside (extra.w bit 2): the back face keeps the
    // authored normal
    let keep_n = (u32(mat.extra.w) & 4u) != 0u;
    if (!front && !keep_n) {
        ng = -ng;
    }
    var s = surface(in, ng);
    // rain: darker, glossier, puddles on the flat (wet_amount)
    let wa = wet_amount(in.world, ng, kind);
    if (wa.x > 0.0) {
        s.albedo = s.albedo * mix(1.0, mix(0.85, 0.5, s.rough), wa.x);
        s.rough = mix(s.rough, mix(0.32, 0.12, s.rough), wa.x);
        s.albedo = s.albedo * mix(1.0, 0.75, wa.y);
        s.rough = mix(s.rough, 0.03, wa.y);
        s.n_ts = mix(s.n_ts, vec3<f32>(0.0, 0.0, 1.0), wa.y);
        // falling rain rings the puddles (and, barely, the wet film), close
        // to the eye only: further out the rings are below a pixel
        let near = 1.0 - smoothstep(12.0, 40.0, length(frame.eye.xyz - in.world));
        let k = frame.wet.y * (wa.y + 0.12 * wa.x) * near;
        if (k > 0.001) {
            let rp = rain_ripples(in.world.xz, frame.wet.z);
            s.n_world = s.n_world + vec3<f32>(rp.x, 0.0, rp.y) * 0.9 * k;
            s.albedo = s.albedo + vec3<f32>(rp.z * 0.25 * k);
        }
    }
    // Alpha test. Under MSAA the coverage comes from a SHARPENED alpha
    // (Golus' trick): a one-pixel-wide ramp around the cutoff, so edges
    // antialias without the card turning soft or thinning out with mips.
    var cov = 1.0;
    if (mat.surf.w > 0.0) {
        if (frame.debug.y > 0.5) {
            cov = clamp((s.alpha - mat.surf.w) / max(fwidth(s.alpha), 1e-4) + 0.5, 0.0, 1.0);
            if (cov <= 0.0) {
                discard;
            }
        } else if (cutout_drop(in.clip.xy, s.alpha)) {
            discard;
        }
    }
    // tangent frame
    var n = ng;
    // (a tangent along the normal leaves nothing after Gram-Schmidt:
    // normalize(0) is NaN, a black pixel that TAA spreads)
    let tp = in.tangent.xyz - ng * dot(ng, in.tangent.xyz);
    if (dot(tp, tp) > 1e-10 && abs(s.n_ts.z) < 0.9999) {
        let t = normalize(tp);
        // the game's frame: B = cross(T, N) * w, green along +B
        let b = cross(t, ng) * select(-1.0, 1.0, in.tangent.w >= 0.0) * BT_GAME;
        let nt = vec3<f32>(s.n_ts.xy * mat.surf.z, s.n_ts.z);
        n = normalize(t * nt.x + b * nt.y + ng * nt.z);
    }
    if (dot(s.n_world, s.n_world) > 0.0) {
        n = normalize(n + s.n_world);
    }
    let v = normalize(frame.eye.xyz - in.world);
    if (dot(n, v) < 0.0 && kind > 2.5 && kind < 3.5 && !keep_n) {
        // foliage cards: light the side we are looking at
        n = reflect(n, ng);
    }
    // GSAA on the final normal, wetness included (Gsaa_Variance 0.5 and
    // Threshold 1 by default): wet and glossy tarmac stops sparkling
    s.rough = gsaa_rough(s.rough, sq_deriv(n) * 0.5, 1.0);
    let ks = spec_aa(sq_deriv(ng));
    let ks_ibl = clamp(ks + 0.75, 0.0, 1.0);

    g_geo = vec4<f32>(ssr_oct(n), s.rough, 1.0 / 3.0);
    var color: vec3<f32>;
    if (frame.misc.y < 0.5) {
        color = s.albedo;
    } else {
        // BRDF as SPEC_evo_shading 2.1-2.3: Frostbite-normalised Burley,
        // GGX + height-correlated Smith, F0 = 0.16 refl^2.
        let l = frame.sun.xyz;
        let sun_e = frame.sun_color.rgb * frame.sun.w;
        let r = s.rough;
        // water on the surface: at least its F0 of 0.02
        let refl = max(mat.surf.y, 0.354 * wa.x);
        let f0 = vec3<f32>(0.16 * refl * refl);
        let nov = clamp(dot(n, v), 1e-4, 1.0);
        // light wrap (foliage): NoL' = (NoL + w) / sqrt(1 + w)... as the game
        // writes it, w = LightWrap
        let wrap = mat.extra.y;
        let nol_raw = dot(n, l);
        let nol = max(clamp((clamp(nol_raw, 1e-4, 1.0) + wrap) / sqrt(1.0 + wrap), 0.0, 1.0), 1e-4)
                * select(1.0, 0.0, nol_raw < -wrap);
        let h = normalize(v + l);
        let noh = max(dot(n, h), 0.0);
        let loh = max(dot(l, h), 0.0);
        let voh = loh;
        let fd90 = 0.5 * r + 2.0 * r * loh * loh;
        let fd = (1.0 / PI - 0.1075086 * r)
               * (1.0 + (fd90 - 1.0) * pow(1.0 - nol, 5.0))
               * (1.0 + (fd90 - 1.0) * pow(1.0 - nov, 5.0));
        let a2 = r * r * r * r;
        let dd = noh * noh * (a2 - 1.0) + 1.0;
        let d_ggx = a2 / max(PI * dd * dd, 1e-7);
        let vis = 0.5 / max(nol * sqrt(nov * nov * (1.0 - a2) + a2)
                          + nov * sqrt(nol * nol * (1.0 - a2) + a2), 1e-5);
        let f = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - voh, 5.0);
        // the map's shadows, and the cumulus deck's on the ground
        let shadow = sun_shadow(in.world, ng) * cloud_shadow_at(in.world);
        // AO micro-shadow on the sun term
        let ms = mix(1.0, pow(clamp(nol / max(sqrt(1.0 - s.ao), 0.001), 0.0, 1.0), 2.0), 0.5);
        color = (s.albedo * fd + d_ggx * vis * f * mat.extra.z * ks) * nol * sun_e * shadow * ms;
        // Grass subsurface (lift 0350dcb0): the sun through the blades, a
        // lobe exp(power (v.-l - 1)) round the sun seen through them, half
        // of it, over pi, in the sun's shadow and the blade's AO
        if (dot(s.sss, s.sss) > 0.0) {
            let back = exp(mat.layer_rough.w * (clamp(dot(v, -l), 0.0, 1.0) - 1.0));
            // Subsurface_ShadowFactor: light still crosses a shaded canopy
            let sh = max(shadow, mat.layer_rough.z * 0.55 * s.ao);
            color += s.sss * sun_e * (0.5 / PI) * back * sh * s.ao;
        }

        // ambient: irradiance volumes, or a sky/ground hemisphere without them
        var irr = iv_irradiance(in.world, n);
        if (irr.x < 0.0) {
            let sky_up = sky_radiance(vec3<f32>(0.0, 1.0, 0.0));
            let gnd = vec3<f32>(0.10, 0.11, 0.07);
            irr = mix(gnd, sky_up * 0.55, clamp(n.y * 0.5 + 0.5, 0.0, 1.0)) * 0.06;
        }
        // x1.8: the shade side of a sunlit scene was ~11:1 darker than the lit side
        // (clear sky is ~5-6:1 on a level surface) and shaded tarmac went black
        // next to the game's own previews, where it keeps its texture
        let amb = irr * frame.misc.x * 1.8;
        let fss = env_brdf(f0, r, nov);
        // Jimenez multi-bounce AO on the diffuse
        let ka = 2.0404 * s.albedo - 0.3324;
        let kb = -4.7951 * s.albedo + 0.6417;
        let kc = 2.7552 * s.albedo + 0.6903;
        let mb = max(vec3<f32>(s.ao), ((s.ao * ka + kb) * s.ao + kc) * s.ao);
        // shadowed ground loses a little of its sky too (IblDiffuseShadowFloor)
        let diff_shadow = mix(0.9, 1.0, shadow);
        color += s.albedo * amb * (vec3<f32>(1.0) - fss) * mb * diff_shadow;
        // ambient specular: the sky, occluded by how much sky the volume sees
        let rr = reflect(-v, n);
        let rd = normalize(vec3<f32>(rr.x, max(rr.y, 0.02), rr.z));
        let sky_occ = clamp(dot(amb, vec3<f32>(0.333)) / max(frame.haze.w * 0.45, 1e-3), 0.1, 1.0);
        let so = clamp(pow(nov + s.ao, exp2(-16.0 * r * r - 1.0)) - 1.0 + s.ao, 0.0, 1.0);
        // A rough surface sees a blurred sky, not the sharp blue overhead:
        // fade toward the hemisphere average (mostly the pale horizon band)
        // or every matte road and roof picks up a blue cast.
        // (constant hemisphere average: one sky evaluation per pixel, not three)
        let sky_avg = vec3<f32>(0.52, 0.55, 0.58) * frame.haze.w;
        let rk = clamp(r * 1.2, 0.0, 1.0);
        var env_c = sky_avg;
        if (rk < 0.98) {
            var sky_d = sky_radiance(rd);
            if (frame.atm.w > 0.5) {
                // glass, wet and polished surfaces mirror the cloud deck too
                let cl = textureSampleLevel(t_pano_cloud, s_cloud, pano_uv(rd, f32(textureDimensions(t_pano_cloud).y)), 0.0);
                sky_d = sky_d * cl.a + cl.rgb;
            }
            env_c = mix(sky_d, sky_avg, rk);
        }
        // the sun's shadow blocks the sun, hardly the sky: shaded asphalt still
        // shows the sky's sheen at a grazing view
        color += env_c * fss * so * sky_occ * mix(0.9, 1.0, shadow) * ks_ibl;
        // what SSR swaps out; a hit on screen is occluded by AO only (the sky
        // occlusion is what the hit replaces)
        g_env = vec4<f32>(env_c * fss * so * sky_occ * mix(0.9, 1.0, shadow) * ks_ibl,
                          dot(fss, vec3<f32>(0.2126, 0.7152, 0.0722)) * so * ks_ibl);
        // night: the lamps' baked bounce (night irradiance x its curve) and
        // the lamps themselves
        if (frame.lamps.w > 0.0) {
            let ni = iv_night(in.world, n);
            if (ni.x >= 0.0) {
                color += s.albedo * ni * frame.lamps.w * mb;
            }
        }
        color += lamp_light(in.world, n, v, s.albedo, f0, r);
        color += mat.emissive.rgb * select(1.0, dot(s.albedo, vec3<f32>(0.333)), mat.emissive.w > 0.5)
               * frame.debug.w;
    }

    let dbg = i32(frame.debug.x + 0.5);
    if (dbg > 0) {
        // raw values straight to the target (sRGB-encoded by the hardware)
        var o = s.albedo;
        if (dbg == 2) { o = max(iv_irradiance(in.world, n), vec3<f32>(0.0)) * 10.0; }
        if (dbg == 3) { o = vec3<f32>(s.ao); }
        if (dbg == 4) { o = n * 0.5 + 0.5; }
        if (dbg == 5) { o = vec3<f32>(sun_shadow(in.world, ng)); }
        if (dbg == 6) { o = s.base_col; }
        // the dry line (red) over the wetness the shading uses (green)
        if (dbg == 7) { o = vec3<f32>(max(dry_line(in.world), 0.0), frame.wet.x, max(-dry_line(in.world), 0.0)); }
        // LOD crossfade: red = this instance is dithering out by that much
        if (dbg == 8) { o = mix(s.albedo * 0.5, vec3<f32>(1.0, 0.0, 0.0), select(0.0, 0.3 + 0.7 * in.inst.y, in.inst.y > 0.0)); }
        return vec4<f32>(o, 1.0);
    }
    let hz = haze(in.world);
    color = color * hz.a + hz.rgb * (1.0 - hz.a);
    g_env = vec4<f32>(tonemap(g_env.rgb * hz.a * frame.sun_color.w), g_env.a * hz.a);
    // alpha-blended materials (glass, blendMode 1 nets and fences) carry
    // their coverage in alpha; everything else is opaque
    // (the opaque pipeline has no blending, so only blended draws see it)
    var alpha = clamp(s.alpha, 0.0, 1.0);
    if (mat.surf.w > 0.0) {
        alpha = cov;
    }
    return vec4<f32>(tonemap(color * frame.sun_color.w), alpha);
}

// ----------------------------------------------------------------- sky ----

struct SkyOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) i: u32) -> SkyOut {
    var o: SkyOut;
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    o.clip = vec4<f32>(p, 1.0, 1.0);
    o.ndc = p;
    return o;
}

// ------------------------------------------------------------- clouds ----
//
// A raymarched cloud deck in the game's own noise (content/weather/
// cloud_noise.dds) with its default settings (clouds_settings/setting_1).
// ⚠ The noise channels' roles and the density / extinction scale are not
// decoded from the game's cloud shader (strict SKY4): base shape from R,
// erosion from G/B at the erode detail scale, a Nubis-style remap. The
// 512x512x64 volume is taken to span 8 km x 8 km x the deck's thickness.

// cloud_tex / cloud_density / cloud_shadow: clouds_common.wgsl

/// cloud_shadow from the map where it covers `w`, else marched here.
fn cloud_shadow_at(w: vec3<f32>) -> f32 {
    if (frame.cmap.w > 0.5) {
        let uv = (w.xz - frame.cmap.xy) / frame.cmap.z;
        if (all(uv > vec2<f32>(0.002)) && all(uv < vec2<f32>(0.998))) {
            return textureSampleLevel(t_cmap, s_clamp, uv, 0.0).r;
        }
    }
    return cloud_shadow(w);
}

/// One texel of the cloud shadow map: the march from the ground (the
/// track's ground height) at that texel's centre.
@fragment
fn fs_cloud_map(in: SkyOut) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(in.ndc.x * 0.5 + 0.5, 0.5 - in.ndc.y * 0.5);
    let xz = frame.cmap.xy + uv * frame.cmap.z;
    return vec4<f32>(cloud_shadow(vec3<f32>(xz.x, frame.haze.z, xz.y)), 0.0, 0.0, 1.0);
}

/// (in-scattered colour, transmittance) of the deck along `d` from the eye.
fn clouds(d: vec3<f32>, px: vec2<f32>, steps_in: i32) -> vec4<f32> {
    if (frame.clouds3.w < 0.5 || frame.clouds.z <= 0.0 || d.y < 0.02) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let e = frame.eye.xyz;
    let bottom = frame.haze.z + frame.clouds.x;
    let top = bottom + frame.clouds.y;
    if (e.y > top) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let t0 = max((bottom - e.y) / d.y, 0.0);
    let t1 = min((top - e.y) / d.y, t0 + 30000.0);
    // EVOVIEW_CLOUD_STEPS (frame.haze2.y when set): more steps for
    // single-frame captures, where TAA is not there to integrate the dither
    let mode = floor(frame.haze2.y / 1000.0);
    let steps = frame.haze2.y - mode * 1000.0;
    var n = select(40, i32(steps), steps > 0.5);
    if (steps_in > 0) {
        n = steps_in;
    }
    let dt = (t1 - t0) / f32(n);
    // per-pixel, per-frame start offset: TAA integrates the banding away
    // IGN stepped 5.588 px a frame (Jimenez's temporal IGN). The old
    // golden-ratio shift of the SAME pattern kept IGN's half-step between
    // horizontal neighbours every frame, so TAA's residual was a fixed
    // checkerboard on every cloud edge. EVOVIEW_CLOUD_NOISE=1 brings it back.
    var jit = dither(px + 5.588238 * f32(u32(frame.haze2.w) & 63u));
    if (mode > 1.5) {
        // 2: white noise per pixel and frame - DLSS keeps IGN's lattice as
        // a checker on cloud edges at 0.667 scale (the default under DLSS)
        let h = vec2<f32>(px) + vec2<f32>(17.13, 3.71) * f32(u32(frame.haze2.w) & 1023u);
        jit = fract(sin(dot(h, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    } else if (mode > 0.5) {
        jit = fract(dither(px) + frame.haze2.z);
    }
    let sun = frame.sun.xyz;
    let sun_rad = frame.sun_color.rgb * frame.sun.w * frame.sky[0].w;
    let mu = dot(d, sun);
    let amb_top = sky_radiance(vec3<f32>(0.0, 1.0, 0.0));
    let amb_bot = sky_radiance(normalize(vec3<f32>(1.0, 0.15, 0.0))) * 0.35;
    // extinction per metre at density 1 (from globalDensity)
    let sigma = frame.clouds.w * 0.16 * frame.atm4.x;
    var tr = 1.0;
    var col = vec3<f32>(0.0);
    for (var i = 0; i < n; i = i + 1) {
        let t = t0 + (f32(i) + jit) * dt;
        let p = e + d * t;
        let h = (p.y - bottom) / max(frame.clouds.y, 1.0);
        let den = cloud_density(p, h, false);
        if (den > 0.001) {
            // optical depth toward the sun: 4 steps through the deck
            var od = 0.0;
            for (var k = 1; k <= 4; k = k + 1) {
                let q = p + sun * (f32(k) * 90.0 * f32(k));
                let hq = (q.y - bottom) / max(frame.clouds.y, 1.0);
                if (hq > 1.0) {
                    break;
                }
                od += cloud_density(q, hq, true) * 90.0 * f32(k);
            }
            // Multiple scattering as octaves (Wrenninge): each order sees
            // less extinction and a flatter phase. A soft powder term darkens
            // thin edges a little. ⚠ Was single scattering x a powder factor
            // (0.034 of the sun at best): every cloud came out grey and see-
            // through, ~130/255 where the game's track previews show ~220-245
            // on the lit side. sun x 2.5 matches the previews' highlights.
            var ms = 0.0;
            var ak = 1.0;
            var bk = 1.0;
            var gk = 1.0;
            for (var o = 0; o < 4; o = o + 1) {
                let ph = mix(hg(mu, -0.15 * gk), hg(mu, 0.7 * gk), 0.6);
                ms += ak * exp(-od * sigma * bk * frame.atm3.z) * ph;
                ak *= 0.6;
                bk *= select(0.3, frame.clouds5.w, frame.clouds5.w > 0.0);
                gk *= 0.5;
            }
            let powder = mix(1.0, 1.0 - exp(-den * sigma * 300.0), frame.atm3.w);
            let amb = mix(amb_bot, amb_top, h) * frame.atm3.y;
            let s = den * sigma;
            let step_tr = exp(-s * dt);
            let lum = sun_rad * ms * powder * frame.atm3.x + amb;
            // energy-conserving integration (Hillaire)
            col += tr * lum * (1.0 - step_tr);
            tr *= step_tr;
            if (tr < 0.02) {
                break;
            }
        }
    }
    // far clouds fade into the haze toward the horizon
    let fade = exp(-max(t0 - 4000.0, 0.0) * frame.haze.x * 0.35);
    return vec4<f32>(col * fade, mix(1.0, tr, fade));
}

/// The cirrus deck along `d`: (colour, opacity). A flat layer high above the
/// cumulus, in the game's own cirrus texture, drifting with the wind.
/// The cirrus layer's base uv along `d` (unbounded where the ray never
/// reaches it; cirrus() rejects those).
fn cirrus_uv(d: vec3<f32>) -> vec2<f32> {
    let e = frame.eye.xyz;
    let t = (frame.haze.z + frame.clouds4.y - e.y) / max(d.y, 0.01);
    let p = e + d * t;
    let wind = frame.clouds3.yz * frame.clouds3.x * 2.5;
    return (p.xz + wind) / 24000.0;
}

/// `g`: the screen derivatives of cirrus_uv (xy along x, zw along y),
/// taken by the caller in uniform control flow. ⚠ It sampled level 0
/// only: toward the horizon each pixel covers many texels, and the
/// streaky texture aliased into thin hard lines across the whole sky.
fn cirrus(d: vec3<f32>, g: vec4<f32>) -> vec4<f32> {
    if (frame.clouds4.z < 0.5 || frame.clouds4.x <= 0.0 || d.y < 0.01) {
        return vec4<f32>(0.0);
    }
    let e = frame.eye.xyz;
    let t = (frame.haze.z + frame.clouds4.y - e.y) / d.y;
    if (t <= 0.0) {
        return vec4<f32>(0.0);
    }
    let p = e + d * t;
    let wind = frame.clouds3.yz * frame.clouds3.x * 2.5;
    // one 1024 texture over 24 km, a second octave over 7 km
    let uv = (p.xz + wind) / 24000.0;
    let a = textureSampleGrad(t_cirrus, s_cloud, uv, g.xy, g.zw);
    let b = textureSampleGrad(t_cirrus, s_cloud, uv * 3.4 + vec2<f32>(0.37, 0.61), g.xy * 3.4, g.zw * 3.4);
    let m = clamp(dot(a.rgb, vec3<f32>(0.65, 0.0, 0.35)) * 1.15 - 0.08, 0.0, 1.0)
          * (0.65 + 0.35 * b.r);
    // thin toward the horizon (long path through haze, and foreshortening)
    let fade = smoothstep(0.01, 0.12, d.y);
    let alpha = m * frame.clouds4.x * fade * 0.85;
    let sun_rad = frame.sun_color.rgb * frame.sun.w * frame.sky[0].w;
    let mu = dot(d, frame.sun.xyz);
    // ice crystals: strong forward scatter, bright where the sun is behind
    let phase = mix(hg(mu, 0.2), hg(mu, 0.85), 0.5) * 4.0 * PI;
    let col = sun_rad * phase * 0.06 + sky_radiance(vec3<f32>(0.0, 1.0, 0.0)) * 1.3;
    return vec4<f32>(col, alpha);
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    // ⚠ Not z = 1: with a far plane kilometres out (or at infinity) the
    // unprojected far point is inf/NaN and the whole sky goes white.
    let a = frame.inv_view_proj * vec4<f32>(in.ndc, 0.5, 1.0);
    let b = frame.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let d = normalize(a.xyz / a.w - b.xyz / b.w);
    var c = sky_radiance(d);
    // the sun disc, 0.53 degrees, with a little limb darkening
    let mu = dot(d, frame.sun.xyz);
    let disc = smoothstep(0.99995, 0.999975, mu);
    c += frame.sun_color.rgb * frame.sun.w * disc * 8.0;
    // cirrus sits above the cumulus: composite it first
    let cuv = cirrus_uv(d);
    // a level softer than the footprint: wisps, not texel detail
    let ci = cirrus(d, vec4<f32>(dpdx(cuv), dpdy(cuv)) * 2.0);
    c = mix(c, ci.rgb, ci.a);
    let cl = clouds_at(d, in.clip.xy);
    c = c * cl.a + cl.rgb;
    return vec4<f32>(tonemap(c * frame.sun_color.w), 1.0);
}
