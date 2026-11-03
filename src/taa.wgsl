// Temporal anti-aliasing: the resolved, jittered frame blended with last
// frame's history, reprojected through the depth buffer (the scenery is
// static; moving cars lean on the neighbourhood clamp instead of motion
// vectors). Output goes to the screen and to the next history.

struct U {
    // current unjittered clip -> world
    inv_vp: mat4x4<f32>,
    // previous unjittered world -> clip
    prev_vp: mat4x4<f32>,
    // x 1 = no usable history (first frame, resize, cut), y blend weight of
    // the current frame
    ctl: vec4<f32>,
    // w, h, 1/w, 1/h
    size: vec4<f32>,
    // the driven car as a box (taa.rs set_object)
    obj_inv: mat4x4<f32>,
    obj_prev: mat4x4<f32>,
    obj_half: vec4<f32>,
    vp: mat4x4<f32>,
    // xyz eye; ambient occlusion: x radius (m), y strength, z frame, w on
    eye: vec4<f32>,
    ao: vec4<f32>,
    // volumetric fog: the sun's shadow cascades, sun, density
    lvp_near: mat4x4<f32>,
    lvp_far: mat4x4<f32>,
    sun: vec4<f32>,
    sun_col: vec4<f32>,
    fog: vec4<f32>,
    amb: vec4<f32>,
    wheels: array<vec4<f32>, 4>,
    // the render size (taa.rs render scale): w, h, 1/w, 1/h
    rsize: vec4<f32>,
    // xy this frame's jitter in render pixels (the scene moved by +x, -y
    // on screen, so render texel k saw the unjittered point k + 0.5 - xy),
    // z 1 = upsampling (render size below the screen's), w 1 = t_mv holds
    // this frame's motion vectors (motion.rs)
    jit: vec4<f32>,
    // the screen rectangle (uv min xy, max zw) the driven car, its road
    // band and its shadow can touch: the car tests run only inside it
    obj_rect: vec4<f32>,
    // x the current frame's weight on sky pixels (EVOVIEW_TAA_SKY)
    sky: vec4<f32>,
    // the middle cascade; per cascade (x near, y far, z mid) the colour
    // pass's depth bias and normal offset m; sbias.w 1 = mid on
    lvp_mid: mat4x4<f32>,
    sbias: vec4<f32>,
    snofs: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_cur: texture_2d<f32>;
@group(0) @binding(2) var t_hist: texture_2d<f32>;
@group(0) @binding(3) var t_depth: DEPTH_TEXTURE;
@group(0) @binding(4) var s_lin: sampler;
// ambient-occlusion history (r occlusion, g view depth), at fs_aofog's size
@group(0) @binding(5) var t_aoh: texture_2d<f32>;
@group(0) @binding(6) var t_shadow: texture_depth_2d_array;
@group(0) @binding(7) var s_shadow: sampler_comparison;
// volumetric fog history (rgb in-scatter, a transmittance)
@group(0) @binding(8) var t_fogh: texture_2d<f32>;
// this frame's occlusion and fog, accumulated (fs_aofog)
@group(0) @binding(9) var t_aoc: texture_2d<f32>;
@group(0) @binding(10) var t_fogc: texture_2d<f32>;
// per-pixel motion vectors at the render size (motion.wgsl): xy last frame's
// position - this frame's in render pixels, z last frame's view depth,
// w 1 = a moving object
@group(0) @binding(11) var t_mv: texture_2d<f32>;

fn mv_at(q: vec2<i32>) -> vec4<f32> {
    let m = vec2<i32>(i32(u.rsize.x) - 1, i32(u.rsize.y) - 1);
    return textureLoad(t_mv, clamp(q, vec2<i32>(0), m), 0);
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    var o: VsOut;
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return o;
}

struct FsOut {
    @location(0) screen: vec4<f32>,
    @location(1) history: vec4<f32>,
};

fn to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(c, vec3<f32>(0.25, 0.5, 0.25)),
                     dot(c, vec3<f32>(0.5, 0.0, -0.5)),
                     dot(c, vec3<f32>(-0.25, 0.5, -0.25)));
}

fn from_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// p in render pixels
fn load_cur(p: vec2<i32>) -> vec3<f32> {
    let m = vec2<i32>(i32(u.rsize.x) - 1, i32(u.rsize.y) - 1);
    return textureLoad(t_cur, clamp(p, vec2<i32>(0), m), 0).rgb;
}

/// Catmull-Rom history fetch from 9 bilinear taps (the full 4x4 kernel).
/// The 5-tap shortcut drops the corners and is softer; resampled every
/// frame, that softness compounds in whatever keeps history longest -
/// distant scenery creeping across the screen while driving.
fn hist_cr(uv: vec2<f32>) -> vec3<f32> {
    let pos = uv * u.size.xy;
    let c = floor(pos - 0.5) + 0.5;
    let f = pos - c;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let t0 = (c - 1.0) * u.size.zw;
    let t3 = (c + 2.0) * u.size.zw;
    let t12 = (c + w2 / w12) * u.size.zw;
    var xs = array<f32, 3>(t0.x, t12.x, t3.x);
    var ys = array<f32, 3>(t0.y, t12.y, t3.y);
    var wx = array<f32, 3>(w0.x, w12.x, w3.x);
    var wy = array<f32, 3>(w0.y, w12.y, w3.y);
    var r = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var j = 0; j < 3; j = j + 1) {
        for (var i = 0; i < 3; i = i + 1) {
            let wt = wx[i] * wy[j];
            r += textureSampleLevel(t_hist, s_lin, vec2<f32>(xs[i], ys[j]), 0.0).rgb * wt;
            wsum += wt;
        }
    }
    return max(r / max(wsum, 1e-4), vec3<f32>(0.0));
}

// q in render pixels
fn depth_r(q: vec2<i32>) -> f32 {
    let m = vec2<i32>(i32(u.rsize.x) - 1, i32(u.rsize.y) - 1);
    return textureLoad(t_depth, clamp(q, vec2<i32>(0), m), 0);
}

// the render texel under screen pixel q
fn to_render(q: vec2<i32>) -> vec2<i32> {
    return vec2<i32>((vec2<f32>(q) + 0.5) * u.rsize.xy * u.size.zw);
}

fn in_obj(wp: vec3<f32>) -> bool {
    let l = u.obj_inv * vec4<f32>(wp, 1.0);
    return all(abs(l.xyz) < u.obj_half.xyz);
}

// q in render pixels
fn world_r(q: vec2<i32>) -> vec3<f32> {
    let uv = (vec2<f32>(q) + 0.5) * u.rsize.zw;
    let w = u.inv_vp * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, min(depth_r(q), 0.9999999), 1.0);
    return w.xyz / w.w;
}

// q in screen pixels
fn world_at(q: vec2<i32>) -> vec3<f32> {
    let uv = (vec2<f32>(q) + 0.5) * u.size.zw;
    let w = u.inv_vp * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, min(depth_r(to_render(q)), 0.9999999), 1.0);
    return w.xyz / w.w;
}

/// The darkening this pixel gets from ambient occlusion and contact shadows,
/// as one multiplier (the history below accumulates it).
///
/// AO is the game's (compute 04732820, a half-res pass): a normal from the
/// neighbouring depths, AO_SAMPLES points on a golden-angle disk whose
/// radius is EVOVIEW_AO_RADIUS metres (growing to 3x by 200 m) on screen,
/// each occluding by max(0, n.v/|v| - bias) with a (1 - |v|^2/r^2)
/// falloff - so a car never darkens the far background it stands in front
/// of (no halo) - then (1 - mean)^AO_POW.
/// The game applies it to the ambient light only; here it scales the whole
/// colour, so where the sun reaches (the shadow map, N.L, and not under
/// the driven car) only 1 - AO_SUNLIT of it applies - the share of a sunlit
/// pixel that is sky light. Contact shadows (contact_shadow) take the sun
/// part away where the map says lit but the depth buffer says blocked.
fn ssao(q: vec2<i32>, wp: vec3<f32>) -> f32 {
    let dist = length(wp - u.eye.xyz);
    if (u.ao.w < 0.5 || dist > 300.0) {
        return 1.0;
    }
    let r = world_r(q + vec2<i32>(1, 0)) - wp;
    let l = wp - world_r(q - vec2<i32>(1, 0));
    let dn = world_r(q + vec2<i32>(0, 1)) - wp;
    let up = wp - world_r(q - vec2<i32>(0, 1));
    let dx = select(l, r, dot(r, r) < dot(l, l));
    let dy = select(up, dn, dot(dn, dn) < dot(up, up));
    var n = normalize(cross(dy, dx));
    if (dot(n, u.eye.xyz - wp) < 0.0) {
        n = -n;
    }
    // EVOVIEW_AO=8: debug - the depth buffer's view depth over this point's
    // own (should be ~0.5 = equal); 7: the normal's y
    if (u.ao.y > 6.5 && u.ao.y < 7.5) {
        return n.y * 0.5 + 0.5;
    }
    if (u.ao.y > 7.5 && u.ao.y < 8.5) {
        let c0 = u.vp * vec4<f32>(wp, 1.0);
        let suv0 = (c0.xy / c0.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
        let s0 = (u.vp * vec4<f32>(world_r(vec2<i32>(suv0 * u.rsize.xy)), 1.0)).w;
        return clamp(s0 / c0.w * 0.5, 0.0, 1.0);
    }
    // interleaved gradient noise, shifted each frame
    let fq = vec2<f32>(q) + vec2<f32>(u.ao.z * 5.588238, u.ao.z * 2.7);
    let noise = fract(52.9829189 * fract(dot(fq, vec2<f32>(0.06711056, 0.00583715))));
    let c0 = u.vp * vec4<f32>(wp, 1.0);
    let rad = u.ao.x * mix(1.0, 3.0, clamp(c0.w / 200.0, 0.0, 1.0));
    // the disk on screen, in render pixels (the game: P00/2 r / z, <= 0.15 uv)
    let p00 = length(vec3<f32>(u.vp[0].x, u.vp[1].x, u.vp[2].x));
    let rpx = min(0.5 * p00 * rad / max(c0.w, 0.01), 0.15) * u.rsize.x;
    var occ = 0.0;
    let ns = AO_SAMPLES;
    let rot = noise * 6.2831853;
    for (var i = 0; i < ns; i = i + 1) {
        let fi = f32(i);
        let ang = fi * 2.399963 + rot;
        let rr = sqrt((fi + 0.5) / f32(ns));
        let o = vec2<f32>(cos(ang), sin(ang)) * rr * rpx;
        let sw = world_r(q + vec2<i32>(round(o)));
        let v = sw - wp;
        let d2 = dot(v, v);
        occ += max(0.0, dot(n, v) * inverseSqrt(d2 + 1e-4) - AO_BIAS)
             * clamp(1.0 - d2 / (rad * rad), 0.0, 1.0);
    }
    var ao = pow(clamp(1.0 - occ / f32(ns), 0.0, 1.0), AO_POW);
    // fade out with distance (far pixels are coarse)
    ao = mix(ao, 1.0, smoothstep(150.0, 300.0, dist));
    let lit = sun_lit(wp, n);
    var m = mix(1.0, ao, 1.0 - AO_SUNLIT * lit);
    // not on the driven car itself: its self-shadow is the hero map's at
    // ~6 mm, and the march over its curved panels (rays passing just behind
    // a badge or a mirror) only smudged the paint
    if (lit > 0.0 && !(u.obj_half.w > 0.5 && in_obj(wp))) {
        m = m * mix(1.0, contact_shadow(q, wp, n, noise, c0.w), AO_SUNLIT * lit);
    }
    return m;
}

/// How sunlit `wp` (normal `n`) is: N.L, the near / far shadow map, and not
/// in the driven car's shadow (the car casts into its own hero map, which
/// this pass does not read: the sun ray crossing the car's box stands in).
fn sun_lit(wp: vec3<f32>, n: vec3<f32>) -> f32 {
    if (u.sun.w <= 0.0) {
        return 0.0;
    }
    let l = normalize(u.sun.xyz);
    let ndl = dot(n, l);
    if (ndl <= 0.0) {
        return 0.0;
    }
    let lp = (u.obj_inv * vec4<f32>(wp, 1.0)).xyz;
    // (the car's own surfaces: its self-shadow is the hero map's, unread here)
    if (u.obj_half.w > 0.5 && any(abs(lp) > u.obj_half.xyz)) {
        let ls = normalize((u.obj_inv * vec4<f32>(l, 0.0)).xyz) * 8.0;
        let hb = u.obj_half.xyz + vec3<f32>(0.1, 0.1, 0.1);
        let sinv = 1.0 / select(ls, vec3<f32>(1e-6), abs(ls) < vec3<f32>(1e-6));
        let s0 = (-hb - lp) * sinv;
        let s1 = (hb - lp) * sinv;
        let sn = min(s0, s1);
        let sf = max(s0, s1);
        if (max(max(sn.x, sn.y), max(sn.z, 0.0)) < min(min(sf.x, sf.y), min(sf.z, 1.0))) {
            return 0.0;
        }
    }
    if (SV_OLD) {
        return sun_vis_old(wp + n * 0.08) * smoothstep(0.0, 0.15, ndl);
    }
    return sun_vis(wp, n, ndl) * smoothstep(0.0, 0.15, ndl);
}

/// Screen-space contact shadow (the game runs Bend's screen-space shadows,
/// compute 03400520, at full res): march CS_STEPS points from `wp` toward
/// the sun over ~CS_PX render pixels; blocked where the depth buffer has a
/// surface in front of a point by less than a thickness. Catches what the
/// shadow map's texels and filter cannot: tyres on the road, kerb edges,
/// posts, a wing's stays. 1 = lit.
fn contact_shadow(q: vec2<i32>, wp: vec3<f32>, n: vec3<f32>, noise: f32, w0: f32) -> f32 {
    if (CS_STEPS == 0 || w0 > 120.0) {
        return 1.0;
    }
    let l = normalize(u.sun.xyz);
    // metres along the sun ray per render pixel here
    let c1 = u.vp * vec4<f32>(wp + l * 0.1, 1.0);
    let c0 = u.vp * vec4<f32>(wp, 1.0);
    let dpx = length((c1.xy / c1.w - c0.xy / c0.w) * 0.5 * u.rsize.xy) * 10.0;
    let len = clamp(CS_PX / max(dpx, 1e-3), 0.02, CS_MAXLEN);
    let start = wp + n * (0.01 + 0.002 * w0);
    var hit = 0.0;
    for (var i = 0; i < CS_STEPS; i = i + 1) {
        let t = (f32(i) + noise) / f32(CS_STEPS);
        let sp = start + l * (len * t * t);
        let c = u.vp * vec4<f32>(sp, 1.0);
        if (c.w <= 0.0) {
            break;
        }
        let suv = (c.xy / c.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
        if (any(suv < vec2<f32>(0.0)) || any(suv > vec2<f32>(1.0))) {
            break;
        }
        let sd = depth_r(vec2<i32>(suv * u.rsize.xy));
        if (sd >= 0.9999999) {
            continue;
        }
        let sw = world_r(vec2<i32>(suv * u.rsize.xy));
        let scene = (u.vp * vec4<f32>(sw, 1.0)).w;
        let gap = c.w - scene;
        // in front of the ray point, but not so far in front that it is a
        // separate nearer object (the car over the road behind it, a fence
        // in front of a wall): Bend's SurfaceThickness, here metres.
        // ⚠ The near limit was 0.008 w - the error a whole-pixel snap makes
        // on ground seen at a slant - against a far limit of 0.05 + 0.003 w:
        // an empty window past 10 m and ~2 cm wide at chase distance, so
        // kerbs, posts and tyres never got one. The receiver's own surface is
        // told apart by height instead: a sample of it lies on its plane,
        // while an occluder the ray is behind stands at least as high as the
        // ray point does. EVOVIEW_CONTACT_OLD=1: the old window.
        var hit_now = false;
        if (CS_OLD) {
            hit_now = gap > 0.008 * c.w && gap < CS_THICK + 0.003 * c.w;
        } else {
            let tol = 0.002 + 0.0005 * c.w;
            // the thickness grows with distance by CS_THICKK x w (it was
            // 0.003: ~0.2 m at 40 m, and a ray passing just behind a person
            // or a pole through pit glass read as blocked)
            hit_now = gap > tol && gap < tol + CS_THICK + CS_THICKK * c.w
                && dot(n, sw - wp) > 0.5 * dot(n, sp - wp);
        }
        if (hit_now) {
            hit = 1.0;
            break;
        }
    }
    // nor at a grazing sun: the march skims its own (depth-buffer-flat,
    // texture-rough) surface and speckles it
    return mix(1.0 - hit, 1.0, max(smoothstep(60.0, 120.0, w0), 1.0 - smoothstep(0.15, 0.35, dot(n, l))));
}

/// Is the sun visible from `w` (normal `n`, `ndl` N.L; 0 and 1 for a point
/// in the air)? The colour pass's cascades in its order (track.wgsl
/// sun_shadow: near, middle, far, each until the middle of its hand-over
/// band) with its bias and normal offset, one compare tap each.
/// ⚠ It read only the near and far maps with one 0.0005 bias for both: 1.2 m
/// in the near map's depth and 6 m in the far one's, and past the near map
/// it fell straight to the 1 m far texels.
fn sun_vis(w: vec3<f32>, n: vec3<f32>, ndl: f32) -> f32 {
    if (SV_OLD) {
        return sun_vis_old(w);
    }
    let slope = clamp(1.0 - ndl, 0.0, 1.0);
    for (var k = 0; k < 3; k = k + 1) {
        var vp = u.lvp_near;
        var b = u.sbias.x;
        var o = u.snofs.x;
        var layer = 0;
        var lim = 0.85;
        if (k == 1) {
            if (u.sbias.w < 0.5) {
                continue;
            }
            vp = u.lvp_mid;
            b = u.sbias.z;
            o = u.snofs.z;
            layer = 2;
            lim = 0.9;
        } else if (k == 2) {
            if (u.amb.w < 0.5) {
                break;
            }
            vp = u.lvp_far;
            b = u.sbias.y;
            o = u.snofs.y;
            layer = 1;
            lim = 0.98;
        }
        let lc = vp * vec4<f32>(w + n * (o * slope), 1.0);
        if (lc.w <= 0.0) {
            continue;
        }
        let ndc = lc.xyz / lc.w;
        let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
        let d = max(abs(uv.x - 0.5), abs(uv.y - 0.5)) * 2.0;
        if (ndc.z <= 0.0 || ndc.z >= 1.0 || d > lim) {
            continue;
        }
        return textureSampleCompareLevel(t_shadow, s_shadow, uv, layer, ndc.z - b * (1.0 + slope));
    }
    return 1.0;
}

/// sun_vis as it was (EVOVIEW_SUNVIS_OLD=1): near, then far, one bias.
fn sun_vis_old(w: vec3<f32>) -> f32 {
    for (var layer = 0; layer < 2; layer = layer + 1) {
        if (layer == 1 && u.amb.w < 0.5) {
            break;
        }
        var vp = u.lvp_near;
        if (layer == 1) {
            vp = u.lvp_far;
        }
        let lc = vp * vec4<f32>(w, 1.0);
        if (lc.w <= 0.0) {
            continue;
        }
        let ndc = lc.xyz / lc.w;
        let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
        if (ndc.z <= 0.0 || ndc.z >= 1.0 || any(uv < vec2<f32>(0.02)) || any(uv > vec2<f32>(0.98))) {
            continue;
        }
        return textureSampleCompareLevel(t_shadow, s_shadow, uv, layer, ndc.z - 0.0005);
    }
    return 1.0;
}

fn hg(cos_t: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (12.566371 * pow(max(1.0 + g2 - 2.0 * g * cos_t, 1e-4), 1.5));
}

/// Volumetric fog along the view ray to `wp` (the game's volumetric fog,
/// summer_sunny settings): sunlit fog where the sun reaches through the
/// shadow map - shafts behind trees and grandstands - with the game's two
/// lobes (0.2 isotropic + 0.6 forward, g 0.8) and its sun boost
/// (exponent 25), plus an ambient part. 24 steps, jittered per pixel and
/// frame; the history below averages it. Returns (in-scatter, transmittance)
/// in display units.
fn vol_fog(q: vec2<i32>, wp: vec3<f32>, sky: bool) -> vec4<f32> {
    if (u.sun_col.w < 0.5 || u.fog.x <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let e = u.eye.xyz;
    var dir = wp - e;
    var dist = length(dir);
    dir = dir / max(dist, 1e-4);
    if (sky) {
        // the sky itself already carries the atmosphere: only the near air
        dist = u.fog.w * 0.45;
    }
    dist = min(dist, u.fog.w);
    let steps = FOG_STEPS;
    let fq = vec2<f32>(q) + vec2<f32>(u.ao.z * 7.13, u.ao.z * 3.71);
    let jit = fract(52.9829189 * fract(dot(fq, vec2<f32>(0.06711056, 0.00583715))));
    let ds = dist / f32(steps);
    let sigma = u.fog.x;
    let cos_t = dot(dir, normalize(u.sun.xyz));
    // phase: the game's weights; the boost is a tight glow around the sun
    let phase = 0.2 * 0.0795775 + 0.6 * hg(cos_t, u.fog.y)
              + 0.06 * pow(max(cos_t, 0.0), 25.0);
    let sun_l = u.sun_col.rgb * u.sun.w;
    var lsum = vec3<f32>(0.0);
    var tr = 1.0;
    for (var i = 0; i < steps; i = i + 1) {
        let s = e + dir * ((f32(i) + jit) * ds);
        // thinner with height above the camera's ground (a ground haze)
        let dens = sigma * exp(-max(s.y - (e.y - 2.0), 0.0) / 120.0);
        let vis = sun_vis(s, vec3<f32>(0.0), 1.0);
        let st = exp(-dens * ds);
        let inscat = (sun_l * vis * phase * 1.6 + u.amb.rgb * 0.06) * dens;
        lsum = lsum + inscat * tr * (1.0 - st) / max(dens, 1e-6);
        tr = tr * st;
    }
    return vec4<f32>(lsum, tr);
}

fn fog_apply(rgb: vec3<f32>, f: vec4<f32>) -> vec3<f32> {
    return rgb * f.a + f.rgb;
}

fn ao_apply(rgb: vec3<f32>, ao: f32) -> vec3<f32> {
    // EVOVIEW_AO=9: show the occlusion itself
    if (u.ao.y > 6.5) {
        return vec3<f32>(ao);
    }
    return rgb * mix(1.0, ao, u.ao.y);
}

struct AoFogOut {
    @location(0) ao: vec4<f32>,
    @location(1) fog: vec4<f32>,
};

/// This frame's occlusion and fog, accumulated over frames at their own
/// size (the resolve only reads them): AOF_STEP 2 = one texel per 2x2 render
/// pixels - both are noisy and averaged anyway, and their full-screen
/// histories were ~100 MB a frame of the resolve's memory traffic.
@fragment
fn fs_aofog(in: VsOut) -> AoFogOut {
    let q = vec2<i32>(in.pos.xy) * AOF_STEP;
    let d = depth_r(q);
    let wp = world_r(q);
    let sky = d >= 0.9999999;
    let aoc = select(ssao(q, wp), 1.0, sky);
    let fogc = vol_fog(q, wp, sky);
    // the sky at exactly the f16-safe ceiling, so it matches itself in the
    // rejection and upsampling tests below whatever the far plane
    let z = select(min((u.vp * vec4<f32>(wp, 1.0)).w, 60000.0), 60000.0, sky && AOFUP_ON);
    var o: AoFogOut;
    o.ao = vec4<f32>(aoc, z, 0.0, 1.0);
    o.fog = fogc;
    if (u.ctl.x > 0.5) {
        return o;
    }
    // reproject: the driven car with its own motion; the road band around
    // it (its contact occlusion moves over a still road) takes mostly this
    // frame, as the resolve did
    var pw = wp;
    var fast = false;
    var band = false;
    if (u.obj_half.w > 0.5) {
        let l = u.obj_inv * vec4<f32>(wp, 1.0);
        if (all(abs(l.xyz) < u.obj_half.xyz)) {
            pw = (u.obj_prev * vec4<f32>(l.xyz, 1.0)).xyz;
            fast = true;
        } else {
            band = all(abs(l.xyz) < u.obj_half.xyz + vec3<f32>(0.8, 0.5, 1.2));
            fast = band;
        }
    }
    var car_w = 0.8;
    let pc0 = u.prev_vp * vec4<f32>(pw, 1.0);
    var puv = (pc0.xy / pc0.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    var pcw = pc0.w;
    if (u.jit.w > 0.5) {
        let mv = mv_at(q);
        puv = (vec2<f32>(q) + 0.5 + mv.xy) * u.rsize.zw;
        pcw = mv.z;
        // ⚠ Every car pixel has w = 1, parked or not, and "fast" took 80 %
        // of this frame's 8-sample occlusion there: the paint and glass
        // wore a frozen speckle of single-frame AO (sharpened, it read as
        // dirt). The vectors reproject the car's surface exactly, so it
        // keeps a history (AO_CAR_W this frame, EVOVIEW_AO_CAR_W; 0.8 =
        // the old look); the road band round the driven car stays fast.
        fast = fast || mv.w > 0.5;
        if (mv.w > 0.5 && !band) {
            car_w = AO_CAR_W;
        }
    }
    let pc = vec4<f32>(0.0, 0.0, 0.0, pcw);
    if (pc.w <= 0.0 || any(puv < vec2<f32>(0.0)) || any(puv > vec2<f32>(1.0))) {
        return o;
    }
    if (AOFUP_ON) {
        // Bilinear history, each of the 4 texels kept only if it saw this
        // depth (the sky included: it held the trees' fog when a tree line
        // swept over it - the grey band above them in motion). The nearest
        // texel alone stepped by whole texels under motion: blotches.
        let pz = select(min(pc.w, 60000.0), 60000.0, sky);
        let dim = vec2<i32>(textureDimensions(t_aoh));
        // texel i holds render pixel i * AOF_STEP (its centre at + 0.5)
        let t = (puv * u.rsize.xy - 0.5) / f32(AOF_STEP);
        let i0 = vec2<i32>(floor(t));
        let f = t - floor(t);
        var ws = 0.0;
        var ah = 0.0;
        var fh = vec4<f32>(0.0);
        for (var j = 0; j < 4; j = j + 1) {
            let c = vec2<i32>(j & 1, j >> 1);
            let hp = clamp(i0 + c, vec2<i32>(0), dim - vec2<i32>(1));
            let a = textureLoad(t_aoh, hp, 0);
            let fc = vec2<f32>(c);
            let wb = mix(1.0 - f.x, f.x, fc.x) * mix(1.0 - f.y, f.y, fc.y);
            let w = select(0.0, wb, abs(a.g - pz) <= 0.1 * pz + 0.05);
            ws += w;
            ah += a.r * w;
            fh += textureLoad(t_fogh, hp, 0) * w;
        }
        // (mostly) disoccluded: this frame alone
        if (ws < 0.1) {
            return o;
        }
        o.ao = vec4<f32>(mix(ah / ws, aoc, select(0.12, car_w, fast)), z, 0.0, 1.0);
        o.fog = mix(fh / ws, fogc, 0.1);
        return o;
    }
    let dim = vec2<i32>(textureDimensions(t_aoh));
    let hp = clamp(vec2<i32>(puv * vec2<f32>(dim)), vec2<i32>(0), dim - vec2<i32>(1));
    let ah = textureLoad(t_aoh, hp, 0);
    // something else was here (disocclusion, a moving prop)
    let moved = !sky && abs(ah.g - min(pc.w, 60000.0)) > 0.1 * min(pc.w, 60000.0) + 0.05;
    if (moved) {
        return o;
    }
    o.ao = vec4<f32>(mix(ah.r, aoc, select(0.12, car_w, fast)), z, 0.0, 1.0);
    o.fog = mix(textureLoad(t_fogh, hp, 0), fogc, 0.1);
    return o;
}

/// This pixel's occlusion (x) and fog (yzw + transmittance in .a of the
/// second) from fs_aofog's targets. At AOF_STEP 2 a texel holds render
/// pixel 2i (not the 2x2 block's middle): the 4 texels around that point,
/// bilinear, each weighted by how well its view depth (.g) matches this
/// pixel's (`zr`, aofog_z) - so a wall never takes the sky's fog nor a tree
/// line the wall's occlusion. Plain bilinear with EVOVIEW_AOFOG_UP=0.
struct AoFogPx {
    ao: f32,
    fog: vec4<f32>,
};

fn aofog_up(uv0: vec2<f32>, zr: vec2<f32>) -> AoFogPx {
    var r: AoFogPx;
    if (!AOFUP_ON) {
        r.ao = textureSampleLevel(t_aoc, s_lin, uv0, 0.0).r;
        r.fog = textureSampleLevel(t_fogc, s_lin, uv0, 0.0);
        return r;
    }
    let dim = vec2<i32>(textureDimensions(t_aoc));
    let t = (uv0 * u.rsize.xy - 0.5) / f32(AOF_STEP);
    let i0 = vec2<i32>(floor(t));
    let f = t - floor(t);
    var ws = 0.0;
    var ao = 0.0;
    var fog = vec4<f32>(0.0);
    // The occlusion over 4 x 4 texels (a tent 2 texels wide), the fog over the
    // 2 x 2: the occlusion's interleaved-gradient noise is applied after the
    // TAA history, so nothing else averages it - on a flat white surface
    // (a truck, a white car) the 2 x 2 left a mosaic of 2 px blocks.
    var ws_ao = 0.0;
    for (var j = 0; j < 16; j = j + 1) {
        let c = vec2<i32>((j & 3) - 1, (j >> 2) - 1);
        let hp = clamp(i0 + c, vec2<i32>(0), dim - vec2<i32>(1));
        let a = textureLoad(t_aoc, hp, 0);
        let dd = abs(vec2<f32>(c) - f);
        let wt = max(1.0 - dd.x * 0.5, 0.0) * max(1.0 - dd.y * 0.5, 0.0) + 1e-3;
        let r0 = abs(a.g - zr.x) / max(zr.x, 0.01) * 33.0;
        let r1 = abs(a.g - zr.y) / max(zr.y, 0.01) * 33.0;
        let w = wt * (1.0 / (1.0 + r0 * r0) + 1.0 / (1.0 + r1 * r1));
        ws_ao += w;
        ao += a.r * w;
    }
    for (var j = 0; j < 4; j = j + 1) {
        let c = vec2<i32>(j & 1, j >> 1);
        let hp = clamp(i0 + c, vec2<i32>(0), dim - vec2<i32>(1));
        let a = textureLoad(t_aoc, hp, 0);
        let fc = vec2<f32>(c);
        let wb = mix(1.0 - f.x, f.x, fc.x) * mix(1.0 - f.y, f.y, fc.y) + 1e-3;
        // 1/2 at a 3 % depth difference; never 0, so a pixel no texel
        // matches (a 1 px pole) still gets the nearest-depth one's value.
        // Against both the nearest and the farthest depth round the pixel:
        // the jitter flips an outline pixel's own depth between the tree
        // and the sky every frame, and its occlusion / fog (applied after
        // the history) flickered with it; an outline takes half of each.
        let r0 = abs(a.g - zr.x) / max(zr.x, 0.01) * 33.0;
        let r1 = abs(a.g - zr.y) / max(zr.y, 0.01) * 33.0;
        let w = wb * (1.0 / (1.0 + r0 * r0) + 1.0 / (1.0 + r1 * r1));
        ws += w;
        fog += textureLoad(t_fogc, hp, 0) * w;
    }
    r.ao = ao / ws_ao;
    r.fog = fog / ws;
    return r;
}

/// The nearest and farthest view depth of render texels `k` +-1, as
/// fs_aofog stores depth (the sky at 60000).
fn aofog_z(k: vec2<i32>) -> vec2<f32> {
    if (!AOFUP_ON) {
        return vec2<f32>(0.0);
    }
    var kn = k;
    var kf = k;
    var dn = depth_r(k);
    var df = dn;
    for (var j = 0; j < 9; j = j + 1) {
        let q = k + vec2<i32>(j % 3 - 1, j / 3 - 1);
        let d = depth_r(q);
        if (d < dn) { dn = d; kn = q; }
        if (d > df) { df = d; kf = q; }
    }
    let zn = select(min((u.vp * vec4<f32>(world_r(kn), 1.0)).w, 60000.0), 60000.0, dn >= 0.9999999);
    let zf = select(min((u.vp * vec4<f32>(world_r(kf), 1.0)).w, 60000.0), 60000.0, df >= 0.9999999);
    return vec2<f32>(zn, zf);
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let p = vec2<i32>(in.pos.xy);
    let up = u.jit.z > 0.5;
    // Upsampling: the render texel whose (jittered) sample lies nearest
    // this pixel, and how far off it is in screen pixels - a sample right
    // on the pixel counts fully, one a pixel away barely (the history
    // fills in between). Same size: the texel itself, as before.
    let pr = in.pos.xy * u.rsize.xy * u.size.zw;
    let k = select(p, vec2<i32>(floor(pr + u.jit.xy)), up);
    let dd = (pr - (vec2<f32>(k) + 0.5 - u.jit.xy)) * u.size.xy * u.rsize.zw;
    let near_w = select(1.0, exp(-2.0 * dot(dd, dd)) * u.size.x * u.size.y * u.rsize.z * u.rsize.w, up);
    let c = load_cur(k);
    // the unjittered colour here (bilinear), for pixels with no history
    let cb = select(c, textureSampleLevel(t_cur, s_lin, (pr + u.jit.xy) * u.rsize.zw, 0.0).rgb, up);
    var o: FsOut;
    // this pixel's view depth, kept in history.a for the next frame's
    // disocclusion test
    let uv0 = in.pos.xy * u.size.zw;
    // upsampling: the nearest depth of the 3x3 (dilated) - the jitter moves
    // k across an edge from frame to frame, and the plain texel's depth
    // jumped with it, failing the disocclusion test on every outline
    var d0 = depth_r(k);
    // the motion vector comes from the nearest texel of the 3x3 too (the
    // usual dilation: an edge pixel follows the object in front)
    var kd = k;
    var dmin = d0;
    let use_mv = u.jit.w > 0.5;
    if (up || use_mv) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let dq = depth_r(k + vec2<i32>(x, y));
                if (dq < dmin) {
                    dmin = dq;
                    kd = k + vec2<i32>(x, y);
                }
            }
        }
        if (up) {
            d0 = dmin;
        }
    }
    let w0 = u.inv_vp * vec4<f32>(uv0.x * 2.0 - 1.0, 1.0 - uv0.y * 2.0, min(d0, 0.9999999), 1.0);
    let world0 = w0.xyz / w0.w;
    // clamped into f16 range: the sky's depth overflowed history.a to inf,
    // so every sky pixel failed the depth test each frame (flickering clouds)
    let zc = min((u.vp * vec4<f32>(world0, 1.0)).w, 60000.0);
    let is_sky = d0 >= 0.9999999;
    // (accumulated already)
    let af = aofog_up(uv0, aofog_z(k));
    let aoc = af.ao;
    let fogc = af.fog;
    if (u.ctl.x > 0.5) {
        o.screen = vec4<f32>(fog_apply(ao_apply(cb, aoc), fogc), 1.0);
        o.history = vec4<f32>(cb, zc);
        return o;
    }
    // neighbourhood statistics (variance clipping, YCoCg)
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var lo = vec3<f32>(1e9);
    var hi = vec3<f32>(-1e9);
    // upsampling: this frame's colour here, reconstructed from the 3x3
    // jittered samples with a Gaussian in screen pixels (the nearest one
    // alone left every edge in render-pixel steps)
    var cf = vec3<f32>(0.0);
    var cfw = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let o = vec2<i32>(x, y);
            let s = to_ycocg(load_cur(k + o));
            m1 += s;
            m2 += s * s;
            lo = min(lo, s);
            hi = max(hi, s);
            let dq = (pr - (vec2<f32>(k + o) + 0.5 - u.jit.xy)) * u.size.xy * u.rsize.zw;
            let gw = exp(-2.29 * dot(dq, dq));
            cf += s * gw;
            cfw += gw;
        }
    }
    let mean = m1 / 9.0;
    let sd = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));

    // reproject through depth
    let uv = in.pos.xy * u.size.zw;
    let d = d0;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, min(d, 0.9999999), 1.0);
    let w = u.inv_vp * ndc;
    var world = w.xyz / w.w;
    // a pixel on the driven car moved with it: take it to where the car was
    var on_obj = false;
    let near_car = u.obj_half.w > 0.5 && all(uv0 >= u.obj_rect.xy) && all(uv0 <= u.obj_rect.zw);
    if (near_car) {
        let l = u.obj_inv * vec4<f32>(world, 1.0);
        if (all(abs(l.xyz) < u.obj_half.xyz)) {
            world = (u.obj_prev * vec4<f32>(l.xyz, 1.0)).xyz;
            on_obj = true;
        }
    }
    // The band just outside the driven car: scenery the car covered a frame
    // ago, and a 3x3 depth match there finds a scenery texel beside the car
    // - the car's colour stayed blended in as a halo around it (the ghost).
    var near_obj = false;
    if (near_car && !on_obj) {
        for (var k = 0; k < 8; k = k + 1) {
            let a = f32(k) * 0.785398;
            let q = p + vec2<i32>(vec2<f32>(cos(a), sin(a)) * 3.0);
            if (in_obj(world_at(q))) {
                near_obj = true;
            }
        }
    }
    var pc = u.prev_vp * vec4<f32>(world, 1.0);
    var puv = (pc.xy / pc.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    // motion vectors (motion.rs): every car, its wheels and the camera
    // exactly, in place of the box above
    var spin_fast = false;
    if (use_mv) {
        let mvv = mv_at(kd);
        puv = uv + mvv.xy * u.rsize.zw;
        pc.w = mvv.z;
        on_obj = mvv.w > 0.5;
        // w = 2: a wheel spinning too fast for its history (motion.wgsl)
        spin_fast = mvv.w > 1.5;
        // the band just outside any car: a ring of texels 3 px out
        near_obj = false;
        if (!on_obj) {
            for (var k = 0; k < 8; k = k + 1) {
                let a = f32(k) * 0.785398;
                let q = p + vec2<i32>(vec2<f32>(cos(a), sin(a)) * 3.0);
                if (mv_at(to_render(q)).w > 0.5) {
                    near_obj = true;
                }
            }
        }
    }
    if (pc.w <= 0.0 || any(puv < vec2<f32>(0.0)) || any(puv > vec2<f32>(1.0))) {
        o.screen = vec4<f32>(fog_apply(ao_apply(cb, aoc), fogc), 1.0);
        o.history = vec4<f32>(cb, zc);
        return o;
    }
    // Disocclusion / motion: the depth that was at this spot last frame
    // should be the depth this point had then. Animated props (waving
    // marshals, flags) and replay cars move without motion vectors - their
    // pixels reproject onto something else, and blending that history
    // smeared them ("stretched textures"). Any of the 3x3 texels agreeing
    // keeps still edges anti-aliased under the jitter.
    let hp = vec2<i32>(puv * u.size.xy);
    let hm = vec2<i32>(i32(u.size.x) - 1, i32(u.size.y) - 1);
    var best = 1e9;
    // by the car only the texel itself may vouch for the history
    let rad = select(1, 0, near_obj);
    for (var y = -rad; y <= rad; y = y + 1) {
        for (var x = -rad; x <= rad; x = x + 1) {
            let zp = textureLoad(t_hist, clamp(hp + vec2<i32>(x, y), vec2<i32>(0), hm), 0).a;
            best = min(best, abs(zp - pc.w));
        }
    }
    // A real jump only: u.ctl.w (taa.rs, EVOVIEW_TAA_TOL) is 10 % with motion
    // vectors, 3 % without. At 3 % the sub-pixel jitter alone fails thin
    // edges (fences) frame after frame - measured at Imola's start, still
    // camera, 8-16 % more fence pixels change frame to frame than at 10 %.
    // 3 % was chosen when the driven car was a box: at 10 % the road just
    // behind it (nearly its depth) kept the car's history - with motion
    // vectors the car's pixels are exact and that ghost does not return.
    // A moving arm or flag still jumps by metres.
    let moved = u.ctl.z > 0.5 && !is_sky && best > u.ctl.w * min(pc.w, 60000.0) + 0.05;
    // the occlusion accumulates over frames like the colour does
    // The road around and under the driven car: its contact shadow, its
    // shadow and the car's own edge sweep across a static surface every
    // frame, and history kept there trailed behind the car over asphalt
    // (grass hid it). Tight clip and more of this frame there.
    var under = false;
    if (near_car && !on_obj) {
        let l = u.obj_inv * vec4<f32>(w.xyz / w.w, 1.0);
        let e = u.obj_half.xyz + vec3<f32>(0.8, 0.0, 1.2);
        under = abs(l.x) < e.x && abs(l.z) < e.z && l.y < -u.obj_half.y + 0.4 && l.y > -u.obj_half.y - 0.5;
        // seen THROUGH the car (its windows): the line from the eye to this
        // point crosses the car's box - what is behind the glass slides
        // past it every frame
        let le = (u.obj_inv * vec4<f32>(u.eye.xyz, 1.0)).xyz;
        // ⚠ not from inside it (cockpit): every line out of the car crosses
        // the box, and the whole world outside got 50 % of each frame -
        // barely anti-aliased, shimmering. From inside, the world through
        // the glass reprojects like any other scenery.
        let eye_in = all(abs(le) < u.obj_half.xyz);
        let d = l.xyz - le;
        let inv = 1.0 / select(d, vec3<f32>(1e-6), abs(d) < vec3<f32>(1e-6));
        let t0 = (-u.obj_half.xyz - le) * inv;
        let t1 = (u.obj_half.xyz - le) * inv;
        let tn = min(t0, t1);
        let tf = max(t0, t1);
        let enter = max(max(tn.x, tn.y), max(tn.z, 0.0));
        let exit = min(min(tf.x, tf.y), min(tf.z, 1.0));
        if (enter < exit && !eye_in) {
            under = true;
        }
        // in the car's shadow: the way from this point to the sun crosses
        // the car (a little grown, for the soft edge) - the shadow moves
        // with the car over a still road
        if (u.sun.w > 0.0 && !under) {
            let ls = normalize((u.obj_inv * vec4<f32>(u.sun.xyz, 0.0)).xyz) * 8.0;
            let hb = u.obj_half.xyz + vec3<f32>(0.3, 0.1, 0.3);
            let sinv = 1.0 / select(ls, vec3<f32>(1e-6), abs(ls) < vec3<f32>(1e-6));
            let s0 = (-hb - l.xyz) * sinv;
            let s1 = (hb - l.xyz) * sinv;
            let sn = min(s0, s1);
            let sf = max(s0, s1);
            if (max(max(sn.x, sn.y), max(sn.z, 0.0)) < min(min(sf.x, sf.y), min(sf.z, 1.0))) {
                under = true;
            }
        }
    }
    // ⚠ the occlusion's history (fs_aofog) takes mostly this frame by the
    // car: kept at 88 %, the car's dark footprint trailed behind it on the
    // road as a stepped smear
    let ao = aoc;
    let fog = fogc;
    var h = to_ycocg(hist_cr(puv));
    let vel = length((puv - uv) * u.size.xy);
    // any motion at all: 0 still, 1 from a quarter pixel a frame
    let moving = smoothstep(0.02, 0.25, vel);
    // Clip (not clamp) the history toward the box centre: a clamp keeps the
    // stale colour's hue on the box face it hits, and at speed that showed
    // as a faint second copy of fences and wall panels.
    // the car itself clips tight too: its glass reflects a world that turns
    // past it, and the loose min/max box kept that as a ghost on the windows
    // moving pixels clip tighter (gamma 1.25 -> 1.0): foliage's wide colour
    // box let softened history through unchallenged
    let g = mix(TAA_GAMMA, min(TAA_GAMMA, 1.0), moving);
    let cmin = max(lo, mean - g * sd);
    let cmax = min(hi, mean + g * sd);
    let cen = 0.5 * (cmax + cmin);
    let ext = max(0.5 * (cmax - cmin), vec3<f32>(1e-4));
    let off = h - cen;
    let a = abs(off / ext);
    let ma = max(a.x, max(a.y, a.z));
    if (ma > 1.0) {
        h = cen + off / ma;
    }
    // luminance-weighted blend (keeps bright specks from flickering)
    let cy = select(to_ycocg(c), cf / max(cfw, 1e-4), up);
    // where the depth jumped, lean on this frame instead of dropping history
    // the further this pixel moved on screen, the less its (resampled,
    // softened) history is worth: 10 % still, 20 % at any motion (from a
    // quarter pixel a frame), up to 35 % at 12 px. ⚠ The ramp started at
    // 0.5 px: distant scenery while driving creeps 0.2-3 px a frame and sat
    // at ~10 %, eight frames of compounding resample blur - the "distant
    // things are blurry" look.
    var wc = select(mix(u.ctl.y, TAA_MOVE_W, moving) + (0.35 - TAA_MOVE_W) * smoothstep(1.0, 12.0, vel), 1.0, moved);
    if (near_obj) {
        wc = max(wc, 0.3);
    }
    if (under) {
        wc = max(wc, 0.5);
    }
    // The sky only reprojects (no disocclusion test applies to it), and the
    // cloud march's jitter wants a longer history: 5 % of this frame by
    // default (EVOVIEW_TAA_SKY) halves the cloud edges' grain and flicker.
    if (is_sky && !moved) {
        wc = min(wc, u.sky.x);
    }
    if (on_obj) {
        wc = max(wc, 0.25);
    }
    // the driven car's spinning wheels: no usable history (with motion
    // vectors the rim's own spin reprojects it; EVOVIEW_MOTION_WHEELS=0
    // keeps this)
    let wp_now = w.xyz / w.w;
    if (spin_fast) {
        wc = max(wc, SPIN_FAST_W);
    }
    if (!use_mv || MV_WHEELS_OFF) {
        for (var k = 0; k < 4; k = k + 1) {
            let wh = u.wheels[k];
            if (wh.w > 0.0 && length(wp_now - wh.xyz) < wh.w + 0.06) {
                wc = 1.0;
            }
        }
    }
    // upsampling: this frame's sample counts by how near it fell; where
    // there is no usable history, the bilinear colour instead
    var cy2 = cy;
    if (up) {
        if (wc >= 1.0) {
            cy2 = to_ycocg(cb);
        } else {
            wc = clamp(wc * near_w, 0.0, 1.0);
        }
    }
    // EVOVIEW_MOTION_DEBUG=1: the motion vectors (r/g |x|/|y| over 8 px,
    // b on a moving object)
    if (MV_DEBUG && use_mv) {
        let mvd = mv_at(kd);
        o.screen = vec4<f32>(min(abs(mvd.xy) / 8.0, vec2<f32>(1.0)), mvd.w * 0.6, 1.0);
        o.history = vec4<f32>(0.0, 0.0, 0.0, zc);
        return o;
    }
    let kc = wc / (1.0 + cy2.x);
    let kh = (1.0 - wc) / (1.0 + h.x);
    let r = from_ycocg((cy2 * kc + h * kh) / (kc + kh));
    let rc = max(r, vec3<f32>(0.0));
    o.screen = vec4<f32>(fog_apply(ao_apply(rc, ao), fog), 1.0);
    o.history = vec4<f32>(rc, zc);
    return o;
}

/// An external upscaler's output (DLSS, dlss.rs) bound as t_hist, at the
/// screen size: only this frame's occlusion and fog go on, as fs_main does.
@fragment
fn fs_ext(in: VsOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.pos.xy);
    let uv0 = in.pos.xy * u.size.zw;
    let c = max(textureLoad(t_hist, p, 0).rgb, vec3<f32>(0.0));
    let af = aofog_up(uv0, aofog_z(to_render(p)));
    return vec4<f32>(fog_apply(ao_apply(c, af.ao), af.fog), 1.0);
}
