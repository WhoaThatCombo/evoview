// Post-processing after TAA (post.rs): the game's veil (bloom), the eye-focus
// vignette, contrast-adaptive sharpening and camera / car motion blur, plus
// a tiny log-luminance image for auto exposure.
//
// The input is already tonemapped (the scene shaders grade to display), so
// the veil's HDR threshold is mapped into display range: t / (1 + t).

struct U {
    inv_vp: mat4x4<f32>,
    prev_vp: mat4x4<f32>,
    obj_inv: mat4x4<f32>,
    obj_prev: mat4x4<f32>,
    // xyz half extents, w 1 = the driven car's box is in use
    obj_half: vec4<f32>,
    // x threshold (display), y intensity, z knee, w mode (0 down+threshold,
    // 1 down, 2 up, 3 composite, 4 luminance, 5 sun shafts)
    veil: vec4<f32>,
    // x vignette intensity, y vignette pow, z sharpen, w motion blur strength
    fx: vec4<f32>,
    // destination w, h, 1/w, 1/h
    dst: vec4<f32>,
    // source texel size (1/w, 1/h), xy; zw unused
    src: vec4<f32>,
    // camera position
    eye: vec4<f32>,
    // x 1 = HDR input: tonemap here through the game's LUT; y LUT z half
    hdr: vec4<f32>,
    // last frame's view-projection from this frame's orientation (travel only)
    prev_vp_t: mat4x4<f32>,
    // x share of the camera rotation's blur, y 1 = t_mv holds motion vectors
    blur: vec4<f32>,
    // depth of field: x focus distance (<= 0 auto), y CoC range, z foreground
    // distance, w background distance
    dof: vec4<f32>,
    // x auto-focus samples, y their radius (pixels), zw focus point (uv)
    dof2: vec4<f32>,
    // xyz camera forward, w 1 = depth of field on
    fwd: vec4<f32>,
    // sun shafts: xy the sun on screen (uv), z intensity, w 1 = on
    shaft: vec4<f32>,
    // x clarity (local contrast) strength, y its width in stops
    look: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_a: texture_2d<f32>;
@group(0) @binding(2) var t_b: texture_2d<f32>;
@group(0) @binding(3) var s_lin: sampler;
@group(0) @binding(4) var t_depth: DEPTH_TEXTURE;
@group(0) @binding(5) var t_lut: texture_3d<f32>;
@group(0) @binding(6) var t_dof_fg: texture_2d<f32>;
@group(0) @binding(7) var t_dof_bg: texture_2d<f32>;
// motion.rs's vectors (render size): xy last frame's pixel minus this one's,
// w 1 on a car
@group(0) @binding(8) var t_mv: texture_2d<f32>;
// the input's blur chain: the local mean for clarity, the sun shafts' source
@group(0) @binding(9) var t_local: texture_2d<f32>;
@group(0) @binding(10) var t_shaft: texture_2d<f32>;

/// The game's tonemap + grade (grade.rs): the log2 shaper (grade.rs) into the 32^3
/// LUT, outdoor half. Exposure is already in the input.
fn tm(c: vec3<f32>) -> vec3<f32> {
    if (u.hdr.x < 0.5) {
        return c;
    }
    let x = max(c, vec3<f32>(0.0));
    var uvw = clamp((log2(x + vec3<f32>(0.0009765625)) + vec3<f32>(10.0)) / 16.0, vec3<f32>(0.0), vec3<f32>(1.0));
    uvw = uvw * (31.0 / 32.0) + vec3<f32>(0.5 / 32.0);
    uvw.z = (uvw.z + u.hdr.y) * 0.5;
    return textureSampleLevel(t_lut, s_lin, uvw, 0.0).rgb;
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    var o: VsOut;
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}

fn lum(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// 13-tap downsample (Jimenez, CoD AW), sampled from the source level
fn down13(uv: vec2<f32>) -> vec3<f32> {
    let t = u.src.xy;
    let a = textureSampleLevel(t_a, s_lin, uv + t * vec2<f32>(-2.0, -2.0), 0.0).rgb;
    let b = textureSampleLevel(t_a, s_lin, uv + t * vec2<f32>(0.0, -2.0), 0.0).rgb;
    let c = textureSampleLevel(t_a, s_lin, uv + t * vec2<f32>(2.0, -2.0), 0.0).rgb;
    let d = textureSampleLevel(t_a, s_lin, uv + t * vec2<f32>(-2.0, 0.0), 0.0).rgb;
    let e = textureSampleLevel(t_a, s_lin, uv, 0.0).rgb;
    let f = textureSampleLevel(t_a, s_lin, uv + t * vec2<f32>(2.0, 0.0), 0.0).rgb;
    let g = textureSampleLevel(t_a, s_lin, uv + t * vec2<f32>(-2.0, 2.0), 0.0).rgb;
    let h = textureSampleLevel(t_a, s_lin, uv + t * vec2<f32>(0.0, 2.0), 0.0).rgb;
    let i = textureSampleLevel(t_a, s_lin, uv + t * vec2<f32>(2.0, 2.0), 0.0).rgb;
    let j = textureSampleLevel(t_a, s_lin, uv + t * vec2<f32>(-1.0, -1.0), 0.0).rgb;
    let k = textureSampleLevel(t_a, s_lin, uv + t * vec2<f32>(1.0, -1.0), 0.0).rgb;
    let l = textureSampleLevel(t_a, s_lin, uv + t * vec2<f32>(-1.0, 1.0), 0.0).rgb;
    let m = textureSampleLevel(t_a, s_lin, uv + t * vec2<f32>(1.0, 1.0), 0.0).rgb;
    return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}

// 9-tap tent upsample
fn up9(uv: vec2<f32>) -> vec3<f32> {
    let t = u.src.xy;
    var s = textureSampleLevel(t_a, s_lin, uv, 0.0).rgb * 4.0;
    s += (textureSampleLevel(t_a, s_lin, uv + vec2<f32>(-t.x, 0.0), 0.0).rgb
        + textureSampleLevel(t_a, s_lin, uv + vec2<f32>(t.x, 0.0), 0.0).rgb
        + textureSampleLevel(t_a, s_lin, uv + vec2<f32>(0.0, -t.y), 0.0).rgb
        + textureSampleLevel(t_a, s_lin, uv + vec2<f32>(0.0, t.y), 0.0).rgb) * 2.0;
    s += textureSampleLevel(t_a, s_lin, uv + vec2<f32>(-t.x, -t.y), 0.0).rgb
        + textureSampleLevel(t_a, s_lin, uv + vec2<f32>(t.x, -t.y), 0.0).rgb
        + textureSampleLevel(t_a, s_lin, uv + vec2<f32>(-t.x, t.y), 0.0).rgb
        + textureSampleLevel(t_a, s_lin, uv + vec2<f32>(t.x, t.y), 0.0).rgb;
    return s / 16.0;
}

// The game's veil read (tonemap shader 04708280): a cubic B-spline through
// four bilinear taps, so the half-size veil carries no bilinear diamonds.
fn veil_bspline(uv: vec2<f32>) -> vec3<f32> {
    let res = vec2<f32>(textureDimensions(t_b));
    let p = uv * res + vec2<f32>(0.5);
    let i = floor(p);
    let f = p - i;
    let w0 = (((vec2<f32>(0.5) - f * (1.0 / 6.0)) * f - vec2<f32>(0.5)) * f) + vec2<f32>(1.0 / 6.0);
    let w1 = f * f * (f * 0.5 - vec2<f32>(1.0)) + vec2<f32>(2.0 / 3.0);
    let w2 = (((vec2<f32>(0.5) - f * 0.5) * f + vec2<f32>(0.5)) * f) + vec2<f32>(1.0 / 6.0);
    let w3 = vec2<f32>(1.0) - w0 - w1 - w2;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    let h0 = min((i - vec2<f32>(1.5) + w1 / g0) / res, vec2<f32>(1.0));
    let h1 = min((i + vec2<f32>(0.5) + w3 / g1) / res, vec2<f32>(1.0));
    let a = textureSampleLevel(t_b, s_lin, h0, 0.0).rgb;
    let b = textureSampleLevel(t_b, s_lin, vec2<f32>(h1.x, h0.y), 0.0).rgb;
    let c = textureSampleLevel(t_b, s_lin, vec2<f32>(h0.x, h1.y), 0.0).rgb;
    let d = textureSampleLevel(t_b, s_lin, h1, 0.0).rgb;
    return (a * g0.x + b * g1.x) * g0.y + (c * g0.x + d * g1.x) * g1.y;
}

fn world_at(px: vec2<i32>, uv: vec2<f32>) -> vec3<f32> {
    // the depth may be smaller than the screen (TAA upsampling): by uv
    let dm = vec2<i32>(textureDimensions(t_depth)) - vec2<i32>(1);
    let d = textureLoad(t_depth, min(vec2<i32>(uv * vec2<f32>(textureDimensions(t_depth))), dm), 0);
    _ = px;
    let w = u.inv_vp * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, min(d, 0.9999999), 1.0);
    return w.xyz / w.w;
}

// Sun shafts at 1/4 size: the sky near the sun (t_a, the 1/8-size blur)
// gathered along the line to the sun where the depth is sky, so gaps in what
// stands in front (trees, stands, the car's pillars) let rays through.
fn shafts(uv: vec2<f32>, fp: vec2<f32>) -> vec3<f32> {
    let asp = u.dst.x * u.dst.w;
    let to = u.shaft.xy - uv;
    let r0 = length(to * vec2<f32>(asp, 1.0));
    if (r0 > 1.2) {
        return vec3<f32>(0.0);
    }
    let dn = vec2<i32>(textureDimensions(t_depth)) - vec2<i32>(1);
    let dz = vec2<f32>(textureDimensions(t_depth));
    // interleaved gradient noise on the start: no banding steps
    let ign = fract(52.9829189 * fract(dot(fp, vec2<f32>(0.06711056, 0.00583715))));
    var acc = vec3<f32>(0.0);
    let n = 32;
    for (var k = 0; k < n; k = k + 1) {
        let t = (f32(k) + ign) / f32(n);
        let p = uv + to * t;
        if (any(p < vec2<f32>(0.0)) || any(p > vec2<f32>(1.0))) {
            continue;
        }
        if (textureLoad(t_depth, min(vec2<i32>(p * dz), dn), 0) < 0.99999) {
            continue;
        }
        // the sky's light near the sun only; the sun's disc is far brighter
        // than the sky, capped so it does not wash the sky white
        let r = length((p - u.shaft.xy) * vec2<f32>(asp, 1.0));
        let w = pow(clamp(1.0 - r / 0.35, 0.0, 1.0), 3.0);
        let sc = textureSampleLevel(t_a, s_lin, p, 0.0).rgb;
        acc += sc * min(1.0, 3.0 / max(lum(sc), 1e-4)) * w * t;
    }
    // falling off away from the sun
    return acc * (clamp(1.0 - r0 / 1.2, 0.0, 1.0) / f32(n));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let mode = i32(u.veil.w + 0.5);
    let uv = in.uv;
    if (mode == 0) {
        // first level: keep what is over the threshold, softly
        let c = down13(uv);
        let l = max(c.r, max(c.g, c.b));
        let knee = u.veil.x * u.veil.z + 1e-4;
        var s = clamp(l - u.veil.x + knee, 0.0, 2.0 * knee);
        s = s * s / (4.0 * knee);
        let w = max(s, l - u.veil.x) / max(l, 1e-4);
        return vec4<f32>(c * w, 1.0);
    }
    if (mode == 1) {
        return vec4<f32>(down13(uv), 1.0);
    }
    if (mode == 2) {
        return vec4<f32>(up9(uv), 1.0);
    }
    if (mode == 5) {
        return vec4<f32>(shafts(uv, in.pos.xy), 1.0);
    }
    if (mode == 4) {
        // auto exposure: log luminance of a box of the frame
        let c = textureSampleLevel(t_a, s_lin, uv, 0.0).rgb;
        return vec4<f32>(log2(max(lum(c), 1e-4)), 0.0, 0.0, 1.0);
    }
    // ---- composite ----
    let px = vec2<i32>(in.pos.xy);
    let maxp = vec2<i32>(i32(u.dst.x) - 1, i32(u.dst.y) - 1);
    var c = textureLoad(t_a, px, 0).rgb;

    // motion blur: this pixel's screen motion since last frame, from the
    // depth buffer (a still scene and a moving camera), the driven car's
    // pixels with the car's own motion (it stays sharp in a chase view)
    // and, with motion vectors, every car's: in a replay the car a TV
    // camera follows took the static scenery's motion and streaked
    var car_mv = false;
    var car_vel = vec2<f32>(0.0);
    if (u.fx.w != 0.0 && u.blur.y > 0.5) {
        let md = vec2<i32>(textureDimensions(t_mv));
        let m = textureLoad(t_mv, min(vec2<i32>(uv * vec2<f32>(md)), md - vec2<i32>(1)), 0);
        if (m.w > 0.5) {
            car_mv = true;
            car_vel = -m.xy / vec2<f32>(md);
        }
    }
    if (u.fx.w != 0.0) {
        var wp = world_at(px, uv);
        if (u.obj_half.w > 0.5 && u.blur.y < 0.5) {
            let l = u.obj_inv * vec4<f32>(wp, 1.0);
            if (all(abs(l.xyz) < u.obj_half.xyz)) {
                wp = (u.obj_prev * vec4<f32>(l.xyz, 1.0)).xyz;
            }
        }
        let pc = u.prev_vp * vec4<f32>(wp, 1.0);
        if (pc.w > 0.0) {
            let puv = (pc.xy / pc.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
            var vel = uv - puv;
            // the camera's rotation blurs at a share only: the travel part
            // (the road rushing by) is this pixel seen from last frame's eye
            // with this frame's orientation
            let pt = u.prev_vp_t * vec4<f32>(wp, 1.0);
            if (pt.w > 0.0) {
                let tuv = (pt.xy / pt.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
                vel = mix(uv - tuv, vel, u.blur.x);
            }
            // a car: its whole motion on screen (rotation included - the
            // camera turning to follow it is what keeps it still)
            if (car_mv) {
                vel = car_vel;
            }
            vel = vel * abs(u.fx.w);
            // EVOVIEW_MOTION_BLUR < 0: show the motion (debug)
            if (u.fx.w < 0.0) {
                return vec4<f32>(abs(vel.x) * 40.0, abs(vel.y) * 40.0, 0.0, 1.0);
            }
            // at most 4 % of the screen
            let vl = length(vel);
            if (vl > 0.04) {
                vel = vel * (0.04 / vl);
            }
            if (length(vel * u.dst.xy) > 1.0) {
                // gather along the motion, but never from something nearer
                // the camera (the car must not smear over the road behind it)
                let d0 = length(world_at(px, uv) - u.eye.xyz);
                var acc = c;
                var n = 1.0;
                for (var k = 1; k <= 8; k = k + 1) {
                    let o = (f32(k) / 8.0 - 0.5) * vel;
                    let suv = uv + o;
                    let sp = clamp(vec2<i32>(suv * u.dst.xy), vec2<i32>(0), maxp);
                    let ds = length(world_at(sp, suv) - u.eye.xyz);
                    if (ds < d0 * 0.92) {
                        continue;
                    }
                    acc += textureSampleLevel(t_a, s_lin, suv, 0.0).rgb;
                    n += 1.0;
                }
                c = acc / n;
            }
        }
    }

    // depth of field (exe 047806d0 + 0477f350): the view depth against the
    // focus distance gives a foreground and a background CoC, each mixing in
    // its blur mip. Auto focus: the mean depth of a ring around the focus point.
    var dofk = 0.0;
    if (u.fwd.w > 0.5) {
        let z = dot(world_at(px, uv) - u.eye.xyz, u.fwd.xyz);
        var f = u.dof.x;
        if (f <= 0.0) {
            let n = max(i32(u.dof2.x), 1);
            var s = 0.0;
            for (var k = 0; k < n; k = k + 1) {
                let a = f32(k) * (6.283185 / f32(n));
                let fuv = clamp(u.dof2.zw + u.dof2.y * u.dst.zw * vec2<f32>(cos(a), sin(a)), vec2<f32>(0.0), vec2<f32>(1.0));
                s += dot(world_at(px, fuv) - u.eye.xyz, u.fwd.xyz);
            }
            f = s / f32(n);
        }
        let cf = clamp((f - z - u.dof.z) / u.dof.y, 0.0, 1.0);
        let cb = clamp((z - f - u.dof.w) / u.dof.y, 0.0, 1.0);
        if (u.fwd.w > 1.5) {
            return vec4<f32>(cf, cb, clamp(f / 200.0, 0.0, 1.0), 1.0);
        }
        c = mix(c, textureSampleLevel(t_dof_fg, s_lin, uv, 0.0).rgb, cf);
        c = mix(c, textureSampleLevel(t_dof_bg, s_lin, uv, 0.0).rgb, cb);
        // no sharpening of what is out of focus (CAS reads the sharp input)
        dofk = max(cf, cb);
    }

    // clarity: local contrast against the blurred surroundings (1/32 size),
    // in stops, on detail within about a stop of its surroundings only - a
    // tree against the sky (a big step) gets no halo. Half on the sky.
    let sky = textureLoad(t_depth, min(vec2<i32>(uv * vec2<f32>(textureDimensions(t_depth))),
                                       vec2<i32>(textureDimensions(t_depth)) - vec2<i32>(1)), 0) >= 0.99999;
    if (u.look.x > 0.0) {
        let m = lum(textureSampleLevel(t_local, s_lin, uv, 0.0).rgb);
        let lr = log2(max(lum(c), 1e-5) / max(m, 1e-5));
        let wd = max(u.look.y * 2.0, 0.05);
        var k = u.look.x * (1.0 - dofk);
        if (sky) {
            k = k * 0.5;
        }
        // brightening at half: the sunlit sand must not blow out
        var e = k * lr * exp(-lr * lr / (2.0 * wd * wd));
        if (e > 0.0) {
            e = e * 0.5;
        }
        c = c * exp2(clamp(e, -0.5, 0.25));
    }
    // sun shafts (mode 5's rays): over the sky itself only a little (the
    // veil already glows there), the rays show across what stands in front
    if (u.shaft.w > 0.5) {
        var ks = u.shaft.z;
        if (sky) {
            ks = ks * 0.25;
        }
        // a 4-tap box over the 1/4-size rays: their start noise must not
        // show as hatching (TAA has already run)
        let ts = 1.0 / vec2<f32>(textureDimensions(t_shaft));
        let sh = textureSampleLevel(t_shaft, s_lin, uv + ts * vec2<f32>(-0.75, -0.75), 0.0).rgb
               + textureSampleLevel(t_shaft, s_lin, uv + ts * vec2<f32>(0.75, -0.75), 0.0).rgb
               + textureSampleLevel(t_shaft, s_lin, uv + ts * vec2<f32>(-0.75, 0.75), 0.0).rgb
               + textureSampleLevel(t_shaft, s_lin, uv + ts * vec2<f32>(0.75, 0.75), 0.0).rgb;
        c += sh * (0.25 * ks);
    }

    // the veil (in linear light with HDR), then the one tonemap
    var bloom0 = textureSampleLevel(t_b, s_lin, uv, 0.0).rgb;
    if (u.hdr.w > 0.5) {
        bloom0 = veil_bspline(uv);
    }
    bloom0 = bloom0 * u.veil.y;
    c = tm(c + bloom0);
    // contrast-adaptive sharpening (AMD CAS, the game's sharpen setting),
    // on display values
    if (u.fx.z > 0.0) {
        let tp = u.dst.zw;
        let n = tm(textureLoad(t_a, clamp(px + vec2<i32>(0, -1), vec2<i32>(0), maxp), 0).rgb
                   + textureSampleLevel(t_b, s_lin, uv - vec2<f32>(0.0, tp.y), 0.0).rgb * u.veil.y);
        let s = tm(textureLoad(t_a, clamp(px + vec2<i32>(0, 1), vec2<i32>(0), maxp), 0).rgb
                   + textureSampleLevel(t_b, s_lin, uv + vec2<f32>(0.0, tp.y), 0.0).rgb * u.veil.y);
        let e = tm(textureLoad(t_a, clamp(px + vec2<i32>(1, 0), vec2<i32>(0), maxp), 0).rgb
                   + textureSampleLevel(t_b, s_lin, uv + vec2<f32>(tp.x, 0.0), 0.0).rgb * u.veil.y);
        let w = tm(textureLoad(t_a, clamp(px + vec2<i32>(-1, 0), vec2<i32>(0), maxp), 0).rgb
                   + textureSampleLevel(t_b, s_lin, uv - vec2<f32>(tp.x, 0.0), 0.0).rgb * u.veil.y);
        let mn = min(c, min(min(n, s), min(e, w)));
        let mx = max(c, max(max(n, s), max(e, w)));
        // ⚠ 1 - mx for this 5-tap cross: AMD's 2 - mx is for the 3x3 form,
        // whose min and max are summed over cross + corners (doubled); with
        // 2 - mx the limiter never engaged on bright edges (sky against
        // trees, white lines), which over-sharpened and clipped
        let amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, vec3<f32>(1e-4)), vec3<f32>(0.0), vec3<f32>(1.0)));
        let peak = -1.0 / mix(8.0, 5.0, u.fx.z);
        let wt = amp * peak * (1.0 - dofk);
        c = clamp((c + (n + s + e + w) * wt) / (1.0 + 4.0 * wt), vec3<f32>(0.0), vec3<f32>(1.0));
    }


    // eye-focus vignette (eyeFocusVignette / eyeFocusPow / eyeFocusIntensity)
    if (u.fx.x > 0.0) {
        let d = (uv - vec2<f32>(0.5)) * vec2<f32>(u.dst.x * u.dst.w, 1.0);
        let r = clamp(length(d) * 1.25, 0.0, 1.0);
        let v = 1.0 - u.fx.x * 0.32 * pow(r, 1.0 / max(u.fx.y, 0.05) + 1.0);
        c = c * v;
    }
    c = max(c, vec3<f32>(0.0));
    // The swapchain is 8-bit sRGB: a triangular (TPDF) dither of +-1 code
    // value, in the encoded domain, so sky, fog and dark gradients do not
    // band. EVOVIEW_DITHER=0 off.
    if (DITHER_ON) {
        let p = vec2<u32>(in.pos.xy);
        let r1 = vec3<f32>(hash3(p)) * (1.0 / 4294967296.0);
        let r2 = vec3<f32>(hash3(p + vec2<u32>(7919u, 104729u))) * (1.0 / 4294967296.0);
        let e = clamp(lin_to_srgb(min(c, vec3<f32>(1.0))) + (r1 - r2) / 255.0, vec3<f32>(0.0), vec3<f32>(1.0));
        c = srgb_to_lin(e);
    }
    return vec4<f32>(c, 1.0);
}

/// Three decorrelated 32-bit hashes of a pixel (pcg3d).
fn hash3(p: vec2<u32>) -> vec3<u32> {
    var v = vec3<u32>(p, p.x ^ p.y) * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

fn lin_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3<f32>(0.0031308));
}

fn srgb_to_lin(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}
