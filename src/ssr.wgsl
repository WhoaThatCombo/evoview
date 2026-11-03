// Screen-space reflections (ssr.rs), the game's FidelityFX SSSR approach
// (cbSSSR intersect 054a9160, lifted): a GGX-VNDF ray per 2x2 quad, a
// hierarchical min-depth march, FFX's hit validation (frustum, self-hit,
// background, back face, depth-thickness confidence squared, 5% border
// vignette), then a reprojected history and an upsampling composite that
// swaps the material's environment reflection for the hit by its confidence.
//
// Depth here is standard Z (0 near .. 1 at infinity, infinite projection);
// the game's is reversed, so its z tests flip.

struct U {
    vp: mat4x4<f32>,
    inv_vp: mat4x4<f32>,
    prev_vp: mat4x4<f32>,
    // the driven car: world -> its box now, its box -> world last frame
    obj_inv: mat4x4<f32>,
    obj_prev: mat4x4<f32>,
    // xyz eye, w frame
    eye: vec4<f32>,
    // render width, height, 1/w, 1/h
    size: vec4<f32>,
    // x roughness cutoff (GGX alpha), y thickness per metre of hit distance,
    // z max march iterations, w strength
    p0: vec4<f32>,
    // x car box in use, y top Hi-Z mip, z history reset, w history weight
    p1: vec4<f32>,
    // x metres per render pixel at 1 m (the start offset off the surface),
    // y prefilter radius cap (half-size texels)
    p2: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var o_hiz: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var t_src: texture_2d<f32>;
@group(0) @binding(4) var t_hiz: texture_2d<f32>;
@group(0) @binding(5) var t_geo: texture_2d<f32>;
@group(0) @binding(6) var t_color: texture_2d<f32>;
@group(0) @binding(7) var o_trace: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var o_dist: texture_storage_2d<r32float, write>;
@group(0) @binding(9) var t_env: texture_2d<f32>;
@group(0) @binding(10) var t_trace: texture_2d<f32>;
@group(0) @binding(11) var t_dist: texture_2d<f32>;
@group(0) @binding(12) var t_hist: texture_2d<f32>;
@group(0) @binding(13) var samp: sampler;
@group(0) @binding(14) var t_hizp: texture_2d<f32>;
@group(0) @binding(15) var o_hist: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265;

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    let f = e * 2.0 - 1.0;
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = clamp(-n.z, 0.0, 1.0);
    n = vec3<f32>(n.x + select(t, -t, n.x >= 0.0), n.y + select(t, -t, n.y >= 0.0), n.z);
    return normalize(n);
}

fn world(uv: vec2<f32>, z: f32) -> vec3<f32> {
    let w = u.inv_vp * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
    return w.xyz / w.w;
}

// (uv, z) of a world point under m; w <= 0 behind the camera
fn project(m: mat4x4<f32>, p: vec3<f32>) -> vec4<f32> {
    let c = m * vec4<f32>(p, 1.0);
    let n = c.xyz / c.w;
    return vec4<f32>(n.x * 0.5 + 0.5, 0.5 - n.y * 0.5, n.z, c.w);
}

// ---- Hi-Z: level 0 is the half-size min depth, padded with the far plane ----

@compute @workgroup_size(8, 8)
fn hiz0(@builtin(global_invocation_id) g: vec3<u32>) {
    let dim = textureDimensions(o_hiz);
    if (any(g.xy >= dim)) {
        return;
    }
    let rs = vec2<i32>(u.size.xy);
    var z = 1.0;
    for (var k = 0; k < 4; k++) {
        let p = vec2<i32>(g.xy) * 2 + vec2<i32>(k & 1, k >> 1u);
        if (all(p < rs)) {
            z = min(z, textureLoad(t_depth, p, 0));
        }
    }
    textureStore(o_hiz, vec2<i32>(g.xy), vec4<f32>(z, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn hizn(@builtin(global_invocation_id) g: vec3<u32>) {
    let dim = textureDimensions(o_hiz);
    if (any(g.xy >= dim)) {
        return;
    }
    let p = vec2<i32>(g.xy) * 2;
    let z = min(min(textureLoad(t_src, p, 0).x, textureLoad(t_src, p + vec2<i32>(1, 0), 0).x),
                min(textureLoad(t_src, p + vec2<i32>(0, 1), 0).x, textureLoad(t_src, p + vec2<i32>(1, 1), 0).x));
    textureStore(o_hiz, vec2<i32>(g.xy), vec4<f32>(z, 0.0, 0.0, 0.0));
}

// ---- trace: one ray per 2x2 quad ----

fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// Heitz 2018 visible-normal sample (FFX's FFX_SSSR_SampleGGXVNDF), tangent space
fn sample_vndf(ve: vec3<f32>, a: f32, u1: f32, u2: f32) -> vec3<f32> {
    let vh = normalize(vec3<f32>(a * ve.x, a * ve.y, ve.z));
    let lensq = vh.x * vh.x + vh.y * vh.y;
    let t1 = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(-vh.y, vh.x, 0.0) * inverseSqrt(lensq), lensq > 0.0);
    let t2 = cross(vh, t1);
    let r = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let p1 = r * cos(phi);
    let s = 0.5 * (1.0 + vh.z);
    let p2 = (1.0 - s) * sqrt(1.0 - p1 * p1) + s * r * sin(phi);
    let nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    return normalize(vec3<f32>(a * nh.x, a * nh.y, max(0.0, nh.z)));
}

@compute @workgroup_size(8, 8)
fn trace(@builtin(global_invocation_id) g: vec3<u32>) {
    let hs = textureDimensions(o_trace);
    if (any(g.xy >= hs)) {
        return;
    }
    let q = vec2<i32>(g.xy);
    let fr = u32(u.eye.w);
    // the quad's pixel this frame: each of the four in turn (FFX traces one
    // per quad and copies it to the other three)
    let k = (fr + u32(q.x + q.y * 2)) & 3u;
    let p = min(q * 2 + vec2<i32>(i32(k & 1u), i32(k >> 1u)), vec2<i32>(u.size.xy) - 1);
    var res = vec4<f32>(0.0);
    var hd = 0.0;
    let geo = textureLoad(t_geo, p, 0);
    let a = geo.z * geo.z;
    let w = textureLoad(t_env, p, 0).a;
    if (geo.w > 0.1 && a < u.p0.x && w > 1e-3) {
        let d = textureLoad(t_depth, p, 0);
        let uv = (vec2<f32>(p) + 0.5) * u.size.zw;
        let pw = world(uv, d);
        let n = oct_decode(geo.xy);
        let vd = normalize(pw - u.eye.xyz);
        var rd = reflect(vd, n);
        if (a > 1e-4) {
            // stochastic: a GGX lobe sample, the history averages them
            let sg = select(-1.0, 1.0, n.z >= 0.0);
            let aa = -1.0 / (sg + n.z);
            let bb = n.x * n.y * aa;
            let tx = vec3<f32>(1.0 + sg * n.x * n.x * aa, sg * bb, -sg * n.x);
            let ty = vec3<f32>(bb, sg + n.y * n.y * aa, -n.y);
            let ve = vec3<f32>(dot(-vd, tx), dot(-vd, ty), dot(-vd, n));
            let fq = vec2<f32>(q) + vec2<f32>(f32(fr % 64u) * 5.588238);
            let u1 = fract(ign(fq) + f32(fr) * 0.618034);
            let u2 = fract(ign(fq.yx + vec2<f32>(47.0, 13.0)) + f32(fr) * 0.754878);
            let h = sample_vndf(ve, a, u1, u2);
            let rr = reflect(vd, tx * h.x + ty * h.y + n * h.z);
            if (dot(rr, n) > 0.0) {
                rd = rr;
            }
        }
        // start two pixels off the surface: level 0 is the quad's NEAREST
        // depth, and a sloped surface would hit itself in its own cell
        let de = length(pw - u.eye.xyz);
        let ow = pw + n * (de * u.p2.x * 2.0);
        let c0 = u.vp * vec4<f32>(ow, 1.0);
        let dw = (u.vp * vec4<f32>(rd, 0.0)).w;
        var len = max(de * 0.25, 1.0);
        if (dw < 0.0) {
            len = min(len, 0.5 * c0.w / -dw);
        }
        let os = project(u.vp, ow);
        let es = project(u.vp, ow + rd * len);
        let half = u.size.xy * 0.5;
        let o = vec3<f32>(os.xy * half, os.z);
        let dir = vec3<f32>(es.xy * half, es.z) - o;
        let inv = vec3<f32>(select(1e30, 1.0 / dir.x, dir.x != 0.0), select(1e30, 1.0 / dir.y, dir.y != 0.0),
                            select(1e30, 1.0 / dir.z, dir.z != 0.0));
        let fl = select(vec2<f32>(0.0), vec2<f32>(1.0), dir.xy >= vec2<f32>(0.0));
        let eps = select(vec2<f32>(-0.005), vec2<f32>(0.005), dir.xy >= vec2<f32>(0.0));
        // FFX_SSSR_InitialAdvanceRay: to the edge of the start cell
        let t0 = (floor(o.xy) + fl + eps - o.xy) * inv.xy;
        var t = min(t0.x, t0.y);
        var pos = o + dir * t;
        var mip = 0;
        let top = i32(u.p1.y);
        let maxi = u32(u.p0.z);
        var hit = false;
        var i = 0u;
        loop {
            if (i >= maxi || any(pos.xy < vec2<f32>(0.0)) || any(pos.xy >= half) || pos.z >= 1.0 || pos.z <= 0.0) {
                break;
            }
            // FFX_SSSR_AdvanceRay: to the nearer of the cell's far edges and
            // the depth plane; up a level over empty cells, down otherwise
            let sc = exp2(f32(mip));
            let c = floor(pos.xy / sc);
            let sz = textureLoad(t_hiz, vec2<i32>(c), mip).x;
            var t3 = (vec3<f32>((c + fl) * sc + eps, sz) - o) * inv;
            t3.z = select(1e30, t3.z, dir.z > 0.0);
            let tmin = min(min(t3.x, t3.y), t3.z);
            let above = sz > pos.z;
            let skipped = above && tmin != t3.z;
            if (above) {
                t = tmin;
                pos = o + dir * t;
            }
            if (skipped) {
                mip = min(mip + 1, top);
            } else {
                mip -= 1;
            }
            i += 1u;
            if (mip < 0) {
                hit = true;
                break;
            }
        }
        // FFX_SSSR_ValidateHit
        let huv = pos.xy / half;
        if (hit && all(huv >= vec2<f32>(0.0)) && all(huv <= vec2<f32>(1.0))
            && any(abs(pos.xy - o.xy) >= vec2<f32>(1.0))) {
            let hp = min(vec2<i32>(pos.xy * 2.0), vec2<i32>(u.size.xy) - 1);
            let sz = textureLoad(t_hiz, vec2<i32>(pos.xy), 0).x;
            let hg = textureLoad(t_geo, hp, 0);
            // not the background, not a back face
            if (sz < 1.0 && !(hg.w > 0.1 && dot(oct_decode(hg.xy), rd) > 0.0)) {
                let ws = world(huv, sz);
                let wh = world(huv, pos.z);
                // FFX holds the depth thickness constant; the far scenery's
                // depth steps here are metres, so it grows with distance
                let thick = 0.015 + u.p0.y * length(ws - u.eye.xyz);
                var conf = 1.0 - smoothstep(0.0, thick, length(ws - wh));
                conf = conf * conf;
                let fv = 0.05 * vec2<f32>(u.size.y * u.size.z, 1.0);
                let bd = smoothstep(vec2<f32>(0.0), fv, huv) * (vec2<f32>(1.0) - smoothstep(vec2<f32>(1.0) - fv, vec2<f32>(1.0), huv));
                conf *= bd.x * bd.y;
                let col = textureLoad(t_color, hp, 0).rgb;
                res = vec4<f32>(max(col, vec3<f32>(0.0)) * conf, conf);
                hd = length(wh - pw);
            }
        }
    }
    textureStore(o_trace, q, res);
    textureStore(o_dist, q, vec4<f32>(hd, 0.0, 0.0, 0.0));
}

// ---- prefilter: FFX's spatial pass, a depth / normal-aware disk around the
// quad sized by the lobe's footprint (alpha x hit distance over the pixel's)

@group(0) @binding(16) var o_filt: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn prefilter(@builtin(global_invocation_id) g: vec3<u32>) {
    let hs = textureDimensions(o_filt);
    if (any(g.xy >= hs)) {
        return;
    }
    let q = vec2<i32>(g.xy);
    let hm = vec2<i32>(hs) - 1;
    let c = textureLoad(t_trace, q, 0);
    let geo = textureLoad(t_geo, min(q * 2, vec2<i32>(u.size.xy) - 1), 0);
    let a = geo.z * geo.z;
    let z = textureLoad(t_hiz, q, 0).x;
    if (geo.w < 0.1 || a >= u.p0.x || z >= 1.0) {
        textureStore(o_filt, q, c);
        return;
    }
    let n = oct_decode(geo.xy);
    let iz = max(1.0 - z, 1e-7);
    // the lobe's footprint in half-size texels (FFX's prefilter is ~4 px)
    let uv = (vec2<f32>(q) * 2.0 + 1.0) * u.size.zw;
    let de = length(world(uv, z) - u.eye.xyz);
    let hd = textureLoad(t_dist, q, 0).x;
    let foot = select(a * 40.0, hd * a / max(de * u.p2.x * 2.0, 1e-5), hd > 0.0);
    let rad = clamp(foot, 0.0, u.p2.y);
    if (rad < 0.75) {
        textureStore(o_filt, q, c);
        return;
    }
    // 12 taps of a golden-angle spiral, turned per texel and frame
    let rot = ign(vec2<f32>(q) + f32(u32(u.eye.w) % 16u) * 7.31) * 6.2831853;
    var acc = c;
    var wsum = 1.0;
    for (var i = 0; i < 12; i++) {
        let fi = f32(i) + 0.5;
        let ang = fi * 2.3999632 + rot;
        let o = vec2<f32>(cos(ang), sin(ang)) * rad * sqrt(fi / 12.0);
        let tp = clamp(q + vec2<i32>(round(o)), vec2<i32>(0), hm);
        let tg = textureLoad(t_geo, min(tp * 2, vec2<i32>(u.size.xy) - 1), 0);
        let tz = max(1.0 - textureLoad(t_hiz, tp, 0).x, 1e-7);
        let wz = exp(-abs(tz - iz) / iz * 20.0);
        let wn = pow(clamp(dot(oct_decode(tg.xy), n), 0.0, 1.0), 16.0);
        let w = wz * wn * select(0.0, 1.0, tg.w > 0.1);
        acc += textureLoad(t_trace, tp, 0) * w;
        wsum += w;
    }
    textureStore(o_filt, q, acc / wsum);
}

// ---- temporal: the reprojected history, clipped to this frame's neighbours ----

@compute @workgroup_size(8, 8)
fn temporal(@builtin(global_invocation_id) g: vec3<u32>) {
    let hs = textureDimensions(o_hist);
    if (any(g.xy >= hs)) {
        return;
    }
    let q = vec2<i32>(g.xy);
    let hm = vec2<i32>(hs) - 1;
    let cur = textureLoad(t_trace, q, 0);
    var m1 = vec4<f32>(0.0);
    var m2 = vec4<f32>(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let s = textureLoad(t_trace, clamp(q + vec2<i32>(dx, dy), vec2<i32>(0), hm), 0);
            m1 += s;
            m2 += s * s;
        }
    }
    m1 = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - m1 * m1, vec4<f32>(0.0)));
    let z = textureLoad(t_hiz, q, 0).x;
    var out = cur;
    if (z < 1.0 && u.p1.z < 0.5) {
        let uv = (vec2<f32>(q) * 2.0 + 1.0) * u.size.zw;
        let pw = world(uv, z);
        let geo = textureLoad(t_geo, min(q * 2, vec2<i32>(u.size.xy) - 1), 0);
        let car = u.p1.x > 0.5 && abs(geo.w - 0.6667) < 0.1;
        var pp = pw;
        var pv = pw;
        let hd = textureLoad(t_dist, q, 0).x;
        if (car) {
            // rides with the car (taa.rs's box, the same motion)
            pp = (u.obj_prev * (u.obj_inv * vec4<f32>(pw, 1.0))).xyz;
            pv = pp;
        } else if (cur.a > 0.0 && hd > 0.0) {
            // a mirror image moves with its virtual point behind the surface
            pv = pw + normalize(pw - u.eye.xyz) * hd;
        }
        let ps = project(u.prev_vp, pp);
        let pvs = project(u.prev_vp, pv);
        var ok = ps.w > 0.0 && pvs.w > 0.0 && all(ps.xy >= vec2<f32>(0.0)) && all(ps.xy <= vec2<f32>(1.0))
                 && all(pvs.xy >= vec2<f32>(0.0)) && all(pvs.xy <= vec2<f32>(1.0));
        if (ok) {
            // disocclusion: last frame's nearest depth there against where the
            // surface was ((1 - z) is 1/view depth up to the near plane)
            let zp = textureLoad(t_hizp, clamp(vec2<i32>(ps.xy * vec2<f32>(hs)), vec2<i32>(0), hm), 0).x;
            ok = abs(zp - ps.z) < 0.06 * max(1.0 - ps.z, 1e-7) || car;
        }
        if (ok) {
            var h = textureSampleLevel(t_hist, samp, pvs.xy, 0.0);
            h = clamp(h, m1 - sigma * 1.5, m1 + sigma * 1.5);
            out = mix(h, cur, u.p1.w);
        } else {
            // no history: this frame's neighbourhood, not one noisy ray
            out = mix(m1, cur, 0.5);
        }
    }
    textureStore(o_hist, q, out);
}

// ---- composite: add (hit - environment) x confidence onto the scene ----

struct VOut {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> VOut {
    var o: VOut;
    o.pos = vec4<f32>(f32(i32(i & 1u) * 4 - 1), f32(i32(i >> 1u) * 4 - 1), 0.0, 1.0);
    return o;
}

@fragment
fn fs_composite(in: VOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.pos.xy);
    let geo = textureLoad(t_geo, p, 0);
    let a = geo.z * geo.z;
    let thr = u.p0.x;
    if (geo.w < 0.1 || a >= thr) {
        discard;
    }
    let env = textureLoad(t_env, p, 0);
    if (env.a <= 1e-3) {
        discard;
    }
    // FFX cuts at the threshold; fade into it instead of a hard edge
    let fade = 1.0 - smoothstep(thr * 0.5, thr, a);
    // depth-aware 2x2 upsample from the half-size history
    let hs = vec2<f32>(textureDimensions(t_hist));
    let hm = vec2<i32>(hs) - 1;
    let hc = in.pos.xy * 0.5 - 0.5;
    let b = vec2<i32>(floor(hc));
    let f = hc - floor(hc);
    let z = textureLoad(t_depth, p, 0);
    let iz = max(1.0 - z, 1e-7);
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1u);
        let tp = clamp(b + o, vec2<i32>(0), hm);
        let bw = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        let tz = textureLoad(t_hiz, tp, 0).x;
        // relative view-depth difference ((1 - z) ~ 1 / depth)
        let rel = abs(max(1.0 - tz, 1e-7) - iz) / iz;
        let wt = bw * exp(-rel * 20.0) + 1e-4 * bw;
        acc += textureLoad(t_hist, tp, 0) * wt;
        wsum += wt;
    }
    let s = acc / max(wsum, 1e-6);
    let r = fade * u.p0.w * (env.a * s.rgb - env.rgb * s.a);
    return vec4<f32>(r, 0.0);
}
