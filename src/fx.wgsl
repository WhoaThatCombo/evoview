// Effect passes that are not material shading (fxgpu.rs):
//   particles  - camera-facing billboards from the game's particle emitters
//   bloom      - downsample / blur / composite of the emissive glow buffer
//   dash       - the car's display pages, drawn into their screen textures

// ---- shared: the camera uniform (prefix of shader.wgsl's Camera) --------
struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    ground: vec4<f32>,
    extent: vec4<f32>,
    mode: vec4<f32>,
    light_vp: mat4x4<f32>,
    shadow: vec4<f32>,
    grade: vec4<f32>,
    fx: array<vec4<f32>, 64>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var t_lut: texture_3d<f32>;
@group(0) @binding(2) var lin_samp: sampler;

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// = shader.wgsl tonemap(): exposure, log2 shaper (grade.rs), the game's LUT
fn tonemap(c: vec3<f32>) -> vec3<f32> {
    // HDR pipeline: exposed linear light, graded once in the post pass
    // (without this branch particles went through the LUT here AND there)
    if (camera.grade.y > 1.5) {
        // capped inside fp16 as hdr_out() in shading_common.wgsl
        let e = c * camera.grade.x;
        return select(clamp(e, vec3<f32>(0.0), vec3<f32>(32000.0)), vec3<f32>(0.0), e != e);
    }
    if (camera.grade.y < 0.5) {
        return aces(c);
    }
    let x = max(c * camera.grade.x, vec3<f32>(0.0));
    var uvw = clamp((log2(x + vec3<f32>(0.0009765625)) + vec3<f32>(10.0)) / 16.0,
                    vec3<f32>(0.0), vec3<f32>(1.0));
    uvw = uvw * (31.0 / 32.0) + vec3<f32>(0.5 / 32.0);
    uvw.z = (uvw.z + camera.grade.z) * 0.5;
    return textureSampleLevel(t_lut, lin_samp, uvw, 0.0).rgb;
}

/// The ceiling of what a pass may write: display 1.0, or (HDR) anything
/// the post pass will tonemap.
fn out_max() -> vec3<f32> {
    return vec3<f32>(select(1.0, 32000.0, camera.grade.y > 1.5));
}

// ---- particles ------------------------------------------------------------
// group 1: the emitter's colour map, opacity map, sampler, params
struct PMat {
    // rgb basecolour, w emissive intensity
    base: vec4<f32>,
    // x cols, y rows, z flipbook on, w 1 = additive (emissive) / 0 = lit smoke
    flip: vec4<f32>,
    // x has colour map, y has opacity map, z gain, w 1 = opacity in alpha
    maps: vec4<f32>,
    // x the material's DepthFade_Distance, m
    fade: vec4<f32>,
    // x water: Refraction_Roughness; y the mist's Ambient_Intensity; z its
    // alpha gain (EVOVIEW_SPRAY) - 1 for everything else; w 1 = the game's
    // water shading (EVOVIEW_SPRAY_GAME)
    refr: vec4<f32>,
};
@group(1) @binding(0) var<uniform> pmat: PMat;
@group(1) @binding(1) var t_pcol: texture_2d<f32>;
@group(1) @binding(2) var t_pop: texture_2d<f32>;
@group(1) @binding(3) var psamp: sampler;
// the scene's depth, for soft particles (fs_particle_soft; own pass)
@group(1) @binding(4) var t_pnorm: texture_2d<f32>;
@group(2) @binding(0) var t_scene_depth: SCENE_DEPTH_TEX;
// the scene behind, for water to refract (1 x 1 where it cannot be read)
@group(2) @binding(1) var t_refr: texture_2d<f32>;

struct PIn {
    @location(0) corner: vec2<f32>,
    // per particle
    @location(1) pos_size: vec4<f32>,
    // x roll (rad), y life fraction, z alpha, w unused
    @location(2) misc: vec4<f32>,
    @location(3) tint: vec4<f32>,
    // xyz world velocity, w streak seconds (0 = round billboard)
    @location(4) vel: vec4<f32>,
};

struct POut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) life: f32,
    @location(2) alpha: f32,
    @location(3) tint: vec3<f32>,
    // view depth (clip w), for the soft fade
    @location(4) vz: f32,
    // the sun in the card's own frame (its texture x, y and the eye axis)
    @location(5) sun: vec3<f32>,
};

@vertex
fn vs_particle(in: PIn) -> POut {
    var o: POut;
    let p = in.pos_size.xyz;
    let to_eye = normalize(camera.eye.xyz - p);
    var right = cross(vec3<f32>(0.0, 1.0, 0.0), to_eye);
    if (dot(right, right) < 1e-6) {
        right = vec3<f32>(1.0, 0.0, 0.0);
    }
    right = normalize(right);
    let up = cross(to_eye, right);
    let c = cos(in.misc.x);
    let s = sin(in.misc.x);
    let k = vec2<f32>(in.corner.x * c - in.corner.y * s, in.corner.x * s + in.corner.y * c);
    var w = p + (right * k.x + up * k.y) * in.pos_size.w;
    // a streak (sparks): along the motion seen from the eye, trailing behind
    let vv = in.vel.xyz - to_eye * dot(in.vel.xyz, to_eye);
    if (in.vel.w > 0.0 && dot(vv, vv) > 1e-4) {
        let ax = normalize(vv);
        let side = normalize(cross(to_eye, ax));
        // at most 35 cm: a spark flying past the lens was a white bar
        let len = min(length(vv) * in.vel.w, 0.35);
        let along = in.corner.y * (in.pos_size.w + len * 0.5) - len * 0.5;
        w = p + ax * along + side * in.corner.x * in.pos_size.w * 0.45;
    }
    o.clip = camera.view_proj * vec4<f32>(w, 1.0);
    o.vz = o.clip.w;
    // the sun: the shadow map's depth axis points away from it
    let l = -normalize(vec3<f32>(camera.light_vp[0].z, camera.light_vp[1].z, camera.light_vp[2].z));
    let tx = right * c + up * s;
    let ty = -right * s + up * c;
    o.sun = vec3<f32>(dot(l, tx), dot(l, ty), dot(l, to_eye));
    o.uv = in.corner * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    o.life = in.misc.y;
    o.alpha = in.misc.z;
    o.tint = in.tint.rgb;
    return o;
}

// The frame cell of flipbook frame `f` (the game's 0x035331f0: cell =
// (frac(uv) + (f mod cols, f / cols)) / (cols, rows))
fn cell_uv(uv: vec2<f32>, f: f32) -> vec2<f32> {
    let cx = f - floor(f / pmat.flip.x) * pmat.flip.x;
    let cy = floor(f / pmat.flip.x);
    // inset a little so filtering never reaches the next frame
    let iu = clamp(uv, vec2<f32>(0.01), vec2<f32>(0.99));
    return (vec2<f32>(cx, cy) + iu) / pmat.flip.xy;
}

// One map through the flipbook with the game's frame blending: life picks
// a position in (frames - 1), the two frames either side are mixed by its
// fraction. Sampled at mip bias -3, clamped (as the game: samLinearClamp,
// globalMaterialMipBias - 3) - a wrapping, mipped read smeared the cells
// into each other.
fn flip_sample(t: texture_2d<f32>, uv: vec2<f32>, life: f32) -> vec4<f32> {
    if (pmat.flip.z < 0.5) {
        return textureSampleBias(t, psamp, uv, -3.0);
    }
    let n = pmat.flip.x * pmat.flip.y;
    let pos = clamp(life, 0.0, 0.9999) * (n - 1.0);
    let f0 = floor(pos);
    let f1 = min(f0 + 1.0, n - 1.0);
    let a = textureSampleBias(t, psamp, cell_uv(uv, f0), -3.0);
    let b = textureSampleBias(t, psamp, cell_uv(uv, f1), -3.0);
    return mix(a, b, pos - f0);
}

fn particle_colour(in: POut) -> vec4<f32> {
    var col = pmat.base.rgb;
    var a = in.alpha;
    if (pmat.maps.x > 0.5) {
        let t = flip_sample(t_pcol, in.uv, in.life);
        col = col * t.rgb;
        // the colour map's own alpha counts too (spray_1's droplets inside
        // the card's soft opacity), except a packed sheet: that is the
        // opacity map itself
        if (pmat.maps.y < 0.5 || pmat.maps.w < 0.5) {
            a = a * t.a;
        }
    }
    if (pmat.maps.y > 0.5) {
        let o = flip_sample(t_pop, in.uv, in.life);
        // maps.w: density in alpha (packed BC3 sheets), else in red
        a = a * select(o.r, o.a, pmat.maps.w > 0.5);
    }
    // no map at all: a soft round puff instead of a square card
    if (pmat.maps.x < 0.5 && pmat.maps.y < 0.5) {
        let r = length(in.uv * 2.0 - vec2<f32>(1.0));
        a = a * (1.0 - smoothstep(0.2, 1.0, r));
    }
    // the card's own border fades to nothing
    let edge = min(in.uv, vec2<f32>(1.0) - in.uv);
    a = a * smoothstep(0.0, 0.12, min(edge.x, edge.y));
    col = col * in.tint;
    return vec4<f32>(col, clamp(a, 0.0, 1.0));
}

/// Smoke and dust outdoors with a normal map: the sun across the puff's
/// relief, a soft wrap around 1 (0.6 shaded .. 1.4 lit) so the mean stays.
fn sun_shade(in: POut) -> f32 {
    if (pmat.fade.w < 0.5 || pmat.fade.y > 0.0 || camera.fx[63].z <= 0.0) {
        return 1.0;
    }
    let t = flip_sample(t_pnorm, in.uv, in.life).xyz * 2.0 - vec3<f32>(1.0);
    let n = normalize(vec3<f32>(t.x, t.y, max(t.z, 0.05)));
    return 0.6 + 0.8 * clamp(dot(n, normalize(in.sun)) * 0.5 + 0.5, 0.0, 1.0);
}

// Flames: emissive, additive. Smoke / dust: lit by the studio ambient, alpha.
@fragment
fn fs_particle(in: POut) -> @location(0) vec4<f32> {
    let c = particle_colour(in);
    if (pmat.flip.w > 0.5) {
        let e = c.rgb * pmat.base.w * pmat.maps.z * c.a;
        return vec4<f32>(tonemap(e), 0.0);
    }
    // fx[63].z > 0: outdoors, the sky's light on smoke (render.rs) instead
    // of the studio ambient - at track exposure that read pitch black
    let amb = select(camera.fx[0].x, camera.fx[63].z, camera.fx[63].z > 0.0);
    let lit = c.rgb * pmat.maps.z * amb * sun_shade(in);
    // water with no scene copy to refract: its transmitted share shows the
    // scene through unbent, which is the lit share at that much less alpha
    return vec4<f32>(tonemap(lit * pmat.refr.y), min(c.a * pmat.refr.z, 1.0) * (1.0 - pmat.fade.y));
}

/// 0 where the sprite is behind the scene, rising to 1 over 0.8 m in front
/// of it. Depth is the infinite-far projection's: ndc = 1 - near / w, so the
/// scene's w is the sprite's w scaled by (1 - z_sprite) / (1 - z_scene).
fn soft_fade(in: POut) -> f32 {
    return soft_fade_at(in, vec2<i32>(in.clip.xy));
}

fn soft_fade_at(in: POut, px: vec2<i32>) -> f32 {
    let d = textureLoad(t_scene_depth, px, 0);
    if (d >= 0.9999999) {
        return 1.0;
    }
    let ws = in.vz * (1.0 - in.clip.z) / max(1.0 - d, 1e-7);
    return clamp((ws - in.vz) / max(pmat.fade.x, 0.05), 0.0, 1.0);
}

@fragment
fn fs_particle_soft(in: POut) -> @location(0) vec4<f32> {
    let c = particle_colour(in);
    let f = soft_fade(in);
    if (pmat.flip.w > 0.5) {
        let e = c.rgb * pmat.base.w * pmat.maps.z * c.a * f;
        return vec4<f32>(tonemap(e), 0.0);
    }
    let amb = select(camera.fx[0].x, camera.fx[63].z, camera.fx[63].z > 0.0);
    let lit = c.rgb * pmat.maps.z * amb * sun_shade(in);
    return vec4<f32>(tonemap(lit * pmat.refr.y), min(c.a * f * pmat.refr.z, 1.0) * (1.0 - pmat.fade.y));
}

// Half resolution (fxgpu draw_particles_half): the fade is the mean over the
// 2x2 full-res depths under this texel, so a sprite behind the car's edge
// covers that texel partly instead of in 2-pixel steps.
@fragment
fn fs_particle_soft_half(in: POut) -> @location(0) vec4<f32> {
    let c = particle_colour(in);
    // most of a soft puff is near-transparent: skip the depth reads and
    // the refraction taps there (rain spray was 7.8 ms at 3440x1440)
    if (c.a * pmat.refr.z < 0.004) {
        discard;
    }
    let m = vec2<i32>(textureDimensions(t_scene_depth)) - vec2<i32>(1);
    let p = vec2<i32>(in.clip.xy) * 2;
    let f = 0.25 * (soft_fade_at(in, min(p, m)) + soft_fade_at(in, min(p + vec2<i32>(1, 0), m))
                  + soft_fade_at(in, min(p + vec2<i32>(0, 1), m)) + soft_fade_at(in, min(p + vec2<i32>(1, 1), m)));
    if (pmat.flip.w > 0.5) {
        let e = c.rgb * pmat.base.w * pmat.maps.z * c.a * f;
        return vec4<f32>(tonemap(e), 0.0);
    }
    let amb = select(camera.fx[0].x, camera.fx[63].z, camera.fx[63].z > 0.0);
    let lit = c.rgb * pmat.maps.z * amb * sun_shade(in);
    // Water (Refraction_Transmission, fade.y): that share of the spray is
    // the scene behind it bent by its normal map, tinted a little by the
    // water; the rest stays lit mist. t_refr is already display-referred.
    let nrm = flip_sample(t_pnorm, in.uv, in.life).xy * 2.0 - vec2<f32>(1.0);
    let rd = vec2<f32>(textureDimensions(t_refr));
    if (pmat.fade.y > 0.0 && rd.x > 1.5) {
        let suv = in.clip.xy * 2.0 / vec2<f32>(textureDimensions(t_scene_depth));
        let bend = select(vec2<f32>(0.0), nrm, pmat.fade.w > 0.5);
        let off = bend * vec2<f32>(1.0, -1.0) * 0.012 * (1.0 + min(pmat.fade.z, 1.0) * 2.0);
        // Refraction_Roughness blurs what is seen through it: four taps on
        // a disc turned per pixel (up to ~1.5% of the screen at roughness 1;
        // TAA-free half-res, the per-pixel spin hides the sparse kernel)
        let rad = pmat.refr.x * 0.015 * vec2<f32>(rd.y / rd.x, 1.0);
        let spin = fract(sin(dot(in.clip.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453) * 6.2832;
        var refr = textureSampleLevel(t_refr, psamp, clamp(suv + off, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
        if (pmat.refr.x > 0.01) {
            for (var k = 0; k < 4; k = k + 1) {
                let a = spin + f32(k) * 2.3999;
                let r = sqrt((f32(k) + 0.5) / 4.0);
                let q = suv + off + vec2<f32>(cos(a), sin(a)) * r * rad;
                refr += textureSampleLevel(t_refr, psamp, clamp(q, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
            }
            refr = refr / 5.0;
        }
        // the transmitted share of the scene, plus the light the mist
        // itself scatters (Ambient_Intensity) on top - not instead: a
        // see-through veil that still lightens what is behind it
        // (0.7 of it: piled up in a long trail the full light glowed)
        let mist = tonemap(lit * pmat.refr.y) * 0.7;
        let a = min(c.a * f * pmat.refr.z, 1.0);
        if (pmat.refr.w > 0.5) {
            // the game's (particle shader 035331f0): the scene through it
            // x its colour, plus its lit diffuse
            return vec4<f32>(min(refr * c.rgb * pmat.fade.y + tonemap(lit), out_max()), a);
        }
        return vec4<f32>(min(refr * pmat.fade.y + mist, out_max()), a);
    }
    return vec4<f32>(tonemap(lit * pmat.refr.y), min(c.a * f * pmat.refr.z, 1.0) * (1.0 - pmat.fade.y));
}

// The same flames into the HDR glow buffer (bloom source).
@fragment
fn fs_particle_glow(in: POut) -> @location(0) vec4<f32> {
    let c = particle_colour(in);
    if (pmat.flip.w < 0.5) {
        discard;
    }
    return vec4<f32>(c.rgb * pmat.base.w * pmat.maps.z * c.a * camera.grade.x, 0.0);
}

// ---- bloom ------------------------------------------------------------------
@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var bsamp: sampler;
struct BloomP {
    // xy texel size of the SOURCE, z blur direction x, w direction y
    px: vec4<f32>,
    // x strength, y threshold (unused), z/w unused
    k: vec4<f32>,
};
@group(0) @binding(2) var<uniform> bp: BloomP;

struct FsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> FsOut {
    var o: FsOut;
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    o.clip = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}

// 4 bilinear taps = a 4x4 box, a clean 2x downsample
@fragment
fn fs_down(in: FsOut) -> @location(0) vec4<f32> {
    let d = bp.px.xy;
    var c = textureSample(t_src, bsamp, in.uv + vec2<f32>(-d.x, -d.y)).rgb;
    c += textureSample(t_src, bsamp, in.uv + vec2<f32>(d.x, -d.y)).rgb;
    c += textureSample(t_src, bsamp, in.uv + vec2<f32>(-d.x, d.y)).rgb;
    c += textureSample(t_src, bsamp, in.uv + vec2<f32>(d.x, d.y)).rgb;
    return vec4<f32>(c * 0.25, 1.0);
}

// 9-tap Gaussian along bp.px.zw using bilinear offsets (5 fetches)
@fragment
fn fs_blur(in: FsOut) -> @location(0) vec4<f32> {
    let dir = bp.px.zw * bp.px.xy;
    var c = textureSample(t_src, bsamp, in.uv).rgb * 0.2270270270;
    c += textureSample(t_src, bsamp, in.uv + dir * 1.3846153846).rgb * 0.3162162162;
    c += textureSample(t_src, bsamp, in.uv - dir * 1.3846153846).rgb * 0.3162162162;
    c += textureSample(t_src, bsamp, in.uv + dir * 3.2307692308).rgb * 0.0702702703;
    c += textureSample(t_src, bsamp, in.uv - dir * 3.2307692308).rgb * 0.0702702703;
    return vec4<f32>(c, 1.0);
}

// Additive onto the tonemapped frame: the glow is exposed HDR, rolled off
// so a lamp core blooms white without the halo clipping.
@fragment
fn fs_composite(in: FsOut) -> @location(0) vec4<f32> {
    let c = textureSample(t_src, bsamp, in.uv).rgb * bp.k.x;
    let g = vec3<f32>(1.0) - exp(-c);
    return vec4<f32>(g, 0.0);
}

// half-res particles over the scene (blend: rgb + dst x a)
@fragment
fn fs_part_comp(in: FsOut) -> @location(0) vec4<f32> {
    return textureSample(t_src, bsamp, in.uv);
}

// ---- dash ---------------------------------------------------------------------
@group(0) @binding(0) var t_dash: texture_2d<f32>;
@group(0) @binding(1) var dsamp: sampler;

struct DIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};
struct DOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_dash(in: DIn) -> DOut {
    var o: DOut;
    o.clip = vec4<f32>(in.pos, 0.0, 1.0);
    o.uv = in.uv;
    o.color = in.color;
    return o;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_dash(in: DOut) -> @location(0) vec4<f32> {
    let t = textureSample(t_dash, dsamp, in.uv);
    // page colours are authored in sRGB; the target encodes on write
    return vec4<f32>(srgb_to_linear(in.color.rgb) * t.rgb, in.color.a * t.a);
}
