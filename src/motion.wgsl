// Per-pixel motion vectors (motion.rs). One render-size Rgba16F target:
//   xy  where this pixel's surface was last frame minus where it is now, in
//       render pixels, both unjittered (DLSS's convention: current -> previous)
//   z   that point's view depth (clip w) last frame, for disocclusion tests
//   w   1 = on a moving object (a car), 0 = static scenery / sky
// fs_cam fills every pixel with the camera's motion through the depth buffer;
// the object pass then draws the cars again with this and last frame's bones
// and instance matrices, over the pixels where they are the visible surface.

struct U {
    // this frame's world -> clip as drawn (jittered), and its inverse
    vp_jit: mat4x4<f32>,
    inv_jit: mat4x4<f32>,
    // this frame's and last frame's unjittered world -> clip
    vp: mat4x4<f32>,
    prev_vp: mat4x4<f32>,
    // xyz eye
    eye: vec4<f32>,
    // render w, h, 1/w, 1/h
    rsize: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_depth: DEPTH_TEXTURE;
// shader.wgsl's bone blocks: 256 a car, block 0 the driven car, 1 + i
// replay car i (instance facing.x = -(1 + i))
@group(0) @binding(2) var<storage, read> bones: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read> prev_bones: array<mat4x4<f32>>;
// per bone: xyz the rest origin in mesh space, w 1 = a wheel bone (spins)
struct Spin { b: array<vec4<f32>, 256> };
@group(0) @binding(4) var<uniform> spin: Spin;

fn depth_at(p: vec2<i32>) -> f32 {
    let m = vec2<i32>(i32(u.rsize.x) - 1, i32(u.rsize.y) - 1);
    return textureLoad(t_depth, clamp(p, vec2<i32>(0), m), 0);
}

// world point -> render pixel (unjittered), and its clip w
fn to_px(m: mat4x4<f32>, w: vec3<f32>) -> vec3<f32> {
    let c = m * vec4<f32>(w, 1.0);
    let ndc = c.xy / max(c.w, 1e-6);
    return vec3<f32>((ndc * vec2<f32>(0.5, -0.5) + 0.5) * u.rsize.xy, c.w);
}

struct FsVs {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> FsVs {
    var o: FsVs;
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return o;
}

/// The point the depth buffer holds at render pixel p (through the jittered
/// matrix it was drawn with, so it is exactly that surface).
fn world_at(p: vec2<i32>, d: f32) -> vec3<f32> {
    let uv = (vec2<f32>(p) + 0.5) * u.rsize.zw;
    let w = u.inv_jit * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, min(d, 0.9999999), 1.0);
    return w.xyz / w.w;
}

MV_OUT

@fragment
fn fs_cam(in: FsVs) -> MvOut {
    let p = vec2<i32>(in.pos.xy);
    let wp = world_at(p, depth_at(p));
    let now = to_px(u.vp, wp);
    let was = to_px(u.prev_vp, wp);
    if (was.z <= 0.0) {
        return mv_out(vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
    return mv_out(vec4<f32>(was.xy - now.xy, min(was.z, 60000.0), 0.0));
}

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(4) bone_idx: vec4<f32>,
    @location(5) bone_wgt: vec4<f32>,
    @location(6) inst0: vec4<f32>,
    @location(7) inst1: vec4<f32>,
    @location(8) inst2: vec4<f32>,
    @location(9) inst3: vec4<f32>,
    @location(10) facing: vec4<f32>,
    @location(11) prev0: vec4<f32>,
    @location(12) prev1: vec4<f32>,
    @location(13) prev2: vec4<f32>,
    @location(14) prev3: vec4<f32>,
};

struct ObjVs {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) prev_world: vec3<f32>,
    // last frame's position without the wheel's own spin (= prev_world off a wheel)
    @location(2) prev_free: vec3<f32>,
    @location(3) @interpolate(flat) wheel: f32,
};

fn bone_base(in: VsIn) -> u32 {
    return u32(max(-in.facing.x, 0.0) + 0.5) * 256u;
}

// shader.wgsl's linear blend skinning, with either bone table
fn skin(in: VsIn, prev: bool) -> vec3<f32> {
    let total = in.bone_wgt.x + in.bone_wgt.y + in.bone_wgt.z + in.bone_wgt.w;
    if (total <= 0.0001) {
        return in.position;
    }
    var p = vec3<f32>(0.0);
    let base = bone_base(in);
    for (var k = 0u; k < 4u; k = k + 1u) {
        let w = in.bone_wgt[k];
        if (w <= 0.0) { continue; }
        let bi = u32(max(in.bone_idx[k], 0.0));
        // as shader.wgsl: indices past its 256 skipped
        if (bi >= 256u) { continue; }
        var m = bones[base + bi];
        if (prev) {
            m = prev_bones[base + bi];
        }
        p = p + (m * vec4<f32>(in.position, 1.0)).xyz * w;
    }
    return p / total;
}

@vertex
fn vs_obj(in: VsIn) -> ObjVs {
    var o: ObjVs;
    let world = mat4x4<f32>(in.inst0, in.inst1, in.inst2, in.inst3);
    let prev = mat4x4<f32>(in.prev0, in.prev1, in.prev2, in.prev3);
    o.world = (world * vec4<f32>(skin(in, false), 1.0)).xyz;
    o.prev_world = (prev * vec4<f32>(skin(in, true), 1.0)).xyz;
    o.prev_free = o.prev_world;
    o.wheel = 0.0;
    // a wheel (rigidly on one spinning bone): the surface moved with the
    // wheel centre only
    let bi = u32(max(in.bone_idx.x, 0.0));
    // (the driven car's: the spin table is its bones)
    if (bone_base(in) == 0u && in.bone_wgt.x > 0.99 && bi < 256u && spin.b[bi].w > 0.5) {
        let c = vec4<f32>(spin.b[bi].xyz, 1.0);
        let c_now = (world * (bones[bi] * c)).xyz;
        let c_was = (prev * (prev_bones[bi] * c)).xyz;
        o.prev_free = o.world - (c_now - c_was);
        o.wheel = 1.0;
    }
    o.clip = u.vp_jit * vec4<f32>(o.world, 1.0);
    return o;
}

@fragment
fn fs_obj(in: ObjVs) -> MvOut {
    // only where this surface is what the main pass kept: the depth buffer's
    // point lies on it (a manual test, so cutout holes and MSAA depth need
    // no depth attachment here)
    let p = vec2<i32>(in.clip.xy);
    let wd = world_at(p, depth_at(p));
    let dist = length(in.world - u.eye.xyz);
    if (abs(length(wd - u.eye.xyz) - dist) > 0.01 * dist + 0.02) {
        discard;
    }
    let now = to_px(u.vp, in.world);
    var was = to_px(u.prev_vp, in.prev_world);
    // ⚠ A fast-spinning rim: its exact vectors point a spoke at another
    // spoke's place (more than WHEEL_MV_PX of spin a frame), so TAA blends
    // spokes into the gaps and the post motion blur smears them over the
    // road along the tangent - the rim faded out at speed. Above the
    // threshold the wheel moves with its centre only and w = 2 tells TAA
    // to take it from this frame (EVOVIEW_WHEEL_MV=0: exact vectors).
    if (in.wheel > 0.5 && WHEEL_MV_PX > 0.0) {
        let free = to_px(u.prev_vp, in.prev_free);
        if (length(was.xy - free.xy) > WHEEL_MV_PX) {
            return mv_out(vec4<f32>(free.xy - now.xy, min(max(free.z, 0.0), 60000.0), 2.0));
        }
    }
    return mv_out(vec4<f32>(was.xy - now.xy, min(max(was.z, 0.0), 60000.0), 1.0));
}
