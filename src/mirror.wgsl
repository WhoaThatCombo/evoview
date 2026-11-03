// The car's mirror glass, showing what its mirror camera rendered
// (mirrors.rs). The texture already holds the scene's light in the scene's
// own units (exposed linear with HDR), so it is written as is.

struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var t_mirror: texture_2d<f32>;
@group(1) @binding(1) var s_mirror: sampler;
struct Params {
    // uv min (xy), 1 / uv range (zw)
    uv: vec4<f32>,
    // flip u, flip v, brightness
    flags: vec4<f32>,
};
@group(1) @binding(2) var<uniform> mp: Params;
// the car's skinning matrices (shader.wgsl): the glass rides its bone -
// a door, the rig's hull lean
struct Bones { m: array<mat4x4<f32>, 128> };
@group(2) @binding(0) var<uniform> bones: Bones;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(4) bone_idx: vec4<f32>,
    @location(5) bone_wgt: vec4<f32>,
    @location(6) inst0: vec4<f32>,
    @location(7) inst1: vec4<f32>,
    @location(8) inst2: vec4<f32>,
    @location(9) inst3: vec4<f32>,
    @location(10) facing: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var o: VsOut;
    var pos = in.position;
    let total = in.bone_wgt.x + in.bone_wgt.y + in.bone_wgt.z + in.bone_wgt.w;
    if (total > 0.0001 && in.facing.y < 0.5) {
        var p = vec3<f32>(0.0);
        for (var k = 0u; k < 4u; k = k + 1u) {
            let w = in.bone_wgt[k];
            let bi = u32(max(in.bone_idx[k], 0.0));
            if (w <= 0.0 || bi >= 128u) { continue; }
            p = p + (bones.m[bi] * vec4<f32>(in.position, 1.0)).xyz * w;
        }
        pos = p / total;
    }
    let world = mat4x4<f32>(in.inst0, in.inst1, in.inst2, in.inst3);
    o.clip = camera.view_proj * (world * vec4<f32>(pos, 1.0));
    o.uv = in.uv;
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var uv = clamp((in.uv - mp.uv.xy) * mp.uv.zw, vec2<f32>(0.0), vec2<f32>(1.0));
    if (mp.flags.x > 0.5) {
        uv.x = 1.0 - uv.x;
    }
    if (mp.flags.y > 0.5) {
        uv.y = 1.0 - uv.y;
    }
    let c = textureSample(t_mirror, s_mirror, uv).rgb * mp.flags.z;
    return vec4<f32>(c, 1.0);
}
