// Shared by shader.wgsl and track.wgsl (appended with the BT_GAME / NRM_COF /
// GSAA constants, render.rs shading_wgsl()).

/// A normal into world space. Under non-uniform scale (75 % of Imola's
/// instances) the model matrix tilts normals toward the stretched axis; the
/// cofactor matrix (columns' cross products = det * inverse transpose) does
/// not. For a mirrored instance it points the normal inward, which is what
/// the reversed winding expects: the back-face flip in the pixel shader
/// turns it out again. (The game's MaterialVS 034a96e0 uses the plain
/// matrix; EVOVIEW_NORMAL_COFACTOR=0 restores that.)
fn normal_to_world(m: mat3x3<f32>, n: vec3<f32>) -> vec3<f32> {
    if (NRM_COF) {
        return mat3x3<f32>(cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1])) * n;
    }
    return m * n;
}

/// What the material passes write to the fp16 HDR target: exposed light,
/// capped well inside fp16's 65504 (a sun glint on wet tarmac or paint could
/// reach Inf, which TAA turned into black pixels and bloom into white
/// blocks), NaN to black so it cannot spread through TAA history either.
fn hdr_out(e: vec3<f32>) -> vec3<f32> {
    return select(clamp(e, vec3<f32>(0.0), vec3<f32>(32000.0)), vec3<f32>(0.0), e != e);
}

/// The game's GSAA (UberBiplanar 034e5ce0, the road shaders, UberVehicle
/// 035a3710 SPEC_uber 7): below 0.72 the roughness is widened by `v`, the
/// normal's screen-space variance (|ddx n|^2 + |ddy n|^2, times the
/// variance factor), capped at `thr`^2. r' >= r always; at most 0.84.
fn gsaa_rough(r: f32, v: f32, thr: f32) -> f32 {
    if (!GSAA || r >= 0.72) {
        return r;
    }
    let a2 = r * r * r * r;
    let s = clamp(min(2.0 * v * v, thr * thr) + a2 / (1.0 - a2), 0.0, 1.0);
    return sqrt(sqrt(s / (s + 1.0)));
}

/// The game's specular fade from the GEOMETRIC normal's variance, kS =
/// sat(1 - 0.3 (|ddx Ng|^2 + |ddy Ng|^2)): direct specular x kS, image-based
/// x sat(kS + 0.75). Curved geometry far away (a car's flanks, kerb edges)
/// stops throwing single-pixel sun glints.
fn spec_aa(ng_var: f32) -> f32 {
    return select(1.0, clamp(1.0 - 0.3 * ng_var, 0.0, 1.0), GSAA);
}

fn sq_deriv(n: vec3<f32>) -> f32 {
    let dx = dpdxCoarse(n);
    let dy = dpdyCoarse(n);
    return dot(dx, dx) + dot(dy, dy);
}

/// The bitangent sign under that transform: a mirror reverses
/// cross(T, N), so w flips with it.
fn tangent_w_to_world(m: mat3x3<f32>, w: f32) -> f32 {
    if (NRM_COF && dot(m[0], cross(m[1], m[2])) < 0.0) {
        return -w;
    }
    return w;
}
