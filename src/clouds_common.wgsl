// Cloud deck functions shared by track.wgsl and shader.wgsl (clouds.rs
// appends this to both). The tokens are replaced per shader:
//   CLD0..CLD4  the clouds.rs block rows (frame.clouds.. / camera.clouds[i])
//   CLD_GROUND  world y the deck heights are measured from
//   CLD_SUN     unit vector toward the sun
// and both declare t_cloud (64-slice noise), s_cloud and t_weather.

fn cloud_tex(p: vec3<f32>) -> vec3<f32> {
    // p.xz in tiles, p.y 0..1 through the 64 slices (trilinear by hand:
    // BC6H cannot be a 3D texture here)
    let z = fract(p.y) * 64.0 - 0.5;
    let z0 = floor(z);
    let f = z - z0;
    let l0 = (i32(z0) + 64) % 64;
    let l1 = (l0 + 1) % 64;
    let a = textureSampleLevel(t_cloud, s_cloud, p.xz, l0, 0.0).rgb;
    let b = textureSampleLevel(t_cloud, s_cloud, p.xz, l1, 0.0).rgb;
    return mix(a, b, f);
}

fn remap(v: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return c + (v - a) / max(b - a, 1e-4) * (d - c);
}

/// (coverage, rain) at world point `w` from the weather map (clouds.rs):
/// its R modulates the deck's coverage around the configured value (the
/// map's mean red is 0.82), its B is a rain band that thickens the cloud.
fn cloud_weather(w: vec3<f32>) -> vec2<f32> {
    let cov = CLD0.z;
    if (CLD4.x < 0.5) {
        return vec2<f32>(cov, 0.0);
    }
    let ca = cos(CLD4.w);
    let sa = sin(CLD4.w);
    let wind = CLD2.yz * CLD2.x;
    let x = (w.x + wind.x) * ca - (w.z + wind.y) * sa;
    let z = (w.x + wind.x) * sa + (w.z + wind.y) * ca;
    let s = textureSampleLevel(t_weather, s_cloud, vec2<f32>(x / CLD4.y, z / CLD4.z), 0.0);
    return vec2<f32>(clamp(cov * s.r / 0.82 + 0.35 * s.b, 0.0, 1.0), s.b);
}

/// Cloud density at world point `w`, height fraction `h` through the deck.
fn cloud_density(w: vec3<f32>, h: f32, cheap: bool) -> f32 {
    let wind = CLD2.yz * CLD2.x;
    let tile = 8000.0 / max(CLD1.x, 0.01);
    let q = vec3<f32>((w.x + wind.x) / tile, h, (w.z + wind.y) / tile);
    let n = cloud_tex(q);
    // cumulus-ish height profile: round bottoms, thinning tops
    let grad = smoothstep(0.0, 0.12, h) * smoothstep(1.0, 0.45, h);
    let wx = cloud_weather(w);
    var d = remap(n.r * grad, 1.0 - wx.x, 1.0, 0.0, 1.0);
    if (d <= 0.0) {
        return 0.0;
    }
    if (!cheap) {
        // erode the edges with the finer channels
        let dq = vec3<f32>(q.x * CLD1.y * 0.125, fract(h * 3.0), q.z * CLD1.y * 0.125);
        let e = cloud_tex(dq);
        let ero = mix(e.g, e.b, h);
        d = remap(d, ero * CLD1.z, 1.0, 0.0, 1.0);
    }
    // rain cloud is denser
    return clamp(d, 0.0, 1.0) * (1.0 + 2.0 * wx.y);
}

/// How much of the sun the cumulus deck lets through to world point `p`:
/// a short march along the sun direction through the deck, coarse shape
/// only (no erosion). Cloud shadows are large and soft, so four samples do.
fn cloud_shadow(p: vec3<f32>) -> f32 {
    let sun = CLD_SUN;
    if (CLD2.w < 0.5 || CLD0.z <= 0.0 || CLD3.w <= 0.0 || sun.y < 0.03) {
        return 1.0;
    }
    let bottom = CLD_GROUND + CLD0.x;
    let top = bottom + CLD0.y;
    let t0 = max((bottom - p.y) / sun.y, 0.0);
    let t1 = max((top - p.y) / sun.y, t0);
    let dt = (t1 - t0) * 0.25;
    var od = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        let q = p + sun * (t0 + (f32(i) + 0.5) * dt);
        let h = (q.y - bottom) / max(CLD0.y, 1.0);
        od += cloud_density(q, h, true) * dt;
    }
    let lit = exp(-od * CLD0.w * 0.16);
    return mix(1.0, lit, CLD3.w);
}
