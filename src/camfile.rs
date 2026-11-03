//! Explicit cameras for headless shots: `--cam <spec>`.
//!
//! The orbit flags (--yaw/--pitch/--zoom) are relative to a framing distance
//! derived from each car's bounds, so the same numbers frame two cars
//! differently. Comparing against the game's own renders needs an absolute
//! camera in car space instead. `<spec>` is one of
//!
//!     {"eye":[x,y,z],"target":[x,y,z],"fov":30}     inline JSON
//!     tools/ref_cameras.json#ks_toyota_gr86          file + entry name
//!     cam.json                                       file holding one entry
//!
//! An entry may also carry "up" (default +Y) and "near" (default 0.05 m).
//! `fov` is VERTICAL, in degrees. A file entry may be an object with a
//! "camera" field (how tools/ref_cameras.json stores them) or the camera
//! itself; when the file has a "default" entry, unknown names fall back to it.

use glam::{Mat4, Vec3};
use serde_json::Value;

fn vec3(v: &Value) -> Option<Vec3> {
    let a = v.as_array()?;
    Some(Vec3::new(a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32,
                   a.get(2)?.as_f64()? as f32))
}

fn entry(spec: &str) -> Result<Value, String> {
    if spec.trim_start().starts_with('{') {
        return serde_json::from_str(spec).map_err(|e| format!("--cam JSON: {e}"));
    }
    let (path, name) = match spec.rsplit_once('#') {
        Some((p, n)) => (p, Some(n)),
        None => (spec, None),
    };
    let text = std::fs::read_to_string(path).map_err(|e| format!("--cam {path}: {e}"))?;
    let root: Value = serde_json::from_str(&text).map_err(|e| format!("--cam {path}: {e}"))?;
    let mut v = match name {
        Some(n) => {
            let cams = root.get("cars").or_else(|| root.get("cameras")).unwrap_or(&root);
            cams.get(n).or_else(|| cams.get("default")).cloned()
                .ok_or_else(|| format!("--cam {path}: no entry {n:?}"))?
        }
        None => root,
    };
    if let Some(c) = v.get("camera") {
        v = c.clone();
    }
    Ok(v)
}

/// (view_proj, eye) for `spec` at the given aspect ratio.
pub fn view(spec: &str, aspect: f32) -> Result<(Mat4, Vec3), String> {
    let v = entry(spec)?;
    let eye = v.get("eye").and_then(vec3).ok_or("--cam: missing \"eye\"")?;
    let target = v.get("target").and_then(vec3).ok_or("--cam: missing \"target\"")?;
    let up = v.get("up").and_then(vec3).unwrap_or(Vec3::Y);
    let fov = v.get("fov").and_then(Value::as_f64).unwrap_or(45.0) as f32;
    let near = v.get("near").and_then(Value::as_f64).unwrap_or(0.05) as f32;
    let proj = Mat4::perspective_infinite_rh(fov.to_radians(), aspect.max(0.01), near);
    println!("camera: eye ({:.3}, {:.3}, {:.3}) -> ({:.3}, {:.3}, {:.3}), fov {fov:.2}",
             eye.x, eye.y, eye.z, target.x, target.y, target.z);
    Ok((proj * Mat4::look_at_rh(eye, target, up), eye))
}
