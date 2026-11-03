//! Loading the .evocar produced by tools/export_car.py.
//!
//! The file is deliberately dumb: a JSON manifest describing offsets into one
//! blob of already-GPU-shaped arrays. Loading is a read plus slices - no
//! parsing, no per-vertex work on the CPU.

use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize, Debug, Clone)]
pub struct Batch {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub material: String,
    #[serde(default)]
    pub start: u32,
    #[serde(default)]
    pub count: u32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct MeshEntry {
    pub name: String,
    pub vertices: u32,
    #[serde(default)]
    pub batches: Vec<Batch>,
    // [offset, length] into the blob
    pub positions: Option<[u64; 2]>,
    pub normals: Option<[u64; 2]>,
    pub uvs: Option<[u64; 2]>,
    pub tangents: Option<[u64; 2]>,
    /// 4 bone indices per vertex (the engine stores them as f32)
    #[serde(default)]
    pub bone_idx: Option<[u64; 2]>,
    /// 4 bone weights per vertex
    #[serde(default)]
    pub bone_wgt: Option<[u64; 2]>,
    pub indices: Option<[u64; 2]>,
    /// World matrices for instanced props/vegetation. Empty = draw once at origin.
    #[serde(default)]
    pub xforms: Vec<[f32; 16]>,
    /// MeshLodData.cameraFacingMode (0 none, 1 Y, 2 XAndY, 3/4 KeepParent).
    #[serde(default)]
    pub facing: u32,
    /// [near, far) camera distance in metres this mesh draws at, when it is
    /// one LOD of an instanced prop (see cull.rs). None = always. Paths with
    /// no culler draw only the near=0 level.
    #[serde(default)]
    pub lod: Option<[f32; 2]>,
    /// The crossfade around that band (load.rs, cull.rs): [metres the
    /// switch-in at `near` blends over, metres the switch-out at `far` blends
    /// over, the whole prop's fade-out start (0 = never), its width].
    #[serde(default)]
    pub lod_fade: [f32; 4],
    /// What casts this LOD's shadow (MeshData forceShadowLod + MeshLodData
    /// shadowLod/castShadows): None = itself, Some(None) = nothing,
    /// Some(Some(i)) = the geometry of mesh i (an index into Car::meshes).
    #[serde(default)]
    pub shadow_mesh: Option<Option<usize>>,
    /// Track meshes only: raw vertex colour, 4 floats/vertex as the engine
    /// stores it (colour / 255). Carries splat weights, AO and variation.
    #[serde(default)]
    pub colors: Option<[u64; 2]>,
    /// Track meshes only: second UV set, 2 floats/vertex.
    #[serde(default)]
    pub uv1: Option<[u64; 2]>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Level {
    pub width: u32,
    pub height: u32,
    pub bytes: u64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct TextureEntry {
    pub name: String,
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<Level>,
    pub data: [u64; 2],
}

#[derive(Deserialize, Debug, Clone)]
pub struct MaterialEntry {
    pub file: String,
    /// Full archive path, lowercased with `/` separators.
    ///
    /// ⚠ The basename is NOT unique. A track ships several `bark_mat.material`
    /// in different vegetation folders, and keying the lookup on the basename
    /// let them overwrite each other - last one loaded won, so a mesh could get
    /// another folder's material. Fuji ships 3 and Spa 4; Nurburgring has 26
    /// names that collide with a materially different entry.
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub shader: Option<String>,
    /// slot name -> index into `textures`
    #[serde(default)]
    pub slots: std::collections::HashMap<String, usize>,
    #[serde(default)]
    pub params: std::collections::HashMap<String, f32>,
    /// vector params (rgb / rgba), e.g. Base_Basecolor
    #[serde(default)]
    pub colors: std::collections::HashMap<String, Vec<f32>>,
}

impl MaterialEntry {
    /// First slot that exists, trying each name in order. The shader exposes
    /// many aliases for the same idea (Base_BaseColorMap, txDiffuse, ...), and
    /// which one a material uses varies.
    pub fn slot(&self, names: &[&str]) -> Option<usize> {
        names.iter().find_map(|n| self.slots.get(*n).copied())
    }
    /// First colour param that exists, as rgba.
    pub fn color(&self, names: &[&str], default: [f32; 4]) -> [f32; 4] {
        for n in names {
            if let Some(v) = self.colors.get(*n) {
                if v.len() >= 3 {
                    return [v[0], v[1], v[2], *v.get(3).unwrap_or(&1.0)];
                }
            }
        }
        default
    }

    /// Is any of these params / colours authored at all?
    pub fn has(&self, names: &[&str]) -> bool {
        names.iter().any(|n| self.params.contains_key(*n) || self.colors.get(*n).map_or(false, |v| v.len() >= 3))
    }

    pub fn param(&self, names: &[&str], default: f32) -> f32 {
        names
            .iter()
            .find_map(|n| self.params.get(*n).copied())
            .unwrap_or(default)
    }
}

#[derive(Deserialize, Debug)]
struct Manifest {
    meshes: Vec<MeshEntry>,
    textures: Vec<TextureEntry>,
    materials: Vec<MaterialEntry>,
}

pub struct Car {
    pub meshes: Vec<MeshEntry>,
    pub textures: Vec<TextureEntry>,
    pub materials: Vec<MaterialEntry>,
    blob: Vec<u8>,
    /// TerrainRenderSettings.world_uv_scale. 0 = not authored; the viewer
    /// keeps its 0.085 / metre grass-tile fallback.
    pub world_uv_scale: f32,
    /// The game's own environment cubemap, when the package was available to
    /// read it from. None falls back to the analytic studio in the shader.
    pub env: Option<crate::load::Cube>,
    /// The game's split-sum BRDF table: (edge, RG16F bytes).
    pub brdf: Option<(u32, Vec<u8>)>,
    /// Track-wide environment (irradiance volumes, location), tracks only.
    pub track_env: Option<Box<crate::track::TrackEnv>>,
    /// Where each texture came from (vfs key), parallel to `textures`; empty
    /// when unknown. Basenames are NOT unique (every SF-25 livery folder has
    /// its own ext_skin_d.texture), so this is what identifies a texture
    /// across reloads (material::TexCache, livery switching).
    pub tex_src: Vec<String>,
    /// Effect data (lights, LEDs, displays, particles), fxdata.rs. None for
    /// tracks and bare .evocar exports.
    pub fx: Option<Box<crate::fxdata::CarFx>>,
    /// common_assets/rain rain_drop_mask / _alpha / _nm (rain.rs)
    pub rain_tex: [Option<usize>; 3],
    /// the side windows' streak maps, packed (rain::pack_streaks)
    pub rain_streaks: Option<std::sync::Arc<Vec<u8>>>,
}

impl Car {
    /// Build from parts already in memory - what the direct folder loader
    /// produces, so nothing has to round-trip through a file.
    pub fn from_parts(
        meshes: Vec<MeshEntry>,
        textures: Vec<TextureEntry>,
        materials: Vec<MaterialEntry>,
        blob: Vec<u8>,
    ) -> Self {
        Car {
            meshes,
            textures,
            materials,
            blob,
            world_uv_scale: 0.0,
            env: None,
            brdf: None,
            track_env: None,
            tex_src: Vec::new(),
            fx: None,
            rain_tex: [None; 3],
            rain_streaks: None,
        }
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if bytes.len() < 24 || &bytes[0..8] != b"EVOCAR01" {
            return Err("not an .evocar file (bad magic)".into());
        }
        let json_len = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
        let json_end = 24 + json_len;
        if json_end > bytes.len() {
            return Err("truncated manifest".into());
        }
        let m: Manifest = serde_json::from_slice(&bytes[24..json_end])
            .map_err(|e| format!("bad manifest: {e}"))?;
        Ok(Car {
            meshes: m.meshes,
            textures: m.textures,
            materials: m.materials,
            blob: bytes[json_end..].to_vec(),
            world_uv_scale: 0.0,
            env: None,
            brdf: None,
            track_env: None,
            tex_src: Vec::new(),
            fx: None,
            rain_tex: [None; 3],
            rain_streaks: None,
        })
    }

    /// The whole data blob. Needed when several loaded cars are merged into
    /// one scene, which has to concatenate blobs and shift every range.
    pub fn blob(&self) -> &[u8] {
        &self.blob
    }

    /// A slice of the data blob, or an empty slice if the range is bogus.
    pub fn slice(&self, r: Option<[u64; 2]>) -> &[u8] {
        match r {
            Some([off, len]) => {
                let (a, b) = (off as usize, (off + len) as usize);
                if b <= self.blob.len() {
                    &self.blob[a..b]
                } else {
                    &[]
                }
            }
            None => &[],
        }
    }

    /// A blob range as f32s.
    ///
    /// ⚠ Do NOT bytemuck::cast_slice the raw bytes: the blob is a Vec<u8>, so
    /// its base pointer carries no alignment guarantee and the cast panics
    /// with TargetAlignmentGreaterAndInputNotAligned on some allocations. The
    /// copy happens once per mesh at load and costs nothing measurable.
    pub fn floats(&self, r: Option<[u64; 2]>) -> Vec<f32> {
        let b = self.slice(r);
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// A blob range as u32s (mesh indices).
    pub fn u32s(&self, r: Option<[u64; 2]>) -> Vec<u32> {
        let b = self.slice(r);
        b.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    pub fn tex_data(&self, t: &TextureEntry) -> &[u8] {
        self.slice(Some(t.data))
    }
}
