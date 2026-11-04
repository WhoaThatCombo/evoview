//! The cloud deck's GPU resources and parameters, shared by the track
//! pipeline (sky, ground shadows) and the car pipeline (cloud shadows on
//! cars), so both see the same clouds in the same place.
//!
//! What is the game's: the noise volume (content/weather/cloud_noise.dds),
//! the cirrus texture, the weather map (content/weather/weather_map6), and
//! the default deck (clouds_settings/setting_1: base 1100 m, 1000 m thick,
//! scale 1, erosion 35.05 / 0.223 / 0.081, density 0.125, coverage 0.294).
//!
//! ⚠ The weather map is a 2048 x 64 strip, not a top-down picture: R is
//! cloud cover with clear lanes through it, B a rain band. How the game
//! lays it on the world is not decoded (strict SKY5): here it is stretched
//! over 200 km x 30 km, turned by mapAngle and scrolled by the wind, and it
//! modulates the coverage around EVOVIEW_CLOUDS (its mean red is 0.82).

use crate::asset::Car;

pub struct CloudGpu {
    pub noise: wgpu::TextureView,
    pub cirrus: wgpu::TextureView,
    pub weather: wgpu::TextureView,
    pub samp: wgpu::Sampler,
    pub has_noise: bool,
    pub has_cirrus: bool,
    pub has_weather: bool,
    pub ground: f32,
    /// the track pipeline draws the scenery, so everything the car shader
    /// draws outdoors is a car (shader.wgsl car_outdoor)
    pub cars_only: bool,
}

static SIM_T: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Seconds since the clouds started drifting (one clock for every user:
/// clouds, vegetation wind, the track's animated props). The frame loop's
/// clock (`advance`), not the wall clock: under EVOVIEW_FIXED_DT a slow
/// capture moved flags and trees ~10x further per frame than live, and no
/// two captures posed them alike.
pub fn time() -> f32 {
    f32::from_bits(SIM_T.load(std::sync::atomic::Ordering::Relaxed))
}

/// Step the clock by one frame's dt.
pub fn advance(dt: f32) {
    let t = time() + dt;
    SIM_T.store(t.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

fn env_f32(name: &str, d: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

impl CloudGpu {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, car: &Car) -> CloudGpu {
        use wgpu::util::DeviceExt;
        let env = car.track_env.as_deref();
        let noise_data = env.and_then(|e| e.cloud_noise.as_ref());
        let has_noise = noise_data.is_some();
        let (w, layers) = if has_noise {
            (crate::track::CLOUD_DIM, crate::track::CLOUD_SLICES)
        } else {
            (4, 1)
        };
        let blank = vec![0u8; 16 * layers as usize];
        let noise = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("cloud noise"),
                size: wgpu::Extent3d { width: w, height: w, depth_or_array_layers: layers },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Bc6hRgbUfloat,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            Default::default(),
            noise_data.map_or(&blank[..], |n| &n[..]),
        )
        .create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let up = |i: Option<usize>| i.and_then(|i| crate::material::upload(device, queue, car, i, false));
        let cirrus = up(env.and_then(|e| e.cirrus_tex));
        let weather = up(env.and_then(|e| e.weather_tex));
        let (has_cirrus, has_weather) = (cirrus.is_some(), weather.is_some());
        let samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cloud noise"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            // the cirrus layer has mips (the noise and the weather map are
            // sampled at level 0 only)
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        CloudGpu {
            noise,
            cirrus: cirrus.unwrap_or_else(|| crate::material::solid(device, queue, [0, 0, 0, 255])),
            weather: weather.unwrap_or_else(|| crate::material::solid(device, queue, [209, 0, 0, 255])),
            samp,
            has_noise,
            has_cirrus,
            has_weather,
            ground: env.map_or(0.0, |e| e.ground_y),
            cars_only: crate::track::owned_material_count(car) > 0,
        }
    }

    /// The shader block (track Frame.clouds.., render::Uniforms.clouds):
    /// 0 base above ground, thickness, coverage, density
    /// 1 noise scale, erosion detail scale, erosion width, hardness
    /// 2 time, wind x, wind z, noise present
    /// 3 cirrus amount, cirrus height, cirrus present, ground-shadow strength
    /// 4 weather map present, map length (m), map width (m), map angle (rad)
    /// 5 ground height (world y the deck is measured from), 1 = car materials
    ///   take the outdoor car path, -, -
    pub fn block(&self) -> [[f32; 4]; 6] {
        [
            [env_f32("EVOVIEW_CLOUD_BASE", 1100.0) * (1.0 - 0.45 * crate::rain::rain()),
             env_f32("EVOVIEW_CLOUD_THICK", 1000.0),
             // rain: a closed deck
             env_f32("EVOVIEW_CLOUDS", 0.294) + (0.9 - env_f32("EVOVIEW_CLOUDS", 0.294)).max(0.0) * crate::rain::rain(),
             env_f32("EVOVIEW_CLOUD_DENSITY", 0.125) * (1.0 + 1.5 * crate::rain::rain())],
            [1.0, 35.053, 0.223, 0.081],
            [time(), 6.0, 2.0, if self.has_noise { 1.0 } else { 0.0 }],
            [env_f32("EVOVIEW_CIRRUS", 0.55), 8000.0, if self.has_cirrus { 1.0 } else { 0.0 },
             env_f32("EVOVIEW_CLOUD_SHADOW", 0.85)],
            [if self.has_weather && env_f32("EVOVIEW_WEATHER_MAP", 1.0) > 0.5 { 1.0 } else { 0.0 },
             // w: the multi-scatter octaves' extinction step (EVOVIEW_CLOUD_MS)
             200_000.0, 30_000.0, env_f32("EVOVIEW_CLOUD_MS", 0.3)],
            [self.ground, if self.cars_only && std::env::var_os("EVOVIEW_OLD_CAR_OUTDOOR").is_none() { 1.0 } else { 0.0 },
             0.0, 0.0],
        ]
    }
}

/// The shared cloud functions (clouds_common.wgsl) with one shader's names
/// for the cloud block, the ground height and the sun.
pub fn common_wgsl(rows: [&str; 5], ground: &str, sun: &str) -> String {
    let mut s = include_str!("clouds_common.wgsl").to_string();
    for (i, r) in rows.iter().enumerate() {
        s = s.replace(&format!("CLD{i}"), r);
    }
    s.replace("CLD_GROUND", ground).replace("CLD_SUN", sun)
}

/// track.wgsl + the shared cloud functions.
pub fn track_source(track: &str) -> String {
    format!("{track}\n{}\n{}", crate::render::shading_wgsl(), common_wgsl(
        ["frame.clouds", "frame.clouds2", "frame.clouds3", "frame.clouds4", "frame.clouds5"],
        "frame.haze.z", "frame.sun.xyz"))
}

/// shader.wgsl + the shared cloud functions.
pub fn car_source(car: &str) -> String {
    format!("{car}\n{}\n{}", crate::render::shading_wgsl(), common_wgsl(
        ["camera.clouds[0]", "camera.clouds[1]", "camera.clouds[2]", "camera.clouds[3]", "camera.clouds[4]"],
        "camera.clouds[5].x", "key_dir()"))
}
