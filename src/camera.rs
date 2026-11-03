//! Orbit camera.
//!
//! Framing is derived from the model's own bounds rather than hardcoded, so
//! any car - or any single mesh - fills the view sensibly on load.

use glam::{Mat4, Vec3};

pub struct Orbit {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
}

impl Orbit {
    /// Frame a bounding box: centre it, and back off far enough to fit.
    pub fn framing(min: Vec3, max: Vec3) -> Self {
        let centre = (min + max) * 0.5;
        let radius = (max - min).length() * 0.5;
        let fov_y = 45f32.to_radians();
        Self {
            target: centre,
            // fit the sphere in the vertical field of view, with headroom
            distance: (radius / (fov_y * 0.5).sin()) * 1.15,
            yaw: 2.5,
            pitch: 0.35,
            fov_y,
        }
    }

    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(cp * cy, sp, cp * sy) * self.distance
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        // Near/far scale with the subject so precision stays sane whether we
        // are framing a whole car or one wheel nut.
        // ⚠ Infinite far plane. A fixed multiple of the orbit distance
        // clipped anything further out, which on a 22 km track meant most of
        // the world simply was not drawn.
        let near = (self.distance * 0.005).max(0.01);
        let proj = Mat4::perspective_infinite_rh(self.fov_y, aspect.max(0.01), near);
        proj * Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.006;
        self.pitch = (self.pitch + dy * 0.006).clamp(-1.5, 1.5);
    }

    pub fn zoom(&mut self, amount: f32) {
        // Multiplicative, so zooming feels the same at every distance.
        // ⚠ No 1000 m ceiling. That was fine for a car and wrong for a
        // track: Shutoko frames from ~13 km out, so the first scroll snapped
        // the camera to 1 km and left the city outside the view - which reads
        // as an empty void rather than as a zoom limit.
        self.distance = (self.distance * (1.0 - amount * 0.1)).clamp(0.05, 1.0e7);
    }
}

/// Cameras that follow a car during replay playback.
///
/// These take a pose rather than owning state, because the car they follow is
/// re-sampled from the replay every frame anyway - keeping a position here as
/// well would just be a second copy to keep in sync.
#[derive(Clone, Copy, PartialEq)]
pub enum Follow {
    /// The fly camera; playback does not drive it.
    Free,
    /// Behind and above, looking ahead of the car.
    Chase,
    /// Driver's eye, looking where the car points.
    Onboard,
    /// Alongside, level with the car - good for watching suspension work.
    Side,
    /// Circling the car slowly, for looking at it from every angle.
    Orbit,
}

impl Follow {
    pub fn next(self) -> Follow {
        match self {
            Follow::Free => Follow::Chase,
            Follow::Chase => Follow::Onboard,
            Follow::Onboard => Follow::Side,
            Follow::Side => Follow::Orbit,
            Follow::Orbit => Follow::Free,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Follow::Free => "free",
            Follow::Chase => "chase",
            Follow::Onboard => "onboard",
            Follow::Side => "side",
            Follow::Orbit => "orbit",
        }
    }

    /// (view_proj, eye). `spin` is a seconds counter, used only by Orbit.
    pub fn view_proj(
        self, aspect: f32, pos: Vec3, fwd: Vec3, up: Vec3, right: Vec3, spin: f32,
    ) -> (Mat4, Vec3) {
        let fov_y = match self {
            // A slightly wider lens onboard, or the cockpit feels like looking
            // down a tube.
            Follow::Onboard => 65f32.to_radians(),
            _ => 45f32.to_radians(),
        };
        let (eye, target) = match self {
            Follow::Free | Follow::Chase => (
                pos - fwd * 8.0 + Vec3::Y * 2.8,
                pos + fwd * 6.0 + Vec3::Y * 0.6,
            ),
            // Driver's head. `pos` is the centre of mass, which sits about
            // 0.37 m above the ground, so 0.62 m above it puts the eye near
            // 1.0 m - real driver height. Going by "about a metre up" from
            // `pos` instead lands above the roof and reads as a bonnet cam.
            Follow::Onboard => {
                let e = pos + up * 0.78 + fwd * 0.10 - right * 0.30;
                // Aimed slightly down: level with the car's own forward axis
                // puts the horizon high and fills the lower half with dash.
                (e, e + fwd - up * 0.09)
            }
            // EVOVIEW_SIDE_EYE=r,u,f,tr,tu,tf: eye and target in car
            // right/up/forward metres from `pos` (wheel close-ups in captures)
            Follow::Side => match std::env::var("EVOVIEW_SIDE_EYE").ok()
                .map(|v| v.split(',').filter_map(|x| x.trim().parse::<f32>().ok()).collect::<Vec<_>>())
                .filter(|v| v.len() == 6)
            {
                Some(v) => (pos + right * v[0] + up * v[1] + fwd * v[2],
                            pos + right * v[3] + up * v[4] + fwd * v[5]),
                None => (pos + right * 9.0 + Vec3::Y * 1.4, pos),
            },
            Follow::Orbit => {
                let (s, c) = (spin * 0.35).sin_cos();
                (pos + Vec3::new(c * 11.0, 3.5, s * 11.0), pos)
            }
        };
        // Near plane close enough for the cockpit, far plane infinite so the
        // rest of the circuit still draws (see Fly::view_proj).
        let proj = Mat4::perspective_infinite_rh(fov_y, aspect.max(0.01), 0.12);
        (proj * Mat4::look_at_rh(eye, target, Vec3::Y), eye)
    }
}

/// Free-look camera: a position and a direction, moved with the keyboard.
///
/// The orbit camera is right for inspecting a car as an object; this one is for
/// getting inside it - under the floor, into the cockpit, along the diffuser -
/// where "orbit a centre point" stops being a useful way to move.
pub struct Fly {
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    /// metres per second, scaled to the car so it feels the same on any model
    pub speed: f32,
}

impl Fly {
    /// Start exactly where the orbit camera is looking from, so toggling does
    /// not teleport the view.
    pub fn from_orbit(o: &Orbit) -> Self {
        let dir = (o.target - o.eye()).normalize_or_zero();
        Self {
            pos: o.eye(),
            yaw: dir.z.atan2(dir.x),
            pitch: dir.y.clamp(-1.0, 1.0).asin(),
            fov_y: o.fov_y,
            // ⚠ Scale the speed to the subject. A flat 2.5 m/s is right for
            // walking around a car and useless on a city-sized track, where
            // it would take twenty minutes to reach anything - so free-look
            // felt like being stuck in empty space.
            speed: (o.distance * 0.06).clamp(2.5, 400.0),
        }
    }

    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * cy, sp, cp * sy)
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize_or_zero()
    }

    pub fn look(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.004;
        // Stop just short of vertical: looking exactly along Y makes the
        // up-vector ambiguous and the view rolls.
        self.pitch = (self.pitch - dy * 0.004).clamp(-1.55, 1.55);
    }

    pub fn view_proj(&self, aspect: f32) -> glam::Mat4 {
        // ⚠ This was hardcoded to 5000 m - on a 22 km highway network that
        // clipped everything past the first few kilometres, so standing on
        // the road still showed an empty horizon. Infinite instead.
        //
        // ⚠ Near is held at 0.25 m rather than scaling all the way down with
        // speed: with an infinite far plane, a tiny near plane spends the
        // whole depth buffer on the first few metres and distant geometry
        // starts z-fighting. 0.25 m is still close enough to inspect a kerb.
        let near = (self.speed * 0.004).clamp(0.25, 2.0);
        let proj = Mat4::perspective_infinite_rh(self.fov_y, aspect.max(0.01), near);
        proj * Mat4::look_at_rh(self.pos, self.pos + self.forward(), Vec3::Y)
    }
}
