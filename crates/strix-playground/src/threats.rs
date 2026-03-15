use nalgebra::Vector3;

use crate::scenario::{ThreatBehavior, ThreatSpec};

// ---------------------------------------------------------------------------
// Active threat actor in the simulation
// ---------------------------------------------------------------------------

/// A living threat entity with position, velocity, and movement model.
#[derive(Debug, Clone)]
pub struct ThreatActor {
    pub id: u32,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub behavior: ThreatBehavior,
    pub spec: ThreatSpec,
    /// Accumulated orbit angle for Circling behavior (radians).
    orbit_angle: f64,
}

impl ThreatActor {
    /// Spawn a threat actor from a spec, placing it relative to a centroid.
    pub fn from_spec(id: u32, spec: &ThreatSpec, centroid: &Vector3<f64>) -> Self {
        let bearing_rad = spec.initial_bearing_deg.to_radians();
        // NED: X=North, Y=East
        let offset = Vector3::new(
            spec.initial_distance * bearing_rad.cos(),
            spec.initial_distance * bearing_rad.sin(),
            spec.altitude,
        );
        let position = centroid + offset;

        Self {
            id,
            position,
            velocity: Vector3::zeros(),
            behavior: spec.behavior.clone(),
            spec: spec.clone(),
            orbit_angle: bearing_rad,
        }
    }

    /// Advance position by `dt` seconds based on behavior and fleet centroid.
    pub fn advance(&mut self, fleet_centroid: &Vector3<f64>, dt: f64) {
        match &self.behavior {
            ThreatBehavior::Approaching { speed } => {
                let diff = fleet_centroid - self.position;
                let dist = diff.norm();
                if dist > 1.0 {
                    let dir = diff / dist;
                    self.velocity = dir * *speed;
                    self.position += self.velocity * dt;
                }
            }

            ThreatBehavior::Flanking { speed, angle_deg } => {
                // Arc approach: direction rotates by angle_deg offset from
                // the direct bearing, creating an oblique attack vector.
                let diff = fleet_centroid - self.position;
                let dist = diff.norm();
                if dist > 1.0 {
                    let direct = diff / dist;
                    let offset_rad = angle_deg.to_radians();
                    // Rotate the 2D bearing (NED X-Y plane) by the offset angle.
                    let rotated_x = direct.x * offset_rad.cos() - direct.y * offset_rad.sin();
                    let rotated_y = direct.x * offset_rad.sin() + direct.y * offset_rad.cos();
                    // Blend: as threat gets closer, converge toward direct path.
                    let blend = (dist / self.spec.initial_distance).clamp(0.0, 1.0);
                    let dir_x = rotated_x * blend + direct.x * (1.0 - blend);
                    let dir_y = rotated_y * blend + direct.y * (1.0 - blend);
                    let dir = Vector3::new(dir_x, dir_y, 0.0).normalize();
                    self.velocity = dir * *speed;
                    self.position += self.velocity * dt;
                }
            }

            ThreatBehavior::Circling { speed, radius } => {
                // Orbit centroid at fixed radius.
                let angular_speed = speed / radius;
                self.orbit_angle += angular_speed * dt;
                let target = Vector3::new(
                    fleet_centroid.x + radius * self.orbit_angle.cos(),
                    fleet_centroid.y + radius * self.orbit_angle.sin(),
                    self.spec.altitude,
                );
                self.velocity = (target - self.position) / dt.max(0.001);
                self.position = target;
            }

            ThreatBehavior::Retreating { speed } => {
                let diff = self.position - fleet_centroid;
                let dist = diff.norm();
                let dir = if dist > 0.1 {
                    diff / dist
                } else {
                    Vector3::new(1.0, 0.0, 0.0)
                };
                self.velocity = dir * *speed;
                self.position += self.velocity * dt;
            }

            ThreatBehavior::Stationary => {
                self.velocity = Vector3::zeros();
            }
        }
    }
}
