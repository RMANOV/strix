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
                let angular_speed = speed / radius.max(0.01);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn centroid() -> Vector3<f64> {
        Vector3::new(0.0, 0.0, 0.0)
    }

    #[test]
    fn approaching_reduces_distance() {
        let spec = ThreatSpec::approaching(500.0, 10.0);
        let c = centroid();
        let mut actor = ThreatActor::from_spec(1, &spec, &c);
        let d0 = (actor.position - c).norm();
        actor.advance(&c, 1.0);
        let d1 = (actor.position - c).norm();
        assert!(d1 < d0, "approaching must reduce distance: {d0} -> {d1}");
    }

    #[test]
    fn retreating_increases_distance() {
        let spec = ThreatSpec::retreating(500.0, 10.0);
        let c = centroid();
        let mut actor = ThreatActor::from_spec(2, &spec, &c);
        let d0 = (actor.position - c).norm();
        actor.advance(&c, 1.0);
        let d1 = (actor.position - c).norm();
        assert!(d1 > d0, "retreating must increase distance: {d0} -> {d1}");
    }

    #[test]
    fn zero_dt_no_movement() {
        let spec = ThreatSpec::approaching(500.0, 10.0);
        let c = centroid();
        let mut actor = ThreatActor::from_spec(3, &spec, &c);
        let pos_before = actor.position;
        actor.advance(&c, 0.0);
        assert_eq!(actor.position, pos_before, "dt=0 must not move");
    }

    #[test]
    fn stationary_no_velocity() {
        let spec = ThreatSpec::stationary(500.0);
        let c = centroid();
        let mut actor = ThreatActor::from_spec(4, &spec, &c);
        let pos_before = actor.position;
        actor.advance(&c, 1.0);
        assert_eq!(actor.position, pos_before, "stationary must not move");
        assert_eq!(actor.velocity, Vector3::zeros());
    }

    #[test]
    fn from_spec_places_at_correct_distance() {
        let spec = ThreatSpec::approaching(500.0, 10.0);
        let c = centroid();
        let actor = ThreatActor::from_spec(5, &spec, &c);
        // XY distance = initial_distance, altitude adds to 3D norm
        let xy_dist = ((actor.position.x - c.x).powi(2) + (actor.position.y - c.y).powi(2)).sqrt();
        assert!(
            (xy_dist - 500.0).abs() < 1.0,
            "XY distance should be ~500m, got {xy_dist}"
        );
    }

    #[test]
    fn circling_maintains_radius() {
        let spec = ThreatSpec::circling(300.0, 20.0);
        let c = centroid();
        let mut actor = ThreatActor::from_spec(6, &spec, &c);
        for _ in 0..100 {
            actor.advance(&c, 0.1);
        }
        // XY plane distance should be ~300m (altitude separate)
        let xy_dist = ((actor.position.x - c.x).powi(2) + (actor.position.y - c.y).powi(2)).sqrt();
        assert!(
            (xy_dist - 300.0).abs() < 10.0,
            "circling should maintain ~300m radius, got {xy_dist}"
        );
    }
}
