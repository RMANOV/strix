use strix_core::cbf::CbfConfig;

use crate::playground::Playground;
use crate::scenario::ThreatSpec;

impl Playground {
    /// **Ambush**: 20 drones in patrol, 3 threats from different angles.
    ///
    /// Tests: regime transitions, intent detection, auction reallocation.
    pub fn ambush() -> Self {
        Playground::new()
            .name("Ambush")
            .drones(20)
            .altitude(-50.0)
            .threats(vec![
                ThreatSpec::approaching(400.0, 8.0),
                ThreatSpec::flanking(500.0, 6.0, 45.0),
                ThreatSpec::flanking(500.0, 6.0, -60.0),
            ])
            .wind([2.0, -1.0, 0.0])
            .cbf(CbfConfig::default())
    }

    /// **GPS Denied**: 10 drones lose GPS at t=20s, CUSUM detects jamming.
    ///
    /// Tests: particle filter resilience, navigation drift, anomaly detection.
    pub fn gps_denied() -> Self {
        Playground::new()
            .name("GPS Denied")
            .drones(10)
            .altitude(-50.0)
            .threats(vec![ThreatSpec::stationary(300.0)])
            .jam_at_sec(20.0)
            .restore_gps_at(80.0)
            .cbf(CbfConfig::default())
    }

    /// **Attrition Cascade**: 20 drones, losses every 15s. Kill zones grow.
    ///
    /// Tests: kill zone expansion, re-auction, RiskLevel → forced EVADE.
    pub fn attrition() -> Self {
        let mut pg = Playground::new()
            .name("Attrition Cascade")
            .drones(20)
            .altitude(-50.0)
            .threats(vec![
                ThreatSpec::approaching(300.0, 5.0),
                ThreatSpec::circling(400.0, 4.0),
            ])
            .cbf(CbfConfig::default());

        // Lose a drone every 15 seconds (drone IDs from grid: 0..19)
        for i in 0..6u32 {
            pg = pg.lose_drone_at(15.0 * (i + 1) as f64, i);
        }
        pg
    }

    /// **Stress Test**: 50 drones, 10 threats, every event type at once.
    ///
    /// Tests: all subsystems under maximum load.
    pub fn stress_test() -> Self {
        let threats: Vec<ThreatSpec> = (0..10)
            .map(|i| {
                let angle = (i as f64) * 36.0;
                if i % 3 == 0 {
                    ThreatSpec::approaching(600.0, 10.0).bearing(angle)
                } else if i % 3 == 1 {
                    ThreatSpec::flanking(700.0, 7.0, angle)
                } else {
                    ThreatSpec::circling(500.0, 5.0).bearing(angle)
                }
            })
            .collect();

        Playground::new()
            .name("Stress Test")
            .drones(50)
            .spacing(40.0)
            .altitude(-80.0)
            .threats(threats)
            .wind([5.0, -3.0, 0.5])
            .cbf(CbfConfig::default())
            .nfz([200.0, 200.0, -50.0], 100.0)
            .nfz([-150.0, 300.0, -50.0], 75.0)
            .jam_at_sec(40.0)
            .restore_gps_at(70.0)
            .wind_change_at(60.0, [-4.0, 6.0, 0.0])
            .lose_drone_at(25.0, 0)
            .lose_drone_at(50.0, 5)
            .lose_drone_at(75.0, 10)
    }
}
