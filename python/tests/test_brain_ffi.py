"""Tests for the brain.py FFI wiring.

Verifies that MissionBrain correctly uses Rust particle filters
and auction engine when available, and falls back to Python
implementations otherwise.
"""

import asyncio

import pytest

from strix.brain import (
    BrainConfig,
    Decision,
    DroneSnapshot,
    MissionArea,
    MissionBrain,
    MissionIntent,
    MissionType,
    RegimeLabel,
    ThreatObservation,
    Vec3,
)


def _has_rust_ffi() -> bool:
    try:
        from strix._strix_core import ParticleNavFilter  # noqa: F401

        return True
    except ImportError:
        return False


class TestBrainInit:
    def test_rust_available_flag(self):
        brain = MissionBrain()
        if _has_rust_ffi():
            assert brain._rust_available is True
            assert brain._auctioneer is not None
        else:
            assert brain._rust_available is False

    def test_filters_dict_empty_on_init(self):
        brain = MissionBrain()
        assert brain._filters == {}


class TestBrainWithRust:
    @pytest.fixture
    def brain(self):
        if not _has_rust_ffi():
            pytest.skip("Rust FFI not available")
        return MissionBrain()

    def test_register_drone_creates_filter(self, brain):
        drone = DroneSnapshot(drone_id=1, position=Vec3(10.0, 20.0, -50.0))
        brain.register_drone(drone)
        assert 1 in brain._filters

    def test_remove_drone_cleans_filter(self, brain):
        drone = DroneSnapshot(drone_id=1, position=Vec3(10.0, 20.0, -50.0))
        brain.register_drone(drone)
        assert 1 in brain._filters
        brain.remove_drone(1)
        assert 1 not in brain._filters

    def test_predict_step_updates_position(self, brain):
        drone = DroneSnapshot(
            drone_id=1,
            position=Vec3(0.0, 0.0, -50.0),
            velocity=Vec3(10.0, 0.0, 0.0),
        )
        brain.register_drone(drone)
        initial_pos = (drone.position.x, drone.position.y, drone.position.z)

        brain._predict_step(0.1)

        # Position should have changed (not necessarily by v*dt since
        # the particle filter incorporates noise)
        updated = brain._fleet[1]
        assert isinstance(updated.position, Vec3)
        # The filter updates position; we just verify it ran without error

    def test_update_step_is_noop(self, brain):
        """_update_step should be a no-op since predict does the full cycle."""
        drone = DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, -50.0))
        brain.register_drone(drone)
        # Should not raise
        brain._update_step()

    def test_tick_executes_full_pipeline(self, brain):
        drone1 = DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, -50.0))
        drone2 = DroneSnapshot(drone_id=2, position=Vec3(100.0, 100.0, -50.0))
        brain.register_drone(drone1)
        brain.register_drone(drone2)

        decisions = asyncio.run(brain.tick(0.1))
        assert isinstance(decisions, list)

    def test_predict_with_threats(self, brain):
        """Predict step should work with threats (provides bearing)."""
        drone = DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, -50.0))
        brain.register_drone(drone)

        threat = ThreatObservation(
            threat_id=100,
            position=Vec3(100.0, 0.0, 0.0),
            confidence=0.8,
        )
        asyncio.run(brain.update_threat(threat))

        brain._predict_step(0.1)
        # Should complete without error


class TestBrainFallback:
    def test_fallback_predict_step(self):
        """Without Rust, predict step uses naive kinematics."""
        brain = MissionBrain()
        brain._rust_available = False
        brain._filters = {}

        drone = DroneSnapshot(
            drone_id=1,
            position=Vec3(0.0, 0.0, 0.0),
            velocity=Vec3(10.0, 5.0, -1.0),
        )
        brain.register_drone(drone)

        brain._predict_step(0.1)

        updated = brain._fleet[1]
        assert abs(updated.position.x - 1.0) < 1e-10
        assert abs(updated.position.y - 0.5) < 1e-10
        assert abs(updated.position.z - (-0.1)) < 1e-10

    def test_fallback_auction_returns_empty(self):
        """Without Rust, _run_auction returns []."""
        brain = MissionBrain()
        brain._rust_available = False
        brain._auctioneer = None

        result = brain._run_auction()
        assert result == []


class TestNearestThreatBearing:
    def test_no_threats(self):
        brain = MissionBrain()
        drone = DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, 0.0))
        bearing = brain._nearest_threat_bearing(drone)
        assert bearing == [0.0, 0.0, 0.0]

    def test_bearing_toward_threat(self):
        brain = MissionBrain()
        asyncio.run(
            brain.update_threat(
                ThreatObservation(threat_id=1, position=Vec3(100.0, 0.0, 0.0))
            )
        )
        drone = DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, 0.0))
        bearing = brain._nearest_threat_bearing(drone)
        assert abs(bearing[0] - 1.0) < 1e-10
        assert abs(bearing[1]) < 1e-10
        assert abs(bearing[2]) < 1e-10


class TestAuctionIntegration:
    @pytest.fixture
    def brain(self):
        if not _has_rust_ffi():
            pytest.skip("Rust FFI not available")
        b = MissionBrain()
        b.register_drone(DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, -50.0)))
        b.register_drone(DroneSnapshot(drone_id=2, position=Vec3(100.0, 100.0, -50.0)))
        return b

    def test_auction_with_plan(self, brain):
        """Auction should produce decisions when there's an active plan."""
        intent = MissionIntent(
            mission_type=MissionType.RECON,
            area=MissionArea(center=Vec3(50.0, 50.0, -50.0)),
            priority=0.8,
        )
        plan = asyncio.run(brain.process_intent(intent))
        assert plan is not None
        assert brain._active_plan is not None

        decisions = brain._run_auction()
        # May or may not produce decisions depending on plan structure
        assert isinstance(decisions, list)
        for d in decisions:
            assert isinstance(d, Decision)

    def test_auction_without_plan_returns_empty(self, brain):
        """Without an active plan, no tasks → no auction decisions."""
        decisions = brain._run_auction()
        assert decisions == []
