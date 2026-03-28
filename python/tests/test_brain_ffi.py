"""Tests for the brain.py FFI wiring.

Verifies that MissionBrain correctly uses Rust particle filters
and auction engine when available, and falls back to Python
implementations otherwise.
"""

import sys
import types

import pytest

from strix.brain import (
    BrainConfig,
    Decision,
    DecisionKind,
    DroneSnapshot,
    MissionArea,
    MissionBrain,
    MissionPlan,
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

        decisions = brain.tick_sync(0.1)
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
        brain.update_threat_sync(threat)

        brain._predict_step(0.1)
        # Should complete without error


class TestBrainFallback:
    def test_register_drone_resets_filter_on_state_update(self):
        created = []

        class FakeFilter:
            def __init__(self, **kwargs):
                self.position = kwargs["position"]
                created.append(self)

        brain = MissionBrain()
        brain._rust_available = True
        brain._ParticleNavFilter = FakeFilter
        brain._filters = {}

        brain.register_drone(DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, 0.0)))
        first_filter = brain._filters[1]

        brain.register_drone(
            DroneSnapshot(
                drone_id=1,
                position=Vec3(5.0, 0.0, 0.0),
                velocity=Vec3(1.0, 0.0, 0.0),
            )
        )

        assert len(created) == 2
        assert brain._filters[1] is not first_filter
        assert brain._filters[1].position == [5.0, 0.0, 0.0]

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
        brain.update_threat_sync(
            ThreatObservation(threat_id=1, position=Vec3(100.0, 0.0, 0.0))
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
        plan = brain.process_intent_sync(intent)
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


def test_tick_replaces_plan_assignments_with_auction_results(monkeypatch):
    class FakeAuctionDroneState:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeTask:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeThreatState:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeAssignment:
        def __init__(self, drone_id, task_id, bid_score):
            self.drone_id = drone_id
            self.task_id = task_id
            self.bid_score = bid_score

    class FakeResult:
        def __init__(self, assignments):
            self.assignments = assignments

    class FakeAuctioneer:
        def run_auction(self, _drones, _tasks, _threats):
            return FakeResult(
                [
                    FakeAssignment(drone_id=2, task_id=0, bid_score=9.0),
                    FakeAssignment(drone_id=1, task_id=1, bid_score=8.0),
                ]
            )

    monkeypatch.setitem(
        sys.modules,
        "strix._strix_core",
        types.SimpleNamespace(
            AuctionDroneState=FakeAuctionDroneState,
            Task=FakeTask,
            ThreatState=FakeThreatState,
        ),
    )

    brain = MissionBrain(BrainConfig(auction_interval_ticks=1))
    brain._rust_available = False
    brain.register_drone(DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, 0.0)))
    brain.register_drone(DroneSnapshot(drone_id=2, position=Vec3(100.0, 0.0, 0.0)))
    brain._rust_available = True
    brain._auctioneer = FakeAuctioneer()
    brain._active_plan = MissionPlan(
        assignments=[
            Decision(
                drone_id=1,
                kind=DecisionKind.GOTO,
                target_position=Vec3(10.0, 0.0, 0.0),
                speed_ms=12.0,
                reason="task alpha",
                confidence=0.7,
            ),
            Decision(
                drone_id=2,
                kind=DecisionKind.LOITER,
                target_position=Vec3(20.0, 0.0, 0.0),
                speed_ms=6.0,
                reason="task bravo",
                confidence=0.6,
            ),
        ]
    )

    decisions = brain.tick_sync(0.1)

    assert len(decisions) == 2
    assert decisions == brain._active_plan.assignments
    assert {d.drone_id for d in decisions} == {1, 2}
    assert any(d.drone_id == 2 and d.target_position == Vec3(10.0, 0.0, 0.0) for d in decisions)
    assert any(d.drone_id == 1 and d.target_position == Vec3(20.0, 0.0, 0.0) for d in decisions)


def test_async_shims_complete_synchronously():
    brain = MissionBrain()
    brain.register_drone(DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, 0.0)))

    coro = brain.tick(0.1)
    try:
        coro.send(None)
    except StopIteration as stop:
        result = stop.value
    else:
        raise AssertionError("tick coroutine unexpectedly yielded")
    finally:
        coro.close()

    assert result == brain.tick_sync(0.1)
