"""Tests for the brain.py FFI wiring.

Verifies that MissionBrain correctly uses Rust particle filters
and auction engine when available, and falls back to Python
implementations otherwise.
"""

import asyncio
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
    _normalize_threat_type_label,
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


class TestPublicThreatAliases:
    @pytest.mark.parametrize("label", [None, "", "   "])
    def test_normalize_threat_type_empty_inputs_return_unknown(self, label):
        assert _normalize_threat_type_label(label) == "unknown"

    def test_normalize_threat_type_alias_without_suffix(self):
        assert _normalize_threat_type_label("ew") == "electronic_warfare"

    def test_normalize_threat_type_alias_with_suffix(self):
        assert _normalize_threat_type_label("air denial:fixed") == "sam:fixed"


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

    def test_process_intent_uses_planner_routing(self):
        """Mission planning should route through tactical planning, not fixed center tasking."""
        brain = MissionBrain(BrainConfig(default_packet_success_rate=0.65, stale_state_age_s=1.0))
        brain._rust_available = False
        brain._filters = {}

        brain.register_drone(
            DroneSnapshot(
                drone_id=1,
                position=Vec3(0.0, 0.0, -50.0),
                velocity=Vec3(8.0, 0.0, 0.0),
                energy=0.9,
            )
        )
        brain.register_drone(
            DroneSnapshot(
                drone_id=2,
                position=Vec3(20.0, 10.0, -50.0),
                velocity=Vec3(6.0, 2.0, 0.0),
                energy=0.8,
            )
        )

        intent = MissionIntent(
            mission_type=MissionType.RECON,
            area=MissionArea(center=Vec3(150.0, 40.0, -50.0), radius_m=120.0),
            priority=0.8,
            drone_count=2,
        )

        plan = asyncio.run(brain.process_intent(intent))

        assert len(plan.assignments) == 2
        assert "Planner valid_fraction" in plan.explanation
        assert all("planner" in assignment.reason.lower() for assignment in plan.assignments)
        assert any(
            assignment.target_position is not None
            and (
                abs(assignment.target_position.x - intent.area.center.x) > 1.0
                or abs(assignment.target_position.y - intent.area.center.y) > 1.0
            )
            for assignment in plan.assignments
        )
        assert any(abs(assignment.confidence - 0.8) > 1e-6 for assignment in plan.assignments)

    def test_real_link_state_overrides_nominal_planner_degradedness(self):
        brain = MissionBrain(BrainConfig(default_packet_success_rate=0.9, stale_state_age_s=0.2))
        brain._rust_available = False
        brain._filters = {}

        brain.register_drone(
            DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, -50.0), velocity=Vec3(7.0, 0.0, 0.0))
        )
        brain.register_drone(
            DroneSnapshot(drone_id=2, position=Vec3(20.0, 0.0, -50.0), velocity=Vec3(7.0, 0.0, 0.0))
        )
        brain.update_network_state(packet_success_rate=0.82, state_age_s=0.4)
        brain.update_link_state(1, packet_success_rate=0.45, state_age_s=1.6, latency_ms=180.0)
        brain.update_link_state(2, packet_success_rate=0.55, state_age_s=0.8, latency_ms=120.0)

        intent = MissionIntent(
            mission_type=MissionType.RECON,
            area=MissionArea(center=Vec3(120.0, 20.0, -50.0), radius_m=80.0),
            priority=0.7,
            drone_count=2,
        )
        plan = asyncio.run(brain.process_intent(intent))

        assert "link=0.50" in plan.explanation
        assert "stale_max=1.60s" in plan.explanation
        assert brain._planner_packet_success_rate([brain._fleet[1], brain._fleet[2]]) == pytest.approx(0.5)
        assert brain._neighbor_state_ages([brain._fleet[1], brain._fleet[2]]) == {1: 1.6, 2: 0.8}


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


class TestRegimeInference:
    def test_weighted_pressure_prefers_engage_for_fast_approach(self):
        brain = MissionBrain()
        brain.register_drone(DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, -50.0)))
        asyncio.run(
            brain.update_threat(
                ThreatObservation(
                    threat_id=7,
                    position=Vec3(200.0, 0.0, -50.0),
                    velocity=Vec3(-10.0, 0.0, 0.0),
                    confidence=0.9,
                    threat_type="sam",
                )
            )
        )

        assert brain._check_regime() == RegimeLabel.ENGAGE

    def test_poor_real_comms_biases_regime_toward_evade(self):
        brain = MissionBrain()
        brain.register_drone(DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, -50.0)))
        brain.update_network_state(packet_success_rate=0.35, state_age_s=1.2)
        asyncio.run(
            brain.update_threat(
                ThreatObservation(
                    threat_id=9,
                    position=Vec3(250.0, 0.0, -50.0),
                    velocity=Vec3(-12.0, 0.0, 0.0),
                    confidence=0.95,
                    threat_type="sam",
                )
            )
        )

        assert brain._check_regime() == RegimeLabel.EVADE


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


def test_run_auction_uses_stable_task_ids(monkeypatch):
    observed_task_ids = []

    class FakeAuctionDroneState:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeTask:
        def __init__(self, **kwargs):
            observed_task_ids.append(kwargs["id"])
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
        def run_auction(self, _drones, tasks, _threats):
            return FakeResult(
                [FakeAssignment(drone_id=1, task_id=tasks[0].id, bid_score=9.0)]
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

    brain = MissionBrain()
    brain._rust_available = False
    brain.register_drone(DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, 0.0)))
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
            )
        ]
    )

    first = brain._run_auction()
    second = brain._run_auction()

    assert first[0].task_id is not None
    assert second[0].task_id == first[0].task_id
    assert observed_task_ids[0] == observed_task_ids[1] == first[0].task_id


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


def test_handle_loss_triggers_reauction():
    class FakeAuctioneer:
        def __init__(self):
            self.needs_reauction = False

        def trigger_reauction(self):
            self.needs_reauction = True

    brain = MissionBrain()
    brain._auctioneer = FakeAuctioneer()
    brain.register_drone(DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, 0.0)))
    brain._active_plan = MissionPlan(assignments=[Decision(drone_id=1, target_position=Vec3(1.0, 0.0, 0.0))])

    brain.handle_loss_sync(1, "test")

    assert brain._auctioneer.needs_reauction is True
    assert brain._active_plan.assignments == []


def test_tick_honors_auctioneer_reauction_flag(monkeypatch):
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
        def __init__(self):
            self.needs_reauction = True
            self.calls = 0

        def trigger_reauction(self):
            self.needs_reauction = True

        def run_auction(self, _drones, tasks, _threats):
            self.calls += 1
            self.needs_reauction = False
            return FakeResult([FakeAssignment(drone_id=1, task_id=tasks[0].id, bid_score=9.0)])

    monkeypatch.setitem(
        sys.modules,
        "strix._strix_core",
        types.SimpleNamespace(
            AuctionDroneState=FakeAuctionDroneState,
            Task=FakeTask,
            ThreatState=FakeThreatState,
        ),
    )

    brain = MissionBrain(BrainConfig(auction_interval_ticks=99))
    brain._rust_available = False
    brain.register_drone(DroneSnapshot(drone_id=1, position=Vec3(0.0, 0.0, 0.0)))
    brain._rust_available = True
    brain._auctioneer = FakeAuctioneer()
    brain._active_plan = MissionPlan(
        assignments=[Decision(drone_id=1, target_position=Vec3(5.0, 0.0, 0.0), reason="alpha")]
    )

    decisions = brain.tick_sync(0.1)

    assert brain._auctioneer.calls == 1
    assert brain._auctioneer.needs_reauction is False
    assert len(decisions) == 1
