"""Tests for the auction FFI bridge (strix._strix_core auction types).

Verifies that the PyO3 bridge correctly exposes:
- Position (construction, getters, distance_to)
- Capabilities (defaults, custom flags)
- AuctionDroneState (construction, getters)
- Task (construction, getters)
- ThreatType (class attributes)
- ThreatState (construction, getters)
- Auctioneer (run_auction, trigger_reauction, needs_reauction)
- AuctionResult (assignments, unassigned_tasks, total_welfare)
- Assignment (drone_id, task_id, bid_score)
- LossAnalyzer (antifragile_score, active_kill_zones)
"""

import pytest


def _import_strix_core():
    """Try to import the native module, skip if not built."""
    try:
        from strix import _strix_core
        return _strix_core
    except ImportError:
        pytest.skip(
            "strix._strix_core not built (run: maturin develop --features python)"
        )


class TestPosition:
    def test_construction(self):
        sc = _import_strix_core()
        p = sc.Position(1.0, 2.0, 3.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0

    def test_distance_to(self):
        sc = _import_strix_core()
        p1 = sc.Position(0.0, 0.0, 0.0)
        p2 = sc.Position(3.0, 4.0, 0.0)
        assert abs(p1.distance_to(p2) - 5.0) < 1e-10

    def test_repr(self):
        sc = _import_strix_core()
        p = sc.Position(1.0, 2.0, 3.0)
        assert "Position" in repr(p)


class TestCapabilities:
    def test_defaults(self):
        sc = _import_strix_core()
        c = sc.Capabilities()
        assert c.has_sensor is True
        assert c.has_weapon is False
        assert c.has_ew is False
        assert c.has_relay is False

    def test_custom(self):
        sc = _import_strix_core()
        c = sc.Capabilities(has_sensor=True, has_weapon=True, has_ew=True)
        assert c.has_weapon is True
        assert c.has_ew is True

    def test_repr(self):
        sc = _import_strix_core()
        c = sc.Capabilities()
        assert "Capabilities" in repr(c)


class TestAuctionDroneState:
    def test_construction_minimal(self):
        sc = _import_strix_core()
        d = sc.AuctionDroneState(id=1, position=[0.0, 0.0, 100.0])
        assert d.id == 1
        assert d.position == [0.0, 0.0, 100.0]
        assert d.velocity == [0.0, 0.0, 0.0]
        assert d.energy == pytest.approx(1.0)
        assert d.alive is True

    def test_construction_full(self):
        sc = _import_strix_core()
        caps = sc.Capabilities(has_sensor=True, has_weapon=True)
        d = sc.AuctionDroneState(
            id=7,
            position=[10.0, 20.0, 30.0],
            velocity=[1.0, 2.0, 3.0],
            regime_index=0,
            capabilities=caps,
            energy=0.75,
            alive=True,
        )
        assert d.id == 7
        assert d.position == [10.0, 20.0, 30.0]
        assert d.velocity == [1.0, 2.0, 3.0]
        assert d.energy == pytest.approx(0.75)

    def test_invalid_regime_index(self):
        sc = _import_strix_core()
        with pytest.raises(Exception):
            sc.AuctionDroneState(id=1, position=[0.0, 0.0, 0.0], regime_index=99)

    def test_repr(self):
        sc = _import_strix_core()
        d = sc.AuctionDroneState(id=3, position=[0.0, 0.0, 0.0])
        assert "AuctionDroneState" in repr(d)


class TestTask:
    def test_construction_minimal(self):
        sc = _import_strix_core()
        t = sc.Task(id=10, location=[5.0, 5.0, 50.0])
        assert t.id == 10
        assert t.location == [5.0, 5.0, 50.0]
        assert t.priority == pytest.approx(0.5)
        assert t.urgency == pytest.approx(0.5)

    def test_construction_full(self):
        sc = _import_strix_core()
        caps = sc.Capabilities(has_sensor=True)
        t = sc.Task(
            id=42,
            location=[1.0, 2.0, 3.0],
            required_capabilities=caps,
            priority=0.9,
            urgency=0.8,
            bundle_id=5,
            dark_pool=2,
        )
        assert t.id == 42
        assert t.location == [1.0, 2.0, 3.0]
        assert t.priority == pytest.approx(0.9)
        assert t.urgency == pytest.approx(0.8)

    def test_repr(self):
        sc = _import_strix_core()
        t = sc.Task(id=1, location=[0.0, 0.0, 0.0])
        assert "Task" in repr(t)


class TestThreatType:
    def test_variants(self):
        sc = _import_strix_core()
        assert sc.ThreatType.Sam is not None
        assert sc.ThreatType.SmallArms is not None
        assert sc.ThreatType.ElectronicWarfare is not None
        assert sc.ThreatType.Unknown is not None

    def test_repr(self):
        sc = _import_strix_core()
        assert "ThreatType" in repr(sc.ThreatType.Sam)


class TestThreatState:
    def test_construction(self):
        sc = _import_strix_core()
        th = sc.ThreatState(id=100, position=[25.0, 25.0, 0.0])
        assert th.id == 100
        assert th.position == [25.0, 25.0, 0.0]
        assert th.lethal_radius == pytest.approx(200.0)

    def test_custom_radius(self):
        sc = _import_strix_core()
        th = sc.ThreatState(id=1, position=[0.0, 0.0, 0.0], lethal_radius=500.0)
        assert th.lethal_radius == pytest.approx(500.0)

    def test_with_threat_type(self):
        sc = _import_strix_core()
        th = sc.ThreatState(
            id=1,
            position=[0.0, 0.0, 0.0],
            threat_type=sc.ThreatType.Sam,
        )
        assert th.id == 1

    def test_repr(self):
        sc = _import_strix_core()
        th = sc.ThreatState(id=1, position=[0.0, 0.0, 0.0])
        assert "ThreatState" in repr(th)


class TestAuctioneer:
    def test_basic_auction(self):
        sc = _import_strix_core()
        auctioneer = sc.Auctioneer()
        drones = [
            sc.AuctionDroneState(id=1, position=[0.0, 0.0, 100.0]),
            sc.AuctionDroneState(id=2, position=[100.0, 100.0, 100.0]),
        ]
        tasks = [
            sc.Task(id=10, location=[5.0, 5.0, 50.0]),
            sc.Task(id=20, location=[95.0, 95.0, 50.0]),
        ]
        result = auctioneer.run_auction(drones, tasks, [])
        assert len(result.assignments) == 2
        assert len(result.unassigned_tasks) == 0
        assert result.total_welfare > 0.0

    def test_empty_auction(self):
        sc = _import_strix_core()
        auctioneer = sc.Auctioneer()
        result = auctioneer.run_auction([], [], [])
        assert len(result.assignments) == 0
        assert result.total_welfare == pytest.approx(0.0)

    def test_reauction_trigger(self):
        sc = _import_strix_core()
        auctioneer = sc.Auctioneer()
        assert not auctioneer.needs_reauction
        auctioneer.trigger_reauction()
        assert auctioneer.needs_reauction

    def test_reauction_cleared_after_run(self):
        sc = _import_strix_core()
        auctioneer = sc.Auctioneer()
        auctioneer.trigger_reauction()
        assert auctioneer.needs_reauction
        auctioneer.run_auction([], [], [])
        assert not auctioneer.needs_reauction

    def test_with_threats(self):
        sc = _import_strix_core()
        auctioneer = sc.Auctioneer()
        drones = [sc.AuctionDroneState(id=1, position=[0.0, 0.0, 100.0])]
        tasks = [sc.Task(id=10, location=[50.0, 50.0, 50.0])]
        threats = [sc.ThreatState(id=100, position=[25.0, 25.0, 0.0])]
        result = auctioneer.run_auction(drones, tasks, threats)
        assert len(result.assignments) == 1

    def test_more_tasks_than_drones(self):
        sc = _import_strix_core()
        auctioneer = sc.Auctioneer()
        drones = [sc.AuctionDroneState(id=1, position=[0.0, 0.0, 100.0])]
        tasks = [
            sc.Task(id=10, location=[5.0, 5.0, 50.0]),
            sc.Task(id=20, location=[50.0, 50.0, 50.0]),
            sc.Task(id=30, location=[90.0, 90.0, 50.0]),
        ]
        result = auctioneer.run_auction(drones, tasks, [])
        assert len(result.assignments) == 1
        assert len(result.unassigned_tasks) == 2

    def test_assignment_fields(self):
        sc = _import_strix_core()
        auctioneer = sc.Auctioneer()
        drones = [sc.AuctionDroneState(id=1, position=[0.0, 0.0, 100.0])]
        tasks = [sc.Task(id=10, location=[5.0, 5.0, 50.0])]
        result = auctioneer.run_auction(drones, tasks, [])
        assert len(result.assignments) == 1
        a = result.assignments[0]
        assert a.drone_id == 1
        assert a.task_id == 10
        assert a.bid_score > 0.0

    def test_repr(self):
        sc = _import_strix_core()
        auctioneer = sc.Auctioneer(min_bid_threshold=0.1)
        assert "Auctioneer" in repr(auctioneer)

    def test_result_repr(self):
        sc = _import_strix_core()
        auctioneer = sc.Auctioneer()
        result = auctioneer.run_auction([], [], [])
        assert "AuctionResult" in repr(result)


class TestLossAnalyzer:
    def test_creation(self):
        sc = _import_strix_core()
        la = sc.LossAnalyzer()
        assert la.antifragile_score() == pytest.approx(0.0)
        assert la.active_kill_zones() == 0

    def test_repr(self):
        sc = _import_strix_core()
        la = sc.LossAnalyzer()
        assert "LossAnalyzer" in repr(la)
