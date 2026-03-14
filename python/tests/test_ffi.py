"""Tests for the Rust FFI bridge (strix._strix_core).

These tests verify that the PyO3 bridge correctly exposes:
- ParticleNavFilter (creation, step, state extraction)
- DroneState (construction, fields)
- Regime (enum values, from_index)
- SensorConfig (defaults)
- detect_jamming (CUSUM)
- detect_regime (regime detection)
"""

import pytest


def _import_strix_core():
    """Try to import the native module, skip if not built."""
    try:
        from strix import _strix_core
        return _strix_core
    except ImportError:
        pytest.skip("strix._strix_core not built (run: maturin develop --features python)")


class TestRegime:
    def test_regime_variants(self):
        sc = _import_strix_core()
        assert sc.Regime.Patrol is not None
        assert sc.Regime.Engage is not None
        assert sc.Regime.Evade is not None

    def test_from_index(self):
        sc = _import_strix_core()
        r = sc.Regime.from_index(0)
        assert r == sc.Regime.Patrol
        r = sc.Regime.from_index(1)
        assert r == sc.Regime.Engage
        r = sc.Regime.from_index(2)
        assert r == sc.Regime.Evade

    def test_from_index_invalid(self):
        sc = _import_strix_core()
        with pytest.raises(ValueError):
            sc.Regime.from_index(5)

    def test_as_index(self):
        sc = _import_strix_core()
        assert sc.Regime.Patrol.as_index() == 0
        assert sc.Regime.Engage.as_index() == 1
        assert sc.Regime.Evade.as_index() == 2

    def test_repr(self):
        sc = _import_strix_core()
        assert "Patrol" in repr(sc.Regime.Patrol)


class TestSensorConfig:
    def test_defaults(self):
        sc = _import_strix_core()
        cfg = sc.SensorConfig()
        assert cfg.imu_accel_noise == pytest.approx(0.1)
        assert cfg.baro_noise == pytest.approx(0.5)

    def test_repr(self):
        sc = _import_strix_core()
        cfg = sc.SensorConfig()
        assert "SensorConfig" in repr(cfg)


class TestDroneState:
    def test_construction(self):
        sc = _import_strix_core()
        ds = sc.DroneState(drone_id=42, position=[1.0, 2.0, 3.0])
        assert ds.drone_id == 42
        assert ds.position == [1.0, 2.0, 3.0]
        assert ds.velocity == [0.0, 0.0, 0.0]
        assert ds.regime == sc.Regime.Patrol

    def test_repr(self):
        sc = _import_strix_core()
        ds = sc.DroneState(drone_id=1, position=[0.0, 0.0, -50.0])
        assert "DroneState" in repr(ds)


class TestParticleNavFilter:
    def test_creation(self):
        sc = _import_strix_core()
        f = sc.ParticleNavFilter(n_particles=100, position=[0.0, 0.0, -50.0])
        assert f.n_particles == 100
        assert f.ess > 0

    def test_step_returns_tuple(self):
        sc = _import_strix_core()
        f = sc.ParticleNavFilter(n_particles=50, position=[0.0, 0.0, -50.0])

        pos, vel, probs = f.step(
            observations=[("barometer", 50.0)],
            threat_bearing=[1.0, 0.0, 0.0],
            vel_gain=1.0,
            dt=0.1,
        )

        assert len(pos) == 3
        assert len(vel) == 3
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 0.01

    def test_step_default_args(self):
        sc = _import_strix_core()
        f = sc.ParticleNavFilter()
        pos, vel, probs = f.step()
        assert len(pos) == 3

    def test_multiple_steps(self):
        sc = _import_strix_core()
        f = sc.ParticleNavFilter(n_particles=100, position=[10.0, 20.0, -50.0])

        for _ in range(10):
            pos, vel, probs = f.step(
                observations=[("barometer", 50.0)],
                dt=0.1,
            )

        # Position should still be roughly near initial
        assert abs(pos[0]) < 200
        assert abs(pos[1]) < 200

    def test_to_drone_state(self):
        sc = _import_strix_core()
        f = sc.ParticleNavFilter(n_particles=50, position=[0.0, 0.0, -50.0], drone_id=7)
        ds = f.to_drone_state()
        assert ds.drone_id == 7
        assert len(ds.position) == 3

    def test_repr(self):
        sc = _import_strix_core()
        f = sc.ParticleNavFilter(n_particles=50, position=[0.0, 0.0, -50.0])
        assert "ParticleNavFilter" in repr(f)


class TestDetectJamming:
    def test_no_jamming_on_stable(self):
        sc = _import_strix_core()
        metrics = [30.0] * 30
        is_jammed, direction, _ = sc.detect_jamming(metrics)
        assert not is_jammed
        assert direction == 0

    def test_detects_jamming(self):
        sc = _import_strix_core()
        metrics = [30.0] * 20 + [5.0] * 20  # sudden SNR drop
        is_jammed, direction, _ = sc.detect_jamming(metrics)
        assert is_jammed
        assert direction == -1


class TestDetectRegime:
    def test_patrol_default(self):
        sc = _import_strix_core()
        r = sc.detect_regime(
            cusum_triggered=False,
            threat_distance=1000.0,
            closing_rate=0.0,
        )
        assert r == sc.Regime.Patrol

    def test_engage_on_approach(self):
        sc = _import_strix_core()
        r = sc.detect_regime(
            cusum_triggered=False,
            threat_distance=300.0,
            closing_rate=-1.0,
            hurst=0.7,
        )
        assert r == sc.Regime.Engage

    def test_evade_on_close_threat(self):
        sc = _import_strix_core()
        r = sc.detect_regime(
            cusum_triggered=True,
            threat_distance=100.0,
            closing_rate=-5.0,
        )
        assert r == sc.Regime.Evade
