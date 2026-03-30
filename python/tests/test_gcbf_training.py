"""Tests for GCBF+ training pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np

from strix.gcbf_training.model import (
    init_params,
    gnn_forward,
    export_weights,
    load_weights,
    WEIGHT_MAGIC,
    WEIGHT_VERSION,
)
from strix.gcbf_training.dataset import (
    generate_random_scenario,
    compute_classical_cbf_labels,
    generate_training_batch,
)
from strix.gcbf_training.train import build_graph, compute_loss


class TestModel:
    def test_init_params(self):
        params = init_params(hidden_dim=8)
        assert params.hidden_dim == 8
        assert params.layer1.w_edge.shape == (8, 7)
        assert params.layer2.w_node.shape == (8, 16)
        assert params.barrier_w.shape == (1, 8)
        assert params.action_w.shape == (3, 8)

    def test_forward_output_shapes(self):
        params = init_params(hidden_dim=8)
        positions = np.array([[0, 0, -50], [10, 0, -50], [0, 10, -50]], dtype=float)
        velocities = np.zeros((3, 3))
        goals = np.array([[100, 0, -50], [0, 100, -50], [50, 50, -50]], dtype=float)
        node_feats, edge_feats, edges, degree = build_graph(
            positions, velocities, goals, k=2
        )
        barriers, actions = gnn_forward(params, node_feats, edge_feats, edges, degree)
        assert barriers.shape == (3,)
        assert actions.shape == (3, 3)

    def test_forward_finite_outputs(self):
        params = init_params(hidden_dim=16)
        n = 20
        rng = np.random.default_rng(42)
        positions = rng.uniform(-100, 100, (n, 3))
        positions[:, 2] = rng.uniform(-200, -20, n)
        velocities = rng.uniform(-5, 5, (n, 3))
        goals = rng.uniform(-100, 100, (n, 3))
        node_feats, edge_feats, edges, degree = build_graph(
            positions, velocities, goals
        )
        barriers, actions = gnn_forward(params, node_feats, edge_feats, edges, degree)
        assert np.all(np.isfinite(barriers))
        assert np.all(np.isfinite(actions))

    def test_action_clamped(self):
        params = init_params(hidden_dim=8)
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        velocities = np.array([[100, 100, 100], [100, 100, 100]], dtype=float)
        goals = np.zeros((2, 3))
        node_feats, edge_feats, edges, degree = build_graph(
            positions, velocities, goals, k=1
        )
        max_corr = 5.0
        _, actions = gnn_forward(params, node_feats, edge_feats, edges, degree, max_corr)
        assert np.all(np.abs(actions) <= max_corr + 1e-10)

    def test_export_load_roundtrip(self):
        params = init_params(hidden_dim=8, seed=123)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        export_weights(params, path)
        loaded = load_weights(path)
        assert loaded.hidden_dim == 8
        np.testing.assert_allclose(loaded.layer1.w_edge, params.layer1.w_edge)
        np.testing.assert_allclose(loaded.barrier_b, params.barrier_b)
        Path(path).unlink()

    def test_rust_compatible_json(self):
        """Verify exported JSON has the right structure for Rust loading."""
        params = init_params(hidden_dim=8)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        export_weights(params, path)
        data = json.loads(Path(path).read_text())
        assert data["magic"] == WEIGHT_MAGIC
        assert data["version"] == WEIGHT_VERSION
        assert data["hidden_dim"] == 8
        assert len(data["layer1_w_edge"]) == 8 * 7
        assert len(data["barrier_w"]) == 8
        assert len(data["action_w"]) == 3 * 8
        Path(path).unlink()


class TestDataset:
    def test_generate_scenario(self):
        pos, vel, goals = generate_random_scenario(10, seed=0)
        assert pos.shape == (10, 3)
        assert vel.shape == (10, 3)
        assert goals.shape == (10, 3)
        # NED: altitude should be negative
        assert np.all(pos[:, 2] < 0)

    def test_classical_cbf_labels(self):
        # Two close drones should trigger correction.
        positions = np.array([[0, 0, -50], [3, 0, -50]], dtype=float)
        velocities = np.array([[5, 0, 0], [0, 0, 0]], dtype=float)
        corrections, labels = compute_classical_cbf_labels(positions, velocities)
        assert corrections.shape == (2, 3)
        # At least one drone should be unsafe (3m apart < 5m min_separation).
        assert np.any(labels < 0), "close drones should be marked unsafe"

    def test_safe_drones_get_zero_correction(self):
        # Two far-apart drones moving away.
        positions = np.array([[0, 0, -50], [100, 0, -50]], dtype=float)
        velocities = np.array([[-5, 0, 0], [5, 0, 0]], dtype=float)
        corrections, labels = compute_classical_cbf_labels(positions, velocities)
        assert np.all(labels > 0), "far drones should be safe"
        assert np.allclose(corrections, 0, atol=1e-6), "no correction needed"

    def test_generate_training_batch(self):
        frames = generate_training_batch(n_scenarios=5, seed=42)
        assert len(frames) == 5
        for f in frames:
            assert f.positions.shape[0] == f.corrections.shape[0]
            assert f.barrier_labels.shape[0] == f.positions.shape[0]


class TestTraining:
    def test_compute_loss(self):
        params = init_params(hidden_dim=8)
        frames = generate_training_batch(n_scenarios=3, seed=42)
        loss, details = compute_loss(params, frames[0])
        assert np.isfinite(loss)
        assert "barrier" in details
        assert "action" in details

    def test_build_graph(self):
        positions = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
        velocities = np.zeros((3, 3))
        goals = np.zeros((3, 3))
        node_feats, edge_feats, edges, degree = build_graph(
            positions, velocities, goals, k=2
        )
        assert node_feats.shape == (3, 9)
        assert edges.shape[1] == 2
        assert len(edges) == sum(degree)
