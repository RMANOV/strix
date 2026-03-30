"""GCBF+ training script.

Trains the GNN model using classical CBF as teacher supervisor.
Uses pure NumPy gradient descent (no JAX dependency for v1).

Usage:
    python -m strix.gcbf_training.train --hidden-dim 16 --epochs 200 --output weights.json
"""

import argparse

import numpy as np

from .dataset import generate_training_batch, TrainingFrame
from .model import (
    GcnLayerParams,
    GnnParams,
    export_weights,
    gnn_forward,
    init_params,
)  # noqa: F401 (GcnLayerParams used in _perturb_params)


def build_graph(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    k: int = 8,
    comm_radius: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build k-NN graph from positions.

    Returns (node_features, edge_features, edges, degree).
    """
    n = positions.shape[0]

    # Node features: [goal_rel, vel, 0,0,0].
    goal_rel = goals - positions
    node_features = np.zeros((n, 9))
    node_features[:, :3] = goal_rel
    node_features[:, 3:6] = velocities

    # k-NN edges.
    edges_list = []
    edge_feats_list = []
    degree = np.zeros(n, dtype=int)

    for i in range(n):
        dists = np.linalg.norm(positions - positions[i], axis=1)
        dists[i] = float("inf")  # exclude self
        within_radius = np.where(dists <= comm_radius)[0]
        if len(within_radius) > k:
            idx = np.argpartition(dists[within_radius], k)[:k]
            neighbors = within_radius[idx]
        else:
            neighbors = within_radius

        for j in neighbors:
            rel_pos = positions[j] - positions[i]
            rel_vel = velocities[j] - velocities[i]
            dist = dists[j]
            edges_list.append([j, i])  # source → target
            edge_feats_list.append([*rel_pos, *rel_vel, dist])

        degree[i] = len(neighbors)

    if edges_list:
        edges = np.array(edges_list, dtype=int)
        edge_features = np.array(edge_feats_list)
    else:
        edges = np.zeros((0, 2), dtype=int)
        edge_features = np.zeros((0, 7))

    return node_features, edge_features, edges, degree


def compute_loss(
    params: GnnParams,
    frame: TrainingFrame,
    max_correction: float = 10.0,
) -> tuple[float, dict[str, float]]:
    """Compute training loss for a single frame.

    Loss = L_barrier + L_action + L_unsafe_penalty
    """
    node_feats, edge_feats, edges, degree = build_graph(
        frame.positions, frame.velocities, frame.goals
    )
    barriers, actions = gnn_forward(
        params, node_feats, edge_feats, edges, degree, max_correction
    )
    if len(barriers) == 0:
        return 0.0, {}

    # L_barrier: barriers should be positive for safe states, negative for unsafe.
    safe_mask = frame.barrier_labels > 0
    unsafe_mask = ~safe_mask

    # Hinge loss: safe states should have h > margin, unsafe should have h < -margin.
    margin = 0.5
    l_safe = np.mean(np.maximum(0, margin - barriers[safe_mask]) ** 2) if safe_mask.any() else 0.0
    l_unsafe = np.mean(np.maximum(0, barriers[unsafe_mask] + margin) ** 2) if unsafe_mask.any() else 0.0
    l_barrier = l_safe + 2.0 * l_unsafe  # weight unsafe more

    # L_action: action corrections should match classical CBF corrections.
    l_action = np.mean((actions - frame.corrections) ** 2)

    total = l_barrier + l_action
    return total, {"barrier": l_barrier, "action": l_action, "safe": l_safe, "unsafe": l_unsafe}


def train(
    hidden_dim: int = 16,
    epochs: int = 100,
    lr: float = 1e-3,
    n_scenarios: int = 50,
    seed: int = 42,
    output_path: str = "gcbf_weights.json",
) -> GnnParams:
    """Train the GCBF+ model and export weights."""
    print(f"Generating {n_scenarios} training scenarios...")
    frames = generate_training_batch(n_scenarios=n_scenarios, seed=seed)
    print(f"Generated {len(frames)} frames, avg {np.mean([len(f.positions) for f in frames]):.0f} drones/frame")

    params = init_params(hidden_dim=hidden_dim, seed=seed)

    print(f"Training for {epochs} epochs with hidden_dim={hidden_dim}...")
    for epoch in range(epochs):
        total_loss = 0.0
        for frame in frames:
            loss, details = compute_loss(params, frame)
            total_loss += loss

        avg_loss = total_loss / len(frames)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}: avg_loss = {avg_loss:.6f}")

        # NOTE: For v1, we use a simplified update.
        # The numerical gradient is too slow for full training.
        # Use the first frame's gradient as a stochastic estimate.
        if avg_loss > 1e-6:
            # Simple perturbation-based optimization (evolution strategy).
            rng = np.random.default_rng(seed + epoch)
            best_loss = avg_loss
            best_params = params
            for trial in range(5):
                # Perturb all parameters slightly.
                p_trial = _perturb_params(params, lr * 0.1, rng)
                trial_loss = sum(compute_loss(p_trial, f)[0] for f in frames[:10]) / min(10, len(frames))
                if trial_loss < best_loss:
                    best_loss = trial_loss
                    best_params = p_trial
            params = best_params

    print(f"Exporting weights to {output_path}")
    export_weights(params, output_path)
    print(f"Done. Final loss: {avg_loss:.6f}")
    return params


def _perturb_params(params: GnnParams, scale: float, rng: np.random.Generator) -> GnnParams:
    """Add small random noise to all parameters."""
    def perturb(arr: np.ndarray) -> np.ndarray:
        return arr + rng.standard_normal(arr.shape) * scale

    return GnnParams(
        layer1=GcnLayerParams(
            w_edge=perturb(params.layer1.w_edge),
            b_edge=perturb(params.layer1.b_edge),
            w_node=perturb(params.layer1.w_node),
            b_node=perturb(params.layer1.b_node),
        ),
        layer2=GcnLayerParams(
            w_edge=perturb(params.layer2.w_edge),
            b_edge=perturb(params.layer2.b_edge),
            w_node=perturb(params.layer2.w_node),
            b_node=perturb(params.layer2.b_node),
        ),
        barrier_w=perturb(params.barrier_w),
        barrier_b=perturb(params.barrier_b),
        action_w=perturb(params.action_w),
        action_b=perturb(params.action_b),
    )


def main():
    parser = argparse.ArgumentParser(description="Train GCBF+ model")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scenarios", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="gcbf_weights.json")
    args = parser.parse_args()
    train(
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        n_scenarios=args.scenarios,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
