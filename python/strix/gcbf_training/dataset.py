"""Training data generation for GCBF+ from classical CBF teacher.

Generates scenarios by randomizing drone positions/velocities,
running the classical CBF, and recording the corrections as labels.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingFrame:
    """One training frame: graph state + CBF labels."""
    positions: np.ndarray       # [n_drones, 3] NED
    velocities: np.ndarray      # [n_drones, 3]
    goals: np.ndarray           # [n_drones, 3]
    corrections: np.ndarray     # [n_drones, 3] — CBF correction vectors
    barrier_labels: np.ndarray  # [n_drones] — 1.0 if safe, -1.0 if active


def generate_random_scenario(
    n_drones: int,
    area_size: float = 200.0,
    min_separation: float = 5.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random drone positions, velocities, and goals.

    Returns (positions, velocities, goals) all [n_drones, 3] in NED.
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-area_size / 2, area_size / 2, (n_drones, 3))
    positions[:, 2] = rng.uniform(-200, -20, n_drones)  # NED altitude
    velocities = rng.uniform(-5, 5, (n_drones, 3))
    goals = rng.uniform(-area_size / 2, area_size / 2, (n_drones, 3))
    goals[:, 2] = rng.uniform(-150, -30, n_drones)
    return positions, velocities, goals


def compute_classical_cbf_labels(
    positions: np.ndarray,
    velocities: np.ndarray,
    min_separation: float = 5.0,
    alpha: float = 1.0,
    max_correction: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute classical pairwise CBF corrections and barrier labels.

    This is the Python equivalent of `cbf::cbf_filter_with_neighbor_states`.
    Used as the training signal (teacher) for the GNN.

    Returns (corrections [n, 3], barrier_labels [n]).
    """
    n = positions.shape[0]
    corrections = np.zeros((n, 3))
    barrier_labels = np.ones(n)  # 1.0 = safe by default

    for i in range(n):
        correction = np.zeros(3)
        for j in range(n):
            if i == j:
                continue
            diff = positions[i] - positions[j]
            dist_sq = np.dot(diff, diff)
            dist = np.sqrt(dist_sq)
            if dist < 1e-6:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = diff / dist

            rel_vel = velocities[i] - velocities[j]
            closing = max(0, -np.dot(rel_vel, direction))
            ttc = dist / closing if closing > 0.1 else float("inf")
            margin_scale = 1.5 if ttc < 3 else (1.0 if ttc < 6 else 0.5)
            eff_sep = min_separation + margin_scale * closing
            h = dist_sq - eff_sep ** 2

            if h < 0:
                penetration = max(0.5, eff_sep - dist)
                radial_rel = np.dot(rel_vel, direction)
                cancel = max(0, -radial_rel)
                push = min(alpha * penetration + cancel, max_correction)
                correction += direction * push
                barrier_labels[i] = -1.0
            else:
                dh_dt = 2.0 * np.dot(diff, velocities[i] - velocities[j])
                constraint = dh_dt + alpha * h
                if constraint < 0:
                    needed = -constraint / max(1e-6, 2.0 * dist)
                    radial_rel = np.dot(rel_vel, direction)
                    cancel = max(0, -radial_rel) * 0.5
                    correction += direction * min(needed + cancel, max_correction)
                    barrier_labels[i] = -1.0

        mag = np.linalg.norm(correction)
        if mag > max_correction:
            correction *= max_correction / mag
        corrections[i] = correction

    return corrections, barrier_labels


def generate_training_batch(
    n_scenarios: int = 100,
    drones_range: tuple[int, int] = (4, 20),
    seed: int = 42,
) -> list[TrainingFrame]:
    """Generate a batch of training frames with varying drone counts."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_scenarios):
        n = rng.integers(drones_range[0], drones_range[1] + 1)
        positions, velocities, goals = generate_random_scenario(
            n, seed=seed + i
        )
        corrections, barrier_labels = compute_classical_cbf_labels(
            positions, velocities
        )
        frames.append(TrainingFrame(
            positions=positions,
            velocities=velocities,
            goals=goals,
            corrections=corrections,
            barrier_labels=barrier_labels,
        ))
    return frames
