"""GNN model matching the Rust GCBF+ architecture.

The model architecture mirrors `strix_core::gcbf::nn::GnnEncoder` exactly:
- 2 GCN layers with mean aggregation
- Barrier head (scalar h value)
- Action head (3D velocity correction with tanh)

This file can be used with either JAX/Flax or pure NumPy for inference.
Training requires JAX (see train.py).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WEIGHT_MAGIC = 0x47434246  # "GCBF"
WEIGHT_VERSION = 1


@dataclass
class GcnLayerParams:
    """Parameters for a single GCN message-passing layer."""
    w_edge: np.ndarray  # [hidden_dim, edge_input_dim]
    b_edge: np.ndarray  # [hidden_dim]
    w_node: np.ndarray  # [hidden_dim, node_input_dim + hidden_dim]
    b_node: np.ndarray  # [hidden_dim]

    @property
    def hidden_dim(self) -> int:
        return self.w_edge.shape[0]


@dataclass
class GnnParams:
    """Complete model parameters matching Rust GnnEncoder."""
    layer1: GcnLayerParams
    layer2: GcnLayerParams
    barrier_w: np.ndarray  # [1, hidden_dim]
    barrier_b: np.ndarray  # [1]
    action_w: np.ndarray   # [3, hidden_dim]
    action_b: np.ndarray   # [3]

    @property
    def hidden_dim(self) -> int:
        return self.layer1.hidden_dim


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def gcn_layer_forward(
    params: GcnLayerParams,
    node_states: np.ndarray,       # [n_nodes, node_dim]
    edge_features: np.ndarray,     # [n_edges, edge_dim]
    edges: np.ndarray,             # [n_edges, 2] — (source, target)
    degree: np.ndarray,            # [n_nodes]
) -> np.ndarray:
    """Forward pass for one GCN layer."""
    n_nodes = node_states.shape[0]
    h = params.hidden_dim

    # Step 1: Edge messages.
    messages = relu(edge_features @ params.w_edge.T + params.b_edge)  # [n_edges, h]

    # Step 2: Mean aggregation per target.
    aggregated = np.zeros((n_nodes, h))
    if len(edges) > 0:
        # Scatter-add messages to targets.
        np.add.at(aggregated, edges[:, 1], messages)
        # Normalize by degree.
        safe_degree = np.maximum(degree, 1).reshape(-1, 1)
        aggregated /= safe_degree

    # Step 3: Node update.
    concat = np.concatenate([node_states, aggregated], axis=1)
    output = relu(concat @ params.w_node.T + params.b_node)

    return output


def gnn_forward(
    params: GnnParams,
    node_features: np.ndarray,     # [n_nodes, 9]
    edge_features: np.ndarray,     # [n_edges, 7]
    edges: np.ndarray,             # [n_edges, 2]
    degree: np.ndarray,            # [n_nodes]
    max_correction: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Full GNN forward pass. Returns (barrier_values, action_corrections)."""
    if node_features.shape[0] == 0:
        return np.array([]), np.array([]).reshape(0, 3)

    # Layer 1.
    h1 = gcn_layer_forward(params.layer1, node_features, edge_features, edges, degree)

    # Layer 2: modulate edge features by hidden state norms.
    if len(edges) > 0:
        src_norms = np.maximum(np.linalg.norm(h1[edges[:, 0]], axis=1, keepdims=True), 1e-6)
        tgt_norms = np.maximum(np.linalg.norm(h1[edges[:, 1]], axis=1, keepdims=True), 1e-6)
        scale = np.sqrt(src_norms * tgt_norms)
        ef2 = edge_features / scale
    else:
        ef2 = edge_features

    h2 = gcn_layer_forward(params.layer2, h1, ef2, edges, degree)

    # Heads.
    barrier_values = (h2 @ params.barrier_w.T + params.barrier_b).flatten()
    action_raw = h2 @ params.action_w.T + params.action_b
    action_corrections = np.tanh(action_raw) * max_correction

    return barrier_values, action_corrections


def init_params(hidden_dim: int = 16, seed: int = 42) -> GnnParams:
    """Initialize random parameters matching Rust `default_weights()`."""
    rng = np.random.default_rng(seed)
    edge_dim = 7
    node_dim = 9

    def xavier(fan_in: int, fan_out: int) -> np.ndarray:
        scale = np.sqrt(2.0 / fan_in)
        return rng.standard_normal((fan_out, fan_in)) * scale

    layer1 = GcnLayerParams(
        w_edge=xavier(edge_dim, hidden_dim),
        b_edge=np.zeros(hidden_dim),
        w_node=xavier(node_dim + hidden_dim, hidden_dim),
        b_node=np.zeros(hidden_dim),
    )
    layer2 = GcnLayerParams(
        w_edge=xavier(edge_dim, hidden_dim),
        b_edge=np.zeros(hidden_dim),
        w_node=xavier(hidden_dim * 2, hidden_dim),
        b_node=np.zeros(hidden_dim),
    )
    return GnnParams(
        layer1=layer1,
        layer2=layer2,
        barrier_w=xavier(hidden_dim, 1),
        barrier_b=np.array([0.5]),  # positive bias → default safe
        action_w=xavier(hidden_dim, 3),
        action_b=np.zeros(3),
    )


def export_weights(params: GnnParams, path: str | Path) -> None:
    """Export model weights to JSON format loadable by Rust."""
    data = {
        "magic": WEIGHT_MAGIC,
        "version": WEIGHT_VERSION,
        "hidden_dim": params.hidden_dim,
        "edge_dim_l1": params.layer1.w_edge.shape[1],
        "node_dim_l1": params.layer1.w_node.shape[1] - params.hidden_dim,
        "layer1_w_edge": params.layer1.w_edge.flatten().tolist(),
        "layer1_b_edge": params.layer1.b_edge.tolist(),
        "layer1_w_node": params.layer1.w_node.flatten().tolist(),
        "layer1_b_node": params.layer1.b_node.tolist(),
        "layer2_w_edge": params.layer2.w_edge.flatten().tolist(),
        "layer2_b_edge": params.layer2.b_edge.tolist(),
        "layer2_w_node": params.layer2.w_node.flatten().tolist(),
        "layer2_b_node": params.layer2.b_node.tolist(),
        "barrier_w": params.barrier_w.flatten().tolist(),
        "barrier_b": params.barrier_b.tolist(),
        "action_w": params.action_w.flatten().tolist(),
        "action_b": params.action_b.tolist(),
    }
    Path(path).write_text(json.dumps(data))


def load_weights(path: str | Path) -> GnnParams:
    """Load weights from JSON format."""
    data = json.loads(Path(path).read_text())
    if data["magic"] != WEIGHT_MAGIC:
        raise ValueError(f"Invalid magic: {data['magic']:#x}, expected {WEIGHT_MAGIC:#x}")
    if data["version"] != WEIGHT_VERSION:
        raise ValueError(f"Unsupported version: {data['version']}, expected {WEIGHT_VERSION}")
    h = data["hidden_dim"]
    ed = data["edge_dim_l1"]
    nd = data["node_dim_l1"]

    layer1 = GcnLayerParams(
        w_edge=np.array(data["layer1_w_edge"]).reshape(h, ed),
        b_edge=np.array(data["layer1_b_edge"]),
        w_node=np.array(data["layer1_w_node"]).reshape(h, nd + h),
        b_node=np.array(data["layer1_b_node"]),
    )
    layer2 = GcnLayerParams(
        w_edge=np.array(data["layer2_w_edge"]).reshape(h, 7),
        b_edge=np.array(data["layer2_b_edge"]),
        w_node=np.array(data["layer2_w_node"]).reshape(h, h * 2),
        b_node=np.array(data["layer2_b_node"]),
    )
    return GnnParams(
        layer1=layer1,
        layer2=layer2,
        barrier_w=np.array(data["barrier_w"]).reshape(1, h),
        barrier_b=np.array(data["barrier_b"]),
        action_w=np.array(data["action_w"]).reshape(3, h),
        action_b=np.array(data["action_b"]),
    )
