"""GCBF+ training pipeline for STRIX drone swarm safety.

This package trains a Graph Neural Network (GNN) to approximate the classical
Control Barrier Function (CBF) safety filter, enabling O(n·k) scaling vs O(n²).

Based on: "GCBF+: A Neural Graph Control Barrier Function Framework for
Distributed Safe Multi-Agent Control" (arXiv:2401.14554)

Workflow:
1. Generate training data from classical CBF via strix-playground simulations.
2. Train the GNN model on 8-agent scenarios.
3. Export weights to JSON format loadable by the Rust `gcbf::weights` module.
"""
