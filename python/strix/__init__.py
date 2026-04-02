"""STRIX -- Swarm Tactical Reasoning and Intelligence eXchange

A drone swarm orchestrator built on quantitative trading mathematics.
The battlefield is a market. Drones are traders. Missions are positions.
The enemy is a counterparty.
"""

from .orientation import (
    BrokenAssumption,
    OrientationConfig,
    OrientationEngine,
    OrientationSnapshot,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BrokenAssumption",
    "OrientationConfig",
    "OrientationEngine",
    "OrientationSnapshot",
]