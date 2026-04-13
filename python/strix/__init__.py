# SPDX-License-Identifier: Apache-2.0

"""STRIX -- Swarm Coordination, Safety, and Explainable Autonomy.

Apache-2.0 public core for multi-agent autonomy research, simulation,
and explainable coordination in degraded environments.
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
