# SPDX-License-Identifier: Apache-2.0

"""Generic language-model hooks for public STRIX workflows.

This module keeps the public interface deliberately neutral. It exposes a
small optional API for richer intent parsing, situation narration, and
decision explanation without embedding domain-specific doctrines or
program-specific source material in the public tree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("strix.llm.autonomy_llm")


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an optional autonomy-oriented language model."""

    model_path: str = ""
    model_name: str = "phi-3-mini-4k"
    backend: str = "llama_cpp"  # llama_cpp | onnx | vllm
    max_tokens: int = 512
    temperature: float = 0.1  # low temperature for deterministic structured output
    context_window: int = 4096
    latency_budget_ms: float = 200.0
    quantization: str = "Q4_K_M"


class AutonomyLLM:
    """Neutral public stub for optional language-model integration."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or LLMConfig()
        self._loaded = False
        logger.info(
            "AutonomyLLM created (stub) -- model=%s backend=%s",
            self._config.model_name,
            self._config.backend,
        )

    def load(self) -> bool:
        """Load the model into memory."""

        logger.info("AutonomyLLM.load() called (stub -- no model file)")
        self._loaded = True
        return True

    def parse_intent(self, text: str) -> dict[str, object]:
        """Parse a natural-language request into structured intent data.

        The public stub keeps a stable return contract and yields an empty
        mapping until a concrete provider is wired in.
        """

        if not self._loaded:
            logger.warning("LLM not loaded -- returning empty parse")
            return {}

        logger.info("parse_intent called (stub): '%s'", text[:80])
        return {}

    def narrate_situation(self, snapshot_dict: dict) -> str:
        """Generate a neutral status summary from a world snapshot."""

        if not self._loaded:
            return "LLM not loaded. Unable to generate briefing."

        n_drones = snapshot_dict.get("active_drone_count", 0)
        n_threats = len(snapshot_dict.get("threats", {}))
        regime = snapshot_dict.get("regime", "UNKNOWN")

        return (
            f"Situation: {n_drones} drones active, {n_threats} tracked contacts. "
            f"Current regime: {regime}."
        )

    def explain_decision(self, decision_dict: dict) -> str:
        """Generate a concise human-readable explanation of a decision."""

        if not self._loaded:
            return "LLM not loaded. Unable to generate explanation."

        drone_id = decision_dict.get("drone_id", "?")
        kind = decision_dict.get("kind", "?")
        return f"Drone {drone_id} was assigned to {kind} based on proximity and capability match."

    def unload(self) -> None:
        """Release model resources."""

        self._loaded = False
        logger.info("AutonomyLLM unloaded")
