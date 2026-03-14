"""Placeholder for military-domain finetuned LLM.

Target models:
    - Phi-3 (3.8B parameters) -- Microsoft, MIT license
    - Llama-3.2 (3B parameters) -- Meta, community license

Finetuning data (synthetic + open-source):
    - NATO ATP-3.3.8.1 (Joint Tactical Air Operations)
    - FM 3-04 Army Aviation
    - JP 3-30 Joint Air Operations
    - Open-source AAR (After Action Reports)
    - Synthetic tactical dialogues generated from doctrine

Inference backends:
    - llama.cpp for edge deployment (CUDA / Metal / CPU)
    - ONNX Runtime for optimized cross-platform inference
    - vLLM for high-throughput server deployment

The LLM enhances three capabilities:

1. **Intent Parsing**: richer understanding of natural language commands
   than the keyword-based parser, including implicit constraints,
   doctrinal conventions, and ambiguity resolution.

2. **Situation Narration**: converts WorldSnapshot into natural language
   briefings for the commander ("Two enemy vehicles approaching from the
   northeast, estimated arrival in 4 minutes. Recommend shifting to
   ENGAGE regime.").

3. **Decision Explanation**: generates human-readable rationale for
   brain decisions ("Drone 7 was assigned to the northern waypoint
   because it had the highest energy and closest proximity, despite
   moderate threat exposure.").

Integration:
    When available, the LLM wraps the keyword parser in
    ``strix.nlp.intent_parser`` and provides fallback if the LLM
    fails or exceeds latency budget (200ms for tactical, 2s for
    strategic).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("strix.llm.military_llm")


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the military LLM."""

    model_path: str = ""
    model_name: str = "phi-3-mini-4k"
    backend: str = "llama_cpp"  # llama_cpp | onnx | vllm
    max_tokens: int = 512
    temperature: float = 0.1  # low temperature for deterministic tactical output
    context_window: int = 4096
    latency_budget_ms: float = 200.0  # max allowed inference time
    quantization: str = "Q4_K_M"  # llama.cpp quantization level


class MilitaryLLM:
    """Military-domain finetuned language model.

    Placeholder implementation -- all methods return stub outputs.
    The real implementation will load a quantized model and run
    inference via the configured backend.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or LLMConfig()
        self._loaded = False
        logger.info(
            "MilitaryLLM created (stub) -- model=%s backend=%s",
            self._config.model_name,
            self._config.backend,
        )

    def load(self) -> bool:
        """Load the model into memory.

        Returns True if successful, False otherwise.
        """
        logger.info("MilitaryLLM.load() called (stub -- no model file)")
        # TODO: Implement actual model loading
        # if self._config.backend == "llama_cpp":
        #     from llama_cpp import Llama
        #     self._model = Llama(model_path=self._config.model_path, ...)
        self._loaded = True
        return True

    def parse_intent(self, text: str) -> dict:
        """Parse a natural language command using the LLM.

        Returns a dictionary compatible with MissionIntent construction.
        Falls back to empty dict if model is not loaded.
        """
        if not self._loaded:
            logger.warning("LLM not loaded -- returning empty parse")
            return {}

        # TODO: Implement actual LLM inference
        # prompt = f"<|system|>You are a military command parser...\n<|user|>{text}\n<|assistant|>"
        # response = self._model(prompt, max_tokens=self._config.max_tokens)
        logger.info("parse_intent called (stub): '%s'", text[:80])
        return {"raw_text": text, "parsed": False, "reason": "LLM not yet implemented"}

    def narrate_situation(self, snapshot_dict: dict) -> str:
        """Generate a natural language situation briefing from world state.

        Parameters
        ----------
        snapshot_dict : dict
            Serialized WorldSnapshot.

        Returns
        -------
        str
            Human-readable situation briefing.
        """
        if not self._loaded:
            return "LLM not loaded. Unable to generate briefing."

        # TODO: Implement actual narration
        n_drones = snapshot_dict.get("active_drone_count", 0)
        n_threats = len(snapshot_dict.get("threats", {}))
        regime = snapshot_dict.get("regime", "UNKNOWN")

        return (
            f"Situation: {n_drones} drones active, {n_threats} threats tracked. "
            f"Current regime: {regime}."
        )

    def explain_decision(self, decision_dict: dict) -> str:
        """Generate a human-readable explanation of a decision.

        Parameters
        ----------
        decision_dict : dict
            Serialized Decision with scoring components.

        Returns
        -------
        str
            Natural language explanation.
        """
        if not self._loaded:
            return "LLM not loaded. Unable to generate explanation."

        # TODO: Implement actual explanation generation
        drone_id = decision_dict.get("drone_id", "?")
        kind = decision_dict.get("kind", "?")
        return f"Drone {drone_id} was assigned to {kind} based on proximity and capability match."

    def unload(self) -> None:
        """Release model from memory."""
        self._loaded = False
        logger.info("MilitaryLLM unloaded")
