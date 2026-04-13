# SPDX-License-Identifier: Apache-2.0

"""Optional public language-model interfaces for STRIX."""

from .autonomy_llm import AutonomyLLM, LLMConfig
from .military_llm import MilitaryLLM

__all__ = ["AutonomyLLM", "LLMConfig", "MilitaryLLM"]
