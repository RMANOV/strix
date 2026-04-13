# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible shim for the generic public LLM interface.

The public repository keeps this legacy import path to avoid breaking older
experiments. New public integrations should prefer
``strix.llm.autonomy_llm.AutonomyLLM``.
"""

from .autonomy_llm import AutonomyLLM, LLMConfig

MilitaryLLM = AutonomyLLM
