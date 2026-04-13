# SPDX-License-Identifier: Apache-2.0

from strix.llm import AutonomyLLM


def test_parse_intent_returns_empty_dict_when_unloaded():
    llm = AutonomyLLM()
    assert llm.parse_intent("survey north ridge") == {}


def test_parse_intent_returns_empty_dict_when_loaded_stub():
    llm = AutonomyLLM()
    llm.load()
    assert llm.parse_intent("survey north ridge") == {}
