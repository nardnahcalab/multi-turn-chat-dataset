"""Unit tests for the shared generator_base helpers."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator_base import (
    CHARS_PER_TOKEN,
    BaseConversationGenerator,
    estimate_output_tokens,
    estimate_tokens,
)


def test_chars_per_token_is_four():
    assert CHARS_PER_TOKEN == 4


@pytest.mark.parametrize("text,expected", [
    ("", 0),
    ("abc", 0),      # 3 // 4
    ("abcd", 1),     # 4 // 4
    ("a" * 40, 10),
])
def test_estimate_tokens(text, expected):
    assert estimate_tokens(text) == expected


def test_estimate_output_tokens_min_one():
    assert estimate_output_tokens("") == 1
    assert estimate_output_tokens("abc") == 1
    assert estimate_output_tokens("a" * 40) == 10


def test_estimate_tokens_accepts_non_str():
    # multimodal content can be a list; helper must not crash
    assert estimate_tokens(["x", "y"]) == len(str(["x", "y"])) // CHARS_PER_TOKEN


def test_new_conversation_id_is_seed_reproducible():
    a = BaseConversationGenerator({}, seed=123)
    b = BaseConversationGenerator({}, seed=123)
    ids_a = [a.new_conversation_id() for _ in range(5)]
    ids_b = [b.new_conversation_id() for _ in range(5)]
    assert ids_a == ids_b
    # different seed -> different stream
    c = BaseConversationGenerator({}, seed=456)
    assert c.new_conversation_id() != ids_a[0]


def _conv(messages):
    return {"conversation_id": "sid", "messages": json.dumps(messages)}


def test_to_aiperf_multi_turn_keeps_user_turns_only():
    gen = BaseConversationGenerator({})
    conv = _conv([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ])
    entries = gen.to_aiperf_multi_turn([conv])
    assert entries == [{"session_id": "sid", "turns": [{"text": "u1"}, {"text": "u2"}]}]


def test_to_aiperf_mooncake_context_and_delay():
    gen = BaseConversationGenerator({})
    conv = _conv([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a" * 40},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "b" * 8},
    ])
    entries = gen.to_aiperf_mooncake([conv])
    assert len(entries) == 2
    # first assistant turn: no delay, full context up to it (3 messages)
    assert len(entries[0]["messages"]) == 3
    assert "delay" not in entries[0]
    assert entries[0]["output_length"] == 10
    # second turn: delay=0, context grows, output_length = 8//4
    assert len(entries[1]["messages"]) == 5
    assert entries[1]["delay"] == 0
    assert entries[1]["output_length"] == 2


def test_default_jsonl_ensure_ascii_is_false():
    assert BaseConversationGenerator.jsonl_ensure_ascii is False
