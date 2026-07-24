"""Unit tests for the shared generator_base helpers."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator_base import (
    CHARS_PER_TOKEN,
    ACTIVE_TOKENIZER,
    BaseConversationGenerator,
    content_to_text,
    count_message_tokens,
    count_tokens,
    estimate_output_tokens,
    estimate_tokens,
)

_USING_TIKTOKEN = ACTIVE_TOKENIZER.startswith("tiktoken")


def test_count_tokens_matches_active_tokenizer():
    text = "Hello, world! This is a test of the tokenizer."
    if _USING_TIKTOKEN:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        assert count_tokens(text) == len(enc.encode(text))
    else:
        assert count_tokens(text) == len(text) // CHARS_PER_TOKEN


def test_count_tokens_scales_with_length():
    assert count_tokens("") == 0
    assert count_tokens("word " * 100) > count_tokens("word " * 10)


def test_estimate_tokens_delegates_to_count_tokens():
    assert estimate_tokens("some sample text") == count_tokens("some sample text")


def test_estimate_output_tokens_min_one():
    assert estimate_output_tokens("") == 1
    assert estimate_output_tokens("a") >= 1
    long_text = "the quick brown fox " * 20
    assert estimate_output_tokens(long_text) == max(1, count_tokens(long_text))


def test_count_tokens_accepts_non_str():
    # multimodal content can be a list; helper must stringify, not crash
    assert count_tokens(["x", "y"]) == count_tokens(str(["x", "y"]))


def test_content_to_text_flattens_multimodal():
    multimodal = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
    ]
    assert content_to_text(multimodal) == "describe this"
    assert content_to_text("plain string") == "plain string"


def test_count_message_tokens_sums_text_content():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello there"},
    ]
    expected = count_tokens("you are helpful") + count_tokens("hello there")
    assert count_message_tokens(messages) == expected


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
    assert entries[0]["output_length"] == estimate_output_tokens("a" * 40)
    # second turn: delay=0, context grows
    assert len(entries[1]["messages"]) == 5
    assert entries[1]["delay"] == 0
    assert entries[1]["output_length"] == estimate_output_tokens("b" * 8)


def test_default_jsonl_ensure_ascii_is_false():
    assert BaseConversationGenerator.jsonl_ensure_ascii is False
