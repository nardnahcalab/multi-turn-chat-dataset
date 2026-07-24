"""Integration tests exercised against every dataset generator.

These assert the invariants that the datasets are supposed to guarantee:
reproducibility from the seed, a consistent core schema, valid message JSON,
the chars/4 token estimate, and well-formed aiperf output.
"""

import json

import pytest

from conftest import DATASETS, make_generator

CORE_COLUMNS = {
    "conversation_id", "num_turns", "num_messages", "system_prompt",
    "messages", "total_characters", "estimated_tokens", "cumulative_char_lengths",
}
# Each generator labels conversations with exactly one of these.
LABEL_COLUMNS = {"topic", "conversation_type", "task_type"}

NUM = 12
SEED = 7


@pytest.fixture(scope="module", params=DATASETS)
def dataset_sample(request):
    gen, config = make_generator(request.param, seed=SEED)
    conversations = gen.generate_dataset(num_conversations=NUM)
    return request.param, gen, conversations


def test_generates_requested_count(dataset_sample):
    _, _, conversations = dataset_sample
    assert len(conversations) == NUM


def test_reproducible_from_seed(dataset_sample):
    name, _, conversations = dataset_sample
    gen2, _ = make_generator(name, seed=SEED)
    again = gen2.generate_dataset(num_conversations=NUM)
    assert [c["conversation_id"] for c in conversations] == [c["conversation_id"] for c in again]
    assert [c["messages"] for c in conversations] == [c["messages"] for c in again]


def test_core_schema_present(dataset_sample):
    _, _, conversations = dataset_sample
    for conv in conversations:
        assert CORE_COLUMNS.issubset(conv.keys())
        assert LABEL_COLUMNS & conv.keys(), "missing topic/conversation_type/task_type"


def test_messages_valid_json_starting_with_system(dataset_sample):
    _, _, conversations = dataset_sample
    for conv in conversations:
        messages = json.loads(conv["messages"])
        assert isinstance(messages, list) and messages
        assert messages[0]["role"] == "system"
        assert conv["num_messages"] == len(messages)


def test_token_estimate_matches_char_heuristic(dataset_sample):
    _, _, conversations = dataset_sample
    for conv in conversations:
        assert conv["estimated_tokens"] == conv["total_characters"] // 4


def test_cumulative_char_lengths_are_monotonic(dataset_sample):
    _, _, conversations = dataset_sample
    for conv in conversations:
        cumulative = json.loads(conv["cumulative_char_lengths"])
        assert cumulative == sorted(cumulative)
        assert cumulative[-1] <= conv["total_characters"]


def test_aiperf_multi_turn_shape(dataset_sample):
    _, gen, conversations = dataset_sample
    entries = gen.to_aiperf_multi_turn(conversations)
    assert entries
    for entry in entries:
        assert set(entry) == {"session_id", "turns"}
        assert entry["turns"]
        assert all("text" in t for t in entry["turns"])


def test_aiperf_mooncake_shape(dataset_sample):
    _, gen, conversations = dataset_sample
    entries = gen.to_aiperf_mooncake(conversations)
    assert entries
    for entry in entries:
        assert {"session_id", "messages", "output_length"}.issubset(entry)
        assert isinstance(entry["output_length"], int)
        assert entry["output_length"] >= 1
