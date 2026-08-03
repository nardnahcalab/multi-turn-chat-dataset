"""Tests specific to the isometric (controlled) text/random/reasoning datasets.

These assert that the three isometric generators share the same conversation
plan and therefore produce identical total word and turn counts, while still
generating style-specific message content.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import conftest  # noqa: E402

ISO_DATASETS = ["iso_text", "iso_random", "iso_reasoning"]
NUM = 12
SEED = 7


def _generate(dataset):
    gen, _ = conftest.make_generator(dataset, seed=SEED)
    return gen.generate_dataset(num_conversations=NUM)


def test_iso_equal_total_words_and_turns():
    """All three isometric datasets must have the same total words and turns."""
    totals = {}
    for ds in ISO_DATASETS:
        convs = _generate(ds)
        totals[ds] = {
            "conversations": len(convs),
            "turns": sum(c["num_turns"] for c in convs),
            "words": sum(c["total_words"] for c in convs),
        }

    first = totals[ISO_DATASETS[0]]
    assert first["conversations"] == NUM
    for ds in ISO_DATASETS[1:]:
        assert totals[ds]["turns"] == first["turns"]
        assert totals[ds]["words"] == first["words"]


def test_iso_shared_plan_per_conversation():
    """Per-conversation turn count and word budget schedules are identical."""
    by_id = {}
    for ds in ISO_DATASETS:
        convs = _generate(ds)
        by_id[ds] = {c["conversation_id"]: c for c in convs}

    ids = set(by_id[ISO_DATASETS[0]].keys())
    for ds in ISO_DATASETS[1:]:
        assert set(by_id[ds].keys()) == ids

    for cid in ids:
        ref = by_id[ISO_DATASETS[0]][cid]
        for ds in ISO_DATASETS[1:]:
            cur = by_id[ds][cid]
            assert cur["num_turns"] == ref["num_turns"]
            assert cur["total_words"] == ref["total_words"]


def test_iso_content_differs_across_styles():
    """Message text should differ between coherent, gibberish, and reasoning."""
    convs = {ds: _generate(ds) for ds in ISO_DATASETS}

    # Compare at least one message from the first conversation of each style.
    samples = []
    for ds in ISO_DATASETS:
        messages = conftest.json.loads(convs[ds][0]["messages"])
        # messages[0] is the system prompt (identical); compare a user turn.
        samples.append(messages[1]["content"])

    assert len(set(samples)) == len(samples), "styles should produce distinct user content"
