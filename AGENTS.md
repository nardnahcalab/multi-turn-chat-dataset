# Repository guide

Synthetic multi-turn chat dataset generators for LLM inference benchmarking
(prefix-cache / KV-cache stress). Each `<type>/generate.py` emits Parquet plus
two aiperf JSONL formats (`multi_turn`, `mooncake_trace`).

## Layout
- `generator_base.py` — `BaseConversationGenerator`: shared scaffolding (RNG/topic
  setup, turn sampling, aiperf writers, token estimation via `CHARS_PER_TOKEN`,
  and `main()`/`write_outputs`). The 6 class-based generators (text, reasoning,
  random, repeat, pdf, image) subclass it. `agentic` and `mixed` are structural
  outliers that reuse the shared helpers.
- `dataset_profile.py` — tagging, descriptive naming, manifests/profiling, payload scoring.
- `<type>/generate.py` + `<type>/config.yaml` — per-dataset generators.
- `tests/` — pytest suite (base helpers + per-generator invariants).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime deps + pytest
```

## Generating data
```bash
python text/generate.py                     # default: config distribution, seed 42
python text/generate.py --num 100 --seed 7  # override count/seed
python text/generate.py --format aiperf      # parquet | aiperf | mooncake | all
python pdf/generate.py --skip-fetch          # pdf/image: reuse cached sources (no network)
```
Output is deterministic for a given seed. The bulk `.parquet`/`.jsonl` are
gitignored (regenerate on demand); only `*_manifest.json` and the source caches
(`pdf/data/arxiv_papers.json`, `image/data/wikipedia_images.json`) are tracked.

## Testing / verification
```bash
pytest tests/ -q          # full suite
pytest tests/ -k reproducible
```
CI (`.github/workflows/ci.yml`) runs the suite on Python 3.11/3.12 plus offline
generator smoke tests.

## Conventions
- Token counts use the `chars // 4` heuristic (`CHARS_PER_TOKEN` in
  `generator_base.py`) — a deliberate approximation, not a real tokenizer.
- Conversation IDs are seed-reproducible (`BaseConversationGenerator.new_conversation_id`).
- When adding a dataset type, subclass `BaseConversationGenerator`, set
  `dataset_type`, implement `generate_conversation`, and delegate `main()`.
