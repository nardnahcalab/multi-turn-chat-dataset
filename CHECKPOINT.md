# Checkpoint: Random & Repeat Dataset Generators

**Date**: April 27, 2026  
**Status**: Complete - pushed to main  
**Branch**: main  
**Last Commit**: `da45591` - feat: Add random and repeat dataset generators

## What Was Accomplished

Added two new dataset categories (`random/` and `repeat/`) following the same patterns as the existing `text/` dataset, plus updated README and example scripts.

## Session Summary

### New Dataset: `random/`
- **Purpose**: Random/gibberish text for benchmarking inference with unpredictable, non-cacheable content (baseline comparison against structured datasets)
- **5 topic types**: `random_words`, `random_chars`, `random_sentences`, `random_mixed`, `random_lorem`
- **Stats**: 500 conversations, ~5M estimated tokens, 1-50 turns (mean ~19)
- **Output sizes**: Parquet ~7 MB, aiperf JSONL ~1.7 MB, mooncake JSONL ~290 MB
- **Files**: `random/config.yaml` (82 lines), `random/generate.py` (662 lines)

### New Dataset: `repeat/`
- **Purpose**: Highly repetitive text for testing tokenizer efficiency and KV-cache behavior with maximally compressible content
- **5 topic types**: `single_word`, `phrase_repeat`, `counting_repeat`, `letter_repeat`, `sentence_repeat`
- **Key design**: Same base word/phrase persists across all turns within a conversation; repetition count varies per turn
- **Stats**: 500 conversations, ~4.8M estimated tokens, 1-50 turns (mean ~19)
- **Output sizes**: Parquet ~2.2 MB, aiperf JSONL ~3 MB, mooncake JSONL ~269 MB
- **Files**: `repeat/config.yaml` (82 lines), `repeat/generate.py` (640 lines)

### Updated Files
- **README.md**: Added `random/` and `repeat/` to dataset types table, full documentation sections (Quick Start, How It Works, Topics, Output Formats, aiperf usage, Schema, Turn Distribution, Stats, Configuration), updated Project Structure tree
- **examples/01_load_and_inspect.py**: Added `random` and `repeat` to `PARQUET_FILES` dict
- **examples/02_analyze_conversations.py**: Added `random` and `repeat` to `PARQUET_FILES` dict
- **examples/03_prepare_for_benchmarking.py**: Added `random` and `repeat` to `PARQUET_FILES` dict

## Generator Interface (both identical to text/)

```bash
python random/generate.py                    # default: 500 convos, seed 42
python random/generate.py --num 1000         # override count
python random/generate.py --seed 123         # override seed
python random/generate.py --format parquet   # parquet only
python random/generate.py --format aiperf    # multi_turn JSONL only
python random/generate.py --format mooncake  # mooncake_trace JSONL only

python repeat/generate.py                    # same interface
python repeat/generate.py --num 1000 --seed 123
```

## Output Schema (both identical to text/)

9 columns: `conversation_id`, `topic`, `num_turns`, `num_messages`, `system_prompt`, `messages`, `total_characters`, `estimated_tokens`, `cumulative_char_lengths`

## Verification

Both generators tested with `--num 10` (quick) and full 500-conversation default runs:
- Schema matches text/ exactly (verified programmatically)
- All 3 output formats produced correctly (Parquet, aiperf JSONL, mooncake JSONL)
- JSONL format validated: correct keys (`session_id`, `turns`/`messages`, `output_length`)
- 500 lines in aiperf JSONL (one per conversation)

## Commit History (this session)

```
da45591 feat: Add random and repeat dataset generators (12 files, 2,708 insertions)
```

## Previous Session Commits

```
17a18a9 docs: Update checkpoint for documentation session
50c4c2e docs: Add comprehensive examples and API reference documentation
```

## Repository State

```
multi-turn-chat-dataset/
├── README.md
├── EXAMPLES.md
├── API_REFERENCE.md
├── FAQ.md
├── ARCHITECTURE.md
├── AGENTIC_SUMMARY.md
├── CHECKPOINT.md                # This file
├── LICENSE
├── requirements.txt
├── examples/
│   ├── README.md
│   ├── 01_load_and_inspect.py
│   ├── 02_analyze_conversations.py
│   ├── 03_prepare_for_benchmarking.py
│   └── 04_agentic_analysis.py
├── text/                        # Original text dataset (500 convos, ~3.9M tokens)
├── pdf/                         # arXiv PDF dataset (500 convos, ~3M tokens)
├── image/                       # Wikipedia image dataset (500 convos, ~3M tokens)
├── reasoning/                   # Deep reasoning dataset (500 convos, ~5.8M tokens)
├── agentic/                     # Agent task dataset (500 convos, ~2.5M tokens)
├── random/                      # NEW - Random text dataset (500 convos, ~5M tokens)
│   ├── generate.py
│   ├── config.yaml
│   └── data/
│       ├── multi_turn_random_chat.parquet
│       ├── multi_turn_random_chat.jsonl
│       └── multi_turn_random_chat_mooncake.jsonl  (gitignored)
└── repeat/                      # NEW - Repetitive text dataset (500 convos, ~4.8M tokens)
    ├── generate.py
    ├── config.yaml
    └── data/
        ├── multi_turn_repeat_chat.parquet
        ├── multi_turn_repeat_chat.jsonl
        └── multi_turn_repeat_chat_mooncake.jsonl  (gitignored)
```

## How to Continue

```bash
cd /mnt/c/Users/rajenb1/apps/src/multi-turn-chat-dataset
source .venv/bin/activate

# Regenerate datasets
python random/generate.py
python repeat/generate.py

# Inspect with example scripts
python examples/01_load_and_inspect.py random
python examples/01_load_and_inspect.py repeat
python examples/02_analyze_conversations.py random
```

## Potential Next Steps

1. Add more dataset categories (e.g., code-only, multilingual)
2. Add Jupyter notebook examples
3. Add visualization examples (matplotlib/plotly charts)
4. Create a unified generation script that builds all datasets
5. Add a performance benchmarking guide comparing dataset types
6. Add a CONTRIBUTING.md guide

---

**Checkpoint created by**: Devin AI  
**Project**: multi-turn-chat-dataset  
**Total dataset types**: 7 (text, pdf, image, reasoning, agentic, random, repeat)  
**Status**: Clean, pushed to origin/main
