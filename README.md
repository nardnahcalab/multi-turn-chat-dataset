# multi-turn-chat-dataset

Synthetic multi-turn conversation datasets for benchmarking LLM inference engines. Designed to simulate real-world conversations with naturally growing context, ideal for stress-testing **prefix caching**, **KV-cache management**, and **long-context inference** performance.

## Dataset Types

| Type | Status | Description |
|------|--------|-------------|
| **text/** | Available | Pure text multi-turn conversations |
| **image/** | Planned | Multi-turn with inline image references |
| **pdf/** | Planned | Multi-turn with PDF document context |

## Text Dataset

### Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate with defaults (500 conversations, seed=42)
python text/generate.py

# Custom generation
python text/generate.py --num 1000 --seed 123
python text/generate.py --config text/config.yaml --output my_dataset.parquet
```

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | string | UUID v4 identifier |
| `topic` | string | Domain category (e.g., `coding_help`, `customer_support`) |
| `num_turns` | int | Number of user-assistant exchange pairs |
| `num_messages` | int | Total messages including system prompt |
| `system_prompt` | string | System-level instruction for the conversation |
| `messages` | string (JSON) | Full message array: `[{"role": "system\|user\|assistant", "content": "..."}]` |
| `total_characters` | int | Character count of entire conversation |
| `estimated_tokens` | int | Approximate token count (~chars/4) |
| `cumulative_char_lengths` | string (JSON) | Array of cumulative character counts after each turn |

### Topics

Conversations are generated across 8 domains with configurable weights:

- **coding_help** (20%) — debugging, code review, architecture, DevOps
- **customer_support** (15%) — billing, returns, shipping, account issues
- **tutoring** (15%) — math, science, history, test prep
- **creative_writing** (10%) — stories, poetry, worldbuilding
- **travel_planning** (10%) — itineraries, budgets, logistics
- **data_analysis** (10%) — SQL, pandas, statistics, visualization
- **business_strategy** (10%) — market analysis, growth, pricing
- **health_fitness** (10%) — workouts, nutrition, training plans

### Turn Distribution

Conversations are distributed across four length buckets to cover different prefix cache scenarios:

| Bucket | Turns | Count | Use Case |
|--------|-------|-------|----------|
| Short | 1-5 | 100 | Cache cold start, minimal reuse |
| Medium | 6-15 | 150 | Moderate prefix sharing |
| Long | 16-30 | 150 | Significant prefix cache benefit |
| Very Long | 31-50 | 100 | Stress-test cache eviction and long context |

### Default Dataset Stats

- **500 conversations**, **~3.9M estimated tokens**
- Turn range: **1-50**, mean: **~19**
- Max single conversation: **~25K tokens**
- File size: **~2 MB** (Parquet)

### Loading the Dataset

```python
import pandas as pd
import json

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Access a conversation
row = df.iloc[0]
messages = json.loads(row["messages"])
for msg in messages:
    print(f"[{msg['role']}] {msg['content'][:100]}...")

# Analyze context growth
lengths = json.loads(row["cumulative_char_lengths"])
print(f"Context growth: {lengths[0]:,} -> {lengths[-1]:,} chars over {row['num_turns']} turns")
```

### Benchmarking Prefix Cache

Each conversation simulates how real inference works: the full conversation history is sent with each new turn, so the prefix grows monotonically. To benchmark:

```python
# Simulate incremental inference requests
messages = json.loads(row["messages"])
for i in range(2, len(messages), 2):  # step through user turns
    prefix = messages[:i]      # everything up to current user message
    new_turn = messages[i:i+2] # current user + expected assistant response
    # Send prefix + new_turn[0] to your inference engine
    # Measure: time-to-first-token, tokens/sec, cache hit rate
```

### Configuration

Edit `text/config.yaml` to customize:
- Number of conversations and distribution across turn-count buckets
- Topic weights and system prompts
- Response length distributions (short/medium/long) by conversation phase
- Random seed for reproducibility

## Project Structure

```
multi-turn-chat-dataset/
├── README.md
├── LICENSE
├── requirements.txt
├── text/
│   ├── generate.py          # Generation script
│   ├── config.yaml          # Configuration
│   └── data/
│       └── multi_turn_text_chat.parquet
├── image/                   # (planned)
└── pdf/                     # (planned)
```

## License

See [LICENSE](LICENSE) for details.
