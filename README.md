# multi-turn-chat-dataset

Synthetic multi-turn conversation datasets for benchmarking LLM inference engines. Designed to simulate real-world conversations with naturally growing context, ideal for stress-testing **prefix caching**, **KV-cache management**, and **long-context inference** performance.

## Dataset Types

| Type | Status | Description |
|------|--------|-------------|
| **text/** | Available | Pure text multi-turn conversations |
| **image/** | Available | Multi-turn with Wikipedia image references |
| **pdf/** | Available | Multi-turn with arXiv PDF document context |

## Text Dataset

### Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate all formats (Parquet + aiperf JSONL + mooncake JSONL)
python text/generate.py

# Generate only specific format
python text/generate.py --format parquet
python text/generate.py --format aiperf      # multi_turn JSONL only
python text/generate.py --format mooncake    # mooncake_trace JSONL only

# Custom generation
python text/generate.py --num 1000 --seed 123
```

### Output Formats

The generator produces three output files:

| File | Format | Size | aiperf `--custom-dataset-type` |
|------|--------|------|-------------------------------|
| `multi_turn_text_chat.parquet` | Parquet | ~2 MB | N/A (analysis/HuggingFace) |
| `multi_turn_text_chat.jsonl` | JSONL | ~0.8 MB | `multi_turn` |
| `multi_turn_text_chat_mooncake.jsonl` | JSONL | ~227 MB | `mooncake_trace` |

### Using with aiperf

The JSONL files are designed for direct use with [NVIDIA aiperf](https://github.com/ai-dynamo/aiperf).

**Option 1: `multi_turn` format** (recommended — lightweight, user messages only)

aiperf sends user messages and accumulates live server responses into conversation history automatically. Each turn includes the growing conversation prefix, exercising prefix caching.

```bash
aiperf profile \
    --model <your-model> \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --input-file text/data/multi_turn_text_chat.jsonl \
    --custom-dataset-type multi_turn \
    --concurrency 10
```

JSONL schema (one line per conversation):
```json
{"session_id": "uuid", "turns": [{"text": "user msg 1"}, {"text": "user msg 2"}, ...]}
```

**Option 2: `mooncake_trace` format** (full control — pre-canned responses included)

Each line is a single turn with the complete message array up to that point. Gives full control over the exact prompt (including system prompt and assistant responses) sent at each turn.

```bash
aiperf profile \
    --model <your-model> \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --input-file text/data/multi_turn_text_chat_mooncake.jsonl \
    --custom-dataset-type mooncake_trace \
    --concurrency 10
```

JSONL schema (one line per turn):
```json
{"session_id": "uuid", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "output_length": 150}
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

## PDF Dataset

Multi-turn conversations about arXiv research papers. Each conversation references a real PDF via URL, simulating document analysis workflows where context grows as users ask follow-up questions.

### Quick Start

```bash
# Setup (same venv as text dataset)
source .venv/bin/activate

# Generate all formats (fetches 100 arXiv papers, generates 500 conversations)
python pdf/generate.py

# Reuse cached papers (skip arXiv API calls)
python pdf/generate.py --skip-fetch

# Generate only aiperf JSONL
python pdf/generate.py --skip-fetch --format aiperf
```

### How It Works

1. **Fetches** metadata for ~100 recent AI/ML papers from arXiv (categories: cs.AI, cs.CL, cs.CV, cs.LG)
2. **Caches** paper metadata locally (`pdf/data/arxiv_papers.json`)
3. **Generates** multi-turn conversations where the first message references the PDF URL
4. The PDF URL is included in the first user message using multimodal content format
5. Subsequent turns are text-only follow-ups about the paper

### Conversation Types

| Type | Weight | Description |
|------|--------|-------------|
| paper_summary | 20% | Summarize and explain the paper |
| methodology_deep_dive | 20% | Deep dive into methods and architecture |
| results_analysis | 15% | Analyze experimental results and tables |
| critical_review | 15% | Critique strengths, weaknesses, limitations |
| comparison | 10% | Compare with related work |
| implementation | 10% | Discuss implementation and reproducibility |
| brainstorm_extensions | 10% | Brainstorm improvements and future work |

### Output Formats

| File | Format | Size | aiperf `--custom-dataset-type` |
|------|--------|------|-------------------------------|
| `multi_turn_pdf_chat.parquet` | Parquet | ~1.6 MB | N/A |
| `multi_turn_pdf_chat.jsonl` | JSONL | ~0.5 MB | `multi_turn` |
| `multi_turn_pdf_chat_mooncake.jsonl` | JSONL | ~97 MB | `mooncake_trace` |

### Using with aiperf

```bash
# multi_turn format (lightweight, PDF URL in first turn text)
aiperf profile \
    --model <your-model> \
    --endpoint-type chat \
    --input-file pdf/data/multi_turn_pdf_chat.jsonl \
    --custom-dataset-type multi_turn \
    --streaming --url localhost:8000 --concurrency 10

# mooncake_trace format (full message arrays with multimodal content)
aiperf profile \
    --model <your-model> \
    --endpoint-type chat \
    --input-file pdf/data/multi_turn_pdf_chat_mooncake.jsonl \
    --custom-dataset-type mooncake_trace \
    --streaming --url localhost:8000 --concurrency 10
```

### Schema (Parquet)

Extends the text dataset schema with PDF-specific columns:

| Column | Type | Description |
|--------|------|-------------|
| `conversation_type` | string | Type of analysis (e.g., `paper_summary`, `critical_review`) |
| `paper_arxiv_id` | string | arXiv paper ID |
| `paper_title` | string | Paper title |
| `paper_pdf_url` | string | Direct PDF URL (e.g., `https://arxiv.org/pdf/2401.12345v1`) |
| `paper_categories` | string (JSON) | arXiv categories |
| `messages` | string (JSON) | First user message uses multimodal format with file reference |

### Default Dataset Stats

- **500 conversations**, **~3M estimated tokens**
- Turn range: **1-30**, mean: **~10**
- **89 unique arXiv papers** across cs.AI, cs.CL, cs.CV, cs.LG
- **7 conversation types** with configurable weights

## Image Dataset

Multi-turn conversations about Wikipedia images across 10 diverse topics. Each conversation references a real image via URL, simulating vision-language model workflows where users ask follow-up questions about visual content.

### Quick Start

```bash
# Setup (same venv as text dataset)
source .venv/bin/activate

# Generate all formats (fetches images from 100 Wikipedia articles, generates 500 conversations)
python image/generate.py

# Reuse cached images (skip Wikipedia API calls)
python image/generate.py --skip-fetch

# Generate only aiperf JSONL
python image/generate.py --skip-fetch --format aiperf
```

### How It Works

1. **Fetches** images from 100 Wikipedia articles across 10 topic categories
2. **Balances** images evenly across topics (15 per topic from the 150 collected)
3. **Caches** image metadata locally (`image/data/wikipedia_images.json`)
4. **Generates** multi-turn conversations where the first message includes the image URL in `image_url` multimodal format
5. Subsequent turns are text-only follow-ups about the image

### Topics

Images are sourced from Wikipedia articles spanning 10 categories:

| Topic | Example Articles |
|-------|-----------------|
| nature | Grand Canyon, Great Barrier Reef, Aurora borealis |
| architecture | Eiffel Tower, Colosseum, Taj Mahal, Sagrada Familia |
| art | Mona Lisa, Starry Night, The Great Wave off Kanagawa |
| science | DNA, Solar System, Large Hadron Collider |
| history | Ancient Egypt, Apollo 11, Pompeii, Stonehenge |
| wildlife | African elephant, Bengal tiger, Emperor penguin |
| geography | Mount Kilimanjaro, Victoria Falls, Ha Long Bay |
| food | Sushi, Pizza, Dim sum, Coffee |
| technology | Microprocessor, Space Shuttle, Robotics |
| culture | Carnival, Tea ceremony, Flamenco, Holi |

### Conversation Types

| Type | Weight | Description |
|------|--------|-------------|
| image_description | 20% | Describe and explain what is shown |
| visual_analysis | 20% | Analyze composition, colors, lighting |
| contextual_discussion | 15% | Historical, cultural, or scientific context |
| creative_interpretation | 15% | Artistic, narrative, or emotional interpretation |
| comparison | 10% | Compare with similar subjects or styles |
| educational | 10% | Use the image as a learning starting point |
| technical_photography | 10% | Discuss photographic techniques |

### Output Formats

| File | Format | Size | aiperf `--custom-dataset-type` |
|------|--------|------|-------------------------------|
| `multi_turn_image_chat.parquet` | Parquet | ~1.6 MB | N/A |
| `multi_turn_image_chat.jsonl` | JSONL | ~0.5 MB | `multi_turn` |
| `multi_turn_image_chat_mooncake.jsonl` | JSONL | ~99 MB | `mooncake_trace` |

### Using with aiperf

```bash
# multi_turn format (lightweight, image URL in first turn text)
aiperf profile \
    --model <your-model> \
    --endpoint-type chat \
    --input-file image/data/multi_turn_image_chat.jsonl \
    --custom-dataset-type multi_turn \
    --streaming --url localhost:8000 --concurrency 10

# mooncake_trace format (full message arrays with multimodal content)
aiperf profile \
    --model <your-model> \
    --endpoint-type chat \
    --input-file image/data/multi_turn_image_chat_mooncake.jsonl \
    --custom-dataset-type mooncake_trace \
    --streaming --url localhost:8000 --concurrency 10
```

### Schema (Parquet)

Extends the text dataset schema with image-specific columns:

| Column | Type | Description |
|--------|------|-------------|
| `conversation_type` | string | Type of analysis (e.g., `image_description`, `visual_analysis`) |
| `image_title` | string | Wikipedia image title |
| `image_url` | string | Direct image URL (Wikimedia Commons) |
| `image_topic` | string | Topic category (e.g., `nature`, `architecture`) |
| `source_article` | string | Wikipedia article the image was sourced from |
| `image_width` | int | Image width in pixels |
| `image_height` | int | Image height in pixels |
| `messages` | string (JSON) | First user message uses `image_url` multimodal format |

### Default Dataset Stats

- **500 conversations**, **~3M estimated tokens**
- Turn range: **1-30**, mean: **~10**
- **141 unique images** across **10 topic categories**
- **7 conversation types** with configurable weights

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
│       ├── multi_turn_text_chat.parquet        # Full dataset (Parquet)
│       ├── multi_turn_text_chat.jsonl           # aiperf multi_turn format
│       └── multi_turn_text_chat_mooncake.jsonl  # aiperf mooncake_trace (generated, not in git)
├── pdf/
│   ├── generate.py          # Generation script (fetches arXiv papers)
│   ├── config.yaml          # Configuration
│   └── data/
│       ├── arxiv_papers.json                   # Cached paper metadata
│       ├── multi_turn_pdf_chat.parquet         # Full dataset (Parquet)
│       ├── multi_turn_pdf_chat.jsonl           # aiperf multi_turn format
│       └── multi_turn_pdf_chat_mooncake.jsonl  # aiperf mooncake_trace (generated, not in git)
├── image/
│   ├── generate.py          # Generation script (fetches Wikipedia images)
│   ├── config.yaml          # Configuration
│   └── data/
│       ├── wikipedia_images.json                  # Cached image metadata
│       ├── multi_turn_image_chat.parquet           # Full dataset (Parquet)
│       ├── multi_turn_image_chat.jsonl             # aiperf multi_turn format
│       └── multi_turn_image_chat_mooncake.jsonl    # aiperf mooncake_trace (generated, not in git)
└── .gitignore
```

## License

See [LICENSE](LICENSE) for details.
