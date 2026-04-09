# Architecture

Comprehensive technical reference for the **multi-turn-chat-dataset** repository. This document covers the system design, data flows, code structure, and output format specifications for all three dataset generators.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Design Principles](#design-principles)
- [Shared Architecture](#shared-architecture)
- [Text Dataset Generator](#text-dataset-generator)
- [PDF Dataset Generator](#pdf-dataset-generator)
- [Image Dataset Generator](#image-dataset-generator)
- [Output Format Specifications](#output-format-specifications)
- [Configuration System](#configuration-system)
- [Extensibility Guide](#extensibility-guide)

---

## Overview

This project generates synthetic multi-turn conversation datasets for benchmarking LLM inference engines. The datasets are designed to simulate real-world conversations with naturally growing context, making them ideal for stress-testing **prefix caching**, **KV-cache management**, and **long-context inference** performance.

Three dataset types target different modalities:

| Dataset | Modality | External Source | Conversations | Tokens | Turn Range |
|---------|----------|-----------------|---------------|--------|------------|
| **text** | Text-only | None (synthetic) | 500 | ~3.9M | 1-50 |
| **pdf** | Text + PDF URL | arXiv papers | 500 | ~3M | 1-30 |
| **image** | Text + Image URL | Wikipedia images | 500 | ~3M | 1-30 |

---

## Repository Structure

```
multi-turn-chat-dataset/
├── README.md                 # User-facing documentation
├── ARCHITECTURE.md           # This file
├── LICENSE                   # Apache License 2.0
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes .venv/, __pycache__/, *_mooncake.jsonl
│
├── text/
│   ├── generate.py           # 1,017 lines — text conversation generator
│   ├── config.yaml           # 97 lines — configuration
│   └── data/
│       ├── multi_turn_text_chat.parquet        # ~2 MB
│       ├── multi_turn_text_chat.jsonl           # ~0.8 MB (aiperf multi_turn)
│       └── multi_turn_text_chat_mooncake.jsonl  # ~227 MB (gitignored)
│
├── pdf/
│   ├── generate.py           # 821 lines — PDF/arXiv conversation generator
│   ├── config.yaml           # 94 lines — configuration
│   └── data/
│       ├── arxiv_papers.json                   # Cached paper metadata (89 papers)
│       ├── multi_turn_pdf_chat.parquet         # ~1.6 MB
│       ├── multi_turn_pdf_chat.jsonl           # ~0.5 MB (aiperf multi_turn)
│       └── multi_turn_pdf_chat_mooncake.jsonl  # ~97 MB (gitignored)
│
└── image/
    ├── generate.py           # 1,059 lines — Wikipedia image conversation generator
    ├── config.yaml           # 210 lines — configuration
    └── data/
        ├── wikipedia_images.json                  # Cached image metadata (150 images)
        ├── multi_turn_image_chat.parquet           # ~1.6 MB
        ├── multi_turn_image_chat.jsonl             # ~0.5 MB (aiperf multi_turn)
        └── multi_turn_image_chat_mooncake.jsonl    # ~99 MB (gitignored)
```

### Dependencies

```
pandas>=2.0       # DataFrame operations, Parquet I/O
pyarrow>=14.0     # Parquet file format engine
pyyaml>=6.0       # YAML configuration parsing
arxiv>=2.0        # arXiv API client (pdf/ only)
requests          # Wikipedia API (image/ only, stdlib-adjacent)
```

---

## Design Principles

### 1. Template-Based Synthetic Generation

All conversations are generated from hand-crafted templates with randomized placeholder fills. This approach provides:

- **Deterministic reproducibility** via seeded RNG (`random.Random(seed)`)
- **Fast generation** with no LLM API calls required
- **Controllable diversity** through configurable template pools and fill value lists
- **Consistent quality** without model-dependent variation

### 2. Growing Context for Prefix Cache Testing

Each conversation simulates real inference: the full message history is sent with every new turn, so the prefix grows monotonically. This is the core property that makes the datasets useful for benchmarking:

```
Turn 1: [system, user1]                          → generate assistant1
Turn 2: [system, user1, assistant1, user2]        → generate assistant2
Turn 3: [system, user1, assistant1, user2, assistant2, user3] → generate assistant3
```

The `cumulative_char_lengths` column tracks this growth per turn.

### 3. Multimodal First Message, Text-Only Follow-ups

For PDF and image datasets, only the first user message includes the multimodal reference (PDF URL or image URL). Subsequent turns are text-only follow-ups. This mirrors real-world usage patterns where:

- A document/image is uploaded once at the start
- Follow-up questions reference it implicitly via conversation context
- The multimodal content remains in the prefix cache across turns

### 4. Three Output Formats

Every generator produces three complementary formats:

| Format | File | Use Case |
|--------|------|----------|
| **Parquet** | `*.parquet` | Data analysis, statistics, HuggingFace upload |
| **aiperf multi_turn** | `*.jsonl` | Lightweight benchmarking (user turns only) |
| **aiperf mooncake_trace** | `*_mooncake.jsonl` | Full-control benchmarking (complete message arrays) |

### 5. External Data Caching

PDF and image generators fetch metadata from external APIs (arXiv, Wikipedia) and cache results locally as JSON files. The `--skip-fetch` flag enables offline regeneration from cached data.

---

## Shared Architecture

All three generators follow an identical architectural pattern. Understanding one makes the others immediately familiar.

### Generation Pipeline

```
config.yaml
    │
    ▼
┌─────────────────────────┐
│   Load Configuration    │  YAML → dict
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Fetch External Data    │  arXiv API / Wikipedia API (pdf, image only)
│  (or load from cache)   │  → JSON cache file
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Initialize Generator   │  ConversationGenerator(config, data, seed)
│  with seeded RNG        │  Isolated random.Random(seed) instance
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Generate Conversations │  For each of 500 conversations:
│  via distribution       │    1. Pick type/topic (weighted)
│  buckets                │    2. Pick turn count (from bucket)
│                         │    3. Generate messages via templates
│                         │    4. Track cumulative char lengths
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Export to 3 Formats    │  Parquet (full metadata)
│                         │  multi_turn JSONL (user turns only)
│                         │  mooncake_trace JSONL (full context per turn)
└─────────────────────────┘
```

### Conversation Generator Class

Each generator has a central class with the same method structure:

| Method | Purpose |
|--------|---------|
| `__init__(config, [data], seed)` | Initialize RNG, extract weights, store config |
| `_fill_template(template, context)` | Multi-phase placeholder substitution |
| `_response_length_bucket(turn_index)` | Pick short/medium/long based on turn position |
| `_generate_user_message(type, context, turn)` | Select opener (turn 0) or followup template |
| `_generate_response(type, context, turn)` | Generate response, pad/trim to target length |
| `generate_conversation(num_turns)` | Build one full conversation with metadata |
| `generate_dataset(num_conversations)` | Generate all conversations using distribution |

### Template System

Every generator uses a three-tier template structure:

```python
TEMPLATES = {
    "conversation_type": {
        "openers": [...],     # First user message (5+ templates per type)
        "followups": [...],   # Subsequent user messages (10+ templates per type)
        "responses": [...]    # Assistant responses (2-5 templates per type)
    }
}
```

Templates contain `{placeholder}` tokens that are filled from `FILL_VALUES` dictionaries:

```python
FILL_VALUES = {
    "placeholder_key": ["value1", "value2", "value3", ...],
    ...
}
```

**Filling algorithm** (3 phases):
1. **Context-specific fills** — Replace placeholders with data from the source (paper title, image URL, etc.)
2. **Generic fills** — Replace remaining placeholders from `FILL_VALUES` using `rng.choice()`
3. **Cleanup** — Remove any unfilled placeholders via `re.sub(r'\{[a-zA-Z_0-9]+\}', '', result)`

### Response Length Control

Responses grow longer as conversations progress, simulating natural deepening engagement:

| Turn Position | Short | Medium | Long |
|--------------|-------|--------|------|
| Early (0-2/0-4) | 20-30% | 50% | 20-30% |
| Middle (3-14/5-19) | 15-20% | 40% | 40-45% |
| Late (15+/20+) | 10% | 30% | 60% |

Word count ranges per bucket:

| Generator | Short | Medium | Long |
|-----------|-------|--------|------|
| text | 20-80 | 80-250 | 250-600 |
| pdf | 30-100 | 100-300 | 300-700 |
| image | 30-100 | 100-300 | 300-700 |

**Padding**: If a response is shorter than the target, extension sentences are appended in a loop until the target is met.

**Trimming**: If a response exceeds 130% of the target, it is truncated at the nearest sentence boundary (last `.` after 70% of the text).

### Turn Distribution

Conversations are allocated across length buckets to cover different prefix cache scenarios:

| Generator | Short | Medium | Long | Very Long | Total |
|-----------|-------|--------|------|-----------|-------|
| text | 100 (1-5) | 150 (6-15) | 150 (16-30) | 100 (31-50) | 500 |
| pdf | 100 (1-3) | 200 (4-10) | 150 (11-20) | 50 (21-30) | 500 |
| image | 100 (1-3) | 200 (4-10) | 150 (11-20) | 50 (21-30) | 500 |

### CLI Interface

All generators share the same CLI flags:

```
--config PATH       Config YAML (default: config.yaml in script directory)
--num N             Override conversation count
--seed N            Override random seed
--output PATH       Override output path
--format FORMAT     all | parquet | aiperf | mooncake
--skip-fetch        Reuse cached external data (pdf, image only)
```

---

## Text Dataset Generator

**File**: `text/generate.py` (1,017 lines)

### Architecture

The text generator is the simplest: no external data fetching, no multimodal content. All conversations are purely synthetic text.

```
text/config.yaml
    │
    ▼
ConversationGenerator(config, seed=42)
    │
    ├── TOPIC_TEMPLATES (8 topics × {openers, followups, responses, fill_values})
    ├── CODE_SNIPPETS (Python, TypeScript, Go, Rust, Java)
    ├── SQL_SNIPPETS (2 complex queries)
    ├── CREATIVE_EXCERPTS (5 literary passages)
    └── PROBLEM_STATEMENTS (6 math/science problems)
    │
    ▼
500 conversations → Parquet + multi_turn JSONL + mooncake JSONL
```

### Topics (8 domains)

| Topic | Weight | Openers | Followups | Responses | Unique Features |
|-------|--------|---------|-----------|-----------|-----------------|
| coding_help | 20% | 8 | 13 | 5 | Embedded code snippets in 5 languages |
| customer_support | 15% | 10 | 14 | 5 | Order IDs, case IDs, escalation flows |
| tutoring | 15% | 6 | 12 | 4 | Math/science problem statements |
| creative_writing | 10% | 6 | 10 | 4 | Literary excerpts, prose refinement |
| travel_planning | 10% | 5 | 12 | 2 | Itineraries, budget breakdowns |
| data_analysis | 10% | 6 | 10 | 4 | SQL snippets, pandas code |
| business_strategy | 10% | 5 | 10 | 2 | GTM frameworks, market analysis |
| health_fitness | 10% | 6 | 12 | 3 | Workout plans with sets/reps |

### Topic-Specific System Prompts

Unlike PDF/image (which use a single system prompt), the text generator assigns **per-topic system prompts** from config:

```yaml
topics:
  - name: "coding_help"
    system_prompt: "You are an expert software engineer assistant..."
  - name: "customer_support"
    system_prompt: "You are a helpful customer support agent..."
```

### Unique Feature: Embedded Domain Content

The text generator includes rich domain-specific content not present in the other generators:

- **CODE_SNIPPETS**: 10 realistic code examples across Python (4), TypeScript (2), Go (2), Rust (1), Java (1)
- **SQL_SNIPPETS**: 2 complex queries (lifetime value analysis, cohort analysis with CTEs)
- **CREATIVE_EXCERPTS**: 5 literary opening passages for creative writing conversations
- **PROBLEM_STATEMENTS**: 6 academic problems (algorithms, calculus, number theory, linear algebra, physics)

### Template Fill System (Extended)

The text generator has the most complex `_fill_template()` method (~280 lines) because it handles:

1. **Topic-specific fill_values** from `TOPIC_TEMPLATES[topic]["fill_values"]`
2. **Special placeholders**: `{order_id}` → random 6-digit number, `{case_id}` → "CS-XXXXX", `{email}` → "userXXX@email.com"
3. **Code insertion**: `{code_snippet}` → language-detected snippet from CODE_SNIPPETS
4. **SQL insertion**: `{sql_code}` → random SQL_SNIPPETS entry
5. **Creative writing**: `{excerpt}` → random CREATIVE_EXCERPTS entry
6. **Problem statements**: `{problem}` → random PROBLEM_STATEMENTS entry
7. **150+ generic fallback** values for catch-all placeholder resolution
8. **Iterative filling** (up to 20 passes) for nested/dependent placeholders

### Message Format

All messages are text-only:

```json
[
  {"role": "system", "content": "You are an expert software engineer..."},
  {"role": "user", "content": "I'm getting a TypeError in my Python code..."},
  {"role": "assistant", "content": "Let me help you debug that. The issue is..."}
]
```

### Code Structure Breakdown

| Section | Lines | % |
|---------|-------|---|
| Topic templates + embedded content | ~400 | 39% |
| ConversationGenerator class | ~414 | 41% |
| aiperf export functions | ~80 | 8% |
| CLI / main() | ~74 | 7% |
| Imports / boilerplate | ~49 | 5% |

---

## PDF Dataset Generator

**File**: `pdf/generate.py` (821 lines)

### Architecture

The PDF generator adds an external data layer (arXiv) and multimodal first messages.

```
pdf/config.yaml
    │
    ▼
fetch_arxiv_papers(config, cache_path)
    ├── Check cache (pdf/data/arxiv_papers.json)
    ├── Fetch from arXiv API (4 categories × 30 papers)
    ├── Deduplicate by arxiv_id
    ├── Rate limit: 3s delay between requests
    └── Cache to JSON
    │
    ▼
PDFConversationGenerator(config, papers, seed=42)
    │
    ├── CONVERSATION_TEMPLATES (7 types × {openers, followups, responses})
    └── FILL_VALUES (39 keys, AI/ML domain-specific)
    │
    ▼
500 conversations → Parquet + multi_turn JSONL + mooncake JSONL
```

### arXiv Paper Fetching

**Function**: `fetch_arxiv_papers(config, cache_path)`

| Step | Details |
|------|---------|
| Cache check | Load `arxiv_papers.json` if exists and has sufficient count |
| API client | `arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)` |
| Categories | cs.AI, cs.CL, cs.CV, cs.LG (configurable) |
| Per-category limit | 30 papers (configurable) |
| Sort order | Submitted date (descending) |
| Deduplication | `seen_ids` set prevents duplicates across categories |
| Rate limiting | 3s between API requests, 1s between category batches |

**Cached paper schema** (`arxiv_papers.json`):

```json
{
  "arxiv_id": "2604.07349v1",
  "title": "Paper Title",
  "authors": ["Author 1", "Author 2"],
  "abstract": "Full abstract...",
  "categories": ["cs.AI", "cs.CL"],
  "primary_category": "cs.AI",
  "published": "2026-04-08T17:59:47+00:00",
  "pdf_url": "https://arxiv.org/pdf/2604.07349v1",
  "entry_url": "http://arxiv.org/abs/2604.07349v1"
}
```

### Conversation Types (7)

| Type | Weight | Openers | Followups | Responses |
|------|--------|---------|-----------|-----------|
| paper_summary | 20% | 5 | 10 | 3 |
| methodology_deep_dive | 20% | 4 | 13 | 3 |
| results_analysis | 15% | 4 | 12 | 2 |
| critical_review | 15% | 4 | 11 | 2 |
| comparison | 10% | 4 | 9 | 2 |
| implementation | 10% | 4 | 10 | 2 |
| brainstorm_extensions | 10% | 4 | 10 | 2 |

**Total**: 29 openers, 75 followups, 16 responses

### Fill Values (AI/ML Domain)

39 keys with domain-specific values including:

- **Research fields**: NLP, computer vision, reinforcement learning, generative models, etc.
- **Techniques**: attention mechanisms, contrastive learning, knowledge distillation, etc.
- **Hardware**: A100 GPU, 8x H100 GPUs, TPU v4 pod, etc.
- **Domains**: healthcare, finance, autonomous driving, robotics, etc.
- **Benchmarks**: MMLU, HumanEval, ImageNet, GLUE, etc.
- **Pre-written analysis blocks**: problem descriptions, results summaries, strengths/weaknesses

### Multimodal First Message

The first user message uses the OpenAI-compatible **file** format:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Can you summarize this paper? ..."},
    {
      "type": "file",
      "file": {
        "url": "https://arxiv.org/pdf/2604.07349v1",
        "mime_type": "application/pdf"
      }
    }
  ]
}
```

Subsequent turns use plain text strings for content.

### Template Fill System

Simpler than text generator (~30 lines):

1. **Paper-specific fills**: `{title}`, `{authors}`, `{first_author}`, `{pdf_url}`, `{abstract}`, `{arxiv_id}`, `{primary_category}`
2. **Generic fills**: Iterate through FILL_VALUES with `rng.choice()`
3. **Cleanup**: Regex removal of unfilled placeholders

### Code Structure Breakdown

| Section | Lines | % |
|---------|-------|---|
| arXiv fetching | ~61 | 7% |
| Templates + fill values | ~358 | 44% |
| PDFConversationGenerator class | ~184 | 22% |
| aiperf export functions | ~57 | 7% |
| CLI / main() | ~100 | 12% |
| System prompt / boilerplate | ~61 | 8% |

---

## Image Dataset Generator

**File**: `image/generate.py` (1,059 lines)

### Architecture

The image generator fetches images from Wikipedia with topic-balanced selection and uses the `image_url` multimodal format.

```
image/config.yaml (10 topics × 10 articles)
    │
    ▼
fetch_wikipedia_images(config, cache_path)
    ├── For each topic (10 categories):
    │   └── For each article (10 per topic):
    │       ├── API call 1: action=query&prop=images (get image titles)
    │       ├── Filter out SVG, GIF, icons, logos, UI elements
    │       ├── API call 2: action=query&prop=imageinfo (get URLs + metadata)
    │       ├── Filter by dimensions (≥400×300) and MIME type
    │       └── Rate limit: 0.5s between articles, 0.3s between batches
    ├── Balance: equal share per topic, round-robin overflow
    ├── Deduplicate by URL
    └── Cache to wikipedia_images.json (150 images)
    │
    ▼
ImageConversationGenerator(config, images, seed=42)
    │
    ├── CONVERSATION_TEMPLATES (7 types × {openers, followups, responses})
    └── FILL_VALUES (32 keys, visual/artistic domain)
    │
    ▼
500 conversations → Parquet + multi_turn JSONL + mooncake JSONL
```

### Wikipedia Image Fetching

**Two-stage API flow per article**:

| Stage | API Parameters | Purpose |
|-------|---------------|---------|
| 1 | `action=query, prop=images, imlimit=50` | Get image file titles from article |
| 2 | `action=query, prop=imageinfo, iiprop=url\|size\|mime\|extmetadata` | Get URLs, dimensions, metadata |

**Filename filters** (excluded from stage 1):

- Extensions: `.svg`, `.gif`, `.tif`
- Keywords: `icon`, `logo`, `flag`, `symbol`, `button`, `arrow`, `commons-logo`, `wiki`, `edit-clear`, `padlock`, `ambox`, `disambiguation`, `stub`, `question_book`, `folder_hexagonal`, `text-x`, `increase2`, `decrease2`, `steady2`

**Image info filters** (excluded from stage 2):

- Width < 400 or height < 300
- MIME not starting with `image/`
- MIME is `image/svg+xml` or `image/gif`

**Topic balancing algorithm**:

```
per_topic_target = target_count // num_topics    # 150 // 10 = 15
for each topic:
    take first 15 images → all_images
    remainder → overflow pool
shortfall = target_count - len(all_images)
all_images += overflow[:shortfall]
```

**Cached image schema** (`wikipedia_images.json`):

```json
{
  "title": "Image filename.jpg",
  "url": "https://upload.wikimedia.org/wikipedia/commons/...",
  "width": 2050,
  "height": 2870,
  "mime_type": "image/jpeg",
  "description": "Image description (HTML stripped, max 500 chars)",
  "artist": "Photographer name (max 200 chars)",
  "license": "CC BY-SA 4.0",
  "topic": "nature",
  "source_article": "Grand Canyon"
}
```

### Topics (10 categories, 100 articles)

| Category | Example Articles |
|----------|-----------------|
| nature | Grand Canyon, Great Barrier Reef, Aurora borealis, Mount Everest |
| architecture | Eiffel Tower, Colosseum, Taj Mahal, Sagrada Familia |
| art | Mona Lisa, Starry Night, The Great Wave off Kanagawa |
| science | DNA, Solar System, Large Hadron Collider, Mars |
| history | Ancient Egypt, Apollo 11, Pompeii, Stonehenge |
| wildlife | African elephant, Bengal tiger, Emperor penguin |
| geography | Mount Kilimanjaro, Victoria Falls, Ha Long Bay |
| food | Sushi, Pizza, Dim sum, Coffee, Chocolate |
| technology | Microprocessor, Space Shuttle, Robotics, Wind turbine |
| culture | Carnival, Tea ceremony, Flamenco, Holi, Origami |

### Conversation Types (7)

| Type | Weight | Openers | Followups | Responses |
|------|--------|---------|-----------|-----------|
| image_description | 20% | 5 | 12 | 4 |
| visual_analysis | 20% | 5 | 12 | 4 |
| contextual_discussion | 15% | 5 | 12 | 4 |
| creative_interpretation | 15% | 5 | 12 | 4 |
| comparison | 10% | 5 | 12 | 4 |
| educational | 10% | 5 | 12 | 4 |
| technical_photography | 10% | 5 | 12 | 4 |

**Total**: 35 openers, 84 followups, 28 responses (147 templates)

### Fill Values (Visual/Artistic Domain)

32 keys with 262 total values including:

- **Visual elements**: lighting, color palette, texture, contrast, depth of field, symmetry, etc. (19 values)
- **Composition techniques**: rule of thirds, leading lines, golden ratio, dynamic asymmetry, etc. (10 values)
- **Color descriptions**: warm golden, cool blue, rich saturated, pastel, monochromatic, etc. (12 values)
- **Moods**: serene tranquility, dramatic intensity, awe-inspiring grandeur, etc. (12 values)
- **Context domains**: art history, natural science, cultural studies, ecology, etc. (12 values)
- **Abstract concepts**: passage of time, tension between order and chaos, beauty in imperfection, etc. (12 values)
- **Photography techniques**: exposure management, selective focus, HDR processing, etc. (10 values)

### Multimodal First Message

The first user message uses the OpenAI-compatible **image_url** format:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Can you describe what you see in this image?..."},
    {
      "type": "image_url",
      "image_url": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/..."
      }
    }
  ]
}
```

### Code Structure Breakdown

| Section | Lines | % |
|---------|-------|---|
| Wikipedia fetching | ~180 | 17% |
| Templates + fill values | ~453 | 43% |
| ImageConversationGenerator class | ~212 | 20% |
| aiperf export functions | ~61 | 6% |
| CLI / main() | ~102 | 10% |
| System prompt / boilerplate | ~51 | 5% |

---

## Output Format Specifications

### Parquet Schema

**Common columns** (all three datasets):

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | string | UUID v4 identifier |
| `num_turns` | int | Number of user-assistant exchange pairs |
| `num_messages` | int | Total messages including system prompt |
| `system_prompt` | string | System-level instruction |
| `messages` | string (JSON) | Full message array |
| `total_characters` | int | Character count of entire conversation |
| `estimated_tokens` | int | Approximate token count (~chars/4) |
| `cumulative_char_lengths` | string (JSON) | Array of cumulative character counts after each turn |

**Text-specific columns**:

| Column | Type | Description |
|--------|------|-------------|
| `topic` | string | Domain category (e.g., `coding_help`) |

**PDF-specific columns**:

| Column | Type | Description |
|--------|------|-------------|
| `conversation_type` | string | e.g., `paper_summary`, `critical_review` |
| `paper_arxiv_id` | string | arXiv paper ID |
| `paper_title` | string | Paper title |
| `paper_pdf_url` | string | Direct PDF URL |
| `paper_categories` | string (JSON) | arXiv categories |

**Image-specific columns**:

| Column | Type | Description |
|--------|------|-------------|
| `conversation_type` | string | e.g., `image_description`, `visual_analysis` |
| `image_title` | string | Wikipedia image title |
| `image_url` | string | Direct Wikimedia Commons URL |
| `image_topic` | string | Topic category (e.g., `nature`) |
| `source_article` | string | Wikipedia article the image was sourced from |
| `image_width` | int | Image width in pixels |
| `image_height` | int | Image height in pixels |

### aiperf multi_turn JSONL

One JSON object per line, one line per conversation. Contains only user turns (aiperf accumulates server responses into conversation history automatically).

**Text example**:

```json
{"session_id": "uuid", "turns": [{"text": "user message 1"}, {"text": "user message 2"}]}
```

**PDF example** (first turn includes PDF URL annotation):

```json
{"session_id": "uuid", "turns": [{"text": "Can you summarize...?\n\n[PDF: https://arxiv.org/pdf/2604.07349v1]"}, {"text": "followup question"}]}
```

**Image example** (first turn includes image URL annotation):

```json
{"session_id": "uuid", "turns": [{"text": "Describe this image...\n\n[Image: https://upload.wikimedia.org/.../example.jpg]"}, {"text": "followup question"}]}
```

**aiperf usage**: `--custom-dataset-type multi_turn` (context mode: `deltas_without_responses`)

### aiperf mooncake_trace JSONL

One JSON object per line, **one line per assistant turn** (not per conversation). Each entry contains the full message array up to that point, giving complete control over the exact prompt.

```json
{
  "session_id": "uuid",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "output_length": 150,
  "delay": 0
}
```

- `output_length`: Estimated output tokens for this turn (`max(1, len(content) // 4)`)
- `delay`: Inter-turn delay in seconds (0 for all turns after the first)
- For PDF/image: first user message preserves the full multimodal content structure

**aiperf usage**: `--custom-dataset-type mooncake_trace` (context mode: `message_array_with_responses`)

### File Size Comparison

| Dataset | Parquet | multi_turn JSONL | mooncake_trace JSONL |
|---------|---------|------------------|----------------------|
| text | ~2 MB | ~0.8 MB | ~227 MB |
| pdf | ~1.6 MB | ~0.5 MB | ~97 MB |
| image | ~1.6 MB | ~0.5 MB | ~99 MB |

The mooncake format is 100-300x larger because it duplicates the full message history for every turn. These files are gitignored and regenerated on demand.

---

## Configuration System

All configuration is externalized to YAML files with identical top-level structure:

```yaml
dataset:           # Output metadata and generation parameters
  name: "..."
  version: "1.0.0"
  num_conversations: 500
  output_dir: "data"
  output_filename: "*.parquet"
  seed: 42

turns:             # Turn count distribution across length buckets
  min: 1
  max: 30          # (50 for text)
  distribution:
    short:     {count: N, min_turns: A, max_turns: B}
    medium:    {count: N, min_turns: A, max_turns: B}
    long:      {count: N, min_turns: A, max_turns: B}
    very_long: {count: N, min_turns: A, max_turns: B}

response_length:   # Word count targets by response bucket
  short:   {min_words: N, max_words: M}
  medium:  {min_words: N, max_words: M}
  long:    {min_words: N, max_words: M}
  length_distribution_by_turn:
    early:   {short: P, medium: P, long: P}
    middle:  {short: P, medium: P, long: P}
    late:    {short: P, medium: P, long: P}
```

**Dataset-specific sections**:

- **text**: `topics` list with name, weight, system_prompt, description
- **pdf**: `papers` (count, categories, cache_file) + `conversation_types` list
- **image**: `images` (count, topics/articles, cache_file, min dimensions) + `conversation_types` list

---

## Extensibility Guide

### Adding a New Topic (text generator)

1. Add template entry to `TOPIC_TEMPLATES` in `text/generate.py`:
   ```python
   "new_topic": {
       "openers": [...],
       "followups": [...],
       "responses": [...],
       "fill_values": {"placeholder": ["val1", "val2", ...]}
   }
   ```
2. Add topic config in `text/config.yaml`:
   ```yaml
   - name: "new_topic"
     weight: 0.10
     system_prompt: "You are..."
   ```
3. Adjust other topic weights to sum to 1.0

### Adding a New Conversation Type (pdf/image generators)

1. Add template entry to `CONVERSATION_TEMPLATES`
2. Add fill values to `FILL_VALUES` for any new placeholders
3. Add type to config `conversation_types` list with weight

### Adding a New Dataset Type

Follow the existing pattern:

1. Create `new_type/config.yaml` mirroring the shared structure
2. Create `new_type/generate.py` with:
   - Data fetching function (if external source) with JSON caching
   - `CONVERSATION_TEMPLATES` dict with openers/followups/responses
   - `FILL_VALUES` dict for domain-specific placeholders
   - Generator class inheriting the shared method pattern
   - `convert_to_aiperf_multi_turn()` and `convert_to_aiperf_mooncake()` functions
   - `main()` with the standard CLI flags
3. Create `new_type/data/` directory
4. Update `requirements.txt` if new dependencies are needed
5. Update `README.md` and this document

### Adding a New Output Format

Add a new conversion function following the pattern:

```python
def convert_to_new_format(conversations: list[dict]) -> list[dict]:
    entries = []
    for conv in conversations:
        messages = json.loads(conv["messages"])
        # Transform to target format
        entries.append({...})
    return entries
```

Then add a new `--format` choice in `main()` and wire up the export logic.
