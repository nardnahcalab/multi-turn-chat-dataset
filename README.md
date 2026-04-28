# multi-turn-chat-dataset

Synthetic multi-turn conversation datasets for benchmarking LLM inference engines. Designed to simulate real-world conversations with naturally growing context, ideal for stress-testing **prefix caching**, **KV-cache management**, and **long-context inference** performance.

## Dataset Types

| Type | Status | Description |
|------|--------|-------------|
| **text/** | Available | Pure text multi-turn conversations |
| **image/** | Available | Multi-turn with Wikipedia image references |
| **pdf/** | Available | Multi-turn with arXiv PDF document context |
| **reasoning/** | Available | Deep reasoning multi-turn conversations |
| **agentic/** | Available | Agent task execution with tool-use and success metrics |
| **random/** | Available | Random/gibberish text multi-turn conversations |
| **repeat/** | Available | Repetitive text multi-turn conversations |

## Examples and Usage

Get started quickly with practical examples. All examples are in the `examples/` directory.

### Run Examples

```bash
# Setup
source .venv/bin/activate

# Generate datasets (if needed)
python text/generate.py
python agentic/generate.py

# Run examples
python examples/01_load_and_inspect.py text
python examples/02_analyze_conversations.py text
python examples/03_prepare_for_benchmarking.py --dataset text
python examples/04_agentic_analysis.py
```

### Example 1: Load and Inspect Datasets

```python
import pandas as pd
import json

# Load dataset
df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Basic info
print(f"Conversations: {len(df)}")
print(f"Total tokens: {df['estimated_tokens'].sum():,}")

# Access a conversation
row = df.iloc[0]
messages = json.loads(row["messages"])

# Print messages
for msg in messages:
    print(f"[{msg['role']}] {msg['content'][:100]}...")
```

**See**: `examples/01_load_and_inspect.py` for complete example

### Example 2: Analyze Context Growth

```python
import json

row = df.iloc[0]
cumulative_lengths = json.loads(row["cumulative_char_lengths"])

# Track context growth
for i, char_count in enumerate(cumulative_lengths):
    token_count = char_count // 4
    print(f"Turn {i+1}: {char_count:,} chars (~{token_count:,} tokens)")
```

**See**: `examples/02_analyze_conversations.py` for complete example

### Example 3: Prepare for Benchmarking

```bash
# Create a subset for testing
python examples/03_prepare_for_benchmarking.py \
    --dataset text \
    --format subset \
    --num-samples 100

# Filter by turn count
python examples/03_prepare_for_benchmarking.py \
    --dataset text \
    --format filtered \
    --min-turns 5 \
    --max-turns 15

# Estimate benchmark time
python examples/03_prepare_for_benchmarking.py \
    --dataset text \
    --tokens-per-sec 100
```

**See**: `examples/03_prepare_for_benchmarking.py` for complete example

### Example 4: Analyze Agentic Tasks

```bash
# Full analysis
python examples/04_agentic_analysis.py

# Compare conversations
python examples/04_agentic_analysis.py --compare 0 1

# Export report
python examples/04_agentic_analysis.py --report
```

**See**: `examples/04_agentic_analysis.py` for complete example

### More Documentation

For detailed tutorials and advanced examples, see:
- **[EXAMPLES.md](EXAMPLES.md)** - Comprehensive tutorials with code walkthroughs
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API and schema reference
- **[FAQ.md](FAQ.md)** - Common questions and troubleshooting
- **[examples/README.md](examples/README.md)** - Quick reference for all examples

## Agentic Task Dataset

Multi-turn conversations where agents execute high-level goals using tool calls, error recovery, and iterative refinement. Designed to benchmark **agent performance** with traceable success metrics that punish partial credit and measure end-to-end task completion.

### Quick Start

```bash
# Setup (same venv as other datasets)
source .venv/bin/activate

# Generate all formats (500 tasks, ~2.5M tokens)
python agentic/generate.py

# Generate only specific format
python agentic/generate.py --format parquet
python agentic/generate.py --format aiperf      # multi_turn JSONL only
python agentic/generate.py --format mooncake    # mooncake_trace JSONL only

# Custom generation
python agentic/generate.py --num 1000 --seed 123
```

### Output Formats

The generator produces three output files:

| File | Format | Size | aiperf `--custom-dataset-type` |
|------|--------|------|-------------------------------|
| `multi_turn_agentic_task.parquet` | Parquet | ~2.5 MB | N/A (analysis/HuggingFace) |
| `multi_turn_agentic_task.jsonl` | JSONL | ~0.6 MB | `multi_turn` |
| `multi_turn_agentic_task_mooncake.jsonl` | JSONL | ~250 MB | `mooncake_trace` |

### Using with aiperf

```bash
# multi_turn format (lightweight, user messages only)
aiperf profile \
    --model <your-model> \
    --endpoint-type chat \
    --input-file agentic/data/multi_turn_agentic_task.jsonl \
    --custom-dataset-type multi_turn \
    --streaming --url localhost:8000 --concurrency 10

# mooncake_trace format (full message arrays with tool calls)
aiperf profile \
    --model <your-model> \
    --endpoint-type chat \
    --input-file agentic/data/multi_turn_agentic_task_mooncake.jsonl \
    --custom-dataset-type mooncake_trace \
    --streaming --url localhost:8000 --concurrency 10
```

### Task Types

Conversations span 6 agent task domains:

| Task Type | Weight | Description |
|-----------|--------|-------------|
| **data_processing** | 20% | Data transformation, validation, aggregation, filtering |
| **api_integration** | 20% | API calls, data fetching, error handling, integration |
| **system_troubleshooting** | 15% | Diagnosis, log analysis, configuration, system repair |
| **code_generation** | 15% | Code writing, testing, debugging, optimization |
| **research_synthesis** | 15% | Information gathering, synthesis, report generation |
| **planning_execution** | 15% | Task decomposition, execution, progress tracking, adaptation |

### Schema (Parquet)

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | string | UUID v4 identifier |
| `task_type` | string | Agent task domain (e.g., `data_processing`, `api_integration`) |
| `num_turns` | int | Number of user-agent exchange pairs |
| `num_messages` | int | Total messages including system prompt |
| `system_prompt` | string | Agent-specific system instruction |
| `messages` | string (JSON) | Full message array with tool calls and results |
| `tool_calls` | string (JSON) | Detailed log of all tool invocations and results |
| `total_characters` | int | Character count of entire conversation |
| `estimated_tokens` | int | Approximate token count (~chars/4) |
| `cumulative_char_lengths` | string (JSON) | Array of cumulative character counts after each turn |
| `success_metric` | string | Metric used to evaluate task completion |
| `success_score` | float | Success score (0.0-1.0) with partial credit penalties |
| `num_tool_calls` | int | Total number of tool invocations |
| `num_errors` | int | Number of failed tool calls |

### Tool Definitions

Agents have access to 24 tools across 6 categories:

**Data Processing Tools:**
- `query_database` — SQL queries with timeout handling
- `transform_data` — Filter, map, aggregate, join operations
- `validate_schema` — Data validation against schema
- `export_data` — Export to CSV, JSON, Parquet, SQL

**API Integration Tools:**
- `call_api` — HTTP requests (GET, POST, PUT, DELETE)
- `parse_response` — Parse JSON, XML, HTML responses
- `retry_with_backoff` — Exponential backoff retry logic
- `log_error` — Error logging with context

**System Troubleshooting Tools:**
- `check_logs` — Search system/application/error logs
- `diagnose_issue` — Run diagnostic checks (CPU, memory, disk, network)
- `apply_fix` — Apply fixes (restart, reconfigure, patch, rollback)
- `verify_resolution` — Verify issue resolution

**Code Generation Tools:**
- `write_code` — Generate code in Python, JavaScript, Go, Rust, Java
- `execute_code` — Run code in sandbox with timeout
- `run_tests` — Execute test suites (pytest, jest, go test)
- `debug_code` — Debug with breakpoints and instrumentation

**Research Synthesis Tools:**
- `search_knowledge_base` — Search documentation/wiki/research
- `fetch_document` — Retrieve documents in various formats
- `summarize_content` — Summarize text/documents
- `generate_report` — Generate structured reports

**Planning & Execution Tools:**
- `decompose_task` — Break down high-level objectives
- `execute_step` — Execute individual steps
- `track_progress` — Monitor progress toward objective
- `adapt_plan` — Adapt plan based on new information

### Success Metrics

Each task type has a specific success metric with **partial credit penalties**:

| Metric | Calculation | Partial Credit | Penalty |
|--------|-----------|---|---------|
| `data_integrity_score` | validated_records / total_records | No | 50% |
| `integration_completeness_score` | integrated_fields / required_fields | Yes | 30% |
| `resolution_success_score` | 1.0 if resolved else 0.0 | No | 0% |
| `test_pass_rate` | passed_tests / total_tests | Yes | 20% |
| `coverage_completeness_score` | covered_topics / required_topics | Yes | 40% |
| `objective_completion_score` | completed_steps / total_steps | Yes | 30% |

**Partial credit punishment:** Tasks with incomplete execution receive reduced scores. For example, a data processing task that validates 95% of records gets 0.95 * base_score, not a binary pass/fail.

### Default Dataset Stats

- **500 conversations**, **~2.5M estimated tokens**
- Turn range: **2-15**, mean: **~7**
- **6 task types** with balanced distribution
- **24 tools** with realistic error injection (~15% failure rate)
- **Tool call distribution:** 0 calls (10%), 1 call (40%), 2 calls (35%), 3+ calls (15%)

### Loading the Dataset

```python
import pandas as pd
import json

df = pd.read_parquet("agentic/data/multi_turn_agentic_task.parquet")

# Access a task conversation
row = df.iloc[0]
messages = json.loads(row["messages"])
tool_calls = json.loads(row["tool_calls"])

print(f"Task: {row['task_type']}")
print(f"Success Score: {row['success_score']:.2f}")
print(f"Tool Calls: {row['num_tool_calls']}, Errors: {row['num_errors']}")

# Analyze tool usage
for tool_call in tool_calls:
    print(f"  {tool_call['tool']}: {tool_call['result']['status']}")

# Analyze context growth
lengths = json.loads(row["cumulative_char_lengths"])
print(f"Context growth: {lengths[0]:,} -> {lengths[-1]:,} chars over {row['num_turns']} turns")
```

### Benchmarking Agent Performance

Each conversation simulates real agent execution: the agent receives a high-level goal, makes tool calls, handles errors, and iteratively refines its approach. To benchmark:

```python
# Simulate agent execution
messages = json.loads(row["messages"])
tool_calls = json.loads(row["tool_calls"])

# Measure:
# - Tool call accuracy (% successful calls)
# - Error recovery (% of errors handled correctly)
# - Goal completion (success_score metric)
# - Token efficiency (tokens_used / estimated_tokens)
# - Iteration efficiency (num_turns / optimal_turns)

tool_success_rate = 1 - (row['num_errors'] / row['num_tool_calls']) if row['num_tool_calls'] > 0 else 1.0
print(f"Tool Success Rate: {tool_success_rate:.2%}")
print(f"Task Success Score: {row['success_score']:.2%}")
```

### Configuration

Edit `agentic/config.yaml` to customize:
- Number of conversations and distribution across turn-count buckets
- Task type weights and system prompts
- Tool definitions and error injection rates
- Response length distributions by conversation phase
- Success metric thresholds and partial credit penalties
- Random seed for reproducibility

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

## Reasoning Dataset

Multi-turn conversations designed to prompt deep chain-of-thought reasoning. Covers mathematical proofs, logic puzzles, algorithmic analysis, scientific reasoning, philosophical arguments, game theory, causal reasoning, and puzzle solving. Ideal for benchmarking reasoning-heavy inference workloads with long output sequences.

### Quick Start

```bash
# Setup (same venv as text dataset)
source .venv/bin/activate

# Generate all formats (500 conversations, ~5.8M tokens)
python reasoning/generate.py

# Generate only specific format
python reasoning/generate.py --format parquet
python reasoning/generate.py --format aiperf
python reasoning/generate.py --format mooncake

# Custom generation
python reasoning/generate.py --num 1000 --seed 123
```

### How It Works

1. **Generates** multi-turn conversations where users pose complex reasoning problems
2. System prompts instruct the model to show step-by-step reasoning, justify each logical step, and verify conclusions
3. Follow-up turns challenge assumptions, ask for alternative approaches, request generalizations, and probe edge cases
4. Responses include formal proofs, case analysis, complexity derivations, and structured arguments

### Topics

Conversations span 8 reasoning domains with configurable weights:

| Topic | Weight | Description |
|-------|--------|-------------|
| mathematical_proofs | 20% | Number theory, algebra, combinatorics, analysis — formal proofs |
| logic_and_deduction | 15% | Formal logic, knight/knave puzzles, syllogisms, truth tables |
| algorithmic_analysis | 15% | Algorithm design, complexity proofs, NP-hardness reductions |
| scientific_reasoning | 15% | Hypothesis testing, experimental design, causal inference |
| philosophical_arguments | 10% | Ethical dilemmas, thought experiments, epistemology |
| game_theory_and_strategy | 10% | Nash equilibrium, mechanism design, decision theory |
| causal_and_counterfactual | 10% | Causal DAGs, counterfactual analysis, Simpson's paradox |
| puzzle_solving | 5% | Brain teasers, constraint satisfaction, combinatorial puzzles |

### Output Formats

| File | Format | Size | aiperf `--custom-dataset-type` |
|------|--------|------|-------------------------------|
| `multi_turn_reasoning_chat.parquet` | Parquet | ~3.2 MB | N/A |
| `multi_turn_reasoning_chat.jsonl` | JSONL | ~0.8 MB | `multi_turn` |
| `multi_turn_reasoning_chat_mooncake.jsonl` | JSONL | ~253 MB | `mooncake_trace` |

### Using with aiperf

```bash
# multi_turn format (lightweight, user reasoning prompts only)
aiperf profile \
    --model <your-model> \
    --endpoint-type chat \
    --input-file reasoning/data/multi_turn_reasoning_chat.jsonl \
    --custom-dataset-type multi_turn \
    --streaming --url localhost:8000 --concurrency 10

# mooncake_trace format (full message arrays with chain-of-thought responses)
aiperf profile \
    --model <your-model> \
    --endpoint-type chat \
    --input-file reasoning/data/multi_turn_reasoning_chat_mooncake.jsonl \
    --custom-dataset-type mooncake_trace \
    --streaming --url localhost:8000 --concurrency 10
```

### Schema (Parquet)

Uses the same schema as the text dataset:

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | string | UUID v4 identifier |
| `topic` | string | Reasoning domain (e.g., `mathematical_proofs`, `logic_and_deduction`) |
| `num_turns` | int | Number of user-assistant exchange pairs |
| `num_messages` | int | Total messages including system prompt |
| `system_prompt` | string | Reasoning-focused system instruction |
| `messages` | string (JSON) | Full message array with chain-of-thought responses |
| `total_characters` | int | Character count of entire conversation |
| `estimated_tokens` | int | Approximate token count (~chars/4) |
| `cumulative_char_lengths` | string (JSON) | Array of cumulative character counts after each turn |

### Turn Distribution

| Bucket | Turns | Count | Use Case |
|--------|-------|-------|----------|
| Short | 1-5 | 80 | Quick single-step reasoning |
| Medium | 6-15 | 200 | Multi-step proofs and analysis |
| Long | 16-25 | 150 | Extended reasoning chains |
| Very Long | 26-40 | 70 | Deep multi-round reasoning dialogues |

### Default Dataset Stats

- **500 conversations**, **~5.8M estimated tokens**
- Turn range: **1-40**, mean: **~15.5**
- Max single conversation: **~36K tokens**
- File size: **~3.2 MB** (Parquet)
- **8 reasoning domains** with configurable weights

### Configuration

Edit `reasoning/config.yaml` to customize:
- Number of conversations and distribution across turn-count buckets
- Topic weights and system prompts (reasoning-focused)
- Response length distributions — skewed longer to accommodate chain-of-thought
- Random seed for reproducibility

## Random Dataset

Multi-turn conversations with randomly generated text — random words, character sequences, gibberish sentences, mixed content, and lorem ipsum. Designed to benchmark inference with **unpredictable, non-cacheable content** where prefix caching provides minimal benefit, serving as a baseline comparison against structured datasets.

### Quick Start

```bash
# Setup (same venv as text dataset)
source .venv/bin/activate

# Generate all formats (500 conversations, ~5M tokens)
python random/generate.py

# Generate only specific format
python random/generate.py --format parquet
python random/generate.py --format aiperf
python random/generate.py --format mooncake

# Custom generation
python random/generate.py --num 1000 --seed 123
```

### How It Works

1. **Generates** multi-turn conversations where user messages are randomly constructed text
2. System prompts instruct the model to respond to random content — analyzing patterns, interpreting gibberish, engaging creatively
3. Each topic type uses a different randomization strategy (words, chars, sentences, mixed, lorem)
4. Content is intentionally unpredictable to minimize prefix cache reuse

### Topics

Conversations span 5 random content types with configurable weights:

| Topic | Weight | Description |
|-------|--------|-------------|
| random_words | 25% | Random words from a ~500-word vocabulary list |
| random_sentences | 25% | Grammatically structured but semantically random sentences |
| random_chars | 20% | Random alphanumeric character sequences |
| random_mixed | 15% | Mixed words, numbers, and symbols |
| random_lorem | 15% | Lorem ipsum style pseudo-Latin placeholder text |

### Output Formats

| File | Format | Size | aiperf `--custom-dataset-type` |
|------|--------|------|-------------------------------|
| `multi_turn_random_chat.parquet` | Parquet | ~7 MB | N/A |
| `multi_turn_random_chat.jsonl` | JSONL | ~1.7 MB | `multi_turn` |
| `multi_turn_random_chat_mooncake.jsonl` | JSONL | ~290 MB | `mooncake_trace` |

### Using with aiperf

```bash
# multi_turn format (lightweight, random user messages only)
aiperf profile \
    --model <your-model> \
    --endpoint-type chat \
    --input-file random/data/multi_turn_random_chat.jsonl \
    --custom-dataset-type multi_turn \
    --streaming --url localhost:8000 --concurrency 10

# mooncake_trace format (full message arrays)
aiperf profile \
    --model <your-model> \
    --endpoint-type chat \
    --input-file random/data/multi_turn_random_chat_mooncake.jsonl \
    --custom-dataset-type mooncake_trace \
    --streaming --url localhost:8000 --concurrency 10
```

### Schema (Parquet)

Uses the same schema as the text dataset:

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | string | UUID v4 identifier |
| `topic` | string | Random content type (e.g., `random_words`, `random_chars`) |
| `num_turns` | int | Number of user-assistant exchange pairs |
| `num_messages` | int | Total messages including system prompt |
| `system_prompt` | string | System-level instruction |
| `messages` | string (JSON) | Full message array |
| `total_characters` | int | Character count of entire conversation |
| `estimated_tokens` | int | Approximate token count (~chars/4) |
| `cumulative_char_lengths` | string (JSON) | Array of cumulative character counts after each turn |

### Turn Distribution

| Bucket | Turns | Count | Use Case |
|--------|-------|-------|----------|
| Short | 1-5 | 100 | Cache cold start, minimal reuse |
| Medium | 6-15 | 150 | Moderate-length random conversations |
| Long | 16-30 | 150 | Long unpredictable conversations |
| Very Long | 31-50 | 100 | Stress-test with extended random content |

### Default Dataset Stats

- **500 conversations**, **~5M estimated tokens**
- Turn range: **1-50**, mean: **~19**
- Max single conversation: **~33K tokens**
- File size: **~7 MB** (Parquet)
- **5 random content types** with configurable weights

### Configuration

Edit `random/config.yaml` to customize:
- Number of conversations and distribution across turn-count buckets
- Topic weights and system prompts
- Response length distributions (short/medium/long) by conversation phase
- Random seed for reproducibility

## Repeat Dataset

Multi-turn conversations with highly repetitive content — single words, phrases, counting sequences, letter patterns, and full sentences repeated many times. Designed to benchmark inference with **maximally compressible, highly cacheable content**, testing tokenizer efficiency and KV-cache behavior with near-identical token sequences.

### Quick Start

```bash
# Setup (same venv as text dataset)
source .venv/bin/activate

# Generate all formats (500 conversations, ~4.8M tokens)
python repeat/generate.py

# Generate only specific format
python repeat/generate.py --format parquet
python repeat/generate.py --format aiperf
python repeat/generate.py --format mooncake

# Custom generation
python repeat/generate.py --num 1000 --seed 123
```

### How It Works

1. **Generates** multi-turn conversations where user messages consist of repeated words/phrases/patterns
2. Within each conversation, the **same base word/phrase/pattern** is used across all turns
3. The **number of repetitions varies per turn** — creating growing context with predictable content
4. Assistant responses acknowledge the repetitive content and respond naturally

### Topics

Conversations span 5 repetition types with configurable weights:

| Topic | Weight | Description |
|-------|--------|-------------|
| single_word | 25% | A single word repeated N times (e.g., "hello hello hello...") |
| sentence_repeat | 20% | A full sentence repeated N times |
| counting_repeat | 20% | Counting sequences repeated (e.g., "1 2 3 1 2 3...") |
| phrase_repeat | 20% | A short phrase repeated N times (e.g., "the cat sat the cat sat...") |
| letter_repeat | 15% | Letters or character patterns repeated (e.g., "aaabbbccc...") |

### Output Formats

| File | Format | Size | aiperf `--custom-dataset-type` |
|------|--------|------|-------------------------------|
| `multi_turn_repeat_chat.parquet` | Parquet | ~2.2 MB | N/A |
| `multi_turn_repeat_chat.jsonl` | JSONL | ~3 MB | `multi_turn` |
| `multi_turn_repeat_chat_mooncake.jsonl` | JSONL | ~269 MB | `mooncake_trace` |

### Using with aiperf

```bash
# multi_turn format (lightweight, repetitive user messages only)
aiperf profile \
    --model <your-model> \
    --endpoint-type chat \
    --input-file repeat/data/multi_turn_repeat_chat.jsonl \
    --custom-dataset-type multi_turn \
    --streaming --url localhost:8000 --concurrency 10

# mooncake_trace format (full message arrays)
aiperf profile \
    --model <your-model> \
    --endpoint-type chat \
    --input-file repeat/data/multi_turn_repeat_chat_mooncake.jsonl \
    --custom-dataset-type mooncake_trace \
    --streaming --url localhost:8000 --concurrency 10
```

### Schema (Parquet)

Uses the same schema as the text dataset:

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | string | UUID v4 identifier |
| `topic` | string | Repetition type (e.g., `single_word`, `phrase_repeat`) |
| `num_turns` | int | Number of user-assistant exchange pairs |
| `num_messages` | int | Total messages including system prompt |
| `system_prompt` | string | System-level instruction |
| `messages` | string (JSON) | Full message array |
| `total_characters` | int | Character count of entire conversation |
| `estimated_tokens` | int | Approximate token count (~chars/4) |
| `cumulative_char_lengths` | string (JSON) | Array of cumulative character counts after each turn |

### Turn Distribution

| Bucket | Turns | Count | Use Case |
|--------|-------|-------|----------|
| Short | 1-5 | 100 | Cache cold start with repetitive content |
| Medium | 6-15 | 150 | Moderate prefix sharing with identical patterns |
| Long | 16-30 | 150 | Significant cache benefit from repetition |
| Very Long | 31-50 | 100 | Stress-test with maximally repetitive context |

### Default Dataset Stats

- **500 conversations**, **~4.8M estimated tokens**
- Turn range: **1-50**, mean: **~19**
- Max single conversation: **~33K tokens**
- File size: **~2.2 MB** (Parquet)
- **5 repetition types** with configurable weights

### Configuration

Edit `repeat/config.yaml` to customize:
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
├── reasoning/
│   ├── generate.py          # Generation script (deep reasoning conversations)
│   ├── config.yaml          # Configuration
│   └── data/
│       ├── multi_turn_reasoning_chat.parquet           # Full dataset (Parquet)
│       ├── multi_turn_reasoning_chat.jsonl             # aiperf multi_turn format
│       └── multi_turn_reasoning_chat_mooncake.jsonl    # aiperf mooncake_trace (generated, not in git)
├── random/
│   ├── generate.py          # Generation script (random/gibberish text)
│   ├── config.yaml          # Configuration
│   └── data/
│       ├── multi_turn_random_chat.parquet              # Full dataset (Parquet)
│       ├── multi_turn_random_chat.jsonl                # aiperf multi_turn format
│       └── multi_turn_random_chat_mooncake.jsonl       # aiperf mooncake_trace (generated, not in git)
├── repeat/
│   ├── generate.py          # Generation script (repetitive text)
│   ├── config.yaml          # Configuration
│   └── data/
│       ├── multi_turn_repeat_chat.parquet              # Full dataset (Parquet)
│       ├── multi_turn_repeat_chat.jsonl                # aiperf multi_turn format
│       └── multi_turn_repeat_chat_mooncake.jsonl       # aiperf mooncake_trace (generated, not in git)
└── .gitignore
```

## License

See [LICENSE](LICENSE) for details.
