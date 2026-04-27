# API Reference

Complete API reference for the multi-turn-chat-dataset project. This document covers the data schemas, configuration options, and programmatic interfaces.

## Table of Contents

- [Dataset Schemas](#dataset-schemas)
- [Configuration Reference](#configuration-reference)
- [Generator APIs](#generator-apis)
- [Output Formats](#output-formats)
- [Data Loading](#data-loading)
- [Common Operations](#common-operations)

---

## Dataset Schemas

### Text Dataset Schema

**File**: `text/data/multi_turn_text_chat.parquet`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `conversation_id` | string | UUID v4 identifier | `"550e8400-e29b-41d4-a716-446655440000"` |
| `topic` | string | Domain category | `"coding_help"`, `"customer_support"` |
| `num_turns` | int | Number of user-assistant exchange pairs | `8` |
| `num_messages` | int | Total messages including system prompt | `17` |
| `system_prompt` | string | System-level instruction | `"You are a helpful coding assistant..."` |
| `messages` | string (JSON) | Full message array | `[{"role": "system", "content": "..."}, ...]` |
| `total_characters` | int | Character count of entire conversation | `3245` |
| `estimated_tokens` | int | Approximate token count (~chars/4) | `811` |
| `cumulative_char_lengths` | string (JSON) | Array of cumulative character counts after each turn | `[245, 512, 1024, ...]` |

**Message Structure**:
```json
{
  "role": "system|user|assistant",
  "content": "Message text content"
}
```

**Topics** (8 types):
- `coding_help` (20%)
- `customer_support` (15%)
- `tutoring` (15%)
- `creative_writing` (10%)
- `travel_planning` (10%)
- `data_analysis` (10%)
- `business_strategy` (10%)
- `health_fitness` (10%)

---

### PDF Dataset Schema

**File**: `pdf/data/multi_turn_pdf_chat.parquet`

Extends text schema with PDF-specific columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `conversation_type` | string | Type of analysis | `"paper_summary"`, `"critical_review"` |
| `paper_arxiv_id` | string | arXiv paper ID | `"2401.12345"` |
| `paper_title` | string | Paper title | `"Attention Is All You Need"` |
| `paper_pdf_url` | string | Direct PDF URL | `"https://arxiv.org/pdf/2401.12345v1"` |
| `paper_categories` | string (JSON) | arXiv categories | `["cs.CL", "cs.AI"]` |

**Conversation Types** (7 types):
- `paper_summary` (20%)
- `methodology_deep_dive` (20%)
- `results_analysis` (15%)
- `critical_review` (15%)
- `comparison` (10%)
- `implementation` (10%)
- `brainstorm_extensions` (10%)

---

### Image Dataset Schema

**File**: `image/data/multi_turn_image_chat.parquet`

Extends text schema with image-specific columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `image_topic` | string | Image topic category | `"nature"`, `"architecture"` |
| `conversation_type` | string | Type of analysis | `"description"`, `"analysis"` |
| `image_url` | string | Image URL | `"https://upload.wikimedia.org/..."` |
| `image_title` | string | Image title | `"Eiffel Tower"` |

**Image Topics** (10 types):
- `nature`, `architecture`, `art`, `science`, `history`
- `wildlife`, `geography`, `food`, `technology`, `culture`

---

### Reasoning Dataset Schema

**File**: `reasoning/data/multi_turn_reasoning_chat.parquet`

Uses the same schema as the text dataset. The `topic` column contains reasoning-specific categories.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `conversation_id` | string | UUID v4 identifier | `"550e8400-e29b-41d4-a716-446655440000"` |
| `topic` | string | Reasoning category | `"mathematical_proofs"`, `"logic_and_deduction"` |
| `num_turns` | int | Number of user-assistant exchange pairs | `12` |
| `num_messages` | int | Total messages including system prompt | `25` |
| `system_prompt` | string | System-level instruction | `"You are a reasoning assistant..."` |
| `messages` | string (JSON) | Full message array | `[{"role": "system", "content": "..."}, ...]` |
| `total_characters` | int | Character count of entire conversation | `8245` |
| `estimated_tokens` | int | Approximate token count (~chars/4) | `2061` |
| `cumulative_char_lengths` | string (JSON) | Array of cumulative character counts after each turn | `[245, 512, 1024, ...]` |

**Topics** (8 types):
- `scientific_reasoning` - Scientific analysis and hypothesis testing
- `mathematical_proofs` - Mathematical proofs and derivations
- `algorithmic_analysis` - Algorithm design and complexity analysis
- `logic_and_deduction` - Formal logic and deductive reasoning
- `philosophical_arguments` - Philosophical analysis and argumentation
- `causal_and_counterfactual` - Causal reasoning and counterfactual analysis
- `game_theory_and_strategy` - Game theory and strategic reasoning
- `puzzle_solving` - Puzzles and problem-solving challenges

---

### Agentic Dataset Schema

**File**: `agentic/data/multi_turn_agentic_task.parquet`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `conversation_id` | string | UUID v4 identifier | `"550e8400-e29b-41d4-a716-446655440000"` |
| `task_type` | string | Agent task domain | `"data_processing"`, `"api_integration"` |
| `num_turns` | int | Number of user-agent exchange pairs | `7` |
| `num_messages` | int | Total messages including system prompt | `15` |
| `system_prompt` | string | Agent-specific system instruction | `"You are a data processing agent..."` |
| `messages` | string (JSON) | Full message array with tool calls | `[{"role": "system", "content": "..."}, ...]` |
| `tool_calls` | string (JSON) | Detailed log of all tool invocations | `[{"tool": "query_database", "result": {...}}, ...]` |
| `total_characters` | int | Character count of entire conversation | `2845` |
| `estimated_tokens` | int | Approximate token count | `711` |
| `cumulative_char_lengths` | string (JSON) | Array of cumulative character counts | `[245, 512, 1024, ...]` |
| `success_metric` | string | Metric used to evaluate task completion | `"data_integrity_score"` |
| `success_score` | float | Success score (0.0-1.0) with partial credit penalties | `0.85` |
| `num_tool_calls` | int | Total number of tool invocations | `12` |
| `num_errors` | int | Number of failed tool calls | `2` |

**Task Types** (6 types):
- `data_processing` (20%) - Data transformation, validation, aggregation
- `api_integration` (20%) - API calls, data fetching, error handling
- `system_troubleshooting` (15%) - Diagnosis, log analysis, repair
- `code_generation` (15%) - Code writing, testing, debugging
- `research_synthesis` (15%) - Information gathering, synthesis, reporting
- `planning_execution` (15%) - Task decomposition, execution, tracking

**Tool Call Structure**:
```json
{
  "tool": "query_database",
  "parameters": {
    "query": "SELECT * FROM users WHERE...",
    "timeout": 30
  },
  "result": {
    "status": "success|failure",
    "result": "Query result data",
    "error": "Error message if failed"
  }
}
```

**Success Metrics**:
- `data_integrity_score` - No partial credit (50% penalty)
- `integration_completeness_score` - Partial credit (30% penalty)
- `resolution_success_score` - Binary (0% penalty)
- `test_pass_rate` - Partial credit (20% penalty)
- `coverage_completeness_score` - Partial credit (40% penalty)
- `objective_completion_score` - Partial credit (30% penalty)

---

## Configuration Reference

### Text Configuration

**File**: `text/config.yaml`

```yaml
dataset:
  name: "multi-turn-text-chat"
  num_conversations: 500
  output_dir: "data"
  seed: 42

turns:
  distribution:
    short:
      count: 100
      min_turns: 1
      max_turns: 5
    medium:
      count: 150
      min_turns: 6
      max_turns: 15
    long:
      count: 150
      min_turns: 16
      max_turns: 30
    very_long:
      count: 100
      min_turns: 31
      max_turns: 50

topics:
  coding_help:
    weight: 0.20
    system_prompt: "You are a helpful coding assistant..."
  customer_support:
    weight: 0.15
    system_prompt: "You are a customer support representative..."
  # ... more topics

response_lengths:
  short: [50, 150]
  medium: [150, 400]
  long: [400, 1000]
```

### Agentic Configuration

**File**: `agentic/config.yaml`

```yaml
dataset:
  name: "multi-turn-agentic-task"
  num_conversations: 500
  output_dir: "data"
  seed: 42

turns:
  distribution:
    short:
      count: 150
      min_turns: 2
      max_turns: 4
    medium:
      count: 200
      min_turns: 5
      max_turns: 8
    long:
      count: 150
      min_turns: 9
      max_turns: 15

task_types:
  data_processing:
    weight: 0.20
    system_prompt: "You are a data processing agent..."
    tools: [query_database, transform_data, validate_schema, export_data]
    success_metric: data_integrity_score
  # ... more task types

tools:
  query_database:
    category: "data_processing"
    parameters:
      - name: query
        type: string
      - name: timeout
        type: integer
    error_modes: [timeout, syntax_error, connection_error]
  # ... more tools

error_injection:
  enabled: true
  failure_rate: 0.15
```

---

## Generator APIs

### Text Generator

```python
from text.generate import TextDatasetGenerator

# Create generator
gen = TextDatasetGenerator(
    num_conversations=500,
    seed=42,
    output_dir="text/data"
)

# Generate all formats
gen.generate_all()

# Generate specific format
gen.generate_parquet()
gen.generate_aiperf_jsonl()
gen.generate_mooncake_jsonl()
```

### Agentic Generator

```python
from agentic.generate import AgenticDatasetGenerator

# Create generator
gen = AgenticDatasetGenerator(
    num_conversations=500,
    seed=42,
    output_dir="agentic/data"
)

# Generate all formats
gen.generate_all()

# Generate specific format
gen.generate_parquet()
gen.generate_aiperf_jsonl()
```

---

## Output Formats

### Parquet Format

**Description**: Full dataset with all columns and metadata

**File Size**: ~2 MB (500 conversations)

**Usage**:
```python
import pandas as pd

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")
print(df.shape)  # (500, 9)
print(df.columns)  # Index(['conversation_id', 'topic', ...])
```

**Advantages**:
- Efficient columnar storage
- Fast filtering and analysis
- Preserves all metadata
- Easy to load with pandas

### aiperf JSONL Format (multi_turn)

**Description**: Lightweight format with user messages only

**File Size**: ~0.8 MB (500 conversations)

**Schema**:
```json
{"session_id": "uuid", "turns": [{"text": "user message 1"}, {"text": "user message 2"}, ...]}
```

**Usage**:
```bash
aiperf profile \
    --model <your-model> \
    --input-file text/data/multi_turn_text_chat.jsonl \
    --custom-dataset-type multi_turn \
    --streaming --url localhost:8000
```

**Advantages**:
- Lightweight and fast
- aiperf automatically accumulates responses
- Good for prefix cache testing
- Easy to parse

### mooncake_trace JSONL Format

**Description**: Full message arrays per turn (one line per turn)

**File Size**: ~227 MB (500 conversations, gitignored)

**Schema**:
```json
{"session_id": "uuid", "messages": [{"role": "system", "content": "..."}, ...], "output_length": 150}
```

**Usage**:
```bash
aiperf profile \
    --model <your-model> \
    --input-file text/data/multi_turn_text_chat_mooncake.jsonl \
    --custom-dataset-type mooncake_trace \
    --streaming --url localhost:8000
```

**Advantages**:
- Full control over exact prompt
- Pre-canned responses included
- Reproducible benchmarks
- Detailed message history

---

## Data Loading

### Load Parquet with Pandas

```python
import pandas as pd

# Load entire dataset
df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Load specific columns
df = pd.read_parquet(
    "text/data/multi_turn_text_chat.parquet",
    columns=['conversation_id', 'num_turns', 'estimated_tokens']
)

# Load in chunks (for large datasets)
for chunk in pd.read_parquet("text/data/multi_turn_text_chat.parquet", chunksize=100):
    print(f"Processing {len(chunk)} conversations")
```

### Load JSONL

```python
import json

# Load multi_turn format
with open("text/data/multi_turn_text_chat.jsonl") as f:
    for line in f:
        record = json.loads(line)
        session_id = record['session_id']
        turns = record['turns']

# Load mooncake_trace format
with open("text/data/multi_turn_text_chat_mooncake.jsonl") as f:
    for line in f:
        record = json.loads(line)
        messages = record['messages']
        output_length = record['output_length']
```

### Parse JSON Fields

```python
import pandas as pd
import json

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")
row = df.iloc[0]

# Parse messages
messages = json.loads(row['messages'])
for msg in messages:
    print(f"[{msg['role']}] {msg['content']}")

# Parse cumulative lengths
lengths = json.loads(row['cumulative_char_lengths'])
print(f"Context growth: {lengths[0]} -> {lengths[-1]} chars")

# For agentic dataset, parse tool calls
tool_calls = json.loads(row['tool_calls'])
for call in tool_calls:
    print(f"{call['tool']}: {call['result']['status']}")
```

---

## Common Operations

### Filter Dataset

```python
import pandas as pd

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Filter by topic
coding = df[df['topic'] == 'coding_help']

# Filter by turn count
long_conversations = df[df['num_turns'] > 20]

# Filter by token count
large = df[df['estimated_tokens'] > 2000]

# Combine filters
filtered = df[
    (df['topic'] == 'coding_help') & 
    (df['num_turns'] > 10) & 
    (df['estimated_tokens'] > 1000)
]

# Save filtered dataset
filtered.to_parquet("custom_dataset.parquet")
```

### Analyze Statistics

```python
import pandas as pd

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Basic statistics
print(f"Total conversations: {len(df)}")
print(f"Total tokens: {df['estimated_tokens'].sum():,}")
print(f"Avg tokens per conversation: {df['estimated_tokens'].mean():,.0f}")

# Distribution by topic
print(df['topic'].value_counts())

# Percentiles
print(f"50th percentile (median): {df['num_turns'].quantile(0.5)}")
print(f"95th percentile: {df['num_turns'].quantile(0.95)}")
```

### Create Custom Subset

```python
import pandas as pd

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Random sample
sample = df.sample(n=100, random_state=42)

# First N conversations
subset = df.head(50)

# Stratified sample (balanced by topic)
sample = df.groupby('topic', group_keys=False).apply(
    lambda x: x.sample(n=min(10, len(x)), random_state=42)
)

# Save
sample.to_parquet("custom_subset.parquet")
```

### Export to Different Format

```python
import pandas as pd
import json

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Export to CSV
df.to_csv("dataset.csv", index=False)

# Export to JSON
df.to_json("dataset.json", orient='records')

# Export to JSONL (aiperf format)
with open("dataset.jsonl", 'w') as f:
    for idx, row in df.iterrows():
        messages = json.loads(row['messages'])
        turns = [{"text": msg['content']} for msg in messages if msg['role'] == 'user']
        record = {"session_id": row['conversation_id'], "turns": turns}
        f.write(json.dumps(record) + '\n')
```

### Estimate Benchmark Time

```python
import pandas as pd

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

total_tokens = df['estimated_tokens'].sum()
tokens_per_second = 100  # Adjust based on your model

total_seconds = total_tokens / tokens_per_second
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)

print(f"Estimated time: {hours}h {minutes}m")

# With concurrency
for concurrency in [1, 5, 10, 20]:
    concurrent_minutes = (total_seconds / concurrency) / 60
    print(f"With {concurrency} concurrent: {concurrent_minutes:.1f} minutes")
```

---

## Error Handling

### Handle Missing Files

```python
from pathlib import Path

parquet_path = Path("text/data/multi_turn_text_chat.parquet")

if not parquet_path.exists():
    print("Dataset not found. Run: python text/generate.py")
else:
    df = pd.read_parquet(parquet_path)
```

### Handle Large Files

```python
import pandas as pd

# Process in chunks to avoid memory issues
try:
    for chunk in pd.read_parquet("text/data/multi_turn_text_chat.parquet", chunksize=100):
        # Process chunk
        print(f"Processing {len(chunk)} conversations")
except MemoryError:
    print("Dataset too large for memory. Use chunksize parameter.")
```

### Validate Data

```python
import pandas as pd
import json

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Check for missing values
print(df.isnull().sum())

# Validate JSON fields
for idx, row in df.iterrows():
    try:
        messages = json.loads(row['messages'])
        assert len(messages) > 0, "Empty messages"
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in row {idx}: {e}")
```

---

## Performance Tips

1. **Use specific columns**: Load only needed columns to save memory
2. **Use chunksize**: Process large datasets in chunks
3. **Use filters**: Filter early to reduce data size
4. **Use Parquet**: More efficient than CSV or JSON
5. **Use indexes**: Create indexes for frequently filtered columns

```python
# Efficient example
df = pd.read_parquet(
    "text/data/multi_turn_text_chat.parquet",
    columns=['conversation_id', 'num_turns', 'estimated_tokens']
)
filtered = df[df['num_turns'] > 10]
```

---

For more examples, see [EXAMPLES.md](EXAMPLES.md).
