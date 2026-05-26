# Examples Directory

Practical examples for working with the multi-turn-chat-dataset. Each example is self-contained and can be run independently.

## Quick Start

```bash
# Setup
cd /path/to/multi-turn-chat-dataset
source .venv/bin/activate

# Generate datasets (if not already done)
python text/generate.py
python agentic/generate.py

# Run examples
python examples/01_load_and_inspect.py text
python examples/02_analyze_conversations.py text
python examples/03_prepare_for_benchmarking.py --dataset text
python examples/04_agentic_analysis.py

# New examples for mixed datasets and payload scoring
python mixed/generate.py --num 100
python examples/06_payload_score_comparison.py --datasets text reasoning agentic
```

## Examples Overview

### 1. Load and Inspect Datasets (`01_load_and_inspect.py`)

Learn how to load and explore dataset structure.

**Features**:
- Load Parquet files with pandas
- Display dataset statistics
- Inspect message structure
- Analyze column information
- Compare datasets

**Usage**:
```bash
python examples/01_load_and_inspect.py text
python examples/01_load_and_inspect.py agentic
python examples/01_load_and_inspect.py pdf
```

**Key Functions**:
- `inspect_dataset()` - Load and display dataset info
- `analyze_statistics()` - Detailed statistical analysis

**Output**:
- Dataset shape and columns
- Sample conversations
- Statistical summaries

---

### 2. Analyze Conversations (`02_analyze_conversations.py`)

Analyze conversation structure and context growth patterns.

**Features**:
- Track context growth across turns
- Analyze message patterns
- Compare datasets
- Agentic-specific analysis
- Tool call patterns

**Usage**:
```bash
python examples/02_analyze_conversations.py text
python examples/02_analyze_conversations.py agentic
```

**Key Functions**:
- `analyze_context_growth()` - Track context growth
- `analyze_message_patterns()` - Message role distribution
- `analyze_agentic_specific()` - Tool and task analysis
- `compare_datasets()` - Compare all datasets

**Output**:
- Turn-by-turn context growth
- Message role distribution
- Tool usage statistics
- Cross-dataset comparison

---

### 3. Prepare for Benchmarking (`03_prepare_for_benchmarking.py`)

Prepare datasets for benchmarking with aiperf and other tools.

**Features**:
- Convert between output formats
- Create custom subsets
- Filter by criteria
- Estimate benchmark time
- Format comparison

**Usage**:
```bash
# Prepare multi_turn format
python examples/03_prepare_for_benchmarking.py --dataset text --format multi_turn

# Prepare mooncake_trace format
python examples/03_prepare_for_benchmarking.py --dataset text --format mooncake_trace

# Create subset
python examples/03_prepare_for_benchmarking.py --dataset text --format subset --num-samples 100

# Filter by turn count
python examples/03_prepare_for_benchmarking.py --dataset text --format filtered --min-turns 5 --max-turns 15

# Estimate time
python examples/03_prepare_for_benchmarking.py --dataset text --tokens-per-sec 100
```

**Key Functions**:
- `prepare_aiperf_multi_turn()` - Lightweight format
- `prepare_for_streaming()` - Full control format
- `prepare_custom_subset()` - Create subsets
- `prepare_by_criteria()` - Filter datasets
- `estimate_benchmark_time()` - Time estimation

**Output**:
- Custom JSONL files
- Filtered Parquet files
- Benchmark time estimates

---

### 4. Agentic Task Analysis (`04_agentic_analysis.py`)

Deep analysis of agentic task dataset.

**Features**:
- Tool usage patterns
- Task performance metrics
- Success score analysis
- Error pattern analysis
- Conversation comparison

**Usage**:
```bash
# Full analysis
python examples/04_agentic_analysis.py

# Compare conversations
python examples/04_agentic_analysis.py --compare 0 1

# Export report
python examples/04_agentic_analysis.py --report
```

**Key Functions**:
- `analyze_tool_usage()` - Tool statistics
- `analyze_task_performance()` - Task metrics
- `analyze_success_metrics()` - Score distribution
- `analyze_error_patterns()` - Error analysis
- `compare_conversations()` - Side-by-side comparison
- `export_analysis_report()` - Generate report

**Output**:
- Tool usage statistics
- Task performance metrics
- Success score distribution
- Error patterns
- Analysis report

---

### 5. Payload Score Comparison (`06_payload_score_comparison.py`)

Compare computational effort across datasets and create payload-balanced subsets.

**Features**:
- Compute multi-dimensional payload scores
- Compare prefill vs decode effort
- Create payload-balanced datasets
- Binary search for target matching
- Fair benchmarking preparation

**Usage**:
```bash
# Compare payload scores
python examples/06_payload_score_comparison.py --datasets text reasoning agentic

# Create balanced subsets
python examples/06_payload_score_comparison.py --datasets text reasoning --target-payload 0.2

# Compare all datasets
python examples/06_payload_score_comparison.py --datasets text pdf image reasoning agentic
```

**Key Functions**:
- `compute_payload_score()` - Calculate payload scores
- `compare_payload_scores()` - Compare across datasets
- `find_payload_balanced_subset()` - Create balanced subsets

**Output**:
- Payload score comparison tables
- Balanced dataset subsets
- Computational effort analysis

---

## Common Workflows

### Workflow 1: Quick Dataset Inspection

```bash
# Load and inspect text dataset
python examples/01_load_and_inspect.py text

# Analyze conversations
python examples/02_analyze_conversations.py text
```

### Workflow 2: Prepare for Benchmarking

```bash
# Create a small subset for testing
python examples/03_prepare_for_benchmarking.py \
    --dataset text \
    --format subset \
    --num-samples 50

# Estimate benchmark time
python examples/03_prepare_for_benchmarking.py \
    --dataset text \
    --tokens-per-sec 100

# Prepare for aiperf
python examples/03_prepare_for_benchmarking.py \
    --dataset text \
    --format multi_turn
```

### Workflow 3: Analyze Agentic Tasks

```bash
# Full agentic analysis
python examples/04_agentic_analysis.py

# Compare two conversations
python examples/04_agentic_analysis.py --compare 0 5

# Export report
python examples/04_agentic_analysis.py --report
```

### Workflow 4: Mixed Dataset Generation

```bash
# Generate mixed dataset combining multiple sources
python mixed/generate.py --num 500

# Compare payload scores across datasets
python examples/06_payload_score_comparison.py --datasets text reasoning agentic

# Create payload-balanced subsets for fair benchmarking
python examples/06_payload_score_comparison.py --datasets text reasoning agentic --target-payload 0.2
```

### Workflow 5: Custom Analysis

```python
# Create custom analysis script
import pandas as pd
import json

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Your custom analysis
for idx, row in df.iterrows():
    messages = json.loads(row['messages'])
    # Your logic here
```

---

## Code Snippets

### Load a Dataset

```python
import pandas as pd
import json

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")
row = df.iloc[0]
messages = json.loads(row['messages'])
```

### Filter Dataset

```python
# By topic
coding = df[df['topic'] == 'coding_help']

# By turn count
long = df[df['num_turns'] > 20]

# Combine
filtered = df[(df['topic'] == 'coding_help') & (df['num_turns'] > 10)]
```

### Analyze Context Growth

```python
cumulative_lengths = json.loads(row['cumulative_char_lengths'])
for i, length in enumerate(cumulative_lengths):
    print(f"Turn {i+1}: {length:,} chars")
```

### Analyze Tool Calls (Agentic)

```python
tool_calls = json.loads(row['tool_calls'])
for call in tool_calls:
    print(f"{call['tool']}: {call['result']['status']}")
```

### Create Subset

```python
subset = df.head(100)
subset.to_parquet("custom_subset.parquet")
```

### Compute Payload Score

```python
from dataset_profile import compute_payload_score

payload = compute_payload_score(df, "text")
print(f"Prefill Score:  {payload['prefill_score']:.4f}")
print(f"Decode Score:   {payload['decode_score']:.4f}")
print(f"Total Payload:  {payload['total_payload_score']:.4f}")
```

### Load Payload from Manifest

```python
import json

with open("text/data/multi_turn_text_chat_manifest.json", "r") as f:
    manifest = json.load(f)

payload = manifest["payload_scores"]
```

---

## Requirements

- Python 3.8+
- pandas >= 2.0
- pyarrow >= 14.0
- pyyaml >= 6.0

Install with:
```bash
pip install -r ../requirements.txt
```

---

## Tips and Tricks

1. **Use specific columns**: Load only columns you need
   ```python
   df = pd.read_parquet("text/data/multi_turn_text_chat.parquet",
                        columns=['conversation_id', 'num_turns'])
   ```

2. **Process in chunks**: For large datasets
   ```python
   for chunk in pd.read_parquet("text/data/multi_turn_text_chat.parquet", chunksize=100):
       # Process chunk
   ```

3. **Use filters early**: Filter before other operations
   ```python
   filtered = df[df['num_turns'] > 10]
   # Then analyze filtered
   ```

4. **Vectorize operations**: Use pandas methods instead of loops
   ```python
   # Fast
   means = df['estimated_tokens'].mean()
   
   # Slow
   # means = sum(df['estimated_tokens']) / len(df)
   ```

5. **Cache results**: Save intermediate results
   ```python
   filtered.to_parquet("cached_result.parquet")
   ```

---

## Troubleshooting

### "Dataset not found" Error

Generate the dataset first:
```bash
python text/generate.py
python agentic/generate.py
```

### Out of Memory

Use chunked processing:
```python
for chunk in pd.read_parquet("text/data/multi_turn_text_chat.parquet", chunksize=50):
    # Process chunk
```

### Slow Performance

Load only needed columns:
```python
df = pd.read_parquet("text/data/multi_turn_text_chat.parquet",
                     columns=['conversation_id', 'num_turns'])
```

---

## Next Steps

1. **Run the examples** to understand the data
2. **Modify the examples** for your use case
3. **Create custom subsets** for your benchmarks
4. **Integrate with aiperf** for performance testing
5. **Extend the examples** with your own analysis

---

## Documentation

- [EXAMPLES.md](../EXAMPLES.md) - Detailed tutorials
- [API_REFERENCE.md](../API_REFERENCE.md) - Technical API reference
- [FAQ.md](../FAQ.md) - Common questions and troubleshooting
- [README.md](../README.md) - Project overview
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System design

---

**Happy analyzing!** 🚀
