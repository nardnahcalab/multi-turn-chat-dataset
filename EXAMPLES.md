# Examples and Tutorials

Comprehensive examples for working with the multi-turn-chat-dataset. All examples are located in the `examples/` directory and can be run independently.

## Quick Start

```bash
# Setup
source .venv/bin/activate

# Generate datasets (if not already generated)
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

---

## Example 1: Loading and Inspecting Datasets

**File**: `examples/01_load_and_inspect.py`

Learn how to load datasets and explore their structure.

### Basic Usage

```bash
# Inspect text dataset
python examples/01_load_and_inspect.py text

# Inspect agentic dataset
python examples/01_load_and_inspect.py agentic

# Inspect any dataset type
python examples/01_load_and_inspect.py pdf
python examples/01_load_and_inspect.py image
python examples/01_load_and_inspect.py reasoning
```

### What You'll Learn

- How to load Parquet files with pandas
- Dataset structure and schema
- Basic statistics (turns, tokens, characters)
- How to access and parse JSON fields
- Message structure and roles

### Code Example

```python
import pandas as pd
import json

# Load dataset
df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Access a conversation
row = df.iloc[0]
messages = json.loads(row["messages"])

# Print basic info
print(f"Topic: {row['topic']}")
print(f"Turns: {row['num_turns']}")
print(f"Tokens: {row['estimated_tokens']}")

# Iterate through messages
for msg in messages:
    print(f"[{msg['role']}] {msg['content'][:100]}...")
```

### Output Example

```
📊 Dataset Statistics:
   Total conversations: 500
   Columns: ['conversation_id', 'topic', 'num_turns', 'num_messages', ...]
   
📈 Column Information:
   conversation_id        | object         | 500/500 non-null
   topic                  | object         | 500/500 non-null
   num_turns              | int64          | 500/500 non-null
   ...

Sample Conversations (first 3):
[Sample 1]
  Conversation ID: 550e8400-e29b-41d4-a716-446655440000
  Topic: coding_help
  Turns: 8
  Messages: 17
  Characters: 3,245
  Estimated Tokens: 811
```

---

## Example 2: Analyzing Conversation Structure

**File**: `examples/02_analyze_conversations.py`

Analyze how context grows across turns and message patterns.

### Basic Usage

```bash
# Analyze context growth in text dataset
python examples/02_analyze_conversations.py text

# Analyze agentic dataset
python examples/02_analyze_conversations.py agentic

# Compare all datasets
python examples/02_analyze_conversations.py text
python examples/02_analyze_conversations.py pdf
python examples/02_analyze_conversations.py image
python examples/02_analyze_conversations.py reasoning
python examples/02_analyze_conversations.py agentic
```

### What You'll Learn

- How context grows turn-by-turn
- Message role distribution
- Message length patterns
- Turn distribution across dataset
- Dataset comparison

### Code Example

```python
import pandas as pd
import json

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")
row = df.iloc[0]

# Analyze context growth
cumulative_lengths = json.loads(row["cumulative_char_lengths"])

for i, char_count in enumerate(cumulative_lengths):
    token_count = char_count // 4
    print(f"Turn {i+1}: {char_count:,} chars (~{token_count:,} tokens)")

# Analyze message roles
messages = json.loads(row["messages"])
role_counts = {}
for msg in messages:
    role = msg['role']
    role_counts[role] = role_counts.get(role, 0) + 1

for role, count in role_counts.items():
    print(f"{role}: {count} messages")
```

### Output Example

```
📈 Context Growth Analysis
Conversation ID: 550e8400-e29b-41d4-a716-446655440000
Total Turns: 8
Total Characters: 3,245
Estimated Tokens: 811

Turn  Characters      Tokens (est.)   Growth
1     245             61              —
2     512             128             +267 chars
3     1,024           256             +512 chars
4     1,536           384             +512 chars
...

📊 Message Distribution:
  system        :   1 messages
  user          :   8 messages
  assistant     :   8 messages
```

---

## Example 3: Preparing Data for Benchmarking

**File**: `examples/03_prepare_for_benchmarking.py`

Prepare datasets for benchmarking with aiperf and other tools.

### Basic Usage

```bash
# Prepare multi_turn format (lightweight)
python examples/03_prepare_for_benchmarking.py --dataset text --format multi_turn

# Prepare mooncake_trace format (full control)
python examples/03_prepare_for_benchmarking.py --dataset text --format mooncake_trace

# Create a subset for testing
python examples/03_prepare_for_benchmarking.py --dataset text --format subset --num-samples 50

# Filter by turn count
python examples/03_prepare_for_benchmarking.py --dataset text --format filtered --min-turns 5 --max-turns 15

# Estimate benchmark time
python examples/03_prepare_for_benchmarking.py --dataset text --tokens-per-sec 100
```

### What You'll Learn

- Different output formats and their use cases
- How to create custom subsets
- How to filter datasets by criteria
- How to estimate benchmark execution time
- Format conversion

### Code Example: Create Multi-Turn Format

```python
import pandas as pd
import json

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

with open("custom_dataset.jsonl", 'w') as f:
    for idx, row in df.iterrows():
        messages = json.loads(row["messages"])
        
        # Extract user messages only
        turns = []
        for msg in messages:
            if msg['role'] == 'user':
                turns.append({"text": msg['content']})
        
        # Create aiperf record
        record = {
            "session_id": row['conversation_id'],
            "turns": turns
        }
        
        f.write(json.dumps(record) + '\n')
```

### Code Example: Create Filtered Subset

```python
import pandas as pd

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Filter conversations with 5-15 turns
filtered = df[(df['num_turns'] >= 5) & (df['num_turns'] <= 15)]

# Save to new parquet
filtered.to_parquet("text/data/multi_turn_text_chat_filtered.parquet")

print(f"Original: {len(df)} conversations")
print(f"Filtered: {len(filtered)} conversations")
print(f"Total tokens: {filtered['estimated_tokens'].sum():,}")
```

### Output Example

```
📊 Output Format Comparison
================================================================================

MULTI_TURN
  Description: Lightweight, user messages only
  Use Case: Fast benchmarking, prefix cache testing
  Schema: {"session_id": "uuid", "turns": [{"text": "..."}, ...]}
  Size: Small (~0.8 MB for 500 conversations)

MOONCAKE_TRACE
  Description: Full message arrays per turn
  Use Case: Full control, exact prompt reproduction
  Schema: {"session_id": "uuid", "messages": [...], "output_length": 150}
  Size: Large (~227 MB for 500 conversations)

⏱️  Benchmark Time Estimation
Dataset: text
Total conversations: 500
Total tokens: 3,900,000
Inference speed: 100 tokens/sec

⏱️  Estimated Time (single-threaded):
  10h 50m 0s

  With  1 concurrent requests:   650.0 minutes
  With  5 concurrent requests:   130.0 minutes
  With 10 concurrent requests:    65.0 minutes
  With 20 concurrent requests:    32.5 minutes
```

---

## Example 4: Agentic Task Analysis

**File**: `examples/04_agentic_analysis.py`

Deep dive into agentic task dataset analysis.

### Basic Usage

```bash
# Analyze tool usage and task performance
python examples/04_agentic_analysis.py

# Compare two conversations
python examples/04_agentic_analysis.py --compare 0 1

# Export analysis report
python examples/04_agentic_analysis.py --report
```

### What You'll Learn

- Tool usage patterns and success rates
- Task performance by type
- Success metric distributions
- Error patterns and recovery
- Conversation comparison

### Code Example: Tool Usage Analysis

```python
import pandas as pd
import json
from collections import defaultdict

df = pd.read_parquet("agentic/data/multi_turn_agentic_task.parquet")

# Aggregate tool statistics
tool_stats = defaultdict(lambda: {"count": 0, "success": 0, "failure": 0})

for tool_calls_json in df["tool_calls"]:
    tool_calls = json.loads(tool_calls_json)
    for call in tool_calls:
        tool = call['tool']
        status = call['result']['status']
        
        tool_stats[tool]['count'] += 1
        if status == 'success':
            tool_stats[tool]['success'] += 1
        else:
            tool_stats[tool]['failure'] += 1

# Display results
for tool in sorted(tool_stats.keys()):
    stats = tool_stats[tool]
    success_rate = (stats['success'] / stats['count'] * 100)
    print(f"{tool}: {stats['count']} calls, {success_rate:.1f}% success rate")
```

### Code Example: Task Performance Analysis

```python
import pandas as pd

df = pd.read_parquet("agentic/data/multi_turn_agentic_task.parquet")

# Group by task type
for task_type in df['task_type'].unique():
    task_df = df[df['task_type'] == task_type]
    
    print(f"\n{task_type}:")
    print(f"  Conversations: {len(task_df)}")
    print(f"  Avg Turns: {task_df['num_turns'].mean():.1f}")
    print(f"  Avg Tool Calls: {task_df['num_tool_calls'].mean():.1f}")
    print(f"  Avg Success Score: {task_df['success_score'].mean():.2f}")
    print(f"  Error Rate: {(task_df['num_errors'].sum() / task_df['num_tool_calls'].sum() * 100):.1f}%")
```

### Output Example

```
🔧 Tool Usage Analysis
================================================================================

Tool                               Count    Success    Failure    Rate
----------------------------------------------------------------------
query_database                     487      414        73         85.0%
transform_data                     456      389        67         85.3%
validate_schema                     412      351        61         85.2%
export_data                         398      338        60         84.9%
call_api                            445      378        67         84.9%
...
----------------------------------------------------------------------
TOTAL                            4,800    4,080       720         85.0%

📊 Task Performance Analysis
================================================================================

Task Type                Avg Turns    Avg Tools    Avg Errors   Avg Score
----------------------------------------------------------------------
data_processing          7.2          10.1         1.5          0.72
api_integration          7.1          9.8          1.4          0.71
system_troubleshooting   6.9          8.5          1.2          0.73
code_generation          7.3          10.5         1.6          0.70
research_synthesis       7.0          9.2          1.3          0.72
planning_execution       7.2          9.9          1.5          0.71
```

---

## Example 5: Mixed Dataset Generation

**File**: `mixed/generate.py`

Create comprehensive benchmarking datasets by combining multiple source types.

### Basic Usage

```bash
# Generate mixed dataset with default configuration
python mixed/generate.py

# Generate with custom conversation count
python mixed/generate.py --num 1000

# Generate with custom seed for reproducibility
python mixed/generate.py --seed 123

# Generate only specific format
python mixed/generate.py --format parquet
python mixed/generate.py --format aiperf
python mixed/generate.py --format mooncake
```

### What You'll Learn

- How to combine multiple dataset types into one
- Configurable source weights and sampling strategies
- Schema standardization across different dataset types
- Source tracking for analysis
- Custom configuration for mixed datasets

### Configuration

Edit `mixed/config.yaml` to customize:

```yaml
sources:
  - type: "text"
    weight: 0.25
    path: "text/data/multi_turn_text_chat.parquet"
    required: true

  - type: "reasoning"
    weight: 0.30
    path: "reasoning/data/multi_turn_reasoning_chat.parquet"
    required: true

  - type: "agentic"
    weight: 0.20
    path: "agentic/data/multi_turn_agentic_task.parquet"
    required: true

  - type: "pdf"
    weight: 0.15
    path: "pdf/data/multi_turn_pdf_chat.parquet"
    required: false

  - type: "image"
    weight: 0.10
    path: "image/data/multi_turn_image_chat.parquet"
    required: false
```

### Code Example: Analyze Mixed Dataset

```python
import pandas as pd
import json

# Load mixed dataset
df = pd.read_parquet("mixed/data/multi_turn_mixed_chat.parquet")

# Analyze source distribution
source_counts = df['source_dataset'].value_counts()
print("Source Distribution:")
for source, count in source_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {source}: {count} ({percentage:.1f}%)")

# Analyze by source dataset type
for source in df['source_dataset'].unique():
    source_df = df[df['source_dataset'] == source]
    print(f"\n{source}:")
    print(f"  Conversations: {len(source_df)}")
    print(f"  Avg Turns: {source_df['num_turns'].mean():.1f}")
    print(f"  Avg Tokens: {source_df['estimated_tokens'].mean():,.0f}")
```

### Output Example

```
Generating mixed dataset with 500 conversations (seed=42)

Loading source datasets:
  Loading text from text/data/multi_turn_text_chat.parquet
    Loaded 500 conversations from text
  Loading pdf from pdf/data/multi_turn_pdf_chat.parquet
    Loaded 500 conversations from pdf
  Loading image from image/data/multi_turn_image_chat.parquet
    Loaded 500 conversations from image
  Loading reasoning from reasoning/data/multi_turn_reasoning_chat.parquet
    Loaded 500 conversations from reasoning
  Loading agentic from agentic/data/multi_turn_agentic_task.parquet
    Loaded 500 conversations from agentic

Sampling 500 conversations using 'weighted' strategy:
    Sampled 125 from text (weight: 0.25)
    Sampled 100 from pdf (weight: 0.20)
    Sampled 100 from image (weight: 0.20)
    Sampled 100 from reasoning (weight: 0.20)
    Sampled 75 from agentic (weight: 0.15)
  Total sampled: 500 conversations

Source Distribution:
  text: 125 (25.0%)
  reasoning: 100 (20.0%)
  pdf: 100 (20.0%)
  image: 100 (20.0%)
  agentic: 75 (15.0%)
```

---

## Example 5: Payload Score Comparison

**File**: `examples/06_payload_score_comparison.py`

Compare computational effort across datasets and create payload-balanced subsets.

### Basic Usage

```bash
# Compare payload scores across datasets
python examples/06_payload_score_comparison.py --datasets text reasoning agentic

# Compare with sample size limit
python examples/06_payload_score_comparison.py --datasets text reasoning --sample-size 100

# Create payload-balanced subsets
python examples/06_payload_score_comparison.py --datasets text reasoning agentic --target-payload 0.2

# Compare all available datasets
python examples/06_payload_score_comparison.py --datasets text pdf image reasoning agentic random repeat
```

### What You'll Learn

- How to compute and compare payload scores
- Understanding prefill vs decode effort
- Creating datasets with identical computational effort
- Binary search for target payload matching
- Fair benchmarking through payload balancing

### Code Example: Compute Payload Scores

```python
import pandas as pd
from dataset_profile import compute_payload_score

# Load datasets
text_df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")
reasoning_df = pd.read_parquet("reasoning/data/multi_turn_reasoning_chat.parquet")

# Compute payload scores
text_payload = compute_payload_score(text_df, "text")
reasoning_payload = compute_payload_score(reasoning_df, "reasoning")

print("Text Dataset:")
print(f"  Prefill Score:  {text_payload['prefill_score']:.4f}")
print(f"  Decode Score:   {text_payload['decode_score']:.4f}")
print(f"  Total Payload:  {text_payload['total_payload_score']:.4f}")

print("\nReasoning Dataset:")
print(f"  Prefill Score:  {reasoning_payload['prefill_score']:.4f}")
print(f"  Decode Score:   {reasoning_payload['decode_score']:.4f}")
print(f"  Total Payload:  {reasoning_payload['total_payload_score']:.4f}")

# Compare breakdown factors
print("\nPrefill Factors Comparison:")
for factor in text_payload['prefill_breakdown']['raw'].keys():
    text_val = text_payload['prefill_breakdown']['raw'][factor]
    reasoning_val = reasoning_payload['prefill_breakdown']['raw'][factor]
    print(f"  {factor}: text={text_val:.2f}, reasoning={reasoning_val:.2f}")
```

### Code Example: Access Payload from Manifest

```python
import json

# Load manifest
with open("text/data/multi_turn_text_chat_manifest.json", "r") as f:
    manifest = json.load(f)

# Access payload scores
payload = manifest["payload_scores"]
print(f"Prefill Score:  {payload['prefill_score']:.4f}")
print(f"Decode Score:   {payload['decode_score']:.4f}")
print(f"Total Payload:  {payload['total_payload_score']:.4f}")

# Access detailed breakdown
print("\nPrefill Breakdown:")
for factor, value in payload['prefill_breakdown']['normalized'].items():
    print(f"  {factor}: {value:.4f}")

print("\nDecode Breakdown:")
for factor, value in payload['decode_breakdown']['normalized'].items():
    print(f"  {factor}: {value:.4f}")
```

### Output Example

```
PAYLOAD SCORE COMPARISON
================================================================================
Dataset              Conversations   Prefill      Decode       Total       
--------------------------------------------------------------------------------
reasoning            500             0.6104       0.4805       0.5455      
pdf                  500             0.5144       0.2919       0.4032      
text                 500             0.2945       0.1396       0.2171      
agentic              500             0.1132       0.2423       0.1777      
================================================================================

Creating payload-balanced datasets with target payload: 0.2000
================================================================================

Processing text...
Finding subset for text with target payload 0.2000...
  Found subset of 340 conversations
  Actual payload: 0.2003 (target: 0.2000)
  Difference: 0.0003
  ✓ Within tolerance (0.0500)
  Saved to balanced_datasets/balanced_text_payload.parquet

Processing reasoning...
Finding subset for reasoning with target payload 0.2000...
  Found subset of 164 conversations
  Actual payload: 0.2015 (target: 0.2000)
  Difference: 0.0015
  ✓ Within tolerance (0.0500)
  Saved to balanced_datasets/balanced_reasoning_payload.parquet
```

### Use Cases

**Fair Benchmarking**: Compare models across different dataset types with identical computational effort

**Resource Planning**: Estimate infrastructure requirements based on payload scores

**Dataset Selection**: Choose datasets that match your computational budget

**Performance Analysis**: Understand which factors contribute most to computational cost

---

## Advanced Examples

### Custom Dataset Generation

```python
import pandas as pd
import json

# Load base dataset
df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Filter by criteria
long_conversations = df[df['num_turns'] > 20]
coding_help = df[df['topic'] == 'coding_help']

# Combine filters
filtered = df[(df['num_turns'] > 10) & (df['estimated_tokens'] > 1000)]

# Save custom dataset
filtered.to_parquet("custom_dataset.parquet")
```

### Analyzing Prefix Cache Efficiency

```python
import pandas as pd
import json

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Analyze prefix reuse
for idx, row in df.iterrows():
    cumulative_lengths = json.loads(row["cumulative_char_lengths"])
    
    # Calculate prefix growth rate
    growth_rates = []
    for i in range(1, len(cumulative_lengths)):
        growth = cumulative_lengths[i] - cumulative_lengths[i-1]
        growth_rates.append(growth)
    
    avg_growth = sum(growth_rates) / len(growth_rates)
    max_growth = max(growth_rates)
    
    print(f"Conversation {idx}: avg growth {avg_growth:.0f}, max growth {max_growth}")
```

### Benchmarking with aiperf

```bash
# Using multi_turn format (lightweight)
aiperf profile \
    --model meta-llama/Llama-2-7b-chat \
    --endpoint-type chat \
    --input-file text/data/multi_turn_text_chat.jsonl \
    --custom-dataset-type multi_turn \
    --streaming \
    --url localhost:8000 \
    --concurrency 10 \
    --num-requests 500

# Using mooncake_trace format (full control)
aiperf profile \
    --model meta-llama/Llama-2-7b-chat \
    --endpoint-type chat \
    --input-file text/data/multi_turn_text_chat_mooncake.jsonl \
    --custom-dataset-type mooncake_trace \
    --streaming \
    --url localhost:8000 \
    --concurrency 10
```

### Analyzing Token Distribution

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(df['estimated_tokens'], bins=50, edgecolor='black')
plt.xlabel('Tokens')
plt.ylabel('Frequency')
plt.title('Token Distribution')
plt.savefig('token_distribution.png')

# Print statistics
print(f"Token Statistics:")
print(f"  Min: {df['estimated_tokens'].min():,}")
print(f"  Max: {df['estimated_tokens'].max():,}")
print(f"  Mean: {df['estimated_tokens'].mean():,.0f}")
print(f"  Median: {df['estimated_tokens'].median():,.0f}")
print(f"  Total: {df['estimated_tokens'].sum():,}")
```

---

## Troubleshooting

### "Dataset not found" Error

```bash
# Generate the dataset first
python text/generate.py
python agentic/generate.py
python pdf/generate.py
python image/generate.py
python reasoning/generate.py
```

### Out of Memory

```python
# Process in chunks instead of loading entire dataset
import pandas as pd

for chunk in pd.read_parquet("text/data/multi_turn_text_chat.parquet", chunksize=100):
    # Process chunk
    print(f"Processing {len(chunk)} conversations")
```

### Slow Performance

```python
# Use specific columns only
df = pd.read_parquet(
    "text/data/multi_turn_text_chat.parquet",
    columns=['conversation_id', 'num_turns', 'estimated_tokens']
)
```

---

## Next Steps

1. **Run the examples** to understand the dataset structure
2. **Modify the examples** for your specific use case
3. **Generate custom subsets** for your benchmarks
4. **Integrate with aiperf** for performance testing
5. **Extend the generators** to create custom datasets

For more information, see:
- [README.md](README.md) - User guide and quick start
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical reference
- [CHECKPOINT.md](CHECKPOINT.md) - Implementation details
