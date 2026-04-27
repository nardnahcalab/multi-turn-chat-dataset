# FAQ and Troubleshooting Guide

Common questions and solutions for working with the multi-turn-chat-dataset.

## Table of Contents

- [Setup and Installation](#setup-and-installation)
- [Dataset Generation](#dataset-generation)
- [Data Loading and Analysis](#data-loading-and-analysis)
- [Benchmarking](#benchmarking)
- [Performance and Optimization](#performance-and-optimization)
- [Troubleshooting](#troubleshooting)
- [Advanced Questions](#advanced-questions)

---

## Setup and Installation

### Q: How do I set up the project?

**A:** Follow these steps:

```bash
# Clone the repository
git clone <repo-url>
cd multi-turn-chat-dataset

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate datasets
python text/generate.py
python agentic/generate.py
python pdf/generate.py
python image/generate.py
python reasoning/generate.py
```

### Q: What are the system requirements?

**A:** 
- Python 3.8+
- ~5 GB disk space for all generated datasets
- ~2 GB RAM for loading and analyzing datasets
- Internet connection for PDF and image datasets (to fetch from arXiv and Wikipedia)

### Q: Can I use this on Windows/Mac/Linux?

**A:** Yes, the project is cross-platform. Use the appropriate activation command for your shell:
- **Linux/Mac**: `source .venv/bin/activate`
- **Windows (PowerShell)**: `.venv\Scripts\Activate.ps1`
- **Windows (cmd)**: `.venv\Scripts\activate.bat`

### Q: Do I need to install all dependencies?

**A:** No, you can install only what you need:

```bash
# Minimal (text dataset only)
pip install pandas pyarrow pyyaml

# With PDF support
pip install pandas pyarrow pyyaml arxiv

# With image support
pip install pandas pyarrow pyyaml requests

# All datasets
pip install -r requirements.txt
```

---

## Dataset Generation

### Q: How long does it take to generate datasets?

**A:**
- **Text**: ~30 seconds (500 conversations)
- **Agentic**: ~1 minute (500 conversations)
- **PDF**: ~2-3 minutes (includes arXiv API calls)
- **Image**: ~2-3 minutes (includes Wikipedia API calls)
- **Reasoning**: ~1-2 minutes (500 conversations)

### Q: Can I generate a custom number of conversations?

**A:** Yes, use the `--num` flag:

```bash
python text/generate.py --num 1000
python agentic/generate.py --num 100
python pdf/generate.py --num 250
```

### Q: How do I make generation reproducible?

**A:** Use the `--seed` flag:

```bash
python text/generate.py --num 500 --seed 42
python agentic/generate.py --num 500 --seed 123
```

The same seed will always produce the same dataset.

### Q: Can I generate only specific formats?

**A:** Yes, use the `--format` flag:

```bash
# Generate only Parquet
python text/generate.py --format parquet

# Generate only aiperf JSONL
python text/generate.py --format aiperf

# Generate only mooncake JSONL
python text/generate.py --format mooncake
```

### Q: What if I get "arXiv API error" when generating PDF dataset?

**A:** The PDF generator fetches papers from arXiv. If you get API errors:

```bash
# Skip fetching and use cached papers
python pdf/generate.py --skip-fetch

# Or manually clear cache and retry
rm pdf/data/arxiv_papers.json
python pdf/generate.py
```

### Q: What if I get "Wikipedia API error" when generating image dataset?

**A:** Similar to PDF, you can skip fetching:

```bash
# Skip fetching and use cached images
python image/generate.py --skip-fetch

# Or manually clear cache and retry
rm image/data/wikipedia_images.json
python image/generate.py
```

### Q: Can I customize the dataset configuration?

**A:** Yes, edit the configuration files:

```bash
# Edit text dataset config
vim text/config.yaml

# Edit agentic dataset config
vim agentic/config.yaml

# Then regenerate
python text/generate.py
python agentic/generate.py
```

See [API_REFERENCE.md](API_REFERENCE.md) for configuration options.

---

## Data Loading and Analysis

### Q: How do I load a dataset?

**A:**

```python
import pandas as pd

# Load Parquet
df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Load JSONL
import json
with open("text/data/multi_turn_text_chat.jsonl") as f:
    for line in f:
        record = json.loads(line)
```

### Q: How do I access the messages in a conversation?

**A:**

```python
import pandas as pd
import json

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")
row = df.iloc[0]

# Parse messages
messages = json.loads(row['messages'])

# Iterate through messages
for msg in messages:
    print(f"[{msg['role']}] {msg['content']}")
```

### Q: How do I analyze context growth?

**A:**

```python
import json

row = df.iloc[0]
cumulative_lengths = json.loads(row['cumulative_char_lengths'])

for i, char_count in enumerate(cumulative_lengths):
    token_count = char_count // 4  # Rough estimation
    print(f"Turn {i+1}: {char_count:,} chars (~{token_count:,} tokens)")
```

### Q: How do I filter the dataset?

**A:**

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
filtered = df[(df['topic'] == 'coding_help') & (df['num_turns'] > 10)]
```

### Q: How do I save a filtered dataset?

**A:**

```python
# Save as Parquet
filtered.to_parquet("custom_dataset.parquet")

# Save as CSV
filtered.to_csv("custom_dataset.csv", index=False)

# Save as JSONL
import json
with open("custom_dataset.jsonl", 'w') as f:
    for idx, row in filtered.iterrows():
        f.write(json.dumps(row.to_dict()) + '\n')
```

### Q: How do I analyze agentic task performance?

**A:**

```python
import pandas as pd
import json

df = pd.read_parquet("agentic/data/multi_turn_agentic_task.parquet")

# Group by task type
for task_type in df['task_type'].unique():
    task_df = df[df['task_type'] == task_type]
    print(f"{task_type}:")
    print(f"  Avg Success Score: {task_df['success_score'].mean():.2f}")
    print(f"  Avg Tool Calls: {task_df['num_tool_calls'].mean():.1f}")
    print(f"  Error Rate: {(task_df['num_errors'].sum() / task_df['num_tool_calls'].sum() * 100):.1f}%")
```

---

## Benchmarking

### Q: How do I use the datasets with aiperf?

**A:**

```bash
# Using multi_turn format (lightweight)
aiperf profile \
    --model meta-llama/Llama-2-7b-chat \
    --endpoint-type chat \
    --input-file text/data/multi_turn_text_chat.jsonl \
    --custom-dataset-type multi_turn \
    --streaming \
    --url localhost:8000 \
    --concurrency 10

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

### Q: What's the difference between multi_turn and mooncake_trace formats?

**A:**

| Aspect | multi_turn | mooncake_trace |
|--------|-----------|-----------------|
| **Size** | Small (~0.8 MB) | Large (~227 MB) |
| **Content** | User messages only | Full message history per turn |
| **Use Case** | Fast benchmarking | Full control, reproducibility |
| **aiperf Behavior** | Auto-accumulates responses | Pre-canned responses |
| **Prefix Cache Testing** | ✓ Good | ✓ Excellent |

### Q: How do I estimate benchmark execution time?

**A:**

```python
import pandas as pd

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")
total_tokens = df['estimated_tokens'].sum()

# Estimate based on inference speed
tokens_per_second = 100  # Adjust based on your model
total_seconds = total_tokens / tokens_per_second

# Convert to hours/minutes
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
print(f"Estimated time: {hours}h {minutes}m")

# With concurrency
for concurrency in [1, 5, 10, 20]:
    concurrent_minutes = (total_seconds / concurrency) / 60
    print(f"With {concurrency} concurrent: {concurrent_minutes:.1f} minutes")
```

Or use the provided example:

```bash
python examples/03_prepare_for_benchmarking.py --dataset text --tokens-per-sec 100
```

### Q: Can I create a custom subset for testing?

**A:** Yes, use the provided example:

```bash
# Create subset of 100 conversations
python examples/03_prepare_for_benchmarking.py --dataset text --format subset --num-samples 100

# Create filtered subset (5-15 turns)
python examples/03_prepare_for_benchmarking.py --dataset text --format filtered --min-turns 5 --max-turns 15
```

---

## Performance and Optimization

### Q: The dataset is too large for my memory. What should I do?

**A:** Process in chunks:

```python
import pandas as pd

# Load in chunks of 100 conversations
for chunk in pd.read_parquet("text/data/multi_turn_text_chat.parquet", chunksize=100):
    # Process chunk
    print(f"Processing {len(chunk)} conversations")
    # Your analysis here
```

### Q: How can I speed up data loading?

**A:** Load only the columns you need:

```python
import pandas as pd

# Instead of loading all columns
df = pd.read_parquet(
    "text/data/multi_turn_text_chat.parquet",
    columns=['conversation_id', 'num_turns', 'estimated_tokens']
)
```

### Q: How can I speed up filtering?

**A:** Filter early and use efficient operations:

```python
import pandas as pd

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Filter early (before other operations)
filtered = df[df['num_turns'] > 10]

# Use vectorized operations (fast)
means = filtered['estimated_tokens'].mean()

# Avoid loops (slow)
# for idx, row in filtered.iterrows():  # Slow!
#     ...
```

### Q: How can I reduce disk space usage?

**A:** The mooncake JSONL files are large. You can delete them:

```bash
# Delete mooncake files (they're gitignored anyway)
rm text/data/multi_turn_text_chat_mooncake.jsonl
rm agentic/data/multi_turn_agentic_task_mooncake.jsonl
rm pdf/data/multi_turn_pdf_chat_mooncake.jsonl
rm image/data/multi_turn_image_chat_mooncake.jsonl
rm reasoning/data/multi_turn_reasoning_chat_mooncake.jsonl
```

---

## Troubleshooting

### Q: "ModuleNotFoundError: No module named 'pandas'"

**A:** Install dependencies:

```bash
pip install -r requirements.txt
# Or
pip install pandas pyarrow pyyaml
```

### Q: "FileNotFoundError: [Errno 2] No such file or directory: 'text/data/multi_turn_text_chat.parquet'"

**A:** Generate the dataset first:

```bash
python text/generate.py
```

### Q: "JSONDecodeError: Expecting value: line 1 column 1"

**A:** The JSONL file might be corrupted. Regenerate it:

```bash
python text/generate.py --format aiperf
```

### Q: "MemoryError: Unable to allocate X GB"

**A:** Use chunked processing:

```python
for chunk in pd.read_parquet("text/data/multi_turn_text_chat.parquet", chunksize=50):
    # Process chunk
    pass
```

### Q: "arXiv API returned 429 Too Many Requests"

**A:** The PDF generator is hitting rate limits. Use cached data:

```bash
python pdf/generate.py --skip-fetch
```

### Q: "Wikipedia API returned 429 Too Many Requests"

**A:** Similar to arXiv, use cached data:

```bash
python image/generate.py --skip-fetch
```

### Q: "YAML parsing error"

**A:** Check the YAML syntax in config files:

```bash
# Validate YAML
python -c "import yaml; yaml.safe_load(open('text/config.yaml'))"
```

### Q: "Seed doesn't produce the same dataset"

**A:** Make sure you're using the exact same seed and number of conversations:

```bash
# This will produce the same dataset
python text/generate.py --num 500 --seed 42
python text/generate.py --num 500 --seed 42

# This will NOT produce the same dataset (different num)
python text/generate.py --num 500 --seed 42
python text/generate.py --num 1000 --seed 42
```

---

## Advanced Questions

### Q: Can I extend the dataset with custom task types?

**A:** Yes, edit the configuration and generator:

1. Add task type to `agentic/config.yaml`
2. Add template to `agentic/generate.py`
3. Regenerate: `python agentic/generate.py`

See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

### Q: Can I add new tools to the agentic dataset?

**A:** Yes, edit `agentic/config.yaml`:

```yaml
tools:
  my_custom_tool:
    category: "data_processing"
    parameters:
      - name: param1
        type: string
    error_modes: [timeout, error]
```

Then update `agentic/generate.py` to use the new tool.

### Q: Can I integrate this with my own LLM framework?

**A:** Yes, load the data and use it:

```python
import pandas as pd
import json

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

for idx, row in df.iterrows():
    messages = json.loads(row['messages'])
    
    # Use with your framework
    response = your_llm.generate(messages)
    # Your logic here
```

### Q: Can I use this for fine-tuning?

**A:** Yes, export to your preferred format:

```python
import pandas as pd
import json

df = pd.read_parquet("text/data/multi_turn_text_chat.parquet")

# Export as JSONL for fine-tuning
with open("finetune_data.jsonl", 'w') as f:
    for idx, row in df.iterrows():
        messages = json.loads(row['messages'])
        f.write(json.dumps({"messages": messages}) + '\n')
```

### Q: How do I contribute improvements?

**A:** 
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See the repository's CONTRIBUTING.md for details.

### Q: Where can I get help?

**A:**
1. Check this FAQ first
2. Read [EXAMPLES.md](EXAMPLES.md) for usage examples
3. Read [API_REFERENCE.md](API_REFERENCE.md) for technical details
4. Check [ARCHITECTURE.md](ARCHITECTURE.md) for design details
5. Open an issue on GitHub

---

## Additional Resources

- [README.md](README.md) - User guide and quick start
- [EXAMPLES.md](EXAMPLES.md) - Comprehensive examples
- [API_REFERENCE.md](API_REFERENCE.md) - Technical API reference
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design and architecture
- [CHECKPOINT.md](CHECKPOINT.md) - Implementation details

---

**Last Updated**: April 2026
