#!/usr/bin/env python3
"""
Example 3: Preparing Data for Benchmarking

This example shows how to prepare datasets for benchmarking with aiperf,
including format conversion, filtering, and custom dataset creation.
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional


# Map dataset type to its parquet filename
PARQUET_FILES = {
    "text": "text/data/multi_turn_text_chat.parquet",
    "pdf": "pdf/data/multi_turn_pdf_chat.parquet",
    "image": "image/data/multi_turn_image_chat.parquet",
    "reasoning": "reasoning/data/multi_turn_reasoning_chat.parquet",
    "agentic": "agentic/data/multi_turn_agentic_task.parquet",
    "random": "random/data/multi_turn_random_chat.parquet",
    "repeat": "repeat/data/multi_turn_repeat_chat.parquet",
}


def get_parquet_path(dataset_type: str) -> Path:
    """Return the correct parquet path for a given dataset type."""
    return Path(PARQUET_FILES.get(dataset_type, f"{dataset_type}/data/multi_turn_{dataset_type}_chat.parquet"))


def prepare_aiperf_multi_turn(dataset_type: str = "text", output_path: Optional[str] = None):
    """
    Prepare data in aiperf multi_turn format (lightweight, user messages only).
    
    Args:
        dataset_type: One of 'text', 'pdf', 'image', 'reasoning', 'agentic'
        output_path: Custom output path (default: {dataset_type}/data/multi_turn_{dataset_type}_chat.jsonl)
    """
    parquet_path = get_parquet_path(dataset_type)
    
    if not parquet_path.exists():
        print(f"❌ Dataset not found: {parquet_path}")
        return
    
    if output_path is None:
        output_path = str(parquet_path).replace('.parquet', '.jsonl')
    
    df = pd.read_parquet(parquet_path)
    
    print(f"\n📝 Preparing aiperf multi_turn format")
    print(f"{'='*80}")
    print(f"Input: {parquet_path}")
    print(f"Output: {output_path}")
    print(f"Conversations: {len(df)}")
    
    # Convert to aiperf format
    with open(output_path, 'w') as f:
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
    
    print(f"✅ Created {output_path}")
    print(f"   Total lines: {len(df)}")


def prepare_custom_subset(dataset_type: str = "text", num_samples: int = 100, 
                         output_path: Optional[str] = None):
    """
    Create a custom subset of the dataset for testing.
    
    Args:
        dataset_type: One of 'text', 'pdf', 'image', 'reasoning', 'agentic'
        num_samples: Number of samples to include
        output_path: Custom output path
    """
    parquet_path = get_parquet_path(dataset_type)
    
    if not parquet_path.exists():
        print(f"❌ Dataset not found: {parquet_path}")
        return
    
    if output_path is None:
        output_path = str(parquet_path).replace('.parquet', f'_subset_{num_samples}.parquet')
    
    df = pd.read_parquet(parquet_path)
    subset = df.head(num_samples)
    
    print(f"\n🔍 Creating custom subset")
    print(f"{'='*80}")
    print(f"Original size: {len(df)} conversations")
    print(f"Subset size: {len(subset)} conversations")
    print(f"Output: {output_path}")
    
    subset.to_parquet(output_path)
    
    print(f"✅ Created {output_path}")
    print(f"   Total tokens: {subset['estimated_tokens'].sum():,}")


def prepare_by_criteria(dataset_type: str = "text", min_turns: int = 5, 
                       max_turns: int = 20, output_path: Optional[str] = None):
    """
    Create a filtered dataset based on criteria.
    
    Args:
        dataset_type: One of 'text', 'pdf', 'image', 'reasoning', 'agentic'
        min_turns: Minimum number of turns
        max_turns: Maximum number of turns
        output_path: Custom output path
    """
    parquet_path = get_parquet_path(dataset_type)
    
    if not parquet_path.exists():
        print(f"❌ Dataset not found: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    filtered = df[(df['num_turns'] >= min_turns) & (df['num_turns'] <= max_turns)]
    
    if output_path is None:
        output_path = str(parquet_path).replace('.parquet', f'_filtered_{min_turns}_{max_turns}.parquet')
    
    print(f"\n🔍 Creating filtered dataset")
    print(f"{'='*80}")
    print(f"Original size: {len(df)} conversations")
    print(f"Filtered size: {len(filtered)} conversations")
    print(f"Criteria: {min_turns} <= turns <= {max_turns}")
    print(f"Output: {output_path}")
    
    filtered.to_parquet(output_path)
    
    print(f"✅ Created {output_path}")
    print(f"   Total tokens: {filtered['estimated_tokens'].sum():,}")
    print(f"   Avg turns: {filtered['num_turns'].mean():.1f}")


def prepare_for_streaming(dataset_type: str = "text", output_path: Optional[str] = None):
    """
    Prepare data for streaming benchmarks (mooncake_trace format).
    
    This creates one line per turn with the full message history up to that point.
    """
    parquet_path = get_parquet_path(dataset_type)
    
    if not parquet_path.exists():
        print(f"❌ Dataset not found: {parquet_path}")
        return
    
    if output_path is None:
        output_path = str(parquet_path).replace('.parquet', '_streaming.jsonl')
    
    df = pd.read_parquet(parquet_path)
    
    print(f"\n📡 Preparing streaming format (mooncake_trace)")
    print(f"{'='*80}")
    print(f"Input: {parquet_path}")
    print(f"Output: {output_path}")
    
    total_lines = 0
    with open(output_path, 'w') as f:
        for idx, row in df.iterrows():
            messages = json.loads(row["messages"])
            
            # Create one line per turn (after each user message)
            for i in range(len(messages)):
                # Include all messages up to current point
                current_messages = messages[:i+1]
                
                # Estimate output length from next assistant message if available
                output_length = 100  # Default
                if i + 1 < len(messages) and messages[i+1]['role'] == 'assistant':
                    output_length = len(messages[i+1]['content'])
                
                record = {
                    "session_id": row['conversation_id'],
                    "messages": current_messages,
                    "output_length": output_length
                }
                
                f.write(json.dumps(record) + '\n')
                total_lines += 1
    
    print(f"✅ Created {output_path}")
    print(f"   Total lines: {total_lines}")


def show_format_comparison():
    """
    Show the differences between output formats.
    """
    print(f"\n📊 Output Format Comparison")
    print(f"{'='*100}")
    
    formats = {
        "multi_turn": {
            "description": "Lightweight, user messages only",
            "use_case": "Fast benchmarking, prefix cache testing",
            "schema": '{"session_id": "uuid", "turns": [{"text": "..."}, ...]}',
            "size": "Small (~0.8 MB for 500 conversations)",
        },
        "mooncake_trace": {
            "description": "Full message arrays per turn",
            "use_case": "Full control, exact prompt reproduction",
            "schema": '{"session_id": "uuid", "messages": [...], "output_length": 150}',
            "size": "Large (~227 MB for 500 conversations)",
        },
        "parquet": {
            "description": "Full dataset with all metadata",
            "use_case": "Analysis, filtering, custom processing",
            "schema": "Structured columns with JSON fields",
            "size": "Medium (~2 MB for 500 conversations)",
        },
    }
    
    for fmt, info in formats.items():
        print(f"\n{fmt.upper()}")
        print(f"  Description: {info['description']}")
        print(f"  Use Case: {info['use_case']}")
        print(f"  Schema: {info['schema']}")
        print(f"  Size: {info['size']}")


def estimate_benchmark_time(dataset_type: str = "text", tokens_per_second: float = 100.0):
    """
    Estimate benchmark execution time.
    
    Args:
        dataset_type: One of 'text', 'pdf', 'image', 'reasoning', 'agentic'
        tokens_per_second: Estimated inference speed (tokens/sec)
    """
    parquet_path = get_parquet_path(dataset_type)
    
    if not parquet_path.exists():
        print(f"❌ Dataset not found: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    total_tokens = df['estimated_tokens'].sum()
    
    print(f"\n⏱️  Benchmark Time Estimation")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_type}")
    print(f"Total conversations: {len(df)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Inference speed: {tokens_per_second:.0f} tokens/sec")
    
    # Calculate time
    total_seconds = total_tokens / tokens_per_second
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    print(f"\n⏱️  Estimated Time (single-threaded):")
    if hours > 0:
        print(f"  {hours}h {minutes}m {seconds}s")
    elif minutes > 0:
        print(f"  {minutes}m {seconds}s")
    else:
        print(f"  {seconds}s")
    
    # With concurrency
    for concurrency in [1, 5, 10, 20]:
        concurrent_seconds = total_seconds / concurrency
        concurrent_minutes = concurrent_seconds / 60
        print(f"  With {concurrency:2d} concurrent requests: {concurrent_minutes:6.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for benchmarking")
    parser.add_argument("--dataset", default="text", 
                       choices=["text", "pdf", "image", "reasoning", "agentic"],
                       help="Dataset type")
    parser.add_argument("--format", default="multi_turn",
                       choices=["multi_turn", "mooncake_trace", "subset", "filtered"],
                       help="Output format")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of samples for subset")
    parser.add_argument("--min-turns", type=int, default=5,
                       help="Minimum turns for filtering")
    parser.add_argument("--max-turns", type=int, default=20,
                       help="Maximum turns for filtering")
    parser.add_argument("--tokens-per-sec", type=float, default=100.0,
                       help="Estimated tokens/sec for time estimation")
    
    args = parser.parse_args()
    
    print(f"\n🚀 Dataset Preparation Tool")
    
    if args.format == "multi_turn":
        prepare_aiperf_multi_turn(args.dataset)
    elif args.format == "mooncake_trace":
        prepare_for_streaming(args.dataset)
    elif args.format == "subset":
        prepare_custom_subset(args.dataset, args.num_samples)
    elif args.format == "filtered":
        prepare_by_criteria(args.dataset, args.min_turns, args.max_turns)
    
    show_format_comparison()
    estimate_benchmark_time(args.dataset, args.tokens_per_sec)
