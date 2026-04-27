#!/usr/bin/env python3
"""
Example 2: Analyzing Conversation Structure and Growth

This example shows how to analyze conversation structure, context growth,
and message patterns across different dataset types.
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict


# Map dataset type to its parquet filename
PARQUET_FILES = {
    "text": "text/data/multi_turn_text_chat.parquet",
    "pdf": "pdf/data/multi_turn_pdf_chat.parquet",
    "image": "image/data/multi_turn_image_chat.parquet",
    "reasoning": "reasoning/data/multi_turn_reasoning_chat.parquet",
    "agentic": "agentic/data/multi_turn_agentic_task.parquet",
}


def get_parquet_path(dataset_type: str) -> Path:
    """Return the correct parquet path for a given dataset type."""
    return Path(PARQUET_FILES.get(dataset_type, f"{dataset_type}/data/multi_turn_{dataset_type}_chat.parquet"))


def analyze_context_growth(dataset_type: str = "text", sample_idx: int = 0):
    """
    Analyze how context grows across turns in a conversation.
    
    Args:
        dataset_type: One of 'text', 'pdf', 'image', 'reasoning', 'agentic'
        sample_idx: Index of the conversation to analyze
    """
    parquet_path = get_parquet_path(dataset_type)
    
    if not parquet_path.exists():
        print(f"❌ Dataset not found: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    row = df.iloc[sample_idx]
    
    print(f"\n📈 Context Growth Analysis")
    print(f"{'='*80}")
    print(f"Conversation ID: {row['conversation_id']}")
    print(f"Dataset Type: {dataset_type}")
    print(f"Total Turns: {row['num_turns']}")
    print(f"Total Characters: {row['total_characters']:,}")
    print(f"Estimated Tokens: {row['estimated_tokens']:,}")
    
    # Parse cumulative lengths
    cumulative_lengths = json.loads(row["cumulative_char_lengths"])
    
    print(f"\n{'Turn':<6} {'Characters':<15} {'Tokens (est.)':<15} {'Growth':<15}")
    print(f"{'-'*50}")
    
    for i, char_count in enumerate(cumulative_lengths):
        token_count = char_count // 4  # Rough estimation
        if i == 0:
            growth = "—"
        else:
            prev_chars = cumulative_lengths[i-1]
            growth = f"+{char_count - prev_chars:,} chars"
        
        print(f"{i+1:<6} {char_count:<15,} {token_count:<15,} {growth:<15}")
    
    # Analyze message roles
    messages = json.loads(row["messages"])
    role_counts = {}
    for msg in messages:
        role = msg['role']
        role_counts[role] = role_counts.get(role, 0) + 1
    
    print(f"\n📊 Message Distribution:")
    for role, count in sorted(role_counts.items()):
        print(f"  {role:15s}: {count:3d} messages")
    
    # Analyze message lengths
    print(f"\n📝 Message Length Analysis:")
    message_lengths = [len(msg['content']) for msg in messages]
    print(f"  Min length: {min(message_lengths):,} chars")
    print(f"  Max length: {max(message_lengths):,} chars")
    print(f"  Mean length: {sum(message_lengths) / len(message_lengths):,.0f} chars")
    
    # Show message preview
    print(f"\n💬 Message Preview:")
    for i, msg in enumerate(messages[:5]):  # Show first 5 messages
        role = msg['role']
        content = msg['content'][:80]
        print(f"  [{i+1}] {role:10s}: {content}...")


def analyze_message_patterns(dataset_type: str = "text"):
    """
    Analyze message patterns across all conversations.
    """
    parquet_path = get_parquet_path(dataset_type)
    
    if not parquet_path.exists():
        print(f"❌ Dataset not found: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    
    print(f"\n📊 Message Pattern Analysis")
    print(f"{'='*80}")
    
    # Analyze role distribution
    all_roles = {}
    for messages_json in df["messages"]:
        messages = json.loads(messages_json)
        for msg in messages:
            role = msg['role']
            all_roles[role] = all_roles.get(role, 0) + 1
    
    total_messages = sum(all_roles.values())
    print(f"\nRole Distribution (across {len(df)} conversations):")
    for role, count in sorted(all_roles.items()):
        percentage = (count / total_messages) * 100
        print(f"  {role:15s}: {count:6,} messages ({percentage:5.1f}%)")
    
    # Analyze turn length distribution
    print(f"\nTurn Length Distribution:")
    turn_lengths = df['num_turns'].value_counts().sort_index()
    for turns, count in turn_lengths.items():
        percentage = (count / len(df)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {turns:2d} turns: {count:3d} conversations ({percentage:5.1f}%) {bar}")
    
    # Analyze message count distribution
    print(f"\nMessage Count Distribution:")
    msg_counts = df['num_messages'].value_counts().sort_index()
    for msgs, count in msg_counts.head(10).items():
        percentage = (count / len(df)) * 100
        print(f"  {msgs:3d} messages: {count:3d} conversations ({percentage:5.1f}%)")


def analyze_agentic_specific(sample_idx: int = 0):
    """
    Analyze agentic-specific features like tool calls and success metrics.
    """
    parquet_path = Path("agentic/data/multi_turn_agentic_task.parquet")
    
    if not parquet_path.exists():
        print(f"❌ Agentic dataset not found: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    row = df.iloc[sample_idx]
    
    print(f"\n🤖 Agentic Task Analysis")
    print(f"{'='*80}")
    print(f"Conversation ID: {row['conversation_id']}")
    print(f"Task Type: {row['task_type']}")
    print(f"System Prompt: {row['system_prompt'][:100]}...")
    
    # Parse and analyze tool calls
    tool_calls = json.loads(row["tool_calls"])
    
    print(f"\n🔧 Tool Call Analysis:")
    print(f"  Total tool calls: {len(tool_calls)}")
    print(f"  Success rate: {((1 - row['num_errors'] / row['num_tool_calls']) * 100) if row['num_tool_calls'] > 0 else 100:.1f}%")
    
    # Tool distribution
    tool_distribution = {}
    for call in tool_calls:
        tool = call['tool']
        tool_distribution[tool] = tool_distribution.get(tool, 0) + 1
    
    print(f"\n  Tool Distribution:")
    for tool, count in sorted(tool_distribution.items(), key=lambda x: x[1], reverse=True):
        status = "✓" if count > 0 else "✗"
        print(f"    {status} {tool:30s}: {count:2d} calls")
    
    # Success metrics
    print(f"\n📊 Success Metrics:")
    print(f"  Metric: {row['success_metric']}")
    print(f"  Score: {row['success_score']:.2f} / 1.00")
    
    # Show sample tool calls
    print(f"\n💾 Sample Tool Calls (first 3):")
    for i, call in enumerate(tool_calls[:3]):
        print(f"\n  [{i+1}] {call['tool']}")
        print(f"      Status: {call['result']['status']}")
        if 'error' in call['result']:
            print(f"      Error: {call['result']['error']}")
        else:
            result_str = str(call['result'].get('result', 'N/A'))[:60]
            print(f"      Result: {result_str}...")


def compare_datasets():
    """
    Compare statistics across all available datasets.
    """
    datasets = ["text", "pdf", "image", "reasoning", "agentic"]
    
    print(f"\n📊 Dataset Comparison")
    print(f"{'='*100}")
    
    stats = []
    for dataset_type in datasets:
        parquet_path = get_parquet_path(dataset_type)
        
        if not parquet_path.exists():
            continue
        
        df = pd.read_parquet(parquet_path)
        
        stats.append({
            "Dataset": dataset_type,
            "Conversations": len(df),
            "Avg Turns": f"{df['num_turns'].mean():.1f}",
            "Avg Tokens": f"{df['estimated_tokens'].mean():,.0f}",
            "Total Tokens": f"{df['estimated_tokens'].sum():,}",
            "Max Tokens": f"{df['estimated_tokens'].max():,}",
        })
    
    if stats:
        stats_df = pd.DataFrame(stats)
        print(stats_df.to_string(index=False))


if __name__ == "__main__":
    import sys
    
    dataset_type = sys.argv[1] if len(sys.argv) > 1 else "text"
    
    print(f"\n🔍 Analyzing {dataset_type} dataset...")
    analyze_context_growth(dataset_type, sample_idx=0)
    analyze_message_patterns(dataset_type)
    
    if dataset_type == "agentic":
        analyze_agentic_specific(sample_idx=0)
    
    compare_datasets()
