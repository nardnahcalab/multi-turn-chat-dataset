#!/usr/bin/env python3
"""
Example 1: Loading and Inspecting Datasets

This example demonstrates how to load the generated datasets and inspect their structure.
It works with all dataset types: text, pdf, image, reasoning, and agentic.
"""

import pandas as pd
import json
from pathlib import Path


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


def inspect_dataset(dataset_type: str = "text", num_samples: int = 3):
    """
    Load and inspect a dataset.
    
    Args:
        dataset_type: One of 'text', 'pdf', 'image', 'reasoning', 'agentic'
        num_samples: Number of samples to display
    """
    # Construct the path to the parquet file
    parquet_path = get_parquet_path(dataset_type)
    
    if not parquet_path.exists():
        print(f"❌ Dataset not found: {parquet_path}")
        print(f"   Run: python {dataset_type}/generate.py")
        return
    
    # Load the dataset
    print(f"\n📊 Loading {dataset_type} dataset from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Display basic statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total conversations: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Shape: {df.shape}")
    
    # Display column info
    print(f"\n📋 Column Information:")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        print(f"   {col:30s} | {str(dtype):15s} | {non_null}/{len(df)} non-null")
    
    # Display sample conversations
    print(f"\n{'='*80}")
    print(f"Sample Conversations (first {num_samples}):")
    print(f"{'='*80}\n")
    
    for idx in range(min(num_samples, len(df))):
        row = df.iloc[idx]
        print(f"[Sample {idx + 1}]")
        print(f"  Conversation ID: {row['conversation_id']}")
        
        # Display type-specific fields
        if dataset_type == "text":
            print(f"  Topic: {row['topic']}")
        elif dataset_type == "pdf":
            print(f"  Paper: {row['paper_title']}")
            print(f"  arXiv ID: {row['paper_arxiv_id']}")
            print(f"  Conversation Type: {row['conversation_type']}")
        elif dataset_type == "image":
            print(f"  Image Topic: {row['image_topic']}")
            print(f"  Conversation Type: {row['conversation_type']}")
        elif dataset_type == "reasoning":
            print(f"  Topic: {row['topic']}")
        elif dataset_type == "agentic":
            print(f"  Task Type: {row['task_type']}")
            print(f"  Success Score: {row['success_score']:.2f}")
            print(f"  Tool Calls: {row['num_tool_calls']}")
            print(f"  Errors: {row['num_errors']}")
        
        print(f"  Turns: {row['num_turns']}")
        print(f"  Messages: {row['num_messages']}")
        print(f"  Characters: {row['total_characters']:,}")
        print(f"  Estimated Tokens: {row['estimated_tokens']:,}")
        
        # Show first message
        messages = json.loads(row["messages"])
        first_msg = messages[0]
        print(f"\n  First Message:")
        print(f"    Role: {first_msg['role']}")
        content = first_msg['content'][:100]
        print(f"    Content: {content}...")
        
        # Show last message
        last_msg = messages[-1]
        print(f"\n  Last Message:")
        print(f"    Role: {last_msg['role']}")
        content = last_msg['content'][:100]
        print(f"    Content: {content}...")
        
        print(f"\n{'-'*80}\n")


def analyze_statistics(dataset_type: str = "text"):
    """
    Analyze and display statistics about the dataset.
    """
    parquet_path = get_parquet_path(dataset_type)
    
    if not parquet_path.exists():
        print(f"❌ Dataset not found: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    
    print(f"\n📊 Detailed Statistics for {dataset_type} Dataset:")
    print(f"{'='*80}\n")
    
    # Turn statistics
    print("Turn Distribution:")
    print(f"  Min turns: {df['num_turns'].min()}")
    print(f"  Max turns: {df['num_turns'].max()}")
    print(f"  Mean turns: {df['num_turns'].mean():.2f}")
    print(f"  Median turns: {df['num_turns'].median():.0f}")
    
    # Token statistics
    print(f"\nToken Statistics:")
    print(f"  Total tokens: {df['estimated_tokens'].sum():,}")
    print(f"  Min tokens: {df['estimated_tokens'].min():,}")
    print(f"  Max tokens: {df['estimated_tokens'].max():,}")
    print(f"  Mean tokens: {df['estimated_tokens'].mean():,.0f}")
    
    # Character statistics
    print(f"\nCharacter Statistics:")
    print(f"  Total characters: {df['total_characters'].sum():,}")
    print(f"  Min characters: {df['total_characters'].min():,}")
    print(f"  Max characters: {df['total_characters'].max():,}")
    print(f"  Mean characters: {df['total_characters'].mean():,.0f}")
    
    # Type-specific statistics
    if dataset_type == "text":
        print(f"\nTopic Distribution:")
        print(df['topic'].value_counts())
    elif dataset_type == "pdf":
        print(f"\nConversation Type Distribution:")
        print(df['conversation_type'].value_counts())
    elif dataset_type == "image":
        print(f"\nImage Topic Distribution:")
        print(df['image_topic'].value_counts())
    elif dataset_type == "reasoning":
        print(f"\nTopic Distribution:")
        print(df['topic'].value_counts())
    elif dataset_type == "agentic":
        print(f"\nTask Type Distribution:")
        print(df['task_type'].value_counts())
        print(f"\nSuccess Score Statistics:")
        print(f"  Min: {df['success_score'].min():.2f}")
        print(f"  Max: {df['success_score'].max():.2f}")
        print(f"  Mean: {df['success_score'].mean():.2f}")
        print(f"  Median: {df['success_score'].median():.2f}")
        print(f"\nTool Call Statistics:")
        print(f"  Total tool calls: {df['num_tool_calls'].sum():,}")
        print(f"  Total errors: {df['num_errors'].sum():,}")
        print(f"  Error rate: {(df['num_errors'].sum() / df['num_tool_calls'].sum() * 100):.1f}%")


if __name__ == "__main__":
    import sys
    
    # Default to text dataset, but allow command-line override
    dataset_type = sys.argv[1] if len(sys.argv) > 1 else "text"
    
    print(f"\n🔍 Inspecting {dataset_type} dataset...")
    inspect_dataset(dataset_type, num_samples=3)
    analyze_statistics(dataset_type)
