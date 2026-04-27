#!/usr/bin/env python3
"""
Example 4: Agentic Task Dataset Analysis

This example demonstrates how to analyze the agentic task dataset,
including tool usage patterns, success metrics, and agent performance.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def analyze_tool_usage():
    """
    Analyze tool usage patterns across all agentic tasks.
    """
    parquet_path = Path("agentic/data/multi_turn_agentic_task.parquet")
    
    if not parquet_path.exists():
        print(f"❌ Agentic dataset not found: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    
    print(f"\n🔧 Tool Usage Analysis")
    print(f"{'='*80}")
    
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
    print(f"\n{'Tool':<35} {'Count':<8} {'Success':<10} {'Failure':<10} {'Rate':<8}")
    print(f"{'-'*70}")
    
    for tool in sorted(tool_stats.keys()):
        stats = tool_stats[tool]
        success_rate = (stats['success'] / stats['count'] * 100) if stats['count'] > 0 else 0
        print(f"{tool:<35} {stats['count']:<8} {stats['success']:<10} {stats['failure']:<10} {success_rate:>6.1f}%")
    
    # Summary statistics
    total_calls = sum(s['count'] for s in tool_stats.values())
    total_success = sum(s['success'] for s in tool_stats.values())
    total_failure = sum(s['failure'] for s in tool_stats.values())
    
    print(f"{'-'*70}")
    print(f"{'TOTAL':<35} {total_calls:<8} {total_success:<10} {total_failure:<10} {(total_success/total_calls*100):>6.1f}%")


def analyze_task_performance():
    """
    Analyze performance metrics by task type.
    """
    parquet_path = Path("agentic/data/multi_turn_agentic_task.parquet")
    
    if not parquet_path.exists():
        print(f"❌ Agentic dataset not found: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    
    print(f"\n📊 Task Performance Analysis")
    print(f"{'='*80}")
    
    # Group by task type
    task_groups = df.groupby('task_type').agg({
        'num_turns': ['mean', 'min', 'max'],
        'num_tool_calls': ['mean', 'sum'],
        'num_errors': ['mean', 'sum'],
        'success_score': ['mean', 'min', 'max'],
        'estimated_tokens': ['mean', 'sum'],
    }).round(2)
    
    print(f"\n{'Task Type':<25} {'Avg Turns':<12} {'Avg Tools':<12} {'Avg Errors':<12} {'Avg Score':<12}")
    print(f"{'-'*73}")
    
    for task_type in df['task_type'].unique():
        task_df = df[df['task_type'] == task_type]
        avg_turns = task_df['num_turns'].mean()
        avg_tools = task_df['num_tool_calls'].mean()
        avg_errors = task_df['num_errors'].mean()
        avg_score = task_df['success_score'].mean()
        
        print(f"{task_type:<25} {avg_turns:<12.1f} {avg_tools:<12.1f} {avg_errors:<12.1f} {avg_score:<12.2f}")
    
    # Detailed metrics by task
    print(f"\n{'='*80}")
    print(f"Detailed Metrics by Task Type:")
    print(f"{'='*80}\n")
    
    for task_type in sorted(df['task_type'].unique()):
        task_df = df[df['task_type'] == task_type]
        
        print(f"\n{task_type.upper()}")
        print(f"  Conversations: {len(task_df)}")
        print(f"  Turns: {task_df['num_turns'].min()}-{task_df['num_turns'].max()} (avg: {task_df['num_turns'].mean():.1f})")
        print(f"  Tool Calls: {task_df['num_tool_calls'].sum():,} total ({task_df['num_tool_calls'].mean():.1f} avg)")
        print(f"  Error Rate: {(task_df['num_errors'].sum() / task_df['num_tool_calls'].sum() * 100):.1f}%")
        print(f"  Success Score: {task_df['success_score'].mean():.2f} (range: {task_df['success_score'].min():.2f}-{task_df['success_score'].max():.2f})")
        print(f"  Tokens: {task_df['estimated_tokens'].sum():,} total ({task_df['estimated_tokens'].mean():,.0f} avg)")


def analyze_success_metrics():
    """
    Analyze success metric distributions and partial credit penalties.
    """
    parquet_path = Path("agentic/data/multi_turn_agentic_task.parquet")
    
    if not parquet_path.exists():
        print(f"❌ Agentic dataset not found: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    
    print(f"\n📈 Success Metric Analysis")
    print(f"{'='*80}")
    
    # Analyze success score distribution
    print(f"\nSuccess Score Distribution:")
    print(f"  Min: {df['success_score'].min():.2f}")
    print(f"  Max: {df['success_score'].max():.2f}")
    print(f"  Mean: {df['success_score'].mean():.2f}")
    print(f"  Median: {df['success_score'].median():.2f}")
    print(f"  Std Dev: {df['success_score'].std():.2f}")
    
    # Score ranges
    print(f"\nScore Range Distribution:")
    ranges = [
        (0.0, 0.2, "0.0-0.2 (Very Poor)"),
        (0.2, 0.4, "0.2-0.4 (Poor)"),
        (0.4, 0.6, "0.4-0.6 (Fair)"),
        (0.6, 0.8, "0.6-0.8 (Good)"),
        (0.8, 1.0, "0.8-1.0 (Excellent)"),
    ]
    
    for min_score, max_score, label in ranges:
        count = len(df[(df['success_score'] >= min_score) & (df['success_score'] < max_score)])
        percentage = (count / len(df)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {label:<25} {count:3d} ({percentage:5.1f}%) {bar}")
    
    # Analyze by metric type
    print(f"\nSuccess Metrics by Task Type:")
    print(f"{'='*80}")
    
    for task_type in sorted(df['task_type'].unique()):
        task_df = df[df['task_type'] == task_type]
        metric = task_df['success_metric'].iloc[0]
        
        print(f"\n{task_type}:")
        print(f"  Metric: {metric}")
        print(f"  Mean Score: {task_df['success_score'].mean():.2f}")
        print(f"  Score Range: {task_df['success_score'].min():.2f} - {task_df['success_score'].max():.2f}")


def analyze_error_patterns():
    """
    Analyze error patterns and recovery strategies.
    """
    parquet_path = Path("agentic/data/multi_turn_agentic_task.parquet")
    
    if not parquet_path.exists():
        print(f"❌ Agentic dataset not found: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    
    print(f"\n⚠️  Error Pattern Analysis")
    print(f"{'='*80}")
    
    # Overall error statistics
    total_calls = df['num_tool_calls'].sum()
    total_errors = df['num_errors'].sum()
    error_rate = (total_errors / total_calls * 100) if total_calls > 0 else 0
    
    print(f"\nOverall Error Statistics:")
    print(f"  Total tool calls: {total_calls:,}")
    print(f"  Total errors: {total_errors:,}")
    print(f"  Error rate: {error_rate:.1f}%")
    
    # Error distribution
    print(f"\nError Distribution:")
    print(f"  No errors: {len(df[df['num_errors'] == 0])} conversations")
    print(f"  1-2 errors: {len(df[(df['num_errors'] > 0) & (df['num_errors'] <= 2)])} conversations")
    print(f"  3-5 errors: {len(df[(df['num_errors'] > 2) & (df['num_errors'] <= 5)])} conversations")
    print(f"  6+ errors: {len(df[df['num_errors'] > 5])} conversations")
    
    # Error rate by task type
    print(f"\nError Rate by Task Type:")
    print(f"{'Task Type':<25} {'Error Rate':<15} {'Avg Errors':<15}")
    print(f"{'-'*55}")
    
    for task_type in sorted(df['task_type'].unique()):
        task_df = df[df['task_type'] == task_type]
        task_total_calls = task_df['num_tool_calls'].sum()
        task_total_errors = task_df['num_errors'].sum()
        task_error_rate = (task_total_errors / task_total_calls * 100) if task_total_calls > 0 else 0
        avg_errors = task_df['num_errors'].mean()
        
        print(f"{task_type:<25} {task_error_rate:<6.1f}%        {avg_errors:<15.1f}")


def compare_conversations(idx1: int = 0, idx2: int = 1):
    """
    Compare two conversations side-by-side.
    """
    parquet_path = Path("agentic/data/multi_turn_agentic_task.parquet")
    
    if not parquet_path.exists():
        print(f"❌ Agentic dataset not found: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    
    if idx1 >= len(df) or idx2 >= len(df):
        print(f"❌ Invalid indices. Dataset has {len(df)} conversations.")
        return
    
    row1 = df.iloc[idx1]
    row2 = df.iloc[idx2]
    
    print(f"\n🔄 Conversation Comparison")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<30} {'Conversation 1':<25} {'Conversation 2':<25}")
    print(f"{'-'*80}")
    
    metrics = [
        ('Task Type', row1['task_type'], row2['task_type']),
        ('Turns', row1['num_turns'], row2['num_turns']),
        ('Messages', row1['num_messages'], row2['num_messages']),
        ('Tool Calls', row1['num_tool_calls'], row2['num_tool_calls']),
        ('Errors', row1['num_errors'], row2['num_errors']),
        ('Success Score', f"{row1['success_score']:.2f}", f"{row2['success_score']:.2f}"),
        ('Tokens', f"{row1['estimated_tokens']:,}", f"{row2['estimated_tokens']:,}"),
        ('Characters', f"{row1['total_characters']:,}", f"{row2['total_characters']:,}"),
    ]
    
    for metric, val1, val2 in metrics:
        print(f"{metric:<30} {str(val1):<25} {str(val2):<25}")
    
    # Show tool calls
    print(f"\n{'='*80}")
    print(f"Tool Calls Comparison:")
    print(f"{'='*80}\n")
    
    for conv_idx, row in [(1, row1), (2, row2)]:
        tool_calls = json.loads(row["tool_calls"])
        
        print(f"Conversation {conv_idx}:")
        tool_dist = defaultdict(int)
        for call in tool_calls:
            tool_dist[call['tool']] += 1
        
        for tool, count in sorted(tool_dist.items()):
            print(f"  {tool}: {count}")
        print()


def export_analysis_report(output_file: str = "agentic_analysis_report.txt"):
    """
    Export a comprehensive analysis report.
    """
    parquet_path = Path("agentic/data/multi_turn_agentic_task.parquet")
    
    if not parquet_path.exists():
        print(f"❌ Agentic dataset not found: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    
    print(f"\n📄 Exporting analysis report to {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AGENTIC TASK DATASET ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overview
        f.write("OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Conversations: {len(df)}\n")
        f.write(f"Total Turns: {df['num_turns'].sum()}\n")
        f.write(f"Total Messages: {df['num_messages'].sum()}\n")
        f.write(f"Total Tool Calls: {df['num_tool_calls'].sum()}\n")
        f.write(f"Total Errors: {df['num_errors'].sum()}\n")
        f.write(f"Total Tokens: {df['estimated_tokens'].sum():,}\n\n")
        
        # Task distribution
        f.write("TASK DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for task_type in sorted(df['task_type'].unique()):
            count = len(df[df['task_type'] == task_type])
            percentage = (count / len(df)) * 100
            f.write(f"{task_type:<25} {count:3d} ({percentage:5.1f}%)\n")
        f.write("\n")
        
        # Success metrics
        f.write("SUCCESS METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Score: {df['success_score'].mean():.2f}\n")
        f.write(f"Min Score: {df['success_score'].min():.2f}\n")
        f.write(f"Max Score: {df['success_score'].max():.2f}\n")
        f.write(f"Median Score: {df['success_score'].median():.2f}\n\n")
    
    print(f"✅ Report exported to {output_file}")


if __name__ == "__main__":
    import sys
    
    print(f"\n🤖 Agentic Task Dataset Analysis")
    
    analyze_tool_usage()
    analyze_task_performance()
    analyze_success_metrics()
    analyze_error_patterns()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        idx1 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        idx2 = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        compare_conversations(idx1, idx2)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        export_analysis_report()
