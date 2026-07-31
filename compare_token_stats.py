#!/usr/bin/env python3
"""Compare token counts and distributions between two JSONL files."""

import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

# Try to use tiktoken, fall back to heuristic
try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
    USE_TIKTOKEN = True
except ImportError:
    TOKENIZER = None
    USE_TIKTOKEN = False

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken or fallback to char heuristic."""
    if USE_TIKTOKEN:
        return len(TOKENIZER.encode(text))
    else:
        return len(text) // 4  # CHARS_PER_TOKEN heuristic

def analyze_jsonl(file_path: str) -> dict:
    """Analyze a JSONL file and return token statistics."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    conversations = []
    turn_counts = []
    total_tokens = 0
    turn_distribution = defaultdict(int)
    
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            turns = data.get('turns', [])
            conv_tokens = 0
            conv_turn_counts = []
            
            for turn in turns:
                text = turn.get('text', '')
                tokens = count_tokens(text)
                conv_tokens += tokens
                conv_turn_counts.append(tokens)
                turn_counts.append(tokens)
                total_tokens += tokens
                
                # Bucket turn sizes
                if tokens < 10:
                    turn_distribution['0-9'] += 1
                elif tokens < 50:
                    turn_distribution['10-49'] += 1
                elif tokens < 100:
                    turn_distribution['50-99'] += 1
                elif tokens < 200:
                    turn_distribution['100-199'] += 1
                elif tokens < 500:
                    turn_distribution['200-499'] += 1
                elif tokens < 1000:
                    turn_distribution['500-999'] += 1
                else:
                    turn_distribution['1000+'] += 1
            
            conversations.append({
                'session_id': data.get('session_id'),
                'num_turns': len(turns),
                'total_tokens': conv_tokens,
                'turn_token_counts': conv_turn_counts
            })
    
    # Calculate statistics
    if turn_counts:
        turn_stats = {
            'min': min(turn_counts),
            'max': max(turn_counts),
            'mean': statistics.mean(turn_counts),
            'median': statistics.median(turn_counts),
            'stdev': statistics.stdev(turn_counts) if len(turn_counts) > 1 else 0
        }
    else:
        turn_stats = {}
    
    if conversations:
        conv_token_counts = [c['total_tokens'] for c in conversations]
        conv_stats = {
            'min': min(conv_token_counts),
            'max': max(conv_token_counts),
            'mean': statistics.mean(conv_token_counts),
            'median': statistics.median(conv_token_counts),
            'stdev': statistics.stdev(conv_token_counts) if len(conv_token_counts) > 1 else 0
        }
        turn_per_conv = [c['num_turns'] for c in conversations]
        turn_per_conv_stats = {
            'min': min(turn_per_conv),
            'max': max(turn_per_conv),
            'mean': statistics.mean(turn_per_conv),
            'median': statistics.median(turn_per_conv),
        }
    else:
        conv_stats = {}
        turn_per_conv_stats = {}
    
    return {
        'file': str(path),
        'num_conversations': len(conversations),
        'total_turns': len(turn_counts),
        'total_tokens': total_tokens,
        'turn_stats': turn_stats,
        'conversation_stats': conv_stats,
        'turns_per_conversation': turn_per_conv_stats,
        'turn_distribution': dict(turn_distribution),
        'conversations': conversations
    }

def print_comparison(analysis1: dict, analysis2: dict):
    """Print a comparison of two analyses."""
    print("=" * 80)
    print("TOKEN COUNT AND DISTRIBUTION COMPARISON")
    print("=" * 80)
    print(f"Using tokenizer: {'tiktoken (cl100k_base)' if USE_TIKTOKEN else 'chars // 4 heuristic'}")
    print()
    
    # File overview
    print("FILE OVERVIEW")
    print("-" * 80)
    print(f"File 1: {Path(analysis1['file']).name}")
    print(f"  - Conversations: {analysis1['num_conversations']}")
    print(f"  - Total turns: {analysis1['total_turns']}")
    print(f"  - Total tokens: {analysis1['total_tokens']:,}")
    print()
    print(f"File 2: {Path(analysis2['file']).name}")
    print(f"  - Conversations: {analysis2['num_conversations']}")
    print(f"  - Total turns: {analysis2['total_turns']}")
    print(f"  - Total tokens: {analysis2['total_tokens']:,}")
    print()
    
    # Turn-level statistics
    print("TURN-LEVEL STATISTICS")
    print("-" * 80)
    print(f"{'Metric':<20} {Path(analysis1['file']).name:<20} {Path(analysis2['file']).name:<20} {'Difference':<15}")
    print("-" * 80)
    
    for metric in ['min', 'max', 'mean', 'median', 'stdev']:
        val1 = analysis1['turn_stats'].get(metric, 0)
        val2 = analysis2['turn_stats'].get(metric, 0)
        diff = val2 - val1
        print(f"{metric:<20} {val1:<20.1f} {val2:<20.1f} {diff:+<15.1f}")
    
    print()
    
    # Conversation-level statistics
    print("CONVERSATION-LEVEL STATISTICS (tokens per conversation)")
    print("-" * 80)
    print(f"{'Metric':<20} {Path(analysis1['file']).name:<20} {Path(analysis2['file']).name:<20} {'Difference':<15}")
    print("-" * 80)
    
    for metric in ['min', 'max', 'mean', 'median', 'stdev']:
        val1 = analysis1['conversation_stats'].get(metric, 0)
        val2 = analysis2['conversation_stats'].get(metric, 0)
        diff = val2 - val1
        print(f"{metric:<20} {val1:<20.1f} {val2:<20.1f} {diff:+<15.1f}")
    
    print()
    
    # Turns per conversation
    print("TURNS PER CONVERSATION")
    print("-" * 80)
    print(f"{'Metric':<20} {Path(analysis1['file']).name:<20} {Path(analysis2['file']).name:<20} {'Difference':<15}")
    print("-" * 80)
    
    for metric in ['min', 'max', 'mean', 'median']:
        val1 = analysis1['turns_per_conversation'].get(metric, 0)
        val2 = analysis2['turns_per_conversation'].get(metric, 0)
        diff = val2 - val1
        print(f"{metric:<20} {val1:<20.1f} {val2:<20.1f} {diff:+<15.1f}")
    
    print()
    
    # Turn size distribution
    print("TURN SIZE DISTRIBUTION")
    print("-" * 80)
    print(f"{'Bucket':<15} {Path(analysis1['file']).name:<20} {Path(analysis2['file']).name:<20}")
    print("-" * 80)
    
    all_buckets = set(analysis1['turn_distribution'].keys()) | set(analysis2['turn_distribution'].keys())
    for bucket in sorted(all_buckets):
        val1 = analysis1['turn_distribution'].get(bucket, 0)
        val2 = analysis2['turn_distribution'].get(bucket, 0)
        print(f"{bucket:<15} {val1:<20} {val2:<20}")
    
    print()
    print("=" * 80)

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_token_stats.py <file1.jsonl> <file2.jsonl>")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    print(f"Analyzing {file1}...")
    analysis1 = analyze_jsonl(file1)
    
    print(f"Analyzing {file2}...")
    analysis2 = analyze_jsonl(file2)
    
    print_comparison(analysis1, analysis2)

if __name__ == "__main__":
    main()
