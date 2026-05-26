#!/usr/bin/env python3
"""
Example: Compare payload scores across datasets and create payload-balanced subsets.

This example demonstrates how to:
1. Load multiple datasets
2. Compute and compare payload scores
3. Create subsets with matching payload scores
4. Visualize the prefill/decode tradeoff
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project root to path for shared module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_profile import compute_payload_score


def load_dataset(dataset_type: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load a dataset by type."""
    # Get the project root (parent of examples directory)
    project_root = Path(__file__).resolve().parent.parent

    # Try standard naming convention first
    path = project_root / dataset_type / "data" / f"multi_turn_{dataset_type}_chat.parquet"
    if not path.exists():
        # Try agentic naming convention
        path = project_root / dataset_type / "data" / f"multi_turn_{dataset_type}_task.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    return df


def compute_all_payload_scores(
    dataset_types: List[str],
    sample_size: Optional[int] = None,
) -> Dict[str, Dict]:
    """Compute payload scores for multiple datasets."""
    results = {}

    for dataset_type in dataset_types:
        print(f"\nLoading {dataset_type} dataset...")
        try:
            df = load_dataset(dataset_type, sample_size)
            print(f"  Loaded {len(df)} conversations")

            payload = compute_payload_score(df, dataset_type)
            results[dataset_type] = {
                "payload": payload,
                "num_conversations": len(df),
                "total_tokens": int(df["estimated_tokens"].sum()) if "estimated_tokens" in df.columns else 0,
            }

            print(f"  Prefill Score:  {payload['prefill_score']:.4f}")
            print(f"  Decode Score:   {payload['decode_score']:.4f}")
            print(f"  Total Payload:  {payload['total_payload_score']:.4f}")

        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
            continue

    return results


def compare_payload_scores(results: Dict[str, Dict]) -> None:
    """Print a comparison table of payload scores."""
    print("\n" + "="*80)
    print("PAYLOAD SCORE COMPARISON")
    print("="*80)
    print(f"{'Dataset':<20} {'Conversations':<15} {'Prefill':<12} {'Decode':<12} {'Total':<12}")
    print("-"*80)

    for dataset_type, data in sorted(results.items(), key=lambda x: x[1]["payload"]["total_payload_score"], reverse=True):
        payload = data["payload"]
        print(f"{dataset_type:<20} {data['num_conversations']:<15} "
              f"{payload['prefill_score']:<12.4f} {payload['decode_score']:<12.4f} "
              f"{payload['total_payload_score']:<12.4f}")

    print("="*80)


def find_payload_balanced_subset(
    df: pd.DataFrame,
    dataset_type: str,
    target_payload: float,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """Find a subset of the dataset that matches the target payload score.

    Uses binary search to find the subset size that achieves the target payload.
    """
    low, high = 10, len(df)
    best_subset = None
    best_diff = float('inf')

    print(f"\nFinding subset for {dataset_type} with target payload {target_payload:.4f}...")

    # Binary search for subset size
    for _ in range(20):  # 20 iterations is sufficient for convergence
        mid = (low + high) // 2
        subset = df.sample(n=mid, random_state=42)
        payload = compute_payload_score(subset, dataset_type)
        diff = abs(payload["total_payload_score"] - target_payload)

        if diff < best_diff:
            best_diff = diff
            best_subset = subset

        if payload["total_payload_score"] < target_payload:
            low = mid + 1
        else:
            high = mid - 1

    if best_subset is not None:
        final_payload = compute_payload_score(best_subset, dataset_type)
        print(f"  Found subset of {len(best_subset)} conversations")
        print(f"  Actual payload: {final_payload['total_payload_score']:.4f} (target: {target_payload:.4f})")
        print(f"  Difference: {best_diff:.4f}")

        if best_diff <= tolerance:
            print(f"  ✓ Within tolerance ({tolerance:.4f})")
        else:
            print(f"  ⚠ Outside tolerance ({tolerance:.4f})")

    return best_subset if best_subset is not None else df.sample(n=10, random_state=42)


def create_balanced_datasets(
    results: Dict[str, Dict],
    target_payload: float,
    output_dir: Path,
) -> None:
    """Create payload-balanced subsets for all datasets."""
    print(f"\nCreating payload-balanced datasets with target payload: {target_payload:.4f}")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_type, data in results.items():
        print(f"\nProcessing {dataset_type}...")
        df = load_dataset(dataset_type)

        balanced_subset = find_payload_balanced_subset(
            df, dataset_type, target_payload
        )

        # Save the balanced subset
        output_path = output_dir / f"balanced_{dataset_type}_payload.parquet"
        balanced_subset.to_parquet(output_path, index=False)
        print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare payload scores across datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["text", "reasoning", "agentic", "pdf", "image"],
        help="Dataset types to compare"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for each dataset (default: use full dataset)"
    )
    parser.add_argument(
        "--target-payload",
        type=float,
        default=None,
        help="Target payload score for creating balanced subsets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="balanced_datasets",
        help="Output directory for balanced subsets"
    )
    args = parser.parse_args()

    print("Payload Score Comparison Tool")
    print("="*80)

    # Compute payload scores for all datasets
    results = compute_all_payload_scores(args.datasets, args.sample_size)

    if not results:
        print("No datasets were successfully loaded.")
        return

    # Print comparison
    compare_payload_scores(results)

    # Create balanced subsets if target payload is specified
    if args.target_payload:
        output_dir = Path(args.output_dir)
        create_balanced_datasets(results, args.target_payload, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
