#!/usr/bin/env python3
"""
Example 5: Dataset Profiling, Tagging, and Distribution Analysis

This example demonstrates the new dataset profiling features:
1. Loading and displaying dataset manifests (tags + distribution profile)
2. Comparing distribution profiles across dataset types
3. Filtering datasets by tags
4. Generating standalone profiles for existing datasets

Usage:
    python examples/05_dataset_profile.py                    # profile all datasets
    python examples/05_dataset_profile.py text               # profile a single dataset
    python examples/05_dataset_profile.py --compare text reasoning  # compare two datasets
    python examples/05_dataset_profile.py --tags             # list all tags across datasets
    python examples/05_dataset_profile.py --filter tag1 tag2 # find datasets matching tags
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_profile import (
    build_manifest,
    compute_distribution_profile,
    generate_tags,
    print_profile_summary,
)

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

# Map dataset type to its config
CONFIG_FILES = {
    "text": "text/config.yaml",
    "pdf": "pdf/config.yaml",
    "image": "image/config.yaml",
    "reasoning": "reasoning/config.yaml",
    "agentic": "agentic/config.yaml",
    "random": "random/config.yaml",
    "repeat": "repeat/config.yaml",
}

# Map dataset type to its manifest
MANIFEST_FILES = {
    "text": "text/data/multi_turn_text_chat_manifest.json",
    "pdf": "pdf/data/multi_turn_pdf_chat_manifest.json",
    "image": "image/data/multi_turn_image_chat_manifest.json",
    "reasoning": "reasoning/data/multi_turn_reasoning_chat_manifest.json",
    "agentic": "agentic/data/multi_turn_agentic_task_manifest.json",
    "random": "random/data/multi_turn_random_chat_manifest.json",
    "repeat": "repeat/data/multi_turn_repeat_chat_manifest.json",
}


def load_manifest(dataset_type: str) -> dict:
    """Load a manifest from disk, or generate one on the fly from parquet + config."""
    import yaml

    project_root = Path(__file__).resolve().parent.parent

    # Try loading pre-generated manifest first
    manifest_path = project_root / MANIFEST_FILES.get(dataset_type, "")
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)

    # Fall back to generating from parquet + config
    parquet_path = project_root / PARQUET_FILES.get(dataset_type, "")
    config_path = project_root / CONFIG_FILES.get(dataset_type, "")

    if not parquet_path.exists():
        print(f"  Dataset not found: {parquet_path}")
        print(f"  Run: python {dataset_type}/generate.py")
        return None

    df = pd.read_parquet(parquet_path)

    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    seed = config.get("dataset", {}).get("seed", 42)
    name = config.get("dataset", {}).get("output_filename", "").replace(".parquet", "")

    manifest = build_manifest(
        df=df,
        config=config,
        dataset_type=dataset_type,
        seed=seed,
        output_files={"parquet": str(parquet_path)},
        descriptive_name=name or dataset_type,
    )
    return manifest


def profile_dataset(dataset_type: str):
    """Display the full profile for a dataset."""
    print(f"\nProfiling {dataset_type} dataset...")
    manifest = load_manifest(dataset_type)
    if manifest:
        print_profile_summary(manifest)


def compare_datasets(types: list):
    """Compare distribution profiles side by side."""
    manifests = {}
    for dt in types:
        m = load_manifest(dt)
        if m:
            manifests[dt] = m

    if len(manifests) < 2:
        print("Need at least 2 available datasets to compare.")
        return

    print(f"\n{'='*80}")
    print(f"  DATASET COMPARISON: {' vs '.join(manifests.keys())}")
    print(f"{'='*80}")

    # Header
    col_width = 20
    header = f"{'Metric':<30s}"
    for dt in manifests:
        header += f"{dt:>{col_width}s}"
    print(f"\n{header}")
    print("-" * (30 + col_width * len(manifests)))

    # Conversations
    row = f"{'Conversations':<30s}"
    for dt, m in manifests.items():
        row += f"{m['dataset']['num_conversations']:>{col_width},}"
    print(row)

    # Turns
    row = f"{'Turn range':<30s}"
    for dt, m in manifests.items():
        td = m["distribution_profile"]["turn_distribution"]
        row += f"{td['min']}-{td['max']:>{col_width - len(str(td['min'])) - 1}}"
    print(row)

    row = f"{'Mean turns':<30s}"
    for dt, m in manifests.items():
        td = m["distribution_profile"]["turn_distribution"]
        row += f"{td['mean']:>{col_width}.1f}"
    print(row)

    # Tokens
    row = f"{'Total tokens':<30s}"
    for dt, m in manifests.items():
        tkd = m["distribution_profile"]["token_distribution"]
        row += f"{tkd['total']:>{col_width},}"
    print(row)

    row = f"{'Mean tokens/conversation':<30s}"
    for dt, m in manifests.items():
        tkd = m["distribution_profile"]["token_distribution"]
        row += f"{tkd['mean']:>{col_width},.0f}"
    print(row)

    row = f"{'Max tokens (single)':<30s}"
    for dt, m in manifests.items():
        tkd = m["distribution_profile"]["token_distribution"]
        row += f"{tkd['max']:>{col_width},}"
    print(row)

    # Context growth
    row = f"{'Context growth (mean ratio)':<30s}"
    for dt, m in manifests.items():
        cg = m["distribution_profile"].get("context_growth", {}).get("growth_ratio", {})
        val = cg.get("mean", 0)
        row += f"{val:>{col_width}.1f}x"
    print(row)

    # Tags
    print(f"\n{'Tags':<30s}")
    for dt, m in manifests.items():
        tags = m["tags"]["all_tags"]
        print(f"  {dt}: {', '.join(tags)}")

    print(f"\n{'='*80}\n")


def list_all_tags():
    """Show all tags across all datasets."""
    print(f"\n{'='*60}")
    print(f"  TAGS ACROSS ALL DATASETS")
    print(f"{'='*60}\n")

    all_tags = {}  # tag -> list of dataset types
    for dt in PARQUET_FILES:
        manifest = load_manifest(dt)
        if manifest:
            for tag in manifest["tags"]["all_tags"]:
                all_tags.setdefault(tag, []).append(dt)

    # Sort by frequency (most common first)
    for tag, datasets in sorted(all_tags.items(), key=lambda x: -len(x[1])):
        print(f"  {tag:<30s} [{len(datasets)}] {', '.join(datasets)}")

    print(f"\n  Total unique tags: {len(all_tags)}")
    print(f"{'='*60}\n")


def filter_by_tags(required_tags: list):
    """Find datasets that match all required tags."""
    print(f"\n  Filtering for tags: {', '.join(required_tags)}")
    print(f"  {'='*50}\n")

    matches = []
    for dt in PARQUET_FILES:
        manifest = load_manifest(dt)
        if manifest:
            dataset_tags = set(manifest["tags"]["all_tags"])
            if all(t in dataset_tags for t in required_tags):
                matches.append((dt, manifest))

    if not matches:
        print("  No datasets match all required tags.")
    else:
        print(f"  Found {len(matches)} matching dataset(s):\n")
        for dt, manifest in matches:
            ds = manifest["dataset"]
            tags = manifest["tags"]["all_tags"]
            tkd = manifest["distribution_profile"]["token_distribution"]
            print(f"  {dt}")
            print(f"    Name: {ds['name']}")
            print(f"    Conversations: {ds['num_conversations']}, Tokens: {tkd['total']:,}")
            print(f"    Tags: {', '.join(tags)}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Dataset profiling, tagging, and distribution analysis"
    )
    parser.add_argument("datasets", nargs="*", default=[],
                        help="Dataset type(s) to profile (default: all)")
    parser.add_argument("--compare", nargs="+", metavar="TYPE",
                        help="Compare distribution profiles of multiple datasets")
    parser.add_argument("--tags", action="store_true",
                        help="List all tags across all datasets")
    parser.add_argument("--filter", nargs="+", metavar="TAG",
                        help="Find datasets matching all specified tags")
    args = parser.parse_args()

    if args.tags:
        list_all_tags()
        return

    if args.filter:
        filter_by_tags(args.filter)
        return

    if args.compare:
        compare_datasets(args.compare)
        return

    # Default: profile specified datasets (or all)
    types_to_profile = args.datasets if args.datasets else list(PARQUET_FILES.keys())
    for dt in types_to_profile:
        if dt not in PARQUET_FILES:
            print(f"  Unknown dataset type: {dt}")
            print(f"  Available: {', '.join(PARQUET_FILES.keys())}")
            continue
        profile_dataset(dt)


if __name__ == "__main__":
    main()
