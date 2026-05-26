#!/usr/bin/env python3
"""
Regenerate manifests for existing datasets to include payload scores.

This script loads existing parquet files and regenerates their manifests
with the new payload scoring functionality, without regenerating the data.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

# Add project root to path for shared module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_profile import (
    build_descriptive_name,
    build_manifest,
    save_manifest,
    print_profile_summary,
)


def load_dataset_config(dataset_type: str) -> Dict:
    """Load configuration for a dataset type."""
    config_path = Path(dataset_type) / "config.yaml"
    if not config_path.exists():
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_parquet_files(dataset_type: str) -> List[Path]:
    """Find all parquet files in a dataset's data directory."""
    data_dir = Path(dataset_type) / "data"
    if not data_dir.exists():
        return []

    return list(data_dir.glob("*.parquet"))


def regenerate_manifest_for_file(
    parquet_path: Path,
    dataset_type: str,
    config: Dict,
) -> Optional[Path]:
    """Regenerate manifest for a single parquet file."""
    print(f"\nProcessing: {parquet_path.name}")

    try:
        # Load the parquet file
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded {len(df)} conversations")

        # Get seed from config or use default
        seed = config.get("dataset", {}).get("seed", 42)

        # Build descriptive name from the filename
        base_name = parquet_path.stem.replace("_mooncake", "").replace("_chat", "")
        descriptive_name = base_name

        # Determine output files (all formats that exist)
        output_files = {}
        data_dir = parquet_path.parent

        # Check for corresponding JSONL files
        jsonl_path = data_dir / f"{parquet_path.stem}.jsonl"
        if jsonl_path.exists():
            output_files["aiperf_multi_turn"] = str(jsonl_path)

        mooncake_jsonl_path = data_dir / f"{parquet_path.stem}_mooncake.jsonl"
        if mooncake_jsonl_path.exists():
            output_files["aiperf_mooncake_trace"] = str(mooncake_jsonl_path)

        output_files["parquet"] = str(parquet_path)

        # Build manifest with payload scores
        manifest = build_manifest(
            df=df,
            config=config,
            dataset_type=dataset_type,
            seed=seed,
            output_files=output_files,
            descriptive_name=descriptive_name,
            include_payload_scores=True,  # Enable payload scoring
        )

        # Save manifest
        manifest_path = save_manifest(manifest, data_dir, descriptive_name)
        print(f"  ✓ Regenerated manifest: {manifest_path.name}")

        return manifest_path

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def regenerate_dataset_manifests(dataset_type: str) -> None:
    """Regenerate manifests for all parquet files in a dataset."""
    print(f"\n{'='*80}")
    print(f"Regenerating manifests for {dataset_type} dataset")
    print(f"{'='*80}")

    # Load configuration
    config = load_dataset_config(dataset_type)
    if not config:
        print(f"Warning: No config.yaml found for {dataset_type}, using defaults")

    # Find all parquet files
    parquet_files = find_parquet_files(dataset_type)
    if not parquet_files:
        print(f"No parquet files found for {dataset_type}")
        return

    print(f"Found {len(parquet_files)} parquet file(s)")

    # Regenerate manifest for each file
    success_count = 0
    for parquet_path in parquet_files:
        # Skip mooncake parquet files (they're derivatives)
        if "mooncake" in parquet_path.name:
            print(f"Skipping derivative: {parquet_path.name}")
            continue

        manifest_path = regenerate_manifest_for_file(parquet_path, dataset_type, config)
        if manifest_path:
            success_count += 1

    print(f"\n{'='*80}")
    print(f"Regenerated {success_count}/{len(parquet_files)} manifests for {dataset_type}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate manifests for existing datasets to include payload scores"
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset types to process (e.g., text reasoning agentic pdf image random repeat mixed)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available datasets"
    )
    args = parser.parse_args()

    if args.all:
        # Process all dataset directories
        dataset_dirs = [
            d for d in Path(".").iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != "examples"
        ]
        dataset_types = [d.name for d in dataset_dirs]
    else:
        dataset_types = args.datasets

    print("Manifest Regeneration Tool")
    print("This will regenerate manifests to include payload scores")
    print("="*80)

    for dataset_type in dataset_types:
        if not Path(dataset_type).exists():
            print(f"Skipping {dataset_type} (directory not found)")
            continue

        regenerate_dataset_manifests(dataset_type)

    print("\nDone!")


if __name__ == "__main__":
    main()
