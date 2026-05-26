#!/usr/bin/env python3
"""
Mixed dataset generator for multi-turn chat benchmarking.

Combines conversations from multiple dataset types (text, pdf, image, reasoning, agentic)
into a single mixed dataset based on configurable weights. Designed for comprehensive
benchmarking across different modalities and task types.

Usage:
    python generate.py                     # uses default config.yaml
    python generate.py --config my.yaml    # custom config
    python generate.py --num 1000          # override conversation count
"""

import argparse
import json
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Add project root to path for shared module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_profile import (
    build_descriptive_name,
    build_manifest,
    print_profile_summary,
    save_manifest,
    compute_payload_score,
    add_payload_scores_to_manifest,
)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_source_dataset(
    source_config: Dict[str, Any],
    project_root: Path,
) -> Optional[pd.DataFrame]:
    """Load a source dataset from parquet file.

    Returns None if the file doesn't exist and the source is not required.
    Raises an error if the file doesn't exist and the source is required.
    """
    path = project_root / source_config["path"]

    if not path.exists():
        if source_config.get("required", False):
            raise FileNotFoundError(
                f"Required source dataset not found: {path}. "
                f"Generate it first or mark it as not required in config."
            )
        else:
            print(f"  Skipping optional source (not found): {source_config['type']}")
            return None

    print(f"  Loading {source_config['type']} from {path}")
    df = pd.read_parquet(path)

    # Add source dataset column
    df["source_dataset"] = source_config["type"]

    return df


def load_all_sources(
    config: Dict[str, Any],
    project_root: Path,
) -> List[Tuple[str, pd.DataFrame]]:
    """Load all available source datasets.

    Returns a list of (dataset_type, dataframe) tuples.
    """
    sources_config = config.get("sources", [])
    loaded_sources = []

    print(f"\nLoading source datasets:")
    for source_config in sources_config:
        df = load_source_dataset(source_config, project_root)
        if df is not None:
            loaded_sources.append((source_config["type"], df))
            print(f"    Loaded {len(df)} conversations from {source_config['type']}")
        else:
            print(f"    Skipped {source_config['type']} (not available)")

    if not loaded_sources:
        raise ValueError("No source datasets could be loaded. Check your config and generate required datasets first.")

    return loaded_sources


def normalize_weights(
    sources_config: List[Dict[str, Any]],
    loaded_types: List[str],
) -> Dict[str, float]:
    """Normalize weights based on available sources.

    If some sources are unavailable, redistribute their weights proportionally
    among available sources.
    """
    original_weights = {s["type"]: s["weight"] for s in sources_config if s["type"] in loaded_types}
    total_weight = sum(original_weights.values())

    if total_weight == 0:
        # Equal weights if all weights are zero
        return {t: 1.0 / len(loaded_types) for t in loaded_types}

    return {t: w / total_weight for t, w in original_weights.items()}


def sample_conversations(
    loaded_sources: List[Tuple[str, pd.DataFrame]],
    config: Dict[str, Any],
    num_conversations: int,
    seed: int,
) -> pd.DataFrame:
    """Sample conversations from source datasets according to configured weights.

    Args:
        loaded_sources: List of (dataset_type, dataframe) tuples
        config: Full configuration dict
        num_conversations: Target number of conversations in mixed dataset
        seed: Random seed for reproducibility

    Returns:
        Combined dataframe with sampled conversations
    """
    rng = random.Random(seed)
    sources_config = config.get("sources", [])
    sampling_config = config.get("sampling", {})
    strategy = sampling_config.get("strategy", "weighted")

    loaded_types = [t for t, _ in loaded_sources]
    loaded_dict = {t: df for t, df in loaded_sources}

    print(f"\nSampling {num_conversations} conversations using '{strategy}' strategy:")

    if strategy == "equal":
        # Sample equal number from each available source
        count_per_source = num_conversations // len(loaded_sources)
        remainder = num_conversations % len(loaded_sources)

        sampled_dfs = []
        for i, (dataset_type, df) in enumerate(loaded_sources):
            count = count_per_source + (1 if i < remainder else 0)
            if count > len(df):
                print(f"  Warning: Requested {count} from {dataset_type} but only {len(df)} available")
                count = len(df)

            sampled = df.sample(n=count, random_state=seed + i)
            sampled_dfs.append(sampled)
            print(f"    Sampled {count} from {dataset_type}")

        combined = pd.concat(sampled_dfs, ignore_index=True)

    elif strategy == "weighted":
        # Sample according to weights
        normalize = sampling_config.get("normalize_weights", True)

        if normalize:
            weights = normalize_weights(sources_config, loaded_types)
        else:
            weights = {s["type"]: s["weight"] for s in sources_config if s["type"] in loaded_types}

        # Calculate counts per source
        counts = {}
        for dataset_type in loaded_types:
            counts[dataset_type] = int(num_conversations * weights[dataset_type])

        # Adjust for rounding
        total_allocated = sum(counts.values())
        if total_allocated < num_conversations:
            # Add remainder to the source with highest weight
            max_type = max(weights, key=weights.get)
            counts[max_type] += num_conversations - total_allocated

        # Sample from each source
        sampled_dfs = []
        for dataset_type, count in counts.items():
            df = loaded_dict[dataset_type]
            if count > len(df):
                print(f"  Warning: Requested {count} from {dataset_type} but only {len(df)} available")
                count = len(df)

            if count > 0:
                sampled = df.sample(n=count, random_state=seed)
                sampled_dfs.append(sampled)
                print(f"    Sampled {count} from {dataset_type} (weight: {weights[dataset_type]:.2f})")

        combined = pd.concat(sampled_dfs, ignore_index=True)

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    # Shuffle the combined dataset
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"  Total sampled: {len(combined)} conversations")

    return combined


def standardize_schema(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Standardize schema across different dataset types.

    Different datasets may have different columns. This function:
    - Adds missing columns with default values
    - Ensures consistent data types
    - Handles dataset-specific columns appropriately
    """
    output_config = config.get("output", {})
    schema_handling = output_config.get("schema_handling", "union")

    # Define the standard column set (union of all possible columns)
    standard_columns = {
        "conversation_id": str,
        "num_turns": int,
        "num_messages": int,
        "system_prompt": str,
        "messages": str,  # JSON string
        "total_characters": int,
        "estimated_tokens": int,
        "cumulative_char_lengths": str,  # JSON string
        "source_dataset": str,  # Added by this generator
    }

    # Dataset-specific columns
    optional_columns = {
        "text": ["topic"],
        "reasoning": ["topic"],
        "random": ["topic"],
        "repeat": ["topic"],
        "agentic": ["task_type", "tool_calls", "success_metric", "success_score", "num_tool_calls", "num_errors"],
        "pdf": ["conversation_type", "paper_arxiv_id", "paper_title", "paper_primary_category"],
        "image": ["conversation_type", "image_topic", "image_url"],
    }

    # Ensure all standard columns exist
    for col, dtype in standard_columns.items():
        if col not in df.columns:
            if col == "conversation_id":
                df[col] = [str(uuid.uuid4()) for _ in range(len(df))]
            elif col in ["system_prompt"]:
                df[col] = ""
            elif col in ["messages", "cumulative_char_lengths"]:
                df[col] = "[]"
            elif dtype == int:
                df[col] = 0
            else:
                df[col] = ""

    # Handle optional columns based on schema_handling
    if schema_handling == "union":
        # Keep all columns from all datasets
        # Fill missing optional columns with default values
        for dataset_type, cols in optional_columns.items():
            for col in cols:
                if col not in df.columns:
                    if col in ["success_score"]:
                        df[col] = 0.0
                    elif col in ["num_tool_calls", "num_errors"]:
                        df[col] = 0
                    else:
                        df[col] = ""

    elif schema_handling == "intersection":
        # Only keep columns that exist in all datasets
        # This is more restrictive but ensures consistent schema
        common_cols = set(standard_columns.keys())
        for dataset_type, cols in optional_columns.items():
            # Only keep optional columns that exist in the current df
            for col in cols:
                if col in df.columns:
                    common_cols.add(col)

        # Drop columns not in common set
        cols_to_drop = [col for col in df.columns if col not in common_cols]
        df = df.drop(columns=cols_to_drop)

    return df


def export_aiperf_multi_turn(df: pd.DataFrame, output_path: Path) -> None:
    """Export to aiperf multi_turn JSONL format (user turns only)."""
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            messages = json.loads(row["messages"])
            # Extract user messages only
            user_turns = []
            for msg in messages:
                if msg["role"] == "user":
                    content = msg["content"]
                    # Handle multimodal content
                    if isinstance(content, list):
                        # For multimodal, extract text or use a placeholder
                        text_content = ""
                        for item in content:
                            if item.get("type") == "text":
                                text_content += item.get("text", "")
                            elif item.get("type") == "file":
                                text_content += f"[FILE: {item['file'].get('url', 'unknown')}] "
                            elif item.get("type") == "image_url":
                                text_content += f"[IMAGE: {item.get('image_url', {}).get('url', 'unknown')}] "
                        user_turns.append({"text": text_content.strip()})
                    else:
                        user_turns.append({"text": content})

            session_id = row.get("conversation_id", str(uuid.uuid4()))
            line = {"session_id": session_id, "turns": user_turns}
            f.write(json.dumps(line) + "\n")


def export_aiperf_mooncake(df: pd.DataFrame, output_path: Path) -> None:
    """Export to aiperf mooncake_trace JSONL format (full message arrays per turn)."""
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            messages = json.loads(row["messages"])
            cumulative_lengths = json.loads(row["cumulative_char_lengths"])

            # Export each turn with the complete message array up to that point
            for i, length in enumerate(cumulative_lengths):
                # Messages up to and including the current turn
                # Each turn consists of user + assistant pair
                turn_messages = messages[: (i + 1) * 2 + 1]  # +1 for system prompt

                session_id = row.get("conversation_id", str(uuid.uuid4()))
                line = {
                    "session_id": session_id,
                    "messages": turn_messages,
                    "output_length": length,
                }
                f.write(json.dumps(line) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate mixed multi-turn chat dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--num", type=int, default=None, help="Override number of conversations")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output", type=str, default=None, help="Override output directory")
    parser.add_argument("--format", type=str, default="all",
                        choices=["all", "parquet", "aiperf", "mooncake"],
                        help="Output format(s)")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with CLI arguments
    if args.num:
        config["dataset"]["num_conversations"] = args.num
    if args.seed:
        config["dataset"]["seed"] = args.seed
    if args.output:
        config["dataset"]["output_dir"] = args.output

    dataset_cfg = config["dataset"]
    num_conversations = dataset_cfg["num_conversations"]
    seed = dataset_cfg["seed"]
    output_dir = Path(dataset_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating mixed dataset with {num_conversations} conversations (seed={seed})")

    # Load source datasets
    project_root = Path(__file__).resolve().parent.parent
    loaded_sources = load_all_sources(config, project_root)

    # Sample conversations
    combined_df = sample_conversations(loaded_sources, config, num_conversations, seed)

    # Standardize schema
    combined_df = standardize_schema(combined_df, config)

    # Generate descriptive name
    descriptive_name = build_descriptive_name(
        config, num_conversations, seed, "mixed", custom_suffix="mixed"
    )

    # Export to requested formats
    output_files = {}

    if args.format in ["all", "parquet"]:
        parquet_path = output_dir / f"{descriptive_name}.parquet"
        combined_df.to_parquet(parquet_path, index=False)
        output_files["parquet"] = str(parquet_path)
        print(f"  Exported Parquet to {parquet_path}")

    if args.format in ["all", "aiperf"]:
        aiperf_path = output_dir / f"{descriptive_name}.jsonl"
        export_aiperf_multi_turn(combined_df, aiperf_path)
        output_files["aiperf_multi_turn"] = str(aiperf_path)
        print(f"  Exported aiperf multi_turn JSONL to {aiperf_path}")

    if args.format in ["all", "mooncake"]:
        mooncake_path = output_dir / f"{descriptive_name}_mooncake.jsonl"
        export_aiperf_mooncake(combined_df, mooncake_path)
        output_files["aiperf_mooncake_trace"] = str(mooncake_path)
        print(f"  Exported aiperf mooncake_trace JSONL to {mooncake_path}")

    # Build and save manifest
    manifest = build_manifest(
        combined_df,
        config,
        "mixed",
        seed,
        output_files,
        descriptive_name,
        extra_tags=["mixed-dataset", "multi-source"],
        include_payload_scores=config.get("payload_scoring", {}).get("enabled", True),
    )

    manifest_path = save_manifest(manifest, output_dir, descriptive_name)
    print(f"  Saved manifest to {manifest_path}")

    # Print summary
    print_profile_summary(manifest)


if __name__ == "__main__":
    main()
