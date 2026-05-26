#!/usr/bin/env python3
"""
Shared module for dataset tagging, naming, and distribution profiling.

Provides four capabilities used by all generators:

1. **Tagging** - Attach descriptive metadata tags (both user-defined and
   auto-generated) to every dataset.
2. **Naming** - Build descriptive filenames that encode key dataset properties
   (count, seed, version, date).
3. **Distribution Profile** - Compute a comprehensive statistical summary
   (histograms, percentiles, distributions) and write it as a sidecar JSON
   manifest alongside the generated data files.
4. **Payload Scoring** - Calculate multi-dimensional payload scores representing
   prefill and decode effort for benchmarking comparisons.
"""

import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------

# Tags that are always generated from dataset characteristics
_COMMON_AUTO_TAGS = [
    "synthetic",
    "multi-turn",
    "benchmarking",
    "inference",
    "prefix-caching",
]


def generate_tags(
    df: pd.DataFrame,
    config: Dict[str, Any],
    dataset_type: str,
    extra_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a tag dictionary from config + computed dataset properties.

    Returns a dict with:
        user_tags   - tags explicitly listed in config.yaml
        auto_tags   - tags derived from dataset characteristics
        all_tags    - deduplicated union of both
    """
    user_tags: List[str] = list(config.get("tags", {}).get("custom", []))

    # Auto-generated tags
    auto_tags = list(_COMMON_AUTO_TAGS)
    auto_tags.append(dataset_type)

    num_convs = len(df)
    if num_convs <= 100:
        auto_tags.append("small")
    elif num_convs <= 500:
        auto_tags.append("medium")
    else:
        auto_tags.append("large")

    auto_tags.append(f"n{num_convs}")

    if "estimated_tokens" in df.columns:
        total_tok = int(df["estimated_tokens"].sum())
        if total_tok < 1_000_000:
            auto_tags.append("sub-1M-tokens")
        elif total_tok < 5_000_000:
            auto_tags.append("1M-5M-tokens")
        else:
            auto_tags.append("5M+-tokens")

    if "num_turns" in df.columns:
        max_turns = int(df["num_turns"].max())
        if max_turns <= 5:
            auto_tags.append("short-context")
        elif max_turns <= 15:
            auto_tags.append("medium-context")
        elif max_turns <= 30:
            auto_tags.append("long-context")
        else:
            auto_tags.append("very-long-context")

    # Dataset-type-specific auto tags
    if dataset_type == "agentic" and "num_tool_calls" in df.columns:
        auto_tags.append("tool-use")
        auto_tags.append("agent-tasks")
    if dataset_type in ("pdf", "image"):
        auto_tags.append("multimodal")
    if dataset_type == "reasoning":
        auto_tags.append("deep-reasoning")
    if dataset_type == "random":
        auto_tags.append("baseline")
        auto_tags.append("gibberish")
    if dataset_type == "repeat":
        auto_tags.append("repetitive")
        auto_tags.append("cache-stress")

    if extra_tags:
        auto_tags.extend(extra_tags)

    all_tags = sorted(set(user_tags + auto_tags))

    return {
        "user_tags": user_tags,
        "auto_tags": auto_tags,
        "all_tags": all_tags,
    }


# ---------------------------------------------------------------------------
# Descriptive Naming
# ---------------------------------------------------------------------------


def build_descriptive_name(
    config: Dict[str, Any],
    num_conversations: int,
    seed: int,
    dataset_type: str,
    custom_suffix: Optional[str] = None,
) -> str:
    """Return a descriptive base filename (no extension).

    Pattern: ``multi_turn_{type}_chat_n{count}_s{seed}_v{version}_{date}``

    Examples:
        multi_turn_text_chat_n500_s42_v1.0.0_20260429
        multi_turn_agentic_task_n1000_s123_v1.0.0_20260429
    """
    ds = config.get("dataset", {})
    version = ds.get("version", "1.0.0")
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    # Derive the base from the original output filename
    original = ds.get("output_filename", f"multi_turn_{dataset_type}_chat.parquet")
    base = original.replace(".parquet", "")

    parts = [base, f"n{num_conversations}", f"s{seed}", f"v{version}", date_str]
    if custom_suffix:
        parts.append(custom_suffix)

    return "_".join(parts)


# ---------------------------------------------------------------------------
# Distribution Profile
# ---------------------------------------------------------------------------


def _percentiles(values: List[float]) -> Dict[str, float]:
    """Compute common percentiles for a list of numeric values."""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _pct(p):
        k = (n - 1) * p / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

    return {
        "p5": round(_pct(5), 2),
        "p25": round(_pct(25), 2),
        "p50": round(_pct(50), 2),
        "p75": round(_pct(75), 2),
        "p95": round(_pct(95), 2),
        "p99": round(_pct(99), 2),
    }


def _histogram(values: List[float], num_bins: int = 10) -> List[Dict[str, Any]]:
    """Build a simple histogram (list of {range, count}) for numeric values."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [{"range": f"{lo}-{hi}", "count": len(values)}]
    bin_width = (hi - lo) / num_bins
    bins: List[Dict[str, Any]] = []
    for i in range(num_bins):
        edge_lo = lo + i * bin_width
        edge_hi = lo + (i + 1) * bin_width
        count = sum(1 for v in values if edge_lo <= v < edge_hi)
        if i == num_bins - 1:
            count = sum(1 for v in values if edge_lo <= v <= edge_hi)
        bins.append({
            "range": f"{edge_lo:.1f}-{edge_hi:.1f}",
            "count": count,
        })
    return bins


def _frequency_distribution(values: List[str]) -> List[Dict[str, Any]]:
    """Compute frequency counts and percentages for categorical values."""
    counter = Counter(values)
    total = len(values)
    return [
        {"value": k, "count": v, "percentage": round(100 * v / total, 2)}
        for k, v in counter.most_common()
    ]


def compute_distribution_profile(
    df: pd.DataFrame,
    config: Dict[str, Any],
    dataset_type: str,
) -> Dict[str, Any]:
    """Compute a comprehensive distribution profile for the dataset.

    Returns a nested dict suitable for JSON serialization containing:
        - turn_distribution
        - token_distribution
        - character_distribution
        - topic / type distribution
        - context_growth statistics
        - response_length analysis
        - dataset-type-specific metrics
    """
    profile: Dict[str, Any] = {}

    # --- Turn distribution ---
    turns = df["num_turns"].tolist()
    profile["turn_distribution"] = {
        "min": int(min(turns)),
        "max": int(max(turns)),
        "mean": round(statistics.mean(turns), 2),
        "median": round(statistics.median(turns), 2),
        "stdev": round(statistics.stdev(turns), 2) if len(turns) > 1 else 0.0,
        "percentiles": _percentiles(turns),
        "histogram": _histogram(turns, num_bins=min(10, int(max(turns) - min(turns) + 1))),
        "bucket_distribution": _turn_bucket_distribution(turns),
    }

    # --- Token distribution ---
    tokens = df["estimated_tokens"].tolist()
    profile["token_distribution"] = {
        "total": int(sum(tokens)),
        "min": int(min(tokens)),
        "max": int(max(tokens)),
        "mean": round(statistics.mean(tokens), 2),
        "median": round(statistics.median(tokens), 2),
        "stdev": round(statistics.stdev(tokens), 2) if len(tokens) > 1 else 0.0,
        "percentiles": _percentiles(tokens),
        "histogram": _histogram(tokens),
    }

    # --- Character distribution ---
    chars = df["total_characters"].tolist()
    profile["character_distribution"] = {
        "total": int(sum(chars)),
        "min": int(min(chars)),
        "max": int(max(chars)),
        "mean": round(statistics.mean(chars), 2),
        "median": round(statistics.median(chars), 2),
        "stdev": round(statistics.stdev(chars), 2) if len(chars) > 1 else 0.0,
        "percentiles": _percentiles(chars),
    }

    # --- Message count distribution ---
    msgs = df["num_messages"].tolist()
    profile["message_count_distribution"] = {
        "min": int(min(msgs)),
        "max": int(max(msgs)),
        "mean": round(statistics.mean(msgs), 2),
        "median": round(statistics.median(msgs), 2),
    }

    # --- Context growth ---
    profile["context_growth"] = _compute_context_growth(df)

    # --- Category distribution (topic / type) ---
    category_col = _get_category_column(dataset_type)
    if category_col and category_col in df.columns:
        profile["category_distribution"] = {
            "column": category_col,
            "distribution": _frequency_distribution(df[category_col].tolist()),
        }

    # --- Dataset-type-specific metrics ---
    if dataset_type == "agentic":
        profile["agentic_metrics"] = _agentic_profile(df)
    if dataset_type in ("pdf", "image"):
        profile["multimodal_info"] = _multimodal_profile(df, dataset_type)

    return profile


def _turn_bucket_distribution(turns: List[int]) -> List[Dict[str, Any]]:
    """Categorize turns into short/medium/long/very_long buckets."""
    buckets = {"short (1-5)": 0, "medium (6-15)": 0, "long (16-30)": 0, "very_long (31+)": 0}
    for t in turns:
        if t <= 5:
            buckets["short (1-5)"] += 1
        elif t <= 15:
            buckets["medium (6-15)"] += 1
        elif t <= 30:
            buckets["long (16-30)"] += 1
        else:
            buckets["very_long (31+)"] += 1
    total = len(turns)
    return [
        {"bucket": k, "count": v, "percentage": round(100 * v / total, 2) if total else 0}
        for k, v in buckets.items()
    ]


def _compute_context_growth(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze how context grows across turns."""
    growth_rates = []
    first_turn_chars = []
    last_turn_chars = []

    for _, row in df.iterrows():
        lengths = json.loads(row["cumulative_char_lengths"])
        if len(lengths) >= 2:
            first_turn_chars.append(lengths[0])
            last_turn_chars.append(lengths[-1])
            growth_rates.append(lengths[-1] / lengths[0] if lengths[0] > 0 else 0)

    result: Dict[str, Any] = {}
    if growth_rates:
        result["growth_ratio"] = {
            "min": round(min(growth_rates), 2),
            "max": round(max(growth_rates), 2),
            "mean": round(statistics.mean(growth_rates), 2),
            "median": round(statistics.median(growth_rates), 2),
        }
    if first_turn_chars:
        result["first_turn_chars"] = {
            "mean": round(statistics.mean(first_turn_chars), 2),
            "median": round(statistics.median(first_turn_chars), 2),
        }
    if last_turn_chars:
        result["last_turn_chars"] = {
            "mean": round(statistics.mean(last_turn_chars), 2),
            "median": round(statistics.median(last_turn_chars), 2),
        }
    return result


def _get_category_column(dataset_type: str) -> Optional[str]:
    """Return the primary category column name for a dataset type."""
    mapping = {
        "text": "topic",
        "reasoning": "topic",
        "random": "topic",
        "repeat": "topic",
        "agentic": "task_type",
        "pdf": "conversation_type",
        "image": "conversation_type",
    }
    return mapping.get(dataset_type)


def _agentic_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Extra metrics for the agentic dataset."""
    result: Dict[str, Any] = {}
    if "success_score" in df.columns:
        scores = df["success_score"].tolist()
        result["success_score"] = {
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "mean": round(statistics.mean(scores), 4),
            "median": round(statistics.median(scores), 4),
            "histogram": _histogram(scores, num_bins=5),
        }
    if "num_tool_calls" in df.columns:
        tc = df["num_tool_calls"].tolist()
        result["tool_calls"] = {
            "total": int(sum(tc)),
            "mean_per_conversation": round(statistics.mean(tc), 2),
            "max_per_conversation": int(max(tc)),
        }
    if "num_errors" in df.columns:
        errs = df["num_errors"].tolist()
        total_tc = int(df["num_tool_calls"].sum()) if "num_tool_calls" in df.columns else 1
        result["errors"] = {
            "total": int(sum(errs)),
            "error_rate": round(sum(errs) / total_tc * 100, 2) if total_tc else 0,
        }
    return result


def _multimodal_profile(df: pd.DataFrame, dataset_type: str) -> Dict[str, Any]:
    """Extra metrics for multimodal (pdf / image) datasets."""
    result: Dict[str, Any] = {}
    if dataset_type == "pdf":
        if "paper_arxiv_id" in df.columns:
            result["unique_papers"] = df["paper_arxiv_id"].nunique()
    if dataset_type == "image":
        if "image_topic" in df.columns:
            result["image_topic_distribution"] = _frequency_distribution(
                df["image_topic"].tolist()
            )
        if "image_url" in df.columns:
            result["unique_images"] = df["image_url"].nunique()
    return result


# ---------------------------------------------------------------------------
# Payload Scoring
# ---------------------------------------------------------------------------


def compute_payload_score(
    df: pd.DataFrame,
    dataset_type: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute multi-dimensional payload scores for a dataset.

    Payload scores represent the computational effort required for:
    - **Prefill**: Processing the input context (tokens, multimodal content)
    - **Decode**: Generating responses (output tokens, tool calls)

    Returns a dict with:
        - prefill_score: Normalized score representing prefill effort
        - decode_score: Normalized score representing decode effort
        - total_payload_score: Combined score
        - breakdown: Detailed factors contributing to each score
    """
    if config is None:
        config = {}

    # Initialize score components
    prefill_factors = {
        "total_tokens": 0.0,
        "avg_context_size": 0.0,
        "multimodal_penalty": 0.0,
        "context_growth": 0.0,
    }

    decode_factors = {
        "total_response_tokens": 0.0,
        "num_turns": 0.0,
        "avg_response_length": 0.0,
        "tool_call_penalty": 0.0,
    }

    # Calculate factors from the dataset
    if "estimated_tokens" in df.columns:
        total_tokens = int(df["estimated_tokens"].sum())
        prefill_factors["total_tokens"] = total_tokens

    if "total_characters" in df.columns:
        # Approximate token count from characters (4 chars ~= 1 token)
        total_chars = int(df["total_characters"].sum())
        prefill_factors["avg_context_size"] = total_chars / len(df) if len(df) > 0 else 0

    if "num_turns" in df.columns:
        turns = df["num_turns"].tolist()
        decode_factors["num_turns"] = sum(turns)
        decode_factors["avg_response_length"] = statistics.mean(turns) if turns else 0

    # Context growth analysis
    if "cumulative_char_lengths" in df.columns:
        growth_rates = []
        for _, row in df.iterrows():
            lengths = json.loads(row["cumulative_char_lengths"])
            if len(lengths) >= 2:
                growth_rates.append(lengths[-1] / lengths[0] if lengths[0] > 0 else 0)
        if growth_rates:
            prefill_factors["context_growth"] = statistics.mean(growth_rates)

    # Multimodal penalty (PDF and Image datasets have higher prefill cost)
    if dataset_type in ("pdf", "image"):
        prefill_factors["multimodal_penalty"] = 1.5  # 50% penalty for multimodal

    # Tool call penalty (agentic datasets have higher decode cost)
    if dataset_type == "agentic" and "num_tool_calls" in df.columns:
        tool_calls = int(df["num_tool_calls"].sum())
        decode_factors["tool_call_penalty"] = tool_calls * 0.1  # Each tool call adds 10% cost

    # Estimate response tokens (roughly half of total tokens are responses)
    if "estimated_tokens" in df.columns:
        total_tokens = int(df["estimated_tokens"].sum())
        decode_factors["total_response_tokens"] = total_tokens * 0.4  # Assume 40% are response tokens

    # Normalize factors to [0, 1] range (using reasonable max values as reference)
    # These reference values can be adjusted based on typical dataset scales
    prefill_normalized = _normalize_payload_factors(
        prefill_factors,
        {
            "total_tokens": 10_000_000,  # 10M tokens as reference max
            "avg_context_size": 50_000,  # 50K chars as reference max
            "multimodal_penalty": 2.0,  # Max penalty
            "context_growth": 10.0,  # 10x growth as reference max
        }
    )

    decode_normalized = _normalize_payload_factors(
        decode_factors,
        {
            "total_response_tokens": 5_000_000,  # 5M response tokens as reference max
            "num_turns": 10_000,  # 10K total turns as reference max
            "avg_response_length": 50,  # 50 turns as reference max
            "tool_call_penalty": 10.0,  # Max tool call penalty
        }
    )

    # Calculate weighted scores (weights can be customized in config)
    weights = config.get("payload_weights", {
        "prefill": {
            "total_tokens": 0.4,
            "avg_context_size": 0.3,
            "multimodal_penalty": 0.2,
            "context_growth": 0.1,
        },
        "decode": {
            "total_response_tokens": 0.4,
            "num_turns": 0.3,
            "avg_response_length": 0.2,
            "tool_call_penalty": 0.1,
        }
    })

    prefill_score = sum(
        prefill_normalized[k] * weights["prefill"].get(k, 0.25)
        for k in prefill_normalized
    )

    decode_score = sum(
        decode_normalized[k] * weights["decode"].get(k, 0.25)
        for k in decode_normalized
    )

    # Combined score (can be weighted differently based on use case)
    prefill_weight = config.get("prefill_decode_balance", 0.5)  # Default: equal weight
    total_payload_score = prefill_score * prefill_weight + decode_score * (1 - prefill_weight)

    return {
        "prefill_score": round(prefill_score, 4),
        "decode_score": round(decode_score, 4),
        "total_payload_score": round(total_payload_score, 4),
        "prefill_breakdown": {
            "normalized": prefill_normalized,
            "raw": prefill_factors,
        },
        "decode_breakdown": {
            "normalized": decode_normalized,
            "raw": decode_factors,
        },
    }


def _normalize_payload_factors(
    factors: Dict[str, float],
    reference_max: Dict[str, float],
) -> Dict[str, float]:
    """Normalize payload factors to [0, 1] range using reference max values."""
    normalized = {}
    for key, value in factors.items():
        ref_max = reference_max.get(key, 1.0)
        if ref_max > 0:
            normalized[key] = min(value / ref_max, 1.0)
        else:
            normalized[key] = 0.0
    return normalized


def add_payload_scores_to_manifest(
    manifest: Dict[str, Any],
    df: pd.DataFrame,
    dataset_type: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add payload scores to an existing manifest."""
    payload_scores = compute_payload_score(df, dataset_type, config)
    manifest["payload_scores"] = payload_scores
    return manifest


# ---------------------------------------------------------------------------
# Manifest Generation
# ---------------------------------------------------------------------------


def build_manifest(
    df: pd.DataFrame,
    config: Dict[str, Any],
    dataset_type: str,
    seed: int,
    output_files: Dict[str, str],
    descriptive_name: str,
    extra_tags: Optional[List[str]] = None,
    include_payload_scores: bool = True,
) -> Dict[str, Any]:
    """Build the complete dataset manifest combining tags, profile, and metadata.

    The manifest is a single JSON document that travels alongside the dataset files.
    """
    tags_info = generate_tags(df, config, dataset_type, extra_tags=extra_tags)
    profile = compute_distribution_profile(df, config, dataset_type)

    ds_cfg = config.get("dataset", {})

    manifest = {
        "manifest_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": ds_cfg.get("name", f"multi-turn-{dataset_type}-chat"),
            "descriptive_name": descriptive_name,
            "type": dataset_type,
            "version": ds_cfg.get("version", "1.0.0"),
            "description": ds_cfg.get("description", ""),
            "num_conversations": len(df),
            "seed": seed,
        },
        "tags": tags_info,
        "output_files": output_files,
        "distribution_profile": profile,
    }

    # Add payload scores if requested
    if include_payload_scores:
        manifest = add_payload_scores_to_manifest(manifest, df, dataset_type, config)

    return manifest


def save_manifest(manifest: Dict[str, Any], output_dir: Path, descriptive_name: str) -> Path:
    """Write the manifest JSON to disk and return the path."""
    manifest_path = output_dir / f"{descriptive_name}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path


def print_profile_summary(manifest: Dict[str, Any]) -> None:
    """Print a human-readable summary of the manifest to stdout."""
    ds = manifest["dataset"]
    tags = manifest["tags"]
    profile = manifest["distribution_profile"]

    print(f"\n{'='*70}")
    print(f"  DATASET MANIFEST: {ds['descriptive_name']}")
    print(f"{'='*70}")

    print(f"\n  Name:           {ds['name']}")
    print(f"  Type:           {ds['type']}")
    print(f"  Version:        {ds['version']}")
    print(f"  Conversations:  {ds['num_conversations']}")
    print(f"  Seed:           {ds['seed']}")
    print(f"  Generated:      {manifest['generated_at']}")

    print(f"\n  Tags: {', '.join(tags['all_tags'])}")

    td = profile["turn_distribution"]
    print(f"\n  Turn Distribution:")
    print(f"    Range: {td['min']} - {td['max']}, Mean: {td['mean']}, Median: {td['median']}")
    for bucket in td.get("bucket_distribution", []):
        bar = "#" * max(1, bucket["count"] // 5)
        print(f"    {bucket['bucket']:20s} {bucket['count']:4d} ({bucket['percentage']:5.1f}%) {bar}")

    tkd = profile["token_distribution"]
    print(f"\n  Token Distribution:")
    print(f"    Total: {tkd['total']:,}")
    print(f"    Range: {tkd['min']:,} - {tkd['max']:,}")
    print(f"    Mean: {tkd['mean']:,.0f}, Median: {tkd['median']:,.0f}")

    if "category_distribution" in profile:
        cd = profile["category_distribution"]
        print(f"\n  {cd['column'].replace('_', ' ').title()} Distribution:")
        for item in cd["distribution"]:
            bar = "#" * max(1, item["count"] // 5)
            print(f"    {item['value']:30s} {item['count']:4d} ({item['percentage']:5.1f}%) {bar}")

    cg = profile.get("context_growth", {})
    if "growth_ratio" in cg:
        gr = cg["growth_ratio"]
        print(f"\n  Context Growth (last/first turn):")
        print(f"    Mean ratio: {gr['mean']:.1f}x, Median: {gr['median']:.1f}x, Max: {gr['max']:.1f}x")

    if "agentic_metrics" in profile:
        am = profile["agentic_metrics"]
        if "success_score" in am:
            ss = am["success_score"]
            print(f"\n  Success Scores:")
            print(f"    Mean: {ss['mean']:.2f}, Median: {ss['median']:.2f}")
        if "tool_calls" in am:
            tc = am["tool_calls"]
            print(f"  Tool Calls: {tc['total']} total, {tc['mean_per_conversation']:.1f} avg/conversation")
        if "errors" in am:
            print(f"  Error Rate: {am['errors']['error_rate']:.1f}%")

    files = manifest.get("output_files", {})
    if files:
        print(f"\n  Output Files:")
        for fmt, path in files.items():
            print(f"    [{fmt}] {path}")

    # Print payload scores if available
    if "payload_scores" in manifest:
        ps = manifest["payload_scores"]
        print(f"\n  Payload Scores:")
        print(f"    Prefill Score:  {ps['prefill_score']:.4f}")
        print(f"    Decode Score:   {ps['decode_score']:.4f}")
        print(f"    Total Payload:  {ps['total_payload_score']:.4f}")

    print(f"\n{'='*70}\n")
