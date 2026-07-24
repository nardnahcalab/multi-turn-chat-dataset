#!/usr/bin/env python3
"""Shared generation infrastructure for the multi-turn chat dataset generators.

Historically every ``<type>/generate.py`` copy-pasted the same scaffolding:
argument parsing, RNG/topic setup, turn-distribution sampling, the aiperf
``multi_turn`` / ``mooncake_trace`` JSONL writers, token estimation, and a
~110 line ``main()``. This module centralizes all of that so each generator
only has to supply its domain-specific pieces (templates and
``generate_conversation``).

The class-based generators (text, reasoning, random, repeat, pdf, image)
subclass :class:`BaseConversationGenerator`. The two structural outliers keep
their own control flow but still import the shared helpers:

* ``agentic`` samples turns with a different algorithm (functional style).
* ``mixed`` loads and reshapes sibling parquet files rather than generating.

Token estimation everywhere uses the single :data:`CHARS_PER_TOKEN` heuristic.
"""

import argparse
import json
import random
import uuid
from pathlib import Path

import pandas as pd
import yaml

from dataset_profile import (
    build_descriptive_name,
    build_manifest,
    print_profile_summary,
    save_manifest,
)

# ---------------------------------------------------------------------------
# Token estimation (single source of truth)
# ---------------------------------------------------------------------------

#: Rough characters-per-token heuristic used for all token estimates. This is a
#: deliberate approximation (no real tokenizer); see the project docs.
CHARS_PER_TOKEN = 4


def estimate_tokens(text) -> int:
    """Estimate token count from character length (the chars/4 heuristic)."""
    return len(str(text)) // CHARS_PER_TOKEN


def estimate_output_tokens(text) -> int:
    """Like :func:`estimate_tokens` but never returns 0 (min 1 decode token)."""
    return max(1, estimate_tokens(text))


# ---------------------------------------------------------------------------
# Shared aiperf JSONL writers
# ---------------------------------------------------------------------------

def default_multi_turn_text(content) -> str:
    """Flatten a (possibly multimodal) user message to plain text for a turn.

    The base implementation returns text content unchanged. Multimodal
    generators (pdf, image) override :meth:`BaseConversationGenerator.
    to_aiperf_multi_turn` to embed file/image references.
    """
    return content


# ---------------------------------------------------------------------------
# Base generator
# ---------------------------------------------------------------------------

class BaseConversationGenerator:
    """Common scaffolding shared by the class-based dataset generators.

    Subclasses must set :attr:`dataset_type` and implement
    :meth:`generate_conversation`. Text-style generators additionally rely on
    ``self.topics`` / ``self.topic_weights`` (populated automatically from
    ``config["topics"]`` when present).
    """

    #: Short dataset identifier, e.g. "text" (used for manifests / naming).
    dataset_type: str = "base"
    #: argparse description for the CLI.
    cli_description: str = "Generate a synthetic multi-turn chat dataset"
    #: Whether JSONL output escapes non-ASCII characters. Most generators emit
    #: raw UTF-8 (False); agentic historically escapes it (True).
    jsonl_ensure_ascii: bool = False

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        if "topics" in config:
            self.topics = config["topics"]
            self.topic_weights = [t["weight"] for t in self.topics]

    # -- sampling helpers ---------------------------------------------------

    def _pick_topic(self) -> dict:
        """Weighted random topic selection."""
        return self.rng.choices(self.topics, weights=self.topic_weights, k=1)[0]

    def _response_length_bucket(self, turn_index: int) -> str:
        """Determine response length bucket based on turn position."""
        dist_config = self.config["response_length"]["length_distribution_by_turn"]
        if turn_index < 5:
            dist = dist_config["early"]
        elif turn_index < 20:
            dist = dist_config["middle"]
        else:
            dist = dist_config["late"]
        buckets = list(dist.keys())
        weights = list(dist.values())
        return self.rng.choices(buckets, weights=weights, k=1)[0]

    def new_conversation_id(self) -> str:
        """Seed-reproducible UUIDv4 drawn from the generator RNG."""
        return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))

    def generate_conversation(self, num_turns: int) -> dict:
        raise NotImplementedError

    def generate_dataset(self, num_conversations: int = None) -> list[dict]:
        """Generate the full dataset.

        With ``num_conversations`` set, turns are drawn uniformly from
        ``config["turns"]["min"..max"]``. Otherwise the configured
        distribution buckets are used and the result is shuffled.
        """
        if num_conversations is not None:
            # Override: uniform random turns
            conversations = []
            turn_min = self.config["turns"]["min"]
            turn_max = self.config["turns"]["max"]
            for _ in range(num_conversations):
                n_turns = self.rng.randint(turn_min, turn_max)
                conversations.append(self.generate_conversation(n_turns))
            return conversations

        # Use configured distribution buckets
        conversations = []
        dist = self.config["turns"]["distribution"]
        for bucket_name, bucket_cfg in dist.items():
            count = bucket_cfg["count"]
            min_t = bucket_cfg["min_turns"]
            max_t = bucket_cfg["max_turns"]
            for _ in range(count):
                n_turns = self.rng.randint(min_t, max_t)
                conversations.append(self.generate_conversation(n_turns))

        self.rng.shuffle(conversations)
        return conversations

    # -- aiperf writers -----------------------------------------------------

    def to_aiperf_multi_turn(self, conversations: list[dict]) -> list[dict]:
        """aiperf ``multi_turn`` entries: user turns only, one line per convo."""
        entries = []
        for conv in conversations:
            messages = json.loads(conv["messages"])
            turns = []
            for msg in messages:
                if msg["role"] == "user":
                    turns.append({"text": default_multi_turn_text(msg["content"])})
            if turns:
                entries.append({
                    "session_id": conv["conversation_id"],
                    "turns": turns,
                })
        return entries

    def to_aiperf_mooncake(self, conversations: list[dict]) -> list[dict]:
        """aiperf ``mooncake_trace`` entries: full context per assistant turn."""
        entries = []
        for conv in conversations:
            session_id = conv["conversation_id"]
            messages = json.loads(conv["messages"])
            context = []
            for msg in messages:
                context.append(msg)
                if msg["role"] == "assistant":
                    entry = {
                        "session_id": session_id,
                        "messages": [m for m in context],
                        "output_length": estimate_output_tokens(msg["content"]),
                    }
                    turn_index = len([m for m in context if m["role"] == "assistant"])
                    if turn_index > 1:
                        entry["delay"] = 0
                    entries.append(entry)
        return entries

    # -- CLI orchestration --------------------------------------------------

    @classmethod
    def build_arg_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.cli_description)
        parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
        parser.add_argument("--num", type=int, default=None, help="Override number of conversations")
        parser.add_argument("--seed", type=int, default=None, help="Override random seed")
        parser.add_argument("--output", default=None, help="Override output path")
        parser.add_argument("--format", choices=["all", "parquet", "aiperf", "mooncake"],
                            default="all", help="Output format(s): parquet, aiperf (multi_turn "
                            "JSONL), mooncake (mooncake_trace JSONL), or all (default)")
        parser.add_argument("--name", default=None,
                            help="Custom suffix for descriptive output filenames")
        parser.add_argument("--descriptive-names", action="store_true", default=False,
                            help="Use descriptive filenames encoding count, seed, version, and date")
        parser.add_argument("--no-profile", action="store_true", default=False,
                            help="Skip generating the dataset manifest/profile JSON")
        cls.add_cli_args(parser)
        return parser

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Hook for subclasses to register extra CLI args (e.g. --skip-fetch)."""

    @classmethod
    def load_config(cls, args) -> dict:
        config_path = Path(args.config)
        if not config_path.exists():
            config_path = cls._module_dir() / "config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    @classmethod
    def _module_dir(cls) -> Path:
        """Directory of the concrete subclass module (for config/output paths)."""
        import sys
        module = sys.modules[cls.__module__]
        return Path(module.__file__).resolve().parent

    @classmethod
    def from_args(cls, config: dict, seed: int, args) -> "BaseConversationGenerator":
        """Construct the generator. Overridden by pdf/image to fetch sources."""
        return cls(config, seed=seed)

    @classmethod
    def main(cls, argv=None) -> None:
        args = cls.build_arg_parser().parse_args(argv)
        config = cls.load_config(args)
        seed = args.seed if args.seed is not None else config["dataset"]["seed"]

        generator = cls.from_args(config, seed, args)

        print(f"Generating conversations (seed={seed})...")
        conversations = generator.generate_dataset(num_conversations=args.num)
        print(f"Generated {len(conversations)} conversations")

        df = pd.DataFrame(conversations)
        generator.print_summary(df)
        generator.write_outputs(conversations, df, args, config, seed)

    def print_summary(self, df: pd.DataFrame) -> None:
        print("\n--- Dataset Summary ---")
        print(f"Total conversations: {len(df)}")
        if "num_turns" in df.columns:
            print(f"Turn count range: {df['num_turns'].min()} - {df['num_turns'].max()}")
            print(f"Mean turns: {df['num_turns'].mean():.1f}")
        group_col = "topic" if "topic" in df.columns else (
            "conversation_type" if "conversation_type" in df.columns else None)
        if group_col:
            print(f"{group_col} distribution:")
            for value, count in df[group_col].value_counts().items():
                print(f"  {value}: {count} ({100 * count / len(df):.1f}%)")
        if "estimated_tokens" in df.columns:
            print(f"Estimated total tokens: {df['estimated_tokens'].sum():,}")
            print(f"Mean tokens/conversation: {df['estimated_tokens'].mean():,.0f}")
            print(f"Max tokens (single conversation): {df['estimated_tokens'].max():,}")

    def write_outputs(self, conversations, df, args, config, seed) -> None:
        output_dir = (Path(args.output).parent if args.output
                      else self._module_dir() / config["dataset"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        actual_count = len(df)
        descriptive_name = build_descriptive_name(
            config, actual_count, seed, self.dataset_type, custom_suffix=args.name
        )
        if args.descriptive_names:
            file_base = descriptive_name
        else:
            file_base = config["dataset"]["output_filename"].replace(".parquet", "")

        fmt = args.format
        output_files = {}

        if fmt in ("all", "parquet"):
            parquet_path = (Path(args.output) if args.output and fmt == "parquet"
                            else output_dir / f"{file_base}.parquet")
            df.to_parquet(parquet_path, engine="pyarrow", index=False)
            size = parquet_path.stat().st_size / (1024 * 1024)
            output_files["parquet"] = str(parquet_path)
            print(f"\nParquet written to: {parquet_path} ({size:.2f} MB)")

        if fmt in ("all", "aiperf"):
            entries = self.to_aiperf_multi_turn(conversations)
            jsonl_path = output_dir / f"{file_base}.jsonl"
            _write_jsonl(jsonl_path, entries, self.jsonl_ensure_ascii)
            size = jsonl_path.stat().st_size / (1024 * 1024)
            output_files["aiperf_multi_turn"] = str(jsonl_path)
            print(f"aiperf multi_turn JSONL written to: {jsonl_path} ({size:.2f} MB)")
            print(f"  Usage: aiperf profile --input-file {jsonl_path} --custom-dataset-type multi_turn ...")

        if fmt in ("all", "mooncake"):
            entries = self.to_aiperf_mooncake(conversations)
            mooncake_path = output_dir / f"{file_base}_mooncake.jsonl"
            _write_jsonl(mooncake_path, entries, self.jsonl_ensure_ascii)
            size = mooncake_path.stat().st_size / (1024 * 1024)
            output_files["mooncake_trace"] = str(mooncake_path)
            print(f"aiperf mooncake_trace JSONL written to: {mooncake_path} ({size:.2f} MB)")
            print(f"  Usage: aiperf profile --input-file {mooncake_path} --custom-dataset-type mooncake_trace ...")

        if not args.no_profile:
            manifest = build_manifest(
                df=df,
                config=config,
                dataset_type=self.dataset_type,
                seed=seed,
                output_files=output_files,
                descriptive_name=descriptive_name,
            )
            manifest_path = save_manifest(manifest, output_dir, file_base)
            output_files["manifest"] = str(manifest_path)
            print(f"\nDataset manifest written to: {manifest_path}")
            print_profile_summary(manifest)


def _write_jsonl(path: Path, entries: list[dict], ensure_ascii: bool = False) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=ensure_ascii) + "\n")
