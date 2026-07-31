#!/usr/bin/env python3
"""
Synthetic multi-turn random-text conversation generator that matches the token 
distribution of the text dataset.

This generates random content (words, characters, sentences, mixed content, lorem ipsum)
but targets the same per-conversation token counts as the text dataset for fair comparison.

Usage:
    python generate_similar.py                     # uses default config
    python generate_similar.py --num 1000          # override conversation count
    python generate_similar.py --seed 42           # override random seed
"""

import json
import sys
from pathlib import Path

# Add project root to path for shared module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generator_base import BaseConversationGenerator, count_message_tokens, count_tokens

# Import word pools from the random generator module
import importlib.util
spec = importlib.util.spec_from_file_location("random_generate", str(Path(__file__).resolve().parent / "generate.py"))
random_generate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(random_generate)

VOCABULARY = random_generate.VOCABULARY
SUBJECTS = random_generate.SUBJECTS
VERBS = random_generate.VERBS
OBJECTS = random_generate.OBJECTS
ADVERBS = random_generate.ADVERBS
PREPOSITIONAL_PHRASES = random_generate.PREPOSITIONAL_PHRASES
LOREM_WORDS = random_generate.LOREM_WORDS
RESPONSE_FILLERS = random_generate.RESPONSE_FILLERS
SYMBOLS = random_generate.SYMBOLS

# Import the random content generators
import random
import string


class SimilarTokenRandomGenerator(BaseConversationGenerator):
    dataset_type = "random_similar"
    cli_description = "Generate random chat dataset matching text dataset token distribution"

    def __init__(self, config: dict, seed: int = 42, target_tokens: list[int] = None):
        super().__init__(config, seed)
        self.target_tokens = target_tokens or []
        self.current_target_index = 0

    def _get_next_target_tokens(self) -> int:
        """Get the next target token count from the text dataset distribution."""
        if not self.target_tokens:
            # Fallback to random reasonable token count
            return self.rng.randint(500, 15000)
        
        target = self.target_tokens[self.current_target_index]
        self.current_target_index = (self.current_target_index + 1) % len(self.target_tokens)
        return target

    # ---- Random content generators (same as original) ----

    def _gen_random_words(self, min_words: int = 3, max_words: int = 30) -> str:
        """Generate a sequence of random words from the vocabulary pool."""
        min_words, max_words = max(1, min_words), max(min_words, max_words)
        n = self.rng.randint(min_words, max_words)
        return " ".join(self.rng.choice(VOCABULARY) for _ in range(n))

    def _gen_random_chars(self, min_len: int = 10, max_len: int = 200) -> str:
        """Generate a random alphanumeric string of varying length."""
        min_len, max_len = max(1, min_len), max(min_len, max_len)
        length = self.rng.randint(min_len, max_len)
        charset = string.ascii_letters + string.digits
        chunk_size = self.rng.randint(4, 12)
        chars = []
        for i in range(length):
            if i > 0 and i % chunk_size == 0:
                chars.append(" ")
                chunk_size = self.rng.randint(4, 12)
            chars.append(self.rng.choice(charset))
        return "".join(chars)

    def _gen_random_sentence(self) -> str:
        """Construct a single grammatically structured but semantically random sentence."""
        subject = self.rng.choice(SUBJECTS)
        verb = self.rng.choice(VERBS)
        obj = self.rng.choice(OBJECTS)

        parts = [subject]
        if self.rng.random() < 0.4:
            parts.append(self.rng.choice(ADVERBS))
        parts.append(verb)
        parts.append(obj)
        if self.rng.random() < 0.5:
            parts.append(self.rng.choice(PREPOSITIONAL_PHRASES))

        return " ".join(parts) + "."

    def _gen_random_sentences(self, min_sentences: int = 1, max_sentences: int = 6) -> str:
        """Generate multiple random sentences."""
        min_sentences, max_sentences = max(1, min_sentences), max(min_sentences, max_sentences)
        n = self.rng.randint(min_sentences, max_sentences)
        return " ".join(self._gen_random_sentence() for _ in range(n))

    def _gen_random_mixed(self, min_parts: int = 5, max_parts: int = 25) -> str:
        """Generate mixed content: words, numbers, and symbols."""
        min_parts, max_parts = max(1, min_parts), max(min_parts, max_parts)
        n = self.rng.randint(min_parts, max_parts)
        parts = []
        for _ in range(n):
            kind = self.rng.choices(["word", "number", "symbol", "chars"], weights=[0.4, 0.25, 0.15, 0.2])[0]
            if kind == "word":
                parts.append(self.rng.choice(VOCABULARY))
            elif kind == "number":
                fmt = self.rng.choice(["int", "float", "hex", "big"])
                if fmt == "int":
                    parts.append(str(self.rng.randint(0, 9999)))
                elif fmt == "float":
                    parts.append(f"{self.rng.uniform(-1000, 1000):.{self.rng.randint(1,6)}f}")
                elif fmt == "hex":
                    parts.append(f"0x{self.rng.randint(0, 0xFFFFFF):06X}")
                else:
                    parts.append(str(self.rng.randint(100000, 99999999)))
                    
            elif kind == "symbol":
                sym_count = self.rng.randint(1, 4)
                parts.append("".join(self.rng.choice(SYMBOLS) for _ in range(sym_count)))
            else:
                length = self.rng.randint(3, 10)
                charset = string.ascii_letters + string.digits
                parts.append("".join(self.rng.choice(charset) for _ in range(length)))
        return " ".join(parts)

    def _gen_random_lorem(self, min_words: int = 10, max_words: int = 60) -> str:
        """Generate lorem-ipsum style pseudo-Latin text."""
        min_words, max_words = max(1, min_words), max(min_words, max_words)
        n = self.rng.randint(min_words, max_words)
        words = [self.rng.choice(LOREM_WORDS) for _ in range(n)]
        sentences = []
        i = 0
        while i < len(words):
            sent_len = self.rng.randint(4, 12)
            sent_words = words[i:i + sent_len]
            if sent_words:
                sent_words[0] = sent_words[0].capitalize()
                if len(sent_words) > 5 and self.rng.random() < 0.4:
                    comma_pos = self.rng.randint(2, len(sent_words) - 2)
                    sent_words[comma_pos] = sent_words[comma_pos] + ","
                sentences.append(" ".join(sent_words) + ".")
            i += sent_len
        return " ".join(sentences)

    # ---- Content generation with token targeting ----

    def _generate_user_content(self, topic_name: str, target_tokens: int) -> str:
        """Generate user message content targeting specific token count."""
        # Random content is less token-dense than natural language
        # Scale up targets to compensate (random content needs ~1.5-2x more characters for same tokens)
        compensation_factor = 1.7
        adjusted_tokens = int(target_tokens * compensation_factor)
        target_words = max(3, adjusted_tokens // 4)
        
        if topic_name == "random_words":
            return self._gen_random_words(max(3, target_words // 2), target_words * 2)
        elif topic_name == "random_chars":
            target_chars = adjusted_tokens * 4
            return self._gen_random_chars(max(10, target_chars // 2), target_chars)
        elif topic_name == "random_sentences":
            target_sentences = max(1, target_words // 8)
            return self._gen_random_sentences(max(1, target_sentences), target_sentences * 3)
        elif topic_name == "random_mixed":
            target_parts = max(5, target_words // 2)
            return self._gen_random_mixed(max(5, target_parts // 2), target_parts * 2)
        elif topic_name == "random_lorem":
            return self._gen_random_lorem(max(10, target_words), target_words * 2)
        else:
            return self._gen_random_words(max(3, target_words // 2), target_words * 2)

    def _generate_response_content(self, topic_name: str, target_tokens: int) -> str:
        """Generate assistant response targeting specific token count."""
        # Apply compensation factor for random content lower token density
        compensation_factor = 1.7
        adjusted_tokens = int(target_tokens * compensation_factor)
        target_words = max(10, adjusted_tokens // 4)
        parts = []
        word_count = 0

        while word_count < target_words:
            strategy = self.rng.choices(
                ["filler", "echo_random", "list", "numbered"],
                weights=[0.35, 0.35, 0.15, 0.15],
            )[0]

            if strategy == "filler":
                sentence = self.rng.choice(RESPONSE_FILLERS)
                parts.append(sentence)
                word_count += len(sentence.split())

            elif strategy == "echo_random":
                remaining_words = target_words - word_count
                if topic_name == "random_words":
                    min_w, max_w = 5, min(30, remaining_words * 2)
                    chunk = self._gen_random_words(min(3, min_w), max(5, max_w))
                elif topic_name == "random_chars":
                    min_c, max_c = 15, min(120, remaining_words * 4)
                    chunk = self._gen_random_chars(max(10, min_c), max(20, max_c))
                elif topic_name == "random_sentences":
                    min_s, max_s = 1, min(5, max(1, remaining_words // 4))
                    chunk = self._gen_random_sentences(min_s, max_s)
                elif topic_name == "random_mixed":
                    min_p, max_p = 5, min(25, remaining_words)
                    chunk = self._gen_random_mixed(min(3, min_p), max(5, max_p))
                elif topic_name == "random_lorem":
                    min_w, max_w = 8, min(50, remaining_words * 2)
                    chunk = self._gen_random_lorem(max(5, min_w), max(10, max_w))
                else:
                    min_w, max_w = 5, min(25, remaining_words * 2)
                    chunk = self._gen_random_words(max(3, min_w), max(5, max_w))
                parts.append(chunk)
                word_count += len(chunk.split())

            elif strategy == "list":
                remaining_words = target_words - word_count
                n_items = min(8, max(2, remaining_words // 6))
                items = []
                for _ in range(n_items):
                    item = self._gen_random_words(3, min(12, remaining_words // len(items) if items else remaining_words))
                    items.append(f"- {item}")
                    word_count += len(item.split())
                parts.append("\n".join(items))

            else:  # numbered
                remaining_words = target_words - word_count
                n_items = min(6, max(1, remaining_words // 15))
                items = []
                for idx in range(1, n_items + 1):
                    item = self.rng.choice(RESPONSE_FILLERS)
                    items.append(f"{idx}. {item}")
                    word_count += len(item.split())
                parts.append("\n".join(items))

        text = "\n\n".join(parts)

        # Trim if significantly over target
        words = text.split()
        if len(words) > target_words * 1.5:
            words = words[:int(target_words * 1.2)]
            text = " ".join(words)
            last_period = text.rfind(".")
            if last_period > len(text) * 0.7:
                text = text[:last_period + 1]

        return text

    def generate_conversation(self, num_turns: int) -> dict:
        """Generate a single multi-turn conversation with random content targeting text dataset tokens."""
        topic = self._pick_topic()
        topic_name = topic["name"]
        system_prompt = topic["system_prompt"]

        # Get target total tokens for this conversation
        target_total_tokens = self._get_next_target_tokens()
        
        messages = [{"role": "system", "content": system_prompt}]
        cumulative_char_lengths = []
        running_chars = len(system_prompt)
        
        # Use exponential growth pattern similar to real conversations
        # Later turns are typically longer
        if num_turns == 1:
            # Single turn - allocate all tokens
            user_target = max(50, target_total_tokens // 3)
            assistant_target = max(100, target_total_tokens - user_target)
        else:
            # Multi-turn - use exponential distribution
            # Early turns: shorter, Later turns: longer
            import math
            total_pairs = num_turns
            
            # Generate exponential weights for each turn
            weights = [math.exp(0.3 * i) for i in range(total_pairs)]
            total_weight = sum(weights)
            
            # Allocate tokens per turn pair (user + assistant)
            tokens_per_pair = []
            for i in range(total_pairs):
                pair_tokens = (weights[i] / total_weight) * target_total_tokens
                tokens_per_pair.append(max(100, int(pair_tokens)))  # Minimum 100 tokens per pair
            
            # Adjust to match total target
            current_total = sum(tokens_per_pair)
            if current_total != target_total_tokens:
                # Scale to match
                scale = target_total_tokens / current_total
                tokens_per_pair = [max(50, int(t * scale)) for t in tokens_per_pair]
        
        for turn_idx in range(num_turns):
            # Get token allocation for this turn
            if num_turns == 1:
                user_target = max(50, target_total_tokens // 3)
                assistant_target = max(100, target_total_tokens - user_target)
            else:
                pair_tokens = tokens_per_pair[turn_idx]
                user_target = max(30, pair_tokens // 3)
                assistant_target = max(50, pair_tokens - user_target)
            
            # User message
            user_msg = self._generate_user_content(topic_name, user_target)
            messages.append({"role": "user", "content": user_msg})
            running_chars += len(user_msg)

            # Assistant response  
            assistant_msg = self._generate_response_content(topic_name, assistant_target)
            messages.append({"role": "assistant", "content": assistant_msg})
            running_chars += len(assistant_msg)

            cumulative_char_lengths.append(running_chars)

        conversation_id = self.new_conversation_id()

        return {
            "conversation_id": conversation_id,
            "topic": topic_name,
            "num_turns": num_turns,
            "num_messages": len(messages),
            "system_prompt": system_prompt,
            "messages": json.dumps(messages),
            "total_characters": running_chars,
            "estimated_tokens": count_message_tokens(messages),
            "cumulative_char_lengths": json.dumps(cumulative_char_lengths),
            "target_tokens": target_total_tokens,  # Track what we aimed for
        }


def load_text_dataset_tokens() -> list[int]:
    """Load actual token counts from the text dataset parquet file."""
    text_parquet_path = Path(__file__).resolve().parent.parent / "text" / "data" / "multi_turn_text_chat.parquet"
    
    if not text_parquet_path.exists():
        print(f"Warning: Text dataset parquet not found at {text_parquet_path}")
        print("Using fallback token distribution from manifest")
        return load_text_dataset_tokens_from_manifest()
    
    try:
        import pandas as pd
        df = pd.read_parquet(text_parquet_path)
        if "estimated_tokens" in df.columns:
            tokens = df["estimated_tokens"].tolist()
            print(f"Loaded {len(tokens)} actual token counts from text dataset")
            return tokens
        else:
            print("estimated_tokens column not found, using manifest distribution")
            return load_text_dataset_tokens_from_manifest()
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        print("Using fallback token distribution from manifest")
        return load_text_dataset_tokens_from_manifest()

def load_text_dataset_tokens_from_manifest() -> list[int]:
    """Load token counts from the text dataset manifest as fallback."""
    text_manifest_path = Path(__file__).resolve().parent.parent / "text" / "data" / "multi_turn_text_chat_manifest.json"
    
    if not text_manifest_path.exists():
        print(f"Warning: Text dataset manifest not found at {text_manifest_path}")
        print("Using fallback token distribution")
        return [5000 + i * 100 for i in range(500)]  # Fallback linear distribution
    
    with open(text_manifest_path) as f:
        manifest = json.load(f)
    
    # Sample from the distribution described in manifest
    import numpy as np
    token_dist = manifest["distribution_profile"]["token_distribution"]
    
    # Sample from a normal distribution with the given parameters
    mean = token_dist["mean"]
    stdev = token_dist["stdev"]
    min_tokens = token_dist["min"]
    max_tokens = token_dist["max"]
    
    # Generate 500 token counts matching the distribution
    rng = np.random.default_rng(42)  # Use same seed for reproducibility
    samples = rng.normal(mean, stdev, 500)
    samples = np.clip(samples, min_tokens, max_tokens).astype(int)
    
    return samples.tolist()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate random chat dataset matching text dataset token distribution")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--num", type=int, default=None, help="Override number of conversations")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output", default=None, help="Override output path")
    parser.add_argument("--format", choices=["all", "parquet", "aiperf", "mooncake"],
                        default="all", help="Output format(s)")
    args = parser.parse_args()
    
    # Load base config from random directory
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent / "config.yaml"
    
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Modify dataset name
    config["dataset"]["name"] = "multi-turn-random-similar-chat"
    config["dataset"]["description"] = "Synthetic multi-turn random-text conversations matching text dataset token distribution"
    config["dataset"]["output_filename"] = "multi_turn_random_similar_chat.parquet"
    
    # Update tags
    config["tags"]["custom"] = ["baseline", "random-text", "token-matched", "cache-baseline"]
    
    seed = args.seed if args.seed is not None else config["dataset"]["seed"]
    
    # Load target tokens from text dataset
    target_tokens = load_text_dataset_tokens()
    
    # Create generator
    generator = SimilarTokenRandomGenerator(config, seed, target_tokens)
    
    print(f"Generating conversations with token distribution matching text dataset (seed={seed})...")
    conversations = generator.generate_dataset(num_conversations=args.num)
    print(f"Generated {len(conversations)} conversations")
    
    import pandas as pd
    df = pd.DataFrame(conversations)
    generator.print_summary(df)
    
    # Override output filename
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = generator._module_dir() / config["dataset"]["output_dir"]
        output_path = output_dir / "multi_turn_random_similar_chat.parquet"
    
    # Create custom args for write_outputs
    class CustomArgs:
        def __init__(self):
            self.output = str(output_path)
            self.format = args.format
            self.descriptive_names = False
            self.no_profile = False
            self.name = "similar"
    
    custom_args = CustomArgs()
    generator.write_outputs(conversations, df, custom_args, config, seed)
    
    # Rename manifest to match requested filename
    manifest_path = generator._module_dir() / config["dataset"]["output_dir"] / "multi_turn_random_similar_chat_manifest.json"
    if manifest_path.exists():
        print(f"\nDataset manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()