#!/usr/bin/env python3
"""
Synthetic multi-turn random-text conversation generator for inference benchmarking.

Generates conversations with randomly generated content (words, characters,
sentences, mixed content, lorem ipsum) to stress-test LLM inference with
unpredictable, non-cacheable prompts.

Usage:
    python generate.py                     # uses default config.yaml
    python generate.py --config my.yaml    # custom config
    python generate.py --num 1000          # override conversation count
    python generate.py --seed 42           # override random seed
    python generate.py --format all        # all|parquet|aiperf|mooncake
"""

import argparse
import hashlib
import json
import random
import string
import uuid
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# ---------------------------------------------------------------------------
# Word pools for random content generation
# ---------------------------------------------------------------------------

# ~500 common English words used as the vocabulary for random_words topic
VOCABULARY = [
    "time", "year", "people", "way", "day", "man", "woman", "child", "world", "life",
    "hand", "part", "place", "case", "week", "company", "system", "program", "question",
    "work", "government", "number", "night", "point", "home", "water", "room", "mother",
    "area", "money", "story", "fact", "month", "lot", "right", "study", "book", "eye",
    "job", "word", "business", "issue", "side", "kind", "head", "house", "service",
    "friend", "father", "power", "hour", "game", "line", "end", "member", "law", "car",
    "city", "community", "name", "president", "team", "minute", "idea", "body", "back",
    "parent", "face", "other", "level", "office", "door", "health", "person", "art",
    "war", "history", "party", "result", "change", "morning", "reason", "research",
    "girl", "guy", "moment", "air", "teacher", "force", "education", "foot", "boy",
    "age", "policy", "process", "music", "market", "sense", "product", "effect",
    "class", "control", "rate", "plan", "figure", "early", "development", "report",
    "student", "view", "activity", "table", "form", "plant", "river", "ground", "tree",
    "course", "land", "cost", "field", "energy", "model", "paper", "group", "center",
    "building", "interest", "period", "practice", "value", "data", "space", "stock",
    "road", "weather", "nature", "fish", "garden", "window", "range", "fire", "rock",
    "language", "action", "thought", "picture", "design", "voice", "color", "machine",
    "light", "problem", "attention", "industry", "current", "surface", "summer", "wall",
    "island", "animal", "ocean", "material", "north", "solution", "standard", "growth",
    "income", "position", "length", "region", "travel", "glass", "decision", "blood",
    "factor", "manager", "opportunity", "society", "economy", "technology", "pressure",
    "spring", "trouble", "memory", "camera", "future", "site", "choice", "function",
    "purpose", "method", "theory", "village", "defense", "evidence", "mission", "sport",
    "kitchen", "oil", "collection", "network", "performance", "band", "audience",
    "finger", "culture", "version", "debate", "environment", "corner", "chapter",
    "weather", "security", "bridge", "tradition", "election", "challenge", "argument",
    "metal", "sugar", "border", "expression", "platform", "revolution", "district",
    "device", "address", "restaurant", "battle", "signal", "progress", "reaction",
    "brain", "desire", "expert", "muscle", "novel", "horror", "storm", "climate",
    "breath", "universe", "spirit", "pattern", "library", "forest", "comfort", "peace",
    "balance", "complex", "panel", "display", "warning", "average", "measure", "channel",
    "package", "ancient", "neighbor", "session", "magazine", "emotion", "volume",
    "sample", "feature", "variety", "article", "journal", "crisis", "captain", "basket",
    "winner", "pocket", "quarter", "cabinet", "orange", "mirror", "shadow", "ceiling",
    "leather", "citizen", "silver", "column", "counter", "valley", "temple", "profit",
    "breath", "kernel", "gravity", "diamond", "monster", "fortune", "chapter", "blanket",
    "whisper", "traffic", "horizon", "volcano", "mineral", "crystal", "harvest", "journey",
    "cabinet", "premium", "railway", "miracle", "fiction", "vitamin", "dynamic", "segment",
    "quantum", "neutral", "plastic", "organic", "missile", "compass", "courage", "fantasy",
    "pension", "habitat", "archive", "gallery", "sunrise", "veteran", "surplus", "mystery",
    "algebra", "calcium", "venture", "insight", "trigger", "barrier", "episode", "paradox",
    "protein", "granite", "thermal", "ecology", "antenna", "formula", "shuttle", "catalog",
    "kingdom", "lecture", "scatter", "pioneer", "mandate", "uniform", "fiction", "circuit",
    "endless", "crimson", "triumph", "passage", "ancient", "embrace", "illusion", "radiant",
    "whisper", "silence", "harmony", "essence", "vibrant", "cascade", "ethereal", "pinnacle",
    "serenity", "zenith", "luminous", "celestial", "twilight", "ember", "frost", "bloom",
    "drift", "spark", "echo", "pulse", "ripple", "glow", "haze", "surge", "tide", "crest",
    "flare", "mist", "veil", "dusk", "dawn", "shard", "prism", "nexus", "void", "flux",
]

# Subjects, verbs, and objects for random_sentences construction
SUBJECTS = [
    "The curious robot", "A forgotten library", "Seven tiny elephants", "My neighbor's telescope",
    "The underwater piano", "A cloud of butterflies", "The midnight baker", "An invisible train",
    "The philosophical cat", "A recursive dream", "The quantum gardener", "A singing mountain",
    "The last dictionary", "An upside-down city", "The marble orchestra", "A reluctant volcano",
    "The paper astronaut", "A clockwork forest", "The diagonal rain", "An electric jellyfish",
    "The suspicious mailbox", "A translucent whale", "The backwards clock", "A portable sunset",
    "The hexagonal moon", "A velvet thunderstorm", "The wandering equation", "A parallel squirrel",
    "The ceramic rocket", "A frozen symphony", "The orbiting teapot", "A magnetic poem",
]

VERBS = [
    "discovered", "computed", "dissolved", "embraced", "translated", "compressed",
    "illuminated", "scattered", "navigated", "encrypted", "harvested", "reorganized",
    "predicted", "questioned", "assembled", "transmitted", "calibrated", "evaporated",
    "synchronized", "manifested", "generated", "processed", "evaluated", "transformed",
    "calculated", "observed", "interpreted", "constructed", "deconstructed", "amplified",
    "catalogued", "simulated", "extrapolated", "reverberated", "contemplated", "orchestrated",
]

OBJECTS = [
    "the invisible equation", "a symphony of colors", "fourteen parallel dimensions",
    "the meaning of Tuesday", "a backwards waterfall", "the texture of silence",
    "an origami universe", "the speed of thought", "a mathematical sunrise",
    "the flavor of gravity", "an acoustic shadow", "the geometry of dreams",
    "a holographic sandwich", "the weight of an idea", "a recursive paradox",
    "the echo of tomorrow", "an alphabetical storm", "the rhythm of starlight",
    "a crystallized memory", "the architecture of wind", "a digital wilderness",
    "the temperature of music", "an elastic horizon", "the density of whispers",
    "a fractured timeline", "the frequency of rain", "a volatile equilibrium",
    "the coordinates of nowhere", "an inverted rainbow", "the probability of magic",
]

ADVERBS = [
    "silently", "frantically", "methodically", "accidentally", "gracefully",
    "reluctantly", "enthusiastically", "mysteriously", "precisely", "chaotically",
    "deliberately", "spontaneously", "cautiously", "vigorously", "elegantly",
    "absurdly", "meticulously", "recklessly", "harmoniously", "paradoxically",
]

PREPOSITIONAL_PHRASES = [
    "beneath the frozen lake", "inside a crystal sphere", "beyond the last digit of pi",
    "across the velvet desert", "through the mirror dimension", "above the sleeping city",
    "within the quantum garden", "under the marble sky", "beside the infinite staircase",
    "along the electric river", "between two parallel truths", "around the forgotten axis",
    "during the silent revolution", "behind the acoustic curtain", "among the digital trees",
]

# Lorem ipsum word pool (pseudo-Latin)
LOREM_WORDS = [
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
    "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore",
    "magna", "aliqua", "enim", "ad", "minim", "veniam", "quis", "nostrud",
    "exercitation", "ullamco", "laboris", "nisi", "aliquip", "ex", "ea", "commodo",
    "consequat", "duis", "aute", "irure", "in", "reprehenderit", "voluptate",
    "velit", "esse", "cillum", "fugiat", "nulla", "pariatur", "excepteur", "sint",
    "occaecat", "cupidatat", "non", "proident", "sunt", "culpa", "qui", "officia",
    "deserunt", "mollit", "anim", "id", "est", "laborum", "viverra", "tellus",
    "pellentesque", "dignissim", "enim", "praesent", "elementum", "facilisis",
    "leo", "vel", "fringilla", "porta", "lacus", "luctus", "accumsan", "tortor",
    "posuere", "morbi", "tristique", "senectus", "netus", "malesuada", "fames",
    "turpis", "egestas", "maecenas", "pharetra", "convallis", "cras", "semper",
    "auctor", "neque", "vitae", "justo", "eget", "risus", "pretium", "quam",
    "vulputate", "sagittis", "massa", "orci", "varius", "natoque", "penatibus",
    "magnis", "dis", "parturient", "montes", "nascetur", "ridiculus", "mus",
    "mauris", "augue", "nunc", "blandit", "gravida", "dictum", "fusce", "placerat",
    "ornare", "arcu", "dui", "curabitur", "sollicitudin", "tempus", "bibendum",
    "imperdiet", "nullam", "fermentum", "iaculis", "nisl", "condimentum", "lacinia",
    "donec", "ultrices", "tincidunt", "suspendisse", "interdum", "metus", "euismod",
    "scelerisque", "purus", "felis", "urna", "mattis", "pulvinar", "sodales",
    "hendrerit", "nec", "habitant", "aliquet", "lectus", "proin", "nibh",
]

# Filler phrases for assistant random responses
RESPONSE_FILLERS = [
    "Analyzing the input, I observe several interesting patterns emerging from the data.",
    "Based on the provided sequence, there appear to be multiple interpretations possible.",
    "The combination of elements suggests a complex underlying structure worth examining.",
    "Processing this information reveals connections that may not be immediately apparent.",
    "Considering the various components, I can identify distinct thematic clusters.",
    "The patterns within this content demonstrate a fascinating range of possibilities.",
    "Examining each element individually and in context provides useful insights.",
    "The structure of this input allows for several complementary analytical approaches.",
    "Cross-referencing these elements with known patterns yields interesting correlations.",
    "The distribution of components across the input suggests a deliberate arrangement.",
    "Taking a holistic view, the relationship between elements becomes clearer.",
    "Each segment contributes to the overall complexity in a meaningful way.",
    "The variance within the content creates opportunities for deeper exploration.",
    "Mapping these elements to established frameworks reveals structural parallels.",
    "The juxtaposition of different components creates an intriguing analytical challenge.",
    "Statistical analysis of the content distribution reveals notable characteristics.",
    "The sequential arrangement of elements follows an interesting progression pattern.",
    "Decomposing the input into constituent parts allows for systematic evaluation.",
    "The semantic density of this content warrants careful multi-level analysis.",
    "Preliminary assessment suggests multiple valid interpretation frameworks apply here.",
    "The interaction between these elements creates emergent properties worth noting.",
    "Contextual analysis indicates several layers of meaning embedded in the content.",
    "Evaluating the structural integrity of this arrangement reveals key pivot points.",
    "The compositional diversity here spans a wide spectrum of representational modes.",
    "Recursive examination of the patterns uncovers self-similar structures at different scales.",
    "The entropy of this sequence suggests a high degree of informational content.",
    "Dimensional reduction of the feature space reveals principal components of interest.",
    "Temporal analysis of the progression shows distinct phase transitions in complexity.",
    "The signal-to-noise ratio in this content can be improved with appropriate filtering.",
    "Meta-analysis across all segments reveals macro-level trends invisible at local scale.",
]

# Symbols used in random_mixed content
SYMBOLS = list("!@#$%^&*()_+-=[]{}|;:',.<>?/~`")


# ---------------------------------------------------------------------------
# Conversation generator
# ---------------------------------------------------------------------------

class ConversationGenerator:
    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.rng = random.Random(seed)
        self.topics = config["topics"]
        self.topic_weights = [t["weight"] for t in self.topics]

    # ---- Random content generators for each topic ----

    def _gen_random_words(self, min_words: int = 3, max_words: int = 30) -> str:
        """Generate a sequence of random words from the vocabulary pool."""
        n = self.rng.randint(min_words, max_words)
        return " ".join(self.rng.choice(VOCABULARY) for _ in range(n))

    def _gen_random_chars(self, min_len: int = 10, max_len: int = 200) -> str:
        """Generate a random alphanumeric string of varying length."""
        length = self.rng.randint(min_len, max_len)
        charset = string.ascii_letters + string.digits
        # Break into space-separated chunks for readability
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

        # Optionally add adverb and/or prepositional phrase
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
        n = self.rng.randint(min_sentences, max_sentences)
        return " ".join(self._gen_random_sentence() for _ in range(n))

    def _gen_random_mixed(self, min_parts: int = 5, max_parts: int = 25) -> str:
        """Generate mixed content: words, numbers, and symbols."""
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
        n = self.rng.randint(min_words, max_words)
        words = [self.rng.choice(LOREM_WORDS) for _ in range(n)]
        # Capitalize first word and add sentence structure
        sentences = []
        i = 0
        while i < len(words):
            sent_len = self.rng.randint(4, 12)
            sent_words = words[i:i + sent_len]
            if sent_words:
                sent_words[0] = sent_words[0].capitalize()
                # Optionally insert a comma
                if len(sent_words) > 5 and self.rng.random() < 0.4:
                    comma_pos = self.rng.randint(2, len(sent_words) - 2)
                    sent_words[comma_pos] = sent_words[comma_pos] + ","
                sentences.append(" ".join(sent_words) + ".")
            i += sent_len
        return " ".join(sentences)

    # ---- Content dispatch ----

    def _generate_user_content(self, topic_name: str, turn_index: int) -> str:
        """Generate user message content based on topic type."""
        if topic_name == "random_words":
            # Longer messages as conversation progresses
            min_w = 3 if turn_index < 3 else 5
            max_w = 15 if turn_index < 5 else 30
            return self._gen_random_words(min_w, max_w)

        elif topic_name == "random_chars":
            min_len = 10 if turn_index < 3 else 20
            max_len = 80 if turn_index < 5 else 200
            return self._gen_random_chars(min_len, max_len)

        elif topic_name == "random_sentences":
            min_s = 1 if turn_index < 3 else 2
            max_s = 3 if turn_index < 5 else 6
            return self._gen_random_sentences(min_s, max_s)

        elif topic_name == "random_mixed":
            min_p = 5 if turn_index < 3 else 8
            max_p = 15 if turn_index < 5 else 25
            return self._gen_random_mixed(min_p, max_p)

        elif topic_name == "random_lorem":
            min_w = 10 if turn_index < 3 else 15
            max_w = 30 if turn_index < 5 else 60
            return self._gen_random_lorem(min_w, max_w)

        else:
            # Fallback: random words
            return self._gen_random_words()

    def _generate_response_content(self, topic_name: str, target_words: int) -> str:
        """Generate an assistant response of approximately target_words length.

        Combines filler sentences with topic-appropriate random content to produce
        varied, unpredictable responses.
        """
        parts = []
        word_count = 0

        while word_count < target_words:
            # Alternate between filler analysis sentences and random content
            strategy = self.rng.choices(
                ["filler", "echo_random", "list", "numbered"],
                weights=[0.35, 0.35, 0.15, 0.15],
            )[0]

            if strategy == "filler":
                sentence = self.rng.choice(RESPONSE_FILLERS)
                parts.append(sentence)
                word_count += len(sentence.split())

            elif strategy == "echo_random":
                # Generate random content similar to the topic style
                if topic_name == "random_words":
                    chunk = self._gen_random_words(5, 20)
                elif topic_name == "random_chars":
                    chunk = self._gen_random_chars(15, 80)
                elif topic_name == "random_sentences":
                    chunk = self._gen_random_sentences(1, 3)
                elif topic_name == "random_mixed":
                    chunk = self._gen_random_mixed(5, 15)
                elif topic_name == "random_lorem":
                    chunk = self._gen_random_lorem(8, 30)
                else:
                    chunk = self._gen_random_words(5, 15)
                parts.append(chunk)
                word_count += len(chunk.split())

            elif strategy == "list":
                # Generate a bullet-point list of random items
                n_items = self.rng.randint(3, 6)
                items = []
                for _ in range(n_items):
                    item = self._gen_random_words(3, 8)
                    items.append(f"- {item}")
                    word_count += len(item.split())
                parts.append("\n".join(items))

            else:  # numbered
                # Generate a numbered analysis
                n_items = self.rng.randint(2, 5)
                items = []
                for idx in range(1, n_items + 1):
                    item = self.rng.choice(RESPONSE_FILLERS)
                    items.append(f"{idx}. {item}")
                    word_count += len(item.split())
                parts.append("\n".join(items))

        text = "\n\n".join(parts)

        # Trim if significantly over target
        words = text.split()
        if len(words) > target_words * 1.3:
            words = words[:target_words]
            text = " ".join(words)
            last_period = text.rfind(".")
            if last_period > len(text) * 0.7:
                text = text[:last_period + 1]

        return text

    # ---- Topic and length selection ----

    def _pick_topic(self) -> dict:
        """Weighted random topic selection."""
        return self.rng.choices(self.topics, weights=self.topic_weights, k=1)[0]

    def _response_length_bucket(self, turn_index: int) -> str:
        """Determine response length based on turn position."""
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

    # ---- Conversation assembly ----

    def generate_conversation(self, num_turns: int) -> dict:
        """Generate a single multi-turn conversation with random content."""
        topic = self._pick_topic()
        topic_name = topic["name"]
        system_prompt = topic["system_prompt"]

        messages = [{"role": "system", "content": system_prompt}]
        cumulative_char_lengths = []
        running_chars = len(system_prompt)

        for turn_idx in range(num_turns):
            # User message
            user_msg = self._generate_user_content(topic_name, turn_idx)
            messages.append({"role": "user", "content": user_msg})
            running_chars += len(user_msg)

            # Assistant response
            bucket = self._response_length_bucket(turn_idx)
            length_config = self.config["response_length"][bucket]
            target_words = self.rng.randint(length_config["min_words"], length_config["max_words"])
            assistant_msg = self._generate_response_content(topic_name, target_words)
            messages.append({"role": "assistant", "content": assistant_msg})
            running_chars += len(assistant_msg)

            cumulative_char_lengths.append(running_chars)

        conversation_id = str(uuid.UUID(int=self.rng.getrandbits(128), version=4))

        return {
            "conversation_id": conversation_id,
            "topic": topic_name,
            "num_turns": num_turns,
            "num_messages": len(messages),
            "system_prompt": system_prompt,
            "messages": json.dumps(messages),
            "total_characters": running_chars,
            "estimated_tokens": running_chars // 4,  # rough approximation
            "cumulative_char_lengths": json.dumps(cumulative_char_lengths),
        }

    def generate_dataset(self, num_conversations: int = None) -> list[dict]:
        """Generate the full dataset according to config distribution."""
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


# ---------------------------------------------------------------------------
# Output format converters
# ---------------------------------------------------------------------------

def convert_to_aiperf_multi_turn(conversations: list[dict]) -> list[dict]:
    """Convert conversations to aiperf multi_turn JSONL format.

    aiperf multi_turn format (one JSON object per line):
        {"session_id": "...", "turns": [{"text": "user msg"}, {"text": "user msg 2"}, ...]}

    This uses the `deltas_without_responses` context mode — aiperf sends only
    user messages and accumulates live server responses into conversation history
    automatically. This is ideal for benchmarking prefix caching because each
    turn's request includes the full conversation prefix.

    Usage with aiperf:
        aiperf profile \\
            --model <model> \\
            --endpoint-type chat \\
            --input-file multi_turn_random_chat.jsonl \\
            --custom-dataset-type multi_turn \\
            --streaming --url localhost:8000

    Returns:
        List of dicts, each representing one conversation line for JSONL output.
    """
    aiperf_entries = []
    for conv in conversations:
        messages = json.loads(conv["messages"])
        turns = []
        for msg in messages:
            if msg["role"] == "user":
                turns.append({"text": msg["content"]})
        if turns:
            aiperf_entries.append({
                "session_id": conv["conversation_id"],
                "turns": turns,
            })
    return aiperf_entries


def convert_to_aiperf_mooncake(conversations: list[dict], block_size: int = 512) -> list[dict]:
    """Convert conversations to aiperf mooncake_trace JSONL format.

    This format sends the full message array (system + user + assistant) per turn,
    using the `message_array_with_responses` context mode. Each line represents a
    single turn within a session, with the complete conversation history up to that
    point. This gives full control over the exact prompt sent to the server.

    Format (one JSON object per line):
        {"session_id": "...", "messages": [...], "output_length": N}

    Usage with aiperf:
        aiperf profile \\
            --model <model> \\
            --endpoint-type chat \\
            --input-file multi_turn_random_chat_mooncake.jsonl \\
            --custom-dataset-type mooncake_trace \\
            --streaming --url localhost:8000

    Returns:
        List of dicts, one per turn across all conversations.
    """
    entries = []
    for conv in conversations:
        messages = json.loads(conv["messages"])
        session_id = conv["conversation_id"]
        # Walk through the conversation, building up context each turn
        context = []
        for i, msg in enumerate(messages):
            context.append(msg)
            # Emit an entry after each assistant message (= end of a turn pair)
            if msg["role"] == "assistant":
                # Estimate output length for this assistant response
                output_tokens = max(1, len(msg["content"]) // 4)
                entry = {
                    "session_id": session_id,
                    "messages": [m for m in context],  # full context up to here
                    "output_length": output_tokens,
                }
                # Add delay for turns after the first (simulate user think time)
                turn_index = len([m for m in context if m["role"] == "assistant"])
                if turn_index > 1:
                    entry["delay"] = 0  # no artificial delay; set >0 to simulate think time
                entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-turn random chat dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--num", type=int, default=None, help="Override number of conversations")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output", default=None, help="Override output path")
    parser.add_argument("--format", choices=["all", "parquet", "aiperf", "mooncake"],
                        default="all", help="Output format(s): parquet, aiperf (multi_turn JSONL), "
                        "mooncake (mooncake_trace JSONL), or all (default)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config["dataset"]["seed"]
    generator = ConversationGenerator(config, seed=seed)

    print(f"Generating conversations (seed={seed})...")
    conversations = generator.generate_dataset(num_conversations=args.num)
    print(f"Generated {len(conversations)} conversations")

    # Build DataFrame
    df = pd.DataFrame(conversations)

    # Summary statistics
    print(f"\n--- Dataset Summary ---")
    print(f"Total conversations: {len(df)}")
    print(f"Turn count range: {df['num_turns'].min()} - {df['num_turns'].max()}")
    print(f"Mean turns: {df['num_turns'].mean():.1f}")
    print(f"Topic distribution:")
    for topic, count in df["topic"].value_counts().items():
        print(f"  {topic}: {count} ({100*count/len(df):.1f}%)")
    print(f"Estimated total tokens: {df['estimated_tokens'].sum():,}")
    print(f"Mean tokens/conversation: {df['estimated_tokens'].mean():,.0f}")
    print(f"Max tokens (single conversation): {df['estimated_tokens'].max():,}")

    # Determine output directory
    output_dir = Path(args.output).parent if args.output else Path(__file__).parent / config["dataset"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format

    # Write Parquet
    if fmt in ("all", "parquet"):
        parquet_path = Path(args.output) if args.output and fmt == "parquet" else output_dir / config["dataset"]["output_filename"]
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        print(f"\nParquet written to: {parquet_path} ({file_size_mb:.2f} MB)")

    # Write aiperf multi_turn JSONL
    if fmt in ("all", "aiperf"):
        aiperf_entries = convert_to_aiperf_multi_turn(conversations)
        jsonl_path = output_dir / config["dataset"]["output_filename"].replace(".parquet", ".jsonl")
        with open(jsonl_path, "w") as f:
            for entry in aiperf_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        file_size_mb = jsonl_path.stat().st_size / (1024 * 1024)
        print(f"aiperf multi_turn JSONL written to: {jsonl_path} ({file_size_mb:.2f} MB)")
        print(f"  Usage: aiperf profile --input-file {jsonl_path} --custom-dataset-type multi_turn ...")

    # Write aiperf mooncake_trace JSONL
    if fmt in ("all", "mooncake"):
        mooncake_entries = convert_to_aiperf_mooncake(conversations)
        mooncake_path = output_dir / config["dataset"]["output_filename"].replace(".parquet", "_mooncake.jsonl")
        with open(mooncake_path, "w") as f:
            for entry in mooncake_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        file_size_mb = mooncake_path.stat().st_size / (1024 * 1024)
        print(f"aiperf mooncake_trace JSONL written to: {mooncake_path} ({file_size_mb:.2f} MB)")
        print(f"  Usage: aiperf profile --input-file {mooncake_path} --custom-dataset-type mooncake_trace ...")


if __name__ == "__main__":
    main()
