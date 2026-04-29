#!/usr/bin/env python3
"""
Synthetic multi-turn repetitive-text conversation generator for inference benchmarking.

Generates conversations where user messages consist of the same word, phrase, or
pattern repeated many times. This creates highly compressible content useful for
testing tokenizer behavior and prefix caching with near-identical content.

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
import sys
import uuid
from pathlib import Path

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
)

# ---------------------------------------------------------------------------
# Word and phrase pools for repetitive content
# ---------------------------------------------------------------------------

SINGLE_WORDS = [
    "hello", "world", "test", "data", "foo", "bar", "baz", "ping", "pong",
    "alpha", "beta", "gamma", "delta", "echo", "loop", "node", "edge",
    "token", "chunk", "block", "cache", "queue", "stack", "tree", "hash",
    "query", "index", "model", "train", "batch", "epoch", "layer", "dense",
    "input", "output", "state", "event", "click", "load", "save", "send",
    "read", "write", "parse", "build", "check", "start", "stop", "reset",
    "apple", "banana", "cherry", "grape", "lemon", "mango", "peach", "plum",
    "blue", "green", "red", "white", "black", "gold", "silver", "bronze",
    "north", "south", "east", "west", "river", "mountain", "ocean", "forest",
    "sun", "moon", "star", "cloud", "rain", "snow", "wind", "fire",
    "table", "chair", "door", "window", "floor", "wall", "roof", "path",
    "book", "page", "word", "line", "note", "song", "bell", "drum",
    "fish", "bird", "wolf", "bear", "deer", "frog", "hawk", "lion",
]

SHORT_PHRASES = [
    "the cat sat", "open the door", "run the test", "hello world again",
    "foo bar baz", "one two three", "red blue green", "up and down",
    "back and forth", "round and round", "over the hill", "under the bridge",
    "left then right", "stop and go", "rise and fall", "push and pull",
    "day by day", "step by step", "side to side", "end to end",
    "fire and ice", "salt and pepper", "bread and butter", "rain or shine",
    "now or never", "more or less", "give and take", "hit or miss",
    "black and white", "loud and clear", "short and sweet", "safe and sound",
    "the quick fox", "a big red ball", "my old friend", "the last train",
    "all the time", "on the way", "in the end", "at the top",
    "by the sea", "for the win", "to the moon", "from the start",
    "once upon a time", "far far away", "the very best", "a brand new day",
    "just like that", "one more time", "all over again", "here we go",
]

COUNTING_SEQUENCES = [
    "1 2 3 4 5", "1 2 3", "0 1 0 1", "10 20 30 40 50",
    "1 1 2 3 5 8", "2 4 6 8 10", "1 3 5 7 9", "100 200 300",
    "A B C D E", "1 A 2 B 3 C", "9 8 7 6 5", "0 0 1 0 0 1",
    "3 1 4 1 5 9", "1 4 9 16 25", "11 22 33 44 55", "5 10 15 20",
]

LETTER_PATTERNS = [
    "aaa", "bbb", "ccc", "abc", "xyz", "aabb", "abab", "abcabc",
    "aaabbbccc", "xyzxyz", "ababab", "aabbaabb", "abcdabcd",
    "mmm", "nnn", "ooo", "mnomno", "pqrpqr", "ststst", "uvwuvw",
    "aaaa", "bbbb", "cccc", "dddd", "aaabbb", "cccddd", "eeefffggg",
    "zzzyyy", "xxww", "qqqrrrsss", "ttttuuuu", "vvvvwwww",
]

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "All work and no play makes Jack a dull boy.",
    "To be or not to be, that is the question.",
    "I think therefore I am.",
    "The only thing we have to fear is fear itself.",
    "In the beginning there was nothing.",
    "It was a dark and stormy night.",
    "Once upon a time in a land far away.",
    "The rain in Spain stays mainly in the plain.",
    "A journey of a thousand miles begins with a single step.",
    "Actions speak louder than words.",
    "Every cloud has a silver lining.",
    "Knowledge is power and power is knowledge.",
    "Time flies when you are having fun.",
    "Practice makes perfect they always say.",
    "The early bird catches the worm.",
    "Better late than never is what they say.",
    "All that glitters is not gold.",
    "Where there is a will there is a way.",
    "You miss every shot you do not take.",
    "The pen is mightier than the sword.",
    "When in Rome do as the Romans do.",
    "Look before you leap into the unknown.",
    "Two wrongs do not make a right.",
    "Fortune favors the bold and the brave.",
    "Necessity is the mother of invention.",
    "The best things in life are free.",
    "Honesty is the best policy to follow.",
    "A picture is worth a thousand words.",
    "Birds of a feather flock together.",
    "What goes around comes around eventually.",
    "The grass is always greener on the other side.",
    "Do not count your chickens before they hatch.",
    "Rome was not built in a single day.",
    "There is no place like home sweet home.",
    "Curiosity killed the cat but satisfaction brought it back.",
    "An apple a day keeps the doctor away.",
    "You cannot judge a book by its cover.",
    "The truth will set you free in the end.",
    "Every rose has its thorn they say.",
    "Silence is golden and patience is a virtue.",
    "Beauty is in the eye of the beholder.",
    "If at first you do not succeed try again.",
    "A stitch in time saves nine stitches later.",
    "Keep your friends close and your enemies closer.",
    "The squeaky wheel gets the grease.",
    "Laughter is the best medicine for the soul.",
    "When life gives you lemons make lemonade.",
    "Do unto others as you would have them do unto you.",
    "The only constant in life is change itself.",
]

# ---------------------------------------------------------------------------
# Response templates for assistant messages
# ---------------------------------------------------------------------------

RESPONSE_TEMPLATES = {
    "single_word": [
        "I notice you've repeated the word '{word}' {count} times. That's quite a lot of repetition! "
        "Repetitive inputs like this are interesting from a linguistic perspective — "
        "they test how systems handle highly redundant content. "
        "Is there something specific you'd like me to help you with regarding '{word}'?",

        "You've sent '{word}' repeated {count} times. I can see the pattern clearly. "
        "If you're testing something or just exploring, I'm happy to play along. "
        "Otherwise, let me know what you need and I'll do my best to assist.",

        "That's '{word}' appearing {count} times in your message. "
        "While the content is repetitive, I'm here to help with whatever you need. "
        "Would you like to discuss something related to this word, or is there another topic on your mind?",

        "I see you've written '{word}' many times — {count} times to be exact. "
        "Repetition can be a form of emphasis or a test of patience. "
        "Either way, I acknowledge your message. How can I assist you today?",

        "Your message contains the word '{word}' repeated {count} times. "
        "That's a straightforward and highly patterned input. "
        "I'm ready to help with whatever comes next — just let me know what you need.",
    ],
    "phrase_repeat": [
        "I see you've repeated the phrase '{phrase}' {count} times. "
        "Phrase repetition creates an interesting rhythm in text. "
        "Is there something specific about this phrase you'd like to explore, "
        "or would you like to move on to a different topic?",

        "You've sent '{phrase}' repeated {count} times. That's a notable amount of repetition. "
        "I can see the pattern and I'm ready to help you with whatever you need. "
        "Just let me know how I can assist.",

        "The phrase '{phrase}' appears {count} times in your message. "
        "Repetitive phrases like this can be used for emphasis, testing, or creative purposes. "
        "What would you like to do next?",

        "I notice the repeated phrase '{phrase}' — it appears {count} times. "
        "Whether you're testing input handling or have a specific purpose, "
        "I'm here and ready to help. What's on your mind?",
    ],
    "counting_repeat": [
        "I see you've repeated the counting sequence '{sequence}' {count} times. "
        "Counting patterns are fundamental in mathematics and computing. "
        "Would you like to explore number sequences, or is this a test of some kind?",

        "You've sent the sequence '{sequence}' repeated {count} times. "
        "Number patterns and repetition are foundational concepts. "
        "I can help with math, pattern recognition, or whatever else you'd like to discuss.",

        "The counting pattern '{sequence}' appears {count} times in your message. "
        "Repetitive numerical sequences have interesting properties in both mathematics and information theory. "
        "How can I help you today?",

        "I notice the repeated number sequence '{sequence}' — {count} repetitions in total. "
        "Whether this is a test input or you're curious about number patterns, "
        "I'm ready to assist. What would you like to know?",
    ],
    "letter_repeat": [
        "I see a repeating pattern of characters — '{pattern}' repeated to form a {length}-character string. "
        "Character-level repetition is interesting from a compression and encoding perspective. "
        "Is there something specific you'd like to explore about this pattern?",

        "Your message consists of the character pattern '{pattern}' repeated many times, "
        "creating a string of {length} characters total. "
        "I can see the repeating structure clearly. How can I assist you?",

        "I notice a character-level repetition pattern: '{pattern}' appears throughout your "
        "{length}-character message. This kind of highly structured input is interesting to analyze. "
        "What would you like to discuss?",

        "That's the pattern '{pattern}' repeated to fill {length} characters. "
        "Repetitive character sequences like this have applications in testing and benchmarking. "
        "Let me know what you need help with.",
    ],
    "sentence_repeat": [
        "I see you've repeated the sentence '{sentence}' {count} times. "
        "Sentence-level repetition creates a strong rhetorical effect. "
        "Is there something about this sentence you'd like to discuss, or is there another way I can help?",

        "You've written the same sentence {count} times: '{sentence}' "
        "That's a lot of repetition. Whether intentional or exploratory, "
        "I'm here to help with whatever you need.",

        "The sentence '{sentence}' appears {count} times in your message. "
        "Repeating a full sentence like this can serve various purposes — from emphasis to testing. "
        "How can I assist you today?",

        "I notice you've sent '{sentence}' repeated {count} times. "
        "I acknowledge the repetitive nature of your message. "
        "I'm ready to help — just let me know what you'd like to do next.",
    ],
}

# Filler sentences to pad responses to the target length
RESPONSE_FILLER = [
    "Language processing systems handle repetitive content in various ways, and understanding these patterns can be quite insightful.",
    "From an information theory perspective, highly repetitive content has very low entropy, which means it can be compressed significantly.",
    "Tokenizers in modern language models often handle repeated patterns efficiently by recognizing recurring subword units.",
    "The way different systems process and store repetitive text can reveal a lot about their underlying architecture and optimization strategies.",
    "In natural language processing, repetition is both a feature and a challenge — it can indicate emphasis but also noise.",
    "Compression algorithms like LZ77 and Huffman coding excel at handling exactly this kind of repetitive content.",
    "When evaluating inference engines, repetitive inputs help isolate the performance characteristics of prefix caching mechanisms.",
    "The relationship between token count and character count varies depending on the content — repetitive text often tokenizes differently than diverse text.",
    "Benchmarking with controlled repetitive patterns allows for more precise measurement of system behavior under specific conditions.",
    "I find that working with structured and patterned inputs can be a useful way to explore the boundaries of text processing systems.",
    "If you're interested in how this content is processed internally, I'd be happy to explain some of the relevant concepts.",
    "Modern language models use attention mechanisms that can both benefit from and be challenged by highly repetitive input patterns.",
    "The study of repetition in language goes back centuries, from rhetorical devices in classical writing to modern computational linguistics.",
    "Understanding how systems handle redundancy is crucial for building efficient and robust text processing pipelines.",
    "Repetitive patterns in text can serve as excellent test cases for evaluating caching strategies and memory management.",
    "The distinction between meaningful repetition and noise is one of the interesting challenges in natural language understanding.",
    "From a practical standpoint, ensuring systems handle repetitive content gracefully is important for production reliability.",
    "Text repetition can be analyzed at multiple levels — character, token, word, phrase, and sentence — each revealing different properties.",
    "I appreciate you sharing this input. It gives me an opportunity to demonstrate consistent and helpful responses regardless of content type.",
    "If you'd like, I can provide more detailed analysis of the patterns in your input or help with any other questions you might have.",
    "Working with patterned inputs is valuable for testing and quality assurance across many different types of software systems.",
    "The amount of information in a message depends not just on its length but on its complexity and variability.",
    "Efficient handling of repetitive content is a key consideration in system design, especially for high-throughput applications.",
    "I'm designed to be helpful and informative regardless of the nature of the input, whether it's complex prose or simple repetition.",
    "Processing repetitive text efficiently requires careful consideration of memory usage, caching, and computational resources.",
]


# ---------------------------------------------------------------------------
# Conversation generator
# ---------------------------------------------------------------------------

class ConversationGenerator:
    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.rng = random.Random(seed)
        self.topics = config["topics"]
        self.topic_weights = [t["weight"] for t in self.topics]

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

    # ----- User message generators per topic -----

    def _generate_single_word(self, word: str, turn_index: int) -> str:
        """Generate a user message: a single word repeated N times (5-100)."""
        count = self.rng.randint(5, 100)
        return " ".join([word] * count)

    def _generate_phrase_repeat(self, phrase: str, turn_index: int) -> str:
        """Generate a user message: a short phrase repeated N times (3-50)."""
        count = self.rng.randint(3, 50)
        return " ".join([phrase] * count)

    def _generate_counting_repeat(self, sequence: str, turn_index: int) -> str:
        """Generate a user message: a counting sequence repeated N times (5-30)."""
        count = self.rng.randint(5, 30)
        return " ".join([sequence] * count)

    def _generate_letter_repeat(self, pattern: str, turn_index: int) -> str:
        """Generate a user message: letter pattern repeated to reach 50-500 chars."""
        target_len = self.rng.randint(50, 500)
        repetitions = (target_len // len(pattern)) + 1
        return (pattern * repetitions)[:target_len]

    def _generate_sentence_repeat(self, sentence: str, turn_index: int) -> str:
        """Generate a user message: a full sentence repeated N times (2-20)."""
        count = self.rng.randint(2, 20)
        return " ".join([sentence] * count)

    def _generate_user_message(self, topic_name: str, base_content: str, turn_index: int) -> str:
        """Generate a repetitive user message based on topic and base content."""
        generators = {
            "single_word": self._generate_single_word,
            "phrase_repeat": self._generate_phrase_repeat,
            "counting_repeat": self._generate_counting_repeat,
            "letter_repeat": self._generate_letter_repeat,
            "sentence_repeat": self._generate_sentence_repeat,
        }
        return generators[topic_name](base_content, turn_index)

    # ----- Base content selection per topic -----

    def _pick_base_content(self, topic_name: str) -> str:
        """Pick the base word/phrase/pattern for a conversation (same across all turns)."""
        if topic_name == "single_word":
            return self.rng.choice(SINGLE_WORDS)
        elif topic_name == "phrase_repeat":
            return self.rng.choice(SHORT_PHRASES)
        elif topic_name == "counting_repeat":
            return self.rng.choice(COUNTING_SEQUENCES)
        elif topic_name == "letter_repeat":
            return self.rng.choice(LETTER_PATTERNS)
        elif topic_name == "sentence_repeat":
            return self.rng.choice(SENTENCES)
        else:
            return self.rng.choice(SINGLE_WORDS)

    # ----- Assistant response generation -----

    def _generate_response_text(self, topic_name: str, base_content: str,
                                user_msg: str, turn_index: int) -> str:
        """Generate an assistant response of appropriate length."""
        templates = RESPONSE_TEMPLATES[topic_name]
        template = self.rng.choice(templates)

        # Fill template placeholders based on topic
        if topic_name == "single_word":
            word_count = user_msg.count(base_content)
            base = template.format(word=base_content, count=word_count)
        elif topic_name == "phrase_repeat":
            phrase_count = user_msg.count(base_content)
            base = template.format(phrase=base_content, count=phrase_count)
        elif topic_name == "counting_repeat":
            seq_count = user_msg.count(base_content)
            base = template.format(sequence=base_content, count=seq_count)
        elif topic_name == "letter_repeat":
            base = template.format(pattern=base_content, length=len(user_msg))
        elif topic_name == "sentence_repeat":
            sent_count = user_msg.count(base_content)
            base = template.format(sentence=base_content, count=sent_count)
        else:
            base = template

        # Determine target length
        bucket = self._response_length_bucket(turn_index)
        length_config = self.config["response_length"][bucket]
        target_words = self.rng.randint(length_config["min_words"], length_config["max_words"])

        # Pad or trim to approximate target
        words = base.split()
        if len(words) < target_words:
            # Extend with filler text
            filler_pool = list(RESPONSE_FILLER)
            self.rng.shuffle(filler_pool)
            filler_idx = 0
            while len(words) < target_words and filler_idx < len(filler_pool):
                words.extend(filler_pool[filler_idx].split())
                filler_idx += 1
            # If still short after exhausting the pool, cycle through again
            while len(words) < target_words:
                words.extend(self.rng.choice(RESPONSE_FILLER).split())
        elif len(words) > target_words * 1.3:
            words = words[:target_words]
            # Try to end on a sentence boundary
            text = " ".join(words)
            last_period = text.rfind(".")
            if last_period > len(text) * 0.7:
                text = text[:last_period + 1]
            return text

        return " ".join(words)

    # ----- Conversation generation -----

    def generate_conversation(self, num_turns: int) -> dict:
        """Generate a single multi-turn conversation with repetitive user messages."""
        topic = self._pick_topic()
        topic_name = topic["name"]
        system_prompt = topic["system_prompt"]

        # Pick the base content used throughout this entire conversation
        base_content = self._pick_base_content(topic_name)

        messages = [{"role": "system", "content": system_prompt}]
        cumulative_char_lengths = []
        running_chars = len(system_prompt)

        for turn_idx in range(num_turns):
            # User message (repetitive content)
            user_msg = self._generate_user_message(topic_name, base_content, turn_idx)
            messages.append({"role": "user", "content": user_msg})
            running_chars += len(user_msg)

            # Assistant response (natural, acknowledging the repetition)
            assistant_msg = self._generate_response_text(
                topic_name, base_content, user_msg, turn_idx
            )
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
            --input-file multi_turn_repeat_chat.jsonl \\
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
            --input-file multi_turn_repeat_chat_mooncake.jsonl \\
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
    parser = argparse.ArgumentParser(description="Generate synthetic multi-turn repeat chat dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--num", type=int, default=None, help="Override number of conversations")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output", default=None, help="Override output path")
    parser.add_argument("--format", choices=["all", "parquet", "aiperf", "mooncake"],
                        default="all", help="Output format(s)")
    parser.add_argument("--name", default=None,
                        help="Custom suffix for descriptive output filenames")
    parser.add_argument("--descriptive-names", action="store_true", default=False,
                        help="Use descriptive filenames encoding count, seed, version, and date")
    parser.add_argument("--no-profile", action="store_true", default=False,
                        help="Skip generating the dataset manifest/profile JSON")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config["dataset"]["seed"]
    generator = ConversationGenerator(config, seed=seed)

    num_conversations = args.num
    print(f"Generating repeat conversations (seed={seed})...")
    conversations = generator.generate_dataset(num_conversations=num_conversations)
    print(f"Generated {len(conversations)} conversations")

    df = pd.DataFrame(conversations)

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

    output_dir = Path(args.output).parent if args.output else Path(__file__).parent / config["dataset"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_count = len(df)
    descriptive_name = build_descriptive_name(
        config, actual_count, seed, "repeat", custom_suffix=args.name
    )

    if args.descriptive_names:
        file_base = descriptive_name
    else:
        file_base = config["dataset"]["output_filename"].replace(".parquet", "")

    fmt = args.format
    output_files = {}

    if fmt in ("all", "parquet"):
        parquet_path = Path(args.output) if args.output and fmt == "parquet" else output_dir / f"{file_base}.parquet"
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        output_files["parquet"] = str(parquet_path)
        print(f"\nParquet written to: {parquet_path} ({file_size_mb:.2f} MB)")

    if fmt in ("all", "aiperf"):
        aiperf_entries = convert_to_aiperf_multi_turn(conversations)
        jsonl_path = output_dir / f"{file_base}.jsonl"
        with open(jsonl_path, "w") as f:
            for entry in aiperf_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        file_size_mb = jsonl_path.stat().st_size / (1024 * 1024)
        output_files["aiperf_multi_turn"] = str(jsonl_path)
        print(f"aiperf multi_turn JSONL written to: {jsonl_path} ({file_size_mb:.2f} MB)")

    if fmt in ("all", "mooncake"):
        mooncake_entries = convert_to_aiperf_mooncake(conversations)
        mooncake_path = output_dir / f"{file_base}_mooncake.jsonl"
        with open(mooncake_path, "w") as f:
            for entry in mooncake_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        file_size_mb = mooncake_path.stat().st_size / (1024 * 1024)
        output_files["mooncake_trace"] = str(mooncake_path)
        print(f"aiperf mooncake_trace JSONL written to: {mooncake_path} ({file_size_mb:.2f} MB)")

    if not args.no_profile:
        manifest = build_manifest(
            df=df,
            config=config,
            dataset_type="repeat",
            seed=seed,
            output_files=output_files,
            descriptive_name=descriptive_name,
        )
        manifest_path = save_manifest(manifest, output_dir, file_base)
        output_files["manifest"] = str(manifest_path)
        print(f"\nDataset manifest written to: {manifest_path}")
        print_profile_summary(manifest)


if __name__ == "__main__":
    main()
