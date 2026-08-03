"""Isometric (controlled) multi-turn chat dataset generator base.

The three subclasses ``IsoTextGenerator``, ``IsoRandomGenerator``, and
``IsoReasoningGenerator`` share a *single deterministic conversation plan*:
the same number of conversations, the same turn counts per conversation, and
exactly the same per-message word budgets.  The only thing that differs is the
style-specific content rendered for each message.

This makes it possible to benchmark LLMs on coherent text, random gibberish,
and deep-reasoning prompts while holding total word count and total turn count
constant.
"""

import json
from typing import List

from generator_base import BaseConversationGenerator, count_message_tokens


# Vocabulary used for the gibberish style.
_GIBBERISH_VOCAB = [
    "time", "year", "people", "way", "day", "man", "woman", "child", "world",
    "life", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "work", "government", "number", "night", "point",
    "home", "water", "room", "mother", "area", "money", "story", "fact",
    "month", "lot", "right", "study", "book", "eye", "job", "word",
    "business", "issue", "side", "kind", "head", "house", "service",
    "friend", "father", "power", "hour", "game", "line", "end", "member",
    "law", "car", "city", "community", "name", "president", "team",
    "minute", "idea", "body", "back", "parent", "face", "other", "level",
    "office", "door", "health", "person", "art", "war", "history", "party",
    "result", "change", "morning", "reason", "research", "girl", "guy",
    "moment", "air", "teacher", "force", "education", "foot", "boy", "age",
    "policy", "process", "music", "market", "sense", "product", "effect",
    "class", "control", "rate", "plan", "figure", "development", "report",
    "student", "view", "activity", "table", "form", "plant", "river", "ground",
    "tree", "course", "land", "cost", "field", "energy", "model", "paper",
    "group", "center", "building", "interest", "period", "practice", "value",
    "data", "space", "stock", "road", "weather", "nature", "fish", "garden",
    "window", "range", "fire", "rock", "language", "action", "thought",
    "picture", "design", "voice", "color", "machine", "light", "problem",
    "attention", "industry", "current", "surface", "summer", "wall", "island",
    "animal", "ocean", "material", "north", "solution", "standard", "growth",
    "income", "position", "length", "region", "travel", "glass", "decision",
    "blood", "factor", "manager", "opportunity", "society", "economy",
    "technology", "pressure", "spring", "trouble", "memory", "camera", "future",
    "site", "choice", "function", "purpose", "method", "theory", "village",
    "defense", "evidence", "mission", "sport", "kitchen", "oil", "collection",
    "network", "performance", "band", "audience", "finger", "culture",
    "version", "debate", "environment", "corner", "chapter", "security",
    "bridge", "tradition", "election", "challenge", "argument", "metal",
    "sugar", "border", "expression", "platform", "revolution", "district",
    "device", "address", "restaurant", "battle", "signal", "progress",
    "reaction", "brain", "desire", "expert", "muscle", "novel", "horror",
    "storm", "climate", "breath", "universe", "spirit", "pattern", "library",
    "forest", "comfort", "peace", "balance", "complex", "panel", "display",
    "warning", "average", "measure", "channel", "package", "ancient",
    "neighbor", "session", "magazine", "emotion", "volume", "sample",
    "feature", "variety", "article", "journal", "crisis", "captain", "basket",
    "winner", "pocket", "quarter", "cabinet", "orange", "mirror", "shadow",
    "ceiling", "leather", "citizen", "silver", "column", "counter", "valley",
    "temple", "profit", "kernel", "gravity", "diamond", "monster", "fortune",
    "blanket", "whisper", "traffic", "horizon", "volcano", "mineral", "crystal",
    "harvest", "journey", "premium", "railway", "miracle", "fiction", "vitamin",
    "dynamic", "segment", "quantum", "neutral", "plastic", "organic",
    "missile", "compass", "courage", "fantasy", "pension", "habitat", "archive",
    "gallery", "sunrise", "veteran", "surplus", "mystery", "algebra", "calcium",
    "venture", "insight", "trigger", "barrier", "episode", "paradox",
    "protein", "granite", "thermal", "ecology", "antenna", "formula",
    "shuttle", "catalog", "kingdom", "lecture", "scatter", "pioneer",
    "mandate", "uniform", "endless", "crimson", "triumph", "passage",
    "embrace", "illusion", "radiant", "silence", "harmony", "essence",
    "vibrant", "cascade", "ethereal", "pinnacle", "serenity", "zenith",
    "luminous", "celestial", "twilight", "ember", "frost", "bloom", "drift",
    "spark", "echo", "pulse", "ripple", "glow", "haze", "surge", "tide",
    "crest", "flare", "mist", "veil", "dusk", "dawn", "shard", "prism",
    "nexus", "void", "flux",
]

# Coherent text fragments.
_COHERENT_USER_OPENERS = [
    "How do I reset my password on this platform?",
    "What are the best exercises for lower back pain?",
    "Can you recommend a quiet place to work in this city?",
    "I'm planning a weekend trip to Kyoto. What should I see?",
    "How can I improve my sleep quality?",
    "What are some healthy breakfast ideas for busy mornings?",
    "Can you help me write a professional email to my manager?",
    "How do I get started with learning Python?",
    "What should I pack for a hiking trip in the mountains?",
    "Why is my computer running slower than usual?",
    "What is the best way to learn a new language quickly?",
    "Can you suggest a book for someone who enjoys mystery novels?",
]

_COHERENT_USER_FOLLOWUPS = [
    "Could you explain that in more detail?",
    "What if my budget is lower?",
    "How long does that usually take?",
    "Are there any risks I should know about?",
    "Can you give me a specific example?",
    "What are the alternatives?",
    "Thanks, is there anything else I should consider?",
    "How does that compare to other options?",
    "Would that work for a beginner?",
    "Can you recommend the next step?",
    "What happens if that doesn't work?",
    "Can you make it simpler?",
]

_COHERENT_USER_EXTENSIONS = [
    "Please include any details that might help.",
    "Let me know if you need more context.",
    "I appreciate your assistance with this.",
    "Could you also suggest a backup plan?",
    "What would you recommend in my situation?",
]

_COHERENT_ASSISTANT_SENTENCES = [
    "Start by identifying the main issue.",
    "Make sure you have a clear goal in mind.",
    "Next, gather any information you need.",
    "Then choose the option that best fits your situation.",
    "After that, monitor the results closely.",
    "If something goes wrong, pause and reassess.",
    "I recommend keeping a simple checklist.",
    "Consistency is usually more important than intensity.",
    "You can adjust the plan as you learn more.",
    "Let me know if you need further help.",
    "First, make a list of your priorities.",
    "Once you have that, sort them by urgency.",
    "Tackle the most important task first.",
    "Break large tasks into smaller, actionable steps.",
    "Set a realistic deadline for each step.",
    "Review your progress at the end of the day.",
    "Don't forget to ask for feedback early.",
    "A good backup plan can save a lot of stress.",
    "Try to keep your workspace organized.",
    "Small improvements add up over time.",
]

# Reasoning fragments designed to trigger step-by-step thinking.
_REASONING_USER_OPENERS = [
    "Prove that the square root of two is irrational.",
    "Walk me through a proof by induction for the sum 1 to n.",
    "Analyze the time complexity of merge sort and justify it.",
    "Is the syllogism 'All A are B; all B are C; therefore all A are C' valid?",
    "Solve the river crossing puzzle with a wolf, a goat, and a cabbage.",
    "Explain why correlation does not imply causation using an example.",
    "Design an experiment to test whether a new drug is effective.",
    "Show that there are infinitely many prime numbers.",
    "Evaluate the argument that free will is impossible in a deterministic universe.",
    "Determine the Nash equilibrium in a simple pricing game.",
    "Prove that the sum of the first n odd numbers equals n squared.",
    "Explain the difference between necessary and sufficient conditions.",
]

_REASONING_USER_FOLLOWUPS = [
    "Why is the base case necessary in this proof?",
    "Can you show the inductive step explicitly?",
    "What is the lower bound for this algorithm?",
    "Where does the argument break down if the assumption changes?",
    "Could we prove this by contradiction instead?",
    "How would you formalize this in logic?",
    "What if the set is infinite?",
    "Can you give a concrete counterexample?",
    "Which assumptions are you relying on?",
    "How do you know the solution is unique?",
    "Is the converse also true?",
    "What is the weakest hypothesis we need?",
]

_REASONING_USER_EXTENSIONS = [
    "Show every step of the proof.",
    "Justify each assumption clearly.",
    "Explain the underlying principle.",
    "Can you provide a concrete example?",
    "Make the logical structure explicit.",
]

_REASONING_ASSISTANT_SENTENCES = [
    "Let me work through this carefully.",
    "First, state the assumptions explicitly.",
    "The base case is straightforward to verify.",
    "For the inductive step, assume the statement holds for k.",
    "Then show it must hold for k plus one.",
    "This follows directly from the definitions.",
    "The contradiction shows the original assumption is false.",
    "Therefore, the conclusion follows necessarily.",
    "The key insight is recognizing the recursive structure.",
    "We can verify this with a small numerical example.",
    "The lower bound is tight because we can construct a matching example.",
    "Each step preserves the invariant we established.",
    "This completes the proof.",
    "Assume the opposite for the sake of contradiction.",
    "Rearranging terms gives the desired equality.",
    "Since the left-hand side equals the right-hand side, the claim holds.",
    "The argument relies on the law of excluded middle.",
    "Without loss of generality, we can consider a representative case.",
    "The contrapositive is easier to prove directly.",
    "Thus the set must be infinite.",
    "Any valid solution must satisfy all constraints simultaneously.",
    "The counterexample disproves the universal claim.",
    "We proceed by case analysis on the possible values.",
    "The induction hypothesis applies because the subproblem is smaller.",
    "Therefore no finite list can contain all primes.",
]


class IsoMetricGenerator(BaseConversationGenerator):
    """Base class for the three isometric (controlled) text datasets.

    Subclasses set ``dataset_type`` and ``style`` to ``coherent``,
    ``gibberish``, or ``reasoning``.
    """

    dataset_type = "iso"
    style = "base"
    cli_description = "Generate an isometric multi-turn chat dataset"

    # Same neutral system prompt for all three styles so the system message
    # contributes the same word count to each dataset.
    _system_prompt: str = "You are a helpful assistant."

    # ------------------------------------------------------------------
    # Style-specific content banks (populated per subclass in __init__)
    # ------------------------------------------------------------------
    _user_openers: List[str] = []
    _user_followups: List[str] = []
    _user_extensions: List[str] = []
    _assistant_sentences: List[str] = []

    def __init__(self, config: dict, seed: int = 42):
        super().__init__(config, seed=seed)
        if self.style == "coherent":
            self._user_openers = _COHERENT_USER_OPENERS
            self._user_followups = _COHERENT_USER_FOLLOWUPS
            self._user_extensions = _COHERENT_USER_EXTENSIONS
            self._assistant_sentences = _COHERENT_ASSISTANT_SENTENCES
        elif self.style == "gibberish":
            # The gibberish style uses the random vocabulary directly.
            pass
        elif self.style == "reasoning":
            self._user_openers = _REASONING_USER_OPENERS
            self._user_followups = _REASONING_USER_FOLLOWUPS
            self._user_extensions = _REASONING_USER_EXTENSIONS
            self._assistant_sentences = _REASONING_ASSISTANT_SENTENCES

        self._system_prompt = config.get("dataset", {}).get(
            "system_prompt", self._system_prompt
        )

    # ------------------------------------------------------------------
    # Conversation plan (shared word/turn budget across styles)
    # ------------------------------------------------------------------

    def _build_plan(self, num_conversations: int) -> List[dict]:
        """Return a deterministic schedule of turn counts and word budgets.

        The plan is independent of the content style, so every IsoMetricGenerator
        subclass that starts from the same seed and config produces the same
        schedule and therefore the same total word/turn counts.
        """
        turns_cfg = self.config.get("turns", {})
        dist = turns_cfg.get("distribution", {})
        default_min = turns_cfg.get("min", 1)
        default_max = turns_cfg.get("max", 10)

        # Build a population of turn-count buckets, repeated by their configured
        # counts, so the default run preserves the exact bucket counts.
        bucket_pool = []
        for bucket in dist.values():
            lo = bucket.get("min_turns", default_min)
            hi = bucket.get("max_turns", default_max)
            count = bucket.get("count", 1)
            bucket_pool.extend([(lo, hi)] * count)

        if not bucket_pool:
            bucket_pool = [(default_min, default_max)] * num_conversations

        if num_conversations <= len(bucket_pool):
            selected = self.rng.sample(bucket_pool, num_conversations)
        else:
            selected = self.rng.choices(bucket_pool, k=num_conversations)

        word_cfg = self.config.get("word_budget", {})
        min_total = word_cfg.get("min_per_turn", 15)
        max_total = word_cfg.get("max_per_turn", 35)
        user_share_min = word_cfg.get("user_share_min", 0.25)
        user_share_max = word_cfg.get("user_share_max", 0.45)

        plan = []
        for lo, hi in selected:
            num_turns = self.rng.randint(lo, hi)
            turns = []
            for _ in range(num_turns):
                total_words = self.rng.randint(min_total, max_total)
                share = self.rng.uniform(user_share_min, user_share_max)
                user_words = max(1, min(total_words - 1, round(total_words * share)))
                assistant_words = total_words - user_words
                turns.append({"user_words": user_words, "assistant_words": assistant_words})
            plan.append({
                "conversation_id": self.new_conversation_id(),
                "num_turns": num_turns,
                "turns": turns,
            })
        return plan

    # ------------------------------------------------------------------
    # Message rendering
    # ------------------------------------------------------------------

    def _render_user_message(self, turn_index: int, target_words: int) -> str:
        """Render a user message of exactly ``target_words`` words."""
        if self.style == "gibberish":
            return " ".join(self.rng.choice(_GIBBERISH_VOCAB) for _ in range(target_words))

        pool = self._user_openers if turn_index == 0 else self._user_followups
        base = self.rng.choice(pool)
        words = base.split()
        if len(words) >= target_words:
            return " ".join(words[:target_words])

        while len(words) < target_words:
            ext = self.rng.choice(self._user_extensions)
            words.extend(ext.split())
        return " ".join(words[:target_words])

    def _render_assistant_message(self, target_words: int) -> str:
        """Render an assistant message of exactly ``target_words`` words."""
        if self.style == "gibberish":
            return " ".join(self.rng.choice(_GIBBERISH_VOCAB) for _ in range(target_words))

        words = []
        # Safety cap: never loop more than ``target_words`` times.
        for _ in range(target_words):
            if len(words) >= target_words:
                break
            fragment = self.rng.choice(self._assistant_sentences)
            frag_words = fragment.split()
            needed = target_words - len(words)
            if len(frag_words) <= needed:
                words.extend(frag_words)
            else:
                words.extend(frag_words[:needed])
                break
        return " ".join(words)

    # ------------------------------------------------------------------
    # Conversation / dataset generation
    # ------------------------------------------------------------------

    def _generate_from_plan(self, plan_entry: dict) -> dict:
        """Materialize one conversation from a plan entry."""
        messages = [{"role": "system", "content": self._system_prompt}]
        cumulative = []
        running_chars = len(self._system_prompt)

        for turn_index, turn in enumerate(plan_entry["turns"]):
            user_msg = self._render_user_message(turn_index, turn["user_words"])
            assistant_msg = self._render_assistant_message(turn["assistant_words"])

            messages.append({"role": "user", "content": user_msg})
            running_chars += len(user_msg)
            messages.append({"role": "assistant", "content": assistant_msg})
            running_chars += len(assistant_msg)
            cumulative.append(running_chars)

        return {
            "conversation_id": plan_entry["conversation_id"],
            "topic": self.style,
            "num_turns": plan_entry["num_turns"],
            "num_messages": len(messages),
            "system_prompt": self._system_prompt,
            "messages": json.dumps(messages),
            "total_characters": running_chars,
            "total_words": sum(len(m["content"].split()) for m in messages),
            "estimated_tokens": count_message_tokens(messages),
            "cumulative_char_lengths": json.dumps(cumulative),
        }

    def generate_dataset(self, num_conversations: int = None) -> List[dict]:
        """Generate the full dataset from a shared isometric plan."""
        if num_conversations is None:
            num_conversations = self.config["dataset"]["num_conversations"]

        plan = self._build_plan(num_conversations)
        conversations = [self._generate_from_plan(entry) for entry in plan]
        self.rng.shuffle(conversations)
        return conversations

    def print_summary(self, df) -> None:
        """Print a concise summary, highlighting total word and turn counts."""
        print("\n--- Dataset Summary ---")
        print(f"Total conversations: {len(df)}")
        print(f"Total turns: {int(df['num_turns'].sum())}")
        if "total_words" in df.columns:
            print(f"Total words: {int(df['total_words'].sum()):,}")
            print(f"Mean words/conversation: {df['total_words'].mean():,.1f}")
        if "num_turns" in df.columns:
            print(f"Turn count range: {df['num_turns'].min()} - {df['num_turns'].max()}")
            print(f"Mean turns: {df['num_turns'].mean():.1f}")
        group_col = "topic"
        if group_col in df.columns:
            print(f"{group_col} distribution:")
            for value, count in df[group_col].value_counts().items():
                print(f"  {value}: {count} ({100 * count / len(df):.1f}%)")
        if "estimated_tokens" in df.columns:
            print(f"Estimated total tokens: {df['estimated_tokens'].sum():,}")
            print(f"Mean tokens/conversation: {df['estimated_tokens'].mean():,.0f}")
            print(f"Max tokens (single conversation): {df['estimated_tokens'].max():,}")
