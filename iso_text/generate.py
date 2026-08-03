#!/usr/bin/env python3
"""Isometric coherent-text multi-turn chat dataset generator.

This generator shares a deterministic conversation plan with
``iso_random`` and ``iso_reasoning`` so that the three datasets have the
same total number of turns and exactly the same total word count.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from iso_generator import IsoMetricGenerator


class ConversationGenerator(IsoMetricGenerator):
    dataset_type = "iso_text"
    style = "coherent"
    cli_description = "Generate an isometric coherent-text chat dataset"


def main():
    ConversationGenerator.main()


if __name__ == "__main__":
    main()
