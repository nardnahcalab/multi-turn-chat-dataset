"""Shared pytest fixtures/helpers for the dataset generator tests.

The generators live in ``<type>/generate.py`` and are loaded dynamically so the
tests don't depend on them being importable as a package. pdf/image need their
cached source metadata (arxiv_papers.json / wikipedia_images.json), which is
tracked in the repo, so the tests never hit the network.
"""

import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generator_base import BaseConversationGenerator  # noqa: E402

# All directly-runnable generators (mixed is a loader over these, tested separately).
DATASETS = ["text", "reasoning", "random", "repeat", "pdf", "image", "agentic"]


def load_generate_module(dataset: str):
    """Import ``<dataset>/generate.py`` under a unique module name."""
    path = ROOT / dataset / "generate.py"
    spec = importlib.util.spec_from_file_location(f"{dataset}_generate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_generator_class(module):
    for obj in vars(module).values():
        if (isinstance(obj, type)
                and issubclass(obj, BaseConversationGenerator)
                and obj is not BaseConversationGenerator):
            return obj
    raise LookupError(f"No BaseConversationGenerator subclass in {module.__name__}")


def make_generator(dataset: str, seed: int = 7):
    """Construct a generator for ``dataset`` plus its loaded config."""
    module = load_generate_module(dataset)
    cls = _find_generator_class(module)
    config = yaml.safe_load((ROOT / dataset / "config.yaml").read_text())

    if dataset == "pdf":
        papers = json.loads((ROOT / "pdf" / "data" / "arxiv_papers.json").read_text())
        return cls(config, papers, seed=seed), config
    if dataset == "image":
        images = json.loads((ROOT / "image" / "data" / "wikipedia_images.json").read_text())
        return cls(config, images, seed=seed), config
    return cls(config, seed=seed), config
