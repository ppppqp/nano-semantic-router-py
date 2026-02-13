from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from llama_cpp import Llama


@dataclass
class ClassificationInput:
    """Input for general classification task (text2text)"""

    model_path: str
    prompt: str


@dataclass
class ClassificationOutput:
    """Output for general classification task (text2text)"""

    result: str
    confidence: float


@lru_cache(maxsize=2)
def get_model(model_path: str) -> Llama:
    """Load and cache a llama.cpp model to avoid repeated disk reads."""
    return Llama(model_path=model_path, n_ctx=2048, verbose=False)
