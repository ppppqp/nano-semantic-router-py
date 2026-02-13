from dataclasses import dataclass
from functools import lru_cache

from llama_cpp import Llama


@dataclass
class ClassificationInput:
    """Input for general classification task (text2text)"""

    model_path: str
    user_content: str


@dataclass
class ClassificationOutput:
    """Output for general classification task (text2text)"""

    raw_result: str
    confidence: float


@lru_cache(maxsize=2)
def get_model(model_path: str) -> Llama:
    """Load and cache a llama.cpp model to avoid repeated disk reads."""
    return Llama(model_path=model_path, n_ctx=2048, verbose=False)


class Classifier:
    """Base classifier interface. Specific classifiers (e.g. complexity, use case) will implement the classify function."""

    @staticmethod
    def _build_prompt(*args, **kwargs) -> str:
        raise NotImplementedError(
            "Classifier subclasses must implement the _build_prompt method."
        )

    @staticmethod
    def classify(input: ClassificationInput) -> ClassificationOutput:
        raise NotImplementedError(
            "Classifier subclasses must implement the classify method."
        )
