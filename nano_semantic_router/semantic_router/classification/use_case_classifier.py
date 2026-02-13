from difflib import get_close_matches
from typing import Any

from .base_classifier import (
    ClassificationInput,
    ClassificationOutput,
    get_model,
    Classifier,
)
from dataclasses import dataclass


@dataclass
class UseCaseSignalOutput(ClassificationOutput):
    use_case: str


class UseCaseClassifier(Classifier):
    @staticmethod
    def _build_prompt(user_prompt: str, use_cases: list[str]) -> str:
        cases = "\n".join(f"- {case}" for case in use_cases)
        return (
            "You are a strict classifier. "
            "Given the text, choose exactly one use case label from the provided list. "
            "Respond with only the label, nothing else.\n\n"
            f"Available use cases:\n{cases}\n\n"
            f"Text:\n{user_prompt}\n\n"
            "Use case:"
        )

    @staticmethod
    def classify(
        input: ClassificationInput, use_cases: list[str]
    ) -> ClassificationOutput:
        """Returns the most likely use case label from the provided list."""

        if not use_cases:
            raise ValueError("use_cases must be a non-empty list")

        max_tokens = (
            max(len(case) for case in use_cases) + 10
        )  # add some buffer for model output
        model = get_model(input.model_path)
        completion: dict[str, Any] = model.create_completion(
            prompt=UseCaseClassifier._build_prompt(input.user_content, use_cases),
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["\n"],
        )

        raw_text = completion.get("choices", [{}])[0].get("text", "")
        use_case = _extract_use_case(raw_text, use_cases)
        confidence = _score_confidence(raw_text, use_cases)

        return ClassificationOutput(raw_result=use_case, confidence=confidence)


def _clean(text: str) -> str:
    return text.strip().strip(".,;:!?").lower()


def _extract_use_case(raw_text: str, use_cases: list[str]) -> str:
    normalized = {_clean(case): case for case in use_cases}
    cleaned_raw = _clean(raw_text)

    if cleaned_raw in normalized:
        return normalized[cleaned_raw]

    closest = get_close_matches(cleaned_raw, list(normalized.keys()), n=1, cutoff=0.6)
    if closest:
        return normalized[closest[0]]

    # Fallback to raw output if no close match found
    return raw_text.strip()


def _score_confidence(raw_text: str, use_cases: list[str]) -> float:
    # TODO: use logprobs for better confidence scoring
    cleaned_raw = _clean(raw_text)
    normalized_keys = [_clean(case) for case in use_cases]

    if cleaned_raw in normalized_keys:
        return 0.95

    if get_close_matches(cleaned_raw, normalized_keys, n=1, cutoff=0.6):
        return 0.7

    return 0.4


def compute_use_case_signal(
    model_path: str,
    use_cases: list[str],
    user_content: str,
    non_user_content: str,
) -> UseCaseSignalOutput:
    """Helper to compute use case signal from request content."""
    model_output = UseCaseClassifier.classify(
        ClassificationInput(model_path=model_path, user_content=user_content),
        use_cases=use_cases,
    )
    return UseCaseSignalOutput(
        raw_result=model_output.raw_result,
        confidence=model_output.confidence,
        use_case=model_output.raw_result,
    )
