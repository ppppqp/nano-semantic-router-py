from difflib import get_close_matches
from typing import Any

from .base_classifier import ClassificationInput, ClassificationOutput, get_model


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


def classify(input: ClassificationInput, use_cases: list[str]) -> ClassificationOutput:
    """Returns the most likely use case label from the provided list."""

    if not use_cases:
        raise ValueError("use_cases must be a non-empty list")

    max_tokens = (
        max(len(case) for case in use_cases) + 10
    )  # add some buffer for model output
    model = get_model(input.model_path)
    completion: dict[str, Any] = model.create_completion(
        prompt=_build_prompt(input.prompt, use_cases),
        max_tokens=max_tokens,
        temperature=0.0,
        stop=["\n"],
    )

    raw_text = completion.get("choices", [{}])[0].get("text", "")
    use_case = _extract_use_case(raw_text, use_cases)
    confidence = _score_confidence(raw_text, use_cases)

    return ClassificationOutput(result=use_case, confidence=confidence)
