from typing import Any
from .base_classifier import ClassificationInput, ClassificationOutput, get_model


def _build_prompt(user_prompt: str) -> str:
    return (
        "You are a strict complexity rater. "
        "Given the text, return a single number between 0 and 10 where 0 is simple and 10 is complex. "
        "Respond with only the number.\n\n"
        f"Text:\n{user_prompt}\n\nScore:"
    )


def _extract_score(raw_text: str) -> float:
    tokens = raw_text.strip().replace("\n", " ").split()
    for tok in tokens:
        try:
            score = float(tok.strip(","))
        except ValueError:
            continue
        return max(0.0, min(10.0, score))
    raise ValueError(f"Could not extract numeric score from model output: {raw_text!r}")


def _score_confidence(raw_text: str) -> float:
    # High confidence if the model returned a clean numeric token
    stripped = raw_text.strip()
    try:
        float(stripped)
        return 0.95
    except ValueError:
        return 0.6


def classify(input: ClassificationInput) -> ClassificationOutput:
    """Returns a complexity score from 0 to 10, where 0 is simple and 10 is complex."""

    model = get_model(input.model_path)
    completion: dict[str, Any] = model.create_completion(
        prompt=_build_prompt(input.prompt),
        max_tokens=8,
        temperature=0.0,
        stop=["\n"],
    )

    raw_text = completion.get("choices", [{}])[0].get("text", "")
    score = _extract_score(raw_text)
    confidence = _score_confidence(raw_text)

    return ClassificationOutput(result=f"{score}", confidence=confidence)
