class ClassificationInput:
    """Input for general classification task (text2text)"""

    model_path: str
    prompt: str


class ClassificationOutput:
    """Output for general classification task (text2text)"""

    result: str
    confidence: float


def classify(input: ClassificationInput) -> float:
    """Returns a complexity score from 0 to 10, where 0 is simple and 10 is complex."""
    # load model
    # run inference with llamacpp
