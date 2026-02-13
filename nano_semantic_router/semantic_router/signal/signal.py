from nano_semantic_router.config.config import (
    SignalConfig,
    ComplexitySignalConfig,
    SignalType,
    UseCaseSignalConfig,
)
import logging
from dataclasses import dataclass


class Signal:
    """Base class for signals extracted from request content. Specific signal types will inherit from this."""

    pass


@dataclass
class ComplexitySignal(Signal):
    score: float  # 0-10 complexity score, where 0 is simple and 10 is complex.


@dataclass
class UseCaseSignal(Signal):
    use_case: str  # categorical label indicating the use case of the request, e.g. "code_generation", "question_answering", etc.


def get_signal_from_content(
    content: str, active_signals: list[SignalConfig]
) -> dict[SignalType, Signal]:
    signal_analysis_result = {}
    if not active_signals:
        logging.warning("No active signals configured; returning empty signal set.")
        return signal_analysis_result

    for signal in active_signals:
        if isinstance(signal, ComplexitySignalConfig):
            complexity_score = compute_complexity_signal(content)
            logging.info(f"Computed complexity signal: {complexity_score}")
            signal_analysis_result[SignalType.COMPLEXITY] = ComplexitySignal(
                score=complexity_score
            )
        elif isinstance(signal, UseCaseSignalConfig):
            use_case_scores = compute_use_case_signal(content, signal.use_cases)
            logging.info(f"Computed use case signals: {use_case_scores}")
            signal_analysis_result[SignalType.USE_CASE] = UseCaseSignal(
                use_case=use_case_scores
            )
        else:
            logging.warning(f"Unknown signal type: {signal.signal_type}")
    return signal_analysis_result


def compute_complexity_signal(content: str) -> float:
    pass


def compute_use_case_signal(content: str, use_cases: list[str]) -> str:
    pass
