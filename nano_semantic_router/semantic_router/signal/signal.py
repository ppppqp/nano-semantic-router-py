from nano_semantic_router.config.config import (
    Condition,
    SignalConfig,
    ComplexitySignalConfig,
    SignalOperator,
    UseCaseSignalConfig,
    SignalType,
)
import logging
from dataclasses import dataclass

from nano_semantic_router.semantic_router.classification.use_case_classifier import (
    compute_use_case_signal,
)
from nano_semantic_router.semantic_router.classification.complexity_classifier import (
    compute_complexity_signal,
)


@dataclass
class Signal:
    """Base class for signals extracted from request content. Specific signal types will inherit from this."""

    signal_type: SignalType


@dataclass
class ComplexitySignal(Signal):
    score: float  # 0-10 complexity score, where 0 is simple and 10 is complex.

    def __init__(self, score: float):
        super().__init__(signal_type=SignalType.COMPLEXITY)
        self.score = score


@dataclass
class UseCaseSignal(Signal):
    use_case: str  # categorical label indicating the use case of the request, e.g. "code_generation", "question_answering", etc.

    def __init__(self, use_case: str):
        super().__init__(signal_type=SignalType.USE_CASE)
        self.use_case = use_case


def get_signals_from_content(
    active_signals: list[SignalConfig], user_content: str
) -> list[Signal]:
    """Return a list of matched signals."""
    signal_analysis_result = []
    if not active_signals:
        logging.warning("No active signals configured; returning empty signal set.")
        return []

    for signal in active_signals:
        if isinstance(signal, ComplexitySignalConfig):
            complexity_score = compute_complexity_signal(
                model_path=signal.model_path,
                user_content=user_content,
            )
            if complexity_score.confidence >= signal.confidence_threshold:
                signal_analysis_result.append(
                    ComplexitySignal(score=complexity_score.complexity_score)
                )
        elif isinstance(signal, UseCaseSignalConfig):
            use_case_result = compute_use_case_signal(
                model_path=signal.model_path,
                use_cases=signal.use_cases,
                user_content=user_content,
            )
            logging.info(f"Computed use case signals: {use_case_result}")
            if use_case_result.confidence >= signal.confidence_threshold:
                signal_analysis_result.append(
                    UseCaseSignal(use_case=use_case_result.use_case)
                )
        else:
            logging.warning(f"Unknown signal type: {signal.signal_type}")
    return signal_analysis_result


def signal_matches_condition(signal: Signal, condition: Condition) -> bool:
    """Check if a signal matches a routing condition. Placeholder for future implementation."""
    if condition.signal.signal_type != SignalConfig.signal_type:
        return False
    if isinstance(signal, ComplexitySignal):
        assert isinstance(condition.signal, ComplexitySignal)
        if condition.operator == SignalOperator.GT:
            return signal.score > condition.signal.score
        elif condition.operator == SignalOperator.LT:
            return signal.score < condition.signal.score
        elif condition.operator == SignalOperator.EQ:
            return signal.score == condition.signal.score
        elif condition.operator == SignalOperator.NEQ:
            return signal.score != condition.signal.score
    elif isinstance(signal, UseCaseSignal):
        assert isinstance(condition.signal, UseCaseSignal)
        if condition.operator == SignalOperator.EQ:
            return signal.use_case == condition.signal.use_case
        elif condition.operator == SignalOperator.NEQ:
            return signal.use_case != condition.signal.use_case
    return False
