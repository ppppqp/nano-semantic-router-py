from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, TYPE_CHECKING

from enum import StrEnum

if TYPE_CHECKING:
    from nano_semantic_router.semantic_router.signal.signal import (
        Signal,
        UseCaseSignal,
        ComplexitySignal,
    )


@dataclass
class ClassifierConfig:
    model_path: str = ""
    threshold: float = 0.0


@dataclass
class CacheConfig:
    size: int = 0
    eviction_policy: str = ""


class SignalType(StrEnum):
    COMPLEXITY = "complexity"
    USE_CASE = "use_case"
    UNKNOWN = "unknown"


@dataclass
class SignalConfig:
    signal_type: SignalType = SignalType.UNKNOWN
    confidence_threshold: float = 0.0
    model_path: str = ""


@dataclass
class ComplexitySignalConfig(SignalConfig):
    """0-10 complexity score, where 0 is simple and 10 is complex."""

    signal_type: Literal[SignalType.COMPLEXITY] = SignalType.COMPLEXITY


@dataclass
class UseCaseSignalConfig(SignalConfig):
    """use_case signals are categorical labels indicating the use case of the request, e.g. "code_generation", "question_answering", etc."""

    signal_type: Literal[SignalType.USE_CASE] = SignalType.USE_CASE
    use_cases: List[str] = field(default_factory=list)


class SignalOperator(StrEnum):
    EQ = "EQ"
    NEQ = "NEQ"
    GT = "GT"
    LT = "LT"


@dataclass
class Condition:
    signal: Signal
    operator: SignalOperator


class ConditionOperator(StrEnum):
    AND = "AND"
    OR = "OR"


@dataclass
class ModelRef:
    model: str
    endpoint: str
    access_key: str
    model_type: (
        str  # e.g. "openai", "anthropic", "custom". Only support openai for now.
    )


@dataclass
class DecisionConfig:
    name: str
    model_ref: ModelRef
    rules: List[Condition] = field(default_factory=list)
    operator: ConditionOperator = ConditionOperator.AND


@dataclass
class RouterConfig:
    default_model: ModelRef
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    decisions: list[DecisionConfig] = field(default_factory=list)
    signals: list[SignalConfig] = field(default_factory=list)
