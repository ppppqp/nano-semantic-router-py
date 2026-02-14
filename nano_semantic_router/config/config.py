from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, TYPE_CHECKING

from enum import StrEnum

if TYPE_CHECKING:
    from nano_semantic_router.semantic_router.signal.signal import (
        Signal,
    )


@dataclass
class ClassifierConfig:
    model_ref: str = ""


class SignalType(StrEnum):
    COMPLEXITY = "complexity"
    USE_CASE = "use_case"
    UNKNOWN = "unknown"


@dataclass
class SignalConfig:
    signal_type: SignalType = SignalType.UNKNOWN
    confidence_threshold: float = 0.0
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)


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
class Model:
    name: str
    endpoint: str
    access_key: str
    model_type: str  # e.g. "openai", "anthropic", "local". Only support openai and local for now.
    is_default: bool = False
    path: str = ""  # optional local path for local models


@dataclass
class DecisionConfig:
    name: str
    model_ref: str
    rules: List[Condition] = field(default_factory=list)
    operator: ConditionOperator = ConditionOperator.AND


@dataclass
class RouterConfig:
    models: dict[str, Model]
    decisions: list[DecisionConfig] = field(default_factory=list)
    signals: list[SignalConfig] = field(default_factory=list)
