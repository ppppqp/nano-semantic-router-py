from dataclasses import dataclass, field
from typing import List, Literal

from enum import StrEnum


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


@dataclass
class DecisionConfig:
    """Placeholder for future decision configuration."""

    pass


@dataclass
class RouterConfig:
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    decisions: DecisionConfig = field(default_factory=DecisionConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
