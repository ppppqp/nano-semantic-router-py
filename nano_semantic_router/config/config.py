from dataclasses import dataclass, field
from typing import List


@dataclass
class ClassifierConfig:
    model_path: str = ""
    threshold: float = 0.0


@dataclass
class CacheConfig:
    size: int = 0
    eviction_policy: str = ""


@dataclass
class SignalConfig:
    signal_type: str = ""
    rules: List[str] = field(default_factory=list)


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
