from dataclasses import dataclass
from nano_semantic_router.config.config import ConditionOperator, DecisionConfig
from nano_semantic_router.semantic_router.signal.signal import (
    Signal,
    signal_matches_condition,
)
from nano_semantic_router.config.config import Condition


@dataclass
class DecisionResult:
    decision: DecisionConfig
    confidence: float
    matched_rules: list[str]  # list of rule names that matched for this decision


def make_routing_decision(
    signals: list[Signal], decisions: list[DecisionConfig]
) -> DecisionResult | None:
    """Make routing decision based on signals."""
    best_decision = None
    best_confidence = -1.0
    for decision in decisions:
        matched, confidence, matched_rules = evaluate_decision(decision, signals)
        if matched:
            if confidence > best_confidence:
                best_confidence = confidence
                best_decision = DecisionResult(
                    decision=decision,
                    confidence=confidence,
                    matched_rules=matched_rules,
                )
    return best_decision


def evaluate_decision(
    decision: DecisionConfig, signals: list[Signal]
) -> tuple[bool, float, list[str]]:
    # naive one level rule combination evaluation
    matched_rules = []
    for condition in decision.rules:
        for signal in signals:
            if signal_matches_condition(signal, condition):
                matched_rules.append(
                    f"{condition.signal.signal_type} {condition.operator} {condition.signal}"
                )
    if decision.operator == ConditionOperator.AND:
        matched = len(matched_rules) == len(decision.rules)
    elif decision.operator == ConditionOperator.OR:
        matched = len(matched_rules) > 0

    # confidence is simply the ratio of matched rules to total rules, which is a very naive approach and can be improved in the future.
    confidence = len(matched_rules) / len(decision.rules) if decision.rules else 0.0
    return matched, confidence, matched_rules
