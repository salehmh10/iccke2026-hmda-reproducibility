"""Tiny mathematical contracts for synthetic documentation tests only."""
import math


def denial_orientation(approval_labels, approval_probabilities):
    if len(approval_labels) != len(approval_probabilities):
        raise ValueError("unaligned inputs")
    if any(y not in (0, 1) for y in approval_labels):
        raise ValueError("labels must be binary")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in approval_probabilities):
        raise ValueError("probabilities must be in [0,1]")
    return [1-y for y in approval_labels], [1-p for p in approval_probabilities]


def synthetic_average_precision(labels, scores):
    """AP at tied score groups; only used with tiny synthetic test examples."""
    if not labels or len(labels) != len(scores) or any(y not in (0, 1) for y in labels):
        raise ValueError("invalid binary example")
    positives = sum(labels)
    if not positives:
        return 0.0
    ordered = sorted(zip(scores, labels), reverse=True)
    tp = seen = 0
    result = 0.0
    for score in sorted(set(scores), reverse=True):
        group = [y for s, y in ordered if s == score]
        increment = sum(group)
        tp += increment
        seen += len(group)
        result += increment / positives * tp / seen
    return result


def routing_strength(probability):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability outside [0,1]")
    return 0.0 if probability <= .75 else .75 * (probability - .75) / .25


def synthetic_primary(global_value, residual, probability):
    if not all(map(math.isfinite, (global_value, residual))):
        raise ValueError("nonfinite synthetic values")
    return global_value + routing_strength(probability) * residual
