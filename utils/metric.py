from __future__ import annotations

from math import sqrt
from typing import Iterable, Sequence

from utils.struct import canonical_pair, parse_dot_bracket, validate_structure


def _pair_set(struct: str) -> set[tuple[int, int]]:
    try:
        return set(parse_dot_bracket(struct))
    except ValueError:
        return set()


def base_pair_precision(pred_struct: str, true_struct: str) -> float:
    pred = _pair_set(pred_struct)
    true = _pair_set(true_struct)
    if not pred:
        return 0.0
    return len(pred & true) / len(pred)


def base_pair_recall(pred_struct: str, true_struct: str) -> float:
    pred = _pair_set(pred_struct)
    true = _pair_set(true_struct)
    if not true:
        return 0.0
    return len(pred & true) / len(true)


def base_pair_f1(pred_struct: str, true_struct: str) -> float:
    precision = base_pair_precision(pred_struct, true_struct)
    recall = base_pair_recall(pred_struct, true_struct)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mcc(pred_struct: str, true_struct: str) -> float:
    length = max(len(pred_struct), len(true_struct))
    if length < 2:
        return 0.0
    pred = _pair_set(pred_struct)
    true = _pair_set(true_struct)
    all_pairs = {(i, j) for i in range(length) for j in range(i + 1, length)}
    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    tn = len(all_pairs - pred - true)
    denom = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denom == 0 else ((tp * tn) - (fp * fn)) / denom


def base_pair_mcc(pred_struct: str, true_struct: str) -> float:
    return mcc(pred_struct, true_struct)


def token_accuracy(
    labels: Sequence[str] | str,
    preds: Sequence[str] | str,
    mask: Iterable[bool] | None = None,
) -> float:
    """Token accuracy on an optional boolean mask.

    Empty masks return 0.0 because no supervised token was available to score.
    """
    if len(labels) != len(preds):
        raise ValueError("Label and prediction token sequences must have the same length.")
    indices = list(range(len(labels)))
    if mask is not None:
        mask_list = list(mask)
        if len(mask_list) != len(labels):
            raise ValueError("Mask length must match token sequence length.")
        indices = [idx for idx, keep in enumerate(mask_list) if keep]
    if not indices:
        return 0.0
    correct = sum(1 for idx in indices if labels[idx] == preds[idx])
    return correct / len(indices)


def valid_structure_rate(
    seqs: Sequence[str],
    structs: Sequence[str],
    allow_wobble: bool = True,
) -> float:
    if len(seqs) != len(structs):
        raise ValueError("Sequence and structure lists must have the same length.")
    if not seqs:
        return 0.0
    valid = sum(
        1 for seq, struct in zip(seqs, structs) if validate_structure(seq, struct, allow_wobble)
    )
    return valid / len(seqs)


def canonical_pair_ratio(
    seq: str,
    struct: str,
    allow_wobble: bool = True,
) -> float:
    try:
        pairs = parse_dot_bracket(struct)
    except ValueError:
        return 0.0
    if not pairs:
        return 0.0
    canonical = sum(1 for i, j in pairs if canonical_pair(seq[i], seq[j], allow_wobble))
    return canonical / len(pairs)


def all_dot_ratio(structs: Sequence[str]) -> float:
    if not structs:
        return 0.0
    return sum(1 for struct in structs if set(struct) <= {"."}) / len(structs)


def average_pair_count(structs: Sequence[str]) -> float:
    if not structs:
        return 0.0
    return sum(len(_pair_set(struct)) for struct in structs) / len(structs)


def evaluate_structures(
    pred_structs: Sequence[str],
    true_structs: Sequence[str],
    seqs: Sequence[str],
    allow_wobble: bool = True,
) -> dict:
    if not (len(pred_structs) == len(true_structs) == len(seqs)):
        raise ValueError("Predicted structures, true structures, and sequences must have equal lengths.")
    if not pred_structs:
        return {
            "pair_precision": 0.0,
            "pair_recall": 0.0,
            "pair_f1": 0.0,
            "mcc": 0.0,
            "valid_structure_rate": 0.0,
            "canonical_pair_ratio": 0.0,
            "all_dot_ratio": 0.0,
            "avg_pred_pair_count": 0.0,
            "avg_true_pair_count": 0.0,
            "pair_count_gap": 0.0,
        }

    precision = sum(base_pair_precision(pred, true) for pred, true in zip(pred_structs, true_structs))
    recall = sum(base_pair_recall(pred, true) for pred, true in zip(pred_structs, true_structs))
    f1 = sum(base_pair_f1(pred, true) for pred, true in zip(pred_structs, true_structs))
    mcc_value = sum(mcc(pred, true) for pred, true in zip(pred_structs, true_structs))
    canonical_values = [
        canonical_pair_ratio(seq, pred, allow_wobble=allow_wobble)
        for seq, pred in zip(seqs, pred_structs)
    ]
    avg_pred = average_pair_count(pred_structs)
    avg_true = average_pair_count(true_structs)
    count = len(pred_structs)
    return {
        "pair_precision": precision / count,
        "pair_recall": recall / count,
        "pair_f1": f1 / count,
        "mcc": mcc_value / count,
        "valid_structure_rate": valid_structure_rate(seqs, pred_structs, allow_wobble=allow_wobble),
        "canonical_pair_ratio": sum(canonical_values) / count,
        "all_dot_ratio": all_dot_ratio(pred_structs),
        "avg_pred_pair_count": avg_pred,
        "avg_true_pair_count": avg_true,
        "pair_count_gap": avg_pred - avg_true,
    }

