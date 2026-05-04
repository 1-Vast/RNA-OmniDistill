from __future__ import annotations

from math import sqrt
from typing import Iterable, Sequence

from utils.structure import canonical_pair, parse_dot_bracket, validate_structure


def _pair_set(struct: str) -> set[tuple[int, int]]:
    try:
        return set(parse_dot_bracket(struct))
    except ValueError:
        return set()


def base_pair_precision(pred_struct: str, true_struct: str) -> float:
    pred = _pair_set(pred_struct)
    true = _pair_set(true_struct)
    if not pred:
        return 1.0 if not true else 0.0
    return len(pred & true) / len(pred)


def base_pair_recall(pred_struct: str, true_struct: str) -> float:
    pred = _pair_set(pred_struct)
    true = _pair_set(true_struct)
    if not true:
        return 1.0 if not pred else 0.0
    return len(pred & true) / len(true)


def base_pair_f1(pred_struct: str, true_struct: str) -> float:
    precision = base_pair_precision(pred_struct, true_struct)
    recall = base_pair_recall(pred_struct, true_struct)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def base_pair_mcc(pred_struct: str, true_struct: str) -> float:
    length = max(len(pred_struct), len(true_struct))
    pred = _pair_set(pred_struct)
    true = _pair_set(true_struct)
    all_pairs = {(i, j) for i in range(length) for j in range(i + 1, length)}
    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    tn = len(all_pairs - pred - true)
    denom = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denom == 0 else ((tp * tn) - (fp * fn)) / denom


def token_accuracy(
    pred_tokens: Sequence[str] | str,
    true_tokens: Sequence[str] | str,
    mask: Iterable[bool] | None = None,
) -> float:
    if len(pred_tokens) != len(true_tokens):
        raise ValueError("Predicted and target token sequences must have the same length.")
    indices = list(range(len(true_tokens)))
    if mask is not None:
        mask_list = list(mask)
        if len(mask_list) != len(true_tokens):
            raise ValueError("Mask length must match token sequence length.")
        indices = [idx for idx, keep in enumerate(mask_list) if keep]
    if not indices:
        return 0.0
    correct = sum(1 for idx in indices if pred_tokens[idx] == true_tokens[idx])
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


def canonical_pair_ratio(seq: str, struct: str, allow_wobble: bool = True) -> float:
    try:
        pairs = parse_dot_bracket(struct)
    except ValueError:
        return 0.0
    if not pairs:
        return 1.0
    canonical = sum(1 for i, j in pairs if canonical_pair(seq[i], seq[j], allow_wobble))
    return canonical / len(pairs)

