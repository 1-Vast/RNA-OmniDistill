from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Set


def random_token_mask(positions: Sequence[int], mask_ratio: float, rng: random.Random | None = None) -> List[int]:
    """Sample token positions to mask, with at least one position when available."""
    if not positions:
        return []
    rng = rng or random
    ratio = min(max(float(mask_ratio), 0.0), 1.0)
    count = max(1, int(round(len(positions) * ratio)))
    count = min(count, len(positions))
    return sorted(rng.sample(list(positions), count))


def pair_aware_mask_positions(
    nucleotide_positions: Iterable[int],
    pairs: Iterable[Sequence[int]],
) -> Set[int]:
    """Expand nucleotide mask positions so paired bases are masked together."""
    selected = {int(pos) for pos in nucleotide_positions}
    for raw_i, raw_j in pairs:
        i, j = int(raw_i), int(raw_j)
        if i in selected or j in selected:
            selected.add(i)
            selected.add(j)
    return selected


def motif_span_mask_positions(
    motifs: Sequence[dict],
    length: int,
    rng: random.Random | None = None,
) -> Set[int]:
    """Pick one motif span and return all nucleotide positions inside it."""
    rng = rng or random
    valid = [
        (max(0, int(motif["start"])), min(length - 1, int(motif["end"])))
        for motif in motifs
        if "start" in motif and "end" in motif and length > 0
    ]
    valid = [(start, end) for start, end in valid if start <= end]
    if not valid:
        return set()
    start, end = rng.choice(valid)
    return set(range(start, end + 1))


def random_span_positions(
    length: int,
    mask_ratio: float,
    rng: random.Random | None = None,
) -> Set[int]:
    """Sample a contiguous span over nucleotide coordinates."""
    if length <= 0:
        return set()
    rng = rng or random
    span_len = max(1, int(round(length * min(max(mask_ratio, 0.0), 1.0))))
    span_len = min(span_len, length)
    start = rng.randint(0, length - span_len)
    return set(range(start, start + span_len))


