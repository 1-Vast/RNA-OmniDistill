"""Reward utilities for RNA structure scoring and preference comparison.

Provides dot-bracket/pair conversion helpers and a multi-factor
structure quality scorer.  No LLM dependency.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from utils.struct import canonical_pair as _canonical_pair
from utils.struct import pairs_to_dot_bracket as _pairs_to_dot_bracket
from utils.struct import parse_dot_bracket as _parse_dot_bracket

Pair = Tuple[int, int]


def dotbracket_to_pairs(struct: str) -> List[Pair]:
    """Parse dot-bracket string into sorted 0-based base pairs.

    Thin wrapper for consistency with the naming convention used
    throughout the preference pipeline.
    """
    return _parse_dot_bracket(struct)


def pairs_to_dotbracket(length: int, pairs: Iterable[Sequence[int]]) -> str:
    """Convert 0-based pair list into dot-bracket string."""
    return _pairs_to_dot_bracket(pairs, length)


def valid_dotbracket(struct: str) -> bool:
    """Return True when *struct* is a syntactically valid dot-bracket string."""
    try:
        _parse_dot_bracket(struct)
        return True
    except ValueError:
        return False


def canonical_pair(a: str, b: str, allow_wobble: bool = True) -> bool:
    """Check whether two RNA bases form a canonical (or wobble) pair."""
    return _canonical_pair(a, b, allow_wobble=allow_wobble)


def score_struct(
    seq: str,
    struct: str,
    *,
    min_loop: int = 3,
    allow_wobble: bool = True,
) -> Dict[str, float | int | bool]:
    """Return a multi-factor quality dictionary for one structure.

    Keys
    ----
    valid : bool
        Whether the dot-bracket is syntactically well-formed.
    pair_count : int
        Number of base pairs.
    pair_density : float
        pair_count / max(1, seq_len).
    canonical_ratio : float
        Fraction of pairs whose bases form canonical/wobble pairs.
    isolated_pairs : int
        Pairs not adjacent to any other pair (stem of length 1).
    stem_continuity : float
        Average fraction of pairs belonging to stems of length >= 2.
        Returns 0.0 when no pairs exist.
    all_dot : bool
        True when struct consists entirely of '.'.
    min_loop_violations : int
        Number of pairs with index distance < *min_loop*.
    """
    length = len(seq)
    result: Dict[str, float | int | bool] = {
        "valid": False,
        "pair_count": 0,
        "pair_density": 0.0,
        "canonical_ratio": 0.0,
        "isolated_pairs": 0,
        "stem_continuity": 0.0,
        "all_dot": True,
        "min_loop_violations": 0,
    }

    try:
        pairs = _parse_dot_bracket(struct)
    except ValueError:
        return result

    result["valid"] = True
    result["pair_count"] = len(pairs)
    result["pair_density"] = len(pairs) / max(1, length)
    result["all_dot"] = len(pairs) == 0

    if not pairs:
        return result

    # -- canonical ratio ---------------------------------------------------
    canonical = 0
    for i, j in pairs:
        a = seq[i] if i < length else "N"
        b = seq[j] if j < length else "N"
        if a != "N" and b != "N" and _canonical_pair(a, b, allow_wobble):
            canonical += 1
    result["canonical_ratio"] = canonical / max(1, len(pairs))

    # -- isolated pairs & stem continuity ----------------------------------
    pair_set = set(pairs)
    in_stem = set()
    for i, j in pairs:
        if (i + 1, j - 1) in pair_set or (i - 1, j + 1) in pair_set:
            in_stem.add((i, j))
    result["isolated_pairs"] = len(pairs) - len(in_stem)
    result["stem_continuity"] = len(in_stem) / max(1, len(pairs))

    # -- min-loop violations -----------------------------------------------
    violations = 0
    for i, j in pairs:
        if abs(j - i) < min_loop:
            violations += 1
    result["min_loop_violations"] = violations

    return result
