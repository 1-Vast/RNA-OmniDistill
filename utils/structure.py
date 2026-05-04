from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


Pair = Tuple[int, int]


OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
CLOSE_TO_OPEN = {v: k for k, v in OPEN_TO_CLOSE.items()}
ALLOWED_STRUCT_CHARS = set(".()[]{}")


def canonical_pair(a: str, b: str, allow_wobble: bool = True) -> bool:
    """Return whether two RNA bases form a canonical or wobble pair."""
    pair = (a.upper().replace("T", "U"), b.upper().replace("T", "U"))
    allowed = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
    if allow_wobble:
        allowed.update({("G", "U"), ("U", "G")})
    return pair in allowed


def parse_dot_bracket(struct: str) -> List[Pair]:
    """Parse dot-bracket structure into 0-based base pairs."""
    stacks: Dict[str, List[int]] = {open_char: [] for open_char in OPEN_TO_CLOSE}
    pairs: List[Pair] = []

    for idx, char in enumerate(struct):
        if char == ".":
            continue
        if char in OPEN_TO_CLOSE:
            stacks[char].append(idx)
            continue
        if char in CLOSE_TO_OPEN:
            open_char = CLOSE_TO_OPEN[char]
            if not stacks[open_char]:
                raise ValueError(f"Unmatched closing bracket {char!r} at position {idx}.")
            left = stacks[open_char].pop()
            pairs.append((left, idx))
            continue
        raise ValueError(f"Unsupported dot-bracket character {char!r} at position {idx}.")

    unmatched = [(char, stack[-1]) for char, stack in stacks.items() if stack]
    if unmatched:
        char, idx = unmatched[0]
        raise ValueError(f"Unmatched opening bracket {char!r} at position {idx}.")

    return sorted(pairs)


def pairs_to_dot_bracket(pairs: Iterable[Sequence[int]], length: int) -> str:
    """Convert non-pseudoknotted 0-based pairs to dot-bracket structure."""
    chars = ["."] * length
    used = set()
    normalized: List[Pair] = []

    for raw_i, raw_j in pairs:
        i, j = int(raw_i), int(raw_j)
        if i == j:
            raise ValueError(f"Self-pair at position {i} is invalid.")
        if i > j:
            i, j = j, i
        if i < 0 or j >= length:
            raise ValueError(f"Pair {(i, j)} is outside structure length {length}.")
        if i in used or j in used:
            raise ValueError(f"Position reused by pair {(i, j)}.")
        used.add(i)
        used.add(j)
        normalized.append((i, j))

    for i, j in normalized:
        chars[i] = "("
        chars[j] = ")"
    return "".join(chars)


def validate_structure(seq: str, struct: str, allow_wobble: bool = True) -> bool:
    """Validate length, bracket balance, one-pair-per-position, and pair chemistry."""
    if len(seq) != len(struct):
        return False
    try:
        pairs = parse_dot_bracket(struct)
    except ValueError:
        return False

    seen = set()
    for i, j in pairs:
        if i in seen or j in seen:
            return False
        seen.add(i)
        seen.add(j)
        if seq[i].upper() != "N" and seq[j].upper() != "N":
            if not canonical_pair(seq[i], seq[j], allow_wobble=allow_wobble):
                return False
    return True


def infer_simple_motifs(
    seq: str | None = None,
    struct: str | None = None,
    pairs: Iterable[Sequence[int]] | None = None,
) -> List[Dict[str, int | str]]:
    """Infer approximate STEM and HAIRPIN motifs from pairs or dot-bracket text."""
    length = len(seq) if seq is not None else len(struct or "")
    if pairs is None and struct:
        try:
            pairs = parse_dot_bracket(struct)
        except ValueError:
            pairs = []
    normalized = sorted((min(int(i), int(j)), max(int(i), int(j))) for i, j in (pairs or []))
    motifs: List[Dict[str, int | str]] = []
    if not normalized:
        if length:
            motifs.append({"type": "MULTI_LOOP", "start": 0, "end": length - 1})
        return motifs

    pair_set = set(normalized)
    visited = set()
    for i, j in normalized:
        if (i, j) in visited:
            continue
        stem_start, stem_end = i, j
        cur_i, cur_j = i, j
        visited.add((cur_i, cur_j))
        while (cur_i + 1, cur_j - 1) in pair_set:
            cur_i += 1
            cur_j -= 1
            visited.add((cur_i, cur_j))
        motifs.append({"type": "STEM", "start": stem_start, "end": stem_end})
        if cur_j - cur_i > 1:
            motifs.append({"type": "HAIRPIN", "start": cur_i + 1, "end": cur_j - 1})

    motifs.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return motifs

