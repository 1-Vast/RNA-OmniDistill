from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from data.tokenizer import RNAOmniTokenizer
from utils.structure import canonical_pair, parse_dot_bracket, pairs_to_dot_bracket


def _forward_model(model: torch.nn.Module, batch: dict) -> dict:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        segment_ids=batch["segment_ids"],
        task_ids=batch["task_ids"],
        time_steps=batch["time_steps"],
        seq_positions=batch.get("seq_positions"),
    )


def _build_inference_batch(
    tokenizer: RNAOmniTokenizer,
    task_name: str,
    seq: str,
    struct: str,
    family: str | None = None,
    motifs: Sequence[dict] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[dict, List[int], List[int]]:
    tokens: List[str] = []
    segment_ids: List[int] = []
    seq_positions: List[int] = []
    struct_positions: List[int] = []

    def add(token: str, segment_id: int) -> int:
        tokens.append(token)
        segment_ids.append(segment_id)
        return len(tokens) - 1

    add(tokenizer.task_token(task_name), 0)
    if task_name == "motif_control":
        add("<FAMILY>", 3)
        add(tokenizer.family_token(family), 3)
        add("</FAMILY>", 3)
        add("<MOTIF>", 3)
        for motif in motifs or []:
            add(tokenizer.motif_token(motif.get("type")), 3)
        add("</MOTIF>", 3)

    add("<SEQ>", 1)
    for base in seq:
        seq_positions.append(add(base, 1))
    add("</SEQ>", 1)
    add("<STRUCT>", 2)
    for char in struct:
        struct_positions.append(add(char, 2))
    add("</STRUCT>", 2)

    input_ids = torch.tensor([tokenizer.encode(tokens)], dtype=torch.long, device=device)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "segment_ids": torch.tensor([segment_ids], dtype=torch.long, device=device),
        "task_ids": torch.tensor([tokenizer.task_to_id[task_name]], dtype=torch.long, device=device),
        "time_steps": torch.ones(1, dtype=torch.float32, device=device),
        "seq_positions": torch.tensor([seq_positions], dtype=torch.long, device=device),
        "lengths": torch.tensor([len(seq)], dtype=torch.long, device=device),
    }
    return batch, seq_positions, struct_positions


@torch.no_grad()
def entropy_iterative_unmask(
    model: torch.nn.Module,
    batch: dict,
    target_positions: Sequence[int],
    allowed_token_ids: Sequence[int],
    mask_id: int,
    num_steps: int = 32,
) -> torch.Tensor:
    """Reveal masked positions with the lowest predictive entropy first."""
    input_ids = batch["input_ids"].clone()
    if not target_positions:
        return input_ids
    target_set = set(int(pos) for pos in target_positions)
    input_ids[0, list(target_set)] = mask_id
    remaining = set(target_set)
    allowed = torch.tensor(list(allowed_token_ids), dtype=torch.long, device=input_ids.device)
    steps = max(1, int(num_steps))

    for step in range(steps):
        if not remaining:
            break
        batch["input_ids"] = input_ids
        batch["time_steps"] = torch.full_like(batch["time_steps"], 1.0 - (step / steps))
        outputs = _forward_model(model, batch)
        logits = outputs["token_logits"][0, list(remaining)]
        restricted = logits.index_select(-1, allowed)
        probs = torch.softmax(restricted, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        reveal_count = max(1, math.ceil(len(remaining) / (steps - step)))
        ordered = torch.argsort(entropy)[:reveal_count].tolist()
        remaining_list = list(remaining)
        for idx in ordered:
            token_pos = remaining_list[idx]
            token_id = allowed[int(torch.argmax(probs[idx]))].item()
            input_ids[0, token_pos] = token_id
            remaining.remove(token_pos)
    return input_ids


def nussinov_decode(
    seq: str,
    pair_probs: torch.Tensor | Sequence[Sequence[float]],
    min_loop_length: int = 3,
    allow_wobble: bool = True,
    min_pair_prob: float = 0.5,
) -> str:
    """Decode a non-pseudoknotted dot-bracket structure from pair probabilities."""
    if isinstance(pair_probs, torch.Tensor):
        probs = pair_probs.detach().float().cpu()
    else:
        probs = torch.tensor(pair_probs, dtype=torch.float32)
    length = len(seq)
    if length == 0:
        return ""
    dp = torch.zeros((length, length), dtype=torch.float32)
    choice: Dict[Tuple[int, int], tuple] = {}

    def pair_score(i: int, j: int) -> float:
        if j - i <= min_loop_length:
            return -1e6
        if seq[i] != "N" and seq[j] != "N" and not canonical_pair(seq[i], seq[j], allow_wobble):
            return -1e6
        prob = float(probs[i, j])
        return prob - min_pair_prob

    for span in range(1, length):
        for i in range(0, length - span):
            j = i + span
            best = dp[i + 1, j] if i + 1 <= j else torch.tensor(0.0)
            choice[(i, j)] = ("skip_i",)
            if dp[i, j - 1] > best:
                best = dp[i, j - 1]
                choice[(i, j)] = ("skip_j",)
            score = pair_score(i, j)
            if score > -1e5:
                paired = (dp[i + 1, j - 1] if i + 1 <= j - 1 else torch.tensor(0.0)) + score
                if paired > best:
                    best = paired
                    choice[(i, j)] = ("pair", i, j)
            for k in range(i + 1, j):
                split = dp[i, k] + dp[k + 1, j]
                if split > best:
                    best = split
                    choice[(i, j)] = ("split", k)
            dp[i, j] = best

    pairs: List[tuple[int, int]] = []

    def backtrack(i: int, j: int) -> None:
        if i >= j:
            return
        action = choice.get((i, j), ("skip_i",))
        if action[0] == "skip_i":
            backtrack(i + 1, j)
        elif action[0] == "skip_j":
            backtrack(i, j - 1)
        elif action[0] == "pair":
            pairs.append((action[1], action[2]))
            backtrack(i + 1, j - 1)
        else:
            k = action[1]
            backtrack(i, k)
            backtrack(k + 1, j)

    backtrack(0, length - 1)
    return pairs_to_dot_bracket(pairs, length)


@torch.no_grad()
def generate_structure_seq2struct(
    model: torch.nn.Module,
    tokenizer: RNAOmniTokenizer,
    seq: str,
    decoding_config: dict,
    device: torch.device | str = "cpu",
) -> str:
    seq = seq.upper().replace("T", "U")
    struct = "." * len(seq)
    batch, _, struct_positions = _build_inference_batch(tokenizer, "seq2struct", seq, struct, device=device)
    allowed = [tokenizer.token_id(token) for token in tokenizer.structure_tokens]
    input_ids = entropy_iterative_unmask(
        model,
        batch,
        struct_positions,
        allowed,
        tokenizer.mask_id,
        num_steps=int(decoding_config.get("num_steps", 32)),
    )
    batch["input_ids"] = input_ids
    outputs = _forward_model(model, batch)
    if decoding_config.get("use_nussinov", True) and outputs["pair_logits"] is not None:
        pair_probs = torch.sigmoid(outputs["pair_logits"][0, : len(seq), : len(seq)])
        return nussinov_decode(
            seq,
            pair_probs,
            min_loop_length=int(decoding_config.get("min_loop_length", 3)),
            allow_wobble=bool(decoding_config.get("allow_wobble", True)),
        )
    ids = input_ids[0, struct_positions].tolist()
    return "".join(tokenizer.decode(ids))


@torch.no_grad()
def generate_sequence_invfold(
    model: torch.nn.Module,
    tokenizer: RNAOmniTokenizer,
    struct: str,
    decoding_config: dict,
    device: torch.device | str = "cpu",
) -> str:
    seq = "N" * len(struct)
    batch, seq_positions, _ = _build_inference_batch(tokenizer, "invfold", seq, struct, device=device)
    allowed = [tokenizer.token_id(token) for token in tokenizer.sequence_tokens]
    input_ids = entropy_iterative_unmask(
        model,
        batch,
        seq_positions,
        allowed,
        tokenizer.mask_id,
        num_steps=int(decoding_config.get("num_steps", 32)),
    )
    seq_tokens = tokenizer.decode(input_ids[0, seq_positions].tolist())
    seq_list = list(seq_tokens)
    try:
        for i, j in parse_dot_bracket(struct):
            if not canonical_pair(seq_list[i], seq_list[j], bool(decoding_config.get("allow_wobble", True))):
                seq_list[i], seq_list[j] = "G", "C"
    except ValueError:
        pass
    return "".join(seq_list)
