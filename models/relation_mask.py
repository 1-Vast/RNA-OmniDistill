"""Multi-Scale Masked Pair-Relation Modeling (MS-MPRM).

Relation-aware loss mask sampler for RNA secondary structure prediction.
Does NOT add model parameters. Changes supervision distribution only.
"""
from __future__ import annotations
import torch
import random

CANONICAL_PAIRS = {("A","U"), ("U","A"), ("G","C"), ("C","G"), ("G","U"), ("U","G")}


class MultiScaleRelationMaskSampler:
    """Multi-scale relation loss mask for pair-relation field supervision.

    Three scales:
      1. Global: long-range valid entries (|i-j| >= threshold)
      2. Stem-local: anti-diagonal span around positive pairs
      3. Canonical hard negatives: valid canonical-compatible but label=0 entries
    """

    def __init__(
        self,
        mode: str = "multiscale",
        global_ratio: float = 0.25,
        stem_span_ratio: float = 0.35,
        hard_negative_ratio: float = 0.40,
        total_ratio: float = 0.25,
        stem_span_len: int = 4,
        min_pair_distance: int = 4,
        long_range_threshold: int = 64,
        canonical_hard_negative: bool = True,
        seed: int = 42,
    ):
        self.mode = mode
        self.global_ratio = global_ratio
        self.stem_span_ratio = stem_span_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.total_ratio = total_ratio
        self.stem_span_len = stem_span_len
        self.min_pair_distance = min_pair_distance
        self.long_range_threshold = long_range_threshold
        self.canonical_hard_negative = canonical_hard_negative
        self.rng = random.Random(seed)

    def sample(
        self,
        seq: str | list,
        pair_labels: torch.Tensor,      # [L, L] float
        valid_pair_mask: torch.Tensor,   # [L, L] bool (True = valid position pair)
    ) -> tuple[torch.Tensor, dict]:
        """Sample a relation_loss_mask from multi-scale components.

        Returns:
            mask: BoolTensor [L, L], symmetric, where True = included in pair loss
            stats: dict with component counts
        """
        L = pair_labels.size(0)
        device = pair_labels.device

        if self.mode == "random":
            # Simple random subsample of valid entries
            total_valid = int(valid_pair_mask.sum().item())
            target = max(1, int(total_valid * self.total_ratio))
            valid_indices = torch.where(valid_pair_mask)
            perm = torch.randperm(len(valid_indices[0]), device=device)[:target]
            mask = torch.zeros(L, L, dtype=torch.bool, device=device)
            mask[valid_indices[0][perm], valid_indices[1][perm]] = True
            mask = mask | mask.T  # ensure symmetry
            return mask, {"total": mask.sum().item(), "mode": "random"}

        # === Multiscale sampling ===
        # Build base valid mask (upper triangular, min distance, not diagonal)
        idx = torch.arange(L, device=device)
        upper = idx.unsqueeze(0) < idx.unsqueeze(1)
        dist_ok = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= self.min_pair_distance
        base_valid = valid_pair_mask & upper & dist_ok

        pos_mask = (pair_labels > 0.5) & base_valid
        neg_mask = (~pos_mask) & base_valid

        total_target = max(1, int(base_valid.sum().item() * self.total_ratio))

        # --- A. Global long-range relation mask ---
        global_valid = base_valid & ((idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= self.long_range_threshold)
        global_target = max(1, int(total_target * self.global_ratio))
        global_indices = torch.where(global_valid)
        if len(global_indices[0]) > 0:
            perm = torch.randperm(len(global_indices[0]), device=device)[:global_target]
            gi, gj = global_indices[0][perm], global_indices[1][perm]
        else:
            gi, gj = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        # --- B. Stem-local relation span mask ---
        stem_target = max(1, int(total_target * self.stem_span_ratio))
        pos_indices = torch.where(pos_mask)
        si_list, sj_list = [], []

        if len(pos_indices[0]) > 0 and stem_target > 0:
            n_pos = len(pos_indices[0])
            per_pair = max(1, stem_target // n_pos)
            for k in range(min(n_pos, stem_target)):
                i, j = int(pos_indices[0][k]), int(pos_indices[1][k])
                # Anti-diagonal span: (i+d, j-d) for d in [0, span_len)
                for d in range(self.stem_span_len):
                    ni, nj = i + d, j - d
                    if 0 <= ni < nj < L and base_valid[ni, nj]:
                        si_list.append(ni)
                        sj_list.append(nj)
                        if len(si_list) >= stem_target:
                            break
                if len(si_list) >= stem_target:
                    break

        if si_list:
            si = torch.tensor(si_list, dtype=torch.long, device=device)
            sj = torch.tensor(sj_list, dtype=torch.long, device=device)
        else:
            si = torch.tensor([], dtype=torch.long, device=device)
            sj = torch.tensor([], dtype=torch.long, device=device)

        # --- C. Canonical-compatible hard negative mask ---
        hn_target = total_target - gi.size(0) - si.size(0)
        hn_target = max(1, hn_target)

        if self.canonical_hard_negative and isinstance(seq, str):
            # Find positions with canonical-compatible bases
            seq_upper = seq.upper().replace("T", "U")
            canon_pairs_mask = torch.zeros(L, L, dtype=torch.bool, device=device)
            for i in range(L):
                for j in range(i + 1, L):
                    if (seq_upper[i], seq_upper[j]) in CANONICAL_PAIRS:
                        canon_pairs_mask[i, j] = True
            hn_valid = neg_mask & canon_pairs_mask
        else:
            hn_valid = neg_mask

        hn_indices = torch.where(hn_valid)
        if len(hn_indices[0]) > 0:
            hn_target = min(hn_target, len(hn_indices[0]))
            perm = torch.randperm(len(hn_indices[0]), device=device)[:hn_target]
            hi, hj = hn_indices[0][perm], hn_indices[1][perm]
        else:
            hi, hj = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        # --- Build final symmetric mask ---
        mask = torch.zeros(L, L, dtype=torch.bool, device=device)
        for ii, jj in [(gi, gj), (si, sj), (hi, hj)]:
            if ii.size(0) > 0:
                mask[ii, jj] = True
        mask = mask | mask.T

        # Ensure at least some positive pairs
        if not (mask & pos_mask).any():
            # Fallback: include some original pair_mask entries
            if pos_indices[0].size(0) > 0:
                k = min(10, pos_indices[0].size(0))
                mask[pos_indices[0][:k], pos_indices[1][:k]] = True
                mask[pos_indices[1][:k], pos_indices[0][:k]] = True

        stats = {
            "global_count": int(gi.size(0)),
            "stem_count": int(si.size(0)),
            "hard_negative_count": int(hi.size(0)),
            "positive_count": int((mask & pos_mask).sum().item()),
            "negative_count": int((mask & (~pos_mask)).sum().item()),
            "total_count": int(mask.sum().item()),
            "mode": self.mode,
        }
        return mask, stats
