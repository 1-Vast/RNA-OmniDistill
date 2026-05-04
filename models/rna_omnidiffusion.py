from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNAOmniDiffusion(nn.Module):
    """Minimal masked discrete diffusion model for RNA sequence/structure tasks."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        num_segments: int = 4,
        num_tasks: int = 5,
        use_pair_head: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.use_pair_head = use_pair_head
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embedding = nn.Embedding(num_segments, hidden_size)
        self.task_embedding = nn.Embedding(num_tasks, hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.sequence_head = nn.Linear(hidden_size, vocab_size)
        self.structure_head = nn.Linear(hidden_size, vocab_size)
        self.general_head = nn.Linear(hidden_size, vocab_size)
        if self.use_pair_head:
            self.pair_left = nn.Linear(hidden_size, hidden_size, bias=False)
            self.pair_right = nn.Linear(hidden_size, hidden_size, bias=False)
            self.pair_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        task_ids: torch.Tensor,
        time_steps: torch.Tensor,
        seq_positions: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | None]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Input token length {seq_len} exceeds max_position_embeddings "
                f"{self.position_embedding.num_embeddings}."
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        time_emb = self.time_mlp(time_steps.float().view(batch_size, 1)).unsqueeze(1)
        task_emb = self.task_embedding(task_ids).unsqueeze(1)
        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.segment_embedding(segment_ids)
            + task_emb
            + time_emb
        )
        padding_mask = attention_mask == 0
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        encoded = self.norm(encoded)

        general_logits = self.general_head(encoded)
        sequence_logits = self.sequence_head(encoded)
        structure_logits = self.structure_head(encoded)
        token_logits = general_logits.clone()
        token_logits = torch.where((segment_ids == 1).unsqueeze(-1), sequence_logits, token_logits)
        token_logits = torch.where((segment_ids == 2).unsqueeze(-1), structure_logits, token_logits)

        pair_logits = None
        if self.use_pair_head and seq_positions is not None:
            pair_logits = self._pair_logits(encoded, seq_positions)

        return {
            "hidden_states": encoded,
            "token_logits": token_logits,
            "sequence_logits": sequence_logits,
            "structure_logits": structure_logits,
            "general_logits": general_logits,
            "pair_logits": pair_logits,
        }

    def _pair_logits(self, hidden: torch.Tensor, seq_positions: torch.Tensor) -> torch.Tensor:
        gather_positions = seq_positions.clamp_min(0)
        expanded = gather_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        seq_hidden = hidden.gather(1, expanded)
        valid = (seq_positions >= 0).float().unsqueeze(-1)
        seq_hidden = seq_hidden * valid
        left = self.pair_left(seq_hidden)
        right = self.pair_right(seq_hidden)
        logits = torch.matmul(left, right.transpose(1, 2)) / (self.hidden_size ** 0.5)
        logits = logits + self.pair_bias
        logits = 0.5 * (logits + logits.transpose(1, 2))
        invalid = valid.squeeze(-1) == 0
        logits = logits.masked_fill(invalid.unsqueeze(1), -20.0)
        logits = logits.masked_fill(invalid.unsqueeze(2), -20.0)
        return logits


def compute_omni_loss(
    outputs: Dict[str, torch.Tensor | None],
    batch: dict,
    lambda_pair: float = 0.5,
    lambda_seq: float = 1.0,
    lambda_struct: float = 1.0,
    token_id_weights: torch.Tensor | None = None,
    pair_pos_weight: torch.Tensor | float | None = None,
    use_pair_loss: bool = True,
) -> Dict[str, torch.Tensor]:
    token_logits = outputs["token_logits"]
    labels = batch["labels"].to(token_logits.device)
    flat_logits = token_logits.view(-1, token_logits.size(-1))
    flat_labels = labels.view(-1)
    per_token = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="none")
    supervised = flat_labels != -100
    flat_segments = batch["segment_ids"].to(token_logits.device).view(-1)
    segment_weights = torch.zeros_like(per_token)
    segment_weights = torch.where(flat_segments == 1, per_token.new_tensor(float(lambda_seq)), segment_weights)
    segment_weights = torch.where(flat_segments == 2, per_token.new_tensor(float(lambda_struct)), segment_weights)
    if token_id_weights is not None:
        id_weights = token_id_weights.to(token_logits.device).float()
        label_weights = torch.ones_like(per_token)
        safe_labels = flat_labels.clamp_min(0)
        label_weights[supervised] = id_weights[safe_labels[supervised]]
        segment_weights = segment_weights * label_weights
    weighted = per_token * segment_weights
    denom = segment_weights[supervised].sum().clamp_min(1.0)
    token_loss = weighted[supervised].sum() / denom

    pair_logits = outputs.get("pair_logits")
    pair_loss = token_loss.new_zeros(())
    pair_mask = batch["pair_mask"].to(token_logits.device)
    positive_count = int(batch.get("pair_positive_counts", pair_mask.new_zeros(1)).sum().item())
    if use_pair_loss and pair_logits is not None and pair_mask.any() and positive_count > 0:
        pair_labels = batch["pair_labels"].to(token_logits.device).float()
        pos_weight = None
        if pair_pos_weight is not None:
            pos_weight = torch.as_tensor(pair_pos_weight, dtype=torch.float32, device=token_logits.device)
        pair_loss = F.binary_cross_entropy_with_logits(
            pair_logits.float()[pair_mask],
            pair_labels[pair_mask],
            pos_weight=pos_weight,
        )

    total = token_loss + float(lambda_pair) * pair_loss
    return {"loss": total, "token_loss": token_loss.detach(), "pair_loss": pair_loss.detach()}
