from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


class RNAFMTeacher:
    """Frozen RNA-FM representation teacher.

    This adapter is intentionally outside ``models.omni`` so RNA-FM stays an
    optional offline teacher and never becomes part of benchmark inference.
    """

    def __init__(
        self,
        model_dir: str | Path = "external",
        checkpoint: str | Path | None = None,
        device: str | torch.device = "cpu",
        dtype: str = "float16",
        dummy: bool = False,
        embedding_dim: int = 640,
    ) -> None:
        self.checkpoint = Path(model_dir)
        self.checkpoint_file = Path(checkpoint) if checkpoint else None
        self.device = torch.device(device)
        self.dtype = dtype
        self.dummy = bool(dummy)
        self.dummy_dim = int(embedding_dim)
        if self.checkpoint_file is not None and not self.checkpoint_file.exists():
            raise FileNotFoundError(f"RNA-FM checkpoint file not found: {self.checkpoint_file}")
        if self.dummy:
            self.vocab = {}
            self.model = None
            return
        self.vocab = self._load_vocab(self.checkpoint)
        self.model = self._load_model(self.checkpoint).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @property
    def embedding_dim(self) -> int:
        if self.dummy:
            return self.dummy_dim
        config_path = self.checkpoint / "config.json" if self.checkpoint.is_dir() else self.checkpoint.with_name("config.json")
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return int(data.get("hidden_size", 0))
        return int(getattr(getattr(self.model, "config", None), "hidden_size", 0))

    def encode_batch(self, seqs: list[str]) -> np.ndarray:
        if self.dummy:
            array = np.stack([self._dummy_embedding(seq) for seq in seqs])
        else:
            array = self.encode(seqs, pool="mean").detach().cpu().numpy()
        return array.astype(np.float16 if self.dtype == "float16" else np.float32)

    def encode(self, sequences: Sequence[str], pool: str = "mean") -> torch.Tensor:
        if self.dummy:
            return torch.tensor(np.stack([self._dummy_embedding(seq) for seq in sequences]), dtype=torch.float32)
        if pool != "mean":
            raise ValueError("RNAFMTeacher currently supports --pool mean only.")
        batch = self._tokenize(sequences)
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
            )
            hidden = outputs.last_hidden_state
            mask = batch["pool_mask"].to(self.device).unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _tokenize(self, sequences: Sequence[str]) -> dict[str, torch.Tensor]:
        cls_id = self.vocab.get("<cls>", self.vocab.get("[CLS]", 1))
        eos_id = self.vocab.get("<eos>", self.vocab.get("[SEP]", 2))
        pad_id = self.vocab.get("<pad>", self.vocab.get("[PAD]", 0))
        unk_id = self.vocab.get("<unk>", self.vocab.get("[UNK]", 3))
        encoded: list[list[int]] = []
        for seq in sequences:
            ids = [cls_id]
            ids.extend(self.vocab.get(base, unk_id) for base in str(seq).upper().replace("T", "U"))
            ids.append(eos_id)
            encoded.append(ids)
        max_len = max(len(ids) for ids in encoded)
        input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
        pool_mask = torch.zeros((len(encoded), max_len), dtype=torch.bool)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
            if len(ids) > 2:
                pool_mask[row, 1 : len(ids) - 1] = True
        return {"input_ids": input_ids, "attention_mask": attention_mask, "pool_mask": pool_mask}

    def _dummy_embedding(self, seq: str) -> np.ndarray:
        values = np.zeros((self.dummy_dim,), dtype="float32")
        state = 2166136261
        for char in str(seq).upper().replace("T", "U"):
            state ^= ord(char)
            state = (state * 16777619) & 0xFFFFFFFF
        rng = np.random.default_rng(state)
        values[:] = rng.standard_normal(self.dummy_dim).astype("float32")
        norm = np.linalg.norm(values)
        return values / max(norm, 1e-6)

    @staticmethod
    def _load_vocab(checkpoint: Path) -> dict[str, int]:
        vocab_path = checkpoint / "vocab.txt" if checkpoint.is_dir() else checkpoint.with_name("vocab.txt")
        if not vocab_path.exists():
            raise FileNotFoundError(f"RNA-FM vocab not found: {vocab_path}")
        return {line.strip(): idx for idx, line in enumerate(vocab_path.read_text(encoding="utf-8").splitlines()) if line.strip()}

    @staticmethod
    def _load_model(checkpoint: Path) -> torch.nn.Module:
        try:
            from transformers import AutoModel

            return AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
        except Exception:
            return RNAFMTeacher._load_bert_compatible_model(checkpoint)

    @staticmethod
    def _load_bert_compatible_model(checkpoint: Path) -> torch.nn.Module:
        from safetensors.torch import load_file
        from transformers import BertConfig, BertModel

        config_path = checkpoint / "config.json"
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        config = BertConfig(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            intermediate_size=int(raw["intermediate_size"]),
            hidden_act=str(raw.get("hidden_act", "gelu")),
            hidden_dropout_prob=float(raw.get("hidden_dropout", 0.1)),
            attention_probs_dropout_prob=float(raw.get("attention_dropout", 0.1)),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            layer_norm_eps=float(raw.get("layer_norm_eps", 1e-12)),
            pad_token_id=int(raw.get("pad_token_id", 0)),
            type_vocab_size=1,
        )
        model = BertModel(config, add_pooling_layer=False)
        state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
        mapped = {}
        for key, value in state.items():
            if not key.startswith("model."):
                continue
            new_key = key.removeprefix("model.")
            new_key = new_key.replace("embeddings.layer_norm.", "embeddings.LayerNorm.")
            new_key = new_key.replace(".attention.layer_norm.", ".attention.output.LayerNorm.")
            new_key = new_key.replace(".layer_norm.", ".output.LayerNorm.")
            mapped[new_key] = value
        model.load_state_dict(mapped, strict=False)
        return model
