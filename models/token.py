from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence


class RNAOmniTokenizer:
    sequence_tokens = ["A", "U", "G", "C", "N"]
    structure_tokens = [".", "(", ")", "[", "]", "{", "}"]
    special_tokens = [
        "<PAD>",
        "<MASK>",
        "<BOS>",
        "<EOS>",
        "<SEQ>",
        "</SEQ>",
        "<STRUCT>",
        "</STRUCT>",
        "<MOTIF>",
        "</MOTIF>",
        "<FAMILY>",
        "</FAMILY>",
    ]
    task_tokens = [
        "<TASK_SEQ2STRUCT>",
        "<TASK_INVFOLD>",
        "<TASK_INPAINT>",
        "<TASK_DENOISE>",
        "<TASK_MOTIF_CONTROL>",
    ]
    motif_tokens = ["<STEM>", "<HAIRPIN>", "<BULGE>", "<INTERNAL_LOOP>", "<MULTI_LOOP>"]

    # Semantic condition tokens (optional, only when semantic.enabled=True)
    semantic_condition_tokens = [
        "<FAMILY_TYPE=tRNA>", "<FAMILY_TYPE=riboswitch>", "<FAMILY_TYPE=ribozyme>",
        "<FAMILY_TYPE=miRNA>", "<FAMILY_TYPE=snRNA>", "<FAMILY_TYPE=snoRNA>",
        "<FAMILY_TYPE=rRNA>", "<FAMILY_TYPE=tmRNA>", "<FAMILY_TYPE=lncRNA>",
        "<FAMILY_TYPE=unknown>",
        "<MOTIF=hairpin>", "<MOTIF=stem_loop>", "<MOTIF=bulge>",
        "<MOTIF=internal_loop>", "<MOTIF=multiloop>", "<MOTIF=pseudoknot>",
        "<MOTIF=unknown>",
        "<BIAS=stem_rich>", "<BIAS=loop_rich>", "<BIAS=cloverleaf>",
        "<BIAS=long_range_pairing>", "<BIAS=balanced>", "<BIAS=unknown>",
        "<HINT=STEM_RICH>", "<HINT=CONSERVED_LOOP>", "<HINT=LONG_RANGE>",
        "<HINT=COMPACT>", "<HINT=UNKNOWN>",
        "<MASK_REGION=stem>", "<MASK_REGION=hairpin_loop>", "<MASK_REGION=internal_loop>",
        "<MASK_REGION=bulge>", "<MASK_REGION=multiloop>", "<MASK_REGION=random>",
        "<REPAIR_TARGET=conserved_loop>", "<REPAIR_TARGET=stem>", "<REPAIR_TARGET=motif>",
        "<REPAIR_TARGET=unknown>",
        "<CONSTRAINT=preserve_stem>", "<CONSTRAINT=preserve_loop>",
        "<CONSTRAINT=maintain_pairing>", "<CONSTRAINT=UNKNOWN>",
    ]
    family_other = "<FAMILY_OTHER>"

    task_name_to_token = {
        "seq2struct": "<TASK_SEQ2STRUCT>",
        "invfold": "<TASK_INVFOLD>",
        "inpaint": "<TASK_INPAINT>",
        "denoise": "<TASK_DENOISE>",
        "motif_control": "<TASK_MOTIF_CONTROL>",
    }

    def __init__(self, families: Iterable[str] | None = None) -> None:
        base_tokens = (
            self.special_tokens
            + self.task_tokens
            + self.motif_tokens
            + self.semantic_condition_tokens
            + [self.family_other]
            + self.sequence_tokens
            + self.structure_tokens
        )
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        for token in base_tokens:
            self.add_token(token)
        self.family_to_token: Dict[str, str] = {}
        for family in sorted(set(f for f in (families or []) if f)):
            self.add_family(family)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def mask_id(self) -> int:
        return self.token_to_id["<MASK>"]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def task_to_id(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.task_name_to_token)}

    def add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)
        return self.token_to_id[token]

    def add_family(self, family: str) -> str:
        token = self._format_family_token(family)
        self.add_token(token)
        self.family_to_token[family] = token
        return token

    @classmethod
    def from_samples(cls, samples: Sequence[dict]) -> "RNAOmniTokenizer":
        return cls(families=[sample.get("family", "") for sample in samples if sample.get("family")])

    def _format_family_token(self, family: str | None) -> str:
        if not family:
            return self.family_other
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(family)).strip("_")
        return f"<FAMILY_{safe}>" if safe else self.family_other

    def family_token(self, family: str | None, add_if_missing: bool = False) -> str:
        if family in self.family_to_token:
            return self.family_to_token[str(family)]
        token = self._format_family_token(family)
        if add_if_missing:
            self.add_token(token)
            self.family_to_token[str(family)] = token
        return token if token in self.token_to_id else self.family_other

    def motif_token(self, motif_type: str | None) -> str:
        if not motif_type:
            return "<MULTI_LOOP>"
        token = f"<{str(motif_type).upper()}>"
        return token if token in self.token_to_id else "<MULTI_LOOP>"

    def task_token(self, task_name: str) -> str:
        try:
            return self.task_name_to_token[task_name]
        except KeyError as exc:
            known = ", ".join(sorted(self.task_name_to_token))
            raise KeyError(f"Unknown task {task_name!r}. Known tasks: {known}") from exc

    def encode(self, tokens: Sequence[str]) -> List[int]:
        ids = []
        for token in tokens:
            if token not in self.token_to_id:
                raise KeyError(f"Token {token!r} is not in the RNAOmniTokenizer vocabulary.")
            ids.append(self.token_to_id[token])
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> List[str]:
        tokens = []
        for idx in ids:
            token = self.id_to_token[int(idx)]
            if skip_special and token.startswith("<") and token.endswith(">"):
                continue
            tokens.append(token)
        return tokens

    def token_id(self, token: str) -> int:
        return self.token_to_id[token]

    def semantic_token(self, prefix: str, value: str, default: str = "UNKNOWN") -> str:
        token = f"<{prefix}={value}>"
        return token if token in self.token_to_id else f"<{prefix}={default}>"

    def to_dict(self) -> dict:
        return {
            "id_to_token": self.id_to_token,
            "family_to_token": self.family_to_token,
        }

    @classmethod
    def from_dict(cls, state: dict) -> "RNAOmniTokenizer":
        tokenizer = cls()
        tokenizer.token_to_id = {token: idx for idx, token in enumerate(state["id_to_token"])}
        tokenizer.id_to_token = list(state["id_to_token"])
        tokenizer.family_to_token = dict(state.get("family_to_token", {}))
        return tokenizer

