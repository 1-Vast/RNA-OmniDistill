"""LLM-guided semantic token definitions for RNA-OmniDistill-LLM.

This module defines the controlled semantic token vocabulary and provides
validation + conversion between LLM JSON output and discrete condition tokens.
It does NOT call any external API — that's handled by scripts/extract_llm_semantics.py.
"""
from __future__ import annotations
from typing import Dict, List, Optional, TypedDict
import json


class LLMSemanticRecord(TypedDict, total=False):
    sequence_id: str
    length_category: str       # short|medium|long|very_long
    gc_level: str              # low|medium|high
    au_level: str              # low|medium|high
    local_complementarity: str # weak|moderate|strong
    palindromic_tendency: str  # low|medium|high
    pairing_density_prior: str # low|medium|high
    stem_loop_tendency: str    # low|medium|high
    hairpin_tendency: str      # low|medium|high
    long_range_pairing_likelihood: str  # low|medium|high
    stability_prior: str       # low|medium|high
    motif_hints: List[str]
    family_hint: str
    reasoning_hops: List[str]
    confidence: float


# ─── Controlled semantic token vocabulary ───
LLM_SEMANTIC_TOKEN_VOCAB: List[str] = [
    # Length
    "<LEN_SHORT>", "<LEN_MEDIUM>", "<LEN_LONG>", "<LEN_VERY_LONG>",
    # GC
    "<GC_LOW>", "<GC_MEDIUM>", "<GC_HIGH>",
    # AU
    "<AU_LOW>", "<AU_MEDIUM>", "<AU_HIGH>",
    # Local complementarity
    "<LOCAL_COMP_WEAK>", "<LOCAL_COMP_MODERATE>", "<LOCAL_COMP_STRONG>",
    # Palindromic
    "<PAL_LOW>", "<PAL_MEDIUM>", "<PAL_HIGH>",
    # Pairing density
    "<PAIR_DENSITY_LOW>", "<PAIR_DENSITY_MEDIUM>", "<PAIR_DENSITY_HIGH>",
    # Stem-loop
    "<STEM_LOOP_LOW>", "<STEM_LOOP_MEDIUM>", "<STEM_LOOP_HIGH>",
    # Hairpin
    "<HAIRPIN_LOW>", "<HAIRPIN_MEDIUM>", "<HAIRPIN_HIGH>",
    # Long-range pairing
    "<LONG_RANGE_LOW>", "<LONG_RANGE_MEDIUM>", "<LONG_RANGE_HIGH>",
    # Stability
    "<STABILITY_LOW>", "<STABILITY_MEDIUM>", "<STABILITY_HIGH>",
    # Motif hints
    "<MOTIF_STEM_LOOP>", "<MOTIF_HAIRPIN>", "<MOTIF_INTERNAL_LOOP>",
    "<MOTIF_BULGE>", "<MOTIF_MULTI_LOOP>", "<MOTIF_UNKNOWN>",
    # Family hints
    "<FAMILY_TRNA_LIKE>", "<FAMILY_RRNA_LIKE>", "<FAMILY_RIBOSWITCH_LIKE>",
    "<FAMILY_MIRNA_LIKE>", "<FAMILY_LNCRNA_LIKE>", "<FAMILY_UNKNOWN>",
    # Confidence
    "<LLM_CONF_LOW>", "<LLM_CONF_MEDIUM>", "<LLM_CONF_HIGH>",
    # Special
    "<SEM_UNKNOWN>",
]

# Valid values for categorical fields
VALID_CATEGORIES: Dict[str, set] = {
    "length_category": {"short", "medium", "long", "very_long"},
    "gc_level": {"low", "medium", "high"},
    "au_level": {"low", "medium", "high"},
    "local_complementarity": {"weak", "moderate", "strong"},
    "palindromic_tendency": {"low", "medium", "high"},
    "pairing_density_prior": {"low", "medium", "high"},
    "stem_loop_tendency": {"low", "medium", "high"},
    "hairpin_tendency": {"low", "medium", "high"},
    "long_range_pairing_likelihood": {"low", "medium", "high"},
    "stability_prior": {"low", "medium", "high"},
}

VALID_MOTIFS = {"stem_loop", "hairpin", "internal_loop", "bulge", "multi_loop", "unknown"}
VALID_FAMILIES = {"tRNA_like", "rRNA_like", "riboswitch_like", "miRNA_like", "lncRNA_like", "unknown"}

# ─── Validation ───
def validate_llm_semantic_record(record: dict, strict: bool = False) -> List[str]:
    """Validate a semantic record dict. Returns list of error messages (empty = valid)."""
    errors = []
    
    # Check required fields
    for field in ["length_category", "gc_level", "au_level", "confidence"]:
        if field not in record:
            errors.append(f"Missing required field: {field}")
    
    # Check categorical values
    for field, valid_set in VALID_CATEGORIES.items():
        if field in record and record[field] not in valid_set:
            errors.append(f"Invalid {field}: {record[field]!r}, expected one of {valid_set}")
    
    # Check motif_hints
    if "motif_hints" in record:
        hints = record["motif_hints"]
        if not isinstance(hints, list):
            errors.append("motif_hints must be a list")
        elif len(hints) > 3:
            errors.append(f"motif_hints has {len(hints)} items, max 3")
        else:
            for h in hints:
                if h not in VALID_MOTIFS:
                    errors.append(f"Invalid motif_hint: {h!r}")
    
    # Check family_hint
    if "family_hint" in record and record["family_hint"] not in VALID_FAMILIES:
        errors.append(f"Invalid family_hint: {record['family_hint']!r}")
    
    # Check reasoning_hops
    if "reasoning_hops" in record:
        hops = record["reasoning_hops"]
        if not isinstance(hops, list):
            errors.append("reasoning_hops must be a list")
        elif len(hops) != 3:
            errors.append(f"reasoning_hops must have exactly 3 items, got {len(hops)}")
    
    # Check confidence
    if "confidence" in record:
        conf = record["confidence"]
        if not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
            errors.append(f"confidence must be float in [0,1], got {conf!r}")
    
    # Structure leakage check
    if "struct" in record:
        errors.append("Leakage: record contains 'struct' field (dot-bracket)")
    if "pairs" in record:
        errors.append("Leakage: record contains 'pairs' field (base-pair positions)")
    if "dot_bracket" in record:
        errors.append("Leakage: record contains 'dot_bracket' field")
    
    if strict and errors:
        raise ValueError(f"Semantic record validation failed: {'; '.join(errors)}")
    
    return errors

# ─── Token conversion ───
def confidence_to_token(confidence: float) -> str:
    if confidence >= 0.7:
        return "<LLM_CONF_HIGH>"
    elif confidence >= 0.3:
        return "<LLM_CONF_MEDIUM>"
    return "<LLM_CONF_LOW>"

# Field prefix mapping
FIELD_PREFIX = {
    "length_category": "LEN",
    "gc_level": "GC",
    "au_level": "AU",
    "local_complementarity": "LOCAL_COMP",
    "palindromic_tendency": "PAL",
    "pairing_density_prior": "PAIR_DENSITY",
    "stem_loop_tendency": "STEM_LOOP",
    "hairpin_tendency": "HAIRPIN",
    "long_range_pairing_likelihood": "LONG_RANGE",
    "stability_prior": "STABILITY",
}

MOTIF_PREFIX = {"stem_loop": "MOTIF_STEM_LOOP", "hairpin": "MOTIF_HAIRPIN",
                "internal_loop": "MOTIF_INTERNAL_LOOP", "bulge": "MOTIF_BULGE",
                "multi_loop": "MOTIF_MULTI_LOOP", "unknown": "MOTIF_UNKNOWN"}

FAMILY_PREFIX = {"tRNA_like": "FAMILY_TRNA_LIKE", "rRNA_like": "FAMILY_RRNA_LIKE",
                 "riboswitch_like": "FAMILY_RIBOSWITCH_LIKE", "miRNA_like": "FAMILY_MIRNA_LIKE",
                 "lncRNA_like": "FAMILY_LNCRNA_LIKE", "unknown": "FAMILY_UNKNOWN"}


def semantic_record_to_tokens(record: dict) -> List[str]:
    """Convert a validated semantic record into a list of discrete condition tokens."""
    tokens = []
    
    for field, prefix in FIELD_PREFIX.items():
        if field in record:
            value = str(record[field]).upper()
            token = f"<{prefix}_{value}>"
            if token in LLM_SEMANTIC_TOKEN_VOCAB:
                tokens.append(token)
    
    # Motif hints
    if "motif_hints" in record:
        for hint in record["motif_hints"]:
            token = f"<{MOTIF_PREFIX.get(hint, 'MOTIF_UNKNOWN')}>"
            if token in LLM_SEMANTIC_TOKEN_VOCAB:
                tokens.append(token)
    else:
        tokens.append("<MOTIF_UNKNOWN>")
    
    # Family hint
    if "family_hint" in record:
        token = f"<{FAMILY_PREFIX.get(record['family_hint'], 'FAMILY_UNKNOWN')}>"
        if token in LLM_SEMANTIC_TOKEN_VOCAB:
            tokens.append(token)
    else:
        tokens.append("<FAMILY_UNKNOWN>")
    
    # Confidence
    if "confidence" in record:
        tokens.append(confidence_to_token(record["confidence"]))
    else:
        tokens.append("<LLM_CONF_LOW>")
    
    return tokens


# ─── Fake record generator (for testing without API) ───
import random as _random

def make_fake_semantic_record(sequence: str, seed: int = 42) -> dict:
    """Generate a deterministic fake semantic record for testing. No API call."""
    rng = _random.Random(hash(sequence) ^ seed)
    length = len(sequence)
    gc_count = sum(1 for c in sequence.upper() if c in "GC")
    gc_ratio = gc_count / max(1, length)
    
    length_cat = "short" if length < 50 else ("medium" if length < 150 else ("long" if length < 300 else "very_long"))
    gc_level = "high" if gc_ratio > 0.55 else ("medium" if gc_ratio > 0.4 else "low")
    au_level = "high" if (1 - gc_ratio) > 0.55 else ("medium" if (1 - gc_ratio) > 0.4 else "low")
    
    return {
        "sequence_id": f"fake_{abs(hash(sequence)) % 1000000:06d}",
        "length_category": length_cat,
        "gc_level": gc_level,
        "au_level": au_level,
        "local_complementarity": rng.choice(["weak", "moderate", "strong"]),
        "palindromic_tendency": rng.choice(["low", "medium", "high"]),
        "pairing_density_prior": rng.choice(["low", "medium", "high"]),
        "stem_loop_tendency": rng.choice(["low", "medium", "high"]),
        "hairpin_tendency": rng.choice(["low", "medium", "high"]),
        "long_range_pairing_likelihood": rng.choice(["low", "medium", "high"]),
        "stability_prior": rng.choice(["low", "medium", "high"]),
        "motif_hints": [rng.choice(["stem_loop", "hairpin", "internal_loop", "bulge", "multi_loop", "unknown"])],
        "family_hint": rng.choice(["tRNA_like", "rRNA_like", "riboswitch_like", "miRNA_like", "unknown"]),
        "reasoning_hops": [
            f"Sequence is {length_cat} with {gc_level} GC content and {au_level} AU richness.",
            f"Local complementarity is {rng.choice(['weak','moderate','strong'])}, suggesting moderate structural tendency.",
            f"Confidence is limited without experimental data; family assignment is uncertain.",
        ],
        "confidence": round(rng.uniform(0.2, 0.8), 2),
    }
