from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.teacher.rnafm_teacher import RNAFMTeacher


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            seq = row.get("seq", row.get("sequence"))
            if not seq:
                raise ValueError(f"{path}:{line_no} missing seq/sequence")
            row["seq"] = str(seq).upper().replace("T", "U")
            rows.append(row)
    return rows


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen RNA-FM mean-pooled embeddings for sequence-only JSONL.")
    parser.add_argument("--input", required=True, help="Input JSONL with seq or sequence field.")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL with teacher_embedding path field.")
    parser.add_argument("--output_npy", required=True, help="Output [N, D] embedding matrix.")
    parser.add_argument("--model_dir", default="external", help="RNA-FM model directory.")
    parser.add_argument("--checkpoint", default=None, help="Optional RNA-FM checkpoint file.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pool", default="mean", choices=["mean"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--embedding_dim", type=int, default=640)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--min_length", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = RNAFMTeacher(
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device=device,
        dtype=args.dtype,
        dummy=args.dummy,
        embedding_dim=args.embedding_dim,
    )
    rows = [
        row for row in read_rows(Path(args.input))
        if len(row["seq"]) >= int(args.min_length)
        and (args.max_length is None or len(row["seq"]) <= int(args.max_length))
    ]
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    output_npy = Path(args.output_npy)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    if output_npy.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {output_npy}")
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {output_path}")

    dtype = np.float16 if args.dtype == "float16" else np.float32
    matrix = np.zeros((len(rows), teacher.embedding_dim), dtype=dtype)
    for start in range(0, len(rows), int(args.batch_size)):
        batch = rows[start : start + int(args.batch_size)]
        matrix[start : start + len(batch)] = teacher.encode_batch([row["seq"] for row in batch]).astype(dtype)
    np.save(output_npy, matrix)
    with output_path.open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows):
            out.write(json.dumps({
                "id": str(row.get("id", f"row_{index:06d}")),
                "seq": row["seq"],
                "teacher_embedding_file": str(output_npy),
                "teacher_embedding_index": index,
                "teacher_embedding_dim": int(matrix.shape[1]),
                "teacher": "rnafm_frozen_mean_dummy" if args.dummy else "rnafm_frozen_mean",
            }, ensure_ascii=False) + "\n")
    print(f"wrote {output_path} and {output_npy} shape={tuple(matrix.shape)} dtype={matrix.dtype}")


if __name__ == "__main__":
    main()
