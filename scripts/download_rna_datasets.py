from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


DATASETS = {
    "archiveii": {
        "url": "https://rna.urmc.rochester.edu/pub/archiveII.tar.gz",
        "filename": "archiveII.tar.gz",
        "note": "Mathews lab ArchiveII RNA secondary structure benchmark.",
    },
    "rfam_seed": {
        "url": "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz",
        "filename": "Rfam.seed.gz",
        "note": "Rfam CURRENT seed alignments.",
    },
    "bprna": {
        "url": None,
        "filename": "download_instruction.txt",
        "note": "bpRNA data should be downloaded from the official bpRNA downloads page.",
        "instruction": "Download bpRNA from https://bprna.cgrb.oregonstate.edu/download.php and place files in this directory.",
    },
    "bprna90": {
        "url": None,
        "filename": "download_instruction.txt",
        "note": "bpRNA-90 should be downloaded from the official bpRNA downloads page if a direct mirror is unavailable.",
        "instruction": "Download bpRNA-90 from https://bprna.cgrb.oregonstate.edu/download.php and place files in this directory.",
    },
    "rnastralign_hf_optional": {
        "url": None,
        "filename": "download_instruction.txt",
        "note": "Optional Hugging Face based RNAStrAlign source; no hard dependency is required.",
        "instruction": "Install datasets or huggingface_hub, or manually place RNAStrAlign files in this directory.",
    },
}


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"download failed for {url}: {exc}") from exc


def extract(path: Path, out_dir: Path) -> list[str]:
    extracted: list[str] = []
    try:
        if path.suffixes[-2:] == [".tar", ".gz"] or path.name.endswith(".tgz"):
            with tarfile.open(path, "r:gz") as archive:
                archive.extractall(out_dir)
                extracted = archive.getnames()
        elif path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                archive.extractall(out_dir)
                extracted = archive.namelist()
        elif path.suffix == ".gz":
            output = out_dir / path.with_suffix("").name
            if not output.exists():
                with gzip.open(path, "rb") as src, output.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            extracted = [str(output.name)]
    except (tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
        raise RuntimeError(f"extract failed for {path}: {exc}") from exc
    return extracted


def write_instruction(out_dir: Path, dataset: str, message: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "download_instruction.txt"
    path.write_text(message + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download common RNA secondary structure datasets.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--no_extract", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    info = DATASETS[args.dataset]
    report = {
        "dataset": args.dataset,
        "out": str(out_dir),
        "status": "ok",
        "downloaded": False,
        "extracted": [],
        "message": info["note"],
    }

    try:
        if args.dataset == "rnastralign_hf_optional":
            try:
                import datasets  # type: ignore  # noqa: F401
            except ImportError:
                path = write_instruction(out_dir, args.dataset, info["instruction"])
                report.update({"status": "instruction_only", "instruction": str(path)})
                return

        url = info.get("url")
        if not url:
            path = write_instruction(out_dir, args.dataset, info["instruction"])
            report.update({"status": "instruction_only", "instruction": str(path)})
            return

        target = out_dir / str(info["filename"])
        if target.exists() and target.stat().st_size > 0:
            report["message"] = f"using existing file: {target}"
        else:
            print(f"downloading {url} -> {target}")
            download(str(url), target)
            report["downloaded"] = True
        if not args.no_extract:
            report["extracted"] = extract(target, out_dir)
    except RuntimeError as exc:
        report["status"] = "failed"
        report["message"] = str(exc)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "download_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"Error: {exc}. See {out_dir / 'download_report.json'}")
    finally:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "download_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"download report -> {out_dir / 'download_report.json'}")


if __name__ == "__main__":
    main()

