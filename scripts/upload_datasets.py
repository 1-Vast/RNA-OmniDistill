"""
Upload raw datasets from local to remote server via SFTP.

Strategy:
  - Upload compressed archives (tar.gz, .gz) where available, extract on remote.
  - Upload individual files for datasets without compressed archives (bpRNA).
  - Skip datasets already present on remote (unless --force).

Security:
  - Never hardcodes passwords. Password is read from stdin prompt or
    the SSH_PASSWORD environment variable (env-var usage is discouraged).
  - Default mode is dry-run: prints the plan without executing.
  - Use --execute to perform the actual transfer.
  - Excludes .env, outputs/, checkpoints/, .git/, and other sensitive paths.

Usage:
  python scripts/upload_datasets.py --dry-run
  python scripts/upload_datasets.py --host connect.nmb1.seetacloud.com --port 49018 --user root --remote-dir /root/RNA-OmniDiffusion/dataset/raw --dry-run
  python scripts/upload_datasets.py --host connect.nmb1.seetacloud.com --port 49018 --user root --remote-dir /root/RNA-OmniDiffusion/dataset/raw --execute

Password is entered manually in terminal; never save in code, docs, .env, or Agent memory.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── Sensitive paths excluded from upload ───────────────────

EXCLUDE_PATTERNS: set[str] = {
    ".env",
    ".git",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "outputs",
    "outputs/",
    "checkpoints",
    "checkpoints/",
    "*.pt",
    "*.pth",
    "*.ckpt",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.log",
    "manifest.json",
    "README_DOWNLOAD.txt",
    ".gitkeep",
}

# ── Upload plan ───────────────────────────────────────────

UPLOAD_PLAN: dict[str, dict] = {
    "ArchiveII": {
        "source": "archiveII.tar.gz",
        "extract": True,
        "extract_cmd": "tar -xzf {archive} -C {dest_dir} 2>&1",
    },
    "RNAStralign": {
        "source": "RNAStralign.tar.gz",
        "extract": True,
        "extract_cmd": "tar -xzf {archive} -C {dest_dir} 2>&1",
    },
    "Rfam": {
        "files": [
            "Rfam.seed.gz",
            "Rfam.full_region.gz",
            "Rfam.cm.gz",
            "Rfam.tar.gz",
        ],
        "extract": False,
    },
    "bpRNA": {
        "mode": "rsync_dir",
        "extract": False,
    },
    "CRW_tRNA": {
        "mode": "rsync_dir",
        "extract": False,
    },
    "RNA3DB": {
        "mode": "rsync_dir",
        "extract": False,
    },
}


# ── Helpers ────────────────────────────────────────────────

def _find_local_dir(name: str, local_raw: Path) -> Path | None:
    """Find a dataset directory by canonical name, case-insensitive."""
    direct = local_raw / name
    if direct.exists():
        return direct
    name_lower = name.lower()
    for entry in local_raw.iterdir():
        if entry.is_dir() and entry.name.lower() == name_lower:
            return entry
    return None


def _should_skip(filename: str) -> bool:
    """Check if a file/dir name matches the exclude list."""
    for pat in EXCLUDE_PATTERNS:
        if pat.endswith("/"):
            if filename == pat[:-1]:
                return True
        elif pat.startswith("*."):
            if filename.endswith(pat[1:]):
                return True
        else:
            if filename == pat:
                return True
    return False


def _file_items(local_dir: Path) -> list[Path]:
    """Return uploadable files (excluding sensitive paths)."""
    items: list[Path] = []
    for item in sorted(local_dir.iterdir()):
        if _should_skip(item.name):
            continue
        if item.is_file():
            items.append(item)
        elif item.is_dir():
            for f in sorted(item.rglob("*")):
                if f.is_file() and not any(_should_skip(p.name) for p in f.parents if p != local_dir):
                    items.append(f)
    return items


def _print_dry_run_plan(selected: list[str], local_raw: Path, remote_raw: str,
                        host: str, port: int, user: str) -> None:
    """Print the upload plan without connecting."""
    print("=" * 60)
    print("DRY-RUN UPLOAD PLAN")
    print("=" * 60)
    print(f"  Local raw:   {local_raw}")
    print(f"  Remote:       {user}@{host}:{port}:{remote_raw}")
    print(f"  Datasets:     {', '.join(selected)}")
    print(f"  Total size:   calculating...")
    total_size = 0
    for ds in selected:
        local_dir = _find_local_dir(ds, local_raw)
        if local_dir is None:
            print(f"  [{ds}] NOT FOUND locally")
            continue
        plan = UPLOAD_PLAN.get(ds, {})
        size_mb = 0
        file_count = 0
        plan_desc = ""
        if "source" in plan and plan.get("extract"):
            archive = local_dir / plan["source"]
            if archive.exists():
                size_mb = archive.stat().st_size / (1024 * 1024)
                file_count = 1
                plan_desc = f"upload {plan['source']} then extract"
        elif "files" in plan:
            for fname in plan["files"]:
                fpath = local_dir / fname
                if fpath.exists():
                    size_mb += fpath.stat().st_size / (1024 * 1024)
                    file_count += 1
            plan_desc = f"upload {file_count} file(s)"
        elif plan.get("mode") == "rsync_dir":
            items = _file_items(local_dir)
            for f in items:
                size_mb += f.stat().st_size / (1024 * 1024)
            file_count = len(items)
            plan_desc = f"upload {file_count} file(s) (rsync_dir)"
        total_size += size_mb
        print(f"  [{ds}] {plan_desc}  ({size_mb:.1f} MB)")
    print(f"  ----------------------------------------")
    print(f"  TOTAL: {total_size:.1f} MB across {len(selected)} dataset(s)")
    print()
    print("Run with --execute to perform the actual transfer.")
    print("Password will be prompted in terminal; never saved.")


# ── SFTP operations (only when --execute) ──────────────────

def _progress_callback(label: str):
    last = [0, time.time()]

    def cb(transferred: int, total_bytes: int):
        pct = transferred / total_bytes * 100 if total_bytes else 0
        now = time.time()
        if pct >= last[0] + 10 or transferred == total_bytes or (now - last[1]) > 5:
            mb = transferred / (1024 * 1024)
            total_mb = total_bytes / (1024 * 1024) if total_bytes else 0
            bar = "#" * (int(pct) // 5) + "." * (20 - int(pct) // 5)
            print(f"  [{label}] {bar} {int(pct):3d}%  {mb:.1f}/{total_mb:.1f} MB", flush=True)
            last[0] = pct
            last[1] = now

    return cb


def _ssh_cmd(ssh, cmd: str, timeout: int = 600) -> tuple:
    """Run command on remote, return (stdout, stderr, exit_code)."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    ec = stdout.channel.recv_exit_status()
    return out, err, ec


def _remote_file_exists(ssh, path: str) -> bool:
    out, _, _ = _ssh_cmd(ssh, f"test -f {path} && echo YES || echo NO")
    return "YES" in out


def _remote_dir_has_files(ssh, path: str) -> bool:
    out, _, _ = _ssh_cmd(ssh, f"ls {path}/* 2>/dev/null | head -1 && echo YES || echo NO")
    return "YES" in out


def _sftp_put_file(sftp, local: Path, remote: str, label: str):
    """Upload a single file with progress."""
    size = local.stat().st_size
    print(f"  [{label}] Uploading {local.name} ({size / (1024*1024):.1f} MB)...")
    sftp.put(str(local), remote, callback=_progress_callback(label))
    print(f"  [{label}] Upload complete: {local.name}")


def _upload_archive(ssh, sftp, name: str, local_dir: Path,
                    remote_raw: str, plan: dict, force: bool) -> bool:
    """Upload a single compressed archive, then extract on remote."""
    source_name = plan["source"]
    local_archive = local_dir / source_name
    remote_archive = f"{remote_raw}/{name}/{source_name}"
    remote_dir = f"{remote_raw}/{name}"

    if not local_archive.exists():
        print(f"  [{name}] Archive not found locally: {local_archive}")
        return False

    if not force and _remote_dir_has_files(ssh, remote_dir):
        out, _, _ = _ssh_cmd(ssh,
            f"ls {remote_dir}/ | grep -v '{source_name}' | grep -v manifest | grep -v README | head -3")
        if out.strip():
            print(f"  [{name}] Remote already has extracted files (skip)")
            return True

    _ssh_cmd(ssh, f"mkdir -p {remote_dir}")

    if not force and _remote_file_exists(ssh, remote_archive):
        print(f"  [{name}] Archive already on remote (skip upload)")
    else:
        _sftp_put_file(sftp, local_archive, remote_archive, name)

    if plan.get("extract"):
        print(f"  [{name}] Extracting on remote...")
        cmd = plan["extract_cmd"].format(archive=remote_archive, dest_dir=remote_dir)
        out, err, ec = _ssh_cmd(ssh, cmd, timeout=600)
        if ec == 0:
            print(f"  [{name}] Extract OK")
        else:
            print(f"  [{name}] Extract ERROR: {err[:300]}")
            return False

    return True


def _upload_files(ssh, sftp, name: str, local_dir: Path,
                  remote_raw: str, plan: dict, force: bool) -> bool:
    """Upload individual files for a dataset."""
    remote_dir = f"{remote_raw}/{name}"
    if not local_dir.exists():
        print(f"  [{name}] Local dir not found")
        return False

    _ssh_cmd(ssh, f"mkdir -p {remote_dir}")

    files = plan["files"]
    all_ok = True
    for fname in files:
        local_file = local_dir / fname
        remote_file = f"{remote_dir}/{fname}"
        if not local_file.exists():
            print(f"  [{name}] File not found locally: {fname}")
            all_ok = False
            continue
        if not force and _remote_file_exists(ssh, remote_file):
            print(f"  [{name}] {fname} already on remote (skip)")
            continue
        _sftp_put_file(sftp, local_file, remote_file, name)

    return all_ok


def _upload_dir(ssh, sftp, name: str, local_dir: Path,
                remote_raw: str, plan: dict, force: bool) -> bool:
    """Upload all files from a local directory to remote via SFTP."""
    remote_dir = f"{remote_raw}/{name}"
    if not local_dir.exists():
        print(f"  [{name}] Local dir not found")
        return False

    if not force and _remote_dir_has_files(ssh, remote_dir):
        print(f"  [{name}] Remote already has files (skip)")
        return True

    _ssh_cmd(ssh, f"mkdir -p {remote_dir}")

    items = _file_items(local_dir)
    uploaded = 0
    skipped = 0
    for f in items:
        rel = str(f.relative_to(local_dir)).replace("\\", "/")
        remote_path = f"{remote_dir}/{rel}"
        remote_parent = "/".join(remote_path.split("/")[:-1])
        _ssh_cmd(ssh, f"mkdir -p {remote_parent}")
        if not force and _remote_file_exists(ssh, remote_path):
            skipped += 1
            continue
        _sftp_put_file(sftp, f, remote_path, name)
        uploaded += 1

    print(f"  [{name}] Uploaded {uploaded} files, skipped {skipped}")
    return True


# ── Main ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload raw datasets to remote server (SFTP).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/upload_datasets.py --dry-run\n"
            "  python scripts/upload_datasets.py --host 1.2.3.4 --port 22 --user root --remote-dir /data --execute\n"
            "\nPassword is entered manually in terminal; never saved in code, docs, .env, or Agent memory."
        ),
    )
    parser.add_argument("--raw-root", default=None, help="Local raw dataset root")
    parser.add_argument("--datasets", nargs="*", default=None, help="Specific datasets")
    parser.add_argument("--force", action="store_true", help="Force re-upload")
    parser.add_argument(
        "--host", default=None, help="Remote SSH host (required for --execute)"
    )
    parser.add_argument(
        "--port", type=int, default=22, help="Remote SSH port (default: 22)"
    )
    parser.add_argument(
        "--user", default=None, help="Remote SSH user (required for --execute)"
    )
    parser.add_argument(
        "--remote-dir", default=None, help="Remote raw dataset root (required for --execute)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Print upload plan without connecting (default)"
    )
    parser.add_argument(
        "--execute", dest="dry_run", action="store_false",
        help="Perform the actual upload (password prompted in terminal)"
    )

    args = parser.parse_args()

    # Resolve local raw root
    if args.raw_root:
        local_raw = Path(args.raw_root).resolve()
    else:
        repo_root = Path(__file__).resolve().parents[1]
        local_raw = repo_root / "dataset" / "raw"

    if not local_raw.exists():
        print(f"ERROR: local raw directory not found: {local_raw}")
        sys.exit(1)

    # Select datasets
    available = [d for d in UPLOAD_PLAN if _find_local_dir(d, local_raw) is not None]
    if args.datasets:
        selected = [d for d in args.datasets if d in available]
    else:
        selected = available

    if not selected:
        print("No datasets to upload.")
        sys.exit(1)

    # ── Dry-run branch ──────────────────────────────────
    if args.dry_run:
        remote_dir = args.remote_dir or "/root/RNA-OmniDiffusion/dataset/raw"
        host = args.host or "<host>"
        port = args.port
        user = args.user or "<user>"
        _print_dry_run_plan(selected, local_raw, remote_dir, host, port, user)
        return

    # ── Execute branch (password prompted) ──────────────
    if not args.host or not args.user or not args.remote_dir:
        print("ERROR: --execute requires --host, --user, and --remote-dir.")
        sys.exit(1)

    host = args.host
    port = args.port
    user = args.user
    remote_raw = args.remote_dir

    # Password: prefer env var (with warning), otherwise prompt
    password: Optional[str] = os.environ.get("SSH_PASSWORD")
    if password:
        print("WARNING: SSH_PASSWORD read from environment variable. This is discouraged.")
        print("         Consider removing it and entering password manually.")
    else:
        password = getpass.getpass(f"Password for {user}@{host}: ")

    if not password:
        print("ERROR: No password provided.")
        sys.exit(1)

    print(f"Local raw:  {local_raw}")
    print(f"Remote:     {user}@{host}:{port}:{remote_raw}")
    print(f"Datasets:   {', '.join(selected)}")
    print()

    # Connect
    print(f"Connecting to {host}:{port}...")
    import paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(host, port=port, username=user, password=password,
                    timeout=30, banner_timeout=30)
    except Exception as e:
        print(f"ERROR: Connection failed: {e}")
        sys.exit(1)
    sftp = ssh.open_sftp()
    print("Connected.\n")

    _ssh_cmd(ssh, f"mkdir -p {remote_raw}")

    results: dict[str, str] = {}
    for ds in selected:
        print(f"{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")
        plan = UPLOAD_PLAN.get(ds, {})
        local_dir = _find_local_dir(ds, local_raw)
        if local_dir is None:
            print(f"  [{ds}] Local dir not found")
            results[ds] = "FAIL"
            continue
        try:
            if "source" in plan and "extract" in plan:
                ok = _upload_archive(ssh, sftp, ds, local_dir, remote_raw, plan, args.force)
            elif "files" in plan:
                ok = _upload_files(ssh, sftp, ds, local_dir, remote_raw, plan, args.force)
            elif plan.get("mode") == "rsync_dir":
                ok = _upload_dir(ssh, sftp, ds, local_dir, remote_raw, plan, args.force)
            else:
                print(f"  Unknown upload strategy for {ds}")
                ok = False
            results[ds] = "OK" if ok else "FAIL"
        except Exception as e:
            print(f"  [{ds}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[ds] = "ERROR"

    # Verify
    print(f"\n{'='*60}")
    print("REMOTE VERIFICATION")
    print(f"{'='*60}")
    out, _, _ = _ssh_cmd(ssh, f"du -sh {remote_raw}/*/ 2>/dev/null")
    print(out)

    sftp.close()
    ssh.close()

    print(f"\n{'='*60}")
    print("UPLOAD SUMMARY")
    print(f"{'='*60}")
    for ds_name, status in results.items():
        print(f"  {ds_name:<20} {status}")


if __name__ == "__main__":
    main()
