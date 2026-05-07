"""
Upload raw datasets from local to remote server via SFTP.

Strategy:
  - Upload compressed archives (tar.gz, .gz) where available → extract on remote
  - Upload individual files for datasets without compressed archives (bpRNA)
  - Skip datasets already present on remote (unless --force)

Usage:
  python scripts/upload_datasets.py
  python scripts/upload_datasets.py --datasets ArchiveII Rfam
  python scripts/upload_datasets.py --force
"""

import argparse
import os
import sys
import time
from pathlib import Path

import paramiko

# Remote connection info
SSH_HOST = "connect.nmb1.seetacloud.com"
SSH_PORT = 49018
SSH_USER = "root"
SSH_PASSWORD = "U7xdqhUvXyt0"
REMOTE_RAW = "/root/RNA-OmniDiffusion/dataset/raw"

# Upload plan: (dataset_name, local_glob_or_file, extract_on_remote)
# local path is relative to local_raw_root
# Upload plan keyed by canonical dataset name.
# The script auto-detects the actual local directory name (case-insensitive on Windows).
UPLOAD_PLAN = {
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
}


def _find_local_dir(name: str, local_raw: Path) -> Path | None:
    """Find a dataset directory by canonical name, case-insensitive."""
    # Try exact match first
    direct = local_raw / name
    if direct.exists():
        return direct
    # Try case-insensitive match
    name_lower = name.lower()
    for entry in local_raw.iterdir():
        if entry.is_dir() and entry.name.lower() == name_lower:
            return entry
    return None


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


def ssh_cmd(ssh: paramiko.SSHClient, cmd: str, timeout: int = 600) -> tuple:
    """Run command on remote, return (stdout, stderr, exit_code)."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    ec = stdout.channel.recv_exit_status()
    return out, err, ec


def remote_file_exists(ssh: paramiko.SSHClient, path: str) -> bool:
    out, _, ec = ssh_cmd(ssh, f"test -f {path} && echo YES || echo NO")
    return "YES" in out


def remote_dir_has_files(ssh: paramiko.SSHClient, path: str) -> bool:
    out, _, ec = ssh_cmd(ssh, f"ls {path}/* 2>/dev/null | head -1 && echo YES || echo NO")
    return "YES" in out


def sftp_put_file(sftp, local: Path, remote: str, label: str):
    """Upload a single file with progress."""
    size = local.stat().st_size
    print(f"  [{label}] Uploading {local.name} ({size / (1024*1024):.1f} MB)...")
    sftp.put(str(local), remote, callback=_progress_callback(label))
    print(f"  [{label}] Upload complete: {local.name}")


def upload_archive_dataset(
    ssh: paramiko.SSHClient,
    sftp,
    name: str,
    local_dir: Path,
    remote_raw: str,
    plan: dict,
    force: bool,
) -> bool:
    """Upload a single compressed archive, then extract on remote."""
    source_name = plan["source"]
    local_archive = local_dir / source_name
    remote_archive = f"{remote_raw}/{name}/{source_name}"
    remote_dir = f"{remote_raw}/{name}"

    if not local_archive.exists():
        print(f"  [{name}] Archive not found locally: {local_archive}")
        return False

    # Check if already extracted on remote
    if not force and remote_dir_has_files(ssh, remote_dir):
        # Check if it seems already extracted (not just the archive)
        out, _, _ = ssh_cmd(ssh, f"ls {remote_dir}/ | grep -v '{source_name}' | grep -v manifest | grep -v README | head -3")
        if out.strip():
            print(f"  [{name}] Remote already has extracted files (skip)")
            return True

    # Ensure remote dir
    ssh_cmd(ssh, f"mkdir -p {remote_dir}")

    # Upload archive
    if not force and remote_file_exists(ssh, remote_archive):
        print(f"  [{name}] Archive already on remote (skip upload)")
    else:
        sftp_put_file(sftp, local_archive, remote_archive, name)

    # Extract
    if plan.get("extract"):
        print(f"  [{name}] Extracting on remote...")
        cmd = plan["extract_cmd"].format(archive=remote_archive, dest_dir=remote_dir)
        out, err, ec = ssh_cmd(ssh, cmd, timeout=600)
        if ec == 0:
            print(f"  [{name}] Extract OK")
        else:
            print(f"  [{name}] Extract ERROR: {err[:300]}")
            return False

    return True


def upload_file_dataset(
    ssh: paramiko.SSHClient,
    sftp,
    name: str,
    local_dir: Path,
    remote_raw: str,
    plan: dict,
    force: bool,
) -> bool:
    """Upload individual files for a dataset."""
    remote_dir = f"{remote_raw}/{name}"

    if not local_dir.exists():
        print(f"  [{name}] Local dir not found")
        return False

    ssh_cmd(ssh, f"mkdir -p {remote_dir}")

    files = plan["files"]
    all_ok = True
    for fname in files:
        local_file = local_dir / fname
        remote_file = f"{remote_dir}/{fname}"
        if not local_file.exists():
            print(f"  [{name}] File not found locally: {fname}")
            all_ok = False
            continue
        if not force and remote_file_exists(ssh, remote_file):
            print(f"  [{name}] {fname} already on remote (skip)")
            continue
        sftp_put_file(sftp, local_file, remote_file, name)

    return all_ok


def upload_dir_dataset(
    ssh: paramiko.SSHClient,
    sftp,
    name: str,
    local_dir: Path,
    remote_raw: str,
    plan: dict,
    force: bool,
) -> bool:
    """Upload all files from a local directory to remote via SFTP."""
    remote_dir = f"{remote_raw}/{name}"

    if not local_dir.exists():
        print(f"  [{name}] Local dir not found")
        return False

    # If remote already has files, optionally skip
    if not force and remote_dir_has_files(ssh, remote_dir):
        print(f"  [{name}] Remote already has files (skip)")
        return True

    ssh_cmd(ssh, f"mkdir -p {remote_dir}")

    # Upload all files (skip manifest.json and README_DOWNLOAD.txt)
    uploaded = 0
    skipped = 0
    for item in local_dir.iterdir():
        if item.name in ("manifest.json", "README_DOWNLOAD.txt", ".gitkeep"):
            continue
        if item.is_file():
            remote_path = f"{remote_dir}/{item.name}"
            if not force and remote_file_exists(ssh, remote_path):
                skipped += 1
                continue
            sftp_put_file(sftp, item, remote_path, name)
            uploaded += 1
        elif item.is_dir():
            # Upload whole directory recursively
            for f in item.rglob("*"):
                if f.is_file():
                    rel = str(f.relative_to(local_dir)).replace("\\", "/")
                    remote_path = f"{remote_dir}/{rel}"
                    remote_parent = "/".join(remote_path.split("/")[:-1])
                    ssh_cmd(ssh, f"mkdir -p {remote_parent}")
                    if not force and remote_file_exists(ssh, remote_path):
                        skipped += 1
                        continue
                    sftp_put_file(sftp, f, remote_path, name)
                    uploaded += 1

    print(f"  [{name}] Uploaded {uploaded} files, skipped {skipped}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload raw datasets to remote server")
    parser.add_argument("--raw-root", default=None, help="Local raw dataset root")
    parser.add_argument("--datasets", nargs="*", default=None, help="Specific datasets")
    parser.add_argument("--force", action="store_true", help="Force re-upload")
    parser.add_argument("--remote-raw", default=REMOTE_RAW, help="Remote raw root")
    args = parser.parse_args()

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

    print(f"Local raw:  {local_raw}")
    print(f"Remote raw: {args.remote_raw}")
    print(f"Datasets:   {', '.join(selected)}")
    print()

    # Connect
    print(f"Connecting to {SSH_HOST}:{SSH_PORT}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, timeout=30, banner_timeout=30)
    sftp = ssh.open_sftp()
    print("Connected.\n")

    # Ensure remote raw exists
    ssh_cmd(ssh, f"mkdir -p {args.remote_raw}")

    results = {}
    for ds in selected:
        print(f"{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")
        plan = UPLOAD_PLAN[ds]
        local_dir = _find_local_dir(ds, local_raw)
        try:
            if "source" in plan and "extract" in plan:
                ok = upload_archive_dataset(ssh, sftp, ds, local_dir, args.remote_raw, plan, args.force)
            elif "files" in plan:
                ok = upload_file_dataset(ssh, sftp, ds, local_dir, args.remote_raw, plan, args.force)
            elif plan.get("mode") == "rsync_dir":
                ok = upload_dir_dataset(ssh, sftp, ds, local_dir, args.remote_raw, plan, args.force)
            else:
                print(f"  Unknown upload strategy for {ds}")
                ok = False
            results[ds] = "OK" if ok else "FAIL"
        except Exception as e:
            print(f"  [{ds}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[ds] = f"ERROR"

    # Verify
    print(f"\n{'='*60}")
    print("REMOTE VERIFICATION")
    print(f"{'='*60}")
    out, _, _ = ssh_cmd(ssh, f"du -sh {args.remote_raw}/*/ 2>/dev/null")
    print(out)

    sftp.close()
    ssh.close()

    print(f"\n{'='*60}")
    print("UPLOAD SUMMARY")
    print(f"{'='*60}")
    for ds, status in results.items():
        print(f"  {ds:<20} {status}")


if __name__ == "__main__":
    main()
