"""
ojasx2/updater.py

Self-Updating Engine & Live Release Synchronizer for OjasX.
===========================================================
Enables automatic discovery and one-command self-updating for OjasX users:
1. Non-blocking asynchronous version check on import (0ms import latency overhead, 12h cache TTL).
2. Command-line updater: `ojasx update` or `python -m ojasx2.updater`.
3. Supports both git-cloned workspaces (`git pull --rebase`) and pip installations (`pip install --upgrade`).
4. Optional continuous auto-update mode via `OJASX_AUTO_UPDATE=1`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

REPO_OWNER = "mkrishna793"
REPO_NAME = "OjasX-open-source-CUDA"
GITHUB_API_RELEASES = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
GITHUB_GIT_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
CACHE_DIR = os.path.expanduser("~/.ojasx")
CACHE_FILE = os.path.join(CACHE_DIR, "update_cache.json")
CACHE_TTL_SECONDS = 12 * 3600  # 12 hours


def _parse_version(v_str: str) -> Tuple[int, ...]:
    """Parse semver string like '0.7.1' or 'v0.7.1' into a tuple of ints."""
    clean = v_str.lstrip("v").strip().split("-")[0].split("+")[0]
    parts = []
    for p in clean.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            break
    return tuple(parts)


def get_current_version() -> str:
    """Return local installed version of OjasX."""
    try:
        from . import __version__
        return __version__
    except Exception:
        return "0.7.1"


def get_repo_root() -> Optional[str]:
    """Return root directory of cloned git repository if running from source."""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git_dir = os.path.join(current_dir, ".git")
    if os.path.exists(git_dir):
        return current_dir
    return None


def fetch_latest_release_info(timeout: float = 3.0) -> Optional[Dict[str, Any]]:
    """Query GitHub REST API for latest release metadata."""
    import urllib.request
    try:
        req = urllib.request.Request(
            GITHUB_API_RELEASES,
            headers={"User-Agent": "OjasX-AutoUpdater", "Accept": "application/vnd.github.v3+json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "tag_name": data.get("tag_name", "").lstrip("v"),
                    "name": data.get("name", ""),
                    "published_at": data.get("published_at", ""),
                    "body": data.get("body", ""),
                    "html_url": data.get("html_url", ""),
                }
    except Exception:
        pass
    return None


def get_cached_update_status() -> Optional[Dict[str, Any]]:
    """Read update status from local cache if within TTL."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        last_check = data.get("last_check_timestamp", 0)
        if time.time() - last_check < CACHE_TTL_SECONDS:
            return data
    except Exception:
        pass
    return None


def save_update_cache(latest_version: str, has_update: bool, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Persist update check timestamp and release metadata to local cache."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        payload = {
            "last_check_timestamp": time.time(),
            "latest_version": latest_version,
            "has_update": has_update,
            "metadata": metadata or {},
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass


def check_for_updates(force: bool = False, timeout: float = 3.0) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Check whether a newer version is available.
    
    Returns:
        (has_update: bool, latest_version: str, metadata: dict)
    """
    current_ver = get_current_version()
    current_tuple = _parse_version(current_ver)

    if not force:
        cached = get_cached_update_status()
        if cached:
            latest_cached = cached.get("latest_version", current_ver)
            has_up = cached.get("has_update", False)
            return has_up, latest_cached, cached.get("metadata")

    info = fetch_latest_release_info(timeout=timeout)
    if not info:
        return False, current_ver, None

    latest_ver = info.get("tag_name", current_ver)
    latest_tuple = _parse_version(latest_ver)
    has_update = latest_tuple > current_tuple

    save_update_cache(latest_ver, has_update, info)
    return has_update, latest_ver, info


def format_update_banner(current_ver: str, latest_ver: str, is_git: bool = False) -> str:
    """Generate visual ASCII alert banner for terminal display."""
    cmd = "ojasx update" if not is_git else "git pull origin main  (or `ojasx update`)"
    lines = [
        "╭─────────────────────────────────────────────────────────────────────────────╮",
        f"│  ✨ OjasX Update Available: v{current_ver} → v{latest_ver}".ljust(78) + "│",
        "│  A new version with performance & mathematical fixes is live on GitHub!     │",
        "│                                                                             │",
        f"│  To update your terminal installation, run:                                │",
        f"│    {cmd}".ljust(78) + "│",
        "│                                                                             │",
        "│  Or in Python:                                                              │",
        "│    import ojasx2; ojasx2.update()                                           │",
        "╰─────────────────────────────────────────────────────────────────────────────╯",
    ]
    return "\n".join(lines)


def notify_if_update_available_background() -> None:
    """Asynchronously check for updates in a non-blocking daemon thread on import."""
    import multiprocessing
    if (
        os.environ.get("OJASX_DISABLE_UPDATE_CHECK", "0") in ("1", "true", "True")
        or "PYTEST_CURRENT_TEST" in os.environ
        or "CI" in os.environ
        or multiprocessing.current_process().name != "MainProcess"
    ):
        return

    def _worker() -> None:
        try:
            has_update, latest_ver, _ = check_for_updates(force=False, timeout=2.0)
            if has_update:
                cur_ver = get_current_version()
                is_git = get_repo_root() is not None
                if os.environ.get("OJASX_AUTO_UPDATE", "0") in ("1", "true", "True"):
                    perform_update(quiet=True)
                elif sys.stdout.isatty():
                    # Only print interactive banner if running in an interactive terminal
                    print("\n" + format_update_banner(cur_ver, latest_ver, is_git) + "\n", file=sys.stderr)
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True, name="ojasx-update-notifier")
    t.start()


def perform_update(target_version: Optional[str] = None, quiet: bool = False) -> bool:
    """Execute live in-place update for either Git cloned repo or Pip package."""
    repo_root = get_repo_root()
    cur_ver = get_current_version()

    if not quiet:
        print("=" * 80)
        print("OJASX LIVE SELF-UPDATING ENGINE")
        print("=" * 80)
        print(f"Current Installed Version : v{cur_ver}")

    if repo_root:
        if not quiet:
            print(f"Detected Installation Mode: Git Cloned Repository ({repo_root})")
            print("Fetching latest commits and tags from GitHub...")
        try:
            subprocess.run(["git", "fetch", "--tags", "origin"], cwd=repo_root, check=True)
            if target_version:
                checkout_target = f"v{target_version.lstrip('v')}"
                if not quiet:
                    print(f"Checking out release tag: {checkout_target}...")
                subprocess.run(["git", "checkout", checkout_target], cwd=repo_root, check=True)
            else:
                if not quiet:
                    print("Pulling latest commits from origin/main...")
                subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=repo_root, check=True)

            # Re-install editable packaging if python pip is present
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"], cwd=repo_root, check=False)
            
            if not quiet:
                print("Update successfully applied from Git origin!")
                print("Running verification doctor...")
            subprocess.run([sys.executable, "-m", "ojasx2.doctor"], check=False)
            return True
        except subprocess.CalledProcessError as e:
            if not quiet:
                print(f"[Error] Git update failed: {e}")
            return False
    else:
        if not quiet:
            print("Detected Installation Mode: Python Pip Package")
            target = f"git+{GITHUB_GIT_URL}"
            if target_version:
                target += f"@v{target_version.lstrip('v')}"
            print(f"Upgrading package via: {sys.executable} -m pip install --upgrade {target}...")
        try:
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", f"git+{GITHUB_GIT_URL}"]
            subprocess.run(cmd, check=True)
            if not quiet:
                print("Package upgraded successfully!")
                print("Running verification doctor...")
            subprocess.run([sys.executable, "-m", "ojasx2.doctor"], check=False)
            return True
        except subprocess.CalledProcessError as e:
            if not quiet:
                print(f"[Error] Pip package upgrade failed: {e}")
            return False


def update() -> bool:
    """Public Python API: `import ojasx2; ojasx2.update()`"""
    return perform_update(quiet=False)


def main() -> None:
    """CLI entry point for `python -m ojasx2.updater` or `ojasx update`."""
    args = sys.argv[1:]
    target_ver = None
    force = True

    if "--help" in args or "-h" in args:
        print("Usage: ojasx update [OPTIONS] [VERSION]")
        print("\nOptions:")
        print("  --check       Check for available updates without applying")
        print("  --auto        Run in quiet automated update mode")
        print("  -h, --help    Show this message and exit")
        return

    if "--check" in args:
        cur_ver = get_current_version()
        has_up, latest_ver, meta = check_for_updates(force=True)
        if has_up:
            print(f"Update available: v{cur_ver} -> v{latest_ver}")
            if meta and meta.get("name"):
                print(f"Release Name: {meta['name']}")
        else:
            print(f"OjasX is up to date (v{cur_ver}).")
        return

    if args and not args[0].startswith("-"):
        target_ver = args[0]

    perform_update(target_version=target_ver)


def cli_router() -> None:
    """Top-level `ojasx` CLI router supporting `ojasx update`, `ojasx doctor`, etc."""
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("OjasX: Math-Level PyTorch Training & Memory Compiler")
        print("\nCommands:")
        print("  ojasx update   Check and update OjasX to the latest release")
        print("  ojasx doctor   Run live hardware, memory, and compiler diagnostics")
        print("  ojasx version  Print current version and update status")
        return

    subcmd = args[0].lower()
    sys.argv = [sys.argv[0]] + args[1:]
    if subcmd in ("update", "upgrade"):
        main()
    elif subcmd in ("doctor", "diagnostics", "check"):
        from .doctor import main as doctor_main
        doctor_main()
    elif subcmd in ("version", "-v", "--version"):
        cur_ver = get_current_version()
        has_up, latest_ver, _ = check_for_updates(force=False)
        print(f"OjasX version {cur_ver}")
        if has_up:
            print(f"  (Note: New version v{latest_ver} available! Run `ojasx update` to upgrade)")
    else:
        print(f"Unknown command: {subcmd}. Run `ojasx --help` for available commands.")


if __name__ == "__main__":
    main()
