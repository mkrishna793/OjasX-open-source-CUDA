#!/usr/bin/env python3
"""Check first-party Markdown links using the paths rendered by GitHub.

Document-relative links are the only accepted local form. Repository-root-style
links inside docs/ are not silently reinterpreted, because GitHub does not do so.
External URLs and backslash-containing LaTeX constructs are not local links.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(
    r"(?P<image>!?)\[(?P<label>[^\]\n]*)\]"
    r"\((?P<destination><[^>\n]+>|[^)\s]+)(?P<suffix>\s+[^)\n]*)?\)"
)
EXTERNAL = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|#|//)")


def markdown_files() -> list[Path]:
    files = set(ROOT.glob("*.md"))
    for directory in ("docs", "reports", "rocq"):
        files.update((ROOT / directory).rglob("*.md"))
    provenance = ROOT / "third_party" / "PROVENANCE.md"
    if provenance.exists():
        files.add(provenance)
    return sorted(path for path in files if path.is_file())


def destination_text(raw: str) -> str:
    return raw[1:-1] if raw.startswith("<") and raw.endswith(">") else raw


def is_external(target: str) -> bool:
    return bool(EXTERNAL.match(target)) or "\\" in target


def local_path(target: str) -> str:
    return unquote(target.split("#", 1)[0].split("?", 1)[0])


def check_file(path: Path) -> tuple[int, list[dict[str, object]]]:
    text = path.read_text(encoding="utf-8")
    checked = 0
    broken: list[dict[str, object]] = []
    for match in LINK.finditer(text):
        target = destination_text(match.group("destination"))
        if not target or is_external(target):
            continue
        checked += 1
        local = local_path(target)
        if not local:
            continue
        resolved = (path.parent / local).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            inside = False
        else:
            inside = True
        if not inside or not resolved.exists():
            try:
                display_path = str(path.relative_to(ROOT))
            except ValueError:
                display_path = str(path)
            broken.append({
                "file": display_path,
                "line": text.count("\n", 0, match.start()) + 1,
                "target": target,
            })
    return checked, broken


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="optional machine-readable result")
    args = parser.parse_args()
    checked = 0
    broken: list[dict[str, object]] = []
    files = markdown_files()
    for path in files:
        count, failures = check_file(path)
        checked += count
        broken.extend(failures)
    result = {
        "schema_version": 1,
        "policy": "document-relative local targets; external URLs and LaTeX excluded",
        "markdown_files": len(files),
        "local_links_checked": checked,
        "broken_count": len(broken),
        "broken": broken,
    }
    if args.json:
        destination = args.json if args.json.is_absolute() else ROOT / args.json
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for failure in broken:
        print(f"{failure['file']}:{failure['line']}: missing {failure['target']}")
    print(f"Checked {checked} local links in {len(files)} Markdown files: {len(broken)} broken")
    return int(bool(broken))


if __name__ == "__main__":
    raise SystemExit(main())
