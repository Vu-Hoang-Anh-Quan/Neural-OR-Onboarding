"""Fix requirements.txt after `pip freeze` so it can be installed on a fresh environment.

Usage (run from the repo root, with the env activated):
    conda activate nor
    pip freeze > requirements.txt   # optional: re-freeze first if needed
    python src/utils/fix_requirements.py [path/to/requirements.txt]

Without an argument, fixes requirements.txt in the current directory.
A .bak copy of the original file is written next to it before changes.

What it fixes:
  1. Prepends `--extra-index-url https://download.pytorch.org/whl/cu128`
     (if not already present) so pip knows where to find torch's cu128 wheels.
  2. Strips the local-version suffix from local builds, e.g.
     `torch==2.11.0+cu128` -> `torch==2.11.0`. Pip then picks the +cu128
     wheel from the extra index automatically (+cu128 is only ever
     fetched from there, so nothing is lost).
  3. Removes the NVIDIA GPU runtime packages:
       - every `nvidia-*==...` package (not hosted on PyPI as-is -> pip
         fails if left in)
       - `cuda-toolkit` (NVIDIA's CUDA toolkit meta-package)
     Once torch + the extra index are in place, torch's wheel pulls the
     nvidia-* CUDA runtime libs it actually needs as its own deps.

  Kept (real PyPI packages, not NVIDIA CUDA binaries):
       - cuda-bindings
       - cuda-pathfinder
"""

import re
import shutil
import sys
from pathlib import Path

EXTRA_INDEX_LINE = "--extra-index-url https://download.pytorch.org/whl/cu128"

# Matches requirement lines like `nvidia-cublas-cu12==12.8.4.1`.
NVIDIA_RE = re.compile(r"^nvidia-[A-Za-z0-9._-]+\s*(==|~=|>=|<=|!=|===)")
# Matches `torch==2.11.0+cu128` so the local "+..." part can be stripped.
LOCAL_SUFFIX_RE = re.compile(r"^(.*?==\d[^+\s]*)\+\w+\s*$")
LOCAL_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(==|~=|>=|<=|!=|===)")


def drop_candidate(stripped):
    """True for lines to remove: nvidia-* packages and the cuda-toolkit meta-package."""
    if NVIDIA_RE.match(stripped):
        return True
    m = LOCAL_NAME_RE.match(stripped)
    return m is not None and m.group(1).lower() == "cuda-toolkit"


def strip_local_suffix(stripped):
    stripped = LOCAL_SUFFIX_RE.sub(r"\1", stripped)
    return stripped


def fix(text):
    dropped, modified = [], []
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#") or line.startswith("--") or line == "":
            out.append(line)
            continue
        if drop_candidate(line):
            dropped.append(line)
            continue
        fixed_line = strip_local_suffix(line)
        if fixed_line != line:
            modified.append(f"    {line}  ->  {fixed_line}")
        out.append(fixed_line)

    lines = [l for l in out if l != ""]
    if EXTRA_INDEX_LINE not in lines:
        lines.insert(0, EXTRA_INDEX_LINE)

    report = ["Fixed requirements file."]
    report.append("")
    report.append(f"1. Extra index line at top (added if missing): {EXTRA_INDEX_LINE}")
    report.append("")
    if modified:
        report.append("2. Stripped local (+) version suffixes:")
        report.extend(modified)
    else:
        report.append("2. Nothing to strip (no +local-version lines present).")
    report.append("")
    if dropped:
        report.append(f"3. Dropped {len(dropped)} NVIDIA/CUDA packages "
                      "(torch's wheel pulls the runtime libs it needs):")
        report.extend(f"    {d}" for d in dropped)
    else:
        report.append("3. Nothing to drop (no nvidia-*/cuda-toolkit lines present).")
    report.append("")
    report.append("Kept: cuda-bindings and cuda-pathfinder (real PyPI packages, not CUDA binaries).")
    return "\n".join(lines) + "\n", "\n".join(report)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("requirements.txt")
    if not path.exists():
        print(f"error: {path} not found (run from repo root or pass a path)", file=sys.stderr)
        sys.exit(1)

    original = path.read_text()
    fixed, report = fix(original)
    backup = path.with_name(path.name + ".bak")
    shutil.copy2(path, backup)
    path.write_text(fixed)
    print(report)
    print(f"Backup of the original: {backup}")


if __name__ == "__main__":
    main()
