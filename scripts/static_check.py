"""Run dependency-free static syntax checks for the educational project."""

from __future__ import annotations

import compileall
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    targets = [ROOT / "credit_simulator", ROOT / "scripts"]
    failures = [str(path) for path in targets if not compileall.compile_dir(path, quiet=1, force=True)]
    dashboard = ROOT / "dashboard.py"
    if not compileall.compile_file(dashboard, quiet=1, force=True):
        failures.append(str(dashboard))
    if failures:
        print("Static syntax check failed:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Static syntax check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
