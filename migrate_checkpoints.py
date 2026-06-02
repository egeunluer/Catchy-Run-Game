"""One-shot: re-save every SB3 checkpoint so pickled class refs point at
the current `catchy_run.*` module paths instead of the legacy `grid.*` ones.

Run from the repo root:
    python -m catchy_run.migrate_checkpoints
"""
from __future__ import annotations

from pathlib import Path

import catchy_run  # noqa: F401  (installs the grid->catchy_run alias)
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parent / "trained_model_checkpoints"


def main() -> None:
    zips = sorted(ROOT.rglob("*.zip"))
    if not zips:
        print(f"no .zip checkpoints found under {ROOT}")
        return
    for p in zips:
        print(f"migrating {p.relative_to(ROOT.parent)} ...", flush=True)
        model = MaskablePPO.load(str(p))
        model.save(str(p))
    print(f"done — migrated {len(zips)} checkpoint(s)")


if __name__ == "__main__":
    main()
