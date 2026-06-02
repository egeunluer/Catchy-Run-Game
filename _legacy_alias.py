"""Make legacy `grid.*` module paths resolve to `catchy_run.*`.

The package was renamed from `grid` to `catchy_run`. Older SB3 checkpoints
pickled fully-qualified class references under `grid.X.Y`, so loading them
with `MaskablePPO.load` triggers `import grid` and fails. Installing this
meta-path finder makes any `import grid[.X.Y]` resolve to `catchy_run[.X.Y]`.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

_LEGACY = "grid"
_CURRENT = "catchy_run"


class _GridAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _LEGACY and not name.startswith(_LEGACY + "."):
            return None
        new_name = _CURRENT + name[len(_LEGACY):]
        mod = importlib.import_module(new_name)
        sys.modules[name] = mod
        return importlib.util.spec_from_loader(name, loader=None)


if not any(type(f).__name__ == "_GridAliasFinder" for f in sys.meta_path):
    sys.meta_path.insert(0, _GridAliasFinder())
