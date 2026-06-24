"""Backward-compatibility shim for checkpoints saved under the old layout.

The packages now live as top-level ``rl_agent`` and ``catchy_run_game`` (the
project is run from inside ``catchy_run/``). Older ``.zip`` checkpoints, however,
were saved while these packages were imported under a ``catchy_run.`` namespace,
so their pickled feature-extractor class is recorded as e.g.
``catchy_run.rl_agent.custom_cnn.<Class>``. ``MaskablePPO.load`` resolves that
path via ``importlib``, which fails once ``catchy_run`` is no longer importable.

``install_legacy_module_alias`` registers a ``catchy_run`` package alias that
points back at this directory, so those legacy module paths resolve again. It is
idempotent and a no-op once anything named ``catchy_run`` is already importable.
New checkpoints save under the top-level names and don't need this.
"""
import importlib
import pathlib
import sys
import types

_LEGACY_ROOT = "catchy_run"
_SUBPACKAGES = ("rl_agent", "catchy_run_game")


def install_legacy_module_alias() -> None:
    if _LEGACY_ROOT in sys.modules:
        return
    pkg = types.ModuleType(_LEGACY_ROOT)
    # Directory that holds rl_agent/ and catchy_run_game/ (i.e. catchy_run/).
    pkg.__path__ = [str(pathlib.Path(__file__).resolve().parent.parent)]
    sys.modules[_LEGACY_ROOT] = pkg
    for sub in _SUBPACKAGES:
        try:
            sys.modules[f"{_LEGACY_ROOT}.{sub}"] = importlib.import_module(sub)
        except ImportError:
            pass
