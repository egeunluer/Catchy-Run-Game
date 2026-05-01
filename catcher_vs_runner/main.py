"""Entry point for the Catcher vs. Runner desktop app.

Run via:

    python -m catcher_vs_runner.main

or double-click this file (assuming `.py` is associated with Python and
`pygame` + `numpy` are installed in the active environment).
"""

from __future__ import annotations

from .render.pygame_app import App


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
