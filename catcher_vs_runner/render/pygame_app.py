"""Pygame renderer + click-to-play interface for Catcher vs. Runner.

The app imports `engine` and uses its public API only — clicks are
translated to action indices and submitted via `engine.step`. The trained
DRL agent (when the user adds one later) plugs into the same loop with
zero changes here.

Modes:
* HVH        — both sides played by humans (default)
* AI_RUNNER  — heuristic plays the runner, human plays the catcher
* AI_CATCHER — heuristic plays the catcher, human plays the runner
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import pygame

from .. import engine
from ..actions import (
    ACTION_MOVE_E,
    ACTION_MOVE_N,
    ACTION_MOVE_S,
    ACTION_MOVE_W,
    ACTION_PLACE_WALL_E,
    ACTION_PLACE_WALL_N,
    ACTION_PLACE_WALL_S,
    ACTION_PLACE_WALL_W,
    ACTION_REMOVE_WALL_E,
    ACTION_REMOVE_WALL_N,
    ACTION_REMOVE_WALL_S,
    ACTION_REMOVE_WALL_W,
    ACTION_SPECIAL_E,
    ACTION_SPECIAL_N,
    ACTION_SPECIAL_S,
    ACTION_SPECIAL_W,
    ACTION_WAIT,
)
from ..agents.heuristic import catcher_policy, runner_policy

# --- Layout --------------------------------------------------------------

CELL = 56
GRID_ORIGIN = (24, 24)
GRID_PIXELS = engine.BOARD_SIZE * CELL
SIDEBAR_X = GRID_ORIGIN[0] + GRID_PIXELS + 24
SIDEBAR_W = 320
WINDOW_W = SIDEBAR_X + SIDEBAR_W + 24
WINDOW_H = GRID_ORIGIN[1] + GRID_PIXELS + 24
FPS = 60
AI_DELAY_MS = 350

# --- Colors --------------------------------------------------------------

C_BG = (24, 26, 34)
C_GRID = (60, 64, 78)
C_CELL = (40, 44, 56)
C_CELL_HL = (74, 96, 130)
C_RUNNER = (90, 168, 230)
C_CATCHER = (220, 96, 96)
C_RUNNER_WALL = (62, 110, 158)
C_CATCHER_WALL = (158, 70, 70)
C_TEXT = (228, 230, 240)
C_TEXT_DIM = (140, 144, 160)
C_BTN = (52, 58, 76)
C_BTN_HOVER = (68, 78, 102)
C_BTN_ACTIVE = (84, 132, 188)
C_BANNER = (32, 36, 48)


# --- Modes ---------------------------------------------------------------

MODE_MOVE = "MOVE"
MODE_PLACE = "PLACE"
MODE_REMOVE = "REMOVE"
MODE_SPECIAL = "SPECIAL"
ACTION_MODES = [MODE_MOVE, MODE_PLACE, MODE_REMOVE, MODE_SPECIAL]

GM_HVH = "HVH"
GM_AI_RUNNER = "AI_RUNNER"   # AI plays runner, human catcher
GM_AI_CATCHER = "AI_CATCHER"  # AI plays catcher, human runner


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    key: str  # arbitrary identifier the click handler dispatches on


# --- Action mapping ------------------------------------------------------

_DIR_TO_INDEX = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}

_MOVE_BY_INDEX = (ACTION_MOVE_N, ACTION_MOVE_E, ACTION_MOVE_S, ACTION_MOVE_W)
_PLACE_BY_INDEX = (
    ACTION_PLACE_WALL_N, ACTION_PLACE_WALL_E,
    ACTION_PLACE_WALL_S, ACTION_PLACE_WALL_W,
)
_REMOVE_BY_INDEX = (
    ACTION_REMOVE_WALL_N, ACTION_REMOVE_WALL_E,
    ACTION_REMOVE_WALL_S, ACTION_REMOVE_WALL_W,
)
_SPECIAL_BY_INDEX = (
    ACTION_SPECIAL_N, ACTION_SPECIAL_E,
    ACTION_SPECIAL_S, ACTION_SPECIAL_W,
)


def _cell_to_action(state: engine.GameState, mode: str, cell: tuple[int, int]) -> Optional[int]:
    """Translate a clicked cell + selected action mode to an action index.

    Returns None if the click does not correspond to a valid direction for
    the selected mode (e.g., diagonal click, or 2-step click in MOVE mode).
    Does not check legality — that is left to `engine.step`.
    """
    actor_pos = state.own_position()
    dx = cell[0] - actor_pos[0]
    dy = cell[1] - actor_pos[1]

    if mode == MODE_SPECIAL:
        if dx == 0 and dy in (-2, 2):
            unit = (0, dy // 2)
        elif dy == 0 and dx in (-2, 2):
            unit = (dx // 2, 0)
        else:
            return None
        return _SPECIAL_BY_INDEX[_DIR_TO_INDEX[unit]]

    if abs(dx) + abs(dy) != 1:
        return None
    unit = (dx, dy)
    idx = _DIR_TO_INDEX[unit]
    if mode == MODE_MOVE:
        return _MOVE_BY_INDEX[idx]
    if mode == MODE_PLACE:
        return _PLACE_BY_INDEX[idx]
    if mode == MODE_REMOVE:
        return _REMOVE_BY_INDEX[idx]
    return None


def _legal_target_cells(state: engine.GameState, mode: str) -> set[tuple[int, int]]:
    """Cells the user could click in the current state under `mode`."""
    if state.terminated:
        return set()
    actor = state.own_position()
    mask = engine.legal_action_mask(state)
    cells: set[tuple[int, int]] = set()

    def add(action: int, dx: int, dy: int, distance: int = 1) -> None:
        if mask[action]:
            cells.add((actor[0] + dx * distance, actor[1] + dy * distance))

    if mode == MODE_MOVE:
        add(ACTION_MOVE_N, 0, -1)
        add(ACTION_MOVE_E, 1, 0)
        add(ACTION_MOVE_S, 0, 1)
        add(ACTION_MOVE_W, -1, 0)
    elif mode == MODE_PLACE:
        add(ACTION_PLACE_WALL_N, 0, -1)
        add(ACTION_PLACE_WALL_E, 1, 0)
        add(ACTION_PLACE_WALL_S, 0, 1)
        add(ACTION_PLACE_WALL_W, -1, 0)
    elif mode == MODE_REMOVE:
        add(ACTION_REMOVE_WALL_N, 0, -1)
        add(ACTION_REMOVE_WALL_E, 1, 0)
        add(ACTION_REMOVE_WALL_S, 0, 1)
        add(ACTION_REMOVE_WALL_W, -1, 0)
    elif mode == MODE_SPECIAL:
        add(ACTION_SPECIAL_N, 0, -1, distance=2)
        add(ACTION_SPECIAL_E, 1, 0, distance=2)
        add(ACTION_SPECIAL_S, 0, 1, distance=2)
        add(ACTION_SPECIAL_W, -1, 0, distance=2)
    return cells


# --- App -----------------------------------------------------------------


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Catcher vs. Runner")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("helvetica", 22, bold=True)
        self.font = pygame.font.SysFont("helvetica", 16)
        self.font_sm = pygame.font.SysFont("helvetica", 13)

        self.state = engine.reset()
        self.mode: str = MODE_MOVE
        self.game_mode: str = GM_HVH
        self.rng = random.Random()
        self.ai_pending_at: Optional[int] = None  # pygame ticks

        self.buttons: list[Button] = self._build_buttons()

    # -- Layout -------------------------------------------------------

    def _build_buttons(self) -> list[Button]:
        buttons: list[Button] = []
        y = 110
        for mode in ACTION_MODES:
            buttons.append(Button(
                pygame.Rect(SIDEBAR_X, y, SIDEBAR_W, 36), mode.title(),
                f"mode:{mode}",
            ))
            y += 42
        # Wait button (special — not a mode, just a one-shot action).
        buttons.append(Button(
            pygame.Rect(SIDEBAR_X, y, SIDEBAR_W, 36), "Wait", "wait",
        ))
        y += 56
        # Game-mode buttons
        for label, key in [
            ("Human vs. Human", f"gm:{GM_HVH}"),
            ("Human catcher (AI runner)", f"gm:{GM_AI_RUNNER}"),
            ("Human runner (AI catcher)", f"gm:{GM_AI_CATCHER}"),
        ]:
            buttons.append(Button(
                pygame.Rect(SIDEBAR_X, y, SIDEBAR_W, 32), label, key,
            ))
            y += 36
        y += 12
        buttons.append(Button(
            pygame.Rect(SIDEBAR_X, y, SIDEBAR_W, 36), "New Game", "new",
        ))
        return buttons

    # -- Cell <-> pixel ----------------------------------------------

    def _cell_at(self, mx: int, my: int) -> Optional[tuple[int, int]]:
        gx, gy = GRID_ORIGIN
        if not (gx <= mx < gx + GRID_PIXELS and gy <= my < gy + GRID_PIXELS):
            return None
        return ((mx - gx) // CELL, (my - gy) // CELL)

    def _cell_rect(self, cell: tuple[int, int]) -> pygame.Rect:
        gx, gy = GRID_ORIGIN
        return pygame.Rect(gx + cell[0] * CELL, gy + cell[1] * CELL, CELL, CELL)

    # -- Game flow ----------------------------------------------------

    def _ai_role(self) -> Optional[str]:
        if self.game_mode == GM_AI_RUNNER:
            return "runner"
        if self.game_mode == GM_AI_CATCHER:
            return "catcher"
        return None

    def _is_ai_turn(self) -> bool:
        return (not self.state.terminated) and self.state.current_agent == self._ai_role()

    def _apply_action(self, action: int) -> None:
        try:
            new_state, *_ = engine.step(self.state, action)
        except ValueError:
            return  # silently ignore illegal click
        self.state = new_state
        if self._is_ai_turn():
            self.ai_pending_at = pygame.time.get_ticks() + AI_DELAY_MS

    def _maybe_run_ai(self) -> None:
        if self.ai_pending_at is None or not self._is_ai_turn():
            self.ai_pending_at = None
            return
        if pygame.time.get_ticks() < self.ai_pending_at:
            return
        if self.state.current_agent == "runner":
            action = runner_policy(self.state, self.rng)
        else:
            action = catcher_policy(self.state, self.rng)
        new_state, *_ = engine.step(self.state, action)
        self.state = new_state
        if self._is_ai_turn():
            self.ai_pending_at = pygame.time.get_ticks() + AI_DELAY_MS
        else:
            self.ai_pending_at = None

    def _new_game(self) -> None:
        self.state = engine.reset()
        self.mode = MODE_MOVE
        self.ai_pending_at = pygame.time.get_ticks() + AI_DELAY_MS if self._is_ai_turn() else None

    # -- Event handlers -----------------------------------------------

    def _on_button(self, key: str) -> None:
        if key == "wait":
            if not self.state.terminated and not self._is_ai_turn():
                self._apply_action(ACTION_WAIT)
        elif key == "new":
            self._new_game()
        elif key.startswith("mode:"):
            self.mode = key.split(":", 1)[1]
        elif key.startswith("gm:"):
            self.game_mode = key.split(":", 1)[1]
            self._new_game()

    def _on_grid_click(self, cell: tuple[int, int]) -> None:
        if self.state.terminated or self._is_ai_turn():
            return
        action = _cell_to_action(self.state, self.mode, cell)
        if action is None:
            return
        self._apply_action(action)

    def _handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            for btn in self.buttons:
                if btn.rect.collidepoint(mx, my):
                    self._on_button(btn.key)
                    return True
            cell = self._cell_at(mx, my)
            if cell is not None:
                self._on_grid_click(cell)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_r:
                self._new_game()
            elif event.key == pygame.K_1:
                self.mode = MODE_MOVE
            elif event.key == pygame.K_2:
                self.mode = MODE_PLACE
            elif event.key == pygame.K_3:
                self.mode = MODE_REMOVE
            elif event.key == pygame.K_4:
                self.mode = MODE_SPECIAL
            elif event.key == pygame.K_SPACE:
                if not self.state.terminated and not self._is_ai_turn():
                    self._apply_action(ACTION_WAIT)
        return True

    # -- Drawing ------------------------------------------------------

    def _draw_grid(self) -> None:
        gx, gy = GRID_ORIGIN
        for cy in range(engine.BOARD_SIZE):
            for cx in range(engine.BOARD_SIZE):
                rect = pygame.Rect(gx + cx * CELL, gy + cy * CELL, CELL, CELL)
                pygame.draw.rect(self.screen, C_CELL, rect)
                pygame.draw.rect(self.screen, C_GRID, rect, 1)

        # Highlight legal target cells for the selected mode.
        if not self.state.terminated and not self._is_ai_turn():
            targets = _legal_target_cells(self.state, self.mode)
            for cell in targets:
                rect = self._cell_rect(cell)
                pygame.draw.rect(self.screen, C_CELL_HL, rect)
                pygame.draw.rect(self.screen, C_GRID, rect, 1)

        # Walls
        for x, y in self.state.runner_walls:
            self._draw_wall((x, y), C_RUNNER_WALL)
        for x, y in self.state.catcher_walls:
            self._draw_wall((x, y), C_CATCHER_WALL)

        # Agents
        self._draw_agent(self.state.runner_pos, C_RUNNER, "R")
        self._draw_agent(self.state.catcher_pos, C_CATCHER, "C")

    def _draw_wall(self, cell: tuple[int, int], color: tuple[int, int, int]) -> None:
        rect = self._cell_rect(cell).inflate(-8, -8)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)

    def _draw_agent(self, cell: tuple[int, int], color: tuple[int, int, int], label: str) -> None:
        rect = self._cell_rect(cell)
        center = rect.center
        pygame.draw.circle(self.screen, color, center, CELL // 2 - 8)
        text = self.font_lg.render(label, True, (255, 255, 255))
        self.screen.blit(text, text.get_rect(center=center))

    def _draw_sidebar(self) -> None:
        x = SIDEBAR_X
        y = 24
        title = self.font_lg.render("Catcher vs. Runner", True, C_TEXT)
        self.screen.blit(title, (x, y))
        y += 32

        # Status line
        if self.state.terminated:
            winner = self.state.winner or "?"
            status = f"Game over — {winner} wins"
            color = C_RUNNER if winner == "runner" else C_CATCHER
        else:
            mover = self.state.current_agent
            color = C_RUNNER if mover == "runner" else C_CATCHER
            ai_tag = " (AI)" if self._ai_role() == mover else ""
            status = f"Turn {self.state.turn}/{engine.TURN_LIMIT} — {mover}{ai_tag}'s move"
        self.screen.blit(self.font.render(status, True, color), (x, y))
        y += 24

        # Stats
        r_walls = len(self.state.runner_walls)
        c_walls = len(self.state.catcher_walls)
        lines = [
            f"Runner @ {self.state.runner_pos}  walls {r_walls}/{engine.RUNNER_WALL_CAP}  sprint {self.state.sprint_charges}/{engine.SPRINT_CHARGES}",
            f"Catcher @ {self.state.catcher_pos}  walls {c_walls}/{engine.CATCHER_WALL_CAP}  vault {self.state.vault_charges}/{engine.VAULT_CHARGES}",
        ]
        for line in lines:
            self.screen.blit(self.font_sm.render(line, True, C_TEXT_DIM), (x, y))
            y += 16

        # Mode-specific helper text
        special_label = "Sprint (2 cells)" if self.state.current_agent == "runner" else "Vault (over wall)"
        helper_lines = [
            "Click a mode, then click a cell.",
            f"Special action: {special_label}",
            "Keys: 1-4 modes, Space wait, R new game",
        ]
        y += 6
        for line in helper_lines:
            self.screen.blit(self.font_sm.render(line, True, C_TEXT_DIM), (x, y))
            y += 14

        # Buttons
        mouse = pygame.mouse.get_pos()
        for btn in self.buttons:
            self._draw_button(btn, mouse)

    def _draw_button(self, btn: Button, mouse: tuple[int, int]) -> None:
        active = (
            (btn.key == f"mode:{self.mode}")
            or (btn.key == f"gm:{self.game_mode}")
        )
        hover = btn.rect.collidepoint(mouse)
        if active:
            color = C_BTN_ACTIVE
        elif hover:
            color = C_BTN_HOVER
        else:
            color = C_BTN
        pygame.draw.rect(self.screen, color, btn.rect, border_radius=6)
        text = self.font.render(btn.label, True, C_TEXT)
        self.screen.blit(text, text.get_rect(center=btn.rect.center))

    def _draw_winner_banner(self) -> None:
        if not self.state.terminated:
            return
        gx, gy = GRID_ORIGIN
        banner = pygame.Surface((GRID_PIXELS, 80), pygame.SRCALPHA)
        banner.fill((*C_BANNER, 220))
        self.screen.blit(banner, (gx, gy + GRID_PIXELS // 2 - 40))
        winner = self.state.winner or "?"
        msg = f"{winner.upper()} WINS"
        text = self.font_lg.render(msg, True, C_TEXT)
        rect = text.get_rect(center=(gx + GRID_PIXELS // 2, gy + GRID_PIXELS // 2))
        self.screen.blit(text, rect)

    # -- Main loop ----------------------------------------------------

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if not self._handle_event(event):
                    running = False
            self._maybe_run_ai()
            self.screen.fill(C_BG)
            self._draw_grid()
            self._draw_sidebar()
            self._draw_winner_banner()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()
