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
    DIRECTIONS_8,
    MOVE_ACTIONS,
    SPECIAL_ACTIONS,
)
from ..agents.heuristic import catcher_policy, runner_policy
from ..agents.rl_runner import rl_runner_policy

# --- Layout --------------------------------------------------------------

CELL = 56
GRID_ORIGIN = (24, 24)
GRID_PIXELS = engine.BOARD_SIZE * CELL
SIDEBAR_X = GRID_ORIGIN[0] + GRID_PIXELS + 24
SIDEBAR_W = 320
WINDOW_W = SIDEBAR_X + SIDEBAR_W + 24
FPS = 60
AI_DELAY_MS = 350

BTN_H = 32
BTN_STEP = 38
NEW_GAME_H = 38
DIV_COLOR = (52, 56, 70)

Y_TITLE = 24
Y_STATUS = 62
Y_STATS_R = 90
Y_STATS_C = 108
Y_SPECIAL = 130
Y_DIV1 = 156
Y_HDR_ACT = 168
Y_BTN_MOVE = 198
Y_BTN_SPECIAL = Y_BTN_MOVE + BTN_STEP
Y_DIV2 = Y_BTN_SPECIAL + BTN_H + 14
Y_HDR_GAME = Y_DIV2 + 12
Y_BTN_GM_1 = Y_HDR_GAME + 28
Y_BTN_GM_2 = Y_BTN_GM_1 + BTN_STEP
Y_BTN_GM_3 = Y_BTN_GM_2 + BTN_STEP
Y_BTN_GM_4 = Y_BTN_GM_3 + BTN_STEP
Y_BTN_NEW = Y_BTN_GM_4 + BTN_STEP + 14
Y_FOOTER = Y_BTN_NEW + NEW_GAME_H + 16

SIDEBAR_BOTTOM = Y_FOOTER + 28
WINDOW_H = max(GRID_ORIGIN[1] + GRID_PIXELS + 24, SIDEBAR_BOTTOM)

# --- Colors --------------------------------------------------------------

C_BG = (24, 26, 34)
C_GRID = (60, 64, 78)
C_CELL = (40, 44, 56)
C_CELL_HL = (74, 96, 130)
C_RUNNER = (90, 168, 230)
C_CATCHER = (220, 96, 96)
C_PROJECTILE = (240, 210, 90)
C_SPECIAL = (200, 180, 60)
C_SPECIAL_CAPTURED = (110, 180, 110)
C_TEXT = (228, 230, 240)
C_TEXT_DIM = (140, 144, 160)
C_BTN = (52, 58, 76)
C_BTN_HOVER = (68, 78, 102)
C_BTN_ACTIVE = (84, 132, 188)
C_BANNER = (32, 36, 48)


# --- Modes ---------------------------------------------------------------

MODE_MOVE = "MOVE"
MODE_SPECIAL = "SPECIAL"
ACTION_MODES = [MODE_MOVE, MODE_SPECIAL]

GM_HVH = "HVH"
GM_AI_RUNNER = "AI_RUNNER"
GM_AI_CATCHER = "AI_CATCHER"
GM_AI_RUNNER_RL = "AI_RUNNER_RL"


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    key: str


# --- Action mapping ------------------------------------------------------

# Direction (dx, dy) -> 8-direction index (matches DIRECTIONS_8 ordering).
_DIR_TO_IDX_8: dict[tuple[int, int], int] = {d: i for i, d in enumerate(DIRECTIONS_8)}


def _cell_to_action(state: engine.GameState, mode: str, cell: tuple[int, int]) -> Optional[int]:
    """Translate a clicked cell + selected action mode to an action index.

    Returns None if the click does not correspond to a valid direction for
    the selected mode. Does not check legality — that is left to `engine.step`.
    """
    actor_pos = state.own_position()
    dx = cell[0] - actor_pos[0]
    dy = cell[1] - actor_pos[1]

    if mode == MODE_SPECIAL:
        if state.current_agent == "runner":
            # Sprint: 3-cell cardinal jump only.
            if dx == 0 and dy in (-3, 3):
                unit = (0, dy // 3)
            elif dy == 0 and dx in (-3, 3):
                unit = (dx // 3, 0)
            else:
                return None
            return SPECIAL_ACTIONS[_DIR_TO_IDX_8[unit]]
        # Catcher shoot: any 1-cell direction (including diagonals).
        if (dx, dy) in _DIR_TO_IDX_8:
            return SPECIAL_ACTIONS[_DIR_TO_IDX_8[(dx, dy)]]
        return None

    # Move: any 1-cell direction.
    if (dx, dy) not in _DIR_TO_IDX_8:
        return None
    idx = _DIR_TO_IDX_8[(dx, dy)]
    if mode == MODE_MOVE:
        return MOVE_ACTIONS[idx]
    return None


def _legal_target_cells(state: engine.GameState, mode: str) -> set[tuple[int, int]]:
    """Cells the user could click in the current state under `mode`."""
    if state.terminated:
        return set()
    actor = state.own_position()
    mask = engine.legal_action_mask(state)
    cells: set[tuple[int, int]] = set()

    def add_8(actions: tuple[int, ...], distance: int = 1) -> None:
        for action, (dx, dy) in zip(actions, DIRECTIONS_8):
            if mask[action]:
                cells.add((actor[0] + dx * distance, actor[1] + dy * distance))

    if mode == MODE_MOVE:
        add_8(MOVE_ACTIONS)
    elif mode == MODE_SPECIAL:
        if state.current_agent == "runner":
            # Sprint: cardinal-only at distance 3; diagonals are illegal so
            # the legality mask filters them out automatically.
            add_8(SPECIAL_ACTIONS, distance=3)
        else:
            # Catcher shoot: highlight where the bullet lands after the
            # post-fire tick (catcher_pos + dir).
            add_8(SPECIAL_ACTIONS)
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
        self.ai_pending_at: Optional[int] = None

        self.buttons: list[Button] = self._build_buttons()

    def _build_buttons(self) -> list[Button]:
        buttons: list[Button] = []
        action_ys = [Y_BTN_MOVE, Y_BTN_SPECIAL]
        for mode, y in zip(ACTION_MODES, action_ys):
            buttons.append(Button(
                pygame.Rect(SIDEBAR_X, y, SIDEBAR_W, BTN_H), mode.title(),
                f"mode:{mode}",
            ))
        gm_entries = [
            (Y_BTN_GM_1, "Human vs. Human", f"gm:{GM_HVH}"),
            (Y_BTN_GM_2, "Human catcher (AI runner)", f"gm:{GM_AI_RUNNER}"),
            (Y_BTN_GM_3, "Human runner (AI catcher)", f"gm:{GM_AI_CATCHER}"),
            (Y_BTN_GM_4, "Human catcher (RL runner)", f"gm:{GM_AI_RUNNER_RL}"),
        ]
        for y, label, key in gm_entries:
            buttons.append(Button(
                pygame.Rect(SIDEBAR_X, y, SIDEBAR_W, BTN_H), label, key,
            ))
        buttons.append(Button(
            pygame.Rect(SIDEBAR_X, Y_BTN_NEW, SIDEBAR_W, NEW_GAME_H), "New Game", "new",
        ))
        return buttons

    def _cell_at(self, mx: int, my: int) -> Optional[tuple[int, int]]:
        gx, gy = GRID_ORIGIN
        if not (gx <= mx < gx + GRID_PIXELS and gy <= my < gy + GRID_PIXELS):
            return None
        return ((mx - gx) // CELL, (my - gy) // CELL)

    def _cell_rect(self, cell: tuple[int, int]) -> pygame.Rect:
        gx, gy = GRID_ORIGIN
        return pygame.Rect(gx + cell[0] * CELL, gy + cell[1] * CELL, CELL, CELL)

    def _ai_role(self) -> Optional[str]:
        if self.game_mode in (GM_AI_RUNNER, GM_AI_RUNNER_RL):
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
            return
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
            if self.game_mode == GM_AI_RUNNER_RL:
                try:
                    action = rl_runner_policy(self.state, self.rng)
                except Exception as e:
                    print(f"[pygame_app] RL runner failed: {e}\nFalling back to HVH.")
                    self.game_mode = GM_HVH
                    self.ai_pending_at = None
                    return
            else:
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

    def _on_button(self, key: str) -> None:
        if key == "new":
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
                self.mode = MODE_SPECIAL
        return True

    def _draw_grid(self) -> None:
        gx, gy = GRID_ORIGIN
        for cy in range(engine.BOARD_SIZE):
            for cx in range(engine.BOARD_SIZE):
                rect = pygame.Rect(gx + cx * CELL, gy + cy * CELL, CELL, CELL)
                pygame.draw.rect(self.screen, C_CELL, rect)
                pygame.draw.rect(self.screen, C_GRID, rect, 1)

        for sq in self.state.special_squares:
            color = C_SPECIAL_CAPTURED if sq in self.state.captured_squares else C_SPECIAL
            self._draw_special_square(sq, color)

        if not self.state.terminated and not self._is_ai_turn():
            targets = _legal_target_cells(self.state, self.mode)
            for cell in targets:
                rect = self._cell_rect(cell)
                pygame.draw.rect(self.screen, C_CELL_HL, rect)
                pygame.draw.rect(self.screen, C_GRID, rect, 1)

        self._draw_agent(self.state.runner_pos, C_RUNNER, "R")
        self._draw_agent(self.state.catcher_pos, C_CATCHER, "C")

        self._draw_projectiles()

    def _draw_special_square(
        self, cell: tuple[int, int], color: tuple[int, int, int]
    ) -> None:
        rect = self._cell_rect(cell)
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, C_GRID, rect, 1)
        text = self.font_lg.render("*", True, C_BG)
        self.screen.blit(text, text.get_rect(center=rect.center))

    def _draw_agent(self, cell: tuple[int, int], color: tuple[int, int, int], label: str) -> None:
        rect = self._cell_rect(cell)
        center = rect.center
        pygame.draw.circle(self.screen, color, center, CELL // 2 - 8)
        text = self.font_lg.render(label, True, (255, 255, 255))
        self.screen.blit(text, text.get_rect(center=center))

    def _draw_projectiles(self) -> None:
        per_cell: dict[tuple[int, int], int] = {}
        for (pos, _) in self.state.projectiles:
            per_cell[pos] = per_cell.get(pos, 0) + 1
        radius = max(4, CELL // 7)
        for cell, count in per_cell.items():
            rect = self._cell_rect(cell)
            cx, cy = rect.center
            offsets = [(0, 0)] if count == 1 else [
                (-radius, -radius), (radius, -radius),
                (-radius, radius), (radius, radius),
            ][:count]
            for ox, oy in offsets:
                pygame.draw.circle(self.screen, C_PROJECTILE, (cx + ox, cy + oy), radius)

    def _draw_sidebar(self) -> None:
        x = SIDEBAR_X

        self.screen.blit(self.font_lg.render("Catcher vs. Runner", True, C_TEXT), (x, Y_TITLE))

        if self.state.terminated:
            winner = self.state.winner or "?"
            status = f"Game over — {winner} wins"
            color = C_RUNNER if winner == "runner" else C_CATCHER
        else:
            mover = self.state.current_agent
            color = C_RUNNER if mover == "runner" else C_CATCHER
            ai_tag = " (AI)" if self._ai_role() == mover else ""
            status = f"Turn {self.state.turn}/{engine.TURN_LIMIT} — {mover}{ai_tag}'s move"
        self.screen.blit(self.font.render(status, True, color), (x, Y_STATUS))

        bullets = len(self.state.projectiles)
        r_line = (
            f"Runner @ {self.state.runner_pos}   {self.state.sprint_charges} sprints left"
        )
        c_line = f"Catcher @ {self.state.catcher_pos}   {bullets} bullets in flight "
        self.screen.blit(self.font_sm.render(r_line, True, C_RUNNER), (x, Y_STATS_R))
        self.screen.blit(self.font_sm.render(c_line, True, C_CATCHER), (x, Y_STATS_C))

        special_label = (
            "Sprint (3 cells, cardinal)"
            if self.state.current_agent == "runner"
            else "Shoot (8 directions, unlimited)"
        )
        self.screen.blit(
            self.font_sm.render(f"Special: {special_label}", True, C_TEXT_DIM),
            (x, Y_SPECIAL),
        )

        pygame.draw.line(self.screen, DIV_COLOR, (x, Y_DIV1), (x + SIDEBAR_W, Y_DIV1), 1)
        self.screen.blit(self.font.render("Action", True, C_TEXT), (x, Y_HDR_ACT))

        pygame.draw.line(self.screen, DIV_COLOR, (x, Y_DIV2), (x + SIDEBAR_W, Y_DIV2), 1)
        self.screen.blit(self.font.render("Game mode", True, C_TEXT), (x, Y_HDR_GAME))

        mouse = pygame.mouse.get_pos()
        for btn in self.buttons:
            self._draw_button(btn, mouse)

        self.screen.blit(
            self.font_sm.render("Keys: 1-2 modes  ·  R new game", True, C_TEXT_DIM),
            (x, Y_FOOTER),
        )

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
