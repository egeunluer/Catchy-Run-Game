from catchy_run.catchy_run_game.engine import GameState, SPECIAL_MAJORITY, TURN_LIMIT, BOARD_SIZE
from catchy_run.catchy_run_game.actions import DIRECTIONS_8, CARDINAL_DIRS


class RewardShaper:
    @staticmethod
    def _cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def shape(self, prev_state: GameState, curr_state: GameState, base_reward: float) -> float:
        if curr_state.terminated:
            return base_reward
        return self._compute(prev_state, curr_state, base_reward)

    def _compute(self, prev_state: GameState, curr_state: GameState, base_reward: float) -> float:
        raise NotImplementedError


class RunnerRewardShaper(RewardShaper):
    CAPTURE_BONUS = 0.24
    ALIVE_BONUS = 0.005
    UNSAFE_CAPTURE_PENALTY = 0.25
    CATCHER_DISTANCE_COEFF = 0.30
    CATCHER_PROXIMITY_COEFF = 0.02
    PROJECTILE_THREAT_COEFF = 0.25
    ATTRACTION_NEAREST = 0.01
    ATTRACTION_SECOND_NEAREST = 0.005
    SPRINT_WASTE_PENALTY = 0.02
    URGENCY_COEFF = 0.005
    DANGER_RADIUS = 2
    SAFE_ZONE_THRESHOLD = 3

    def _capture_bonus(self, prev: GameState, curr: GameState) -> float:
        newly_captured = len(curr.captured_squares) - len(prev.captured_squares)
        return self.CAPTURE_BONUS * newly_captured

    def _catcher_distance_rewarding(self, curr: GameState, prev: GameState) -> float:
        runner_move_dist = self._cheb(curr.runner_pos, prev.catcher_pos)
        delta_dist = runner_move_dist - self._cheb(prev.runner_pos, prev.catcher_pos)
        current_dist = self._cheb(curr.runner_pos, curr.catcher_pos)
        if current_dist <= self.DANGER_RADIUS:
            if delta_dist <= 0:
                return -self.CATCHER_DISTANCE_COEFF / current_dist
            return -self.CATCHER_PROXIMITY_COEFF / current_dist
        else:
            return -self.CATCHER_PROXIMITY_COEFF / current_dist


    def _projectile_threat_penalty(self, curr: GameState) -> float:
        penalty = 0.0
        for (px, py), (dx, dy) in curr.projectiles:
            next_cell = (px + dx, py + dy)
            next2_cell = (px + 2 * dx, py + 2 * dy)
            d1 = self._cheb(curr.runner_pos, next_cell)
            d2 = self._cheb(curr.runner_pos, next2_cell)
            penalty -= self.PROJECTILE_THREAT_COEFF / max(1, min(d1, d2))
        return penalty

    def _special_attraction(self, curr: GameState) -> float:
        remaining = curr.special_squares - curr.captured_squares
        safe = [
            s for s in remaining
            if self._cheb(s, curr.catcher_pos) > self.DANGER_RADIUS
        ]
        if not safe:
            return 0.0
        safe.sort(key=lambda s: self._cheb(s, curr.runner_pos))
        reward = self.ATTRACTION_NEAREST / max(1, self._cheb(safe[0], curr.runner_pos))
        if len(safe) >= 2:
            reward += self.ATTRACTION_SECOND_NEAREST / max(1, self._cheb(safe[1], curr.runner_pos))
        return reward

    def _sprint_waste_penalty(self, prev: GameState, curr: GameState) -> float:
        sprint_used = prev.sprint_charges > curr.sprint_charges
        in_safe_zone = self._cheb(curr.runner_pos, curr.catcher_pos) > self.SAFE_ZONE_THRESHOLD
        if sprint_used and in_safe_zone:
            return -self.SPRINT_WASTE_PENALTY
        return 0.0

    def _urgency_penalty(self, curr: GameState) -> float:
        shortfall = SPECIAL_MAJORITY - len(curr.captured_squares)
        if shortfall <= 0:
            return 0.0
        turns_elapsed = curr.turn / TURN_LIMIT
        return -self.URGENCY_COEFF * shortfall * turns_elapsed

    def _unsafe_capture_penalty(self, prev: GameState, curr: GameState) -> float:
        #The model before adding the unsafe capture penalty: catchy_run_runner_stage0_v1_4_0
        newly_captured = len(curr.captured_squares) - len(prev.captured_squares)
        if newly_captured > 0 and self._cheb(curr.runner_pos, prev.catcher_pos) <= 1:
            return -self.UNSAFE_CAPTURE_PENALTY
        return 0.0

    def _compute(self, prev_state: GameState, curr_state: GameState, base_reward: float) -> float:
        return (base_reward
                + self._capture_bonus(prev_state, curr_state)
                + self.ALIVE_BONUS
                + self._catcher_distance_rewarding(curr_state, prev_state)
                + self._projectile_threat_penalty(curr_state)
                + self._special_attraction(curr_state)
                + self._sprint_waste_penalty(prev_state, curr_state)
                + self._urgency_penalty(curr_state)
                + self._unsafe_capture_penalty(prev_state, curr_state))


class CatcherRewardShaper(RewardShaper):
    CAPTURE_BLOCK_PENALTY = 0.20
    DISTANCE_CLOSURE_COEFF = 0.02
    BULLET_COVERAGE_COEFF = 0.10
    BULLET_COVERAGE_CAP = 3
    SPECIAL_DEFENSE_COEFF = 0.10
    SPECIAL_DEFENSE_NEAR_MAX = 3
    SPECIAL_DEFENSE_FAR_MIN = 4
    SPECIAL_DEFENSE_FAR_MAX = 5
    BULLET_SPAM_PENALTY = 0.1

    @staticmethod
    def _runner_reaches_in_one(state: GameState, target: tuple[int, int]) -> bool:
        runner = state.runner_pos
        catcher = state.catcher_pos
        if max(abs(runner[0] - target[0]), abs(runner[1] - target[1])) <= 1:
            return True
        if state.sprint_charges <= 0:
            return False
        dx = target[0] - runner[0]
        dy = target[1] - runner[1]
        if not ((dx == 0 and abs(dy) == 3) or (dy == 0 and abs(dx) == 3)):
            return False
        step_dx = 0 if dx == 0 else (1 if dx > 0 else -1)
        step_dy = 0 if dy == 0 else (1 if dy > 0 else -1)
        one_ahead = (runner[0] + step_dx, runner[1] + step_dy)
        if one_ahead == catcher or target == catcher:
            return False
        return True

    @staticmethod
    def _runner_reaches_in_exactly_two(state: GameState, target: tuple[int, int]) -> bool:
        if CatcherRewardShaper._runner_reaches_in_one(state, target):
            return False
        runner = state.runner_pos
        catcher = state.catcher_pos
        specials_open = state.special_squares - state.captured_squares

        candidates: list[tuple[tuple[int, int], int]] = []
        for sdx, sdy in DIRECTIONS_8:
            mid = (runner[0] + sdx, runner[1] + sdy)
            if not (0 <= mid[0] < BOARD_SIZE and 0 <= mid[1] < BOARD_SIZE):
                continue
            charges_after = state.sprint_charges + (1 if mid in specials_open else 0)
            candidates.append((mid, charges_after))
        if state.sprint_charges > 0:
            for sdx, sdy in CARDINAL_DIRS:
                mid = (runner[0] + 3 * sdx, runner[1] + 3 * sdy)
                one_ahead = (runner[0] + sdx, runner[1] + sdy)
                if not (0 <= mid[0] < BOARD_SIZE and 0 <= mid[1] < BOARD_SIZE):
                    continue
                if one_ahead == catcher or mid == catcher:
                    continue
                charges_after = state.sprint_charges - 1 + (1 if mid in specials_open else 0)
                candidates.append((mid, charges_after))

        for mid, charges_after in candidates:
            if max(abs(mid[0] - target[0]), abs(mid[1] - target[1])) <= 1:
                return True
            if charges_after <= 0:
                continue
            dxt = target[0] - mid[0]
            dyt = target[1] - mid[1]
            if not ((dxt == 0 and abs(dyt) == 3) or (dyt == 0 and abs(dxt) == 3)):
                continue
            sx = 0 if dxt == 0 else (1 if dxt > 0 else -1)
            sy = 0 if dyt == 0 else (1 if dyt > 0 else -1)
            one_ahead = (mid[0] + sx, mid[1] + sy)
            if one_ahead == catcher or target == catcher:
                continue
            return True
        return False

    def _capture_block_penalty(self, prev: GameState, curr: GameState) -> float:
        newly_captured = len(curr.captured_squares) - len(prev.captured_squares)
        return -self.CAPTURE_BLOCK_PENALTY * newly_captured

    def _distance_closure_bonus(self, curr: GameState) -> float:
        return self.DISTANCE_CLOSURE_COEFF / max(1, self._cheb(curr.runner_pos, curr.catcher_pos))

    def _bullet_coverage_bonus(self, curr: GameState) -> float:
        if not curr.projectiles:
            return 0.0
        scores = []
        for (px, py), (dx, dy) in curr.projectiles:
            next_cell = (px + dx, py + dy)
            next2_cell = (px + 2 * dx, py + 2 * dy)
            d1 = self._cheb(curr.runner_pos, next_cell)
            d2 = self._cheb(curr.runner_pos, next2_cell)
            scores.append(self.BULLET_COVERAGE_COEFF / max(1, min(d1, d2)))
        scores.sort(reverse=True)
        return sum(scores[:self.BULLET_COVERAGE_CAP])

    @staticmethod
    def _newly_fired_bullet(prev: GameState, curr: GameState):
        expected = set()
        for (px, py), (dx, dy) in prev.projectiles:
            nxt = (px + dx, py + dy)
            if 0 <= nxt[0] < BOARD_SIZE and 0 <= nxt[1] < BOARD_SIZE:
                expected.add((nxt, (dx, dy)))
        for entry in curr.projectiles:
            if entry not in expected:
                return entry
        return None

    def _special_defense_bonus(self, prev: GameState, curr: GameState) -> float:
        new_bullet = self._newly_fired_bullet(prev, curr)
        if new_bullet is None:
            return 0.0
        _, (dx, dy) = new_bullet
        cx, cy = prev.catcher_pos
        remaining = curr.special_squares - curr.captured_squares

        if remaining:
            for k in range(1, self.SPECIAL_DEFENSE_NEAR_MAX + 1):
                cell = (cx + k * dx, cy + k * dy)
                if not (0 <= cell[0] < BOARD_SIZE and 0 <= cell[1] < BOARD_SIZE):
                    break
                if cell in remaining and self._runner_reaches_in_one(prev, cell):
                    return self.SPECIAL_DEFENSE_COEFF

            for k in range(self.SPECIAL_DEFENSE_FAR_MIN, self.SPECIAL_DEFENSE_FAR_MAX + 1):
                cell = (cx + k * dx, cy + k * dy)
                if not (0 <= cell[0] < BOARD_SIZE and 0 <= cell[1] < BOARD_SIZE):
                    break
                if cell in remaining and self._runner_reaches_in_exactly_two(prev, cell):
                    return self.SPECIAL_DEFENSE_COEFF

        return -self.BULLET_SPAM_PENALTY

    def _compute(self, prev_state: GameState, curr_state: GameState, base_reward: float) -> float:
        return (base_reward
                + self._capture_block_penalty(prev_state, curr_state)
                + self._distance_closure_bonus(curr_state)
                + self._bullet_coverage_bonus(curr_state)
                + self._special_defense_bonus(prev_state, curr_state))
