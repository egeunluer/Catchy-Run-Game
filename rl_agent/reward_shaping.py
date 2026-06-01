from grid.catcher_vs_runner.engine import Agent, GameState, SPECIAL_MAJORITY, TURN_LIMIT


class RewardShaper:
    CAPTURE_BONUS = 0.24
    ALIVE_BONUS = 0.005
    CATCHER_DISTANCE_COEFF = 0.30
    CATCHER_PROXIMITY_COEFF = 0.02
    PROJECTILE_THREAT_COEFF = 0.25
    ATTRACTION_NEAREST = 0.01
    ATTRACTION_SECOND_NEAREST = 0.005
    SPRINT_WASTE_PENALTY = 0.02
    URGENCY_COEFF = 0.005
    DANGER_RADIUS = 2
    SAFE_ZONE_THRESHOLD = 3

    def __init__(self, trainee_role: Agent):
        self.trainee_role = trainee_role

    @staticmethod
    def _cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

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

    def shape(self, prev_state: GameState, curr_state: GameState, base_reward: float) -> float:
        if self.trainee_role != "runner" or curr_state.terminated:
            return base_reward
        return (base_reward
                + self._capture_bonus(prev_state, curr_state)
                + self.ALIVE_BONUS
                + self._catcher_distance_rewarding(curr_state, prev_state)
                + self._projectile_threat_penalty(curr_state)
                + self._special_attraction(curr_state)
                + self._sprint_waste_penalty(prev_state, curr_state)
                + self._urgency_penalty(curr_state))
