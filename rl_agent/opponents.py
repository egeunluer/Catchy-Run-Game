from catchy_run.catchy_run_game.agents import heuristic

def heuristic_opponent(state):
    if state.current_agent == "runner":
        return heuristic.runner_policy(state)
    else:
        return heuristic.catcher_policy(state)