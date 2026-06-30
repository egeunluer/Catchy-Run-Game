# Training Guide

End-to-end plan for training the runner agent. The catcher is always a fixed
heuristic opponent — it is not trained.

Training escalates in opponent strength across three stages:
random/heuristic → heuristic/shooting → shooting only.

---

## Where am I right now?

Tick boxes as you progress:

- [ ] **Stage 0** — explore the grid against a 70% random / 30% heuristic pool; learn to capture squares
- [ ] **Stage 1** — refine against a heuristic / shooting-heuristic pool
- [ ] **Stage 2** — sharpen against the shooting heuristic only

---

## Architecture summary

```
rl_agent/
  environment.py      CatchyRunEnv — single-agent Gym env with the catcher
                      opponent played inline inside step(). trainee_role is
                      always "runner".
  opponents.py        heuristic_opponent, defensive_shooter_opponent — thin
                      wrappers that adapt catchy_run_game.agents.heuristic
                      to the (state) -> int contract the env expects.
  custom_cnn.py       CustomGridCNN feature extractor (3× Conv2d + linear),
                      sized for the 9×7×7 observation tensor.
  model.py            MaskablePPO + CnnPolicy + custom extractor. Entry point
                      for training.
```

Key design choices:
- **One network.** The runner is the only trained agent.
- **Algorithm:** `MaskablePPO` from `sb3-contrib` — handles action masking via `info["action_mask"]`.
- **Observation:** `(9, 7, 7)` — the engine's 9 channels, runner perspective.
- **Reward:** sparse ±1 on termination plus per-step shaping from `RunnerRewardShaper` in `rl_agent/reward_shaping.py`. Nine components: capture bonus, alive bonus, catcher-distance, projectile threat, special-attraction, sprint-waste, urgency, unsafe-capture, plus the base engine reward. All magnitudes are tuned so the engine's terminal ±1 still dominates. See `rl_agent/environment_explanations/reward_shaping.md` for per-component derivations.
- **Opponent strategy:** catcher plays inline inside `env.step`, so SB3 sees a normal single-agent env.

---

## Stage 0 — Exploration

**Goal:** the runner learns the *objective* with minimal pressure. The 70% random episodes give low-stakes opportunities to stumble onto specials and learn the capture loop; the 30% heuristic episodes add just enough chasing pressure to prevent the policy from ignoring the catcher entirely. By the end of Stage 0 the runner should reliably head for special squares and capture (not just hover adjacent to) them.

### Setup

```python
def make_env(trainee_role: Agent = "runner"):
    env = CatchyRunEnv(trainee_role=trainee_role)
    env.set_opponent_pool(
        [env._default_opponent, heuristic_opponent],
        weights=[0.7, 0.3],
    )
    env = ActionMasker(env, mask_fn)
    return env
```

`env._default_opponent` is the built-in random catcher (70% random moves, 30% random shots). Train fresh — no `load_from`.

### Run

```bash
python -m rl_agent.model
```

### Output

`catchy_run_runner_stage0.zip` — seed for Stage 1.

---

## Stage 1 — Skill floor

**Goal:** raise the skill floor against a smarter, bullet-firing catcher. Transitions from pure exploration to refined evasion: the runner must now dodge purposeful defensive shots, not just random ones.

### Setup

Load the Stage 0 checkpoint and switch the opponent pool to heuristic + shooting heuristic:

```python
def make_env(trainee_role: Agent = "runner"):
    env = CatchyRunEnv(trainee_role=trainee_role)
    env.set_opponent_pool(
        [heuristic_opponent, defensive_shooter_opponent],
        weights=[0.5, 0.5],
    )
    env = ActionMasker(env, mask_fn)
    return env

train(load_from="catchy_run_runner_stage0",
      save_to="catchy_run_runner_stage1",
      tb_log_name="runner_stage1",
      ent_coef=0.005)
```

Lower entropy keeps the runner refining its Stage 0 strategy rather than re-exploring.

### Output

`catchy_run_runner_stage1.zip` — seed for Stage 2.

---

## Stage 2 — Refinement (final)

**Goal:** sharpen the policy to its ceiling against the strongest heuristic opponent. The runner trains exclusively against the shooting heuristic catcher — no easier fallback.

### Setup

Load the Stage 1 checkpoint and switch to shooting heuristic only:

```python
def make_env(trainee_role: Agent = "runner"):
    env = CatchyRunEnv(trainee_role=trainee_role)
    env.set_opponent_pool(
        [defensive_shooter_opponent],
        weights=[1.0],
    )
    env = ActionMasker(env, mask_fn)
    return env

train(load_from="catchy_run_runner_stage1",
      save_to="catchy_run_runner_stage2",
      tb_log_name="runner_stage2",
      ent_coef=0.002,
      learning_rate=1e-4)
```

Lower learning rate and entropy for fine-grained convergence.

### Output

`catchy_run_runner_final.zip`.
