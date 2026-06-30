# Training Guide

End-to-end plan for training the runner agent. The catcher is always a fixed
heuristic or pre-trained opponent — it is not trained.

Training escalates in opponent strength: random → heuristic → fixed-RL → pool.

---

## Where am I right now?

Tick boxes as you progress:

- [ ] **Stage 0** — explore the grid against a 20% random / 80% defensive-shooter pool; learn to capture squares under light pressure
- [ ] **Stage 1** — train against the heuristic catcher + defensive shooter
- [ ] **Stage 2** — train against the best available fixed catcher checkpoint
- [ ] **Stage 3** — train against a pool of past runner snapshots + heuristic (league)

---

## Architecture summary

```
rl_agent/
  environment.py      CatchyRunEnv — single-agent Gym env with the catcher
                      opponent played inline inside step(). trainee_role is
                      always "runner".
  opponents.py        heuristic_opponent, defensive_shooter_opponent — adapt
                      the bundled heuristic agents to the (state) -> int
                      contract the env expects. make_rl_opponent can load a
                      fixed checkpoint for Stage 2+.
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

## Stage 0 — Runner vs. mixed pool (exploration)

**Goal:** the runner learns the *objective* under light catcher pressure. The 20% random episodes give low-stakes opportunities to stumble onto specials and learn the capture loop; the 80% defensive-shooter episodes apply just enough purposeful bullet pressure that the policy learns to dodge meaningful shots rather than random spray. By the end of Stage 0 the runner should reliably head for special squares and capture (not just hover adjacent to) them.

### How it works

The env exposes a weighted opponent pool via `set_opponent_pool(opponents, weights)`. The pool is sampled **per episode** in `reset()`, so the opponent is fixed for the duration of an episode.

```python
def make_env(trainee_role: Agent = "runner"):
    env = CatchyRunEnv(trainee_role=trainee_role)
    env.set_opponent_pool(
        [heuristic_opponent, defensive_shooter_opponent],
        weights=[0.2, 0.8],
    )
    env = ActionMasker(env, mask_fn)
    return env
```

Train fresh — no `load_from`. Any prior checkpoint was fit to a different shaping/opponent regime and would carry over a biased value function.

### Run

```bash
python -m rl_agent.model
```

Budget: 300k steps. Extend to 500k–1M if the capture rate hasn't crossed the 4-square win threshold.

### Move-on criteria

- [ ] Runner captures **≥4 special squares** in the majority of episodes (the on-timeout win threshold)
- [ ] Win rate against the pool stabilizes at **≥70%**
- [ ] Episode length trends *down* over training — the runner is finishing faster, not stalling
- [ ] Behavioral check: run `trace_rewards.py` against the checkpoint and confirm the runner is *capturing* specials (capture component fires regularly), not loitering adjacent to them to farm the attraction term

### Common failures

- **Capture rate plateaus at ~3** with episode length near the 40-turn cap → policy is exploiting shaping (attraction farming or alive-bonus farming). Inspect with `trace_rewards.py` and rebalance `ATTRACTION_NEAREST` vs `CAPTURE_BONUS` in `reward_shaping.py`.
- **`ep_rew_mean` stuck near 0** → the runner isn't finding specials. Temporarily lower the defensive-shooter weight so the runner sees more low-pressure episodes; once captures appear, restore it.
- **Capture rate near 0 with very short episodes** → the catcher is killing the runner before it learns. Same fix: lower defensive-shooter weight temporarily.
- **Entropy stuck high (~`ln(n_legal)`) with flat losses** → no signal is reaching the policy. Re-check `_shape_reward` and the mask threading before extending the run.

### Output

`catchy_run_runner_stage0.zip` — seed for Stage 1.

---

## Stage 1 — Skill floor

**Goal:** establish a skill floor against the bundled heuristic catcher. Load the Stage 0 checkpoint and switch to the default heuristic pool.

### Setup

```python
train(load_from="catchy_run_runner_stage0",
      save_to="catchy_run_runner_stage1",
      tb_log_name="runner_stage1",
      ent_coef=0.005)   # lower entropy so it refines instead of thrashing
```

### Monitor

```bash
tensorboard --logdir ./tb_logs/
```

| Metric | Group | Healthy signal |
|---|---|---|
| `ep_rew_mean` | rollout | Drifts from ~0 toward positive. Plateau = ceiling vs. heuristic. |
| `ep_len_mean` | rollout | Stabilizes as policy commits to a strategy. |
| `entropy_loss` | train | Decreases *slowly*. Sudden collapse → bump `ent_coef`. |
| `clip_fraction` | train | 0.1–0.3 healthy. >0.5 → drop `learning_rate`. |
| `approx_kl` | train | <0.02 fine. >0.05 → updates too aggressive. |

### Move-on criteria

- [ ] Win rate vs. heuristic plateaus at **≥70%** for several evaluations
- [ ] Plateau holds for ~50k–100k steps with no further improvement
- [ ] Policy entropy has stabilized (not collapsed to zero, not still drifting)

200k steps is the initial budget. If a plateau hasn't been reached, extend to 500k or 1M.

### Output

`catchy_run_runner_stage1.zip` — seed for Stage 2.

---

## Stage 2 — Fixed RL catcher

**Status:** not yet implemented.

**Goal:** climb past the heuristic ceiling by training against the best available pre-trained catcher checkpoint (`trained_model_checkpoints/catcher_models/`). The catcher is frozen — not retrained — so this is one-sided improvement.

### Setup

```python
from rl_agent.opponents import make_rl_opponent

fixed_catcher = make_rl_opponent(
    "trained_model_checkpoints/catcher_models/catchy_run_catcher_stage1_v1.zip",
    role="catcher", deterministic=False
)
env.set_opponent_pool([heuristic_opponent, defensive_shooter_opponent, fixed_catcher],
                      weights=[0.1, 0.4, 0.5])
```

### Move-on criteria

- [ ] Win rate vs. **heuristic** holds at ≥80% (regression check)
- [ ] Win rate vs. **fixed RL catcher** stabilizes at ≥60%

### Output

`catchy_run_runner_stage2.zip` — seed for Stage 3.

---

## Stage 3 — Opponent pool (league)

**Status:** not yet implemented.

**Goal:** prevent overfitting to one fixed opponent by training against a *mix* of past runner snapshots of the runner itself alongside the fixed catchers. This is runner self-play within the opponent pool.

The env already supports this via `set_opponent_pool` — no env changes needed.

### Move-on criteria

- [ ] Win rate vs. **heuristic** stays at ≥80%
- [ ] Win rate vs. **every pool member** is roughly balanced

### Output

`catchy_run_runner_final.zip`.

---

## TensorBoard cheatsheet

```bash
# Start
tensorboard --logdir ./tb_logs/

# Different port if 6006 is taken
tensorboard --logdir ./tb_logs/ --port 6007
```

- Each `model.learn(...)` call creates a subdirectory `<name>_N/`. Use distinct `tb_log_name` per run so runs don't pile into the same line.
- Smoothing slider (~0.6) is essential for `ep_rew_mean` — raw curve is jittery.

---

## Hyperparameter notes

Current values in `model.py`:

| Param | Value | Reason |
|---|---|---|
| `learning_rate` | `3e-4` | PPO standard. Drop to `1e-4` when continuing from a checkpoint (Stage 1+). |
| `n_steps` | `4096` | ~100 episodes per rollout at 40-turn max. |
| `batch_size` | `256` | Fine for network size. |
| `gamma` | `0.99` | Discount. Short episodes mean this barely matters. |
| `gae_lambda` | `0.95` | GAE smoothing. Default. |
| `clip_range` | `0.2` | PPO clip. Default. |
| `ent_coef` | `0.05` | Entropy bonus. Drop to `~0.005` once a stage's policy is moving. Bump to `0.05` if the policy collapses early. |

Don't tune anything before a stage run has clearly failed.

---

## Quick troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ep_rew_mean` flat at 0 (Stage 0) | Runner not finding special squares | Bump capture shaping to `0.2` |
| `ep_rew_mean` flat at 0 (Stage 1) | Mask not reaching policy, or opponent broken | Verify `ActionMasker` is wrapping the env; print the mask inside `mask_fn` |
| `ep_rew_mean` strongly negative | Reward sign flipped, or opponent way too strong | Check `_play_opponent_turn` reward extraction; consider warming up with a weaker opponent |
| `entropy_loss` near 0 within 10k steps | Premature commitment | Bump `ent_coef` to `0.05` |
| Entropy stays near max with losses flat | No reward signal reaching the policy | Stop the run; check shaping and opponent strength |
| `approx_kl` repeatedly >0.1 | Updates too aggressive | Lower `learning_rate` to `1e-4` or `n_epochs` to 5 |
| Training crashes with shape mismatch | `observation_space` not matching `_obs()` output | Confirm both are `(9, 7, 7)` |

---

## File map at a glance

| File | Purpose |
|---|---|
| `rl_agent/environment.py` | The Gym env. `trainee_role` is always `"runner"`; binds `RunnerRewardShaper`. |
| `rl_agent/reward_shaping.py` | `RewardShaper` base + `RunnerRewardShaper`. |
| `rl_agent/opponents.py` | Opponent callables: `heuristic_opponent`, `defensive_shooter_opponent`, `make_rl_opponent` (for loading fixed catcher checkpoints in Stage 2+). |
| `rl_agent/custom_cnn.py` | CNN feature extractor + `policy_kwargs`. Sized for 9×7×7. |
| `rl_agent/model.py` | Training entry point. |
| `catchy_run_game/engine.py` | Pure game logic. Do not modify for RL needs. |
| `tb_logs/` | TensorBoard event files (created on first run). |
| `trained_model_checkpoints/runner_models/` | Runner model checkpoints. |
| `trained_model_checkpoints/catcher_models/` | Pre-trained catcher checkpoints (frozen; used as fixed opponents in Stage 2+). |
