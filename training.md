# Training Guide

End-to-end plan for training a self-play agent on Catcher vs. Runner.

The same network plays both roles. Training happens in three stages of escalating opponent strength: heuristic → frozen self → league of past snapshots.

---

## Where am I right now?

Tick boxes as you progress:

- [ ] **Stage 1** — train against the bundled heuristic agent
- [ ] **Stage 2** — train against a frozen snapshot of the current policy
- [ ] **Stage 3** — train against a sampled pool of past snapshots + heuristic

---

## Architecture summary

```
rl_agent/
  environment.py      CatchyRunEnv — single-agent Gym env with the opponent
                      played inline inside step(). Trainee role is randomized
                      per episode and exposed via channel 0 of the observation.
  opponents.py        heuristic_opponent — adapts the bundled heuristic agent
                      to the (state) -> int contract the env expects.
  custom_cnn.py       CustomGridCNN feature extractor (3× Conv2d + linear),
                      sized for the 10×7×7 observation tensor.
  model.py            MaskablePPO + CnnPolicy + custom extractor. Entry point
                      for training.
```

Key design choices already made:
- **Algorithm:** `MaskablePPO` from `sb3-contrib` — handles action masking via `info["action_mask"]`.
- **Observation:** `(10, 7, 7)` — engine's 9 channels plus an explicit role indicator (channel 0).
- **Reward:** sparse ±1 on termination, sign flipped to match the trainee's role.
- **Self-play strategy:** opponent plays inline inside `env.step`, so SB3 sees a normal single-agent env.

---

## Stage 1 — Heuristic opponent

**Goal:** establish a skill floor. The bundled heuristic gives a non-trivial, stationary target for the policy to climb to before we introduce the moving target of self-play.

### Run

From the project root:

```bash
python -m rl_agent.model
```

In a second terminal:

```bash
tensorboard --logdir ./tb_logs/
```

Open `http://localhost:6006` in a browser.

### What to watch on TensorBoard

| Metric | Group | Healthy signal |
|---|---|---|
| `ep_rew_mean` | rollout | Drifts from ~0 toward positive. Plateau = ceiling vs. heuristic. |
| `ep_len_mean` | rollout | Stabilizes as policy commits to a strategy. |
| `entropy_loss` | train | Decreases *slowly*. Sudden collapse → bump `ent_coef`. |
| `clip_fraction` | train | 0.1–0.3 healthy. >0.5 → drop `learning_rate`. |
| `approx_kl` | train | <0.02 fine. >0.05 → updates too aggressive. |

### Move-on criteria

Stage 1 is complete when **all** of these are true:

- [ ] Win rate vs. heuristic plateaus at **≥70%** for several evaluations
- [ ] Plateau holds for ~50k–100k steps with no further improvement
- [ ] Policy entropy has stabilized (not collapsed to zero, not still drifting)

200k steps is the initial budget. If the plateau hasn't been reached, extend to 500k or 1M. If `ep_rew_mean` hasn't moved off zero after 50k steps, **stop and debug** — don't keep burning compute. Common causes:
- Mask not threading through `ActionMasker` correctly
- Reward sign flipped
- Opponent function broken
- Observation/role channel misaligned

### Output

`catchy_run_stage1.zip` — the trained model. Used as the seed snapshot for Stage 2.

---

## Stage 2 — Frozen self-play

**Status:** not yet implemented.

**Goal:** climb past the heuristic ceiling by training against ever-improving copies of yourself.

### What needs to be built

1. **Snapshot loader** in `opponents.py`:
   ```python
   def make_snapshot_opponent(model_path: str):
       model = MaskablePPO.load(model_path)
       def opponent(state):
           obs = build_obs_for_opponent(state)        # use opponent's perspective
           mask = engine.legal_action_mask(state)
           action, _ = model.predict(obs, action_masks=mask, deterministic=True)
           return int(action)
       return opponent
   ```

2. **`SelfPlayCallback`** — periodically snapshots the live policy and pushes it into the env as the new opponent. Fires every ~20k steps:
   ```python
   class SelfPlayCallback(BaseCallback):
       def _on_step(self):
           if self.num_timesteps - self.last_snapshot >= 20_000:
               snapshot = copy.deepcopy(self.model.policy).eval()
               self.training_env.env_method("set_opponent", wrap_as_callable(snapshot))
               self.last_snapshot = self.num_timesteps
           return True
   ```

3. **Initialize from Stage 1.** Load `catchy_run_stage1.zip` as both the trainee starting weights *and* the initial opponent.

### Move-on criteria

- [ ] Win rate vs. **the heuristic** holds at ≥80% (regression check)
- [ ] Win rate vs. **last snapshot** stays near 50% over several updates (sign of healthy self-play convergence — neither side dominates after each snapshot rotation)
- [ ] Episode length and entropy are stable

### Output

`catchy_run_stage2.zip` plus a directory of intermediate snapshots — seeds the Stage 3 pool.

---

## Stage 3 — Opponent pool (league)

**Status:** not yet implemented.

**Goal:** prevent overfitting to the latest self by training against a *mix* of past opponents. Catches the "policy beats current opponent but forgets how to beat older strategies" failure mode.

### What needs to be built

1. **Pool assembly** — collect heuristic + last N snapshots (N ≈ 5–10):
   ```python
   pool = [heuristic_opponent]
   for snapshot_path in recent_snapshots(n=8):
       pool.append(make_snapshot_opponent(snapshot_path))
   env.set_opponent_pool(pool)
   ```

2. **Pool refresh in callback** — every snapshot, append the new snapshot and drop the oldest non-heuristic.

The env already supports this via `set_opponent_pool` — no env changes needed.

### Move-on criteria

- [ ] Win rate vs. **every member of the pool** is roughly balanced (no single opponent dominates or gets dominated by ≥80%)
- [ ] Win rate vs. **heuristic** stays at ≥80%

### Output

The final trained model.

---

## TensorBoard cheatsheet

```bash
# Start
tensorboard --logdir ./tb_logs/

# Different port if 6006 is taken
tensorboard --logdir ./tb_logs/ --port 6007
```

- Each `model.learn(...)` call creates a subdirectory `<name>_N/`. Use `tb_log_name="stage1_v1"` etc. in `model.learn` to name runs.
- Smoothing slider (~0.6) is essential for `ep_rew_mean` — raw curve is jittery.
- Toggle runs in the left sidebar to compare experiments.

---

## Hyperparameter notes

Current values in `model.py`:

| Param | Value | Reason |
|---|---|---|
| `learning_rate` | `3e-4` | PPO standard. Drop to `1e-4` if `approx_kl` consistently >0.05. |
| `n_steps` | `2048` | ~50–100 episodes per rollout at 40-turn max. |
| `batch_size` | `64` | Default; fine for tiny network. |
| `gamma` | `0.99` | Discount. Short episodes mean this barely matters. |
| `gae_lambda` | `0.95` | GAE smoothing. Default. |
| `clip_range` | `0.2` | PPO clip. Default. |
| `ent_coef` | `0.01` | Entropy bonus. **Bump to 0.05 if policy collapses early in Stage 1.** |

Don't tune anything before a sparse-reward Stage 1 run has clearly failed.

---

## Quick troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ep_rew_mean` flat at 0 | Mask not reaching policy, or opponent broken | Verify `ActionMasker` is wrapping the env; print the mask inside `mask_fn` |
| `ep_rew_mean` strongly negative | Reward sign flipped, or opponent is too strong | Check `_play_opponent_turn` reward extraction |
| `entropy_loss` near 0 within 10k steps | Premature commitment | Bump `ent_coef` to `0.05` |
| `approx_kl` repeatedly >0.1 | Updates too aggressive | Lower `learning_rate` to `1e-4` or `n_epochs` to 5 |
| Training crashes with shape mismatch | `observation_space` not matching `_obs()` output | Confirm both are `(10, 7, 7)` |

---

## File map at a glance

| File | Purpose |
|---|---|
| `rl_agent/environment.py` | The Gym env. |
| `rl_agent/opponents.py` | Opponent callables. Heuristic now; snapshot loader for Stage 2. |
| `rl_agent/custom_cnn.py` | CNN feature extractor + `policy_kwargs`. |
| `rl_agent/model.py` | Training entry point. |
| `catcher_vs_runner/engine.py` | Pure game logic. Do not modify for RL needs. |
| `tb_logs/` | TensorBoard event files (created on first run). |
| `catchy_run_stage1.zip` | Stage 1 model output. |
