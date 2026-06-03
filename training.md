# Training Guide

End-to-end plan for training Catcher vs. Runner agents. **One network per role** — the runner and catcher are trained as separate `MaskablePPO` models that only meet at evaluation (and during cross-play in Stage 2+).

Training escalates in opponent strength: random → heuristic → cross-play → league.

---

## Where am I right now?

Tick boxes as you progress:

### Runner
- [ ] **Stage 0** — explore the grid against a 70% random / 30% heuristic catcher mix; learn to capture squares under light pressure
- [ ] **Stage 1** — train against the bundled heuristic catcher
- [ ] **Stage 2** — train against the latest catcher snapshot
- [ ] **Stage 3** — train against a pool of past catcher snapshots + heuristic

### Catcher
- [ ] **Stage 1** — train against the trained runner snapshot `catchy_run_runner_stage0_v1_4_1`
- [ ] **Stage 2** — train against the latest runner snapshot
- [ ] **Stage 3** — train against a pool of past runner snapshots + heuristic

The catcher has no Stage 0: it starts directly against the trained runner checkpoint `catchy_run_runner_stage0_v1_4_1`. That snapshot actively pursues special squares (~4.8 captures per episode against a random catcher), so the engine's terminal ±1 gives a real gradient from step 1 instead of paying out free timeout wins for inaction. The catcher additionally gets its own lightweight reward shaper (`CatcherRewardShaper`) to scaffold projectile tactics — the sparse signal would otherwise teach pure stepping (see `rl_agent/environment_explanations/catcher_reward_shaping.md`).

---

## Architecture summary

```
rl_agent/
  environment.py      CatchyRunEnv — single-agent Gym env with the opponent
                      played inline inside step(). Trainee role is fixed at
                      construction via the trainee_role kwarg.
  opponents.py        heuristic_opponent — adapts the bundled heuristic agent
                      to the (state) -> int contract the env expects.
  custom_cnn.py       CustomGridCNN feature extractor (3× Conv2d + linear),
                      sized for the 9×7×7 observation tensor.
  model.py            MaskablePPO + CnnPolicy + custom extractor. Entry point
                      for training. trainee_role is threaded through.
```

Key design choices:
- **Two networks.** One model per role. No shared weights, no role channel.
- **Algorithm:** `MaskablePPO` from `sb3-contrib` — handles action masking via `info["action_mask"]`.
- **Observation:** `(9, 7, 7)` — the engine's 9 channels. Each model sees only its own role's perspective.
- **Reward:** sparse ±1 on termination, sign flipped to the trainee's role. Both roles additionally get per-step shaping from `rl_agent/reward_shaping.py`, with separate subclasses (`RunnerRewardShaper`, `CatcherRewardShaper`) extending a shared `RewardShaper` base. The runner shaper has 9 components (capture bonus, alive bonus, catcher-distance, projectile threat, special-attraction, sprint-waste, urgency, unsafe-capture, plus the base engine reward). The catcher shaper has 4 components (capture-block, distance-closure, bullet-coverage, plus the base engine reward), intentionally lighter — its job is to scaffold projectile tactics that a sparse catcher would otherwise skip. All magnitudes are tuned so the engine's terminal ±1 still dominates. See `rl_agent/environment_explanations/reward_shaping.md` and `catcher_reward_shaping.md` for per-component derivations.
- **Opponent strategy:** opponent plays inline inside `env.step`, so SB3 sees a normal single-agent env.

---

## Stage 0 — Runner vs. mixed pool (exploration)

**Goal:** the runner learns the *objective* under light catcher pressure. The 70% random episodes give low-stakes opportunities to stumble onto specials and learn the capture loop; the 30% heuristic episodes apply just enough pressure that the policy can't ignore the catcher and over-fit to a passive opponent. By the end of Stage 0 the runner should reliably head for special squares and capture (not just hover adjacent to) them.

**Why no catcher version:** the catcher's reward density doesn't need this scaffolding — it can learn directly vs. the heuristic runner.

### How it works

The env exposes a weighted opponent pool via `set_opponent_pool(opponents, weights)`. The pool is sampled **per episode** in `reset()`, so the opponent is fixed for the duration of an episode — 30% of *episodes* are heuristic, not 30% of *turns within an episode*.

```python
def make_env(trainee_role: Agent):
    env = CatchyRunEnv(trainee_role=trainee_role)
    env.set_opponent_pool(
        [env._default_opponent, heuristic_opponent],
        weights=[0.7, 0.3],
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
- [ ] Win rate against the 70/30 pool stabilizes at **≥70%**
- [ ] Episode length trends *down* over training — the runner is finishing faster, not stalling
- [ ] Behavioral check: run `trace_rewards.py` against the checkpoint and confirm the runner is *capturing* specials (capture component fires regularly), not loitering adjacent to them to farm the attraction term

### Common failures

- **Capture rate plateaus at ~3** with episode length near the 40-turn cap → policy is exploiting shaping (attraction farming or alive-bonus farming). Inspect with `trace_rewards.py` and rebalance `ATTRACTION_NEAREST` vs `CAPTURE_BONUS` in `reward_shaping.py`.
- **`ep_rew_mean` stuck near 0** → the runner isn't finding specials. Drop the heuristic share to 10% temporarily so the runner sees more low-pressure episodes; once captures appear, restore 30%.
- **Capture rate near 0 with very short episodes** → the heuristic catcher is killing the runner before it learns. Same fix: drop heuristic share to 10%, retrain from scratch.
- **Entropy stuck high (~`ln(n_legal)`) with flat losses** → no signal is reaching the policy. Re-check `_shape_reward` and the mask threading before extending the run.

### Output

`catchy_run_runner_stage0.zip` — seed for Stage 1.

---

## Stage 1 — Skill floor

**Goal:** establish a skill floor. The runner trains against the bundled heuristic catcher; the catcher trains against the trained runner snapshot `catchy_run_runner_stage0_v1_4_1`.

### Setup

**Runner** — load Stage 0 checkpoint, switch opponent to the heuristic catcher:

```python
def make_env():
    env = CatchyRunEnv(trainee_role="runner", opponent_policy=heuristic_opponent)
    env = ActionMasker(env, mask_fn)
    return env

train(load_from="catchy_run_runner_stage0",
      save_to="catchy_run_runner_stage1",
      tb_log_name="runner_stage1",
      ent_coef=0.005)   # lower entropy so it refines instead of thrashing
```

**Catcher** — from scratch against the trained runner snapshot `catchy_run_runner_stage0_v1_4_1`. `make_env(trainee_role="catcher")` in `model.py` plugs `rl_runner_policy` (from `catchy_run_game/agents/rl_runner.py`, which lazily loads the runner checkpoint) in as the inline opponent, and `environment.py` binds `CatcherRewardShaper` automatically based on `trainee_role`:

```python
train(trainee_role="catcher",
      save_to="catchy_run_catcher_stage1_v0",
      tb_log_name="catcher_stage1_v0")
```

The trained runner captures aggressively enough that catcher episodes terminate on real outcomes (kill, projectile kill, or contested timeout) rather than the runner failing to reach majority. That makes the engine's terminal ±1 the dominant training signal from the first rollout. Catcher shaping (capture-block, distance-closure, bullet-coverage, special-defense) is light by design — it nudges the policy toward projectile use without overriding the terminal verdict. See `rl_agent/environment_explanations/catcher_reward_shaping.md` before tuning coefficients.

The two trainings are independent — run them sequentially or in parallel processes.

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

### Move-on criteria (per model)

- [ ] Win rate vs. opposing heuristic plateaus at **≥70%** for several evaluations
- [ ] Plateau holds for ~50k–100k steps with no further improvement
- [ ] Policy entropy has stabilized (not collapsed to zero, not still drifting)

200k steps is the initial budget. If a plateau hasn't been reached, extend to 500k or 1M. If `ep_rew_mean` hasn't moved off zero after 50k steps, **stop and debug** — don't keep burning compute. Common causes:
- Mask not threading through `ActionMasker` correctly
- Reward sign flipped
- Opponent function broken
- For the runner: forgot to `load_from` the Stage 0 checkpoint, so it doesn't know to capture squares

### Output

`catchy_run_runner_stage1.zip` and `catchy_run_catcher_stage1.zip` — seeds for Stage 2.

---

## Stage 2 — Cross-play (each model vs. the other)

**Status:** not yet implemented.

**Goal:** climb past the heuristic ceiling by training each model against the latest snapshot of the *other* model. With two separate networks, cross-play replaces "frozen self-play" — each side has a real, learning adversary instead of a copy of itself.

### What needs to be built

1. **Snapshot loader** in `opponents.py`:
   ```python
   def make_snapshot_opponent(model_path: str, opponent_role: str):
       model = MaskablePPO.load(model_path)
       def opponent(state):
           obs = engine.encode_observation(state, opponent_role)
           mask = engine.legal_action_mask(state)
           action, _ = model.predict(obs, action_masks=mask, deterministic=True)
           return int(action)
       return opponent
   ```

2. **`CrossPlayCallback`** — every ~20k steps, snapshot the model being trained to a shared directory:
   ```python
   class CrossPlayCallback(BaseCallback):
       def _on_step(self):
           if self.num_timesteps - self.last_snapshot >= 20_000:
               path = f"snapshots/{self.role}_{self.num_timesteps}.zip"
               self.model.save(path)
               self.last_snapshot = self.num_timesteps
           return True
   ```
   Each training process polls the *other* role's snapshot directory and rotates its env opponent (via `env.set_opponent(...)`) when a new snapshot appears.

3. **Initialize from Stage 1.** Each model loads its Stage 1 checkpoint as starting weights; the initial opponent is the other model's Stage 1 checkpoint.

### Move-on criteria (per model)

- [ ] Win rate vs. **opposing heuristic** holds at ≥80% (regression check)
- [ ] Win rate vs. **latest opposing snapshot** stays near 50% over several updates (sign of healthy convergence — neither side dominates after rotation)
- [ ] Episode length and entropy are stable

### Output

`catchy_run_runner_stage2.zip`, `catchy_run_catcher_stage2.zip`, plus a directory of intermediate snapshots — seeds the Stage 3 pool.

---

## Stage 3 — Opponent pool (league)

**Status:** not yet implemented.

**Goal:** prevent overfitting to the *latest* opposing model by training against a *mix* of past opponents. Catches the "policy beats current opponent but forgets older strategies" failure mode.

### What needs to be built

1. **Pool assembly** (per model) — heuristic of opposing role + last N snapshots of opposing role (N ≈ 5–10):
   ```python
   pool = [heuristic_opponent]
   for snap in recent_snapshots(role="catcher", n=8):
       pool.append(make_snapshot_opponent(snap, opponent_role="catcher"))
   env.set_opponent_pool(pool)
   ```

2. **Pool refresh** — every new snapshot, append it and drop the oldest non-heuristic entry.

The env already supports this via `set_opponent_pool` — no env changes needed.

### Move-on criteria (per model)

- [ ] Win rate vs. **every member of the pool** is roughly balanced (no single opponent dominates or gets dominated by ≥80%)
- [ ] Win rate vs. **opposing heuristic** stays at ≥80%

### Output

Final trained models: `catchy_run_runner_final.zip`, `catchy_run_catcher_final.zip`.

---

## TensorBoard cheatsheet

```bash
# Start
tensorboard --logdir ./tb_logs/

# Different port if 6006 is taken
tensorboard --logdir ./tb_logs/ --port 6007
```

- Each `model.learn(...)` call creates a subdirectory `<name>_N/`. Use distinct `tb_log_name` per run (e.g. `runner_stage0`, `catcher_stage1_v2`) so runs don't pile into the same line.
- Smoothing slider (~0.6) is essential for `ep_rew_mean` — raw curve is jittery.
- Toggle runs in the left sidebar to compare across roles and stages.

---

## Hyperparameter notes

Current values in `model.py`:

| Param | Value | Reason |
|---|---|---|
| `learning_rate` | `3e-4` | PPO standard. Drop to `1e-4` when continuing from a checkpoint (Stage 1+). |
| `n_steps` | `2048` | ~50–100 episodes per rollout at 40-turn max. |
| `batch_size` | `64` | Default; fine for tiny network. |
| `gamma` | `0.99` | Discount. Short episodes mean this barely matters. |
| `gae_lambda` | `0.95` | GAE smoothing. Default. |
| `clip_range` | `0.2` | PPO clip. Default. |
| `ent_coef` | `0.01` | Entropy bonus. Drop to `~0.005` once a stage's policy is moving. Bump to `0.05` if the policy collapses early. |

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
| Catcher training silently still trains runner | `trainee_role` arg ignored in `__init__` | Confirm `self.trainee_role = trainee_role` (not hard-coded to `"runner"`) |

---

## File map at a glance

| File | Purpose |
|---|---|
| `rl_agent/environment.py` | The Gym env. `trainee_role` fixed at construction; binds the right `*RewardShaper`. |
| `rl_agent/reward_shaping.py` | `RewardShaper` base + `RunnerRewardShaper` and `CatcherRewardShaper` subclasses. |
| `rl_agent/opponents.py` | Opponent callables. Heuristic now; snapshot loader for Stage 2. |
| `rl_agent/custom_cnn.py` | CNN feature extractor + `policy_kwargs`. Sized for 9×7×7. |
| `rl_agent/model.py` | Training entry point. Role-aware `make_env`, checkpoint naming. |
| `catcher_vs_runner/engine.py` | Pure game logic. Do not modify for RL needs. |
| `tb_logs/` | TensorBoard event files (created on first run). |
| `catchy_run_runner_stageN.zip` | Runner model checkpoints. |
| `catchy_run_catcher_stageN.zip` | Catcher model checkpoints. |
