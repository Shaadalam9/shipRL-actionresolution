# Trained models

Each run directory is self-contained: the policy, the exact configuration that
produced it, and the training history. Nothing here needs to be cross-referenced
against the code to be understood.

```
models/<experiment>/<run name>/
├── config.yaml                 configuration that reproduces this run
├── original_parameters.txt     parameter dump from the original training run
├── training_history.json       per-episode returns and losses
└── policy/                     TensorFlow SavedModel, loaded by evaluate.py
```

Runs produced by the current `scripts/train.py` additionally contain
`metrics.json`, `checkpoints/ep_<n>/`, and `plots/` holding the return and loss
curves as PNG, HTML, and PDF.

## Released policies

All four were trained in calm water, with no wind and no waves. The checkpoint
column records the point in training the released policy was taken from, which
is not always the end of the run.

| Experiment | Run | Algorithm | Rudder command | Checkpoint | Seed | Original name |
|---|---|---|---|---|---|---|
| `dqn_3_action` | `calm_water_v1` | DQN | 3 fixed angles (−35°, 0°, +35°) | episode 7000 | 69574 | `model_012` |
| `dqn_5_action` | `calm_water_v1` | DQN | 5 fixed angles (−35°, −20°, 0°, +20°, +35°) | episode 10000 | 95832 | `model_002` |
| `ddpg` | `calm_water_v1` | DDPG | continuous, ±35° | episode 5000 | 38153 | `model_001` |
| `ppo` | `calm_water_v1` | PPO | continuous, ±35° | iteration 60 | 43738 | `model_004` |

## Using a released policy

```bash
uv run scripts/evaluate.py models/ddpg/calm_water_v1
```

This writes figures (PNG, HTML, and PDF) and a `results.json` of tracking
metrics to `models/ddpg/calm_water_v1/evaluation/`. To compare every controller
at once instead, run `uv run scripts/analyze.py`. Add `--wind-speed` / `--wave-height`
to test a policy outside the calm water it was trained in — the controllers
were never shown a disturbance during training, so this is the interesting
experiment rather than a formality.

## Reproducing a run

```bash
uv run scripts/train.py configs/ddpg.yaml --run-name my_rerun
```

The shipped policies predate the restructuring of this repository, so a rerun
will not be bit-identical to them. See "Behaviour changes" in the top-level
[README](../README.md) for what differs.

## Naming

Run names describe the training conditions, not an index: `calm_water_v1`,
`wind_45deg_v1`, `dqn5_seed_95832`. Prefer something a reader can interpret
without opening the config.
