# DQN (5 actions) — calm water v1

Five fixed rudder settings: −35°, −20°, 0°, +20°, +35°.

| | |
|---|---|
| Algorithm | DQN |
| Conditions | Calm water (no wind, no waves) |
| Checkpoint | episode 10000 of 10001 |
| Random seed | 95832 |
| Original name | `model_002` |
| Config | [`configs/dqn_5_action.yaml`](../../../configs/dqn_5_action.yaml) |

## Evaluate it

```bash
uv run scripts/evaluate.py models/dqn_5_action/calm_water_v1
```

Add `--wind-speed 6 --wind-dir 90` or `--wave-height 3 --wave-period 10` to
test it outside the conditions it was trained in.

## Retrain it

```bash
uv run scripts/train.py configs/dqn_5_action.yaml --run-name calm_water_v2
```

## Notes

- The two intermediate angles let the agent hold a gentler turn than the 3-action variant, which generally produces a smoother track.
- Epsilon-greedy exploration anneals to zero over 6000 episodes, so the last third of training is effectively greedy.

## Files

- `config.yaml` — configuration this run is reproduced from
- `original_parameters.txt` — parameter dump written by the original training script
- `training_history.json` — per-episode returns and losses
- `policy/` — TensorFlow SavedModel loaded by `evaluate.py`
