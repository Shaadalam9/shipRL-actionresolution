# DQN (3 actions) — calm water v1

Three fixed rudder settings: hard-over port (−35°), midships (0°), hard-over starboard (+35°).

| | |
|---|---|
| Algorithm | DQN |
| Conditions | Calm water (no wind, no waves) |
| Checkpoint | episode 7000 of 10001 |
| Random seed | 69574 |
| Original name | `model_012` |
| Config | [`configs/dqn_3_action.yaml`](../../../configs/dqn_3_action.yaml) |

## Evaluate it

```bash
uv run scripts/evaluate.py models/dqn_3_action/calm_water_v1
```

Add `--wind-speed 6 --wind-dir 90` or `--wave-height 3 --wave-period 10` to
test it outside the conditions it was trained in.

## Retrain it

```bash
uv run scripts/train.py configs/dqn_3_action.yaml --run-name calm_water_v2
```

## Notes

- With only hard-over and midships available, this controller cannot hold a steady moderate turn; expect a visibly zig-zagging track. That behaviour is the reason the variant is in the comparison.
- Epsilon-greedy exploration keeps a floor of ε = 0.2 for the whole run, unlike the 5-action variant which anneals to zero.
- This checkpoint was originally stored outside `saved_models/`, which is why it is easy to miss in the pre-restructure layout.

## Files

- `config.yaml` — configuration this run is reproduced from
- `original_parameters.txt` — parameter dump written by the original training script
- `training_history.json` — per-episode returns and losses
- `policy/` — TensorFlow SavedModel loaded by `evaluate.py`
