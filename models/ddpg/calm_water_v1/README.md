# DDPG — calm water v1

Continuous rudder angle in ±35°, produced directly by the actor network.

| | |
|---|---|
| Algorithm | DDPG |
| Conditions | Calm water (no wind, no waves) |
| Checkpoint | episode 5000 of 10001 |
| Random seed | 38153 |
| Original name | `model_001` |
| Config | [`configs/ddpg.yaml`](../../../configs/ddpg.yaml) |

## Evaluate it

```bash
uv run scripts/evaluate.py models/ddpg/calm_water_v1
```

Add `--wind-speed 6 --wind-dir 90` or `--wave-height 3 --wave-period 10` to
test it outside the conditions it was trained in.

## Retrain it

```bash
uv run scripts/train.py configs/ddpg.yaml --run-name calm_water_v2
```

## Notes

- The actor and critic both used a **constant** learning rate. The original `ddpg_train.py` built exponential decay schedules but never passed them to the optimisers, and `original_parameters.txt` records `decay_steps:0`, confirming the schedules were inactive. `config.yaml` therefore omits decay, matching what actually ran.
- Exploration was Gaussian noise added to the actor output, decaying linearly to zero over 9000 episodes.

## Files

- `config.yaml` — configuration this run is reproduced from
- `original_parameters.txt` — parameter dump written by the original training script
- `training_history.json` — per-episode returns and losses
- `policy/` — TensorFlow SavedModel loaded by `evaluate.py`
