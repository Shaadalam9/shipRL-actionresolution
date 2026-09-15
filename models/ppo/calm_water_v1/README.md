# PPO — calm water v1

Continuous rudder angle in ±35°, sampled from the actor's output distribution.

| | |
|---|---|
| Algorithm | PPO |
| Conditions | Calm water (no wind, no waves) |
| Checkpoint | iteration 60 of 100 (3000 episodes) |
| Random seed | 43738 |
| Original name | `model_004` |
| Config | [`configs/ppo.yaml`](../../../configs/ppo.yaml) |

## Evaluate it

```bash
uv run scripts/evaluate.py models/ppo/calm_water_v1
```

Add `--wind-speed 6 --wind-dir 90` or `--wave-height 3 --wave-period 10` to
test it outside the conditions it was trained in.

## Retrain it

```bash
uv run scripts/train.py configs/ppo.yaml --run-name calm_water_v2
```

## Notes

- PPO is on-policy: each iteration collects 50 fresh episodes, takes 10 optimisation passes over them, and discards the batch.
- The original `ppo_train.py` called `agent.train(experience)` twice per iteration — once discarding the result, once to read the loss — so this policy received two full 10-epoch updates per batch. The current loop trains once, as intended. A rerun will therefore not match this checkpoint step for step.
- `greedy_eval` was `false` for this run, so the saved policy is stochastic.

## Files

- `config.yaml` — configuration this run is reproduced from
- `original_parameters.txt` — parameter dump written by the original training script
- `training_history.json` — per-episode returns and losses
- `policy/` — TensorFlow SavedModel loaded by `evaluate.py`
