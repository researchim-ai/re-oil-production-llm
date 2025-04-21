# re-oil-production-llm

```
python -m grpo.main --forecast_days 1 --simulation_max_time 7 --wandb
```
```
python -m grpo.main --forecast_days 1 --simulation_max_time 4 --kl_weight 0.05 --lr 1e-5 --clip_eps 0.3 --gamma 0.99 --total_steps 1000 --rollouts_per_step 8 --train_batch_size 8 --temperature 0.7 --wandb
```