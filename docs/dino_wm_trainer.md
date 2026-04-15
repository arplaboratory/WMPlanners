# DinoWM World Model

## Run Training

```bash
cd /path/to/NYU_DS_GA_3001

HYDRA_FULL_ERROR=1 WANDB_MODE=disabled conda run -n cjepa \
  python scripts/world_model/train_world_model_dinowm.py \
  2>&1 | tee /tmp/wm_train.log
```

- `WANDB_MODE=disabled` — skips W&B upload (remove if you want to log to W&B).
- `HYDRA_FULL_ERROR=1` — shows full tracebacks on Hydra errors.
- Training config is read from `configs/dino-wm/trial.yaml`.
- Checkpoints are saved to `checkpoints/model_latest.pth` and `checkpoints/model_<epoch>.pth`.
- If a checkpoint already exists, training **resumes** from it automatically.

### Override Config Values from the Command Line

```bash
# Change number of epochs
WANDB_MODE=disabled conda run -n cjepa \
  python scripts/world_model/train_world_model_dinowm.py \
  training.epochs=10

# Change dataset
WANDB_MODE=disabled conda run -n cjepa \
  python scripts/world_model/train_world_model_dinowm.py \
  env=collect1
```

## 6. Monitor Training

```bash
# Follow the log live
tail -f /tmp/wm_train.log

# Or grep for losses only
grep "Training loss\|Validation loss" /tmp/wm_train.log
```

## Config Reference

| File | Purpose |
|---|---|
| `configs/dino-wm/trial.yaml` | Main training config (epochs, encoder, frameskip, etc.) |
| `configs/dino-wm/env/collect1.yaml` | Dataset path + normalization stats |
| `configs/dino-wm/encoder/dino.yaml` | Visual encoder (DINOv2) |
| `configs/dino-wm/predictor/vit.yaml` | Latent predictor |