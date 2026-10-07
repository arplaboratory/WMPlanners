# Training and evaluation

`scripts/world_model/train_wm.py` dispatches to the standard WorldModel or DinoWM trainer.

## Standard WorldModel: CNN or pretrained vision backbone, MLP or RSSM

### Example data loading options:

A local CNN run with generated NPZ data:
```bash
python scripts/world_model/make_toy_dataset.py --out_dir data/toy --episodes 200
python scripts/world_model/train_wm.py --method wm \
  --data_path data/toy --dataset_type npz \
  --backbone cnn --dynamics_type mlp \
  --epochs 20 --out_dir runs/cnn
```

A local CNN run with generated parquet data (`frames.parquet` and `windows.parquet`):
```bash
python scripts/world_model/train_wm.py --method wm \
  --data_path /path/to/parquet_dataset --dataset_type hf \
  --backbone cnn --dynamics_type rssm \
  --epochs 20 --out_dir runs/wm
```
The Parquet tables may contain image paths and array-valued sensor columns.

The same trainer supports CNN, DINOv2, DINOv3, or V-JEPA2 with RSSM. Set `--backbone cnn` on that command for CNN + RSSM. `--dynamics_type mlp` selects the non-RSSM dynamics path.

For a Hugging Face dataset and a frozen DINOv2 encoder:
```bash
python scripts/world_model/train_wm.py --method wm \
  --data_path lquan9/collect1 --dataset_type hf \
  --backbone dinov2 --backbone_name dinov2_vitb14 \
  --freeze_backbone --dynamics_type rssm \
  --epochs 20 --batch_size 64 --out_dir runs/dino_rssm
```
Hub rows may embed images and numbered sensor columns. `hf` accepts a Hub ID.

For DINOv3, obtain an official checkpoint from Meta and pass its local path or
URL. DINOv3 currently supports the ViT variants because the standard world
model consumes patch tokens:

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/world_model/train_world_model.py --ddp \
  --data_path /path/to/parquet_dataset --dataset_type hf \
  --backbone dinov3 --backbone_name dinov3_vitb16 \
  --backbone_weights /path/to/dinov3_vitb16_pretrain_lvd1689m.pth \
  --freeze_backbone --wm_mode jepa-lite --dynamics_type mlp \
  --horizon 36 --img_size 224 --out_dir runs/dinov3_jepa_lite_h36
```

---

For the existing two-GPU DDP setup, the underlying trainer accepts `--ddp`:
```bash
torchrun --standalone --nproc_per_node=2 \
  scripts/world_model/train_world_model.py --ddp \
  --data_path lquan9/collect1 --backbone dinov2 \
  --backbone_name dinov2_vitb14 --freeze_backbone \
  --epochs 20 --batch_size 64 --lr 3e-4 \
  --horizon 8 --img_size 224 --wandb --out_dir runs/dinov2
```

---

## JEPA-lite and state decoder

JEPA-lite is the `wm` trainer with a latent matching objective inspired by joint-embedding predictive architectures {cite:p}`assran2023ijepa`.

```bash
python scripts/world_model/train_wm.py --method wm \
  --wm_mode jepa-lite --data_path lquan9/collect1 \
  --backbone cnn --dynamics_type mlp \
  --epochs 20 --batch_size 64 --lr 3e-4 \
  --horizon 8 --img_size 224 --out_dir runs/jepa_lite_c1_512
```

The same objective accepts DINOv2 or DINOv3 with a frozen backbone. Both apply ImageNet RGB normalization inside the backbone so training, evaluation, and runtime preprocessing remain identical. A frozen backbone is shared with the EMA target model to avoid a duplicate foundation encoder on each GPU. The state-head trainer supports `--latent_source observed` (decode encoded sensor frames) and `--latent_source rollout` (decode predicted states used by MPC). A two-stage run initializes from observed frames, then fine-tunes on predicted rollouts:

For a CNN trained without depth, pass `--rgb_only`. The resulting checkpoint
records a three-channel vision input and all checkpoint loaders ignore depth.

```bash
python scripts/world_model/train_state_head_decoder.py \
  --checkpoint runs/jepa_lite_c1_512/latest.pt \
  --dataset /path/to/matching/dataset --dataset_type hf \
  --heads state --latent_source observed --epochs 5 --batch_size 8

python scripts/world_model/train_state_head_decoder.py \
  --checkpoint runs/jepa_lite_c1_512/latest_with_observed_heads.pt \
  --dataset /path/to/matching/dataset --dataset_type hf \
  --heads state --latent_source rollout --epochs 10 --batch_size 8
```

Each stage reports both observed and rollout validation position RMSE; select a planner checkpoint by held-out rollout error and planning performance. You can run `rollout` directly from `latest.pt` if you do not need the observed-latent probe. The dataset must match the checkpoint's odometry, IMU, action dimensions and action convention. The trainer loads `norm_stats.pt` beside the input checkpoint when present, or accepts `--norm_stats_path`; it does not create new statistics. For a state-only head, set the ROS planner's `risk_weight` to zero. Use `--heads both` only with valid collision and non-collision labels.

For long rollouts, train the dynamics at the deployment horizon before fitting
the rollout head. A staged curriculum can resume the same run at increasing
`--horizon` values. `--epochs` is the cumulative final epoch, not the number of
additional epochs, so a four-stage run can use horizons `8, 16, 24, 36` with
epoch limits `10, 20, 30, 45`. Checkpoints include and restore the frozen EMA
target model; legacy checkpoints without it initialize the target from the
online model and print a warning. Do not fit a head beyond the checkpoint's
`trained_horizon`; the head trainer rejects that mismatch.

Standard training writes `action_bounds.pt` from the transformed training
samples by default. Disable this only when bounds are supplied separately with
`--no-save_action_bounds`. The model checkpoint records a `training_contract`
containing the trained rollout horizon, history length, and dataset action
semantics. Head training verifies this action contract against its dataset and
preserves it in the output checkpoint.

---

## DinoWM

DinoWM {cite:p}`zhou2024dinowm` uses a frozen DINOv2 encoder and a ViT latent predictor. The trainer takes CLI flags, including a required data path:

```bash
WANDB_MODE=disabled python scripts/world_model/train_wm.py \
  --method dinowm --data_path lquan9/collect1 \
  --epochs 10 --out_dir runs/dinowm
```

For a local frames/windows Parquet dataset, use `--data_path /path/to/dataset` with the same command. A separate `val` split is used when available.

The options `--frameskip`, `--num_hist`, `--num_pred`, `--encoder`, and `--encoder_name` control DinoWM's sequence and visual encoder. For image reconstruction, add `--has_decoder --train_decoder` and choose `--decoder_type vqvae` or `transposed_conv`. Image metrics work with the core dependencies; install `.[decoder]` to request optional LPIPS scores from `world_model.metrics.eval_images(..., include_lpips=True)`. `configs/dino-wm/trial.yaml` records an earlier configuration; the current trainer uses argparse flags.

## Checkpoints and evaluation

Use a separate `--out_dir` for each run. Checkpoint paths below are relative to that directory:

| Trainer | Latest checkpoint | Numbered checkpoints | Best validation checkpoint |
|---|---|---|---|
| Standard WorldModel (including JEPA-lite) | `latest.pt` after every epoch | `epoch_NNN.pt` every `--save_every` epochs (default 10), and at the final epoch | `best.pt` |
| DinoWM | `checkpoints/model_latest.pth` at each save interval | `checkpoints/model_N.pth` every `--save_every_x_epoch` epochs (default 1) | `checkpoints/model_best.pth` |

Both trainers select the best checkpoint by validation loss at their checkpoint intervals, when a separate validation split exists. A train-only dataset has no best validation checkpoint.

To evaluate the standard WorldModel trained on the toy dataset above, run the toy-environment MPC example:

```bash
python scripts/world_model/eval_mpc.py --ckpt runs/cnn/latest.pt --episodes 20
```

| Option | Meaning | Default |
|---|---|---|
| `--data_path` | Hugging Face dataset ID or local path | required |
| `--dataset_type` | `hf` for Hub rows or local Parquet tables; `npz` for NPZ episodes. DinoWM currently supports `hf`. | `hf` |
| `--schema_config` | Optional YAML mapping for local Parquet table paths, columns, sensor policy, and split fallback. | canonical names |
| `--precompute_windows` | Cache local Parquet windows in RAM before standard training. | off |
| `--max_windows` | Limit cached Parquet windows per split; requires `--precompute_windows`. | unlimited |
| `--backbone` | Standard visual encoder: `cnn`, `dinov2`, `dinov3`, or `vjepa2` | `cnn` |
| `--backbone_name` | Model variant; defaults per backend | backend-specific |
| `--backbone_weights` | Official DINOv3 checkpoint path or URL | unset |
| `--dynamics_type` | Standard dynamics: `mlp`, `rssm`, or `td-jepa` | `mlp` |
| `--wm_mode` | `standard` or `jepa-lite` objective | `standard` |
| `--freeze_backbone` | Keep pretrained visual weights fixed | off |
| `--ddp` | Distributed standard training | off |
| `--epochs` | Training epochs | `20` for standard training |
| `--out_dir` | Checkpoint and run directory | `runs` |
| `--save_every` | Standard numbered checkpoint interval | `10` |
| `--save_action_bounds` | Write transformed per-dimension action statistics for CEM. | on |
| `--resume` | Restore model, optimizer, epoch, and EMA target state. | automatic `out_dir/latest.pt` |
| `--save_every_x_epoch` | DinoWM checkpoint interval | `1` |

**Read next**
- [ROS2](ros2.md) contains ROS2 integration details (e.g., launch arguments, checkpoint requirements, topics and node parameters).
