# Training and evaluation

`scripts/world_model/train_wm.py` dispatches to the standard WorldModel or DinoWM trainer.

## Standard WorldModel: CNN or DINOv2, MLP or RSSM

A local CNN run with generated NPZ data:

```bash
python scripts/world_model/make_toy_dataset.py --out_dir data/toy --episodes 200
python scripts/world_model/train_wm.py --method wm \
  --data_path data/toy --dataset_type npz \
  --backbone cnn --dynamics_type mlp \
  --epochs 20 --out_dir runs/cnn
```

```bash
python scripts/world_model/train_wm.py --method wm \
  --data_path /path/to/parquet_dataset --dataset_type hf \
  --backbone cnn --dynamics_type rssm \
  --epochs 20 --out_dir runs/wm
```

`hf` accepts a Hub ID, saved Hugging Face rows, or a local directory with `frames.parquet` and `windows.parquet`. Local tables use one `ParquetWindowDataset` reader. The Parquet tables may contain image paths and array-valued sensor columns, whereas Hub rows may embed images and numbered sensor columns.

The same trainer supports CNN or DINOv2 with RSSM. For a Hugging Face dataset and a frozen DINOv2 encoder:
```bash
python scripts/world_model/train_wm.py --method wm \
  --data_path lquan9/collect1 --dataset_type hf \
  --backbone dinov2 --backbone_name dinov2_vitb14 \
  --freeze_backbone --dynamics_type rssm \
  --epochs 20 --batch_size 64 --out_dir runs/dino_rssm
```

Set `--backbone cnn` on that command for CNN + RSSM. The visual encoder can also be `vjepa2` in the standard architecture. `--dynamics_type mlp` selects the non-RSSM dynamics path.

For the existing two-GPU DDP setup, the underlying trainer accepts `--ddp`:
```bash
torchrun --standalone --nproc_per_node=2 \
  scripts/world_model/train_world_model.py --ddp \
  --data_path lquan9/collect1 --backbone dinov2 \
  --backbone_name dinov2_vitb14 --freeze_backbone \
  --epochs 20 --batch_size 64 --lr 3e-4 \
  --horizon 8 --img_size 224 --wandb --out_dir runs/dinov2
```

## JEPA-lite and state decoder

JEPA-lite is the `wm` trainer with a latent matching objective inspired by joint-embedding predictive architectures {cite:p}`assran2023ijepa`.

```bash
python scripts/world_model/train_wm.py --method wm \
  --wm_mode jepa-lite --data_path lquan9/collect1 \
  --backbone cnn --dynamics_type mlp \
  --epochs 20 --batch_size 64 --lr 3e-4 \
  --horizon 8 --img_size 512 --out_dir runs/jepa_lite_c1_512
```

The same objective accepts the DINOv2 backbone using `--backbone dinov2 --backbone_name dinov2_vitb14 --freeze_backbone`. A trained JEPA-lite checkpoint can be given to the state-decoder and replay MPC scripts:

```bash
python scripts/world_model/train_state_head_decoder.py \
  --checkpoint runs/jepa_lite_c1_512/latest.pt \
  --dataset lquan9/collect1 --epochs 5 --batch_size 16 --lr 1e-3

python scripts/world_model/eval_mpc_replay.py \
  --checkpoint runs/jepa_lite_c1_512/latest_with_state_head.pt \
  --dataset lquan9/collect1_unfiltered --horizon 16 --num_episodes 50

python scripts/world_model/eval_mpc_recovery_replay.py \
  --checkpoint runs/jepa_lite_c1_512/latest_with_state_head.pt \
  --dataset lquan9/collect1_unfiltered --horizon 16 --num_episodes 10
```

## DinoWM

DinoWM {cite:p}`zhou2024dinowm` uses a frozen DINOv2 encoder and a ViT latent predictor. The trainer takes CLI flags, including a required data path:

```bash
WANDB_MODE=disabled python scripts/world_model/train_wm.py \
  --method dinowm --data_path lquan9/collect1 \
  --epochs 10 --out_dir runs/dinowm
```

For a local frames/windows Parquet dataset, use `--data_path /path/to/dataset` with the same command. A separate `val` split is used when available.

Checkpoints are written in `runs/dinowm/checkpoints/` as `model_latest.pth` and numbered epoch files. The options `--frameskip`, `--num_hist`, `--num_pred`, `--encoder`, and `--encoder_name` control the sequence and visual encoder. For image reconstruction, add `--has_decoder --train_decoder` and choose `--decoder_type vqvae` or `transposed_conv`. Image metrics work with the core dependencies; install `.[decoder]` to request optional LPIPS scores from `world_model.metrics.eval_images(..., include_lpips=True)`. `configs/dino-wm/trial.yaml` records an earlier configuration, while the current trainer uses argparse flags.

## Evaluation and key options

The standard trainer writes `latest.pt` after every epoch and `epoch_NNN.pt` every `--save_every` epochs (default 10), including the final epoch. DinoWM writes `model_latest.pth` and `model_N.pth` every `--save_every_x_epoch` epochs (default 1). With a separate validation split, both trainers compare validation loss at checkpoint epochs and keep the lowest-loss model as `best.pt` (standard) or `checkpoints/model_best.pth` (DinoWM). A train-only dataset has no `best` checkpoint because training loss is not a validation criterion. Use a distinct `--out_dir` for each run.

The standard trainer writes `latest.pt` under its output directory. The existing MPC evaluation entry point is:

```bash
python scripts/world_model/eval_mpc.py --ckpt runs/dino_rssm/latest.pt --episodes 20
```

| Option | Meaning | Default |
|---|---|---|
| `--data_path` | Hugging Face dataset ID or local path | required |
| `--dataset_type` | `hf` for Hub rows or local Parquet tables; `npz` for NPZ episodes. DinoWM currently supports `hf`. | `hf` |
| `--schema_config` | Optional YAML mapping for local Parquet table paths, columns, sensor policy, and split fallback. | canonical names |
| `--precompute_windows` | Cache local Parquet windows in RAM before standard training. | off |
| `--max_windows` | Limit cached Parquet windows per split; requires `--precompute_windows`. | unlimited |
| `--backbone` | Standard visual encoder: `cnn`, `dinov2`, or `vjepa2` | `cnn` |
| `--dynamics_type` | Standard dynamics: `mlp`, `rssm`, or `td-jepa` | `mlp` |
| `--wm_mode` | `standard` or `jepa-lite` objective | `standard` |
| `--freeze_backbone` | Keep pretrained visual weights fixed | off |
| `--ddp` | Distributed standard training | off |
| `--epochs` | Training epochs | `20` for standard training |
| `--out_dir` | Checkpoint and run directory | `runs` |
| `--save_every` | Standard numbered checkpoint interval | `10` |
| `--save_every_x_epoch` | DinoWM checkpoint interval | `1` |

## Read next
- [ROS2](ros2.md) contains ROS2 integration details (e.g., launch arguments, checkpoint requirements, topics and node parameters).