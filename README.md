# DS_GA_3001
This repo supports the DS_GA_3001 - Embodied Learning and Vision Course Project

`src/cjepa` is a lightweight integration layer for adapting continual learning to JEPA.
`external/vjepa2` official [VJEPA2](https://github.com/facebookresearch/vjepa2) repository.

- V-JEPA 2 experiments scaffolding for two-phase continual learning:
  - **Task 1**: self-supervised representation learning on broad video data.
  - **Task 2**: post-training/action-conditioned adaptation on robot data.

- Utilities to build V-JEPA 2 CSV datasets (`/path/to/video.mp4 label`).
- A small CLI (`cjepa`) to generate a reproducible local training
  command for `src/vjepa2/app/main.py`.

## Environment setup

> The bundled V-JEPA 2 project in `src/vjepa2` targets Linux environments like
> Ubuntu 24.04 and commonly requires CUDA + multi-GPU for full-scale training.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,docs]

# V-JEPA2 runtime dependencies (decord can be platform-sensitive)
pip install -r external/vjepa2/requirements.txt
```

## Data preparation (CSV format)

V-JEPA 2 dataset manifests are plain CSV-like text files where each row is:

```text
/path/to/video.mp4 <label>
```

Labels are ignored in self-supervised pre-training, but the column is still
required by the loader contract.

You can generate these manifests with `cjepa.write_dataset_csv`.

## CL mapping to V-JEPA 2

When adapting V-JEPA 2 as a world model, keep the following module boundaries:

- **Encoder (ViT context/target)**: latent state extraction; often frozen or
  regularized in CL to reduce catastrophic forgetting.
- **Predictor**: latent dynamics model; primary target for adaptation/expansion.
- **V-JEPA 2-AC predictor**: action-conditioned dynamics for control/robotics.

A practical CL setup is phase-based:

1. Train on general videos (Task 1).
2. Adapt to robot interaction videos (Task 2) while applying replay,
   regularization (EWC/MAS), or predictor-only expansion.

Generate a launch command:

```bash
cjepa \
  --config configs/train/vitl16/pretrain-256px-16f.yaml \
  --devices cuda:0 \
  --root-path datasets/task1 \
  --train-csv datasets/task1/train.csv \
  --val-csv datasets/task1/val.csv \
  --task-name task1_pretrain
```

Then run it from the V-JEPA 2 directory:

```bash
cd src/vjepa2
python -m app.main \
  --fname configs/train/vitl16/pretrain-256px-16f.yaml \
  --devices cuda:0 \
  --root_path ../../datasets/task1 \
  --train_csv ../../datasets/task1/train.csv \
  --val_csv ../../datasets/task1/val.csv \
  --tag task1_pretrain
```

For a task-2 adaptation pass, reuse the same command pattern with robot-specific
CSVs and a new `--tag`, then incorporate your CL method in the trainer/loss path
inside `src/vjepa2/app/vjepa/train.py` (or AC equivalent).

## Extra documentation

See `docs/getting_started.md` for a concise step-by-step.

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE).
