# V-JEPA 2 + Continual Learning

## 1) Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,docs]
pip install -r src/vjepa2/requirements.txt
```

## 2) Build manifests for each CL task

Create at least two dataset roots/manifests:

- `datasets/task1/*.csv` (broad dynamics pre-training)
- `datasets/task2/*.csv` (robot/environment-specific adaptation)

Each line in each CSV is:

```text
/path/to/video.mp4 <label>
```

## 3) Launch Task 1 (base world model)

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

## 4) Launch Task 2 (continual adaptation)

```bash
cd src/vjepa2
python -m app.main \
  --fname configs/train/vitg16/droid-256px-8f.yaml \
  --devices cuda:0 \
  --root_path ../../datasets/task2 \
  --train_csv ../../datasets/task2/train.csv \
  --val_csv ../../datasets/task2/val.csv \
  --tag task2_robot_adapt
```

## 5) Inject continual learning methods

Typical hooks:

- Freeze/partially freeze encoder weights.
- Add replay batches sampled from task 1 manifests.
- Add EWC/MAS-style penalties in the training loss.
- Swap or expand predictor heads (including action-conditioned predictor).

Good implementation entry points in the local V-JEPA 2 tree:

- `src/vjepa2/app/vjepa/train.py`
- `src/vjepa2/src/models/predictor.py`
- `src/vjepa2/src/models/ac_predictor.py`
