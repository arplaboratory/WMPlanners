# wm_planner Data Pipeline

This note explains the data pipeline used by `wm_planner`, based on the notebook
`docs/notebooks/world_model_pipeline.ipynb` and the implementation under
`scripts/world_model` and `src/world_model/data`.

There are two connected pipelines:

1. Offline dataset creation: raw robot logs become training episodes and,
   optionally, a Hugging Face dataset.
2. Training-time loading: NPZ or Hugging Face episodes become tensors consumed
   by the standard world model or by DinoWM.

## 1. Big Picture

```text
                         OFFLINE DATA CREATION

  ROS2 MCAP / ROS1 bag
          |
          v
  +-------------------+
  | Stage 0 Extract   |  scripts/world_model/extract_dataset.py
  |                   |
  | - decode images   |
  | - write CSVs      |
  | - write manifest  |
  +---------+---------+
            |
            v
  extracted bag directory
  - shard_0000/color_*.jpg
  - shard_0000/depth_*.png
  - image_index.csv
  - imu.csv
  - odom.csv
  - position_cmd.csv or so3_cmd.csv
  - manifest.json
            |
            v
  +-------------------+
  | Stage 1 Episodes  |  scripts/world_model/build_episodes.py
  |                   |
  | - sync sensors    |
  | - filter bad data |
  | - frame skip      |
  | - slice windows   |
  +---------+---------+
            |
            v
  NPZ episodes: ep_00000.npz, ep_00001.npz, ...
            |
            v
  +-------------------+
  | Stage 2 Select    |  scripts/world_model/select_episodes.py
  |                   |
  | - descriptors     |
  | - z-score blocks  |
  | - FPS selection   |
  +---------+---------+
            |
            v
  selected NPZ subset
            |
            v
  +-------------------+
  | Stage 3 HF Upload |  scripts/world_model/data_pipeline/hf_upload.py
  |                   |
  | - explode frames  |
  | - build HF schema |
  | - save/push       |
  +---------+---------+
            |
            v
  Hugging Face dataset
```

The wrapper CLI is `scripts/world_model/data_pipeline/run_pipeline.py`.

```bash
python -m scripts.world_model.data_pipeline.run_pipeline \
  --mcap path/to/bag.mcap \
  --robot race16 \
  --frame-skip 6 \
  --episode-len 50 \
  --stride 50 \
  --target-count 5000 \
  --push-to-hub user/repo
```

If Stage 0 has already been run, use `--bags path/to/extracted`.

## 2. Stage 0: Extract Bags

Code path:

- `scripts/world_model/data_pipeline/stage0_extract.py`
- `scripts/world_model/extract_dataset.py`
- `scripts/world_model/robot_config.py`

Stage 0 reads raw bag files and writes a simple file-and-CSV dataset. The robot
configuration decides which topics are treated as color, depth, IMU, odometry,
and action/command topics.

```text
             raw bag messages
                   |
       +-----------+-----------+
       |                       |
       v                       v
  image topics            sensor topics
  color/depth             imu/odom/cmd
       |                       |
       v                       v
  shard directories          CSV files
  color_000000.jpg           imu.csv
  depth_000000.png           odom.csv
                             position_cmd.csv
                             so3_cmd.csv
       |                       |
       +-----------+-----------+
                   |
                   v
             image_index.csv
             manifest.json
```

Main details:

- Color images are written as JPEGs into `shard_0000/`, `shard_0001/`, etc.
- Depth images are written as PNGs. If incoming depth is `32FC1`, it is converted
  from metres to millimetres and stored as `uint16`.
- `image_index.csv` records `frame_id`, `timestamp_ns`, `color_path`, and
  `depth_path`.
- Sensor CSVs are written lazily, only when messages for that sensor are seen.
- `manifest.json` records the robot config and the list of extracted bags.
- The extractor handles both MCAP and ROS1 bag input. It also has topic fallback
  logic for common topic suffixes.

The output has this shape:

```text
pipeline_out/extracted/
  manifest.json
  my_bag/
    image_index.csv
    imu.csv
    odom.csv
    position_cmd.csv
    shard_0000/
      color_000000.jpg
      depth_000000.png
      ...
```

## 3. Stage 1: Build Fixed-Length NPZ Episodes

Code path:

- `scripts/world_model/build_episodes.py`
- `scripts/world_model/sync_utils.py`

Stage 1 turns each extracted bag into fixed-length trajectory episodes.

```text
  manifest.json
       |
       v
  for each bag directory
       |
       +--> load image_index.csv
       +--> load odom.csv / imu.csv / action CSV
       |
       v
  sort every stream by timestamp
       |
       v
  sync sensors to image timestamps
       |
       v
  build valid-frame mask
       |
       +--> remove long idle runs
       +--> split at timestamp gaps
       |
       v
  contiguous valid runs
       |
       v
  slice episodes with frame_skip and stride
       |
       v
  ep_00000.npz, ep_00001.npz, ...
```

### 3.1 Time Synchronization

Images are the clock for the episode. For each image timestamp, the pipeline
finds the nearest sensor row from each CSV.

```text
image timestamps:   I0      I1      I2      I3      I4
                    |       |       |       |       |
odom timestamps:   O0   O1      O2      O3      O4
                    |       |       |       |       |
synced odom:       nn(I0)  nn(I1)  nn(I2)  nn(I3)  nn(I4)
```

`max_sync_gap_ms` controls how far a nearest sensor row may be from the image
timestamp. Missing sensor files are zero-filled using dimensions from the robot
config.

### 3.2 Quality Filtering

Two quality checks define valid runs:

```text
valid frame = not idle and no timestamp gap before this frame
```

Idle detection:

- Computes speed from synced odometry linear velocity.
- Marks a frame idle only if it belongs to a run of at least
  `idle_persistence` consecutive frames with speed below `min_speed`.

Gap detection:

- Computes differences between adjacent image timestamps.
- Marks a gap before frame `i` when `dt > 2 * median(dt)`.

Only contiguous valid runs are sliced into episodes.

### 3.3 Frame Skipping and Episode Slicing

The arguments interact like this:

```text
episode_len = number of frames kept in the final NPZ episode
frame_skip  = raw-frame spacing between kept frames
stride      = episode start spacing before multiplying by frame_skip

raw_window = episode_len * frame_skip
starts     = range(run_start, run_end - raw_window + 1, stride * frame_skip)
kept idxs  = start, start + frame_skip, start + 2*frame_skip, ...
```

Example with `episode_len=5`, `frame_skip=3`:

```text
raw frames:  00 01 02 03 04 05 06 07 08 09 10 11 12 13 14
kept:        XX       XX       XX       XX       XX

final episode has 5 frames but spans 15 raw frames.
```

### 3.4 NPZ Episode Contents

Each episode is saved with `np.savez_compressed`:

```text
ep_00000.npz
  rgb     : (T, H, W, 3)  uint8
  depth   : (T, H, W)     float32, metres
  imu     : (T, D_imu)    float32
  odom    : (T, D_odom)   float32
  action  : (T, D_act)    float32
```

For the documented Race16-style config, expected dimensions are often:

```text
imu    : 10
odom   : 13
action : 14 for position_cmd, or whatever the configured action source exposes
```

The code also writes `metadata.json` with total episode counts, per-bag counts,
sensor dimensions, quality parameters, and episode parameters.

## 4. Stage 2: Select Diverse Episodes

Code path:

- `scripts/world_model/select_episodes.py`

Stage 2 reduces the full episode pool to a diverse subset using Farthest Point
Sampling (FPS). It is dynamics/action focused, not visual focused.

```text
  ep_*.npz
     |
     v
  per-episode descriptor
     |
     +--> action mean        (D_act)
     +--> action std         (D_act)
     +--> odom start->end    (6)
     |
     v
  [action_mean | action_std | odom_delta]
     |
     v
  z-score action block and odom block separately
     |
     v
  apply weights
     |
     v
  FPS: repeatedly choose episode farthest from selected set
     |
     v
  selected/ep_00000.npz, selected/ep_00001.npz, ...
```

`odom_delta` is:

```text
[dx, dy, dz, ax, ay, az]
```

where displacement is expressed in the start-frame body coordinate system and
rotation is represented as an axis-angle vector. Episodes with invalid
quaternions are skipped.

For `encoder="none"`, the code weights descriptor blocks as:

```text
action stats weight = 0.75
odom delta weight   = 0.25
```

The selected directory receives:

```text
selected/
  ep_00000.npz
  ep_00001.npz
  metadata.json
  selection_stats.json
```

## 5. Stage 3: Build / Upload Hugging Face Dataset

Code path:

- `scripts/world_model/data_pipeline/hf_upload.py`

Stage 3 converts selected NPZ episodes into a row-per-frame Hugging Face dataset.

```text
  selected NPZ episodes
          |
          v
  for each episode
    for each frame t
      row = {
        episode_id,
        frame_idx,
        image,
        odom_0 ... odom_N,
        action_0 ... action_N,
        imu_0 ... imu_N,
        depth_mean
      }
          |
          v
  datasets.Dataset
          |
          +--> save_to_disk(selected/hf_dataset)
          |
          +--> optional push_to_hub(repo_id)
```

Important detail: Stage 3 does not store the full depth image in the Hugging
Face dataset. It stores only `depth_mean`.

## 6. Training-Time Data Loading

The notebook `docs/notebooks/world_model_pipeline.ipynb` demonstrates two model
families:

- Standard `WorldModel`
- `DinoWM`

They share the same upstream episode/HF data, but their collators return
different batch structures.

## 7. Standard WorldModel Loading Path

Code path:

- `src/world_model/data/dataloaders.py`
- `src/world_model/data/hf_datasets.py`
- `src/world_model/data/types.py`
- `src/world_model/engine/trainer.py`

There are two sources.

### 7.1 Local NPZ Source

```text
  ep_*.npz
     |
     v
  NPZEpisodeDataset
     |
     +--> indexes sliding windows within each episode
     +--> optionally resizes images
     +--> returns numpy dict
     |
     v
  collate_npz
     |
     v
  SampleBatch
```

`SampleBatch` tensors:

```text
rgb       : (B, T, 3, H, W) float, [0, 1]
depth     : (B, T, 1, H, W) float
imu       : (B, T, D_imu)
odom      : (B, T, D_odom)
action    : (B, T, D_act)
collision : (B, T, 1)
```

### 7.2 Hugging Face Source

```text
  HF dataset rows
     |
     v
  HFWindowDataset
     |
     +--> groups rows by episode_id or window_id
     +--> sorts each group by frame_idx
     +--> transforms PIL images to tensors
     +--> reconstructs odom/action/imu arrays from numbered columns
     |
     v
  collate_hf
     |
     v
  SampleBatch
```

The notebook loads:

```python
ds_hf = HFWindowDataset("lquan9/collect1", img_size=224)
train_loader = DataLoader(ds_hf, batch_size=4, shuffle=True, collate_fn=collate_hf)
```

Then training uses:

```text
SampleBatch
    |
    v
rollout_loss(model, batch, TrainConfig)
    |
    +--> choose random sub-window if training
    +--> call model(rgb, depth, odom, imu, action)
    +--> compare predicted latent/state/risk against targets
```

Loss terms:

```text
loss = lambda_z     * latent MSE
     + lambda_state * odom/state MSE
     + lambda_risk  * collision BCE
```

## 8. Standard WorldModel Pipeline Diagram

```text
      SampleBatch
  rgb/depth/odom/imu/action/collision
             |
             v
      +--------------+
      | encode obs   |
      | visual+state |
      +------+-------+
             |
             v
         latent z0
             |
             v
      +--------------+
      | dynamics     | <-- action sequence
      | rollout      |
      +------+-------+
             |
             v
   predicted latents z_pred
             |
       +-----+------+
       |            |
       v            v
  state head     risk head
       |            |
       v            v
  odom/state     collision/risk
  prediction     prediction
       |            |
       +-----+------+
             |
             v
     rollout_loss
```

For inference and planning, the notebook uses CEM:

```text
  current observation -> encode -> z0
                              |
                              v
  sample candidate action sequences
                              |
                              v
  rollout each sequence through model
                              |
                              v
  score by cost(goal distance + risk penalty)
                              |
                              v
  keep/refit best sequence distribution
                              |
                              v
  output best first action
```

## 9. DinoWM Loading Path

The DinoWM section of the notebook uses:

```python
ds_dinowm = HFWindowDataset(
    hf_repo,
    img_size=224,
    frameskip=FRAMESKIP,
    sample_mode=True,
    sample_k=NUM_HIST + NUM_PRED,
)
dinowm_loader = DataLoader(ds_dinowm, collate_fn=collate_dinowm)
```

Intended sample-mode behavior:

```text
  full episode/window from HF
        |
        v
  choose fixed-length sample of k frames
  with stride = frameskip
        |
        v
  dict:
    rgb    : (k, 3, H, W)
    imu    : (k, D_imu)
    action : (k, D_act)
    state  : (k, D_odom + D_imu + 1)
        |
        v
  collate_dinowm
        |
        v
  obs = {
    "visual"  : (B, T, 3, H, W),
    "proprio" : (B, T, D_imu)
  }
  action      : (B, T, D_act)
  state       : (B, T, D_state)
```

## 10. DinoWM Model Data Flow

DinoWM replaces the trainable visual backbone with frozen DINOv2 and trains a
latent predictor.

```text
 obs["visual"]                         obs["proprio"]          action
 (B,T,3,H,W)                           (B,T,D_imu)             (B,T,D_act)
      |                                      |                       |
      v                                      v                       v
 +------------+                    +----------------+       +----------------+
 | DINOv2     | frozen             | proprio embed  |       | action embed   |
 | encoder    |                    | Conv1d-like    |       | Conv1d-like    |
 +-----+------+                    +--------+-------+       +--------+-------+
       |                                    |                        |
       v                                    v                        v
 visual patch tokens                 proprio tokens             action tokens
 (B,T,P,D_v)                         (B,T,D_p)                  (B,T,D_a)
       |                                    |                        |
       +------------------+-----------------+------------------------+
                          |
                          v
        fuse tokens by concat_dim
        - concat_dim=1: tile proprio/action across patches
          and concatenate along feature dimension
                          |
                          v
        context tokens for NUM_HIST frames
                          |
                          v
                  +---------------+
                  | ViTPredictor  |
                  +-------+-------+
                          |
                          v
              predicted future latent tokens
                          |
                          v
              MSE against target DINO/proprio tokens
```

In the notebook example:

```text
NUM_HIST = 3
NUM_PRED = 1
FRAMESKIP = 5
visual encoder = DINOv2 ViT-S/14
encoder emb_dim = 384
num_patches = 256
proprio emb dim = 10
action emb dim = 10
concat_dim = 1
predictor token dim = 384 + 10 + 10 = 404
```

The forward pass returns:

```text
z_pred
visual_pred
visual_reconstructed
loss
loss_components
```

With `decoder=None`, there is no pixel reconstruction and `visual_pred` is
`None`. The loss is embedding-space MSE.

## 11. DinoWM Rollout

The notebook also demonstrates open-loop rollout:

```text
  obs_0 = first NUM_HIST frames
  act   = history actions + future planned/zero actions
             |
             v
       WM.rollout(obs_0, act)
             |
             v
  autoregressive latent sequence
             |
             v
  z_obses["visual"] : (B, total_T, P, D_v)
  z_full            : full fused token sequence
```

This is the latent prediction primitive that can be used for downstream
evaluation or planning.

## 12. Key Files

```text
docs/notebooks/world_model_pipeline.ipynb
  Demonstrates HF loading, standard WorldModel training, MPC, and DinoWM.

docs/data_pipeline.md
  Short existing overview of MCAP -> NPZ -> HF.

scripts/world_model/data_pipeline/run_pipeline.py
  End-to-end CLI orchestrating stages 0-3.

scripts/world_model/extract_dataset.py
  Reads MCAP/ROS1 bags and writes image shards plus sensor CSVs.

scripts/world_model/build_episodes.py
  Synchronizes sensors, filters frames, slices episodes, writes NPZ files.

scripts/world_model/select_episodes.py
  Computes action/odom descriptors and selects diverse episodes via FPS.

scripts/world_model/data_pipeline/hf_upload.py
  Converts selected NPZ episodes to a Hugging Face dataset.

src/world_model/data/dataloaders.py
  NPZ dataset and collator for standard WorldModel.

src/world_model/data/hf_datasets.py
  Hugging Face window dataset and collators, including DinoWM collator.

src/world_model/engine/trainer.py
  Multi-step rollout loss for standard WorldModel.
```

## 13. Implementation Caveat

While reading `src/world_model/data/hf_datasets.py`, note that the current file
appears to contain duplicated or stale constructor code for `HFWindowDataset`.
The visible class defines two `__init__` methods; in Python, the second one
overrides the first. However, methods later in the class still reference
attributes such as `sample_mode`, `sample_k`, `frameskip`, `odom_stats`,
`imu_stats`, `action_stats`, and `collision_stats`.

The notebook expects those sample-mode arguments to work:

```python
HFWindowDataset(..., sample_mode=True, sample_k=..., frameskip=...)
```

but the active constructor shown in the source only accepts:

```python
HFWindowDataset(dataset, img_size=224, norm_stats=None)
```

So the documented intended pipeline is clear, but the current source may need
cleanup before the DinoWM notebook cell runs exactly as written.
