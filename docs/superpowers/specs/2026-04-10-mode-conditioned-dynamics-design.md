# Mode-Conditioned Dynamics with Depth Fusion Design

**Date:** 2026-04-10
**Author:** Deepanshu (dm6262) + Claude
**Status:** Approved
**Goal:** Improve world model prediction accuracy (especially flight mode) while staying MPC-friendly (<50ms per dynamics step).

---

## Context

### Current State (Approach B results, 50 epochs)

Best model: DINOv2+MLP+unfiltered

| Horizon | Overall ADE | Ground ADE | Flight ADE |
|---------|------------|------------|------------|
| 1.6s | 0.74m | 0.46m | 0.92m |
| 3.2s | 1.04m | 0.62m | 1.32m |
| 5.0s | 1.43m | 0.83m | 1.82m |

### Problems Identified

1. **Flight is 2x worse than ground** — the single MLP dynamics model averages ground and flight physics, compromising both
2. **State loss mixes incompatible units** — position (meters, std ~2-3m) and quaternion (unitless, std ~0.3) in one MSE, causing orientation to dominate gradients
3. **Cumulative delta error** — predicting displacement from frame 0 means errors at step 5 corrupt all subsequent steps
4. **DINOv2 is depth-blind** — only gets RGB; depth is critical for 3D flight prediction
5. **No risk/collision data** — risk head trains on zeros (unchanged, not addressed here)

### Constraints

- **MPC-friendly**: prediction must run in <50ms per dynamics step
- **No reprocessing**: use existing HF datasets as-is
- **Incremental**: changes build on current codebase, don't require architectural rewrite

---

## Design

### 1. Separate Position and Orientation Losses

**File:** `src/world_model/engine/trainer.py`

Replace single state MSE with weighted split:

```
loss_state = λ_pos * MSE(pred[:, :, :3], gt[:, :, :3])       # position
           + λ_ori * MSE(pred[:, :, 3:state_dim], gt[:, :, 3:state_dim])  # orientation
```

Defaults: `λ_pos = 1.0`, `λ_ori = 0.5`

Add `lambda_pos` and `lambda_ori` to `TrainConfig`.

### 2. Per-Frame Delta Prediction

**Files:** `src/world_model/data/hf_datasets.py`, `scripts/world_model/eval_trajectory.py`

Currently odom positions are deltas from frame 0: `odom[t] = pos[t] - pos[0]`. This means a prediction error at step 5 propagates to all later steps.

Change to **per-frame deltas**: `odom[t] = pos[t] - pos[t-1]` for t > 0, `odom[0] = 0`. The model predicts "how much will I move in the next timestep?" — small values (~0.1-0.3m) that don't accumulate.

Frame 0 remains `[0, 0, 0, ...]` (no displacement yet). This only affects position dims (0:3). Orientation (3:7), velocity (7:10), angular velocity (10:13) are unchanged.

**Normalization:** NormStats recomputed on per-frame deltas. Std will be much smaller (~0.1-0.3m vs ~2-3m for cumulative deltas).

**Eval update:** To compute ADE, reconstruct the trajectory by cumulative sum of predicted per-frame deltas, then compare to cumulative sum of GT per-frame deltas.

### 3. Parallel Depth Encoder for DINOv2

**File:** `src/world_model/backbones/depth_encoder.py` (new), `src/world_model/architecture.py`

DINOv2 is pretrained on 3-channel RGB and cannot accept 4 channels. Add a parallel lightweight depth encoder:

```python
class DepthEncoder(nn.Module):
    """Small CNN to encode depth map into a latent vector."""
    def __init__(self, z_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, z_dim),
            nn.LayerNorm(z_dim),
        )

    def forward(self, depth):  # (B, 1, H, W)
        return self.net(depth)  # (B, z_dim)
```

Modify `WorldModel.__init__`: if backbone is DINOv2, create a `DepthEncoder` and update fusion to accept `z_vis + z_depth + z_prop`:

```
Fusion input: concat(z_vis, z_depth, z_prop) = 128 + 128 + 128 = 384
Fusion output: z (256-dim)
```

Update `Fusion` module's input dim from `z_vis + z_prop` to `z_vis + z_depth + z_prop`.

For CNN backbone (already 4-ch RGBD), no depth encoder needed — works as before.

### 4. Mode-Conditioned Dynamics

**File:** `src/world_model/components/dynamics.py` (new class), `src/world_model/architecture.py`

#### Mode Classifier

```python
class ModeClassifier(nn.Module):
    """Predicts operating mode from latent state."""
    def __init__(self, z_dim, num_modes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64), nn.ReLU(),
            nn.Linear(64, num_modes),
        )

    def forward(self, z):  # (B, z_dim)
        return self.net(z)  # (B, num_modes) logits
```

Modes: 0 = ground, 1 = flight, 2 = transition

#### Mode-Conditioned Dynamics

```python
class ModeConditionedDynamics(nn.Module):
    """Action-conditioned dynamics with mode-specific residuals."""
    def __init__(self, z_dim, act_dim, hidden=256, mode_hidden=128, num_modes=3):
        super().__init__()
        self.shared = mlp(z_dim + act_dim, hidden, z_dim, depth=3)
        self.mode_heads = nn.ModuleList([
            mlp(z_dim + act_dim, mode_hidden, z_dim, depth=2)
            for _ in range(num_modes)
        ])
        self.classifier = ModeClassifier(z_dim, num_modes)

    def forward(self, z, a):
        za = torch.cat([z, a], dim=-1)
        base = self.shared(za)

        mode_logits = self.classifier(z)
        mode_weights = F.softmax(mode_logits, dim=-1)  # (B, num_modes)

        mode_residual = torch.zeros_like(base)
        for i, head in enumerate(self.mode_heads):
            mode_residual += mode_weights[:, i:i+1] * head(za)

        return z + base + mode_residual, mode_logits
```

The dynamics returns both the next latent state AND mode logits (for the classification loss).

#### Pseudo-Labels

Generated in the dataset loader by looking at raw absolute Z from the HF data:
- Read `odom_2` (absolute Z) at each frame from the underlying HF dataset
- Per episode: if abs(Z at frame 0) >= 1.5m → all frames labeled "flight" (1)
- If std(Z within episode) >= 0.5m → labeled "flight"
- Otherwise → "ground" (0)
- Transition (2): first and last 5 frames of episodes where Z crosses the 1.5m threshold

Labels stored as an additional return field in `HFWindowDataset.__getitem__` and passed through `SampleBatch`.

#### Mode Classification Loss

Added to `rollout_loss` in `trainer.py`:

```
loss_mode = λ_mode * CrossEntropy(mode_logits, mode_labels)
```

`λ_mode = 0.1` — small weight, auxiliary signal.

The mode classifier output at step 0 is compared to the episode's label. During rollout steps 1..H, mode logits are predicted but not supervised (no ground truth for future mode since we only have the episode-level label).

### 5. Integration into Training Script

**File:** `scripts/world_model/train_world_model.py`

New CLI arg: `--dynamics_type mode_conditioned` (alongside existing `mlp`, `rssm`)

When `mode_conditioned` is selected:
- `ModeConditionedDynamics` is used instead of `LatentDynamics`
- Mode pseudo-labels are computed and passed through the batch
- Mode classification loss is added to the total loss
- Rollout handles the extra `mode_logits` return value

### 6. Updated Rollout in Architecture

**File:** `src/world_model/architecture.py`

The `rollout()` method needs to handle `ModeConditionedDynamics` which returns `(z_next, mode_logits)` instead of just `z_next`:

```python
if hasattr(self.dynamics, 'classifier'):
    z_next, mode_logits = self.dynamics(z, a_seq[:, t])
    all_mode_logits.append(mode_logits)
else:
    z_next = self.dynamics(z, a_seq[:, t])
```

Mode logits are collected during rollout and returned for loss computation.

---

## File Changes Summary

| File | Change | Description |
|------|--------|-------------|
| `src/world_model/engine/trainer.py` | Modify | Split state loss into position + orientation, add mode loss |
| `src/world_model/data/hf_datasets.py` | Modify | Per-frame deltas, add mode pseudo-labels |
| `src/world_model/data/types.py` | Modify | Add `mode_labels` field to SampleBatch |
| `src/world_model/backbones/depth_encoder.py` | **New** | Lightweight depth CNN encoder |
| `src/world_model/components/dynamics.py` | Modify | Add ModeConditionedDynamics, ModeClassifier |
| `src/world_model/architecture.py` | Modify | Depth encoder integration, mode-conditioned rollout |
| `scripts/world_model/train_world_model.py` | Modify | `--dynamics_type mode_conditioned`, mode labels |
| `scripts/world_model/eval_trajectory.py` | Modify | Reconstruct trajectory from per-frame deltas |
| `experiments/slurm_template_v2.sh` | Modify | Add mode_conditioned experiments |
| `experiments/run_all_v3.sh` | **New** | V3 experiment grid |

**Untouched:** DINOv2 backbone weights, CNN backbone, RSSM dynamics, HF datasets on Hub, data pipeline.

---

## Experiment Grid (v3)

Focus on the best backbone (DINOv2) + unfiltered dataset, testing the new components:

| # | Backbone | Dynamics | Depth Enc | Per-frame Delta | Notes |
|---|----------|----------|-----------|-----------------|-------|
| 1 | DINOv2 | MLP | No | Yes | Baseline v3 (delta fix only) |
| 2 | DINOv2 | MLP | Yes | Yes | + depth encoder |
| 3 | DINOv2 | mode_conditioned | No | Yes | + mode dynamics |
| 4 | DINOv2 | mode_conditioned | Yes | Yes | Full Approach 2 |

All on unfiltered dataset, 50 epochs, A100, with WANDB.

Compare against v2 best (DINOv2+MLP+unfiltered, cumulative deltas) as baseline.

---

## Success Criteria

1. **Flight ADE < 1.2m at 3.2s** (currently 1.32m) — mode conditioning should specialize for flight
2. **Overall ADE < 0.8m at 3.2s** (currently 1.04m) — per-frame deltas + depth should help everywhere
3. **Ground ADE stays < 0.7m at 3.2s** (currently 0.62m) — don't regress ground performance
4. **MPC step latency < 50ms** — measured on A100
5. **Mode classifier accuracy > 85%** — can correctly identify ground vs flight from latent state

---

## Execution Order

1. Per-frame delta conversion in `hf_datasets.py` + eval update
2. Split position/orientation loss in `trainer.py`
3. Depth encoder module + integration into WorldModel
4. ModeConditionedDynamics + ModeClassifier + pseudo-labels
5. Training script updates (`--dynamics_type mode_conditioned`)
6. V3 experiment grid (4 runs)
7. Compare results
