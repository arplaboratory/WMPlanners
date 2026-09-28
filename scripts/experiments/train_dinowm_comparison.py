"""
Apples-to-apples comparison: Train Akash's DINO-WM architecture using
our training infrastructure (same data, same val split, same eval metrics).

Uses Akash's core components:
  - DINOv2 frozen encoder → patch tokens (B, T, 256, 768)
  - Proprio encoder (sinusoidal + linear)
  - Action encoder (linear)
  - Causal masked ViT predictor (latent dynamics)

But with our infrastructure:
  - HFWindowDataset with position deltas + normalization
  - 80/20 val split by episode_id
  - ADE/FDE evaluation at multiple horizons
  - Same checkpointing and logging

Usage:
    python experiments/train_dinowm_comparison.py \
        --dataset lquan9/collect1 --epochs 20 --out_dir /scratch/$USER/wm_dinowm_comparison
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange, repeat

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from world_model.data import HFWindowDataset, collate_hf
from world_model.data.normalization import NormStats
from world_model.utils import seed_everything


# ── Akash's components (simplified, self-contained) ──────────────────

class ProprioEncoder(nn.Module):
    """Linear projection for proprioceptive data."""
    def __init__(self, input_dim, emb_dim=10):
        super().__init__()
        self.net = nn.Linear(input_dim, emb_dim)

    def forward(self, x):
        return self.net(x)


class ActionEncoder(nn.Module):
    """Linear projection for actions."""
    def __init__(self, input_dim, emb_dim=10):
        super().__init__()
        self.net = nn.Linear(input_dim, emb_dim)

    def forward(self, x):
        return self.net(x)


class CausalAttention(nn.Module):
    """Multi-head self-attention with causal masking across frames."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # Causal mask: each token can only attend to earlier tokens
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        dots.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CausalViTPredictor(nn.Module):
    """Causal masked Vision Transformer for latent dynamics prediction."""
    def __init__(self, dim, depth=6, heads=8, dim_head=64, mlp_dim=1024, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CausalAttention(dim, heads, dim_head, dropout),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout),
                )
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return x


class DinoWMComparison(nn.Module):
    """Simplified DINO-WM for apples-to-apples comparison.

    Architecture:
      - DINOv2 frozen → patch tokens (B, T, num_patches, 768)
      - Proprio + Action encoded and appended as extra tokens
      - Causal masked ViT predicts next-frame tokens
      - State head decodes pose from mean-pooled predicted patches

    Key simplification vs Akash's full code:
      - No VQ-VAE decoder (decoder-free, like our model)
      - No Hydra config (argparse like ours)
      - State head added for ADE/FDE comparison
    """

    def __init__(self, odom_dim, imu_dim, act_dim, state_dim=6,
                 num_hist=3, vit_depth=4, vit_heads=8, vit_dim=768,
                 proprio_emb_dim=10, action_emb_dim=10, device='cuda'):
        super().__init__()
        self.num_hist = num_hist
        self.state_dim = state_dim

        # Frozen DINOv2 encoder
        import torch.hub
        dino_repo = os.path.join(os.path.dirname(__file__), '..', 'external', 'dino')
        self.encoder = torch.hub.load(dino_repo, 'dinov2_vitb14',
                                       source='local', pretrained=True)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Get patch token dim (768 for ViT-B/14)
        self.patch_dim = 768
        # DINOv2 with 224x224 input, patch_size=14: 16x16 = 256 patches
        self.num_patches = 256

        # Proprio and action encoders
        self.proprio_encoder = ProprioEncoder(odom_dim + imu_dim, proprio_emb_dim)
        self.action_encoder = ActionEncoder(act_dim, action_emb_dim)

        # Project proprio/action to patch dim for token concatenation
        self.proprio_proj = nn.Linear(proprio_emb_dim, self.patch_dim)
        self.action_proj = nn.Linear(action_emb_dim, self.patch_dim)

        # Causal ViT predictor
        self.predictor = CausalViTPredictor(
            dim=self.patch_dim, depth=vit_depth, heads=vit_heads,
            dim_head=64, mlp_dim=self.patch_dim * 2,
        )

        # State head: mean-pool predicted patches → pose
        self.state_head = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim),
        )

    def encode_visual(self, rgb):
        """Encode RGB frames to patch tokens.
        Args: rgb: (B*T, 3, 224, 224)
        Returns: (B*T, num_patches, 768)
        """
        with torch.no_grad():
            # DINOv2 forward_features returns dict with 'x_norm_patchtokens'
            out = self.encoder.forward_features(rgb)
            patches = out['x_norm_patchtokens']  # (B*T, 256, 768)
        return patches

    def forward(self, rgb, odom, imu, action, horizon):
        """Forward pass for training.

        Args:
            rgb: (B, T, 3, H, W)
            odom: (B, T, D_odom)
            imu: (B, T, D_imu)
            action: (B, T, D_act)
            horizon: prediction horizon H (uses num_hist frames to predict H frames ahead)

        Returns:
            dict with z_pred_pooled, z_tgt_pooled, state_pred, for loss computation
        """
        B, T = rgb.shape[0], rgb.shape[1]
        H = min(horizon, T - self.num_hist)
        total_frames = self.num_hist + H

        # Encode all frames to patches
        rgb_flat = rearrange(rgb[:, :total_frames], 'b t c h w -> (b t) c h w')
        patches = self.encode_visual(rgb_flat)  # (B*T, 256, 768)
        patches = rearrange(patches, '(b t) p d -> b t p d', b=B)

        # Encode proprio and action
        proprio = torch.cat([odom[:, :total_frames], imu[:, :total_frames]], dim=-1)
        proprio_emb = self.proprio_encoder(proprio)  # (B, T, 10)
        proprio_tokens = self.proprio_proj(proprio_emb).unsqueeze(2)  # (B, T, 1, 768)

        action_emb = self.action_encoder(action[:, :total_frames])  # (B, T, 10)
        action_tokens = self.action_proj(action_emb).unsqueeze(2)  # (B, T, 1, 768)

        # Concatenate: patches + proprio_token + action_token per frame
        # (B, T, 258, 768)
        z_all = torch.cat([patches, proprio_tokens, action_tokens], dim=2)
        tokens_per_frame = z_all.shape[2]  # 258

        # Source: first num_hist frames
        z_src = z_all[:, :self.num_hist]  # (B, num_hist, 258, 768)
        z_src_flat = rearrange(z_src, 'b t p d -> b (t p) d')

        # Target: frames shifted by 1 (predict next frame from history)
        z_tgt = patches[:, 1:self.num_hist + 1]  # (B, num_hist, 256, 768) — visual only

        # Predict
        z_pred_flat = self.predictor(z_src_flat)  # (B, num_hist*258, 768)
        z_pred = rearrange(z_pred_flat, 'b (t p) d -> b t p d', t=self.num_hist)

        # Extract visual patches from prediction (exclude proprio/action tokens)
        z_pred_visual = z_pred[:, :, :self.num_patches, :]  # (B, num_hist, 256, 768)

        # Mean-pool patches for state prediction
        z_pred_pooled = z_pred_visual.mean(dim=2)  # (B, num_hist, 768)
        z_tgt_pooled = z_tgt.mean(dim=2)  # (B, num_hist, 768)

        # Decode state from predicted pooled features
        state_pred = self.state_head(z_pred_pooled)  # (B, num_hist, state_dim)

        return {
            'z_pred': z_pred_pooled,
            'z_targets': z_tgt_pooled.detach(),
            'state_pred': state_pred,
        }

    def rollout(self, rgb_init, odom_init, imu_init, action_seq):
        """Autoregressive rollout for evaluation.

        Args:
            rgb_init: (B, num_hist, 3, H, W) — initial history frames
            odom_init: (B, num_hist, D_odom)
            imu_init: (B, num_hist, D_imu)
            action_seq: (B, H, D_act) — future actions

        Returns:
            state_preds: (B, H, state_dim) — predicted states for each future step
        """
        B = rgb_init.shape[0]
        H = action_seq.shape[1]

        # Encode initial history frames
        rgb_flat = rearrange(rgb_init, 'b t c h w -> (b t) c h w')
        patches = self.encode_visual(rgb_flat)
        patches = rearrange(patches, '(b t) p d -> b t p d', b=B)

        proprio = torch.cat([odom_init, imu_init], dim=-1)
        proprio_emb = self.proprio_encoder(proprio)
        proprio_tokens = self.proprio_proj(proprio_emb).unsqueeze(2)

        # For initial history, use zero actions (or actual initial actions)
        act_emb = self.action_encoder(torch.zeros(B, self.num_hist, action_seq.shape[-1], device=action_seq.device))
        act_tokens = self.action_proj(act_emb).unsqueeze(2)

        z_history = torch.cat([patches, proprio_tokens, act_tokens], dim=2)

        state_preds = []
        for t in range(H):
            # Flatten history and predict
            z_flat = rearrange(z_history[:, -self.num_hist:], 'b t p d -> b (t p) d')
            z_pred_flat = self.predictor(z_flat)
            z_pred = rearrange(z_pred_flat, 'b (t p) d -> b t p d', t=self.num_hist)

            # Get last frame's predicted visual patches
            z_new_visual = z_pred[:, -1:, :self.num_patches, :]  # (B, 1, 256, 768)

            # Decode state
            pooled = z_new_visual.mean(dim=2)  # (B, 1, 768)
            state = self.state_head(pooled)  # (B, 1, state_dim)
            state_preds.append(state)

            # Create next history token with predicted patches + action
            act_t = self.action_encoder(action_seq[:, t:t+1])
            act_tok = self.action_proj(act_t).unsqueeze(2)
            # Use zero proprio for predicted frames (we don't have ground truth)
            zero_proprio = torch.zeros(B, 1, proprio_tokens.shape[-1], device=z_history.device)
            z_new = torch.cat([z_new_visual, zero_proprio.unsqueeze(2), act_tok], dim=2)
            z_history = torch.cat([z_history, z_new], dim=1)

        return torch.cat(state_preds, dim=1)  # (B, H, state_dim)


def main():
    ap = argparse.ArgumentParser(description="DINO-WM comparison training")
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--num_hist', type=int, default=3)
    ap.add_argument('--vit_depth', type=int, default=4)
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_dir', default='runs/dinowm_comparison')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load dataset with position deltas + normalization (same as our pipeline)
    print(f"Loading dataset: {args.dataset}")
    ds_init = HFWindowDataset(args.dataset, img_size=args.img_size)

    all_ids = ds_init._window_ids
    val_ids = set(eid for eid in all_ids if eid % 5 == 0)
    train_ids = [eid for eid in all_ids if eid not in val_ids]

    norm_stats = ds_init.compute_norm_stats(episode_ids=train_ids)
    norm_stats.save(os.path.join(args.out_dir, 'norm_stats.pt'))
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")

    ds_train = HFWindowDataset(args.dataset, img_size=args.img_size, norm_stats=norm_stats)
    ds_train._window_ids = train_ids
    ds_train._window_rows = {k: v for k, v in ds_train._window_rows.items() if k in set(train_ids)}

    ds_val = HFWindowDataset(args.dataset, img_size=args.img_size, norm_stats=norm_stats)
    ds_val._window_ids = [eid for eid in all_ids if eid in val_ids]
    ds_val._window_rows = {k: v for k, v in ds_val._window_rows.items() if k in val_ids}

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate_hf,
                          drop_last=True, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_hf,
                        drop_last=True, pin_memory=True)

    odom_dim = ds_init.D_odom
    imu_dim = ds_init.D_imu
    act_dim = ds_init.D_act
    state_dim = min(6, odom_dim)

    print(f"Building DinoWM: odom={odom_dim}, imu={imu_dim}, act={act_dim}, "
          f"num_hist={args.num_hist}, vit_depth={args.vit_depth}")

    model = DinoWMComparison(
        odom_dim=odom_dim, imu_dim=imu_dim, act_dim=act_dim,
        state_dim=state_dim, num_hist=args.num_hist,
        vit_depth=args.vit_depth, vit_heads=8,
        device=args.device,
    ).to(args.device)

    # Only train predictor + encoders + state head (encoder frozen)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    # Training loop
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dl_train:
            rgb = batch.rgb.to(args.device)
            odom = batch.odom.to(args.device)
            imu = batch.imu.to(args.device)
            action = batch.action.to(args.device)

            T = rgb.shape[1]
            if T < args.num_hist + 1:
                continue

            outputs = model(rgb, odom, imu, action, horizon=args.horizon)

            # Latent prediction loss
            loss_z = F.mse_loss(outputs['z_pred'], outputs['z_targets'])

            # State prediction loss (same as our pipeline)
            H = outputs['state_pred'].shape[1]
            state_gt = odom[:, 1:H+1, :state_dim]
            loss_state = F.mse_loss(outputs['state_pred'], state_gt)

            loss = loss_z + loss_state

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            epoch_loss += loss.item()
            epoch_steps += 1

        avg_loss = epoch_loss / max(epoch_steps, 1)

        # Val loss
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in dl_val:
                rgb = batch.rgb.to(args.device)
                odom = batch.odom.to(args.device)
                imu = batch.imu.to(args.device)
                action = batch.action.to(args.device)
                T = rgb.shape[1]
                if T < args.num_hist + 1:
                    continue
                outputs = model(rgb, odom, imu, action, horizon=args.horizon)
                loss_z = F.mse_loss(outputs['z_pred'], outputs['z_targets'])
                H = outputs['state_pred'].shape[1]
                state_gt = odom[:, 1:H+1, :state_dim]
                loss_state = F.mse_loss(outputs['state_pred'], state_gt)
                val_loss += (loss_z + loss_state).item()
                val_steps += 1

        avg_val = val_loss / max(val_steps, 1)
        print(f"[Epoch {epoch:03d}] train_loss={avg_loss:.4f}  val_loss={avg_val:.4f}")

        # Save checkpoint
        torch.save({
            'model': {k: v for k, v in model.state_dict().items() if 'encoder.' not in k},
            'epoch': epoch,
            'args': vars(args),
            'model_config': {
                'odom_dim': odom_dim, 'imu_dim': imu_dim,
                'act_dim': act_dim, 'state_dim': state_dim,
            },
            'has_norm_stats': True,
        }, os.path.join(args.out_dir, 'latest.pt'))

    train_time = time.time() - t0
    print(f"Training complete. Time: {train_time:.1f}s")

    # ── Evaluation: ADE/FDE (same protocol as our pipeline) ──
    print("\n" + "=" * 60)
    print("Evaluating trajectory prediction (ADE/FDE)...")
    print("=" * 60)

    model.eval()
    horizons = [8, 16, 25]
    max_h = max(horizons)
    results = {h: {"ade": [], "fde": []} for h in horizons}

    with torch.no_grad():
        for idx in range(len(ds_val)):
            sample = ds_val[idx]
            T = sample["odom"].shape[0]
            if T < max_h + args.num_hist:
                continue

            rgb = torch.from_numpy(sample["rgb"]).unsqueeze(0).to(args.device)
            odom = torch.from_numpy(sample["odom"]).unsqueeze(0).to(args.device)
            imu = torch.from_numpy(sample["imu"]).unsqueeze(0).to(args.device)
            action = torch.from_numpy(sample["action"]).unsqueeze(0).to(args.device)

            # Use first num_hist frames as history
            rgb_init = rgb[:, :args.num_hist]
            odom_init = odom[:, :args.num_hist]
            imu_init = imu[:, :args.num_hist]
            action_future = action[:, args.num_hist:args.num_hist + max_h]

            state_preds = model.rollout(rgb_init, odom_init, imu_init, action_future)
            # (1, max_h, state_dim)

            # Denormalize
            odom_mean = norm_stats.odom_mean[:state_dim].to(args.device)
            odom_std = norm_stats.odom_std[:state_dim].to(args.device)
            pred_denorm = state_preds * odom_std + odom_mean

            # GT: odom at frames num_hist .. num_hist+max_h (denormalized)
            gt_odom = odom[:, args.num_hist:args.num_hist + max_h, :state_dim]
            gt_denorm = gt_odom * odom_std + odom_mean

            pred_pos = pred_denorm[:, :, :3].squeeze(0)  # (max_h, 3)
            gt_pos = gt_denorm[:, :, :3].squeeze(0)

            displacements = torch.norm(pred_pos - gt_pos, dim=-1)  # (max_h,)

            for h in horizons:
                if h <= displacements.shape[0]:
                    ade = displacements[:h].mean().item()
                    fde = displacements[h - 1].item()
                    results[h]["ade"].append(ade)
                    results[h]["fde"].append(fde)

    # Print results
    print(f"\n{'Horizon':>10s} | {'ADE':>8s} | {'FDE':>8s} | {'N':>5s}")
    print("-" * 40)
    summary = {}
    for h in horizons:
        r = results[h]
        key = f"{h/5.0:.1f}s"
        ade = float(np.mean(r["ade"])) if r["ade"] else 0.0
        fde = float(np.mean(r["fde"])) if r["fde"] else 0.0
        n = len(r["ade"])
        print(f"{key:>10s} | {ade:8.4f} | {fde:8.4f} | {n:5d}")
        summary[key] = {"ade": ade, "fde": fde, "count": n, "horizon_steps": h}
    print("=" * 40)

    # Save
    out_path = os.path.join(args.out_dir, 'trajectory_eval.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {out_path}")

    metrics_path = os.path.join(args.out_dir, 'eval_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({"train_time_s": train_time, "experiment": "dinowm_comparison"}, f, indent=2)


if __name__ == '__main__':
    main()
