import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from world_model.data import HFWindowDataset, collate_hf
from world_model.backbones import create_model

# Add src to pythonpath
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(REPO_ROOT / "src"))

class PosePredictor(nn.Module):
    """Linear Probe / MLP for predicting 7-DoF pose from a frozen vision backbone."""
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        # Common trunk to extract relevant geometric features
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1) # Prevents overfitting to specific sim textures
        )

        # Position Head: Output (x, y, z)
        self.pos_head = nn.Linear(hidden_dim, 3)

        # Orientation Head: Output (qw, qx, qy, qz)
        self.ori_head = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Input: (B, input_dim) features from frozen backbone"""
        features = self.trunk(x)
        pos = self.pos_head(features)
        ori = self.ori_head(features)

        # Quaternions MUST be normalized to be valid rotations
        ori = F.normalize(ori, p=2, dim=1)

        # Return as [batch, 7]
        return torch.cat((pos, ori), dim=1)

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Pose MSE for Frozen Vision Backbones")
    parser.add_argument("--dataset", type=str, required=True, help="HF Dataset ID (e.g. lquan9/collect1)")
    parser.add_argument("--backbone", type=str, required=True, help="Backbone type: cnn, dinov2, vjepa2")
    parser.add_argument("--backbone_name", type=str, default="", help="Specific model name (e.g. dinov2_vitb14)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for orientation MSE loss")
    return parser.parse_args()

def main():
    """main"""
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset {args.dataset}...")
    dataset = HFWindowDataset(
        repo_id=args.dataset,
        window_size=1,
        img_size=args.img_size
    )

    # Split into Train/Val (80/20) based on episodes
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, collate_fn=collate_hf, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, collate_fn=collate_hf, pin_memory=True)

    bkw = {"model_name": args.backbone_name} if args.backbone_name else {}
    if args.backbone in ["dinov2", "vjepa2"]:
        bkw["in_ch"] = 3
    else:
        bkw["in_ch"] = 4

    print(f"Initializing backbone: {args.backbone} ({args.backbone_name})")
    bundle = create_model(args.backbone, **bkw)
    backbone_model = bundle.model
    backbone_model.eval()
    for p in backbone_model.parameters():
        p.requires_grad = False
    backbone_model.to(device)

    feature_dim = bundle.feature_dim
    print(f"Initializing PosePredictor with input dim: {feature_dim}")
    probe = PosePredictor(input_dim=feature_dim).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    
    def weighted_pose_loss(pred_pose, gt_pose):
        """weighted MSE loss that balances position and orientation errors. Beta controls the relative weight of orientation loss."""
        # pred_pose: (B, 7), gt_pose: (B, 13) where 0:3 is pos, 3:7 is ori
        pos_gt = gt_pose[:, 0:3]
        ori_gt = gt_pose[:, 3:7]
 
        pos_pred = pred_pose[:, 0:3]
        ori_pred = pred_pose[:, 3:7]

        loss_pos = F.mse_loss(pos_pred, pos_gt)
        loss_ori = F.mse_loss(ori_pred, ori_gt)
        return loss_pos + args.beta * loss_ori, loss_pos, loss_ori

    print("Starting Probe Training...")
    for epoch in range(args.epochs):
        probe.train()
        train_loss = 0.0
        train_pos = 0.0
        train_ori = 0.0
        start_t = time.time()

        for batch in train_loader:
            # HFWindowDataset returns 5D tensors (B, T, C, H, W). We take T=0 since window_size=1
            rgb = batch.rgb[:, 0].to(device)
            depth = batch.depth[:, 0].to(device)
            odom_gt = batch.odom[:, 0].to(device)

            if bundle.modality == "image" and bundle.feature_dim == 384: # DINOv2 etc.
                x = rgb
            elif bundle.modality == "image":
                x = torch.cat([rgb, depth], dim=1)
            else: # Video (V-JEPA2) expects 5D tubelets (B, C, T, H, W)
                x = rgb.unsqueeze(2)  # (B, 3, 1, H, W)

            with torch.no_grad():
                features = backbone_model(x)
                # DINOv2 returns a vector directly. V-JEPA2 returns a spatial token grid, we must pool it.
                if len(features.shape) > 2:
                    features = features.mean(dim=[d for d in range(2, len(features.shape))])

            pred_pose = probe(features)
            loss, lp, lo = weighted_pose_loss(pred_pose, odom_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pos += lp.item()
            train_ori += lo.item()

        train_loss /= len(train_loader)
        train_pos /= len(train_loader)
        train_ori /= len(train_loader)

        # Validation
        probe.eval()
        val_loss, val_pos, val_ori = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch.rgb[:, 0].to(device)
                depth = batch.depth[:, 0].to(device)
                odom_gt = batch.odom[:, 0].to(device)

                if bundle.modality == "image" and bundle.feature_dim == 384:
                    x = rgb
                elif bundle.modality == "image":
                    x = torch.cat([rgb, depth], dim=1)
                else:
                    x = rgb.unsqueeze(2)

                features = backbone_model(x)
                if len(features.shape) > 2:
                    features = features.mean(dim=[d for d in range(2, len(features.shape))])

                pred_pose = probe(features)
                loss, lp, lo = weighted_pose_loss(pred_pose, odom_gt)

                val_loss += loss.item()
                val_pos += lp.item()
                val_ori += lo.item()

        val_loss /= len(val_loader)
        val_pos /= len(val_loader)
        val_ori /= len(val_loader)

        elapsed = time.time() - start_t
        print(f"Epoch [{epoch+1}/{args.epochs}] | Time: {elapsed:.1f}s")
        print(f"  Train -> Loss: {train_loss:.4f} (Pos: {train_pos:.4f}, Ori: {train_ori:.4f})")
        print(f"  Val   -> Loss: {val_loss:.4f} (Pos: {val_pos:.4f}, Ori: {val_ori:.4f})")

    print("Evaluation Complete.")

if __name__ == "__main__":
    main()
