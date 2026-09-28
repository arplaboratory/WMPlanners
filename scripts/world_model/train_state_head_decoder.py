"""Train state_head separately on frozen world model latents.

This script is intended for JEPA-lite checkpoints where latent dynamics may be
learned first and the state decoder is fitted post-hoc.

Usage:
    python scripts/world_model/train_state_head_decoder.py \
        --checkpoint runs/jepa_lite_smoke/latest.pt \
        --dataset lquan9/collect1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from world_model.architecture import WorldModel
from world_model.data import HFWindowDataset, collate_hf
from world_model.data.normalization import NormStats


def load_model_and_stats(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = ckpt["args"]
    mcfg = ckpt["model_config"]

    dynamics_type = args_dict.get("dynamics_type", "mlp")
    if dynamics_type == "td-jepa":
        dynamics_type = "mlp"

    model = WorldModel(
        odom_dim=mcfg["odom_dim"],
        imu_dim=mcfg["imu_dim"],
        act_dim=mcfg["act_dim"],
        state_dim=mcfg["state_dim"],
        backbone=args_dict.get("backbone", "cnn"),
        backbone_kwargs={
            "model_name": args_dict.get("backbone_name", "dinov2_vitb14"),
            "device": device,
        },
        dynamics_type=dynamics_type,
        image_size=args_dict.get("img_size", 224),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)

    norm_dir = str(Path(checkpoint_path).parent)
    norm_path = os.path.join(norm_dir, "norm_stats.pt")
    norm_stats = NormStats.load(norm_path) if os.path.exists(norm_path) else None

    return model, norm_stats, ckpt, args_dict, mcfg


def set_trainable_state_head_only(model: WorldModel):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.state_head.parameters():
        p.requires_grad = True


def train_state_head(
    model: WorldModel,
    dataloader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    max_steps_per_epoch: int,
):
    set_trainable_state_head_only(model)
    model.train()
    opt = torch.optim.AdamW(model.state_head.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(dataloader):
            if max_steps_per_epoch > 0 and step >= max_steps_per_epoch:
                break

            rgb = batch.rgb.to(device)
            depth = batch.depth.to(device)
            imu = batch.imu.to(device)
            odom = batch.odom.to(device)

            bsz, tlen = rgb.shape[0], rgb.shape[1]

            with torch.no_grad():
                out = model.encode(
                    rgb.reshape(bsz * tlen, *rgb.shape[2:]),
                    depth.reshape(bsz * tlen, *depth.shape[2:]),
                    odom.reshape(bsz * tlen, odom.shape[-1]),
                    imu.reshape(bsz * tlen, imu.shape[-1]),
                )
                z = out.z.reshape(bsz, tlen, -1)

            pred_state = model.state_head(z)
            target_state = odom[..., : pred_state.shape[-1]]

            loss = F.mse_loss(pred_state, target_state)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.state_head.parameters(), 1.0)
            opt.step()

            loss_val = float(loss.item())
            epoch_loss += loss_val
            epoch_steps += 1

            if global_step % 50 == 0:
                print(f"epoch={epoch:03d} step={global_step:06d} state_loss={loss_val:.6f}")
            global_step += 1

        avg_loss = epoch_loss / max(epoch_steps, 1)
        history.append({"epoch": epoch, "avg_state_loss": avg_loss, "steps": epoch_steps})
        print(f"[Epoch {epoch:03d}] avg_state_loss={avg_loss:.6f} steps={epoch_steps}")

    return history


def main():
    ap = argparse.ArgumentParser(description="Train state_head decoder separately")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--max_steps_per_epoch", type=int, default=0,
                    help="If >0, cap training steps per epoch for quick runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_checkpoint", type=str, default="",
                    help="Optional path to save updated checkpoint; default is alongside input checkpoint")
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, norm_stats, ckpt, train_args, _ = load_model_and_stats(args.checkpoint, args.device)

    img_size = args.img_size or train_args.get("img_size", 224)
    print(f"Loading dataset: {args.dataset}")
    ds = HFWindowDataset(args.dataset, img_size=img_size, norm_stats=norm_stats)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_hf,
        drop_last=len(ds) > args.batch_size,
        pin_memory="cuda" in str(args.device),
    )

    history = train_state_head(
        model=model,
        dataloader=dl,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        max_steps_per_epoch=args.max_steps_per_epoch,
    )

    ckpt["model"] = model.state_dict()

    ckpt_dir = str(Path(args.checkpoint).parent)
    out_ckpt = args.out_checkpoint or os.path.join(ckpt_dir, "latest_with_state_head.pt")
    torch.save(ckpt, out_ckpt)

    state_head_path = os.path.join(ckpt_dir, "state_head_decoder.pt")
    torch.save(model.state_head.state_dict(), state_head_path)

    metrics_path = os.path.join(ckpt_dir, "state_head_train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"history": history, "checkpoint": out_ckpt}, f, indent=2)

    print("\n" + "=" * 60)
    print("State head training complete")
    print("=" * 60)
    print(f"Updated checkpoint: {out_ckpt}")
    print(f"State head weights: {state_head_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
