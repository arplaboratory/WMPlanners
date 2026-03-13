from __future__ import annotations
import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from world_model.data import HFWindowDataset, collate_hf
from world_model.architecture import WorldModel
from world_model.engine.trainer import TrainConfig, rollout_loss
from world_model.utils import seed_everything

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True,
                    help='HuggingFace dataset path (local dir or Hub ID)')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', type=str, default='runs')
    # Backbone configs
    ap.add_argument('--backbone', type=str, default='cnn', help='cnn, dinov2, or vjepa2')
    ap.add_argument('--backbone_name', type=str, default='dinov2_vitb14', help='Specific model variant for the backbone')
    ap.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone weights during training')
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = HFWindowDataset(args.dataset, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                    collate_fn=collate_hf, drop_last=True)

    odom_dim = ds.D_odom
    imu_dim = ds.D_imu
    act_dim = ds.D_act
    state_dim = min(6, odom_dim)

    model = WorldModel(
        odom_dim=odom_dim,
        imu_dim=imu_dim,
        act_dim=act_dim,
        state_dim=state_dim,
        backbone=args.backbone,
        backbone_kwargs={"model_name": args.backbone_name, "device": args.device},
        freeze_backbone=args.freeze_backbone,
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    cfg = TrainConfig(device=args.device, lr=args.lr, horizon=args.horizon)

    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in dl:
            loss, logs = rollout_loss(model, batch, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 50 == 0:
                print(f"epoch={epoch:03d} step={step:06d} loss={logs['loss']:.4f} z={logs['loss_z']:.4f} state={logs['loss_state']:.4f} risk={logs['loss_risk']:.4f}")
            step += 1

        # Save checkpoint each epoch
        ckpt = {
            'model': model.state_dict(),
            'epoch': epoch,
            'args': vars(args),
        }
        path = os.path.join(args.out_dir, f'epoch_{epoch:03d}.pt')
        torch.save(ckpt, path)
        torch.save(ckpt, os.path.join(args.out_dir, 'latest.pt'))

    print(f"Done. Total time: {time.time()-t0:.1f}s")
    print(f"Latest checkpoint: {os.path.join(args.out_dir, 'latest.pt')}")

if __name__ == '__main__':
    main()
