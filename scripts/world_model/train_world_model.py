from __future__ import annotations
import argparse
import os
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from world_model.data import HFWindowDataset, NPZEpisodeDataset, collate_hf, collate_npz
from world_model.architecture import WorldModel
from world_model.engine.trainer import TrainConfig, rollout_loss
from world_model.utils import seed_everything


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process(args) -> bool:
    if not args.ddp:
        return True
    return int(os.environ.get("RANK", 0)) == 0


# Post-training evaluation
@torch.no_grad()
def evaluate_pose(model, dataloader, cfg: TrainConfig, threshold: float = 0.1):
    """Evaluate pose prediction quality over the full dataset.

    Returns a dict with:
      - pose_mse: mean squared error of state_head predictions vs GT odom
      - pose_accuracy: fraction of predictions within `threshold` (L2 norm)
    """
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    device = torch.device(cfg.device)
    total_mse = 0.0
    total_within = 0
    total_count = 0

    for batch in dataloader:
        rgb = batch.rgb.to(device)
        depth = batch.depth.to(device)
        imu = batch.imu.to(device)
        odom = batch.odom.to(device)

        B, T = rgb.shape[0], rgb.shape[1]

        for t in range(T):
            out = raw_model.encode(rgb[:, t], depth[:, t], odom[:, t], imu[:, t])
            pred_state = out.state                          # (B, state_dim)
            gt_state = odom[:, t, :pred_state.shape[-1]]    # (B, state_dim)

            mse = F.mse_loss(pred_state, gt_state, reduction='sum').item()
            l2 = torch.norm(pred_state - gt_state, dim=-1)  # (B,)
            within = (l2 < threshold).sum().item()

            total_mse += mse
            total_within += within
            total_count += B

    if total_count == 0:
        return {"pose_mse": 0.0, "pose_accuracy": 0.0}

    return {
        "pose_mse": total_mse / total_count,
        "pose_accuracy": total_within / total_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', '--data_dir', type=str, required=True, dest='dataset',
                    help='Path to dataset (HuggingFace dir/ID or local .npz dir)')
    ap.add_argument('--dataset_type', type=str, default='hf', choices=['hf', 'npz'],
                    help='Type of dataset to load')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', type=str, default='runs')
    ap.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (autodetects "latest.pt")')
    # Backbone configs
    ap.add_argument('--backbone', type=str, default='cnn', help='cnn, dinov2, or vjepa2')
    ap.add_argument('--backbone_name', type=str, default='dinov2_vitb14', help='Specific model variant for the backbone')
    ap.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone weights during training')
    ap.add_argument('--dynamics_type', type=str, default='mlp', choices=['mlp', 'rssm', 'td-jepa'], help='Type of dynamics module')
    ap.add_argument('--ema_tau', type=float, default=0.996, help='EMA coefficient for TD-JEPA target model')
    # Experiment / logging
    ap.add_argument('--experiment_name', type=str, default=None, help='WANDB run name (default: auto-generated)')
    ap.add_argument('--wandb', action='store_true', default=False, help='Enable WANDB logging')
    ap.add_argument('--pose_threshold', type=float, default=0.1, help='L2 threshold for pose accuracy metric')
    ap.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers')
    # DDP
    ap.add_argument('--ddp', action='store_true', default=False, help='Enable DistributedDataParallel training')
    args = ap.parse_args()

    local_rank = 0
    if args.ddp:
        local_rank = setup_ddp()
        args.device = f"cuda:{local_rank}"

    main_process = is_main_process(args)

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.experiment_name is None:
        ds_short = args.dataset.replace('/', '_')
        args.experiment_name = f"{ds_short}_{args.backbone}"

    # WANDB init (rank 0 only)
    use_wandb = args.wandb and os.environ.get('WANDB_API_KEY') and main_process
    if use_wandb:
        import wandb
        wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'world_model'),
            name=os.environ.get('WANDB_RUN_NAME', args.experiment_name),
            config=vars(args),
        )
    elif args.wandb and main_process:
        print("[WARN] --wandb enabled but WANDB_API_KEY not set. Logging disabled.")
        use_wandb = False

    # Dataset
    if args.dataset_type == 'hf':
        ds = HFWindowDataset(args.dataset, img_size=args.img_size)
        collate_fn = collate_hf
    else:
        ds = NPZEpisodeDataset(args.dataset, window=args.horizon + 1, img_size=args.img_size)
        collate_fn = collate_npz

    sampler = None
    shuffle = True
    if args.ddp:
        sampler = DistributedSampler(ds, shuffle=True)
        shuffle = False

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
                    num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
                    pin_memory=True if "cuda" in str(args.device) else False)

    odom_dim = ds.D_odom
    imu_dim = ds.D_imu
    act_dim = ds.D_act
    state_dim = min(6, odom_dim)

    # Model
    model = WorldModel(
        odom_dim=odom_dim,
        imu_dim=imu_dim,
        act_dim=act_dim,
        state_dim=state_dim,
        backbone=args.backbone,
        backbone_kwargs={"model_name": args.backbone_name, "device": args.device},
        freeze_backbone=args.freeze_backbone,
        dynamics_type="mlp" if args.dynamics_type == "td-jepa" else args.dynamics_type,
    ).to(args.device)

    target_model = None
    if args.dynamics_type == "td-jepa":
        import copy
        target_model = copy.deepcopy(model)
        for p in target_model.parameters():
            p.requires_grad = False

    if args.ddp:
        # rollout_loss calls model(...) (i.e. DDP.forward()) so DDP hooks fire
        # correctly. For td-jepa the online backbone is unused (targets come
        # from the frozen EMA model), so we need find_unused_parameters=True.
        if args.dynamics_type == "td-jepa":
            model = DDP(model, device_ids=[local_rank],
                        find_unused_parameters=True)
        else:
            model = DDP(model, device_ids=[local_rank])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    cfg = TrainConfig(device=args.device, lr=args.lr, horizon=args.horizon)
    
    # Auto-resume logic
    start_epoch = 0
    checkpoint_path = args.resume
    if checkpoint_path is None:
        # Check if latest.pt exists in out_dir
        latest = os.path.join(args.out_dir, 'latest.pt')
        if os.path.exists(latest):
            checkpoint_path = latest
            if main_process:
                print(f"[RECOVERY] Found {latest}. Resuming automatically...")

    if checkpoint_path and os.path.exists(checkpoint_path):
        if main_process:
            print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=args.device)
        
        # Load weights into the raw model
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(ckpt['model'])
        
        if 'optimizer' in ckpt:
            opt.load_state_dict(ckpt['optimizer'])
        
        start_epoch = ckpt.get('epoch', -1) + 1
        if main_process:
            print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    if main_process:
        print(f"[INFO] Starting training: epochs={args.epochs}, batch_size={args.batch_size}, "
              f"backbone={args.backbone}, dynamics={args.dynamics_type}, ddp={args.ddp}")
    t0 = time.time()
    step = 0
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)  # Ensure proper shuffling each epoch

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dl:
            loss, logs = rollout_loss(model, batch, cfg, target_model=target_model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if target_model is not None:
                with torch.no_grad():
                    for online_p, target_p in zip(model.parameters(), target_model.parameters()):
                        target_p.data.mul_(args.ema_tau).add_(online_p.data, alpha=1.0 - args.ema_tau)

            epoch_loss += logs['loss']
            epoch_steps += 1

            if step % 50 == 0 and main_process:
                print(f"epoch={epoch:03d} step={step:06d} loss={logs['loss']:.4f} z={logs['loss_z']:.4f} state={logs['loss_state']:.4f} risk={logs['loss_risk']:.4f}")
                if use_wandb:
                    wandb.log({
                        "train/loss": logs['loss'],
                        "train/loss_z": logs['loss_z'],
                        "train/loss_state": logs['loss_state'],
                        "train/loss_risk": logs['loss_risk'],
                        "train/step": step,
                        "train/epoch": epoch,
                    }, step=step)
            step += 1

        # Epoch summary (rank 0 only)
        if main_process:
            avg_loss = epoch_loss / max(epoch_steps, 1)
            print(f"[Epoch {epoch:03d}] avg_loss={avg_loss:.4f}")
            if use_wandb:
                wandb.log({"epoch/avg_loss": avg_loss, "epoch": epoch}, step=step)

            # Save checkpoint each epoch (rank 0 only)
            raw_model = model.module if isinstance(model, DDP) else model
            ckpt = {
                'model': raw_model.state_dict(),
                'optimizer': opt.state_dict(),
                'epoch': epoch,
                'args': vars(args),
                'model_config': {
                    'odom_dim': odom_dim,
                    'imu_dim': imu_dim,
                    'act_dim': act_dim,
                    'state_dim': state_dim,
                },
            }
            path = os.path.join(args.out_dir, f'epoch_{epoch:03d}.pt')
            torch.save(ckpt, path)
            torch.save(ckpt, os.path.join(args.out_dir, 'latest.pt'))

        # Sync all ranks before next epoch
        if args.ddp:
            dist.barrier()

    train_time = time.time() - t0

    # Post-training evaluation (rank 0 only)
    if main_process:
        print(f"Training complete. Total time: {train_time:.1f}s")

        print("\n" + "=" * 60)
        print("[Evaluation] Computing pose MSE and accuracy ...")
        print("=" * 60)

        eval_dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False,
                             pin_memory=True if "cuda" in str(args.device) else False)
        eval_metrics = evaluate_pose(model, eval_dl, cfg, threshold=args.pose_threshold)

        print(f"  Pose MSE:      {eval_metrics['pose_mse']:.6f}")
        print(f"  Pose Accuracy: {eval_metrics['pose_accuracy']:.4f}  (threshold={args.pose_threshold})")

        if use_wandb:
            wandb.log({
                "eval/pose_mse": eval_metrics['pose_mse'],
                "eval/pose_accuracy": eval_metrics['pose_accuracy'],
            })
            wandb.summary["pose_mse"] = eval_metrics["pose_mse"]
            wandb.summary["pose_accuracy"] = eval_metrics["pose_accuracy"]
            wandb.summary["train_time_s"] = train_time
            wandb.finish()

        # Save final metrics to disk alongside checkpoint
        import json
        metrics_path = os.path.join(args.out_dir, 'eval_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({**eval_metrics, "train_time_s": train_time, "experiment": args.experiment_name}, f, indent=2)
        print(f"  Metrics saved to {metrics_path}")

        print(f"\nLatest checkpoint: {os.path.join(args.out_dir, 'latest.pt')}")

    if args.ddp:
        cleanup_ddp()

if __name__ == '__main__':
    main()
