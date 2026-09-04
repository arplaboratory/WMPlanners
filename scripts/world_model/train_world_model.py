from __future__ import annotations
import argparse
import copy
import json
import os
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from world_model.data import (
    collate_for_dataset,
    open_window_dataset,
)
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


@torch.no_grad()
def evaluate_pose(model, dataloader, cfg: TrainConfig, threshold: float = 0.1):
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
            pred_state = out.state
            gt_state = odom[:, t, :pred_state.shape[-1]]
            mse = F.mse_loss(pred_state, gt_state, reduction='sum').item()
            l2 = torch.norm(pred_state - gt_state, dim=-1)
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


class Trainer:
    def __init__(self, args):
        self.args = args
        self.local_rank = 0
        if args.ddp:
            self.local_rank = setup_ddp()
            args.device = f"cuda:{self.local_rank}"
        self.main_process = is_main_process(args)

        seed_everything(args.seed)
        os.makedirs(args.out_dir, exist_ok=True)

        if args.experiment_name is None:
            ds_short = args.data_path.replace('/', '_')
            args.experiment_name = f"{ds_short}_{args.backbone}"

        self.use_wandb = args.wandb and os.environ.get('WANDB_API_KEY') and self.main_process
        if self.use_wandb:
            import wandb
            wandb.init(
                project=os.environ.get('WANDB_PROJECT', 'world_model'),
                name=os.environ.get('WANDB_RUN_NAME', args.experiment_name),
                config=vars(args),
            )
        elif args.wandb and self.main_process:
            print("[WARN] --wandb enabled but WANDB_API_KEY not set. Logging disabled.")

        self.ds = open_window_dataset(
            dataset_type=args.dataset_type, path=args.data_path, split=args.split,
            image_size=args.img_size, window_size=args.num_hist + args.horizon,
        )
        self.collate_fn = collate_for_dataset(self.ds)
        self.dataset_provenance = {
            key: self.ds.manifest.get(key)
            for key in ("dataset_id", "dataset_hash", "release_id", "release_hash", "recipe_sha256")
            if hasattr(self.ds, "manifest") and self.ds.manifest.get(key) is not None
        }

        sampler = None
        shuffle = True
        if args.ddp:
            sampler = DistributedSampler(self.ds, shuffle=True)
            shuffle = False

        self.dl = DataLoader(
            self.ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=args.num_workers, collate_fn=self.collate_fn, drop_last=True,
            pin_memory=True if "cuda" in str(args.device) else False,
        )

        odom_dim = self.ds.D_odom
        imu_dim = self.ds.D_imu
        act_dim = self.ds.D_act
        state_dim = min(6, odom_dim)

        self.model = WorldModel(
            odom_dim=odom_dim,
            imu_dim=imu_dim,
            act_dim=act_dim,
            state_dim=state_dim,
            backbone=args.backbone,
            backbone_kwargs={"model_name": args.backbone_name, "device": args.device},
            freeze_backbone=args.freeze_backbone,
            dynamics_type="mlp" if args.dynamics_type == "td-jepa" else args.dynamics_type,
        ).to(args.device)

        self.target_model = None
        if args.dynamics_type == "td-jepa":
            self.target_model = copy.deepcopy(self.model)
            for p in self.target_model.parameters():
                p.requires_grad = False

        if args.ddp:
            find_unused = args.dynamics_type == "td-jepa"
            self.model = DDP(self.model, device_ids=[self.local_rank],
                             find_unused_parameters=find_unused)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.cfg = TrainConfig(
            device=args.device, lr=args.lr, horizon=args.horizon,
            num_hist=args.num_hist,
        )

        self.start_epoch = 0
        checkpoint_path = args.resume
        if checkpoint_path is None:
            latest = os.path.join(args.out_dir, 'latest.pt')
            if os.path.exists(latest):
                checkpoint_path = latest
                if self.main_process:
                    print(f"[RECOVERY] Found {latest}. Resuming automatically...")

        if checkpoint_path and os.path.exists(checkpoint_path):
            if self.main_process:
                print(f"Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=args.device)
            raw_model = self.model.module if isinstance(self.model, DDP) else self.model
            raw_model.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt:
                self.opt.load_state_dict(ckpt['optimizer'])
            self.start_epoch = ckpt.get('epoch', -1) + 1
            if self.main_process:
                print(f"Resuming from epoch {self.start_epoch}")

    def run(self):
        args = self.args
        if self.main_process:
            print(f"[INFO] Starting training: epochs={args.epochs}, batch_size={args.batch_size}, "
                  f"backbone={args.backbone}, dynamics={args.dynamics_type}, ddp={args.ddp}")

        t0 = time.time()
        step = 0

        for epoch in range(self.start_epoch, args.epochs):
            if self.dl.sampler is not None and hasattr(self.dl.sampler, 'set_epoch'):
                self.dl.sampler.set_epoch(epoch)

            self.model.train()
            epoch_loss = 0.0
            epoch_steps = 0

            for batch in self.dl:
                loss, logs = rollout_loss(self.model, batch, self.cfg, target_model=self.target_model)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                if self.target_model is not None:
                    with torch.no_grad():
                        for op, tp in zip(self.model.parameters(), self.target_model.parameters()):
                            tp.data.mul_(args.ema_tau).add_(op.data, alpha=1.0 - args.ema_tau)

                epoch_loss += logs['loss']
                epoch_steps += 1

                if step % 50 == 0 and self.main_process:
                    print(f"epoch={epoch:03d} step={step:06d} loss={logs['loss']:.4f} "
                          f"z={logs['loss_z']:.4f} state={logs['loss_state']:.4f} "
                          f"risk={logs['loss_risk']:.4f}")
                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss": logs['loss'],
                            "train/loss_z": logs['loss_z'],
                            "train/loss_state": logs['loss_state'],
                            "train/loss_risk": logs['loss_risk'],
                            "train/step": step,
                            "train/epoch": epoch,
                        }, step=step)
                step += 1

            if self.main_process:
                avg_loss = epoch_loss / max(epoch_steps, 1)
                print(f"[Epoch {epoch:03d}] avg_loss={avg_loss:.4f}")
                if self.use_wandb:
                    import wandb
                    wandb.log({"epoch/avg_loss": avg_loss, "epoch": epoch}, step=step)

                raw_model = self.model.module if isinstance(self.model, DDP) else self.model
                ckpt = {
                    'model': raw_model.state_dict(),
                    'optimizer': self.opt.state_dict(),
                    'epoch': epoch,
                    'args': vars(args),
                    'model_config': {
                        'odom_dim': self.ds.D_odom,
                        'imu_dim': self.ds.D_imu,
                        'act_dim': self.ds.D_act,
                        'state_dim': min(6, self.ds.D_odom),
                    },
                    'dataset_provenance': self.dataset_provenance,
                }
                torch.save(ckpt, os.path.join(args.out_dir, f'epoch_{epoch:03d}.pt'))
                torch.save(ckpt, os.path.join(args.out_dir, 'latest.pt'))

            if args.ddp:
                dist.barrier()

        train_time = time.time() - t0

        if self.main_process:
            print(f"Training complete. Total time: {train_time:.1f}s")
            print("\n" + "=" * 60)
            print("[Evaluation] Computing pose MSE and accuracy ...")
            print("=" * 60)

            eval_dl = DataLoader(
                self.ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, collate_fn=self.collate_fn, drop_last=False,
                pin_memory=True if "cuda" in str(args.device) else False,
            )
            eval_metrics = evaluate_pose(self.model, eval_dl, self.cfg, threshold=args.pose_threshold)

            print(f"  Pose MSE:      {eval_metrics['pose_mse']:.6f}")
            print(f"  Pose Accuracy: {eval_metrics['pose_accuracy']:.4f}  "
                  f"(threshold={args.pose_threshold})")

            if self.use_wandb:
                import wandb
                wandb.log({
                    "eval/pose_mse": eval_metrics['pose_mse'],
                    "eval/pose_accuracy": eval_metrics['pose_accuracy'],
                })
                wandb.summary["pose_mse"] = eval_metrics["pose_mse"]
                wandb.summary["pose_accuracy"] = eval_metrics["pose_accuracy"]
                wandb.summary["train_time_s"] = train_time
                wandb.finish()

            metrics_path = os.path.join(args.out_dir, 'eval_metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {**eval_metrics, "train_time_s": train_time, "experiment": args.experiment_name},
                    f, indent=2,
                )
            print(f"  Metrics saved to {metrics_path}")
            print(f"\nLatest checkpoint: {os.path.join(args.out_dir, 'latest.pt')}")

        if args.ddp:
            cleanup_ddp()


def build_parser(parents=None):
    ap = argparse.ArgumentParser(
        description="Train a standard WorldModel",
        parents=parents or [],
        add_help=parents is None,
    )
    ap.add_argument('--data_path', '--dataset', '--data_dir', type=str, required=True,
                    dest='data_path', help='HuggingFace dataset ID or local path')
    ap.add_argument('--dataset_type', type=str, default='hf', choices=['hf', 'npz', 'gold'])
    ap.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    ap.add_argument('--out_dir', type=str, default='runs')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--horizon', type=int, default=16)
    ap.add_argument('--num_hist', type=int, default=1, help='Number of history frames (1=original, 3=multi-frame context)')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--backbone', type=str, default='cnn')
    ap.add_argument('--backbone_name', type=str, default='dinov2_vitb14')
    ap.add_argument('--freeze_backbone', action='store_true')
    ap.add_argument('--dynamics_type', type=str, default='mlp', choices=['mlp', 'rssm', 'td-jepa'])
    ap.add_argument('--ema_tau', type=float, default=0.996)
    ap.add_argument('--experiment_name', type=str, default=None)
    ap.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--pose_threshold', type=float, default=0.1)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--ddp', action='store_true', default=False)
    return ap


def main():
    args = build_parser().parse_args()
    Trainer(args).run()


if __name__ == '__main__':
    main()
