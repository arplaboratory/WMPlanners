from __future__ import annotations
import argparse
import copy
import json
import os
import subprocess
import sys
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from world_model.data import (
    HFWindowDataset, ParquetWindowDataset, CachedWindowDataset,
    open_window_dataset, collate_for_dataset,
)
from world_model.architecture import WorldModel
from world_model.engine.trainer import TrainConfig, rollout_loss
from world_model.utils import seed_everything


def prune_checkpoints(out_dir: str, keep_last: int = 3):
    """Keep only the most recent epoch checkpoints to limit disk usage."""
    if keep_last <= 0:
        return
    epoch_files = []
    for name in os.listdir(out_dir):
        if name.startswith("epoch_") and name.endswith(".pt"):
            try:
                epoch_num = int(name[len("epoch_"):-len(".pt")])
            except ValueError:
                continue
            epoch_files.append((epoch_num, os.path.join(out_dir, name)))
    epoch_files.sort(key=lambda x: x[0])
    to_delete = epoch_files[:-keep_last]
    for _, path in to_delete:
        try:
            os.remove(path)
        except OSError:
            pass


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

        if args.max_windows is not None and not args.precompute_windows:
            raise ValueError("--max_windows requires --precompute_windows")
        self.ds = open_window_dataset(
            dataset_type=args.dataset_type, path=args.data_path, split=args.split,
            image_size=args.img_size, window_size=args.num_hist + args.horizon,
            schema_config=args.schema_config,
        )
        if args.precompute_windows:
            if not isinstance(self.ds, ParquetWindowDataset):
                raise ValueError("--precompute_windows requires local Parquet tables")
            self.ds = CachedWindowDataset(self.ds, max_windows=args.max_windows)
        self.collate_fn = collate_for_dataset(self.ds)
        self.val_ds = self.ds
        self.has_validation_split = False
        if args.split == "train" and args.dataset_type == "hf":
            try:
                self.val_ds = open_window_dataset(
                    dataset_type="hf", path=args.data_path, split="val",
                    image_size=args.img_size, window_size=args.num_hist + args.horizon,
                    schema_config=args.schema_config,
                )
                if args.precompute_windows:
                    self.val_ds = CachedWindowDataset(self.val_ds, max_windows=args.max_windows)
                self.has_validation_split = True
            except (ValueError, RuntimeError) as exc:
                if not any(marker in str(exc) for marker in (
                    "has no windows in split", "no requested Silver split",
                    "Parquet dataset has no windows in split",
                )):
                    raise
        manifest = getattr(self.ds, "manifest", {})
        self.dataset_provenance = {
            key: manifest[key]
            for key in ("dataset_id", "dataset_hash", "release_id", "release_hash", "recipe_sha256")
            if manifest.get(key) is not None
        }

        sampler = None
        shuffle = True
        if args.ddp:
            sampler = DistributedSampler(self.ds, shuffle=True)
            shuffle = False

        train_drop_last = len(self.ds) > args.batch_size
        if self.main_process and not train_drop_last:
            print(
                f"[WARN] Dataset has {len(self.ds)} windows with batch_size={args.batch_size}; "
                "using drop_last=False to avoid skipping all training batches."
            )

        self.dl = DataLoader(
            self.ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=args.num_workers, collate_fn=self.collate_fn, drop_last=train_drop_last,
            pin_memory=True if "cuda" in str(args.device) else False,
        )

        odom_dim = self.ds.D_odom
        imu_dim = self.ds.D_imu
        act_dim = self.ds.D_act
        state_dim = args.state_dim if args.state_dim else min(6, odom_dim)

        self.model = WorldModel(
            odom_dim=odom_dim,
            imu_dim=imu_dim,
            act_dim=act_dim,
            state_dim=state_dim,
            backbone=args.backbone,
            backbone_kwargs={"model_name": args.backbone_name, "device": args.device},
            freeze_backbone=args.freeze_backbone,
            dynamics_type="mlp" if args.dynamics_type == "td-jepa" else args.dynamics_type,
            image_size=args.img_size,
        ).to(args.device)

        self.target_model = None
        if args.dynamics_type == "td-jepa" or args.wm_mode == "jepa-lite":
            self.target_model = copy.deepcopy(self.model)
            for p in self.target_model.parameters():
                p.requires_grad = False

        if args.ddp:
            find_unused = (args.dynamics_type == "td-jepa") or (args.wm_mode == "jepa-lite")
            self.model = DDP(self.model, device_ids=[self.local_rank],
                             find_unused_parameters=find_unused)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        lambda_state = 0.0 if args.wm_mode == "jepa-lite" else args.lambda_state
        lambda_risk = 0.0 if args.wm_mode == "jepa-lite" else args.lambda_risk
        lambda_var = args.lambda_var if args.wm_mode == "jepa-lite" else 0.0
        self.cfg = TrainConfig(
            device=args.device,
            lr=args.lr,
            horizon=args.horizon,
            num_hist=args.num_hist,
            lambda_z=args.lambda_z,
            lambda_state=lambda_state,
            lambda_risk=lambda_risk,
            objective=args.wm_mode,
            jepa_mask_ratio=args.jepa_mask_ratio,
            lambda_var=lambda_var,
        )

        self.best_validation_loss = float("inf")
        best_path = os.path.join(args.out_dir, "best.pt")
        if self.has_validation_split and os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            self.best_validation_loss = best_ckpt.get("validation_loss", float("inf"))
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

    @torch.no_grad()
    def validation_loss(self) -> float:
        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.eval()
        loader = DataLoader(
            self.val_ds, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.num_workers, collate_fn=self.collate_fn,
            drop_last=False,
        )
        total = 0.0
        count = 0
        for batch in loader:
            loss, _ = rollout_loss(model, batch, self.cfg, target_model=self.target_model)
            total += loss.item() * batch.rgb.shape[0]
            count += batch.rgb.shape[0]
        return total / count

    def run(self):
        args = self.args
        if self.main_process:
            print(f"[INFO] Starting training: epochs={args.epochs}, batch_size={args.batch_size}, "
                  f"backbone={args.backbone}, dynamics={args.dynamics_type}, wm_mode={args.wm_mode}, ddp={args.ddp}")

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
                          f"risk={logs['loss_risk']:.4f} var={logs.get('loss_var', 0.0):.4f}")
                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss": logs['loss'],
                            "train/loss_z": logs['loss_z'],
                            "train/loss_state": logs['loss_state'],
                            "train/loss_risk": logs['loss_risk'],
                            "train/loss_var": logs.get('loss_var', 0.0),
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
                        'state_dim': args.state_dim if args.state_dim else min(6, self.ds.D_odom),
                    },
                    'dataset_provenance': self.dataset_provenance,
                }
                torch.save(ckpt, os.path.join(args.out_dir, 'latest.pt'))
                if args.save_every > 0 and ((epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs):
                    if self.has_validation_split:
                        val_loss = self.validation_loss()
                        ckpt["validation_loss"] = val_loss
                        print(f"[Epoch {epoch:03d}] validation loss={val_loss:.4f}")
                        if val_loss < self.best_validation_loss:
                            self.best_validation_loss = val_loss
                            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
                    torch.save(ckpt, os.path.join(args.out_dir, f'epoch_{epoch:03d}.pt'))
                    prune_checkpoints(args.out_dir, keep_last=args.keep_last_checkpoints)

            if args.ddp:
                dist.barrier()

        train_time = time.time() - t0

        if self.main_process:
            print(f"Training complete. Total time: {train_time:.1f}s")
            print("\n" + "=" * 60)
            print("[Evaluation] Computing pose MSE and accuracy ...")
            print("=" * 60)

            eval_dl = DataLoader(
                self.val_ds, batch_size=args.batch_size, shuffle=False,
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
            latest_ckpt = os.path.join(args.out_dir, 'latest.pt')
            print(f"\nLatest checkpoint: {latest_ckpt}")

            if args.run_mpc_after_train:
                self._run_mpc_eval(latest_ckpt)

        if args.ddp:
            cleanup_ddp()

    def _run_mpc_eval(self, checkpoint_path: str):
        args = self.args
        base_cmd = [sys.executable]

        if args.mpc_eval_type == "toy":
            cmd = base_cmd + [
                "scripts/world_model/eval_mpc.py",
                "--ckpt", checkpoint_path,
                "--episodes", str(args.mpc_num_episodes),
                "--horizon", str(args.mpc_horizon),
                "--device", str(args.device),
            ]
        else:
            if not isinstance(self.ds, HFWindowDataset):
                print("[WARN] Skipping replay MPC eval (supports HFWindowDataset rows only).")
                return

            cmd = base_cmd + [
                "scripts/world_model/eval_mpc_replay.py",
                "--checkpoint", checkpoint_path,
                "--dataset", args.data_path,
                "--horizon", str(args.mpc_horizon),
                "--num_episodes", str(args.mpc_num_episodes),
                "--cem_population", str(args.mpc_cem_population),
                "--cem_iters", str(args.mpc_cem_iters),
                "--goal_success_threshold", str(args.mpc_goal_success_threshold),
                "--device", str(args.device),
            ]

        print("\n" + "=" * 60)
        print("[Post-Train MPC] Running:")
        print("  " + " ".join(cmd))
        print("=" * 60)

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] Post-train MPC evaluation failed with exit code {exc.returncode}.")


def build_parser(parents=None):
    ap = argparse.ArgumentParser(
        description="Train a standard WorldModel",
        parents=parents or [],
        add_help=parents is None,
    )
    ap.add_argument('--data_path', '--dataset', '--data_dir', type=str, required=True,
                    dest='data_path', help='HuggingFace dataset ID or local path')
    ap.add_argument('--dataset_type', type=str, default='hf', choices=['hf', 'npz'])
    ap.add_argument('--schema_config', type=str, default=None,
                    help='YAML layout and sensor settings for local frames/windows Parquet tables')
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
    ap.add_argument('--wm_mode', type=str, default='standard', choices=['standard', 'jepa-lite'],
                    help='standard: latent+state+risk losses; jepa-lite: latent JEPA-style objective only')
    ap.add_argument('--ema_tau', type=float, default=0.996)
    ap.add_argument('--jepa_mask_ratio', type=float, default=0.6,
                    help='Fraction of rollout steps to supervise in jepa-lite objective')
    ap.add_argument('--lambda_z', type=float, default=1.0)
    ap.add_argument('--lambda_state', type=float, default=1.0)
    ap.add_argument('--lambda_risk', type=float, default=0.2)
    ap.add_argument('--lambda_var', type=float, default=1.0,
                    help='jepa-lite only: weight for the variance-preservation anti-collapse term')
    ap.add_argument('--experiment_name', type=str, default=None)
    ap.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--pose_threshold', type=float, default=0.1)
    ap.add_argument('--state_dim', type=int, default=None, help='Number of state dimensions to predict (default: min(6, odom_dim))')
    ap.add_argument('--save_every', type=int, default=10, help='Save numbered epoch checkpoint every N epochs (latest.pt is always updated)')
    ap.add_argument('--keep_last_checkpoints', type=int, default=3, help='Keep only last K numbered epoch checkpoints')
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--precompute_windows', action='store_true', default=False,
                    help='Cache local Parquet windows in RAM before training')
    ap.add_argument('--max_windows', type=int, default=None,
                    help='Cap cached Parquet windows per split; requires --precompute_windows')
    ap.add_argument('--ddp', action='store_true', default=False)
    ap.add_argument('--run_mpc_after_train', action='store_true', default=False,
                    help='Run MPC evaluation automatically after training finishes (main process only)')
    ap.add_argument('--mpc_eval_type', type=str, default='replay', choices=['replay', 'toy'],
                    help='Post-train MPC evaluation type: replay (dataset) or toy (Toy2DNavEnv)')
    ap.add_argument('--mpc_num_episodes', type=int, default=20,
                    help='Number of episodes for post-train MPC evaluation')
    ap.add_argument('--mpc_horizon', type=int, default=16,
                    help='Planning horizon for post-train MPC evaluation')
    ap.add_argument('--mpc_cem_population', type=int, default=128,
                    help='CEM population for replay MPC evaluation')
    ap.add_argument('--mpc_cem_iters', type=int, default=5,
                    help='CEM iterations for replay MPC evaluation')
    ap.add_argument('--mpc_goal_success_threshold', type=float, default=0.1,
                    help='Goal success threshold in meters for replay MPC evaluation')
    return ap


def main():
    args = build_parser().parse_args()
    Trainer(args).run()


if __name__ == '__main__':
    main()
