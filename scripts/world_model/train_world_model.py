"""Unified world-model trainer.

Modes
-----
  --mode wm      Original WorldModel with rollout_loss (default).
  --mode dinowm  DINO-WM modular trainer, Hydra-free.

Usage examples
--------------
  # Original WM
  python train_world_model.py --mode wm --dataset lquan9/collect1 --backbone dinov2

  # DINO-WM
  python train_world_model.py --mode dinowm --dataset lquan9/collect1 \
      --backbone_name dinov2_vits14 --num_hist 3 --num_pred 1 --frameskip 5 \
      --encoder_lr 1e-6 --predictor_lr 5e-4 --action_encoder_lr 5e-4
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import utils as vutils
from tqdm import tqdm

# Shared data / training utilities
from world_model.data import NPZEpisodeDataset, collate_npz
from world_model.data.hf_datasets import (
    HFWindowDataset,
    EpisodeDataset,
    collate_hf,
    collate_dinowm,
)
from world_model.utils import seed_everything, slice_trajdict_with_t, seed, sample_tensors
from world_model.metrics import eval_images

# WM mode imports
from world_model.architecture import WorldModel
from world_model.engine.trainer import TrainConfig, rollout_loss

# DINO-WM mode imports
from world_model.backbones.wm import WM
from world_model.backbones.dino import build_dinov2
from world_model.backbones.proprio import ProprioceptiveEmbedding
from world_model.backbones.latentdynamics import ViTPredictor
from world_model.backbones.base import ModelBundle

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Shared DDP helpers
# ──────────────────────────────────────────────────────────────────────────────

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


def barrier(args) -> None:
    if args.ddp:
        dist.barrier()


def gather_scalar(tensor: torch.Tensor, args) -> torch.Tensor:
    """All-reduce a scalar tensor across DDP ranks, returning the per-rank mean."""
    if not args.ddp:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / dist.get_world_size()


def gather_dict(d: dict, args) -> dict:
    """All-reduce every tensor value in a dict across DDP ranks."""
    if not args.ddp:
        return d
    return {k: gather_scalar(v.clone(), args) for k, v in d.items()}


# ──────────────────────────────────────────────────────────────────────────────
# WM mode: post-training pose evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_pose(model, dataloader, cfg: TrainConfig, threshold: float = 0.1):
    """Evaluate pose prediction quality over the full dataset.

    Returns:
      pose_mse      – mean squared error of state_head predictions vs GT odom.
      pose_accuracy – fraction of predictions within ``threshold`` (L2 norm).
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
            pred_state = out.state                        # (B, state_dim)
            gt_state = odom[:, t, :pred_state.shape[-1]]  # (B, state_dim)
            mse = F.mse_loss(pred_state, gt_state, reduction="sum").item()
            l2 = torch.norm(pred_state - gt_state, dim=-1)
            total_mse += mse
            total_within += (l2 < threshold).sum().item()
            total_count += B

    if total_count == 0:
        return {"pose_mse": 0.0, "pose_accuracy": 0.0}
    return {
        "pose_mse": total_mse / total_count,
        "pose_accuracy": total_within / total_count,
    }


# ──────────────────────────────────────────────────────────────────────────────
# DINO-WM config dataclass (notebook-friendly alternative to argparse.Namespace)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DinoWMConfig:
    """Drop-in replacement for the argparse Namespace used by DinoWMTrainer.

    All fields match the argparse defaults so that ``DinoWMTrainer(DinoWMConfig(...))``
    is equivalent to ``DinoWMTrainer(args)`` from the CLI.  Only ``dataset`` is
    mandatory; everything else has a sensible default.

    Notebook usage::

        from train_world_model import DinoWMTrainer, DinoWMConfig
        cfg = DinoWMConfig(dataset="lquan9/collect1", epochs=5, batch_size=4)
        trainer = DinoWMTrainer(cfg)
        trainer.run()
    """

    # ── Required ─────────────────────────────────────────────────────────────
    dataset: str = ""

    # ── Shared ────────────────────────────────────────────────────────────────
    epochs: int = 20
    batch_size: int = 64
    img_size: int = 224
    seed: int = 0
    out_dir: str = "runs"
    num_workers: int = 4
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    experiment_name: Optional[str] = None
    wandb: bool = False
    ddp: bool = False
    local_rank: int = 0

    # ── Encoder ───────────────────────────────────────────────────────────────
    backbone_name: str = "dinov2_vits14"
    train_encoder: bool = False
    encoder_lr: float = 1e-6

    # ── Predictor ─────────────────────────────────────────────────────────────
    num_hist: int = 3
    num_pred: int = 1
    frameskip: int = 5
    concat_dim: int = 1
    has_predictor: bool = True
    train_predictor: bool = True
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_mlp_dim: int = 2048
    predictor_dropout: float = 0.1
    predictor_lr: float = 5e-4

    # ── Action / proprio encoders ─────────────────────────────────────────────
    action_emb_dim: int = 10
    proprio_emb_dim: int = 10
    num_action_repeat: int = 1
    num_proprio_repeat: int = 1
    action_encoder_lr: float = 5e-4

    # ── Decoder (optional) ────────────────────────────────────────────────────
    has_decoder: bool = False
    train_decoder: bool = False
    decoder_path: Optional[str] = None
    decoder_n_embed: int = 2048
    decoder_n_res_block: int = 4
    decoder_n_res_channel: int = 128
    decoder_lr: float = 3e-4

    # ── Checkpointing / logging ───────────────────────────────────────────────
    save_every_x_epoch: int = 1
    num_reconstruct_samples: int = 6
    debug: bool = False

    # ── Data normalisation stats (inline JSON strings) ────────────────────────
    odom_stats_json: Optional[str] = None
    imu_stats_json: Optional[str] = None
    action_stats_json: Optional[str] = None

    # ── Planning eval integration (optional) ──────────────────────────────────
    plan_cfg_path: Optional[str] = None
    plan_planner: List[str] = field(default_factory=lambda: ["gd", "cem"])
    plan_goal_source: List[str] = field(default_factory=lambda: ["dset", "random_state"])
    plan_goal_H: List[int] = field(default_factory=lambda: [5])
    plan_alpha: List[float] = field(default_factory=lambda: [0.1, 1.0])

    # ── Training improvements ─────────────────────────────────────────────────
    use_amp: bool = False
    accum_steps: int = 1
    clip_grad_norm: float = 1.0
    ema_tau: float = 0.0
    lr_schedule: str = "none"
    lr_warmup_epochs: int = 0
    lambda_z: float = 1.0
    lambda_action_decode: float = 1.0

    # ── Swappable predictor ───────────────────────────────────────────────────
    predictor_type: str = "vit"     # "vit" | "mlp" | "rssm"

    # ── State / risk auxiliary supervision ───────────────────────────────────
    state_dim:    int   = 0         # 0 = disabled; e.g. 6 for 6-DOF odom
    lambda_state: float = 1.0
    lambda_risk:  float = 0.2


# ──────────────────────────────────────────────────────────────────────────────
# DINO-WM Trainer (Hydra-free)
# ──────────────────────────────────────────────────────────────────────────────

class DinoWMTrainer:
    """DINO-WM trainer driven by argparse instead of Hydra.

    Preserves all original per-component learning rates, modular model
    construction, per-epoch validation, image-quality metrics, open-loop
    rollout evaluation, and optional planning-job integration.
    Replaces Accelerator with manual DDP to match the rest of the codebase.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.main_process = is_main_process(args)

        # Convenience aliases matching attribute names used throughout the class
        self.num_hist = args.num_hist
        self.num_pred = args.num_pred
        self.frameskip = args.frameskip
        self.num_reconstruct_samples = args.num_reconstruct_samples
        self.total_epochs = args.epochs
        self.epoch = 0

        os.makedirs(args.out_dir, exist_ok=True)

        # Experiment name / model identifier
        ds_short = args.dataset.replace("/", "_")
        self.model_name = (
            args.experiment_name
            if args.experiment_name
            else f"{ds_short}_h{args.num_hist}_p{args.num_pred}"
        )

        # ── WandB (main process only) ─────────────────────────────────────────
        self.use_wandb = args.wandb and self.main_process
        self.wandb_run = None
        if self.use_wandb:
            import wandb as _wandb
            run_id_file = os.path.join(args.out_dir, "wandb_run_id.txt")
            wandb_run_id = None
            if os.path.exists(run_id_file):
                with open(run_id_file) as fh:
                    wandb_run_id = fh.read().strip() or None
                log.info(f"Resuming WandB run {wandb_run_id}")
            project = "dino_wm_debug" if args.debug else "dino_wm"
            self.wandb_run = _wandb.init(
                project=project,
                name=self.model_name,
                config=vars(args),
                id=wandb_run_id,
                resume="allow",
            )
            with open(run_id_file, "w") as fh:
                fh.write(self.wandb_run.id)
        elif args.wandb and self.main_process:
            log.warning("--wandb requested but WandB may not be configured.")

        # ── Dataset ───────────────────────────────────────────────────────────
        odom_stats = self._parse_stats(args.odom_stats_json)
        imu_stats = self._parse_stats(args.imu_stats_json)
        action_stats = self._parse_stats(args.action_stats_json)
        trajectory_length = args.num_hist + args.num_pred

        log.info(f"Loading dataset from {args.dataset} ...")
        _window_ds = HFWindowDataset(
            args.dataset,
            img_size=args.img_size,
            num_hist=args.num_hist,
            num_pred=args.num_pred,
            frameskip=args.frameskip,
            odom_stats=odom_stats,
            imu_stats=imu_stats,
            collision_stats=None,
            action_stats=action_stats,
            sample_mode=True,
            sample_k=trajectory_length,
        )
        self.datasets = {"train": _window_ds, "valid": _window_ds}
        # Odom slice width — used to extract odom from the state tensor in train/val
        self.D_odom = _window_ds.D_odom

        # Episode-level dataset for open-loop rollout (full trajectories)
        _traj_ds = HFWindowDataset(
            args.dataset,
            img_size=args.img_size,
            odom_stats=odom_stats,
            imu_stats=imu_stats,
            collision_stats=None,
            action_stats=action_stats,
            sample_mode=False,
        )
        self.train_traj_dset = EpisodeDataset(_traj_ds)
        self.val_traj_dset = EpisodeDataset(_traj_ds)

        # Per-process batch size
        world_size = dist.get_world_size() if args.ddp else 1
        assert args.batch_size % world_size == 0, (
            f"batch_size ({args.batch_size}) must be divisible by "
            f"world_size ({world_size})"
        )
        gpu_batch_size = args.batch_size // world_size

        self._sampler_train = (
            DistributedSampler(_window_ds, shuffle=False) if args.ddp else None
        )
        self._sampler_valid = (
            DistributedSampler(_window_ds, shuffle=False) if args.ddp else None
        )
        self.dataloaders = {
            "train": DataLoader(
                _window_ds,
                batch_size=gpu_batch_size,
                shuffle=(self._sampler_train is None),
                sampler=self._sampler_train,
                num_workers=args.num_workers,
                collate_fn=collate_dinowm,
                drop_last=True,
                pin_memory="cuda" in args.device,
            ),
            "valid": DataLoader(
                _window_ds,
                batch_size=gpu_batch_size,
                shuffle=False,
                sampler=self._sampler_valid,
                num_workers=args.num_workers,
                collate_fn=collate_dinowm,
                drop_last=False,
                pin_memory="cuda" in args.device,
            ),
        }
        log.info(f"Dataloader per-GPU batch size: {gpu_batch_size}")

        # ── Per-component train flags ─────────────────────────────────────────
        self.train_encoder = args.train_encoder
        self.train_predictor = args.train_predictor
        self.train_decoder = args.train_decoder

        # Keys saved in each checkpoint
        self._keys_to_save = ["epoch"]
        if self.train_encoder:
            self._keys_to_save += ["encoder", "encoder_optimizer"]
        if self.train_predictor and args.has_predictor:
            self._keys_to_save += ["predictor", "predictor_optimizer"]
        if self.train_decoder and args.has_decoder:
            self._keys_to_save += ["decoder", "decoder_optimizer"]
        self._keys_to_save += ["action_encoder", "proprio_encoder"]
        if getattr(args, "state_dim", 0) > 0:
            self._keys_to_save += ["state_head", "risk_head"]

        # Model attributes populated by init_models()
        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.model = None
        self.target_encoder = None  # EMA encoder for I-JEPA (None when ema_tau=0)

        self.init_models()
        self.init_optimizers()
        # AMP scaler (no-op when use_amp=False)
        self._scaler = torch.cuda.amp.GradScaler(enabled=getattr(args, 'use_amp', False))
        self._best_val_loss = float("inf")
        self.epoch_log = OrderedDict()

    # ── Dataset stat helpers ─────────────────────────────────────────────────

    @staticmethod
    def _parse_stats(json_str):
        """Parse a JSON string '{"mean":[...], "std":[...]}' or return None."""
        if not json_str:
            return None
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid stats JSON: {json_str!r}") from exc

    # ── Model / optimizer initialisation ─────────────────────────────────────

    def init_models(self):
        args = self.args
        # Auto-resume from latest checkpoint if present
        ckpt_path = Path(args.out_dir) / "checkpoints" / "model_latest.pth"
        if ckpt_path.exists():
            self.load_ckpt(ckpt_path)
            log.info(f"Resuming from epoch {self.epoch}: {ckpt_path}")

        # ── Encoder ──────────────────────────────────────────────────────────
        if self.encoder is None:
            enc = build_dinov2(model_name=args.backbone_name, device=args.device)
            if isinstance(enc, ModelBundle):
                enc = enc.model
            self.encoder = enc
        if not self.train_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.encoder = self.encoder.to(self.device)

        # EMA target encoder (I-JEPA style) — created from raw encoder before any DDP wrap
        ema_tau = getattr(args, 'ema_tau', 0.0)
        if ema_tau > 0:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
            self.target_encoder = self.target_encoder.to(self.device)
            log.info(f"Initialized EMA target encoder with tau={ema_tau}")

        # ── Proprio encoder ───────────────────────────────────────────────────
        self.proprio_encoder = ProprioceptiveEmbedding(
            in_chans=self.datasets["train"].proprio_dim,
            emb_dim=args.proprio_emb_dim,
        )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        self.proprio_encoder = self.proprio_encoder.to(self.device)

        # ── Action encoder ────────────────────────────────────────────────────
        self.action_encoder = ProprioceptiveEmbedding(
            in_chans=self.datasets["train"].action_dim,
            emb_dim=args.action_emb_dim,
        )
        action_emb_dim = self.action_encoder.emb_dim
        self.action_encoder = self.action_encoder.to(self.device)

        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        # ── Predictor ─────────────────────────────────────────────────────────
        predictor_type = getattr(args, "predictor_type", "vit")

        if self.encoder.latent_ndim == 1:
            num_patches = 1
        elif hasattr(self.encoder, "num_patches"):
            num_patches = self.encoder.num_patches
        else:
            num_side_patches = args.img_size // 16
            num_patches = num_side_patches ** 2

        if args.concat_dim == 0:
            num_patches += 2

        predictor_dim = self.encoder.emb_dim + (
            proprio_emb_dim * args.num_proprio_repeat
            + action_emb_dim * args.num_action_repeat
        ) * args.concat_dim

        if args.has_predictor and self.predictor is None:
            if predictor_type == "vit":
                self.predictor = ViTPredictor(
                    num_patches=num_patches,
                    num_frames=args.num_hist,
                    dim=predictor_dim,
                    depth=args.predictor_depth,
                    heads=args.predictor_heads,
                    mlp_dim=args.predictor_mlp_dim,
                    dropout=args.predictor_dropout,
                    emb_dropout=0.0,
                    pool="mean",
                )
            else:
                # Frame-level dynamics: MLP or RSSM
                # z_dim = visual encoder output dim, act_dim = raw action embedding dim
                from world_model.components.dynamics import LatentDynamics, RSSMDynamics
                z_dim = self.encoder.emb_dim
                if predictor_type == "rssm":
                    self.predictor = RSSMDynamics(z_dim=z_dim, act_dim=action_emb_dim)
                else:  # "mlp"
                    self.predictor = LatentDynamics(z_dim=z_dim, act_dim=action_emb_dim)
                log.info(
                    f"Using frame-level {predictor_type.upper()} predictor: "
                    f"z_dim={z_dim}, act_dim={action_emb_dim}"
                )

        if self.predictor is not None:
            if not self.train_predictor:
                for p in self.predictor.parameters():
                    p.requires_grad = False
            self.predictor = self.predictor.to(self.device)

        # ── Decoder ───────────────────────────────────────────────────────────
        if args.has_decoder and self.decoder is None:
            if args.decoder_path:
                ckpt = torch.load(args.decoder_path, weights_only=False)
                self.decoder = ckpt["decoder"] if isinstance(ckpt, dict) else ckpt
                log.info(f"Loaded decoder from {args.decoder_path}")
            else:
                from world_model.backbones.decoder import VQVAE
                self.decoder = VQVAE(
                    channel=self.encoder.emb_dim,
                    n_embed=args.decoder_n_embed,
                    n_res_block=args.decoder_n_res_block,
                    n_res_channel=args.decoder_n_res_channel,
                    quantize=False,
                )
        if self.decoder is not None:
            if not self.train_decoder:
                for p in self.decoder.parameters():
                    p.requires_grad = False
            self.decoder = self.decoder.to(self.device)

        # ── Composite WM model ────────────────────────────────────────────────
        predictor_mode = "frame" if predictor_type in ("mlp", "rssm") else "patch"
        self.model = WM(
            image_size=args.img_size,
            num_hist=args.num_hist,
            num_pred=args.num_pred,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=args.concat_dim,
            num_action_repeat=args.num_action_repeat,
            num_proprio_repeat=args.num_proprio_repeat,
            raw_action_dim=self.datasets["train"].action_dim,
            train_encoder=args.train_encoder,
            train_predictor=args.train_predictor,
            train_decoder=args.has_decoder and args.train_decoder,
            predictor_mode=predictor_mode,
            state_dim=getattr(args, "state_dim", 0),
            lambda_state=getattr(args, "lambda_state", 1.0),
            lambda_risk=getattr(args, "lambda_risk", 0.2),
        ).to(self.device)

        # Expose state/risk heads as trainer attributes so save_ckpt() can persist them
        self.state_head = self.model.state_head
        self.risk_head  = self.model.risk_head

        # DDP-wrap per-component (mirrors Accelerator.prepare behaviour)
        if args.ddp:
            local_rank = int(os.environ["LOCAL_RANK"])
            self.encoder = DDP(self.encoder, device_ids=[local_rank])
            self.action_encoder = DDP(self.action_encoder, device_ids=[local_rank])
            self.proprio_encoder = DDP(self.proprio_encoder, device_ids=[local_rank])
            if self.predictor is not None:
                self.predictor = DDP(self.predictor, device_ids=[local_rank])
            if self.decoder is not None:
                self.decoder = DDP(self.decoder, device_ids=[local_rank])

    def init_optimizers(self):
        """Create per-component optimizers with independent learning rates and optional LR schedulers."""
        args = self.args

        # Adam for the (usually frozen) vision encoder
        raw_enc = self.encoder.module if isinstance(self.encoder, DDP) else self.encoder
        self.encoder_optimizer = torch.optim.Adam(
            raw_enc.parameters(), lr=args.encoder_lr
        )

        if args.has_predictor:
            # AdamW for the ViT predictor
            raw_pred = (
                self.predictor.module if isinstance(self.predictor, DDP) else self.predictor
            )
            self.predictor_optimizer = torch.optim.AdamW(
                raw_pred.parameters(), lr=args.predictor_lr
            )
            # AdamW for action encoder, proprio encoder, and the WM action decoder head
            raw_act = (
                self.action_encoder.module
                if isinstance(self.action_encoder, DDP)
                else self.action_encoder
            )
            raw_prop = (
                self.proprio_encoder.module
                if isinstance(self.proprio_encoder, DDP)
                else self.proprio_encoder
            )
            aux_params = itertools.chain(
                raw_act.parameters(),
                raw_prop.parameters(),
                self.model.action_decoder.parameters(),
            )
            if self.model.state_head is not None:
                aux_params = itertools.chain(
                    aux_params,
                    self.model.state_head.parameters(),
                    self.model.risk_head.parameters(),
                )
            self.action_encoder_optimizer = torch.optim.AdamW(
                aux_params,
                lr=args.action_encoder_lr,
            )

        if args.has_decoder:
            # Adam for the image decoder
            raw_dec = (
                self.decoder.module if isinstance(self.decoder, DDP) else self.decoder
            )
            self.decoder_optimizer = torch.optim.Adam(
                raw_dec.parameters(), lr=args.decoder_lr
            )

        # Per-component cosine LR schedulers with linear warmup
        self._schedulers = {}
        lr_schedule = getattr(args, 'lr_schedule', 'none')
        if lr_schedule != 'none':
            self._schedulers['encoder'] = self._make_scheduler(self.encoder_optimizer)
            if args.has_predictor:
                self._schedulers['predictor'] = self._make_scheduler(self.predictor_optimizer)
                self._schedulers['action_encoder'] = self._make_scheduler(self.action_encoder_optimizer)
            if args.has_decoder:
                self._schedulers['decoder'] = self._make_scheduler(self.decoder_optimizer)

        # Restore scheduler states saved from a previous checkpoint
        for k, state in getattr(self, '_pending_sched_states', {}).items():
            if k in self._schedulers and self._schedulers[k] is not None:
                self._schedulers[k].load_state_dict(state)
                log.info(f"Restored LR scheduler state for '{k}'")
        self._pending_sched_states = {}

    # ── LR / optimizer helpers ───────────────────────────────────────────────

    def _make_scheduler(self, optimizer):
        """Linear warmup followed by cosine annealing, keyed to total training epochs."""
        args = self.args
        warmup = getattr(args, 'lr_warmup_epochs', 0)
        if warmup <= 0:
            return None
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - warmup)
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup]
        )

    def _optimizer_map(self):
        """Map component name → optimizer (for LR logging)."""
        m = {"encoder": self.encoder_optimizer}
        if self.args.has_predictor:
            m["predictor"] = self.predictor_optimizer
            m["action_encoder"] = self.action_encoder_optimizer
        if self.args.has_decoder:
            m["decoder"] = self.decoder_optimizer
        return m

    def _active_optimizers(self):
        """Return optimizers for components that are currently being trained."""
        opts = []
        if self.model.train_encoder:
            opts.append(self.encoder_optimizer)
        if self.args.has_predictor and self.model.train_predictor:
            opts.append(self.predictor_optimizer)
            opts.append(self.action_encoder_optimizer)
        if self.args.has_decoder and self.model.train_decoder:
            opts.append(self.decoder_optimizer)
        return opts

    def _zero_grads(self):
        self.encoder_optimizer.zero_grad(set_to_none=True)
        if self.args.has_predictor:
            self.predictor_optimizer.zero_grad(set_to_none=True)
            self.action_encoder_optimizer.zero_grad(set_to_none=True)
        if self.args.has_decoder:
            self.decoder_optimizer.zero_grad(set_to_none=True)

    def _clip_grads(self, max_norm: float):
        """Clip gradients for all trainable components."""
        params = []
        if self.model.train_encoder:
            raw = self.encoder.module if isinstance(self.encoder, DDP) else self.encoder
            params.extend(p for p in raw.parameters() if p.grad is not None)
        if self.args.has_predictor and self.model.train_predictor:
            for mod in (self.predictor, self.action_encoder, self.proprio_encoder):
                raw = mod.module if isinstance(mod, DDP) else mod
                params.extend(p for p in raw.parameters() if p.grad is not None)
        if self.args.has_decoder and self.model.train_decoder:
            raw = self.decoder.module if isinstance(self.decoder, DDP) else self.decoder
            params.extend(p for p in raw.parameters() if p.grad is not None)
        if params:
            torch.nn.utils.clip_grad_norm_(params, max_norm)

    def _ema_update(self):
        """Exponential moving average update of the target encoder (I-JEPA style)."""
        tau = getattr(self.args, 'ema_tau', 0.0)
        if tau <= 0 or self.target_encoder is None:
            return
        raw_enc = self.encoder.module if isinstance(self.encoder, DDP) else self.encoder
        with torch.no_grad():
            for online_p, target_p in zip(raw_enc.parameters(), self.target_encoder.parameters()):
                target_p.data.mul_(tau).add_(online_p.data, alpha=1.0 - tau)

    # ── Checkpoint helpers ───────────────────────────────────────────────────

    def save_ckpt(self, tag=None):
        barrier(self.args)
        if not self.main_process:
            return None, self.model_name, self.epoch

        ckpt_dir = os.path.join(self.args.out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt = {}
        for k in self._keys_to_save:
            obj = self.__dict__.get(k)
            if obj is None:
                continue
            ckpt[k] = obj.module if isinstance(obj, DDP) else obj

        # Persist LR scheduler states so training resumes at the correct LR
        for k, sched in getattr(self, '_schedulers', {}).items():
            if sched is not None:
                ckpt[f"sched_{k}"] = sched.state_dict()

        torch.save(ckpt, os.path.join(ckpt_dir, "model_latest.pth"))
        torch.save(ckpt, os.path.join(ckpt_dir, f"model_{self.epoch}.pth"))
        if tag is not None:
            torch.save(ckpt, os.path.join(ckpt_dir, f"model_{tag}.pth"))

        label = f" (tag={tag})" if tag else ""
        log.info(f"Saved checkpoint epoch {self.epoch} → {ckpt_dir}{label}")
        return os.path.join(ckpt_dir, f"model_{self.epoch}.pth"), self.model_name, self.epoch

    def load_ckpt(self, filename="model_latest.pth"):
        ckpt = torch.load(str(filename), weights_only=False)
        sched_states = {}
        for k, v in ckpt.items():
            if k.startswith("sched_"):
                sched_states[k[len("sched_"):]] = v
            else:
                self.__dict__[k] = v
        # Scheduler states are applied after init_optimizers() creates them;
        # stash them here so init_optimizers() can pick them up.
        self._pending_sched_states = sched_states
        missing = set(self._keys_to_save) - {k for k in ckpt if not k.startswith("sched_")}
        if missing:
            log.warning("Keys not found in checkpoint: %s", missing)

    # ── Main run loop ────────────────────────────────────────────────────────

    def run(self):
        lock = None
        if self.main_process:
            self.job_set = set()
            lock = threading.Lock()
            monitor_thread = threading.Thread(
                target=self._monitor_jobs, args=(lock,), daemon=True
            )
            monitor_thread.start()

        init_epoch = self.epoch + 1
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            if self._sampler_train is not None:
                self._sampler_train.set_epoch(epoch)

            barrier(self.args)
            self.train()
            barrier(self.args)
            self.val()

            # Step per-component LR schedulers once per epoch
            for sched in self._schedulers.values():
                if sched is not None:
                    sched.step()

            # Log current learning rates alongside loss metrics
            lr_logs = {f"lr_{k}": [opt.param_groups[0]["lr"]] for k, opt in self._optimizer_map().items()}
            self.logs_update(lr_logs)

            self.logs_flash(step=self.epoch)

            if epoch % self.args.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt()
                if (
                    self.args.plan_cfg_path is not None
                    and ckpt_path is not None
                    and self.main_process
                ):
                    from plan import build_plan_cfg_dicts, launch_plan_jobs  # noqa: PLC0415

                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=self.args.plan_cfg_path,
                        ckpt_base_path=self.args.out_dir,
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=self.args.plan_planner,
                        goal_source=self.args.plan_goal_source,
                        goal_H=self.args.plan_goal_H,
                        alpha=self.args.plan_alpha,
                    )
                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            self.args.out_dir, "submitit-evals", f"epoch_{self.epoch}"
                        ),
                    )
                    with lock:
                        self.job_set.update(jobs)

    def _monitor_jobs(self, lock):
        """Poll async planning-eval futures and log results to WandB."""
        while True:
            with lock:
                finished = [j for j in self.job_set if j[2].done()]
                for epoch, job_name, job in finished:
                    result = job.result()
                    log.info(f"Planning result {job_name} @ epoch {epoch}: {result}")
                    if self.wandb_run is not None:
                        log_data = {f"{job_name}/{k}": v for k, v in result.items()}
                        log_data["epoch"] = epoch
                        self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    # ── Per-epoch train / val ────────────────────────────────────────────────

    def train(self):
        args = self.args
        accum_steps = getattr(args, 'accum_steps', 1)
        use_amp = getattr(args, 'use_amp', False)
        clip_grad = getattr(args, 'clip_grad_norm', 0.0)
        lambda_z = getattr(args, 'lambda_z', 1.0)
        lambda_act = getattr(args, 'lambda_action_decode', 1.0)

        loader = self.dataloaders["train"]
        self._zero_grads()

        for i, data in enumerate(
            tqdm(loader, desc=f"Epoch {self.epoch} Train", disable=not self.main_process)
        ):
            obs, act, state = data
            is_last_batch = (i + 1 == len(loader))
            should_update = ((i + 1) % accum_steps == 0) or is_last_batch
            plot = (i == 0)

            self.model.train()

            state_dim = getattr(args, "state_dim", 0)
            if state_dim > 0:
                odom      = state[:, :, :self.D_odom].to(self.device)
                collision = state[:, :, -1:].to(self.device)
            else:
                odom      = None
                collision = None

            with torch.cuda.amp.autocast(enabled=use_amp):
                z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                    obs, act,
                    target_encoder=self.target_encoder,
                    odom=odom,
                    collision=collision,
                )

                if z_out is not None:
                    act_tgt = act[:, self.model.num_pred:].detach()
                    act_pred = self.model.decode_act(z_out)
                    action_decode_loss = F.mse_loss(act_pred, act_tgt)
                    loss_components["action_decode_loss"] = action_decode_loss
                    # Weighted JEPA + action-decode loss
                    loss = (
                        lambda_z * loss_components["z_loss"]
                        + lambda_act * action_decode_loss
                    )

                scaled_loss = loss / accum_steps  # normalize for gradient accumulation

            if use_amp:
                self._scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if should_update:
                active_opts = self._active_optimizers()
                if use_amp:
                    if clip_grad > 0:
                        for opt in active_opts:
                            self._scaler.unscale_(opt)
                        self._clip_grads(clip_grad)
                    for opt in active_opts:
                        self._scaler.step(opt)
                    self._scaler.update()
                else:
                    if clip_grad > 0:
                        self._clip_grads(clip_grad)
                    for opt in active_opts:
                        opt.step()

                self._zero_grads()
                self._ema_update()

            # Gather unscaled loss components from all ranks for logging
            loss_components_gathered = gather_dict(
                {
                    k: (v.detach() if isinstance(v, torch.Tensor) else torch.tensor(float(v)))
                    for k, v in loss_components.items()
                },
                self.args,
            )
            loss_components_logged = {k: v.mean().item() for k, v in loss_components_gathered.items()}

            if args.has_decoder and plot:
                self._log_image_metrics(obs, z_out, visual_out, visual_reconstructed, phase="train")

            self.logs_update({f"train_{k}": [v] for k, v in loss_components_logged.items()})

    def val(self):
        self.model.eval()
        use_amp = getattr(self.args, 'use_amp', False)

        # Open-loop rollout metrics
        if len(self.train_traj_dset) > 0 and self.args.has_predictor:
            with torch.no_grad():
                train_rollout = self.openloop_rollout(self.train_traj_dset, mode="train")
                self.logs_update({f"train_{k}": [v] for k, v in train_rollout.items()})
                val_rollout = self.openloop_rollout(self.val_traj_dset, mode="val")
                self.logs_update({f"val_{k}": [v] for k, v in val_rollout.items()})

        barrier(self.args)

        # Per-batch validation loss and image metrics
        val_losses = []
        for i, data in enumerate(
            tqdm(
                self.dataloaders["valid"],
                desc=f"Epoch {self.epoch} Valid",
                disable=not self.main_process,
            )
        ):
            obs, act, state = data
            plot = (i == 0)

            state_dim = getattr(self.args, "state_dim", 0)
            if state_dim > 0:
                odom_val      = state[:, :, :self.D_odom].to(self.device)
                collision_val = state[:, :, -1:].to(self.device)
            else:
                odom_val      = None
                collision_val = None

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                        obs, act,
                        odom=odom_val,
                        collision=collision_val,
                    )

                    if z_out is not None:
                        act_tgt = act[:, self.model.num_pred:].detach()
                        act_pred = self.model.decode_act(z_out)
                        action_decode_loss = F.mse_loss(act_pred, act_tgt)
                        loss_components["action_decode_loss"] = action_decode_loss

                val_losses.append(loss.item())

                loss_components_gathered = gather_dict(
                    {
                        k: (v.detach() if isinstance(v, torch.Tensor) else torch.tensor(float(v)))
                        for k, v in loss_components.items()
                    },
                    self.args,
                )
                loss_components_logged = {k: v.mean().item() for k, v in loss_components_gathered.items()}

                if self.args.has_decoder and plot:
                    self._log_image_metrics(obs, z_out, visual_out, visual_reconstructed, phase="val")

            self.logs_update({f"val_{k}": [v] for k, v in loss_components_logged.items()})

        # Save a "best" checkpoint whenever validation loss improves
        if val_losses and self.main_process:
            mean_val = sum(val_losses) / len(val_losses)
            if mean_val < self._best_val_loss:
                self._best_val_loss = mean_val
                self.save_ckpt(tag="best")
                log.info(f"New best val loss {mean_val:.4f} at epoch {self.epoch}")

    def _log_image_metrics(self, obs, z_out, visual_out, visual_reconstructed, phase):
        """Compute embedding errors and image quality scores; save sample plots."""
        if self.args.has_predictor and z_out is not None:
            z_obs_out, _ = self.model.separate_emb(z_out)
            z_gt = self.model.encode_obs(obs)
            z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)
            err_logs = self.err_eval(z_obs_out, z_tgt)
            err_logs = gather_dict({k: v.detach() for k, v in err_logs.items()}, self.args)
            self.logs_update(
                {f"{phase}_{k}": [v.mean().item()] for k, v in err_logs.items()}
            )

        if visual_out is not None:
            for t in range(self.args.num_hist, self.args.num_hist + self.args.num_pred):
                scores = eval_images(
                    visual_out[:, t - self.args.num_pred], obs["visual"][:, t]
                )
                scores = gather_dict({k: v.detach() for k, v in scores.items()}, self.args)
                self.logs_update(
                    {f"{phase}_img_{k}_pred": [v.mean().item()] for k, v in scores.items()}
                )

        if visual_reconstructed is not None:
            for t in range(obs["visual"].shape[1]):
                scores = eval_images(visual_reconstructed[:, t], obs["visual"][:, t])
                scores = gather_dict({k: v.detach() for k, v in scores.items()}, self.args)
                self.logs_update(
                    {f"{phase}_img_{k}_reconstructed": [v.mean().item()] for k, v in scores.items()}
                )

        self.plot_samples(
            obs["visual"],
            visual_out,
            visual_reconstructed,
            self.epoch,
            batch=0,
            num_samples=self.num_reconstruct_samples,
            phase=phase,
        )

    # ── Open-loop rollout evaluation ─────────────────────────────────────────

    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        np.random.seed(self.args.seed)
        min_horizon = min_horizon + self.num_hist
        plotting_dir = os.path.join(
            self.args.out_dir, "rollout_plots", f"e{self.epoch}_rollout"
        )
        if self.main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        barrier(self.args)
        logs = {}
        num_past_configs = [(self.num_hist, ""), (1, "_1framestart")]

        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.frameskip

            for k in obs.keys():
                obs[k] = obs[k][start : start + horizon * self.frameskip + 1 : self.frameskip]
            act = act[start : start + horizon * self.frameskip : self.frameskip]

            obs_g = {k: obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device) for k in obs}
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0)

            for n_past, postfix in num_past_configs:
                obs_0 = {k: obs[k][:n_past].unsqueeze(0).to(self.device) for k in obs}
                z_obses, _ = self.model.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k, v in div_loss.items():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    logs.setdefault(log_key, []).append(v)

                if self.args.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        os.path.join(
                            plotting_dir, f"e{self.epoch}_{mode}_{idx}{postfix}.png"
                        ),
                    )

        return {
            key: sum(vals) / len(vals) for key, vals in logs.items() if vals
        }

    # ── Embedding error helpers ──────────────────────────────────────────────

    def err_eval_single(self, z_pred, z_tgt):
        return {k: self.model.emb_criterion(z_pred[k], z_tgt[k]) for k in z_pred}

    def err_eval(self, z_out, z_tgt):
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_s = slice_trajdict_with_t(z_out, start_idx=start_idx, end_idx=end_idx)
            z_tgt_s = slice_trajdict_with_t(z_tgt, start_idx=start_idx, end_idx=end_idx)
            logs.update(
                {f"z_{k}_err_{name}": v for k, v in self.err_eval_single(z_out_s, z_tgt_s).items()}
            )
        return logs

    # ── Logging helpers ──────────────────────────────────────────────────────

    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = [value.detach().cpu().item()]
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (count + len(value), total + sum(value))

    def logs_flash(self, step):
        epoch_log = OrderedDict(
            {k: total / count for k, (count, total) in self.epoch_log.items()}
        )
        epoch_log["epoch"] = step
        train_loss = epoch_log.get("train_loss", float("nan"))
        val_loss = epoch_log.get("val_loss", float("nan"))

        if self.main_process:
            log.info(
                f"Epoch {self.epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )
            if self.wandb_run is not None:
                self.wandb_run.log(epoch_log)

        self.epoch_log = OrderedDict()

    # ── Image plotting ───────────────────────────────────────────────────────

    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        """Save a grid of GT / predicted / reconstructed frames for visual inspection."""
        num_frames = gt_imgs.shape[1]
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )
        num_samples = min(num_samples, gt_imgs.shape[0])

        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.model.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        phase_dir = os.path.join(self.args.out_dir, phase)
        if self.main_process:
            os.makedirs(phase_dir, exist_ok=True)
        barrier(self.args)

        self.plot_imgs(
            imgs,
            num_samples * num_frames,
            os.path.join(phase_dir, f"{phase}_e{str(epoch).zfill(5)}_b{batch}.png"),
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        if self.main_process:
            vutils.save_image(
                imgs, img_name, nrow=num_columns, normalize=True, value_range=(-1, 1)
            )


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unified world-model trainer (--mode wm | dinowm)"
    )

    # ── Mode ─────────────────────────────────────────────────────────────────
    ap.add_argument(
        "--mode", type=str, default="wm", choices=["wm", "dinowm"],
        help="Training mode: 'wm' (original WorldModel) or 'dinowm' (DINO-WM modular)",
    )

    # ── Shared args ───────────────────────────────────────────────────────────
    ap.add_argument(
        "--dataset", "--data_dir", type=str, required=True, dest="dataset",
        help="Path to dataset (HuggingFace dir/ID or local .npz dir)",
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--experiment_name", type=str, default=None,
                    help="WandB run name (default: auto-generated)")
    ap.add_argument("--wandb", action="store_true", default=False,
                    help="Enable WandB logging")
    ap.add_argument("--ddp", action="store_true", default=False,
                    help="Enable DistributedDataParallel training")

    # ── WM-mode args ──────────────────────────────────────────────────────────
    wm = ap.add_argument_group("wm mode options")
    wm.add_argument("--dataset_type", type=str, default="hf", choices=["hf", "npz"])
    wm.add_argument("--horizon", type=int, default=8,
                    help="[wm] Multi-step rollout horizon")
    wm.add_argument("--lr", type=float, default=3e-4,
                    help="[wm] Global learning rate (single AdamW optimizer)")
    wm.add_argument("--backbone", type=str, default="cnn",
                    help="[wm] Backbone: cnn, dinov2, or vjepa2")
    wm.add_argument("--freeze_backbone", action="store_true")
    wm.add_argument(
        "--dynamics_type", type=str, default="mlp", choices=["mlp", "rssm", "td-jepa"]
    )
    wm.add_argument("--ema_tau", type=float, default=0.0,
                    help="EMA coefficient: for [wm] TD-JEPA use ~0.996; for [dinowm] I-JEPA use ~0.996 (0 = disabled)")
    wm.add_argument("--pose_threshold", type=float, default=0.1,
                    help="[wm] L2 threshold for post-training pose accuracy metric")
    wm.add_argument("--resume", type=str, default=None,
                    help="[wm] Checkpoint path to resume from (auto-detects latest.pt)")

    # ── DINO-WM mode args ─────────────────────────────────────────────────────
    dw = ap.add_argument_group("dinowm mode options")
    dw.add_argument(
        "--backbone_name", type=str, default="dinov2_vits14",
        help="[dinowm] DINOv2 variant: dinov2_vits14, dinov2_vitb14, …",
    )
    dw.add_argument("--num_hist", type=int, default=3,
                    help="[dinowm] Number of history frames")
    dw.add_argument("--num_pred", type=int, default=1,
                    help="[dinowm] Number of prediction steps (currently only 1 supported)")
    dw.add_argument("--frameskip", type=int, default=5)
    dw.add_argument("--concat_dim", type=int, default=1,
                    help="[dinowm] 0=cat along patch dim; 1=cat along embedding dim")
    dw.add_argument("--action_emb_dim", type=int, default=10)
    dw.add_argument("--proprio_emb_dim", type=int, default=10)
    dw.add_argument("--num_action_repeat", type=int, default=1)
    dw.add_argument("--num_proprio_repeat", type=int, default=1)
    # Component toggles
    dw.add_argument("--has_predictor", action="store_true", default=True)
    dw.add_argument("--no_predictor", dest="has_predictor", action="store_false")
    dw.add_argument("--has_decoder", action="store_true", default=False)
    dw.add_argument("--train_encoder", action="store_true", default=False)
    dw.add_argument("--train_predictor", action="store_true", default=True)
    dw.add_argument("--no_train_predictor", dest="train_predictor", action="store_false")
    dw.add_argument("--train_decoder", action="store_true", default=False)
    # Per-component learning rates (preserved from original DINO-WM defaults)
    dw.add_argument("--encoder_lr", type=float, default=1e-6,
                    help="[dinowm] Encoder learning rate (Adam)")
    dw.add_argument("--predictor_lr", type=float, default=5e-4,
                    help="[dinowm] ViT predictor learning rate (AdamW)")
    dw.add_argument("--decoder_lr", type=float, default=3e-4,
                    help="[dinowm] Decoder learning rate (Adam)")
    dw.add_argument("--action_encoder_lr", type=float, default=5e-4,
                    help="[dinowm] Action/proprio encoder + action decoder head LR (AdamW)")
    # Predictor architecture
    dw.add_argument("--predictor_depth", type=int, default=6)
    dw.add_argument("--predictor_heads", type=int, default=16)
    dw.add_argument("--predictor_mlp_dim", type=int, default=2048)
    dw.add_argument("--predictor_dropout", type=float, default=0.1)
    # Decoder architecture (VQVAE, used when --has_decoder with no --decoder_path)
    dw.add_argument("--decoder_path", type=str, default=None,
                    help="[dinowm] Pre-trained decoder checkpoint (skips VQVAE init)")
    dw.add_argument("--decoder_n_embed", type=int, default=2048)
    dw.add_argument("--decoder_n_res_block", type=int, default=4)
    dw.add_argument("--decoder_n_res_channel", type=int, default=128)
    # Checkpointing
    dw.add_argument("--save_every_x_epoch", type=int, default=1)
    dw.add_argument("--num_reconstruct_samples", type=int, default=6)
    # Data normalisation (inline JSON strings for portability)
    dw.add_argument(
        "--odom_stats_json", type=str, default=None,
        help='[dinowm] JSON {"mean":[...], "std":[...]} for odom normalisation',
    )
    dw.add_argument("--imu_stats_json", type=str, default=None,
                    help="[dinowm] JSON stats for IMU normalisation")
    dw.add_argument("--action_stats_json", type=str, default=None,
                    help="[dinowm] JSON stats for action normalisation")
    # Optional planning eval integration
    dw.add_argument("--plan_cfg_path", type=str, default=None,
                    help="[dinowm] Path to planning config for async eval jobs")
    dw.add_argument("--plan_planner", type=str, nargs="+", default=["gd", "cem"])
    dw.add_argument("--plan_goal_source", type=str, nargs="+",
                    default=["dset", "random_state"])
    dw.add_argument("--plan_goal_H", type=int, nargs="+", default=[5])
    dw.add_argument("--plan_alpha", type=float, nargs="+", default=[0.1, 1.0])
    dw.add_argument("--debug", action="store_true", default=False,
                    help="[dinowm] Use dino_wm_debug WandB project")
    # Training improvements
    dw.add_argument("--use_amp", action="store_true", default=False,
                    help="[dinowm] Enable automatic mixed precision (bfloat16/float16)")
    dw.add_argument("--accum_steps", type=int, default=1,
                    help="[dinowm] Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    dw.add_argument("--clip_grad_norm", type=float, default=1.0,
                    help="[dinowm] Max gradient norm for clipping (0.0 = disabled)")
    # Note: --ema_tau is shared with the wm group (default 0.996); pass --ema_tau 0 to disable for dinowm
    dw.add_argument("--lr_schedule", type=str, default="none", choices=["none", "cosine"],
                    help="[dinowm] LR schedule: none or cosine with linear warmup")
    dw.add_argument("--lr_warmup_epochs", type=int, default=0,
                    help="[dinowm] Linear warmup epochs before cosine decay")
    dw.add_argument("--lambda_z", type=float, default=1.0,
                    help="[dinowm] Weight for the JEPA embedding prediction loss")
    dw.add_argument("--lambda_action_decode", type=float, default=1.0,
                    help="[dinowm] Weight for the action decode auxiliary loss")
    # Swappable predictor
    dw.add_argument(
        "--predictor_type", type=str, default="vit", choices=["vit", "mlp", "rssm"],
        help="[dinowm] Dynamics predictor: 'vit' (patch-level JEPA), 'mlp' (frame-level residual), 'rssm' (recurrent)",
    )
    # State / risk auxiliary supervision
    dw.add_argument("--state_dim", type=int, default=0,
                    help="[dinowm] Odom dimensions to supervise via state_head (0 = disabled)")
    dw.add_argument("--lambda_state", type=float, default=1.0,
                    help="[dinowm] Weight for state prediction MSE loss")
    dw.add_argument("--lambda_risk", type=float, default=0.2,
                    help="[dinowm] Weight for collision risk BCE loss")

    return ap


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    # ── DDP setup ─────────────────────────────────────────────────────────────
    local_rank = 0
    if args.ddp:
        local_rank = setup_ddp()
        args.device = f"cuda:{local_rank}"
    args.local_rank = local_rank

    main_process = is_main_process(args)
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Route to DINO-WM trainer ──────────────────────────────────────────────
    if args.mode == "dinowm":
        trainer = DinoWMTrainer(args)
        trainer.run()
        if args.ddp:
            cleanup_ddp()
        return

    # ─── Original WM training ─────────────────────────────────────────────────
    if args.experiment_name is None:
        ds_short = args.dataset.replace("/", "_")
        args.experiment_name = f"{ds_short}_{args.backbone}"

    use_wandb = args.wandb and os.environ.get("WANDB_API_KEY") and main_process
    if use_wandb:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "world_model"),
            name=os.environ.get("WANDB_RUN_NAME", args.experiment_name),
            config=vars(args),
        )
    elif args.wandb and main_process:
        print("[WARN] --wandb enabled but WANDB_API_KEY not set. Logging disabled.")
        use_wandb = False

    if args.dataset_type == "hf":
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

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory="cuda" in str(args.device),
    )

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
        dynamics_type="mlp" if args.dynamics_type == "td-jepa" else args.dynamics_type,
    ).to(args.device)

    target_model = None
    if args.dynamics_type == "td-jepa":
        target_model = copy.deepcopy(model)
        for p in target_model.parameters():
            p.requires_grad = False

    if args.ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=(args.dynamics_type == "td-jepa"),
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cfg = TrainConfig(device=args.device, lr=args.lr, horizon=args.horizon)

    # Auto-resume
    start_epoch = 0
    checkpoint_path = args.resume
    if checkpoint_path is None:
        latest = os.path.join(args.out_dir, "latest.pt")
        if os.path.exists(latest):
            checkpoint_path = latest
            if main_process:
                print(f"[RECOVERY] Found {latest}. Resuming automatically...")

    if checkpoint_path and os.path.exists(checkpoint_path):
        if main_process:
            print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=args.device)
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", -1) + 1
        if main_process:
            print(f"Resuming from epoch {start_epoch}")

    if main_process:
        print(
            f"[INFO] Starting training: epochs={args.epochs}, batch_size={args.batch_size}, "
            f"backbone={args.backbone}, dynamics={args.dynamics_type}, ddp={args.ddp}"
        )

    t0 = time.time()
    step = 0
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

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
                    for op, tp in zip(model.parameters(), target_model.parameters()):
                        tp.data.mul_(args.ema_tau).add_(op.data, alpha=1.0 - args.ema_tau)

            epoch_loss += logs["loss"]
            epoch_steps += 1

            if step % 50 == 0 and main_process:
                print(
                    f"epoch={epoch:03d} step={step:06d} loss={logs['loss']:.4f} "
                    f"z={logs['loss_z']:.4f} state={logs['loss_state']:.4f} "
                    f"risk={logs['loss_risk']:.4f}"
                )
                if use_wandb:
                    import wandb
                    wandb.log(
                        {
                            "train/loss": logs["loss"],
                            "train/loss_z": logs["loss_z"],
                            "train/loss_state": logs["loss_state"],
                            "train/loss_risk": logs["loss_risk"],
                            "train/step": step,
                            "train/epoch": epoch,
                        },
                        step=step,
                    )
            step += 1

        if main_process:
            avg_loss = epoch_loss / max(epoch_steps, 1)
            print(f"[Epoch {epoch:03d}] avg_loss={avg_loss:.4f}")
            if use_wandb:
                import wandb
                wandb.log({"epoch/avg_loss": avg_loss, "epoch": epoch}, step=step)

            raw_model = model.module if isinstance(model, DDP) else model
            ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "model_config": {
                    "odom_dim": odom_dim,
                    "imu_dim": imu_dim,
                    "act_dim": act_dim,
                    "state_dim": state_dim,
                },
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"))
            torch.save(ckpt, os.path.join(args.out_dir, "latest.pt"))

        if args.ddp:
            dist.barrier()

    train_time = time.time() - t0

    if main_process:
        print(f"Training complete. Total time: {train_time:.1f}s")
        print("\n" + "=" * 60)
        print("[Evaluation] Computing pose MSE and accuracy ...")
        print("=" * 60)

        eval_dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory="cuda" in str(args.device),
        )
        eval_metrics = evaluate_pose(model, eval_dl, cfg, threshold=args.pose_threshold)

        print(f"  Pose MSE:      {eval_metrics['pose_mse']:.6f}")
        print(f"  Pose Accuracy: {eval_metrics['pose_accuracy']:.4f}  (threshold={args.pose_threshold})")

        if use_wandb:
            import wandb
            wandb.log(
                {
                    "eval/pose_mse": eval_metrics["pose_mse"],
                    "eval/pose_accuracy": eval_metrics["pose_accuracy"],
                }
            )
            wandb.summary["pose_mse"] = eval_metrics["pose_mse"]
            wandb.summary["pose_accuracy"] = eval_metrics["pose_accuracy"]
            wandb.summary["train_time_s"] = train_time
            wandb.finish()

        metrics_path = os.path.join(args.out_dir, "eval_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(
                {**eval_metrics, "train_time_s": train_time, "experiment": args.experiment_name},
                fh,
                indent=2,
            )
        print(f"  Metrics saved to {metrics_path}")
        print(f"\nLatest checkpoint: {os.path.join(args.out_dir, 'latest.pt')}")

    if args.ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
