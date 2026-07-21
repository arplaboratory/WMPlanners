from __future__ import annotations
import os
import time
import json
import argparse
import torch
import logging
import warnings
import threading
import itertools
import numpy as np
from tqdm import tqdm
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from world_model.metrics import eval_images
from world_model.utils import slice_trajdict_with_t, seed, sample_tensors
from world_model.data.hf_datasets import collate_dinowm, HFWindowDataset, EpisodeDataset
from world_model.backbones.base import ModelBundle
from world_model.backbones.proprio import ProprioceptiveEmbedding
from world_model.backbones.latentdynamics import ViTPredictor
from world_model.backbones.wm import WM
# VQVAE is imported lazily in _build_decoder to avoid pulling in distributed_fn
# when no decoder is used

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args):
        self.args = args
        os.makedirs(args.out_dir, exist_ok=True)

        out_dir_abs = os.path.abspath(args.out_dir)
        log.info(f"Model saved dir: {out_dir_abs}")
        if "outputs/" in out_dir_abs:
            model_name = out_dir_abs.split("outputs/")[-1]
        else:
            model_name = os.path.basename(out_dir_abs) or "run"
        model_name += f"_{args.env_name}_f{args.frameskip}_h{args.num_hist}_p{args.num_pred}"

        # SLURM multi-node DDP setup
        if int(os.environ.get("SLURM_NTASKS", 1)) > 1:
            log.info("SLURM multi-node setup begin...")
            log.info(f"SLURM_JOB_NODELIST={os.environ.get('SLURM_JOB_NODELIST', '')}")
            log.info(f"DEBUGVAR={os.environ.get('DEBUGVAR', '')}")
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),
                )
                log.info("Multi-node setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()

        self.accelerator = Accelerator(log_with="wandb" if args.wandb else None)
        log.info(f"rank: {self.accelerator.local_process_index}  model_name: {model_name}")
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = args.num_reconstruct_samples
        self.total_epochs = args.epochs
        self.epoch = 0

        assert args.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {args.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        args.effective_batch_size = args.batch_size
        args.gpu_batch_size = args.batch_size // self.accelerator.num_processes

        self.wandb_run = None
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            run_info_path = os.path.join(args.out_dir, "run_info.json")
            if os.path.exists(run_info_path):
                with open(run_info_path) as f:
                    run_info = json.load(f)
                wandb_run_id = run_info.get("wandb_run_id")
                log.info(f"Resuming Wandb run {wandb_run_id}")

            if args.wandb:
                import wandb
                wandb_dict = vars(args)
                if args.debug:
                    log.info("WARNING: Running in debug mode...")
                    self.wandb_run = wandb.init(
                        project="dino_wm_debug",
                        config=wandb_dict,
                        id=wandb_run_id,
                        resume="allow",
                    )
                else:
                    self.wandb_run = wandb.init(
                        project="dino_wm",
                        config=wandb_dict,
                        id=wandb_run_id,
                        resume="allow",
                    )
                args.wandb_run_id = self.wandb_run.id
                self.wandb_run.name = model_name
                with open(run_info_path, "w") as f:
                    json.dump({"wandb_run_id": self.wandb_run.id}, f)

        seed(args.seed)

        odom_stats = imu_stats = collision_stats = action_stats = None
        if args.stats_file is not None and os.path.exists(args.stats_file):
            with open(args.stats_file) as f:
                stats = json.load(f)
            odom_stats = stats.get("odom")
            imu_stats = stats.get("imu")
            collision_stats = stats.get("collision")
            action_stats = stats.get("action")

        if args.trajectory_length is None:
            args.trajectory_length = args.num_hist + args.num_pred

        log.info(f"Loading dataset from {args.data_path} ...")
        _ds = HFWindowDataset(
            args.data_path,
            img_size=args.img_size,
            num_hist=args.num_hist,
            num_pred=args.num_pred,
            frameskip=args.frameskip,
            odom_stats=odom_stats,
            imu_stats=imu_stats,
            collision_stats=collision_stats,
            action_stats=action_stats,
            sample_mode=True,
            sample_k=args.trajectory_length,
        )
        self.datasets = {"train": _ds, "valid": _ds}

        _traj_ds = HFWindowDataset(
            args.data_path,
            img_size=args.img_size,
            odom_stats=odom_stats,
            imu_stats=imu_stats,
            collision_stats=collision_stats,
            action_stats=action_stats,
            sample_mode=False,
        )
        self.train_traj_dset = EpisodeDataset(_traj_ds)
        self.val_traj_dset = EpisodeDataset(_traj_ds)

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=args.gpu_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_dinowm,
            )
            for x in ["train", "valid"]
        }

        log.info(f"dataloader batch size: {args.gpu_batch_size}")

        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.train_encoder = args.train_encoder
        self.train_predictor = args.train_predictor
        self.train_decoder = args.train_decoder
        log.info(
            f"Train encoder, predictor, decoder: {args.train_encoder} {args.train_predictor} {args.train_decoder}"
        )

        self._keys_to_save = ["epoch"]
        self._keys_to_save += ["encoder", "encoder_optimizer"] if self.train_encoder else []
        self._keys_to_save += (
            ["predictor", "predictor_optimizer"]
            if self.train_predictor and args.has_predictor
            else []
        )
        self._keys_to_save += ["decoder", "decoder_optimizer"] if self.train_decoder else []
        self._keys_to_save += ["action_encoder", "proprio_encoder"]

        self.init_models()
        self.init_optimizers()

        self.epoch_log = OrderedDict()

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            ckpt_dir = os.path.join(self.args.out_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = {}
            for k in self._keys_to_save:
                if hasattr(self.__dict__[k], "module"):
                    ckpt[k] = self.accelerator.unwrap_model(self.__dict__[k])
                else:
                    ckpt[k] = self.__dict__[k]
            torch.save(ckpt, os.path.join(ckpt_dir, "model_latest.pth"))
            torch.save(ckpt, os.path.join(ckpt_dir, f"model_{self.epoch}.pth"))
            log.info("Saved model to {}".format(self.args.out_dir))
            ckpt_path = os.path.join(ckpt_dir, f"model_{self.epoch}.pth")
        else:
            ckpt_path = None
        out_dir_abs = os.path.abspath(self.args.out_dir)
        model_name = out_dir_abs.split("outputs/")[-1] if "outputs/" in out_dir_abs else os.path.basename(out_dir_abs)
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        ckpt = torch.load(filename, weights_only=False)
        for k, v in ckpt.items():
            self.__dict__[k] = v
        not_in_ckpt = set(self._keys_to_save) - set(ckpt.keys())
        if len(not_in_ckpt):
            log.warning("Keys not found in ckpt: %s", not_in_ckpt)

    def _build_encoder(self):
        if self.args.encoder == "dino":
            from world_model.backbones.dino import build_dinov2
            encoder = build_dinov2(model_name=self.args.encoder_name)
        elif self.args.encoder == "vjepa":
            from world_model.backbones.jepa import build_vjepa2
            encoder = build_vjepa2(model_name=self.args.encoder_name)
        else:
            raise ValueError(f"Unknown encoder: {self.args.encoder!r}")
        if isinstance(encoder, ModelBundle):
            encoder = encoder.model
        return encoder

    def _build_decoder(self):
        args = self.args
        if args.decoder_path is not None:
            decoder_path = os.path.join(self.base_path, args.decoder_path)
            ckpt = torch.load(decoder_path, weights_only=False)
            decoder = ckpt["decoder"] if isinstance(ckpt, dict) else ckpt
            log.info(f"Loaded decoder from {decoder_path}")
            return decoder
        if args.decoder_type == "vqvae":
            from world_model.backbones.decoder import VQVAE
            return VQVAE(
                channel=args.decoder_channel,
                n_embed=args.decoder_n_embed,
                n_res_block=args.decoder_n_res_block,
                n_res_channel=args.decoder_n_res_channel,
                emb_dim=self.encoder.emb_dim,
                quantize=args.decoder_quantize,
            )
        if args.decoder_type == "transposed_conv":
            from models.decoder.transposed_conv import TransposedConvDecoder
            return TransposedConvDecoder(
                observation_shape=[3, args.img_size, args.img_size],
                depth=args.tconv_depth,
                kernel_size=args.tconv_kernel_size,
                stride=args.tconv_stride,
            )
        raise ValueError(f"Unknown decoder_type: {args.decoder_type!r}")

    def init_models(self):
        model_ckpt = Path(self.args.out_dir) / "checkpoints" / "model_latest.pth"
        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")

        if self.encoder is None:
            self.encoder = self._build_encoder()
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.proprio_encoder = ProprioceptiveEmbedding(
            num_frames=1,
            tubelet_size=1,
            use_3d_pos=False,
            in_chans=self.datasets["train"].proprio_dim,
            emb_dim=self.args.proprio_emb_dim,
        )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        self.action_encoder = ProprioceptiveEmbedding(
            num_frames=1,
            tubelet_size=1,
            use_3d_pos=False,
            in_chans=self.datasets["train"].action_dim,
            emb_dim=self.args.action_emb_dim,
        )
        action_emb_dim = self.action_encoder.emb_dim
        print(f"Action encoder type: {type(self.action_encoder)}")
        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        if self.accelerator.is_main_process and self.wandb_run is not None:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        if self.encoder.latent_ndim == 1:
            num_patches = 1
        elif hasattr(self.encoder, "num_patches"):
            num_patches = self.encoder.num_patches
        else:
            num_side_patches = self.args.img_size // 16
            num_patches = num_side_patches ** 2

        if self.args.concat_dim == 0:
            num_patches += 2

        if self.args.has_predictor:
            if self.predictor is None:
                self.predictor = ViTPredictor(
                    num_patches=num_patches,
                    num_frames=self.args.num_hist,
                    dim=(
                        self.encoder.emb_dim
                        + (
                            proprio_emb_dim * self.args.num_proprio_repeat
                            + action_emb_dim * self.args.num_action_repeat
                        )
                        * self.args.concat_dim
                    ),
                    depth=self.args.predictor_depth,
                    heads=self.args.predictor_heads,
                    mlp_dim=self.args.predictor_mlp_dim,
                    dropout=self.args.predictor_dropout,
                    emb_dropout=self.args.predictor_emb_dropout,
                    pool=self.args.predictor_pool,
                )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        if self.args.has_decoder:
            if self.decoder is None:
                self.decoder = self._build_decoder()
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False

        self.encoder, self.predictor, self.decoder = self.accelerator.prepare(
            self.encoder, self.predictor, self.decoder
        )

        self.model = WM(
            image_size=self.args.img_size,
            num_hist=self.args.num_hist,
            num_pred=self.args.num_pred,
            train_encoder=self.args.train_encoder,
            train_predictor=self.args.train_predictor,
            train_decoder=self.args.train_decoder,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.args.concat_dim,
            num_action_repeat=self.args.num_action_repeat,
            num_proprio_repeat=self.args.num_proprio_repeat,
            raw_action_dim=self.datasets["train"].action_dim,
        )
        self.model = self.model.to(self.device)

    def init_optimizers(self):
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.args.encoder_lr,
        )
        self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)

        if self.args.has_predictor:
            self.predictor_optimizer = torch.optim.AdamW(
                self.predictor.parameters(),
                lr=self.args.predictor_lr,
            )
            self.predictor_optimizer = self.accelerator.prepare(self.predictor_optimizer)

            self.action_encoder_optimizer = torch.optim.AdamW(
                itertools.chain(
                    self.action_encoder.parameters(),
                    self.proprio_encoder.parameters(),
                    self.model.action_decoder.parameters(),
                ),
                lr=self.args.action_encoder_lr,
            )
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )

        if self.args.has_decoder:
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(), lr=self.args.decoder_lr
            )
            self.decoder_optimizer = self.accelerator.prepare(self.decoder_optimizer)

    def monitor_jobs(self, lock):
        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    result = job.result()
                    print(f"Logging result for {job_name} at epoch {epoch}: {result}")
                    if self.wandb_run is not None:
                        log_data = {
                            f"{job_name}/{key}": value for key, value in result.items()
                        }
                        log_data["epoch"] = epoch
                        self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    def run(self):
        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()

        init_epoch = self.epoch + 1
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()
            self.train()
            self.accelerator.wait_for_everyone()
            self.val()
            self.logs_flash(step=self.epoch)
            if self.epoch % self.args.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt()
                if (
                    self.args.plan_cfg_path is not None
                    and ckpt_path is not None
                ):
                    from plan import build_plan_cfg_dicts, launch_plan_jobs

                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=os.path.join(
                            self.base_path, self.args.plan_cfg_path
                        ),
                        ckpt_base_path=self.args.ckpt_base_path,
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=self.args.planner,
                        goal_source=self.args.goal_source,
                        goal_H=self.args.goal_H,
                        alpha=self.args.alpha,
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

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(z_out, start_idx=start_idx, end_idx=end_idx)
            z_tgt_slice = slice_trajdict_with_t(z_tgt, start_idx=start_idx, end_idx=end_idx)
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)
            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})
        return logs

    def train(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            obs, act, state = data
            plot = i == 0
            self.model.train()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(obs, act)

            if z_out is not None:
                act_tgt = act[:, self.model.num_pred:].detach()
                act_pred = self.model.decode_act(z_out)
                action_decode_loss = torch.nn.functional.mse_loss(act_pred, act_tgt)
                loss = loss + action_decode_loss
                loss_components["action_decode_loss"] = action_decode_loss

            self.encoder_optimizer.zero_grad()
            if self.args.has_decoder:
                self.decoder_optimizer.zero_grad()
            if self.args.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

            self.accelerator.backward(loss)

            if self.model.train_encoder:
                self.encoder_optimizer.step()
            if self.args.has_decoder and self.model.train_decoder:
                self.decoder_optimizer.step()
            if self.args.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            if self.args.has_decoder and plot:
                if self.args.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)
                    err_logs = self.err_eval(z_obs_out, z_tgt)
                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {key: value.mean().item() for key, value in err_logs.items()}
                    err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}
                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(self.args.num_hist, self.args.num_hist + self.args.num_pred):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.args.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(img_pred_scores)
                        img_pred_scores = {
                            f"train_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"train_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )

            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def val(self):
        self.model.eval()
        if len(self.train_traj_dset) > 0 and self.args.has_predictor:
            with torch.no_grad():
                train_rollout_logs = self.openloop_rollout(self.train_traj_dset, mode="train")
                self.logs_update({f"train_{k}": [v] for k, v in train_rollout_logs.items()})
                val_rollout_logs = self.openloop_rollout(self.val_traj_dset, mode="val")
                self.logs_update({f"val_{k}": [v] for k, v in val_rollout_logs.items()})

        self.accelerator.wait_for_everyone()
        for i, data in enumerate(
            tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")
        ):
            obs, act, state = data
            plot = i == 0
            self.model.eval()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(obs, act)

            if z_out is not None:
                act_tgt = act[:, self.model.num_pred:].detach()
                act_pred = self.model.decode_act(z_out)
                action_decode_loss = torch.nn.functional.mse_loss(act_pred, act_tgt)
                loss_components["action_decode_loss"] = action_decode_loss

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            if self.args.has_decoder and plot:
                if self.args.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)
                    err_logs = self.err_eval(z_obs_out, z_tgt)
                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {key: value.mean().item() for key, value in err_logs.items()}
                    self.logs_update({f"val_{k}": [v] for k, v in err_logs.items()})

                if visual_out is not None:
                    for t in range(self.args.num_hist, self.args.num_hist + self.args.num_pred):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.args.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(img_pred_scores)
                        img_pred_scores = {
                            f"val_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"val_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="valid",
                )

            loss_components = {f"val_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        np.random.seed(self.args.seed)
        min_horizon = min_horizon + self.args.num_hist
        plotting_dir = os.path.join(self.args.out_dir, f"rollout_plots/e{self.epoch}_rollout")
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        num_past = [(self.args.num_hist, ""), (1, "_1framestart")]

        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.args.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.args.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.args.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.args.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                    start:
                    start + horizon * self.args.frameskip + 1:
                    self.args.frameskip
                ]
            act = act[start: start + horizon * self.args.frameskip: self.args.frameskip]

            obs_g = {k: obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device) for k in obs}
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0)

            for n_past, postfix in num_past:
                obs_0 = {
                    k: obs[k][:n_past].unsqueeze(0).to(self.device) for k in obs
                }

                z_obses, z = self.model.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    logs.setdefault(log_key, []).append(div_loss[k])

                if self.args.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        os.path.join(plotting_dir, f"e{self.epoch}_{mode}_{idx}{postfix}.png"),
                    )

        logs = {key: sum(values) / len(values) for key, values in logs.items() if values}
        return logs

    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            length = len(value)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (count + length, total + sum(value))

    def logs_flash(self, step):
        epoch_log = OrderedDict()
        for key, (count, total) in self.epoch_log.items():
            epoch_log[key] = total / count
        epoch_log["epoch"] = step
        train_loss = epoch_log.get("train_loss", float("nan"))
        val_loss = epoch_log.get("val_loss", float("nan"))
        log.info(
            f"Epoch {self.epoch}  Training loss: {train_loss:.4f}  Validation loss: {val_loss:.4f}"
        )
        if self.accelerator.is_main_process and self.wandb_run is not None:
            self.wandb_run.log(epoch_log)
        self.epoch_log = OrderedDict()

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
        reconstructed_gt_imgs = rearrange(reconstructed_gt_imgs, "b t c h w -> (b t) c h w")
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        phase_dir = os.path.join(self.args.out_dir, phase)
        if self.accelerator.is_main_process:
            os.makedirs(phase_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=os.path.join(phase_dir, f"{phase}_e{str(epoch).zfill(5)}_b{batch}.png"),
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        utils.save_image(imgs, img_name, nrow=num_columns, normalize=True, value_range=(-1, 1))


def build_parser(parents=None):
    ap = argparse.ArgumentParser(
        description="Train a DinoWM world model",
        parents=parents or [],
        add_help=parents is None,
    )

    # Dataset
    ap.add_argument("--data_path", type=str, required=True, help="HuggingFace dataset ID or local path")
    ap.add_argument("--env_name", type=str, default="collect1")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--stats_file", type=str, default=None,
                    help="JSON file with odom/imu/collision/action normalisation stats")

    # Output
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--ckpt_base_path", type=str, default="./")

    # Training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every_x_epoch", type=int, default=1)
    ap.add_argument("--num_reconstruct_samples", type=int, default=6)
    ap.add_argument("--encoder_lr", type=float, default=1e-6)
    ap.add_argument("--predictor_lr", type=float, default=5e-4)
    ap.add_argument("--action_encoder_lr", type=float, default=5e-4)
    ap.add_argument("--decoder_lr", type=float, default=3e-4)

    # Sequence geometry
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--num_hist", type=int, default=3)
    ap.add_argument("--num_pred", type=int, default=1)
    ap.add_argument("--trajectory_length", type=int, default=None,
                    help="Defaults to num_hist + num_pred")
    ap.add_argument("--concat_dim", type=int, default=1)
    ap.add_argument("--action_emb_dim", type=int, default=10)
    ap.add_argument("--proprio_emb_dim", type=int, default=10)
    ap.add_argument("--num_action_repeat", type=int, default=1)
    ap.add_argument("--num_proprio_repeat", type=int, default=1)

    # Module flags
    ap.add_argument("--has_predictor", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--has_decoder", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--train_encoder", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--train_predictor", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--train_decoder", action=argparse.BooleanOptionalAction, default=False)

    # Encoder
    ap.add_argument("--encoder", type=str, default="dino", choices=["dino", "vjepa"])
    ap.add_argument("--encoder_name", type=str, default="dinov2_vits14",
                    help="Model variant, e.g. dinov2_vits14 or vjepa2_vit_large")

    # Predictor (ViTPredictor)
    ap.add_argument("--predictor_depth", type=int, default=6)
    ap.add_argument("--predictor_heads", type=int, default=16)
    ap.add_argument("--predictor_mlp_dim", type=int, default=2048)
    ap.add_argument("--predictor_dropout", type=float, default=0.1)
    ap.add_argument("--predictor_emb_dropout", type=float, default=0.0)
    ap.add_argument("--predictor_pool", type=str, default="mean")

    # Decoder
    ap.add_argument("--decoder_type", type=str, default="vqvae",
                    choices=["vqvae", "transposed_conv"],
                    help="Decoder architecture when has_decoder is True")
    ap.add_argument("--decoder_path", type=str, default=None,
                    help="Load decoder from checkpoint instead of building from scratch")
    ap.add_argument("--decoder_channel", type=int, default=384)
    ap.add_argument("--decoder_n_embed", type=int, default=2048)
    ap.add_argument("--decoder_n_res_block", type=int, default=4)
    ap.add_argument("--decoder_n_res_channel", type=int, default=128)
    ap.add_argument("--decoder_quantize", action="store_true", default=False)
    ap.add_argument("--tconv_depth", type=int, default=64)
    ap.add_argument("--tconv_kernel_size", type=int, default=5)
    ap.add_argument("--tconv_stride", type=int, default=3)

    # Logging
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--debug", action="store_true", default=False)

    # Planning eval (launched after each checkpoint save)
    ap.add_argument("--plan_cfg_path", type=str, default=None)
    ap.add_argument("--planner", nargs="+", default=None)
    ap.add_argument("--goal_source", nargs="+", default=None)
    ap.add_argument("--goal_H", nargs="+", type=int, default=None)
    ap.add_argument("--alpha", nargs="+", type=float, default=None)

    return ap


def main():
    args = build_parser().parse_args()
    Trainer(args).run()


if __name__ == "__main__":
    main()
