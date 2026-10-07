"""Fit state and optional risk heads on a frozen world model.

Observed mode decodes encoded sensor frames. Rollout mode decodes predicted
latents, matching the states used by MPC. Both validation errors are reported.
The dataset must match the checkpoint's sensor preprocessing and actions.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from world_model.architecture import WorldModel, checkpoint_backbone_kwargs
from world_model.data.factory import collate_for_dataset, is_parquet_dataset, open_window_dataset
from world_model.data.normalization import NormStats


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    config = checkpoint["model_config"]
    backbone = str(args.get("backbone", "cnn")).lower()
    dynamics = str(args.get("dynamics_type", "mlp")).lower()
    if dynamics == "td-jepa":
        dynamics = "mlp"
    backbone_kwargs = checkpoint_backbone_kwargs(checkpoint, str(device))
    model = WorldModel(
        odom_dim=config["odom_dim"],
        imu_dim=config["imu_dim"],
        act_dim=config["act_dim"],
        state_dim=config["state_dim"],
        backbone=None if backbone == "cnn" else backbone,
        backbone_kwargs=backbone_kwargs,
        dynamics_type=dynamics,
        image_size=int(args.get("img_size", 224)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


class WindowPrefix(Dataset):
    """Read only the frames needed for head supervision after a window loads."""

    def __init__(self, base: Dataset, length: int, norm_stats: NormStats | None = None):
        self.base = base
        self.length = length
        self.norm_stats = norm_stats

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        sample = {
            key: value[:self.length]
            if key in {"rgb", "depth", "imu", "odom", "action", "collision",
                       "depth_valid", "imu_valid", "odom_valid", "action_valid",
                       "collision_valid", "timestamp_ns", "frame_indices"}
            else value
            for key, value in sample.items()
        }
        if self.norm_stats is not None:
            for key, mean, std in (
                ("odom", self.norm_stats.odom_mean, self.norm_stats.odom_std),
                ("imu", self.norm_stats.imu_mean, self.norm_stats.imu_std),
                ("action", self.norm_stats.act_mean, self.norm_stats.act_std),
            ):
                value = sample[key]
                if value.shape[-1] != mean.numel():
                    raise ValueError(f"{key} dimension does not match norm_stats.pt")
                if isinstance(value, torch.Tensor):
                    sample[key] = (value - mean) / std
                else:
                    sample[key] = (value - mean.numpy()) / std.numpy()
        return sample


def open_splits(cli, image_size: int, window_size: int):
    options = dict(dataset_type=cli.dataset_type, path=cli.dataset,
                   image_size=image_size, window_size=window_size,
                   schema_config=cli.schema_config)
    train = open_window_dataset(split=cli.train_split, **options)
    try:
        val = open_window_dataset(split=cli.val_split, **options)
        return train, val
    except ValueError as error:
        if not is_parquet_dataset(cli.dataset) or "no windows in split" not in str(error):
            raise
    # Some Parquet releases label every window "train". Hold out complete
    # recordings so adjacent windows from the same run cannot leak into val.
    all_data = open_window_dataset(split=None, **options)
    by_recording = {}
    for window in all_data.windows:
        recording = str(all_data._field(window, "recording_id", "windows"))
        by_recording.setdefault(recording, []).append(window)
    if len(by_recording) < 2:
        raise ValueError("At least two recordings are needed for a held-out validation split")
    recordings = sorted(by_recording)
    random.Random(cli.seed).shuffle(recordings)
    target = max(1, round(len(all_data) * cli.validation_fraction))
    val_names = set()
    held_out = 0
    for recording in recordings:
        if held_out >= target and val_names:
            break
        if len(val_names) == len(recordings) - 1:
            break
        val_names.add(recording)
        held_out += len(by_recording[recording])
    train = copy.copy(all_data)
    val = copy.copy(all_data)
    train.windows = [row for row in all_data.windows
                     if str(all_data._field(row, "recording_id", "windows")) not in val_names]
    val.windows = [row for row in all_data.windows
                   if str(all_data._field(row, "recording_id", "windows")) in val_names]
    train._frame_cache = OrderedDict()
    val._frame_cache = OrderedDict()
    print(f"Validation recordings: {sorted(val_names)} ({len(val)} windows)", flush=True)
    return train, val


def _valid(mask: torch.Tensor | None, shape: tuple[int, int], device: torch.device):
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    return mask.to(device=device, dtype=torch.bool).reshape(shape)


def batch_predictions(model: WorldModel, batch, horizon: int, device: torch.device,
                      latent_source: str):
    """Decode observed or action-rolled latents against the same state targets."""
    if batch.rgb.shape[1] < horizon + 1:
        raise ValueError(f"Dataset window has {batch.rgb.shape[1]} frames; need {horizon + 1}")
    rgb = batch.rgb[:, :horizon + 1].to(device)
    depth = torch.nan_to_num(batch.depth[:, :horizon + 1].to(device),
                             nan=0.0, posinf=0.0, neginf=0.0)
    odom = batch.odom[:, :horizon + 1].to(device)
    imu = batch.imu[:, :horizon + 1].to(device)
    action = batch.action[:, :horizon].to(device)
    collision = batch.collision[:, :horizon + 1].to(device)
    count, steps = odom.shape[:2]
    inputs = (rgb, odom, imu, action)
    if not all(torch.isfinite(value).all().item() for value in inputs):
        raise ValueError("Training observations or actions contain nonfinite values")
    if odom.shape[-1] != model.proprio.net[0].in_features - imu.shape[-1]:
        raise ValueError("Dataset odometry dimension differs from checkpoint")
    with torch.no_grad():
        if latent_source == "observed":
            encoded = model.encode(
                rgb.reshape(count * steps, *rgb.shape[2:]),
                depth.reshape(count * steps, *depth.shape[2:]),
                odom.reshape(count * steps, -1),
                imu.reshape(count * steps, -1),
            ).z.reshape(count, steps, -1)
            latents = encoded
        elif latent_source == "rollout":
            z0 = model.encode(rgb[:, 0], depth[:, 0], odom[:, 0], imu[:, 0]).z
            latents = model.rollout(z0, action)
        else:
            raise ValueError(f"Unknown latent source: {latent_source}")
    state = model.state_head(latents)
    risk = model.risk_head(latents)
    state_target = odom[..., :state.shape[-1]]
    state_valid = _valid(batch.odom_valid, (count, batch.odom.shape[1]), device)[:, :steps]
    if latent_source == "rollout":
        action_valid = _valid(batch.action_valid, (count, batch.action.shape[1]), device)[:, :horizon]
        prefix_valid = torch.cat((
            torch.ones((count, 1), dtype=torch.bool, device=device),
            action_valid.cumprod(dim=1).bool(),
        ), dim=1)
        state_valid = state_valid & state_valid[:, :1] & prefix_valid
    risk_valid = _valid(batch.collision_valid, (count, batch.collision.shape[1]), device)[:, :steps]
    risk_valid = risk_valid & state_valid
    return state, risk, state_target, collision, state_valid, risk_valid


def batch_losses(model, batch, horizon, device, latent_source, train_risk):
    state, risk, state_target, collision, state_valid, risk_valid = batch_predictions(
        model, batch, horizon, device, latent_source)
    if not state_valid.any().item():
        raise ValueError("Batch has no valid state targets")
    state_loss = F.mse_loss(state[state_valid], state_target[state_valid])
    position_error = state[state_valid, :3] - state_target[state_valid, :3]
    position_squared = position_error.square().sum(dim=-1)
    valid_risk = collision[risk_valid]
    if train_risk and valid_risk.numel():
        if not torch.isfinite(valid_risk).all().item() or (valid_risk < 0).any() or (valid_risk > 1).any():
            raise ValueError("Valid collision labels must be finite probabilities in [0, 1]")
        risk_loss = F.binary_cross_entropy(risk[risk_valid], valid_risk)
    else:
        risk_loss = state_loss.new_zeros(())
    return {
        "state_loss": state_loss,
        "risk_loss": risk_loss,
        "position_squared": position_squared.detach().sum().item(),
        "position_count": position_squared.numel(),
        "state_count": int(state_valid.sum().item()),
        "risk_count": valid_risk.numel(),
        "risk_positive": int((valid_risk > 0.5).sum().item()),
    }


def run_epoch(model, loader, optimizer, horizon, device, latent_source, train_risk,
              risk_weight, max_steps, evaluate_both=False):
    model.state_head.train(optimizer is not None)
    model.risk_head.train(optimizer is not None and train_risk)
    sources = ("observed", "rollout") if evaluate_both else (latent_source,)
    totals = {
        source: {key: 0.0 for key in (
            "state_loss", "risk_loss", "position_squared", "position_count",
            "state_count", "risk_count", "risk_positive")}
        for source in sources
    }
    batches = 0
    for batch in loader:
        if max_steps and batches >= max_steps:
            break
        for source in sources:
            with torch.set_grad_enabled(optimizer is not None and source == latent_source):
                losses = batch_losses(
                    model, batch, horizon, device, source,
                    train_risk and source == latent_source)
                if optimizer is not None and source == latent_source:
                    loss = losses["state_loss"] + risk_weight * losses["risk_loss"]
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(model.state_head.parameters())
                        + (list(model.risk_head.parameters()) if train_risk else []), 1.0)
                    optimizer.step()
            for key in ("state_loss", "risk_loss"):
                totals[source][key] += float(losses[key].detach())
            for key in ("position_squared", "position_count", "state_count",
                        "risk_count", "risk_positive"):
                totals[source][key] += losses[key]
        batches += 1
    if not batches:
        raise ValueError("Dataset split contains no batches")

    def metrics(source):
        total = totals[source]
        if not total["position_count"]:
            raise ValueError(f"No valid {source} state targets in this split")
        return {
            "state_mse": total["state_loss"] / batches,
            "position_rmse": (total["position_squared"] / total["position_count"]) ** 0.5,
            "state_steps": int(total["state_count"]),
        }

    result = metrics(latent_source)
    selected = totals[latent_source]
    result.update({
        "risk_bce": selected["risk_loss"] / batches if train_risk else None,
        "risk_labels": int(selected["risk_count"]),
        "risk_positives": int(selected["risk_positive"]),
        "batches": batches,
    })
    if evaluate_both:
        result["observed"] = metrics("observed")
        result["rollout"] = metrics("rollout")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="Dataset used by this checkpoint")
    parser.add_argument("--dataset_type", choices=("hf", "npz"), default="hf",
                        help="Current Parquet frames/windows datasets use hf")
    parser.add_argument("--schema_config", help="Optional Parquet column/sensor mapping YAML")
    parser.add_argument("--norm_stats_path", type=Path,
                        help="Default: norm_stats.pt beside checkpoint, if present")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--validation_fraction", type=float, default=0.2,
                        help="Recording holdout if Parquet has no validation split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heads", choices=("state", "both"), default="state")
    parser.add_argument("--latent_source", choices=("observed", "rollout"), default="rollout",
                        help="Latents used to train the head; validation measures both")
    parser.add_argument("--horizon", type=int, help="Default: horizon saved in checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--risk_weight", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_checkpoint", type=Path,
                        help="Default: latest_with_<latent_source>_heads.pt beside checkpoint")
    cli = parser.parse_args()
    device = torch.device(cli.device)
    model, checkpoint = load_model(cli.checkpoint, device)
    norm_path = cli.norm_stats_path or cli.checkpoint.with_name("norm_stats.pt")
    if cli.norm_stats_path is not None and not norm_path.is_file():
        parser.error(f"normalization statistics not found: {norm_path}")
    norm_stats = NormStats.load(norm_path) if norm_path.is_file() else None
    print(f"Normalization statistics: {norm_path if norm_stats is not None else 'none'}", flush=True)
    trained_horizon = int(
        checkpoint.get("training_contract", {}).get(
            "trained_horizon", checkpoint["args"].get("horizon", 8)
        )
    )
    horizon = cli.horizon or trained_horizon
    if horizon < 1 or cli.epochs < 1 or cli.batch_size < 1:
        parser.error("horizon, epochs, and batch_size must be positive")
    if horizon > trained_horizon:
        parser.error(
            f"head horizon {horizon} exceeds dynamics training horizon {trained_horizon}"
        )
    image_size = int(checkpoint["args"].get("img_size", 224))
    train_risk = cli.heads == "both"
    if train_risk and cli.risk_weight <= 0:
        parser.error("risk_weight must be positive when training risk_head")
    if not 0 < cli.validation_fraction < 1:
        parser.error("validation_fraction must be between 0 and 1")
    datasets = open_splits(cli, image_size, horizon + 1)
    expected = checkpoint["model_config"]
    for split, dataset in zip(
        (cli.train_split, cli.val_split), datasets, strict=True
    ):
        dims = (dataset.D_odom, dataset.D_imu, dataset.D_act)
        required = (expected["odom_dim"], expected["imu_dim"], expected["act_dim"])
        if dims != required:
            raise ValueError(f"{split} dataset dimensions {dims} differ from checkpoint {required}")
        if not len(dataset):
            raise ValueError(f"{split} dataset has no windows")
        expected_action = checkpoint.get("training_contract", {}).get("action", {})
        actual_action = getattr(dataset, "manifest", {}).get(
            "sensor_semantics", {}
        ).get("action", {})
        for key in ("source", "semantic", "frame", "components"):
            if (
                expected_action.get(key)
                and actual_action.get(key) != expected_action[key]
            ):
                raise ValueError(
                    f"{split} dataset action {key}={actual_action.get(key)!r} differs "
                    f"from checkpoint {expected_action[key]!r}"
                )
    rebase_positions = is_parquet_dataset(cli.dataset)
    loaders = [DataLoader(
        WindowPrefix(dataset, horizon + 1, norm_stats), batch_size=cli.batch_size,
        shuffle=(index == 0), num_workers=cli.num_workers,
        collate_fn=collate_for_dataset(dataset),
    ) for index, dataset in enumerate(datasets)]
    for parameter in model.state_head.parameters():
        parameter.requires_grad_(True)
    if train_risk:
        for parameter in model.risk_head.parameters():
            parameter.requires_grad_(True)
    params = list(model.state_head.parameters())
    if train_risk:
        params += list(model.risk_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cli.lr)
    output = cli.out_checkpoint or cli.checkpoint.with_name(
        f"latest_with_{cli.latent_source}_heads.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    if norm_stats is not None and output.parent.resolve() != norm_path.parent.resolve():
        shutil.copy2(norm_path, output.with_name("norm_stats.pt"))
    best_score = float("inf")
    history = []
    for epoch in range(cli.epochs):
        train = run_epoch(model, loaders[0], optimizer, horizon, device,
                          cli.latent_source, train_risk, cli.risk_weight,
                          cli.max_steps_per_epoch)
        val = run_epoch(model, loaders[1], None, horizon, device,
                        cli.latent_source, train_risk, cli.risk_weight,
                        cli.max_steps_per_epoch, evaluate_both=True)
        if train_risk and (train["risk_labels"] == 0
                           or train["risk_positives"] in {0, train["risk_labels"]}):
            raise ValueError(
                "Training risk_head requires valid collision and non-collision labels"
            )
        if train_risk and val["risk_labels"] == 0:
            raise ValueError("Validation split needs valid collision labels")
        score = val["state_mse"] + (cli.risk_weight * val["risk_bce"] if train_risk else 0.0)
        history.append({"epoch": epoch + 1, "train": train, "val": val})
        print(f"epoch={epoch + 1} train={train} val={val}", flush=True)
        if score < best_score:
            best_score = score
            result = {key: value for key, value in checkpoint.items() if key != "optimizer"}
            result["model"] = model.state_dict()
            result["head_training"] = {
                "state_head": True, "risk_head": train_risk, "dataset": cli.dataset,
                "latent_source": cli.latent_source, "horizon": horizon,
                "position_origin": "window_start" if rebase_positions else "dataset",
                "best_epoch": epoch + 1, "validation": val,
            }
            torch.save(result, output)
            output.with_suffix(".metrics.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best head-trained checkpoint: {output}")


if __name__ == "__main__":
    main()
