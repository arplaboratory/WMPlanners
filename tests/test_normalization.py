"""Tests for NormStats normalization module."""

import numpy as np
import pytest
import torch

from world_model.data.normalization import NormStats


@pytest.fixture
def random_arrays():
    rng = np.random.default_rng(42)
    N = 500
    odom = rng.standard_normal((N, 13)).astype(np.float32)
    imu = rng.standard_normal((N, 10)).astype(np.float32)
    action = rng.standard_normal((N, 14)).astype(np.float32)
    return odom, imu, action


def test_from_arrays_shapes(random_arrays):
    odom, imu, action = random_arrays
    ns = NormStats.from_arrays(odom, imu, action)
    assert ns.odom_mean.shape == (13,)
    assert ns.odom_std.shape == (13,)
    assert ns.imu_mean.shape == (10,)
    assert ns.imu_std.shape == (10,)
    assert ns.act_mean.shape == (14,)
    assert ns.act_std.shape == (14,)


def test_normalize_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    N = 5000
    odom = rng.standard_normal((N, 13)).astype(np.float32)
    imu = rng.standard_normal((N, 10)).astype(np.float32)
    action = rng.standard_normal((N, 14)).astype(np.float32)
    ns = NormStats.from_arrays(odom, imu, action)

    odom_t = torch.from_numpy(odom)
    normed = ns.normalize_odom(odom_t)

    assert torch.abs(normed.mean(dim=0)).max().item() < 0.05
    assert torch.abs(normed.std(dim=0) - 1.0).max().item() < 0.05


def test_denormalize_round_trip(random_arrays):
    odom, imu, action = random_arrays
    ns = NormStats.from_arrays(odom, imu, action)

    odom_t = torch.from_numpy(odom)
    recovered = ns.denormalize_odom(ns.normalize_odom(odom_t))
    assert torch.allclose(odom_t, recovered, atol=1e-5)

    act_t = torch.from_numpy(action)
    recovered_act = ns.denormalize_action(ns.normalize_action(act_t))
    assert torch.allclose(act_t, recovered_act, atol=1e-5)


def test_constant_channel_no_nan():
    rng = np.random.default_rng(7)
    N = 100
    odom = rng.standard_normal((N, 13)).astype(np.float32)
    odom[:, 3] = 5.0  # constant column
    imu = rng.standard_normal((N, 10)).astype(np.float32)
    action = rng.standard_normal((N, 14)).astype(np.float32)

    ns = NormStats.from_arrays(odom, imu, action)
    normed = ns.normalize_odom(torch.from_numpy(odom))
    assert not torch.isnan(normed).any()


def test_save_load_round_trip(random_arrays, tmp_path):
    odom, imu, action = random_arrays
    ns = NormStats.from_arrays(odom, imu, action)

    path = tmp_path / "norm_stats.pt"
    ns.save(str(path))
    ns2 = NormStats.load(str(path))

    assert torch.allclose(ns.odom_mean, ns2.odom_mean)
    assert torch.allclose(ns.odom_std, ns2.odom_std)
    assert torch.allclose(ns.imu_mean, ns2.imu_mean)
    assert torch.allclose(ns.imu_std, ns2.imu_std)
    assert torch.allclose(ns.act_mean, ns2.act_mean)
    assert torch.allclose(ns.act_std, ns2.act_std)
