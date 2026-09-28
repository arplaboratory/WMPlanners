import json
import os
import numpy as np


def _make_episodes(out_dir, n_episodes=50, ep_len=16, img_size=8, odom_dim=13, act_dim=14):
    """Create synthetic NPZ episodes with varying dynamics."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(n_episodes):
        speed = 0.5 + i * 0.1
        angle = i * 0.2
        np.savez_compressed(
            os.path.join(out_dir, f"ep_{i:05d}.npz"),
            rgb=rng.integers(0, 255, (ep_len, img_size, img_size, 3), dtype=np.uint8),
            depth=rng.random((ep_len, img_size, img_size)).astype(np.float32),
            imu=rng.standard_normal((ep_len, 10)).astype(np.float32),
            odom=np.column_stack([
                np.linspace(0, speed, ep_len),  # pos_x
                np.zeros((ep_len, 2)),  # pos_y, pos_z
                np.zeros((ep_len, 3)),  # orient_xyz
                np.ones((ep_len, 1)),   # orient_w
                np.full((ep_len, 1), speed),  # lin_vel_x
                np.zeros((ep_len, 2)),  # lin_vel_yz
                np.full((ep_len, 1), angle),  # ang_vel_x
                np.zeros((ep_len, 2)),  # ang_vel_yz
            ]).astype(np.float32),
            action=rng.standard_normal((ep_len, act_dim)).astype(np.float32) * (i + 1),
        )
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({"total_episodes": n_episodes, "sensor_dims": {
            "odom": odom_dim, "imu": 10, "action": act_dim,
        }}, f)


class TestDescriptorConstruction:
    def test_action_stats_shape(self, tmp_path):
        from scripts.world_model.select_episodes import compute_descriptors
        _make_episodes(str(tmp_path), n_episodes=10, act_dim=14)
        descs, _ = compute_descriptors(str(tmp_path), encoder="none")
        # action stats (2*14) + odom delta (6) = 34
        assert descs.shape == (10, 34)

    def test_descriptors_vary(self, tmp_path):
        from scripts.world_model.select_episodes import compute_descriptors
        _make_episodes(str(tmp_path), n_episodes=10)
        descs, _ = compute_descriptors(str(tmp_path), encoder="none")
        assert not np.allclose(descs[0], descs[-1])

    def test_spot_profile_shape(self, tmp_path):
        from scripts.world_model.select_episodes import compute_descriptors
        _make_episodes(str(tmp_path), n_episodes=10)
        descs, _ = compute_descriptors(str(tmp_path), encoder="none", profile="spot")
        assert descs.shape[0] == 10
        assert descs.shape[1] >= 10

    def test_rover_profile_shape(self, tmp_path):
        from scripts.world_model.select_episodes import compute_descriptors
        _make_episodes(str(tmp_path), n_episodes=10, act_dim=3)
        descs, _ = compute_descriptors(str(tmp_path), encoder="none", profile="rover")
        assert descs.shape[0] == 10
        assert descs.shape[1] >= 10


class TestFPSSelection:
    def test_selects_target_count(self, tmp_path):
        from scripts.world_model.select_episodes import select_episodes
        _make_episodes(str(tmp_path), n_episodes=50)
        out = str(tmp_path / "selected")
        stats = select_episodes(str(tmp_path), out, target_count=10, encoder="none", dynamics_weight=0.6)
        assert stats["selected"] == 10
        eps = [f for f in os.listdir(out) if f.endswith(".npz")]
        assert len(eps) == 10

    def test_target_exceeds_available(self, tmp_path):
        from scripts.world_model.select_episodes import select_episodes
        _make_episodes(str(tmp_path), n_episodes=5)
        out = str(tmp_path / "selected")
        stats = select_episodes(str(tmp_path), out, target_count=100, encoder="none", dynamics_weight=0.6)
        assert stats["selected"] == 5

    def test_selection_stats_written(self, tmp_path):
        from scripts.world_model.select_episodes import select_episodes
        _make_episodes(str(tmp_path), n_episodes=20)
        out = str(tmp_path / "selected")
        select_episodes(str(tmp_path), out, target_count=5, encoder="none", dynamics_weight=0.6)
        assert os.path.exists(os.path.join(out, "selection_stats.json"))
        assert os.path.exists(os.path.join(out, "metadata.json"))

    def test_spot_profile_selects_target_count(self, tmp_path):
        from scripts.world_model.select_episodes import select_episodes
        _make_episodes(str(tmp_path), n_episodes=12)
        out = str(tmp_path / "selected_spot")
        stats = select_episodes(
            str(tmp_path),
            out,
            target_count=6,
            encoder="none",
            dynamics_weight=0.6,
            profile="spot",
        )
        assert stats["selected"] == 6
        eps = [f for f in os.listdir(out) if f.endswith(".npz")]
        assert len(eps) == 6

    def test_rover_profile_selects_target_count(self, tmp_path):
        from scripts.world_model.select_episodes import select_episodes
        _make_episodes(str(tmp_path), n_episodes=12, act_dim=3)
        out = str(tmp_path / "selected_rover")
        stats = select_episodes(
            str(tmp_path),
            out,
            target_count=6,
            encoder="none",
            dynamics_weight=0.6,
            profile="rover",
        )
        assert stats["selected"] == 6
        eps = [f for f in os.listdir(out) if f.endswith(".npz")]
        assert len(eps) == 6
