"""Tests for Stage 3 window-level diversity selection."""

import os
import json

import numpy as np
import pandas as pd
import pytest

from scripts.world_model.data_pipeline.stage3_select import (
    normalize_descriptors,
    farthest_point_sampling,
    extract_windows,
    run_stage3,
    VISUAL_DIM,
    DEFAULT_WINDOW_SIZE,
)


class TestNormalizeDescriptors:
    """Test per-component z-score normalization."""

    def test_output_shape(self):
        desc = np.random.randn(100, 390).astype(np.float32)
        normed = normalize_descriptors(desc)
        assert normed.shape == (100, 390)

    def test_independent_normalization(self):
        n = 200
        desc = np.zeros((n, 390), dtype=np.float32)
        desc[:, :VISUAL_DIM] = np.random.randn(n, VISUAL_DIM) * 10 + 5
        desc[:, VISUAL_DIM:] = np.random.randn(n, 6) * 0.01

        normed = normalize_descriptors(desc)
        cls_part = normed[:, :VISUAL_DIM]
        odom_part = normed[:, VISUAL_DIM:]
        assert abs(cls_part.mean()) < 0.2
        assert abs(odom_part.mean()) < 0.2
        assert abs(cls_part.std() - 1.0) < 0.2
        assert abs(odom_part.std() - 1.0) < 0.2


class TestFPS:
    """Test farthest-point sampling."""

    def test_selects_target_count(self):
        desc = np.random.randn(100, 390).astype(np.float32)
        selected = farthest_point_sampling(desc, target_count=20)
        assert len(selected) == 20

    def test_no_duplicates(self):
        desc = np.random.randn(50, 390).astype(np.float32)
        selected = farthest_point_sampling(desc, target_count=25)
        assert len(set(selected)) == len(selected)

    def test_selects_all_when_target_exceeds(self):
        desc = np.random.randn(10, 390).astype(np.float32)
        selected = farthest_point_sampling(desc, target_count=100)
        assert len(selected) == 10


class TestExtractWindows:
    """Test window extraction from transitions."""

    def test_basic_extraction(self):
        """All-valid transitions should produce windows."""
        n_frames = 32
        cls_tokens = np.random.randn(n_frames, 384).astype(np.float32)
        odom_deltas = np.random.randn(n_frames - 1, 6).astype(np.float64)
        valid_mask = np.ones(n_frames - 1, dtype=bool)
        frame_ids = np.arange(n_frames, dtype=np.int64)
        timestamps = np.arange(n_frames, dtype=np.int64) * 33_000_000

        descs, meta = extract_windows(
            cls_tokens, odom_deltas, valid_mask,
            frame_ids, timestamps, "test_bag",
            window_size=16, window_stride=16,
        )
        # 32 frames = 31 valid transitions -> 2 non-overlapping windows of 16
        assert descs.shape == (2, 390)
        assert len(meta) == 2
        assert meta[0]["n_frames"] == 16
        assert meta[0]["bag_name"] == "test_bag"

    def test_gap_splits_windows(self):
        """A gap in valid_mask should prevent windows spanning it."""
        n_frames = 40
        cls_tokens = np.random.randn(n_frames, 384).astype(np.float32)
        odom_deltas = np.random.randn(n_frames - 1, 6).astype(np.float64)
        valid_mask = np.ones(n_frames - 1, dtype=bool)
        # Put a gap at transition 19 (between frames 19 and 20)
        valid_mask[19] = False
        frame_ids = np.arange(n_frames, dtype=np.int64)
        timestamps = np.arange(n_frames, dtype=np.int64) * 33_000_000

        descs, meta = extract_windows(
            cls_tokens, odom_deltas, valid_mask,
            frame_ids, timestamps, "test_bag",
            window_size=16, window_stride=16,
        )
        # Run 1: transitions 0-18 = 19 valid -> 1 window of 16
        # Run 2: transitions 20-38 = 19 valid -> 1 window of 16
        assert descs.shape[0] == 2
        # Verify windows don't span the gap
        for m in meta:
            fids = m["frame_ids"]
            assert 19 not in fids or 20 not in fids  # can't have both sides of gap

    def test_too_short_segment_no_windows(self):
        """Segment shorter than window_size produces no windows."""
        n_frames = 10
        cls_tokens = np.random.randn(n_frames, 384).astype(np.float32)
        odom_deltas = np.random.randn(n_frames - 1, 6).astype(np.float64)
        valid_mask = np.ones(n_frames - 1, dtype=bool)
        frame_ids = np.arange(n_frames, dtype=np.int64)
        timestamps = np.arange(n_frames, dtype=np.int64) * 33_000_000

        descs, meta = extract_windows(
            cls_tokens, odom_deltas, valid_mask,
            frame_ids, timestamps, "test_bag",
            window_size=16,
        )
        assert descs.shape[0] == 0
        assert len(meta) == 0

    def test_descriptor_is_mean(self):
        """Window descriptor should be mean CLS + mean odom."""
        n_frames = 16
        cls_tokens = np.ones((n_frames, 384), dtype=np.float32) * 2.0
        odom_deltas = np.ones((n_frames - 1, 6), dtype=np.float64) * 0.5
        valid_mask = np.ones(n_frames - 1, dtype=bool)
        frame_ids = np.arange(n_frames, dtype=np.int64)
        timestamps = np.arange(n_frames, dtype=np.int64) * 33_000_000

        descs, meta = extract_windows(
            cls_tokens, odom_deltas, valid_mask,
            frame_ids, timestamps, "test_bag",
            window_size=16,
        )
        assert descs.shape == (1, 390)
        np.testing.assert_allclose(descs[0, :384], 2.0, atol=1e-6)
        np.testing.assert_allclose(descs[0, 384:], 0.5, atol=1e-6)

    def test_overlapping_windows(self):
        """Stride < window_size should produce overlapping windows."""
        n_frames = 24
        cls_tokens = np.random.randn(n_frames, 384).astype(np.float32)
        odom_deltas = np.random.randn(n_frames - 1, 6).astype(np.float64)
        valid_mask = np.ones(n_frames - 1, dtype=bool)
        frame_ids = np.arange(n_frames, dtype=np.int64)
        timestamps = np.arange(n_frames, dtype=np.int64) * 33_000_000

        descs, meta = extract_windows(
            cls_tokens, odom_deltas, valid_mask,
            frame_ids, timestamps, "test_bag",
            window_size=16, window_stride=8,
        )
        assert descs.shape[0] == 2


class TestRunStage3:
    """Integration tests for run_stage3."""

    def test_output_files_created(self, tmp_dir):
        _setup_stage3_inputs(tmp_dir, n_frames=64)
        selected_df, stats = run_stage3(
            per_bag_dir=tmp_dir,
            output_dir=os.path.join(tmp_dir, "corpus"),
            target_count=2,
            window_size=16,
        )
        corpus_dir = os.path.join(tmp_dir, "corpus")
        assert os.path.isfile(os.path.join(corpus_dir, "selected_windows.csv"))
        assert os.path.isfile(os.path.join(corpus_dir, "selected_windows_full.json"))
        assert os.path.isfile(os.path.join(corpus_dir, "corpus_stats.json"))
        assert os.path.isdir(os.path.join(corpus_dir, "gap_analysis"))

    def test_selected_count(self, tmp_dir):
        _setup_stage3_inputs(tmp_dir, n_frames=128)
        selected_df, stats = run_stage3(
            per_bag_dir=tmp_dir,
            output_dir=os.path.join(tmp_dir, "corpus"),
            target_count=3,
            window_size=16,
        )
        assert len(selected_df) == 3

    def test_selected_csv_columns(self, tmp_dir):
        _setup_stage3_inputs(tmp_dir, n_frames=64)
        selected_df, _ = run_stage3(
            per_bag_dir=tmp_dir,
            output_dir=os.path.join(tmp_dir, "corpus"),
            target_count=2,
            window_size=16,
        )
        required = ["bag_name", "window_id", "start_frame_id", "end_frame_id",
                     "start_timestamp", "end_timestamp", "n_frames",
                     "odom_dx_mean", "odom_dy_mean", "odom_dz_mean",
                     "odom_ax_mean", "odom_ay_mean", "odom_az_mean"]
        for col in required:
            assert col in selected_df.columns, f"Missing column: {col}"

    def test_stats_has_window_info(self, tmp_dir):
        _setup_stage3_inputs(tmp_dir, n_frames=64)
        _, stats = run_stage3(
            per_bag_dir=tmp_dir,
            output_dir=os.path.join(tmp_dir, "corpus"),
            target_count=2,
            window_size=16,
        )
        assert "window_size" in stats
        assert "total_selected_frames" in stats
        assert stats["window_size"] == 16


def _setup_stage3_inputs(base_dir, n_frames=64, bag_name="test_bag"):
    """Create a transitions_compact.npz for Stage 3 testing."""
    n_transitions = n_frames - 1
    bag_dir = os.path.join(base_dir, bag_name)
    os.makedirs(bag_dir, exist_ok=True)

    cls_tokens = np.random.randn(n_frames, 384).astype(np.float32)
    odom_deltas = np.random.randn(n_transitions, 6).astype(np.float64)
    descriptors = np.random.randn(n_transitions, 390).astype(np.float32)
    frame_ids = np.arange(n_frames, dtype=np.int64)
    timestamps = np.arange(n_frames, dtype=np.int64) * 33_000_000 + 1_000_000_000
    valid_mask = np.ones(n_transitions, dtype=bool)

    np.savez_compressed(
        os.path.join(bag_dir, f"{bag_name}_transitions_compact.npz"),
        descriptors=descriptors,
        odom_deltas=odom_deltas,
        cls_tokens=cls_tokens,
        frame_t_ids=frame_ids[:-1],
        frame_t1_ids=frame_ids[1:],
        timestamps_t=timestamps[:-1],
        timestamps_t1=timestamps[1:],
        valid_mask=valid_mask,
        bag_name=np.array(bag_name),
        color_paths=np.array([f"img_{i}.jpg" for i in range(n_frames)]),
        depth_paths=np.array([f"depth_{i}.png" for i in range(n_frames)]),
    )
