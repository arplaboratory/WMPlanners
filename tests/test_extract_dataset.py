"""Tests for extract_dataset manifest generation and topic dispatch."""
import json
import os
import pytest


class TestManifest:
    def test_write_manifest(self, tmp_path):
        from scripts.world_model.extract_dataset import write_manifest

        config = {"version": 1, "name": "race16", "topics": {}, "action": {"source": "position_cmd", "dim": 14}}
        bags_info = [
            {"name": "bag1", "path": "bag1/", "frames": 100},
            {"name": "bag2", "path": "bag2/", "frames": 200},
        ]
        write_manifest(str(tmp_path), config, bags_info)

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            m = json.load(f)
        assert m["robot"] == "race16"
        assert len(m["bags"]) == 2
        assert m["bags"][0]["frames"] == 100
        assert m["config"]["action"]["source"] == "position_cmd"

    def test_empty_bags_list(self, tmp_path):
        from scripts.world_model.extract_dataset import write_manifest

        config = {"version": 1, "name": "test"}
        write_manifest(str(tmp_path), config, [])
        with open(tmp_path / "manifest.json") as f:
            m = json.load(f)
        assert m["bags"] == []


class TestTopicDispatch:
    def test_build_topic_handlers(self):
        from scripts.world_model.extract_dataset import build_topic_handlers

        config = {
            "topics": {
                "color": "/robot/color",
                "depth": "/robot/depth",
                "imu": "/robot/imu",
                "odom": "/robot/odom",
                "position_cmd": "/robot/cmd",
            },
            "sensors": {"color_encoding": "compressed", "depth_encoding": "16UC1"},
        }
        handlers = build_topic_handlers(config)
        assert "/robot/color" in handlers
        assert "/robot/imu" in handlers
        assert handlers["/robot/color"]["type"] == "color"
        assert handlers["/robot/imu"]["type"] == "imu"

    def test_optional_topic_excluded(self):
        from scripts.world_model.extract_dataset import build_topic_handlers

        config = {
            "topics": {"color": "/robot/color", "odom": "/robot/odom"},
            "sensors": {"color_encoding": "raw"},
        }
        handlers = build_topic_handlers(config)
        assert len(handlers) == 2
