import os
import pytest
import yaml

from scripts.world_model.robot_config import (
    load_robot_config,
    apply_overrides,
)

def test_load_by_name():
    cfg = load_robot_config("race16")
    assert cfg["name"] == "race16"
    assert cfg["topics"]["color"] == "/race16/cam1/color/image_raw/compressed"
    assert cfg["action"]["source"] == "position_cmd"
    assert cfg["action"]["dim"] == 14

def test_load_by_path(tmp_path):
    custom = {
        "version": 1, "name": "test_bot",
        "topics": {"color": "/test/color", "odom": "/test/odom"},
        "sensors": {"color_encoding": "raw", "odom_dim": 6},
        "action": {"source": "odom", "dim": 6},
    }
    p = tmp_path / "test_bot.yaml"
    p.write_text(yaml.dump(custom))
    cfg = load_robot_config(str(p))
    assert cfg["name"] == "test_bot"

def test_apply_overrides():
    cfg = {"topics": {"color": "/old"}, "action": {"dim": 14}}
    result = apply_overrides(cfg, {"topics.color": "/new", "action.dim": "6"})
    assert result["topics"]["color"] == "/new"
    assert result["action"]["dim"] == 6  # YAML parsing converts "6" → int 6

def test_load_with_overrides():
    cfg = load_robot_config("race16", overrides={"topics.color": "/override/topic"})
    assert cfg["topics"]["color"] == "/override/topic"
    assert cfg["action"]["source"] == "position_cmd"

def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_robot_config("nonexistent_robot_xyz")
