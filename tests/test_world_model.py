"""Tests for the world_model package after the modular restructure.

Covers:
- Backbone registry (registration, creation, CNN forward pass)
- WorldModel instantiation with different backbones
- WorldModel encode / rollout
- Backbone freezing
- Components (dynamics, fusion, heads) basic forward shapes
"""
import pytest
import torch

from world_model.backbones import create_model, ModelBundle
from world_model.backbones.registry import _BUILDERS
from world_model.architecture import WorldModel, WMOutput
from world_model.components import LatentDynamics, Fusion, ProprioEncoder, StateHead, RiskHead


# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------


class TestBackboneRegistry:
    """Verify the backbone registry and CNN backend."""

    def test_expected_backends_registered(self):
        assert "cnn" in _BUILDERS
        assert "dinov2" in _BUILDERS
        assert "vjepa2" in _BUILDERS

    def test_create_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown backbone"):
            create_model("nonexistent_model_xyz")

    def test_create_cnn(self):
        bundle = create_model("cnn", in_ch=4, z_dim=64, device="cpu")
        assert isinstance(bundle, ModelBundle)
        assert bundle.feature_dim == 64
        assert bundle.modality == "image"

    def test_cnn_forward_shape(self):
        bundle = create_model("cnn", in_ch=4, z_dim=64, device="cpu")
        x = torch.randn(2, 4, 64, 64)
        with torch.no_grad():
            out = bundle.model(x)
        assert out.shape == (2, 64)


# ---------------------------------------------------------------------------
# WorldModel instantiation
# ---------------------------------------------------------------------------


class TestWorldModelInstantiation:
    """WorldModel can be created with different backbone configs."""

    def test_default_backbone_is_cnn(self):
        m = WorldModel(odom_dim=6, imu_dim=6, act_dim=2)
        from world_model.backbones.cnn import VisionEncoder
        assert isinstance(m.vision, VisionEncoder)

    def test_explicit_cnn_backbone(self):
        m = WorldModel(odom_dim=6, imu_dim=6, act_dim=2, backbone="cnn")
        from world_model.backbones.cnn import VisionEncoder
        assert isinstance(m.vision, VisionEncoder)

    def test_freeze_backbone(self):
        m = WorldModel(odom_dim=6, imu_dim=6, act_dim=2, freeze_backbone=True)
        trainable = sum(p.numel() for p in m.vision.parameters() if p.requires_grad)
        assert trainable == 0, "Expected all backbone params frozen"

    def test_non_frozen_backbone_has_grad(self):
        m = WorldModel(odom_dim=6, imu_dim=6, act_dim=2, freeze_backbone=False)
        trainable = sum(p.numel() for p in m.vision.parameters() if p.requires_grad)
        assert trainable > 0


# ---------------------------------------------------------------------------
# WorldModel forward pass
# ---------------------------------------------------------------------------


class TestWorldModelForward:
    """Encode and rollout produce correct shapes."""

    @pytest.fixture()
    def model(self):
        return WorldModel(
            odom_dim=6, imu_dim=6, act_dim=2,
            z_vis_dim=32, z_prop_dim=32, z_dim=64, state_dim=6,
        )

    def test_encode_shapes(self, model):
        B = 3
        out = model.encode(
            rgb=torch.randn(B, 3, 64, 64),
            depth=torch.randn(B, 1, 64, 64),
            odom=torch.randn(B, 6),
            imu=torch.randn(B, 6),
        )
        assert isinstance(out, WMOutput)
        assert out.z.shape == (B, 64)
        assert out.state.shape == (B, 6)
        assert out.risk.shape == (B, 1)

    def test_rollout_shapes(self, model):
        B, T = 2, 5
        z0 = torch.randn(B, 64)
        actions = torch.randn(B, T, 2)
        z_seq = model.rollout(z0, actions)
        assert z_seq.shape == (B, T + 1, 64)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


class TestComponents:
    """Verify individual world-model components produce correct shapes."""

    def test_latent_dynamics(self):
        dyn = LatentDynamics(z_dim=64, act_dim=2)
        z = torch.randn(4, 64)
        a = torch.randn(4, 2)
        z_next = dyn(z, a)
        assert z_next.shape == (4, 64)

    def test_proprio_encoder(self):
        enc = ProprioEncoder(odom_dim=6, imu_dim=6, z_dim=32)
        z = enc(torch.randn(4, 6), torch.randn(4, 6))
        assert z.shape == (4, 32)

    def test_fusion(self):
        fuse = Fusion(z_vis=32, z_prop=32, z_out=64)
        z = fuse(torch.randn(4, 32), torch.randn(4, 32))
        assert z.shape == (4, 64)

    def test_state_head(self):
        head = StateHead(z_dim=64, state_dim=6)
        out = head(torch.randn(4, 64))
        assert out.shape == (4, 6)

    def test_risk_head(self):
        head = RiskHead(z_dim=64)
        out = head(torch.randn(4, 64))
        assert out.shape == (4, 1)
        assert (out >= 0).all() and (out <= 1).all(), "RiskHead should output probabilities"
