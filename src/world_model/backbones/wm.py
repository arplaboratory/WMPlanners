# adapted from https://github.com/gaoyuezhou/dino_wm/blob/main/models/visual_world_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange, repeat
from logging import getLogger
from .base import ModelBundle
from .registry import register
from world_model.components.heads import StateHead, RiskHead
logger = getLogger(__name__)

class WM(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        predictor,
        decoder=None,
        proprio_dim=0,
        action_dim=0,
        raw_action_dim=4,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=False,
        # Predictor mode: "patch" (ViT, default) or "frame" (MLP/RSSM)
        predictor_mode="patch",
        # State/risk auxiliary supervision (disabled when state_dim=0)
        state_dim=0,
        lambda_state=1.0,
        lambda_risk=0.2,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.predictor = predictor  # predictor could be None
        self.decoder = decoder
        self.action_decoder = StateHead(action_dim, raw_action_dim)
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat
        self.action_dim = action_dim * num_action_repeat
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used

        # Predictor mode: "patch" (ViT causal attention) or "frame" (MLP/RSSM frame-level dynamics)
        assert predictor_mode in ("patch", "frame"), f"predictor_mode must be 'patch' or 'frame', got {predictor_mode!r}"
        if predictor_mode == "frame":
            assert concat_dim == 1, (
                "predictor_mode='frame' requires concat_dim=1 so that action embeddings have "
                "dimension action_emb_dim (not encoder.emb_dim). Set concat_dim=1 in your config."
            )
        self.predictor_mode = predictor_mode

        # State and risk auxiliary heads (disabled when state_dim=0)
        self.state_dim = state_dim
        self.state_head = StateHead(self.encoder.emb_dim, state_dim) if state_dim > 0 else None
        self.risk_head  = RiskHead(self.encoder.emb_dim)             if state_dim > 0 else None
        self.lambda_state = lambda_state
        self.lambda_risk  = lambda_risk

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")
        print(f"predictor_mode: {self.predictor_mode}")
        print(f"state_dim: {self.state_dim}")

        logger.info(f"Initialized WM with encoder: {encoder}, predictor: {predictor}, proprio_encoder: {proprio_encoder}, action_encoder: {action_encoder}, concat_dim: {concat_dim}, num_action_repeat: {num_action_repeat}, num_proprio_repeat: {num_proprio_repeat}, predictor_mode: {predictor_mode}, state_dim: {state_dim}")

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)


        #Removed Decoder specific code from here. might be a traceback in future.
        self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()

    def encode(self, obs, act):
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size)
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        return z

    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act

    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def _predict_frame_level(self, z):
        """
        Frame-level dynamics path for MLP and RSSM predictors.

        Pools patch tokens to frame vectors, runs dynamics autoregressively,
        then broadcasts predicted frame vectors back to patch token shape.

        input : z (b, T, P, D) — full latent (visual + proprio + action)
        output: z (b, T, P, D) — same shape, with visual tokens predicted
        """
        B, T, P, D = z.shape
        z_obs, z_act = self.separate_emb(z)
        z_frame = z_obs["visual"].mean(dim=2)       # (B, T, emb_dim) — pool over patches

        # z_act: (B, T, action_emb_dim) for concat_dim=1 (clean separation)
        # For concat_dim=0, z_act has dim = encoder.emb_dim (action stored as a patch token)

        preds = []
        h = (
            self.predictor.get_initial_state(B, z.device)
            if hasattr(self.predictor, "get_initial_state")
            else None
        )
        for t in range(T):
            if h is not None:
                z_next, h = self.predictor(z_frame[:, t], z_act[:, t], h)
            else:
                z_next = self.predictor(z_frame[:, t], z_act[:, t])
            preds.append(z_next)    # each: (B, emb_dim)

        z_pred_frame = torch.stack(preds, dim=1)                           # (B, T, emb_dim)
        z_pred_visual = z_pred_frame.unsqueeze(2).expand(B, T, P, -1)     # (B, T, P, emb_dim)

        # Reconstruct full z with predicted visual tokens + original proprio/action
        if self.concat_dim == 0:
            return torch.cat(
                [z_pred_visual, z_obs["proprio"].unsqueeze(2), z_act.unsqueeze(2)], dim=2
            )
        else:  # concat_dim == 1
            proprio_tiled = repeat(
                z_obs["proprio"].unsqueeze(2), "b t 1 a -> b t f a", f=P
            ).repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(
                z_act.unsqueeze(2), "b t 1 a -> b t f a", f=P
            ).repeat(1, 1, 1, self.num_action_repeat)
            return torch.cat([z_pred_visual, proprio_tiled, act_tiled], dim=3)

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        if self.predictor_mode == "frame":
            return self._predict_frame_level(z)

        # Original ViT patch-level path
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_act(self, z):
        """
        input :  z: (b, num_frames, num_patches, emb_dim)
        output: act: (b, num_frames, action_dim)
        """
        _, z_act = self.separate_emb(z)
        return self.action_decoder(z_act)

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"], # Note: no decoder for proprio for now!
        }
        return obs, diff

    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def forward(self, obs, act, target_encoder=None, odom=None, collision=None):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
                target_encoder: optional EMA encoder for I-JEPA-style stable latent targets
                odom: optional (b, num_frames, D_odom) for state prediction auxiliary loss
                collision: optional (b, num_frames, 1) binary labels for risk head BCE loss
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]
        visual_tgt = obs['visual'][:, self.num_pred :, ...]

        if target_encoder is not None:
            # I-JEPA: encode target frames with frozen EMA encoder for stable prediction targets
            b = visual_tgt.shape[0]
            vt_flat = rearrange(visual_tgt, "b t ... -> (b t) ...")
            vt_flat = self.encoder_transform(vt_flat)
            with torch.no_grad():
                tgt_visual_embs = target_encoder(vt_flat)
            tgt_visual_embs = rearrange(tgt_visual_embs, "(b t) p d -> b t p d", b=b)
            z_tgt = z[:, self.num_pred :, :, :].clone()
            if self.concat_dim == 0:
                z_tgt[:, :, :-2, :] = tgt_visual_embs.detach()
            else:  # concat_dim == 1
                z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)] = tgt_visual_embs.detach()
        else:
            z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)

        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if self.decoder is not None:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim],
                    z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim],
                    z_tgt[:, :, :, :-self.action_dim].detach()
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss

            # ── Auxiliary state and risk heads ────────────────────────────────
            if self.state_head is not None and odom is not None:
                z_obs_pred, _ = self.separate_emb(z_pred)
                z_pooled = z_obs_pred["visual"].mean(dim=2)     # (B, num_hist, emb_dim)

                state_pred = self.state_head(z_pooled)          # (B, num_hist, state_dim)
                state_tgt  = odom[:, self.num_pred:, :self.state_dim].detach()
                loss_state = F.mse_loss(state_pred, state_tgt)
                loss_components["state_loss"] = loss_state
                loss = loss + self.lambda_state * loss_state

                if collision is not None:
                    risk_pred = self.risk_head(z_pooled)        # (B, num_hist, 1)
                    risk_tgt  = collision[:, self.num_pred:].detach()
                    loss_risk = F.binary_cross_entropy(risk_pred, risk_tgt)
                    loss_components["risk_loss"] = loss_risk
                    loss = loss + self.lambda_risk * loss_risk
        else:
            visual_pred = None
            z_pred = None

        loss_components["loss"] = loss

        # Reconstruct all frames through decoder (if present)
        visual_reconstructed = None
        if self.decoder is not None:
            z_obs, _ = self.separate_emb(z)
            obs_reconstructed, _ = self.decode_obs(z_obs)
            visual_reconstructed = obs_reconstructed["visual"]

        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z


    def rollout(self, obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:]
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z

def build_wm(
    encoder,
    predictor,
    proprio_encoder,
    action_encoder,
    concat_dim=0,
    num_action_repeat=7,
    num_proprio_repeat=7,
    predictor_mode="patch",
    state_dim=0,
    lambda_state=1.0,
    lambda_risk=0.2,
    **kwargs
) -> ModelBundle:
    model = WM(
        encoder=encoder,
        predictor=predictor,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        concat_dim=concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=num_proprio_repeat,
        predictor_mode=predictor_mode,
        state_dim=state_dim,
        lambda_state=lambda_state,
        lambda_risk=lambda_risk,
        **kwargs
    )
    logger.info(f"Built WM model with encoder: {encoder}, predictor: {predictor}, proprio_encoder: {proprio_encoder}, action_encoder: {action_encoder}, concat_dim: {concat_dim}, num_action_repeat: {num_action_repeat}, num_proprio_repeat: {num_proprio_repeat}, predictor_mode: {predictor_mode}, state_dim: {state_dim}")
    return ModelBundle(model=model)
