from math import isqrt

import torch
import torch.distributed as dist
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .base import ModelBundle
from .registry import register

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch

# Adapted from https://github.com/gaoyuezhou/dino_wm/blob/main/models/vqvae.py
class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        distances = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-distances).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(embed_onehot_sum)
                dist.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for _ in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for _ in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        emb_dim=64,
        n_embed=512,
        decay=0.99,
        quantize=True,
        image_size=None,
    ):
        super().__init__()

        self.image_size = image_size
        self.quantize = quantize
        self.quantize_b = Quantize(emb_dim, n_embed)

        if not quantize:
            for param in self.quantize_b.parameters():
                param.requires_grad = False

        self.upsample_b = Decoder(
            emb_dim, emb_dim, channel, n_res_block, n_res_channel, stride=4
        )
        self.dec = Decoder(
            emb_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.info = (
            f"in_channel: {in_channel}, channel: {channel}, "
            f"n_res_block: {n_res_block}, n_res_channel: {n_res_channel}, "
            f"emb_dim: {emb_dim}, n_embed: {n_embed}, decay: {decay}"
        )

    def forward(self, input):
        '''
            input: (b, t, num_patches, emb_dim)
        '''
        num_patches = input.shape[2]
        num_side_patches = isqrt(num_patches)
        if num_side_patches ** 2 != num_patches:
            raise ValueError("VQVAE requires a square grid of patch tokens")
        input = rearrange(
            input, "b t (h w) e -> (b t) h w e",
            h=num_side_patches, w=num_side_patches,
        )

        if self.quantize:
            quant_b, diff_b, _ = self.quantize_b(input)
        else:
            quant_b, diff_b = input, torch.zeros(1).to(input.device)

        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        dec = self.decode(quant_b)
        if self.image_size is not None and dec.shape[-2:] != (
            self.image_size, self.image_size
        ):
            dec = F.interpolate(
                dec, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        return dec, diff_b # diff is 0 if no quantization

    def decode(self, quant_b):
        upsample_b = self.upsample_b(quant_b) 
        dec = self.dec(upsample_b) # quant: (128, 64, 64)
        return dec

    def decode_code(self, code_b): # not used (only used in sample.py in original repo)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        dec = self.decode(quant_b)
        if self.image_size is not None and dec.shape[-2:] != (
            self.image_size, self.image_size
        ):
            dec = F.interpolate(
                dec, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        return dec


class TransposedConvDecoder(nn.Module):
    """Decode a square patch-token grid into RGB images."""

    def __init__(self, observation_shape, emb_dim, depth=64, kernel_size=5, stride=3):
        super().__init__()
        if len(observation_shape) != 3 or min(observation_shape) <= 0:
            raise ValueError("observation_shape must be (channels, height, width)")
        if emb_dim <= 0 or depth < 2 or kernel_size <= 0 or stride <= 0:
            raise ValueError(
                "decoder dimensions, kernel_size, and stride must be positive"
            )
        self.observation_shape = tuple(observation_shape)
        padding = kernel_size // 2
        output_padding = stride - 1 if kernel_size % 2 else stride - 2
        if output_padding < 0:
            output_padding = 0
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                emb_dim, depth, kernel_size, stride, padding, output_padding
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                depth, depth // 2, kernel_size, stride, padding, output_padding
            ),
            nn.GELU(),
            nn.Conv2d(depth // 2, observation_shape[0], 3, padding=1),
        )

    def forward(self, tokens):
        if tokens.ndim != 4:
            raise ValueError(
                "expected tokens with shape (batch, time, patches, embedding)"
            )
        batch, time, patches, embedding = tokens.shape
        side = isqrt(patches)
        if side ** 2 != patches:
            raise ValueError(
                "TransposedConvDecoder requires a square grid of patch tokens"
            )
        grid = tokens.permute(0, 1, 3, 2).reshape(
            batch * time, embedding, side, side
        )
        image = self.layers(grid)
        size = self.observation_shape[1:]
        if image.shape[-2:] != size:
            image = F.interpolate(
                image, size=size, mode="bilinear", align_corners=False
            )
        return image, image.new_zeros(())


@register("decoder")
def build_vqvae(**kwargs) -> ModelBundle:
    model = VQVAE(**kwargs)
    return ModelBundle(
        model=model, preprocess=lambda x: x, feature_dim=None, modality="image"
    )