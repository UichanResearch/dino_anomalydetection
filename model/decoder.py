# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from tools.pos_embed import get_2d_sincos_pos_embed
from tools.patchify import unpatchify


class ViTDecoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, patch_size=14, in_chans=1,
                 embed_dim=768, decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, img_size = 224,
                 mem_slot = 100,device = "cuda"):
        super().__init__()
        self.decoder_depth = decoder_depth
        self.device = device
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.patch_embed = PatchEmbed(img_size , patch_size, in_chans, embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        num_patches = self.patch_embed.num_patches

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        self.feature_mask1 = torch.zeros([1,self.patch_embed.num_patches,1]).to(device)
        for i in range(self.patch_embed.num_patches):
            if i % 2 == 0:
                self.feature_mask1[0,i,0] = 1
        self.feature_mask2 = torch.ones_like(self.feature_mask1).to(device) - self.feature_mask1
        self.feature_black = torch.zeros([1,self.patch_embed.num_patches+1,1]).to(device)
        self.feature_black[0,-1,0] = 1
        self.feature_loss = torch.nn.MSELoss(reduction='mean').to(device)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_decoder(self, x1, x2):
        # embed tokens
        x1 = self.decoder_embed(x1)
        x2 = self.decoder_embed(x2)

        # add pos embed
        x1 = x1 + self.decoder_pos_embed
        x2 = x2 + self.decoder_pos_embed

        total_feature_loss = 0.0
        # apply Transformer blocks
        for i in range(self.decoder_depth):
            x1 = self.decoder_blocks[i](x1)
            x2 = self.decoder_blocks[i](x2)
            cls_index = x1.shape[1]
            total_feature_loss += self.feature_loss(x1,x2)

        x1 = self.decoder_norm(x1)
        x2 = self.decoder_norm(x2)

        # predictor projection
        x1 = self.decoder_pred(x1)
        x2 = self.decoder_pred(x2)

        # remove cls token
        x1 = x1[:, 1:, :]
        x2 = x2[:, 1:, :]

        return x1,x2,total_feature_loss

    def forward(self, x1,x2):
        x1,x2,loss = self.forward_decoder(x1,x2)  # [N, L, p*p*3]
        x1 = unpatchify(x1) # B P*P 14 14
        x2 = unpatchify(x2)
        return x1,x2,loss

if __name__ == "__main__":
    model = ViTDecoder(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))