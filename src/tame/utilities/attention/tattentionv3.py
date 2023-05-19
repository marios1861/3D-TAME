from typing import List

import torch
import torch.nn as nn
import torchvision

from . import generic_atten as ga


class TAttentionV3(ga.AttentionMech):
    def __init__(self, ft_size: List[torch.Size]):
        super().__init__(ft_size)
        # noinspection PyTypeChecker
        ln_dim = ft_size[0][2]
        self.lns_1 = nn.ModuleList(
            [
                nn.LayerNorm(
                    ln_dim,
                    eps=1e-06,
                )
                for _ in ft_size
            ]
        )
        self.mhas = nn.ModuleList(
            [nn.MultiheadAttention(ft[2], 12, batch_first=True) for ft in ft_size]
        )
        self.lns_2 = nn.ModuleList(
            [
                nn.LayerNorm(
                    ln_dim,
                    eps=1e-06,
                )
                for _ in ft_size
            ]
        )

        self.mlps = nn.ModuleList(
            [
                torchvision.ops.MLP(ft[2], [4 * ft[2], ft[2]], activation_layer=nn.GELU)
                for ft in ft_size
            ]
        )

        # special initialization of MLP layers
        for mlp in self.mlps:
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.normal_(m.bias, std=1e-6)

        def reshape_transform(tensor, height=14, width=14):
            result = tensor[:, 1:, :].reshape(
                tensor.size(0), height, width, tensor.size(2)
            )

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result

        self.reshape = reshape_transform
        fuse_channels = sum(ft[2] for ft in ft_size)

        self.cnn_fuser = nn.Conv2d(
            in_channels=fuse_channels,
            out_channels=1000,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, features):
        feature_maps = features
        # layer norm 1
        xs = [op(feature) for op, feature in zip(self.lns_1, feature_maps)]
        # Multihead Attention
        xs = [op(x, x, x, need_weights=False)[0] for op, x in zip(self.mhas, xs)]
        # add (skip connection 1)
        xs = [x + feature_map for x, feature_map in zip(xs, feature_maps)]
        # layer norm 2
        ys = [op(x) for op, x in zip(self.lns_1, xs)]
        # MLP
        ys = [op(y) for op, y in zip(self.mlps, ys)]
        # add (skip connection 2)
        ys = [y + x for y, x in zip(ys, xs)]
        # Reshape
        ys = [self.reshape(y) for y in ys]
        # Concat
        ys = torch.cat(ys, 1)
        # fuse into 1000 channels
        c = self.cnn_fuser(ys)  # batch_size x1xWxH
        # activation
        a = torch.sigmoid(c)

        return a, c
