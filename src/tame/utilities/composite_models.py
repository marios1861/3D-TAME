# checked, should be working correctly
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

from tame.utilities.attention.factory import AttentionMechFactory


class Generic(nn.Module):
    def __init__(
        self,
        name: str,
        mdl: nn.Module,
        feature_layers: List[str],
        attn_version: str,
        noisy_masks: str = "random",
    ):
        """Args:
        mdl (nn.Module): the model which we would like to use for interpretability
        feature_layers (list): the layers, as printed by get_graph_node_names,
            which we would like to get feature maps from
        """
        super(Generic, self).__init__()
        # get model feature extractor
        train_names, eval_names = get_graph_node_names(mdl)

        output = (train_names[-1], eval_names[-1])
        if output[0] != output[1]:
            print("WARNING! THIS MODEL HAS DIFFERENT OUTPUTS FOR TRAIN AND EVAL MODE")

        self.output = output[0]

        self.body = create_feature_extractor(
            mdl, return_nodes=(feature_layers + [self.output])
        )

        # Dry run to get number of channels for the attention mechanism
        inp = torch.randn(2, 3, 224, 224)
        self.body.eval()
        with torch.no_grad():
            out = self.body(inp)
        out.pop(self.output)

        # Required for attention mechanism initialization
        ft_size = [o.shape for o in out.values()]
        # Build Attention mechanism
        if Generic.is_transformer(mdl) and "TAttention" not in attn_version:
            if "vit_b_16" == name:
                ft_size = [torch.Size([2, 768, 14, 14]) for _ in out.values()]
            else:
                raise NotImplementedError(
                    f"TAME not implemented for the transformer {name}."
                )
            self.attn_mech = AttentionMechFactory.create_attention(
                attn_version, ft_size
            )
            self.attn_mech = nn.Sequential(Generic.PreprocessSeq(name), self.attn_mech)
        else:
            self.attn_mech = AttentionMechFactory.create_attention(
                attn_version, ft_size
            )

        # Get loss and forward training method
        self.arr = "1-1"
        arrangement = Arrangement("1-1", self.body, self.output)
        self.train_policy, self.get_loss = (arrangement.train_policy, arrangement.loss)

        self.a: Optional[torch.Tensor] = None
        self.c: Optional[torch.Tensor] = None
        self.noisy_masks = noisy_masks

    @staticmethod
    def is_transformer(mdl: nn.Module) -> bool:
        for module in mdl.modules():
            if isinstance(module, nn.MultiheadAttention):
                return True
        return False

    class PreprocessSeq(nn.Module):
        def __init__(self, name: str):
            super(Generic.PreprocessSeq, self).__init__()
            implementations = {"vit_b_16": self.forward_vit_b_16}
            try:
                self.forward = implementations[name]
            except KeyError:
                raise KeyError(
                    f"Feature preprocessing not implemented for transformer {name}"
                )

        def forward_vit_b_16(self, seq_list: List[torch.Tensor]) -> List[torch.Tensor]:
            # discard class tocken
            seq_list = [seq[:, 1:, :] for seq in seq_list]
            # reshape
            seq_list = [
                seq.reshape(seq.size(0), 14, 14, seq.size(2)) for seq in seq_list
            ]
            # bring channels after batch dimension
            seq_list = [seq.transpose(2, 3).transpose(1, 2) for seq in seq_list]
            return seq_list

    def forward(
        self, x: torch.Tensor, label: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        x_norm = x

        features: Dict[str, torch.Tensor] = self.body(x_norm)
        x_norm = features.pop(self.output)

        # features now only has the feature maps since we popped the output in case we are in eval mode

        # Attention mechanism
        a, c = self.attn_mech(features.values())
        self.a = a
        self.c = c
        # if in training mode we need to do another forward pass with our masked input as input

        if self.training:
            assert label is not None
            logits = self.train_policy(a, label, x)
            self.logits = logits
            return logits
        else:
            return x_norm

    @staticmethod
    def select_max_masks(
        masks: torch.Tensor, logits: torch.Tensor, N: int
    ) -> torch.Tensor:
        """Select the N masks with the max logits"""
        max_indexes = logits.topk(N)[1]
        return masks[max_indexes, :, :]

    def get_c(self, labels: torch.Tensor) -> torch.Tensor:
        assert self.c is not None
        if self.noisy_masks == "random":
            return self.c[:, labels, :, :]
        elif self.noisy_masks == "diagonal":
            batches = self.c.size(0)
            return self.c[:, labels, :, :][
                torch.arange(batches), torch.arange(batches), :, :
            ]

        elif self.noisy_masks == "max":
            batched_select_max_masks = torch.vmap(
                Generic.select_max_masks, in_dims=(0, 0, None)
            )
            return batched_select_max_masks(self.c, self.logits, self.logits.size(0))
        else:
            raise NotImplementedError

    def get_a(self, labels: torch.Tensor) -> torch.Tensor:
        assert self.a is not None
        if self.noisy_masks == "random":
            return self.a[:, labels, :, :]
        elif self.noisy_masks == "diagonal":
            batches = self.a.size(0)
            return self.a[:, labels, :, :][
                torch.arange(batches), torch.arange(batches), :, :
            ]
            batched_select_max_masks = torch.vmap(
                Generic.select_max_masks, in_dims=(0, 0, None)
            )
            return batched_select_max_masks(self.c, self.logits, self.logits.size(0))
        else:
            raise NotImplementedError


class Arrangement(nn.Module):
    r"""The train_policy and get_loss components of Generic"""

    def __init__(
        self, version: str, body: nn.Module, output_name: str, noisy_masks=True
    ):
        super(Arrangement, self).__init__()
        arrangements = {"1-1": (self.train_policy1, self.loss1)}

        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.body = body
        self.output_name = output_name

        if noisy_masks:
            self.ce_coeff = 1.5  # lambda3
            self.area_loss_coeff = 2  # lambda2
            self.smoothness_loss_coeff = 0.01  # lambda1
            self.area_loss_power = 0.3  # lambda4
        else:
            self.ce_coeff = 1.7  # lambda3
            self.area_loss_coeff = 1.5  # lambda2
            self.smoothness_loss_coeff = 0.1  # lambda1
            self.area_loss_power = 0.3  # lambda4

        self.extra_masks = None
        self.train_policy, self.loss = arrangements[version]

    def area_loss(self, masks):
        if self.area_loss_power != 1:
            # add e to prevent nan (derivative of sqrt at 0 is inf)
            masks = (masks + 0.0005) ** self.area_loss_power
        return torch.mean(masks)

    @staticmethod
    def smoothness_loss(masks, power=2, border_penalty=0.3):
        B, _, _, _ = masks.size()
        x_loss = torch.sum(
            (torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :])) ** power
        )
        y_loss = torch.sum(
            (torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1])) ** power
        )
        if border_penalty > 0:
            border = float(border_penalty) * torch.sum(
                masks[:, :, -1, :] ** power
                + masks[:, :, 0, :] ** power
                + masks[:, :, :, -1] ** power
                + masks[:, :, :, 0] ** power
            )
        else:
            border = 0.0
        return (x_loss + y_loss + border) / float(
            power * B
        )  # watch out, normalised by the batch size!

    def loss1(
        self, logits: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
    ) -> List[torch.Tensor]:
        labels = labels.long()
        variation_loss = self.smoothness_loss_coeff * Arrangement.smoothness_loss(masks)
        area_loss = self.area_loss_coeff * self.area_loss(masks)
        cross_entropy = self.ce_coeff * self.loss_cross_entropy(logits, labels)

        loss = cross_entropy + area_loss + variation_loss

        return [loss, cross_entropy, area_loss, variation_loss]

    def train_policy1(
        self, masks: torch.Tensor, labels: torch.Tensor, inp: torch.Tensor
    ) -> torch.Tensor:
        batches = masks.size(0)
        masks = masks[:, labels, :, :][
            torch.arange(batches), torch.arange(batches), :, :
        ]
        masks = F.interpolate(
            masks, size=(224, 224), mode="bilinear", align_corners=False
        )
        x_norm = masks * inp
        return self.body(x_norm)[self.output_name]
