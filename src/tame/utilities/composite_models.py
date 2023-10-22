# checked, should be working correctly
from typing import Dict, List, Optional, Union, Tuple
from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from torchvision import transforms

from tame.utilities.attention.factory import AMBuilder


class Generic(nn.Module):
    normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __init__(
        self,
        name: str,
        mdl: nn.Module,
        feature_layers: Optional[List[str]],
        attention_version: str,
        masking: Literal["random", "diagonal", "max"] = "random",
        train_method: Literal[
            "new", "renormalize", "raw_normalize", "layernorm", "batchnorm"
        ] = "new",
        input_dim: Optional[torch.Size] = None,
        num_classes=1000,
    ):
        """Args:
        mdl (nn.Module): the model which we would like to use for interpretability
        feature_layers (list): the layers, as printed by get_graph_node_names,
            which we would like to get feature maps from
        """
        super().__init__()
        # get model feature extractor
        train_names, eval_names = get_graph_node_names(mdl)
        if feature_layers == [] or feature_layers is None:
            print(train_names)
            quit()

        output = (train_names[-1], eval_names[-1])
        if output[0] != output[1]:
            print("WARNING! THIS MODEL HAS DIFFERENT OUTPUTS FOR TRAIN AND EVAL MODE")

        self.output = output[0]

        self.body = create_feature_extractor(
            mdl, return_nodes=(feature_layers + [self.output])
        )

        # Dry run to get number of channels for the attention mechanism
        if input_dim:
            inp = torch.randn(input_dim)
        else:
            inp = torch.randn(2, 3, 224, 224)
        self.body.eval()
        with torch.no_grad():
            out = self.body(inp)
        out.pop(self.output)

        # Required for attention mechanism initialization
        ft_size = [o.shape for o in out.values()]

        # Build AM
        if num_classes != 1000:
            self.attn_mech = AMBuilder.create_attention(
                name, mdl, attention_version, ft_size, num_classes=num_classes
            )
        else:
            self.attn_mech = AMBuilder.create_attention(
                name, mdl, attention_version, ft_size
            )
        # Get loss and forward training method
        self.train_method: Literal[
            "new", "renormalize", "raw_normalize", "layernorm", "batchnorm"
        ] = train_method
        self.arrangement = Arrangement(self.train_method, self.body, self.output)
        self.train_policy, self.get_loss = (
            self.arrangement.train_policy,
            self.arrangement.loss,
        )

        self.a: Optional[torch.Tensor] = None
        self.c: Optional[torch.Tensor] = None
        self.masking: Literal["random", "diagonal", "max"] = masking

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.train_method == "raw_normalize":
            x_norm = Generic.normalization(x)
        else:
            x_norm = x

        features: Dict[str, torch.Tensor] = self.body(x_norm)
        old_logits = features.pop(self.output)
        label = old_logits.argmax(dim=1)

        # features now only has the feature maps since we popped the output in case we are in eval mode

        # Attention mechanism
        a, c = self.attn_mech(features.values())
        self.a = a
        self.c = c
        # if in training mode we need to do another forward pass with our masked input as input

        if self.training:
            logits = self.train_policy(a, label, x)
            self.logits = logits
            return logits, label
        else:
            self.logits = old_logits
            return old_logits

    @staticmethod
    def select_max_masks(
        masks: torch.Tensor, logits: torch.Tensor, N: int
    ) -> torch.Tensor:
        """Select the N masks with the max logits"""
        if logits.size(0) < N:
            max_indexes = logits.topk(logits.size(0))[1]
        else:
            max_indexes = logits.topk(N)[1]
        return masks[max_indexes, :, :]

    def get_c(self, labels: torch.Tensor) -> torch.Tensor:
        assert self.c is not None
        if self.masking == "random":
            return self.c[:, labels, :, :]
        elif self.masking == "diagonal":
            batches = self.c.size(0)
            return self.c[torch.arange(batches), labels, :, :].unsqueeze(1)

        elif self.masking == "max":
            batched_select_max_masks = torch.vmap(
                Generic.select_max_masks, in_dims=(0, 0, None)
            )
            return batched_select_max_masks(self.c, self.logits, self.logits.size(0))
        else:
            raise NotImplementedError

    def get_a(self, labels: torch.Tensor) -> torch.Tensor:
        assert self.a is not None
        if self.masking == "random":
            return self.a[:, labels, :, :]
        elif self.masking == "diagonal":
            batches = self.a.size(0)
            return self.a[torch.arange(batches), labels, :, :].unsqueeze(1)
        elif self.masking == "max":
            batched_select_max_masks = torch.vmap(
                Generic.select_max_masks, in_dims=(0, 0, None)
            )
            return batched_select_max_masks(self.a, self.logits, self.logits.size(0))
        else:
            raise NotImplementedError


class Arrangement(nn.Module):
    r"""The train_policy and get_loss components of Generic"""

    def __init__(self, version: str, body: nn.Module, output_name: str):
        super(Arrangement, self).__init__()
        arrangements = {
            "new": (self.new_train_policy, self.classic_loss),
            "renormalize": (self.old_train_policy, self.classic_loss),
            "raw_normalize": (self.legacy_train_policy, self.classic_loss),
            "layernorm": (self.ln_train_policy, self.classic_loss),
            "batchnorm": (self.bn_train_policy, self.classic_loss),
        }

        if version == "layernorm":
            self.norm = nn.LayerNorm([3, 224, 224])
        elif version == "batchnorm":
            self.norm = nn.BatchNorm2d(3)

        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.body = body
        self.output_name = output_name

        self.ce_coeff = 1.7  # lambda3
        self.area_loss_coeff = 1.5  # lambda2
        self.smoothness_loss_coeff = 0.1  # lambda1
        self.area_loss_power = 0.3  # lambda4

        self.train_policy, self.loss = arrangements[version]

    def area_loss(self, masks):
        if self.area_loss_power != 1:
            # add e to prevent nan (derivative of sqrt at 0 is inf)
            masks = (masks + 0.0005) ** self.area_loss_power
        return torch.mean(masks)

    @staticmethod
    def smoothness_loss(masks, power=2, border_penalty=0.3):
        if masks.dim() == 4:
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
        else:
            B, _, _, _, _ = masks.size()
            x_loss = torch.sum(
                (torch.abs(masks[:, :, :, 1:, :] - masks[:, :, :, :-1, :])) ** power
            )
            y_loss = torch.sum(
                (torch.abs(masks[:, :, :, :, 1:] - masks[:, :, :, :, :-1])) ** power
            )
            z_loss = torch.sum(
                (torch.abs(masks[:, :, 1:, :, :] - masks[:, :, :-1, :, :])) ** power
            )

            if border_penalty > 0:
                border = float(border_penalty) * torch.sum(
                    (masks[:, :, :, -1, :] ** power).sum()
                    + (masks[:, :, :, 0, :] ** power).sum()
                    + (masks[:, :, :, :, -1] ** power).sum()
                    + (masks[:, :, :, :, 0] ** power).sum()
                    + (masks[:, :, -1, :, :] ** power).sum()
                    + (masks[:, :, 0, :, :] ** power).sum()
                )
            else:
                border = 0.0
            return (x_loss + y_loss + z_loss + border) / float(
                power * B
            )  # watch out, normalised by the batch size!

    def classic_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
    ) -> List[torch.Tensor]:
        labels = labels.long()
        variation_loss = self.smoothness_loss_coeff * Arrangement.smoothness_loss(masks)
        area_loss = self.area_loss_coeff * self.area_loss(masks)
        cross_entropy = self.ce_coeff * self.loss_cross_entropy(logits, labels)

        loss = cross_entropy + area_loss + variation_loss

        return [loss, cross_entropy, area_loss, variation_loss]

    def bn_train_policy(
        self, masks: torch.Tensor, labels: torch.Tensor, inp: torch.Tensor
    ) -> torch.Tensor:
        batches = masks.size(0)
        masks = masks[torch.arange(batches), labels, :, :].unsqueeze(1)
        masks = F.interpolate(
            masks, size=(224, 224), mode="bilinear", align_corners=False
        )
        # normalize the mask
        x_norm = self.norm(masks * inp)
        return self.body(x_norm)[self.output_name]

    def ln_train_policy(
        self, masks: torch.Tensor, labels: torch.Tensor, inp: torch.Tensor
    ) -> torch.Tensor:
        batches = masks.size(0)
        masks = masks[torch.arange(batches), labels, :, :].unsqueeze(1)
        masks = F.interpolate(
            masks, size=(224, 224), mode="bilinear", align_corners=False
        )
        # normalize the mask
        x_norm = self.norm(masks * inp)
        return self.body(x_norm)[self.output_name]

    def new_train_policy(
        self, masks: torch.Tensor, labels: torch.Tensor, inp: torch.Tensor
    ) -> torch.Tensor:
        batches = masks.size(0)
        if masks.dim() == 4:
            masks = masks[torch.arange(batches), labels, :, :].unsqueeze(1)
            masks = F.interpolate(
                masks, size=(224, 224), mode="bilinear", align_corners=False
            )
        else:
            masks = masks[torch.arange(batches), labels, :, :, :].unsqueeze(1)
            masks = F.interpolate(
                masks, size=inp.shape[-3:], mode="trilinear", align_corners=False
            )
        x_norm = masks * inp
        return self.body(x_norm)[self.output_name]

    def old_train_policy(
        self, masks: torch.Tensor, labels: torch.Tensor, inp: torch.Tensor
    ) -> torch.Tensor:
        invTrans = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                ),
            ]
        )
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        batches = masks.size(0)
        masks = masks[torch.arange(batches), labels, :, :].unsqueeze(1)
        masks = F.interpolate(
            masks, size=(224, 224), mode="bilinear", align_corners=False
        )
        x_norm = normalize(masks * invTrans(inp))
        return self.body(x_norm)[self.output_name]

    def legacy_train_policy(
        self, masks: torch.Tensor, labels: torch.Tensor, inp: torch.Tensor
    ) -> torch.Tensor:
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        batches = masks.size(0)
        masks = masks[torch.arange(batches), labels, :, :].unsqueeze(1)
        masks = F.interpolate(
            masks, size=(224, 224), mode="bilinear", align_corners=False
        )
        x_norm = normalize(masks * inp)
        return self.body(x_norm)[self.output_name]
