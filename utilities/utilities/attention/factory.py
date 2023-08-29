from typing import ClassVar, Dict, List, Type

import torch
import torch.nn as nn

from .generic_atten import AttentionMech
from .old_attention import (
    AttentionV3d2,
    AttentionV3d2dd1,
    AttentionV5d1,
)
from .tame import AttentionTAME
from .tattentionv1 import (
    TAttentionV1,
    TAttentionV1_1,
    TAttentionV1_2,
)
from .tattentionv2 import (
    TAttentionV2,
    TAttentionV2_1,
    TAttentionV2_2,
)
from .tattentionv3 import TAttentionV3


class AMBuilder(object):
    r"""The attention mechanism component of Generic"""
    versions: ClassVar[Dict[str, Type[AttentionMech]]] = {
        "TAME": AttentionTAME,
        "Noskipconnection": AttentionV3d2dd1,
        "NoskipNobatchnorm": AttentionV3d2,
        "Sigmoidinfeaturebranch": AttentionV5d1,
        "TAttentionV1": TAttentionV1,
        "TAttentionV1_1": TAttentionV1_1,
        "TAttentionV1_2": TAttentionV1_2,
        "TAttentionV2": TAttentionV2,
        "TAttentionV2_1": TAttentionV2_1,
        "TAttentionV2_2": TAttentionV2_2,
        "TAttentionV3": TAttentionV3,
    }

    @classmethod
    def register_attention(cls, name: str, new_attention: Type[AttentionMech]):
        cls.versions.update({name: new_attention})

    @classmethod
    def create_attention(
        cls, mdl_name: str, mdl: nn.Module, version: str, ft_size: List[torch.Size]
    ) -> nn.Module:
        """ Build Attention mechanism based on version and model

        Args:
            mdl_name (str): name of mdl
            mdl (nn.Module): model module
            version (str): name of version
            ft_size (List[torch.Size]): size of features

        Raises:
            NotImplementedError: Not implemented for given transformer
            NotImplementedError: Not implemented for given version

        Returns:
            Attention mechanism
        """
        try:
            # Build Attention mechanism
            # Vision Transformer and not TAttention type attention module
            if is_transformer(mdl) and "TAttention" not in version:
                if "vit_b_16" == mdl_name:
                    ft_size = [torch.Size([2, 768, 14, 14]) for _ in ft_size]
                else:
                    raise NotImplementedError(
                        f"TAME not implemented for the transformer {mdl_name}."
                    )
                attn_mech = cls.versions[version](ft_size)
                attn_mech = nn.Sequential(PreprocessSeq(mdl_name), attn_mech)
            # CNN and TAttention type attention module
            elif not is_transformer(mdl) and "TAttention" in version:
                ft_size = [torch.Size([ft[0], ft[2] * ft[3], ft[1]]) for ft in ft_size]
                attn_mech = cls.versions[version](ft_size)
                attn_mech = nn.Sequential(PreprocessSeq("cnn"), attn_mech)
            # CNN and not Tattention type attention module or Vision Transformer and
            # TAttention type attention module
            else:
                attn_mech = cls.versions[version](ft_size)
            return attn_mech
        except KeyError:
            raise NotImplementedError(version)


def is_transformer(mdl: nn.Module) -> bool:
    for module in mdl.modules():
        if isinstance(module, nn.MultiheadAttention):
            return True
    return False


class PreprocessSeq(nn.Module):
    def __init__(self, name: str):
        super(PreprocessSeq, self).__init__()
        implementations = {
            "vit_b_16": self.forward_vit_b_16,
            "cnn": self.forward_cnn,
        }
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
        seq_list = [seq.reshape(seq.size(0), 14, 14, seq.size(2)) for seq in seq_list]
        # bring channels after batch dimension
        seq_list = [seq.transpose(2, 3).transpose(1, 2) for seq in seq_list]
        return seq_list

    def forward_cnn(self, fmap_list: List[torch.Tensor]) -> List[torch.Tensor]:
        # bring channels to the last dimension
        fmap_list = [fmap.transpose(1, 2).transpose(2, 3) for fmap in fmap_list]
        # now we have a list of tensors of shape (batch, H, W, channels)
        # reshape
        seq_list = [
            seq.reshape(seq.size(0), seq.size(1) * seq.size(2), seq.size(3))
            for seq in fmap_list
        ]
        # now we have a list of tensors of shape (batch, H*W, channels), we can
        # use H*W as the sequence length and channels as the hidden dimension
        return seq_list
