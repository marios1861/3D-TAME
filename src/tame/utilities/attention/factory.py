from typing import ClassVar, Dict, List, Type

import torch

from tame.utilities.attention.generic_atten import AttentionMech
from tame.utilities.attention.old_attention import (
    AttentionV3d2,
    AttentionV3d2dd1,
    AttentionV5d1,
)
from tame.utilities.attention.tame import AttentionTAME
from tame.utilities.attention.tattentionv1 import (
    TAttentionV1,
    TAttentionV1_1,
    TAttentionV1_2,
)
from tame.utilities.attention.tattentionv2 import (
    TAttentionV2,
    TAttentionV2_1,
    TAttentionV2_2,
)
from tame.utilities.attention.tattentionv3 import TAttentionV3


class AttentionMechFactory(object):
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
    def create_attention(cls, version: str, ft_size: List[torch.Size]) -> AttentionMech:
        try:
            return cls.versions[version](ft_size)
        except KeyError:
            raise NotImplementedError(version)
