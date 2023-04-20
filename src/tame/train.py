"""
Train an attention mechanism model on a pretrained classifier

Usage:
    $ python -m tame.train --cfg resnet50_SGD.yaml --epoch -1
"""

from lightning.pytorch.cli import LightningCLI
import torch

from .utilities.pl_module import TAMELIT, LightnightDataset


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.model_name", "data.model")
        parser.link_arguments("data.crop_size", "model.img_size")
        parser.link_arguments("model.epochs", "trainer.max_epochs")
        parser.link_arguments("data.num_classes", "model.num_classes")


def cli_main():
    torch.set_float32_matmul_precision('high')
    trainer_defaults = {"accelerator": "gpu"}
    cli = MyLightningCLI(  # noqa: F841
        TAMELIT, LightnightDataset, trainer_defaults=trainer_defaults
    )
