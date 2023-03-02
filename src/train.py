"""
Train an attention mechanism model on a pretrained classifier

Usage:
    $ python -m tame.train --cfg resnet50_SGD.yaml --epoch -1
"""

from pytorch_lightning.cli import LightningCLI

from utilities.pl_module import TAMELIT, LightnightDataset


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.model", "model.model")
        parser.link_arguments("data.crop_size", "model.img_size")
        parser.link_arguments("trainer.max_epochs", "model.epochs")
        parser.link_arguments(
            "data.num_classes", "model.num_classes", apply_on="instantiate"
        )


def cli_main():
    trainer_defaults = {"accelerator": "gpu"}
    cli = MyLightningCLI(  # noqa: F841
        TAMELIT, LightnightDataset, trainer_defaults=trainer_defaults
    )
