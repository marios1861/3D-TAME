"""
Train an attention mechanism model on a pretrained classifier

Usage:
    $ python -m tame.train --cfg resnet50_SGD.yaml --epoch -1
"""

from pytorch_lightning import Trainer
from pytorch_lightning.cli import LightningCLI

from utilities.pl_module import TAMELIT, LightnightDataset
from utilities.restore import get_checkpoint_dir
import yaml

import utilities as utils


def train(args):
    print("Running parameters:\n")
    print(yaml.dump(args, indent=4))
    cfg = utils.load_config(args["cfg"])
    print(yaml.dump(cfg, indent=4))

    train_loader, val_loader, _ = utils.data_loader(cfg)

    pl_model = TAMELIT(cfg, 1000, 224, [0.0, 0.5, 0.85])

    trainer = Trainer(
        accelerator='gpu',
        default_root_dir=get_checkpoint_dir(args["cfg"], cfg, 0).parent.absolute(),
        max_epochs=cfg["epochs"],
    )
    trainer.fit(pl_model, train_loader, val_loader)


def cli_main():
    cli = LightningCLI(TAMELIT, LightnightDataset)  # noqa: F841
