"""
Train an attention mechanism model on a pretrained classifier

Usage:
    $ python -m tame.train --cfg resnet50_SGD.yaml --epoch -1
"""

import warnings
from typing import Any, Dict
from old_train import train
from pytorch_lightning import Trainer

import torch
from utilities.pl_module import TAMELIT
import yaml
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import utilities as utils
import val
from utilities import AverageMeter, metrics


def main(args):
    print("Running parameters:\n")
    print(yaml.dump(args, indent=4))
    cfg = utils.load_config(args["cfg"])
    print(yaml.dump(cfg, indent=4))

    train_loader, val_loader, _ = utils.data_loader(cfg)
    pl_model = TAMELIT(cfg, len(train_loader), 1000)
    trainer = Trainer()
    trainer.fit(pl_model, train_loader, val_loader)
