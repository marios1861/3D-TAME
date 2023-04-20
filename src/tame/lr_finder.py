"""LR finder test
Generate the loss to lr curve of a model given a configuration
Usage:
    $ python -m tame.val --cfg resnet50_new.yaml
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.cuda import amp
from tqdm.auto import tqdm
from .utilities.proj_utilities import get_project_root
import yaml

from . import utilities as utils
from .utilities import AverageMeter


def find_lr(cfg: Dict[str, Any], args: Dict[str, Any]
            ) -> Tuple[List[float], List[float]]:
    beta = args["beta"]

    # Dataloader
    dataloader = utils.data_loader(cfg)[0]

    # Model
    model = utils.get_model(cfg)

    # Optimizer
    optimizer = utils.get_optim(cfg, model)

    model.requires_grad_(False)
    model.attn_mech.requires_grad_()
    model.train()

    num = len(dataloader) - 1
    mult = (args["final"] / args["init"]) ** (1 / num)
    lr = args["init"]
    for group in optimizer.param_groups:
        group["lr"] = lr

    avg_loss = AverageMeter(a=(1 - beta))
    best_loss = 0.0
    loss_list = []
    lrs = []
    print(f"{'GPU mem':>8}{'train loss: current':>20}{'minimum':>8}{'lr':>10}")
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    )
    scaler = amp.GradScaler()
    for idx, (images, labels) in pbar:
        idx += 1
        images, labels = images.cuda(), labels.cuda()

        # forward pass
        with amp.autocast():
            logits = model(images, labels)
            masks = model.get_a(labels)
            losses = model.get_loss(logits, labels, masks)
            loss = losses[0]

        # Backward pass
        scaler.scale(loss).backward()  # type: ignore
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        # Update the lr for the next step
        # Compute the smoothed loss
        avg_loss.update(loss.item())
        smoothed_loss = avg_loss() / (1 - beta**idx)
        # Stop if the loss is exploding
        if idx > 1 and smoothed_loss > 4 * best_loss:
            return lrs, loss_list
        # Record the best loss
        if smoothed_loss < best_loss or idx == 1:
            best_loss = smoothed_loss
        loss_list.append(smoothed_loss)
        mem = "%.3gG" % (
            torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        )
        pbar.desc = f"{mem:>8}{smoothed_loss:>20.2f}{best_loss:>8.2f}{lr:>10.3e}"

        lrs.append(lr)
        lr *= mult
        for group in optimizer.param_groups:
            group["lr"] = lr

    return lrs, loss_list


def save_data(data: Tuple[List[float], List[float]], file_path: Path):
    with open(file_path, mode="x") as file:
        json.dump(data, file)


def main(args: Dict[str, Any]):
    print("Running parameters:\n")
    cfg = utils.load_config(args["cfg"])
    print(yaml.dump(cfg, indent=4))
    data_dir = get_project_root() / "LR"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / Path(args["cfg"]).with_suffix(".json")
    data = find_lr(cfg, args)
    save_data(data, file_path)
