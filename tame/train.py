"""
Train an attention mechanism model on a pretrained classifier

Usage:
    $ python tame/train.py --cfg resnet50_SGD.yaml --epoch -1
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.cuda import amp
from tqdm.auto import tqdm

from . import utilities as utils
from . import val
from .utilities import AverageMeter, metrics


def train(cfg: Dict[str, Any], args: Dict[str, Any]):
    # Dataloaders
    train_loader, val_loader, _ = utils.data_loader(cfg)

    # Model
    model = utils.get_model(cfg)

    # Optimizer
    optimizer = utils.get_optim(cfg, model)

    # Attempt to reload
    if args["epoch"] != -1:
        # load the latest epoch, or the epoch supplied by args
        last_epoch = utils.load_model(
            args["cfg"], cfg, model, optimizer, args.get("epoch")
        )
    else:
        last_epoch = 0

    # Scheduler
    scheduler = utils.get_schedule(cfg, optimizer, len(train_loader), last_epoch)

    # Train
    scaler = amp.GradScaler()
    epochs = cfg["epochs"]
    print(
        f"{'Epoch':>6}{'GPU mem':>8}"
        f"{'train loss: total':>18}{'CE':>6}{'Area':>6}{'Var':>6}{'top 1':>6}{'top 5':>6}"
        f"{'val loss: total':>16}{'CE':>6}{'Area':>6}{'Var':>6}"
        f"{'AD/IC: 100%':>12}{'50%':>12}{'15%':>12}"
    )  # 75 characters
    # Epoch loop
    for epoch in range(last_epoch, epochs):
        top1 = AverageMeter()
        top5 = AverageMeter()
        loss = AverageMeter()
        loss_ce = AverageMeter()
        loss_mean_mask = AverageMeter()  # Mask energy loss
        loss_var_mask = AverageMeter()  # Mask variation loss
        model.train()
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        # Batch loop
        for i, (images, labels) in pbar:
            images, labels = images.cuda(), labels.cuda()

            # forward pass
            with amp.autocast():
                logits = model(images, labels)
                masks = model.get_a(labels.long())
                (
                    loss_val,
                    loss_ce_val,
                    loss_mean_mask_val,
                    loss_var_mask_val,
                ) = model.get_loss(logits, labels, masks)
            # Backward
            scaler.scale(loss_val).backward()  # type: ignore
            # Optimize
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.attn_mech.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Skip 10 first batches to make sure that loss gradient
            # has stabilized enough for step to run step
            if i > 10:
                scheduler.step()

            logits1 = torch.squeeze(logits)
            top1_val, top5_val = metrics.accuracy(logits1, labels.long(), topk=(1, 5))
            top1.update(top1_val[0])
            top5.update(top5_val[0])

            loss.update(loss_val.item())
            loss_mean_mask.update(loss_mean_mask_val.item())
            loss_var_mask.update(loss_var_mask_val.item())
            loss_ce.update(loss_ce_val.item())
            mem = "%.3gG" % (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )  # (GB)
            pbar.desc = (
                f"{f'{epoch + 1}/{epochs}':>6}{mem:>8}"
                f"{loss.avg:>18.2f}{loss_ce.avg:>6.2f}{loss_mean_mask.avg:>6.2f}"
                f"{loss_var_mask.avg:>6.2f}{top1.avg:>6.2f}{top5.avg:>6.2f}" + " " * 58
            )
            remaining = (
                (pbar.total - pbar.n) / pbar.format_dict["rate"]
                if pbar.format_dict["rate"] and pbar.total
                else 0
            )
            total = remaining + (pbar.format_dict["elapsed"] + remaining) * (
                epochs - epoch - 1
            )
            pbar.set_postfix({"ETA": pbar.format_interval(total)})

            # Val
            if i == len(pbar) - 1:
                _ = val.run(model=model.eval(), dataloader=val_loader, pbar=pbar)

        # first epoch: 1, during training it is current_epoch == 0, saved as epoch_1 ...
        # last epoch: 8, during training it is current_epoch ==7, saved as epoch_8
        utils.save_model(cfg["cfg"], cfg, model, optimizer, epoch)


def get_arguments():
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument(
        "--cfg", type=str, default="default.yaml", help="config script name (not path)"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to load, defaults to latest epoch saved. -1 to restart training",
    )
    return parser.parse_args()


def main(args):
    FILE = Path(__file__).resolve()
    ROOT_DIR = FILE.parents[1]
    print("Running parameters:\n")
    print(yaml.dump(vars(args), indent=4))
    cfg = utils.load_config(ROOT_DIR / "configs", args.cfg)
    train(cfg, vars(args))


if __name__ == "__main__":
    cmd_args = get_arguments()
    main(cmd_args)
