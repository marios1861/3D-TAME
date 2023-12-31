"""
Train an attention mechanism model on a pretrained classifier

Usage:
    $ python -m tame.train --cfg resnet50_SGD.yaml --epoch -1
"""

import warnings
from typing import Any, Dict

import torch
import yaml
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from . import utilities as utils
from . import val
from .utilities import AverageMeter, metrics


def train(cfg: Dict[str, Any], args: Dict[str, Any]):
    # Dataloaders
    train_loader, val_loader, _ = utils.data_loader(cfg)

    # Model
    model = utils.get_model(cfg)  # included

    # Optimizer
    optimizer = utils.get_optim(cfg, model)  # included

    # Attempt to reload
    if args["epoch"] != -1:
        # load the latest epoch, or the epoch supplied by args
        last_epoch = utils.load_model(args["cfg"], cfg, model, optimizer, args["epoch"])
    else:
        last_epoch = 0

    # Scheduler
    scheduler = utils.get_schedule(cfg, optimizer, len(train_loader), last_epoch)  # included

    # Train

    # Tensorboard logger
    writer = None
    iters = None
    if args["tensorboard"]:
        writer = SummaryWriter(
            log_dir=args["tblogdir"],
            purge_step=len(train_loader) * last_epoch,
        )
        iters = 0
    scaler = amp.GradScaler()
    epochs = cfg["epochs"]
    print(
        f"{'Epoch':>6}{'GPU mem':>8}"
        f"{'train loss: total':>18}{'CE':>6}{'Area':>6}{'Var':>6}{'top 1':>6}{'top 5':>6}"
        f"{'val: AD/IC: 100%':>20}{'ROAD: 10%/20%':>16}"
    )  # 75 characters
    # Epoch loop
    for epoch in range(last_epoch, epochs):
        top1 = AverageMeter()
        top5 = AverageMeter()
        loss = AverageMeter()
        loss_ce = AverageMeter()
        loss_mean_mask = AverageMeter()  # Mask energy loss
        loss_var_mask = AverageMeter()  # Mask variation loss

        # freeze classifier
        model.requires_grad_(False)
        model.attn_mech.requires_grad_()

        model.train()
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        # Batch loop
        for i, (images, labels) in pbar:
            images, labels = images.cuda(), labels.cuda()  # type: ignore
            images: torch.Tensor
            labels: torch.LongTensor

            # forward pass
            with amp.autocast():
                logits = model(images, labels)
                masks = model.get_a(labels)
                (
                    loss_val,
                    loss_ce_val,
                    loss_mean_mask_val,
                    loss_var_mask_val,
                ) = model.get_loss(logits, labels, masks)
            # # gradients that aren't computed are set to None
            # optimizer.zero_grad(set_to_none=True)

            # # backwards pass
            # loss_val.backward()

            # # optimizer step
            # optimizer.step()

            # Backward
            scaler.scale(loss_val).backward()  # type: ignore
            # Optimize
            # scaler.unscale_(optimizer)  # unscale gradients
            # torch.nn.utils.clip_grad_norm_(model.attn_mech.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # lr reduction step
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
                f"{loss():>18.2f}{loss_ce():>6.2f}{loss_mean_mask():>6.2f}"
                f"{loss_var_mask():>6.2f}{top1():>6.2f}{top5():>6.2f}" + " " * 36
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

            if args["tensorboard"]:
                assert writer is not None
                assert iters is not None
                writer.add_scalar("LR", scheduler.get_last_lr()[0], iters)
                writer.add_scalars(
                    "Losses",
                    {
                        "Total Loss": loss_val.item(),
                        "Mean Mask Loss": loss_mean_mask_val.item(),
                        "Var Loss": loss_var_mask_val.item(),
                        "CE Loss": loss_ce_val.item(),
                    },
                    global_step=iters,
                )
                writer.add_scalars(
                    "Accuracy",
                    {"Top 1": top1_val[0],
                     "Top 5": top1_val[0]},
                    global_step=iters
                )
                writer.flush()
                iters += 1

            # Val
            if i == len(pbar) - 1:
                model.noisy_masks = False
                val.run(model=model.eval(), dataloader=val_loader, pbar=pbar)
                model.noisy_masks = True

        # first epoch: 1, during training it is current_epoch == 0, saved as epoch_1 ...
        # last epoch: 8, during training it is current_epoch ==7, saved as epoch_8
        utils.save_model(args["cfg"], cfg, model, optimizer, epoch)


def main(args):
    print("Running parameters:\n")
    print(yaml.dump(args, indent=4))
    cfg = utils.load_config(args["cfg"])

    if args["tensorboard"]:
        args["tblogdir"] = f'./.tb_logs/{args["cfg"]}/'
        # run this in a separate shell to get results on tensorboard.dev
        # f'tensorboard dev upload --logdir {args["tblogdir"]} --name {args["cfg"]} &> /dev/null'
    print(yaml.dump(cfg, indent=4))
    train(cfg, args)
