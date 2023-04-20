from typing import Any, Dict, List, Optional

from torch import nn, optim
from torch.optim import lr_scheduler as lr
from torchvision import models

from .composite_models import Generic


def get_model(cfg: Dict[str, Any]) -> Generic:
    mdl = model_prep(cfg["model"])
    mdl = Generic(cfg["model"], mdl, cfg["layers"], cfg["version"], cfg["noisy_masks"])
    mdl.cuda()
    return mdl


def pl_get_config(
    model_name: str,
    layers: List[str],
    attention_version: str,
    noisy_masks: bool,
    optimizer: str,
    momentum: float,
    decay: float,
    schedule: str,
    lr: float,
    epochs: int,
) -> Dict:
    cfg = {
        "model": model_name,
        "layers": layers,
        "version": attention_version,
        "noisy_masks": noisy_masks,
        "optimizer": optimizer,
        "momentum": momentum,
        "decay": decay,
        "schedule": schedule,
        "lr": lr,
        "epochs": epochs,
    }
    return cfg


def model_prep(model_name: str) -> nn.Module:
    model = models.__dict__[model_name](weights="IMAGENET1K_V1")
    return model


def get_optim(cfg: Dict[str, Any], model: Generic) -> optim.Optimizer:
    g = [], [], []  # optimizer parameter groups
    # normalization layers, i.e. BatchNorm2d()
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    for v in model.attn_mech.modules():
        for p_name, p in v.named_parameters(recurse=False):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:  # weight (with decay)
                g[0].append(p)
    if cfg["optimizer"] == "Adam":
        # adjust beta1 to momentum
        optimizer = optim.Adam(g[2], lr=1e-7, betas=(cfg["momentum"], 0.999))
    elif cfg["optimizer"] == "AdamW":
        optimizer = optim.AdamW(
            g[2], lr=1e-7, betas=(cfg["momentum"], 0.999), weight_decay=0.0
        )
    elif cfg["optimizer"] == "RMSProp":
        optimizer = optim.RMSprop(g[2], lr=1e-7, momentum=cfg["momentum"])
    elif cfg["optimizer"] == "SGD":
        optimizer = optim.SGD(g[2], lr=1e-7, momentum=cfg["momentum"], nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {cfg["optimizer"]} not implemented.')

    # add g0 with weight_decay
    optimizer.add_param_group({"params": g[0], "weight_decay": cfg["decay"]})
    # add g1 (BatchNorm2d weights)
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    return optimizer


def get_schedule(
    cfg: Dict[str, Any],
    optimizer: optim.Optimizer,
    currect_epoch: int,
    steps_per_epoch: Optional[int] = None,
    total_steps: Optional[int] = None,
) -> lr._LRScheduler:
    if cfg["schedule"] == "NEW":
        schedule = lr.OneCycleLR(  # type: ignore
            optimizer,
            cfg["lr"],
            epochs=cfg["epochs"] if total_steps is None else None,  # type:ignore
            steps_per_epoch=steps_per_epoch,  # type:ignore
            total_steps=total_steps,  # type:ignore
            # this denotes the last iteration, if we are just starting out it should be its default
            # value, -1
            last_epoch=(currect_epoch * steps_per_epoch)  # type:ignore
            if currect_epoch != 0
            else -1,
        )
    elif cfg["schedule"] == "CLASSIC":
        schedule = lr.OneCycleLR(  # type: ignore
            optimizer,
            [cfg["lr"], 2 * cfg["lr"], 2 * cfg["lr"]],
            epochs=cfg["epochs"] if total_steps is None else None,  # type:ignore
            steps_per_epoch=steps_per_epoch,  # type:ignore
            total_steps=total_steps,  # type:ignore
            # this denotes the last iteration, if we are just starting out it should be its default
            # value, -1
            last_epoch=(currect_epoch * steps_per_epoch)  # type:ignore
            if currect_epoch != 0
            else -1,
        )
    else:
        raise NotImplementedError(f'Schedule {cfg["schedule"]} not implemented.')
    return schedule
