"""Evaluation Script
Evaluate an attention mechanism model on a pretrained classifier

Usage:
    $ python -m tame.val --cfg resnet50_new.yaml --test --with-val
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
from torchvision import transforms
from tqdm import tqdm

import utilities as utils
from masked_print import save_heatmap
from utilities import metrics


@torch.no_grad()
def run(
    cfg: Optional[Dict[str, Any]] = None,
    args: Optional[Dict[str, Any]] = None,
    percent_list: List[float] = [0.0, 0.5, 0.85],
    model: Optional[utils.Generic] = None,
    dataloader: Optional[torch.utils.data.DataLoader] = None,  # type: ignore
    pbar: Optional[tqdm] = None,
    example_gen: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    if cfg is not None:
        assert args is not None
        # Dataloader
        if args["with_val"]:
            dataloader = utils.data_loader(cfg)[1]
        else:
            dataloader = utils.data_loader(cfg)[2]
        # Model
        cfg["noisy_masks"] = False
        model = utils.get_model(cfg)
        # Load model
        utils.load_model(args["cfg"], cfg, model, epoch=args.get("epoch"))
        # model.half()
        model.eval()
    assert dataloader is not None
    assert model is not None

    # dig through dataloader to find input image size
    transform: transforms.Compose = dataloader.dataset.transform  # type: ignore
    img_size = next(
        tsfm for tsfm in transform.transforms if isinstance(tsfm, transforms.CenterCrop)
    ).size[0]

    n = len(dataloader)
    action = (
        ("validating" if (False if args is None else args["with_val"]) else "testing")
        if args
        else "validating"
    )
    desc = f"{pbar.desc[:-36]}{action:>36}" if pbar else f"{action}"
    bar = tqdm(
        enumerate(dataloader),
        desc,
        n,
        False,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        position=0,
    )
    if example_gen is not None:
        image, _ = dataloader.dataset[example_gen]
        image = image.unsqueeze(0)
        logits: torch.Tensor = model.cpu()(image)
        logits = logits.softmax(dim=1)
        chosen_logits, model_truth = logits.max(dim=1)
        mask = model.get_c(model_truth)
        mask = metrics.normalizeMinMax(mask)
        mask = torch.nn.functional.interpolate(
            mask,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        image = image.squeeze()
        save_heatmap(
            mask,
            image,
            Path("evaluation data")
            / "examples"
            / f"val_{example_gen}_{datetime.now().time().strftime('%H-%M-%S')}.jpg",
        )
        quit()

    metric_AD_IC = metrics.AD_IC(model, img_size, percent_list=percent_list)
    metric_ROAD = metrics.ROAD(model, ROADMostRelevantFirst)
    for _, (images, _) in bar:
        # with torch.cuda.amp.autocast():
        images = images.cuda()  # type: ignore
        images: torch.Tensor
        logits: torch.Tensor = model(images)

        logits = logits.softmax(dim=1)
        chosen_logits, model_truth = logits.max(dim=1)
        masks = model.get_c(model_truth)
        masks = metrics.normalizeMinMax(masks)
        metric_AD_IC(images, chosen_logits, model_truth, masks)
        masks = torch.nn.functional.interpolate(
            masks,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks.squeeze().cpu().detach().numpy()
        metric_ROAD(images, model_truth, masks)

    ADs, ICs = metric_AD_IC.get_results()
    ROADs = metric_ROAD.get_results()
    if pbar:
        adic = f"{ADs[0]:.2f}/{ICs[0]:.2f}"
        roads = f"{ROADs[0]:.2f}/{ROADs[1]:.2f}"
        pbar.desc = f"{pbar.desc[:-36]}{adic:>20}{roads:>16}"

    return [*ADs, *ICs], ROADs


def main(args):
    print("Running parameters:\n")
    print(yaml.dump(args, indent=4))
    cfg = utils.load_config(args["cfg"])
    print(yaml.dump(cfg, indent=4))

    adic_data = []
    road_data = []
    if not args.get("epoch"):
        for epoch in range(0, cfg["epochs"]):
            args["epoch"] = epoch
            adic, road = run(cfg, args, example_gen=args["example_gen"])
            adic_data.append(adic)
            road_data.append(road)
        index = [f"Epoch {i}" for i in range(cfg["epochs"])]
    else:
        adic, road = run(cfg, args, example_gen=args["example_gen"])
        adic_data.append(adic)
        road_data.append(road)
        index = [f"Chosen Epoch {args['epoch']}"]
    columns = [
        "AD 100%",
        "AD 50%",
        "AD 15%",
        "IC 100%",
        "IC 50%",
        "IC 15%",
    ]
    data = pd.DataFrame(adic_data, columns=columns, index=index)
    new_columns = [
        "AD 100%",
        "IC 100%",
        "AD 50%",
        "IC 50%",
        "AD 15%",
        "IC 15%",
    ]
    data = data.reindex(columns=new_columns, copy=False)
    data.to_csv("evaluation data/data.csv", float_format="%.2f")
    road_data = pd.DataFrame(
        road_data, index=index, columns=[10, 20, 30, 40, 50, 70, 90]
    )
    road_data.to_csv("evaluation data/tame_road_data.csv", float_format="%.2f")
