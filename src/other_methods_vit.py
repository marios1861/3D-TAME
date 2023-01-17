"""Evaluation Script
Evaluate other explainability methods on a pretrained classifier

Usage:
    $ python -m tame.val --cfg resnet50_new.yaml --test --with-val
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type
import pandas as pd

import torch
import yaml
from torchvision import transforms
from tqdm import tqdm
from pytorch_grad_cam.metrics.road import (
    ROADMostRelevantFirst,
)
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
)
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.ablation_layer import AblationLayerVit

import utilities as utils
from utilities import metrics


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def run(
    cfg: Dict[str, Any],
    args: Dict[str, Any],
    cam_method: Dict[str, Type[BaseCAM]],
    name: str,
    percent_list: List[float] = [0.0, 0.5, 0.85],
) -> Tuple[List[float], List[float]]:

    # Dataloader
    if args["with_val"]:
        dataloader = utils.data_loader(cfg)[1]
    else:
        dataloader = utils.data_loader(cfg)[2]
    model = torch.hub.load(
        "facebookresearch/deit:main", "deit_tiny_patch16_224", pretrained=True
    )
    model.eval()

    # dig through dataloader to find input image size
    transform: transforms.Compose = dataloader.dataset.transform  # type: ignore
    img_size = next(
        tsfm for tsfm in transform.transforms if isinstance(tsfm, transforms.CenterCrop)
    ).size[0]

    n = len(dataloader)

    bar = tqdm(
        enumerate(dataloader),
        total=n,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    )

    metric_AD_IC = metrics.AD_IC(model, img_size, percent_list=percent_list)
    metric_ROAD = metrics.ROAD(model, ROADMostRelevantFirst)
    if "vit" in cfg["model"]:
        target_layers = [model.blocks[-1].norm1]  # type: ignore
    elif "resnet" in cfg["model"]:
        target_layers = [model.layer4[-1]]  # type: ignore
    else:
        target_layers = [model.features[-1]]  # type: ignore

    if name == "scorecam":
        use_cuda = False
    else:
        use_cuda = True

    if name == "ablationcam":
        cam_model = cam_method[name](
            model=model,
            target_layers=target_layers,
            use_cuda=use_cuda,
            reshape_transform=reshape_transform,
            ablation_layer=AblationLayerVit(),  # type: ignore
        )
    else:
        cam_model = cam_method[name](
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
            use_cuda=use_cuda,
        )
    for _, (images, labels) in bar:
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()  # type: ignore

        images: torch.Tensor
        labels: torch.LongTensor
        logits: torch.Tensor = model(images)

        logits = logits.softmax(dim=1)
        chosen_logits, model_truth = logits.max(dim=1)
        masks = cam_model(input_tensor=images)  # type: ignore
        metric_ROAD(images, model_truth, masks)
        if use_cuda:
            masks = torch.tensor(masks).cuda().unsqueeze(dim=1)
        else:
            masks = torch.tensor(masks).unsqueeze(dim=1)
        metric_AD_IC(images, chosen_logits, model_truth, masks)
        break

    ADs, ICs = metric_AD_IC.get_results()
    ROADs = metric_ROAD.get_results()

    return [*ADs, *ICs], ROADs


def main(args: Any):
    FILE = Path(__file__).resolve()
    ROOT_DIR = FILE.parents[1]
    print("Running parameters:\n")
    print(yaml.dump(args, indent=4))
    cfg = utils.load_config(ROOT_DIR / "configs" / args["cfg"])
    print(yaml.dump(cfg, indent=4))
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        # "fullgrad": FullGrad,
    }
    stats: List[List[float]] = []
    road_data = []
    if not args.get("method"):
        index = list(methods.keys())
        for name, method in methods.items():
            print(f"Evaluating {name} method")
            try:
                stat, data = run(cfg, args, methods, name)
                stats.append(stat)
                road_data.append(data)
            except RuntimeError as e:
                print(e)
                stats.append([])

    else:
        index = args["method"]
        print(f"Evaluating {args['method']} method")
        try:
            stat, data = run(cfg, args, methods, args["method"])
            stats.append(stat)
            road_data.append(data)
        except RuntimeError as e:
            print(e)
            stats.append([])
    columns = [
        "AD 100%",
        "AD 50%",
        "AD 15%",
        "IC 100%",
        "IC 50%",
        "IC 15%",
    ]
    data = pd.DataFrame(stats, columns=columns, index=index)
    new_columns = [
        "AD 100%",
        "IC 100%",
        "AD 50%",
        "IC 50%",
        "AD 15%",
        "IC 15%",
    ]
    data = data.reindex(columns=new_columns)
    data.to_csv("evaluation data/other_method_data.csv", float_format="%.2f")
    road_data = pd.DataFrame(
        road_data, index=index, columns=[10, 20, 30, 40, 50, 70, 90]
    )
    road_data.to_csv("evaluation data/other_road_data.csv", float_format="%.2f")
