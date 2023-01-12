"""Evaluation Script
Evaluate other explainability methods on a pretrained classifier

Usage:
    $ python -m tame.val --cfg resnet50_new.yaml --test --with-val
"""
import argparse
from pathlib import Path
from typing import Any, Dict, List, Type
import pandas as pd

import torch
import yaml
from torchvision import transforms
from tqdm import tqdm
from pytorch_grad_cam.metrics.road import (
    ROADMostRelevantFirst,
    ROADLeastRelevantFirst,
    ROADLeastRelevantFirstAverage,
    ROADMostRelevantFirstAverage,
    ROADCombined,
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
    FullGrad,
)
from pytorch_grad_cam.base_cam import BaseCAM

from . import utilities as utils
from .utilities import metrics


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def run(
    cfg: Dict[str, Any],
    args: Dict[str, Any],
    cam_method: Type[BaseCAM],
    percent_list: List[float] = [0.0, 0.5, 0.85],
) -> List[List[float]]:

    # Dataloader
    if args["with_val"]:
        dataloader = utils.data_loader(cfg)[1]
    else:
        dataloader = utils.data_loader(cfg)[2]
    model = torch.hub.load(
        "facebookresearch/deit:main", "deit_tiny_patch16_224", pretrained=True
    )
    model.cuda()
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
    metric_ROAD = metrics.ROAD(ROADLeastRelevantFirst(), model)

    if "vit" in cfg["model"]:
        target_layers = [model.blocks[-1].norm1]  # type: ignore
    elif "resnet" in cfg["model"]:
        target_layers = [model.layer4[-1]]  # type: ignore
    else:
        target_layers = [model.features[-1]]  # type: ignore
    cam_model = cam_method(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
        use_cuda=True,
    )
    for _, (images, labels) in bar:
        # with torch.cuda.amp.autocast():
        images, labels = images.cuda(), labels.cuda()  # type: ignore

        images: torch.Tensor
        labels: torch.LongTensor
        logits: torch.Tensor = model(images)

        logits = logits.softmax(dim=1)
        chosen_logits, model_truth = logits.max(dim=1)
        targets = metrics.MaxSelect(model_truth)
        masks = torch.tensor(cam_model(input_tensor=images, targets=targets())).cuda()
        if masks.dim() == 3:
            masks = masks.unsqueeze(dim=1)
        metric_AD_IC(images, chosen_logits, model_truth, torch.tensor(masks).cuda())
        metric_ROAD(images, chosen_logits, torch.tensor(masks).cuda(), targets())

    ADs, ICs = metric_AD_IC.get_results()
    ROADs = metric_ROAD.get_results()

    return [ADs, ICs, ROADs]


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluation script for methods included in pytorch_grad_cam library"
    )
    parser.add_argument(
        "--cfg", type=str, default="default.yaml", help="config script name (not path)"
    )
    parser.add_argument("--with-val", action="store_true", help="test with val dataset")
    parser.add_argument(
        "--method",
        type=str,
        default="GradCam",
        help="explainability method",
        choices=["GradCam", "HiResCAM", "ScoreCAM", "AblationCam"],
    )
    return parser.parse_args()


def main(args: Any):
    FILE = Path(__file__).resolve()
    ROOT_DIR = FILE.parents[1]
    print("Running parameters:\n")
    args = vars(args)
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
        "fullgrad": FullGrad,
    }
    stats = []
    if args.get("method", False):
        for name, method in methods.items():
            print(f"Evaluating {name} method")
            try:
                stats.append(run(cfg, args, method))
            except RuntimeError as e:
                print(e)
                stats.append(([], [], []))
    else:
        print(f"Evaluating {args['method']} method")
        try:
            stats.append(run(cfg, args, args["method"]))
        except RuntimeError as e:
            print(e)
            stats.append(([], [], []))
    data = pd.DataFrame(stats)
    data.columns = ["AD", "IC", "ROAD"]  # type: ignore
    data.to_csv("other_method_data.csv", float_format="%.2f")


if __name__ == "__main__":
    cmd_opt = get_arguments()
    main(cmd_opt)
