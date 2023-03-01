"""Evaluation Script
Evaluate other explainability methods on a pretrained classifier

Usage:
    $ python -m tame.val --cfg resnet50_new.yaml --test --with-val
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd
import torch
from utilities.model_prep import model_prep
import yaml
from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    ScoreCAM,
    XGradCAM,
)
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
from torchvision import transforms
from tqdm import tqdm

import utilities as utils
from masked_print import save_heatmap
from utilities import metrics


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def run(
    cfg: Dict[str, Any],
    cam_method: Dict[str, Type[BaseCAM]],
    name: str,
    percent_list: List[float] = [0.0, 0.5, 0.85],
    example_gen: Optional[int] = None,
) -> Tuple[List[float], List[float]]:

    # Dataloader
    dataloader = utils.data_loader(cfg)[2]
    if 'vit' in cfg['model']:
        model = torch.hub.load(
            "facebookresearch/deit:main", "deit_tiny_patch16_224", pretrained=True
        )
    else:
        model = model_prep(cfg['model'])
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
        if 'vit' in cfg['model']:
            cam_model = cam_method[name](
                model=model,
                target_layers=target_layers,
                use_cuda=use_cuda,
                reshape_transform=reshape_transform,
                ablation_layer=AblationLayerVit(),  # type: ignore
            )
        else:
            raise NotImplementedError(f"AblationCAM not implemented for model: {cfg['model']}")
    else:
        cam_model = cam_method[name](
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform if 'vit' in cfg['model'] else None,
            use_cuda=use_cuda,
        )

    if example_gen is not None:
        image, _ = dataloader.dataset[example_gen]
        image = image.unsqueeze(0)
        mask = torch.tensor(cam_model(input_tensor=image))
        image = image.squeeze()
        save_heatmap(
            mask,
            image,
            Path("evaluation data") / "examples" / f"grad_{name}_{example_gen}.jpg",
        )
        quit()

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

    ADs, ICs = metric_AD_IC.get_results()
    ROADs = metric_ROAD.get_results()

    return [*ADs, *ICs], ROADs


def main(args: Any):
    print("Running parameters:\n")
    print(yaml.dump(args, indent=4))
    cfg = utils.load_config(args["cfg"])
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
    if 'resnet' in cfg['model']:
        cfg['batch_size'] = 8
    stats: List[List[float]] = []
    road_data = []
    if not args.get("method"):
        index = list(methods.keys()) if not args["classic"] else list(methods.keys())[0:3]
        for name in index:
            print(f"Evaluating {name} method")
            try:
                stat, data = run(cfg, methods, name, example_gen=args["example_gen"])
                stats.append(stat)
                road_data.append(data)
            except (RuntimeError, AttributeError, ValueError) as e:
                print(e)
                stats.append([])
                road_data.append([])

    else:
        index = args["method"]
        print(f"Evaluating {args['method']} method")
        try:
            stat, data = run(
                cfg, methods, args["method"], example_gen=args["example_gen"]
            )
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
    data.to_csv(f"evaluation data/{cfg['model']}_other_adic.csv", float_format="%.2f")
    road_data = pd.DataFrame(
        road_data, index=index, columns=[10, 20, 30, 40, 50, 70, 90]
    )
    road_data.to_csv(
        f"evaluation data/{cfg['model']}_other_road.csv", float_format="%.2f"
    )
