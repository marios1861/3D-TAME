"""Evaluation Script
Evaluate other explainability methods on a pretrained classifier

Usage:
    $ python -m tame.val --cfg resnet50_new.yaml --test --with-val
"""
import argparse
from pathlib import Path
from typing import Any, Dict, List, cast
import pandas as pd

import torch
import yaml
from torchvision import transforms
from tqdm import tqdm
from pytorch_grad_cam.metrics.road import (
    ROADMostRelevantFirst,
)
from .transformer_explainability.baselines.ViT.ViT_LRP import (
    vit_base_patch16_224 as vit_LRP,
)
from .transformer_explainability.baselines.ViT.ViT_explanation_generator import LRP

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
    percent_list: List[float] = [0.0, 0.5, 0.85],
) -> List[float]:

    # Dataloader
    # set batch size to 1 ***
    cfg["batch_size"] = 1
    if args["with_val"]:
        dataloader = utils.data_loader(cfg)[1]
    else:
        dataloader = utils.data_loader(cfg)[2]
    model = vit_LRP(pretrained=True).cuda()
    attribution_generator = LRP(model)
    model.eval()

    def gen_mask(image):
        transformer_attribution = attribution_generator.generate_LRP(
            image.cuda(),
            method="transformer_attribution",
        ).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(
            transformer_attribution, scale_factor=16, mode="bilinear"
        )
        transformer_attribution = (
            transformer_attribution.reshape(1, 224, 224).cuda().data.cpu().numpy()
        )
        transformer_attribution = (
            transformer_attribution - transformer_attribution.min()
        ) / (transformer_attribution.max() - transformer_attribution.min())
        return transformer_attribution

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
    metric_ROAD = metrics.ROAD(model, ROADMostRelevantFirst())

    use_cuda = True

    for _, (images, labels) in bar:
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()  # type: ignore

        images: torch.Tensor
        labels: torch.LongTensor
        logits: torch.Tensor = model(images)

        logits = logits.softmax(dim=1)
        chosen_logits, model_truth = logits.max(dim=1)
        targets = metrics.SoftmaxSelect(model_truth)
        masks = gen_mask(images)
        metric_ROAD(images, chosen_logits, masks, targets())
        if use_cuda:
            masks = torch.tensor(masks).cuda().unsqueeze(dim=1)
        else:
            masks = torch.tensor(masks).unsqueeze(dim=1)
        metric_AD_IC(images, chosen_logits, model_truth, masks)

    ADs, ICs = metric_AD_IC.get_results()
    ROADs = cast(List[float], metric_ROAD.get_results())

    return [*ADs, *ICs, *ROADs]


def main(args):
    FILE = Path(__file__).resolve()
    ROOT_DIR = FILE.parents[1]
    print("Running parameters:\n")
    print(yaml.dump(args, indent=4))
    cfg = utils.load_config(ROOT_DIR / "configs" / args["cfg"])
    print(yaml.dump(cfg, indent=4))

    stats = []
    stats.append(run(cfg, args))

    columns = [
        "AD 100%",
        "AD 50%",
        "AD 15%",
        "IC 100%",
        "IC 50%",
        "IC 15%",
        "ROAD AD",
        "ROAD IC",
    ]
    data = pd.DataFrame(stats, columns=columns)
    new_columns = [
        "AD 100%",
        "IC 100%",
        "AD 50%",
        "IC 50%",
        "AD 15%",
        "IC 15%",
        "ROAD AD",
        "ROAD IC",
    ]
    data = data.reindex(columns=new_columns, copy=False)
    data.to_csv("evaluation data/new_method_data.csv", float_format="%.2f")
