"""Evaluation Script
Evaluate an attention mechanism model on a pretrained classifier

Usage:
    $ python -m tame.val --cfg resnet50_new.yaml --test --with-val
"""
import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

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

from . import utilities as utils
from .utilities import AverageMeter, metrics


@dataclass
class AD_IC:
    model: torch.nn.Module
    img_size: int
    percent_list: List[float] = [0.0, 0.5, 0.85]
    noisy_masks: bool = False
    ADs: List[AverageMeter] = []
    ICs: List[AverageMeter] = []

    def __post_init__(self):
        self.ADs = [AverageMeter(type="avg") for _ in self.percent_list]
        self.ICs = [AverageMeter(type="avg") for _ in self.percent_list]

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        chosen_logits: torch.Tensor,
        model_truth: torch.Tensor,
        masks: torch.Tensor,
    ):
        masks = metrics.normalizeMinMax(masks)
        masked_images_list = metrics.get_masked_inputs(
            images,
            masks,
            self.img_size,
            self.percent_list,
            self.noisy_masks,
        )

        new_logits_list = [
            new_logits.softmax(dim=1).gather(1, model_truth.unsqueeze(-1)).squeeze()
            for new_logits in [
                self.model(masked_images) for masked_images in masked_images_list
            ]
        ]

        for AD, IC, new_logits in zip(self.ADs, self.ICs, new_logits_list):
            AD.update(metrics.get_AD(chosen_logits, new_logits))
            IC.update(metrics.get_IC(chosen_logits, new_logits))

    def get_results(self) -> Tuple[List[float], List[float]]:
        return [AD() for AD in self.ADs], [IC() for IC in self.ICs]


@dataclass
class ROAD:
    road: Union[
        List[ROADMostRelevantFirst],
        List[ROADLeastRelevantFirst],
        ROADMostRelevantFirst,
        ROADLeastRelevantFirst,
        ROADMostRelevantFirstAverage,
        ROADLeastRelevantFirstAverage,
        ROADCombined,
    ]
    model: torch.nn.Module
    metric: Union[List[AverageMeter], AverageMeter] = []

    def post_init(self):
        if isinstance(self.road, (List)):
            self.metric = []
            for i in range(len(self.road)):
                self.metric.append(AverageMeter(type="avg"))
        else:
            self.metric = AverageMeter(type="avg")

    @torch.no_grad()
    def __call__(
        self,
        input: torch.Tensor,
        cams: torch.Tensor,
        targets: List[Callable],
    ):
        if isinstance(self.road, (List)):
            assert isinstance(self.metric, List)
            scores = [
                metric(input, cams.cpu().detach().numpy(), targets, self.model)
                for metric in self.road
            ]
            for metric, score in zip(self.metric, scores):
                metric.update(score)  # type: ignore
        else:
            assert isinstance(self.metric, AverageMeter)
            score = self.road(input, cams.cpu().detach().numpy(), targets, self.model)
            self.metric.update(score)  # type: ignore

    def get_results(self) -> Union[float, List[float]]:
        if isinstance(self.metric, List):
            return [metric() for metric in self.metric]
        else:
            assert isinstance(self.metric, AverageMeter)
            return self.metric()


class SoftmaxSelect:
    def __init__(self, model_truth):
        self.model_truth = model_truth

    class SingleSelect:
        def __init__(self, label):
            self.label = label

        def __call__(self, single_output):
            single_output.softmax(dim=-1)[self.label]

    def __call__(self):
        callables = []
        for label in zip(self.model_truth):
            callables.append(self.SingleSelect(label))
        return callables


@torch.no_grad()
def run(
    cfg: Optional[Dict[str, Any]] = None,
    args: Optional[Dict[str, Any]] = None,
    percent_list: List[float] = [0.0, 0.5, 0.85],
    model: Optional[utils.Generic] = None,
    dataloader: Optional[torch.utils.data.DataLoader] = None,  # type: ignore
    pbar: Optional[tqdm] = None,
) -> List[List[float]]:
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
    desc = f"{pbar.desc[:-35]}{action:>35}" if pbar else f"{action}"
    bar = tqdm(
        enumerate(dataloader),
        desc,
        n,
        False,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        position=0,
    )

    losses = [AverageMeter(type="avg") for _ in range(0, 4)]
    metric_AD_IC = AD_IC(model, img_size, percent_list=percent_list)
    metric_ROAD = ROAD(ROADLeastRelevantFirst(), model)
    for _, (images, labels) in bar:
        # with torch.cuda.amp.autocast():
        images, labels = images.cuda(), labels.cuda()  # type: ignore
        images: torch.Tensor
        labels: torch.LongTensor
        logits: torch.Tensor = model(images)
        masks_for_loss = model.get_a(labels)

        losses_vals = model.get_loss(logits, labels, masks_for_loss)

        for loss, loss_val in zip(losses, losses_vals):
            loss.update(loss_val.item())

        logits = logits.softmax(dim=1)
        chosen_logits, model_truth = logits.max(dim=1)
        masks = model.get_c(model_truth)
        metric_AD_IC(images, chosen_logits, model_truth, masks)
        targets = SoftmaxSelect(model_truth)
        metric_ROAD(images, masks, targets())

    ADs, ICs = metric_AD_IC.get_results()

    if pbar:
        losses_str = "".join(
            f"{loss():>{align}.2f}" for loss, align in zip(losses, [16, 6, 6, 6])
        )
        AD_IC_str = "".join(
            f"{num_pair:>{align}}"
            for num_pair, align in zip(
                (f"{AD:.2f}/{IC:.2f}" for AD, IC in zip(ADs, ICs)), [12, 12, 12]
            )
        )
        pbar.desc = f"{pbar.desc[:-70]}{losses_str}{AD_IC_str}"
    return [
        [loss.avg for loss in losses],
        ADs,
        ICs,
    ]


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument(
        "--cfg", type=str, default="default.yaml", help="config script name (not path)"
    )
    parser.add_argument("--with-val", action="store_true", help="test with val dataset")
    return parser.parse_args()


def main(args: Any):
    FILE = Path(__file__).resolve()
    ROOT_DIR = FILE.parents[1]
    print("Running parameters:\n")
    args = vars(args)
    print(yaml.dump(args, indent=4))
    cfg = utils.load_config(ROOT_DIR / "configs" / args["cfg"])
    print(yaml.dump(cfg, indent=4))
    stats = []
    for epoch in range(0, cfg["epochs"]):
        args["epoch"] = epoch
        stats.append(run(cfg, args))
        print(stats[epoch])
    # print(stats)


if __name__ == "__main__":
    cmd_opt = get_arguments()
    main(cmd_opt)
