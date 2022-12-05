"""
Evaluate an attention mechanism model on a pretrained classifier

Usage:
    $ python -m tame.val --cfg resnet50_new.yaml --test --with-val
"""
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from torchvision import transforms
from tqdm import tqdm

from . import utilities as utils
from .utilities import AverageMeter, metrics


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
        cfg['noisy_masks'] = False
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
    ADs = [AverageMeter(type="avg") for _ in percent_list]
    ICs = [AverageMeter(type="avg") for _ in percent_list]
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
        masks = metrics.normalizeMinMax(masks)
        masked_images_list = metrics.get_masked_inputs(
            images, masks, img_size, percent_list, model.noisy_masks
        )

        new_logits_list = [model(masked_images) for masked_images in masked_images_list]
        new_logits_list = [
            new_logits.softmax(dim=1).gather(1, model_truth.unsqueeze(-1)).squeeze()
            for new_logits in new_logits_list
        ]
        for AD, IC, new_logits in zip(ADs, ICs, new_logits_list):
            AD.update(metrics.get_AD(chosen_logits, new_logits))
            IC.update(metrics.get_IC(chosen_logits, new_logits))

    if pbar:
        losses_str = "".join(
            f"{loss():>{align}.2f}" for loss, align in zip(losses, [16, 6, 6, 6])
        )
        AD_IC_str = "".join(
            f"{num_pair:>{align}}"
            for num_pair, align in zip(
                (f"{AD():.2f}/{IC():.2f}" for AD, IC in zip(ADs, ICs)),
                [12, 12, 12]
            )
        )
        pbar.desc = f"{pbar.desc[:-70]}{losses_str}{AD_IC_str}"
    return [
        [loss.avg for loss in losses],
        [AD.avg for AD in ADs],
        [IC.avg for IC in ICs],
    ]


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument(
        "--cfg", type=str, default="default.yaml", help="config script name (not path)"
    )
    parser.add_argument("--with-val", action="store_true", help="test with val dataset")
    return parser.parse_args()


def main(args):
    FILE = Path(__file__).resolve()
    ROOT_DIR = FILE.parents[1]
    print("Running parameters:\n")
    args = vars(args)
    print(yaml.dump(args, indent=4))
    cfg = utils.load_config(ROOT_DIR / "configs", args['cfg'])
    print(yaml.dump(cfg, indent=4))
    stats = []
    for epoch in range(0, cfg['epochs']):
        args['epoch'] = epoch
        stats.append(run(cfg, args))
        print(stats[epoch])
    # print(stats)


if __name__ == "__main__":
    cmd_opt = get_arguments()
    main(cmd_opt)
