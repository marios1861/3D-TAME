"""
Generate a list holding the model truth

Usage:
    $ python -m tame.gen_trainlist --cfg resnet50_new.yaml
"""
import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torchvision import transforms
from tqdm import tqdm

from . import utilities as utils


@torch.inference_mode()
def run(cfg: Dict[str, Any], train_list: Path) -> None:
    if train_list.is_file():
        print(f"Training list already exists at {train_list} !")
        return

    model = utils.model_prep.model_prep(cfg['model']).cuda()
    # use the vgg16 train list to bootstrap new train lists
    transform = transforms.Compose(
        [transforms.Resize(cfg['input_size']),
         transforms.CenterCrop(cfg['crop_size']),
         transforms.ToTensor(),
         ])
    dataset = utils.load_data.MyDataset(
        Path(cfg['dataset']) / 'ILSVRC2012_img_train',
        Path(cfg['datalist']) / "vgg16_train.txt",
        transform=transform)
    dataloader = torch.utils.data.DataLoader(  # type: ignore
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True)
    samples = utils.load_data.read_labeled_image_list(
        Path(cfg['dataset']) / 'ILSVRC2012_img_train',
        Path(cfg['datalist']) / "vgg16_train.txt")

    def update_samples(samples, new_labels, batch):
        old_samples = samples[batch * cfg["batch_size"]: (batch + 1) * cfg["batch_size"]]
        new_samples = []
        for (dir, _), new_label in zip(old_samples, new_labels):
            new_samples.append((Path(*Path(dir).parts[-2: None]), str(new_label.item())))
        samples[batch * cfg["batch_size"]: (batch + 1) * cfg["batch_size"]] = new_samples

    for i, (images, _) in enumerate(tqdm(dataloader)):
        images = images.cuda()  # type: ignore
        images: torch.Tensor
        logits: torch.Tensor = model(images)
        _, model_truth = logits.max(dim=1)
        update_samples(samples, model_truth, i)

    with open(train_list, 'w') as f:
        for item in samples:
            line = ' '.join(item) + '\n'
            f.write(line)


def get_arguments():
    parser = argparse.ArgumentParser(description="Train list generation script")
    parser.add_argument(
        "--cfg", type=str, default="default.yaml", help="config script name (not path)"
    )
    return parser.parse_args()


def main(args):
    FILE = Path(__file__).resolve()
    ROOT_DIR = FILE.parents[1]
    print("Running parameters:\n")
    args = vars(args)
    print(yaml.dump(args, indent=4))
    cfg = utils.load_config(ROOT_DIR / "configs", args["cfg"])
    print(yaml.dump(cfg, indent=4))
    train_list = ROOT_DIR / "datalist" / "ILSVRC" / f"{cfg['model']}_train.txt"
    run(cfg, train_list)


if __name__ == "__main__":
    cmd_opt = get_arguments()
    main(cmd_opt)
