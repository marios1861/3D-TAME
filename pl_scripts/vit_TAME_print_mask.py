import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from tame.utilities.pl_module import TAMELIT, LightnightDataset

load_dotenv()

torch.set_float32_matmul_precision("medium")

# # vgg
# model_name = "vgg16"
# model = TAMELIT.load_from_checkpoint("logs/TAME_vgg16_oldnorm/version_0/checkpoints/epoch=7-step=320296.ckpt", train_method="legacy")
# vit
model_name = "vit_b_16"
model = TAMELIT.load_from_checkpoint("logs/TAME_vit_b_16/version_0/checkpoints/epoch=7-step=320296.ckpt", train_method="legacy")
# resnet50
# model_name = "resnet50"
# model = TAMELIT.load_from_checkpoint("logs/TAME_resnet50_oldnorm/version_0/checkpoints/epoch=7-step=320296.ckpt", train_method="legacy")

# model: pl.LightningModule = torch.compile(model)  # type: ignore

dataset = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model=model_name,
    batch_size=32,
    legacy=True
)
dataset.test_dataloader()
dataset = dataset.test_dataset
id = 10
model.save_masked_image(dataset[id][0], id, model_name)
