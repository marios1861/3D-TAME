from pathlib import Path
from lightning.pytorch.loggers import CSVLogger
import lightning.pytorch as pl
import torch
from utilities.pl_module import TAMELIT, LightnightDataset
import os
os.environ["MASTER_ADDR"] = "160.40.53.85"
os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "3"
torch.set_float32_matmul_precision("medium")
model = TAMELIT.load_from_checkpoint("lightning_logs/version_1/checkpoints/epoch=7-step=13352.ckpt")

dataset = LightnightDataset(
    dataset_path=Path("/home/marios/Documents/imagenet-1k"),
    datalist_path=Path("/home/marios/Documents/T-TAME/datalist/ILSVRC"),
    model="vit_b_16",
    batch_size=64,
)

if os.environ["NODE_RANK"] == "0":
    tester = pl.Trainer(
        accelerator="gpu",
        logger=CSVLogger("logs", name="vit_b_16_ddp_v1"),
    )
    tester.test(model, dataset)
