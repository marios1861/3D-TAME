import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger

from tame.utilities.pl_module import TAMELIT, LightnightDataset
from tame.utilities import send_email

load_dotenv()

os.environ["MASTER_ADDR"] = "160.40.53.85"
os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "3"
torch.set_float32_matmul_precision("medium")
version = "TAME"
postfix = "_resnet"
model = TAMELIT(
    model_name="resnet50",
    layers=[
        "layer2",
        "layer3",
        "layer4",
    ],
    attention_version=version,
    schedule="NEW",
    lr=0.0001,
    epochs=8,
)
# compiled_model: pl.LightningModule = torch.compile(model)  # type: ignore

dataset = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model="resnet50",
    batch_size=64,
)
# torch._dynamo.config.verbose=True
trainer = pl.Trainer(
    accelerator="gpu",
    num_nodes=3,
    strategy="ddp",
    precision="16-mixed",
    accumulate_grad_batches=4,
    gradient_clip_algorithm="norm",
    max_epochs=8,
)

trainer.fit(model, dataset)

if os.environ["NODE_RANK"] == "0":
    tester = pl.Trainer(
        accelerator="gpu",
        logger=CSVLogger("logs", name=(version + postfix)),
    )
    tester.test(model, dataset)
    send_email(version + postfix, os.environ["PASS"])
