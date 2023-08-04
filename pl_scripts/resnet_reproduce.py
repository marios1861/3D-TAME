import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from tame.utilities import send_email
from tame.utilities.pl_module import TAMELIT, LightnightDataset

load_dotenv()

os.environ["MASTER_ADDR"] = "160.40.53.85"
os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "3"
torch.set_float32_matmul_precision("medium")
pl.seed_everything(42, workers=True)
version = "TAME"
model_name = "resnet50"
postfix = "_resnet_reproduce"
epochs = 8
model = TAMELIT(
    model_name=model_name,
    layers=[
        "layer2",
        "layer3",
        "layer4",
    ],
    attention_version=version,
    optimizer="OLDSGD",
    eval_protocol="old",
    eval_length="short",
    lr=0.0001,
    epochs=epochs,
)
# model: pl.LightningModule = torch.compile(model)  # type: ignore

dataset = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model=model_name,
    batch_size=32,
)
checkpointer = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)

# torch._dynamo.config.verbose=True
trainer = pl.Trainer(
    precision="16-mixed",
    gradient_clip_algorithm="norm",
    max_epochs=epochs,
    callbacks=[checkpointer],
    logger=CSVLogger("logs", name=(version + postfix)),
)

trainer.fit(model, dataset)
tester = pl.Trainer(
    logger=CSVLogger("logs", name=(version + postfix)),
)
tester.test(model, dataset)
send_email(version + postfix, os.environ["PASS"])
