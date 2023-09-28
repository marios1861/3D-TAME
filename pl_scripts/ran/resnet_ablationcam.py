import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from tame.utilities import send_email, COMPAREVIT, model_prep
from tame.utilities.pl_module import TAMELIT, LightnightDataset


load_dotenv()

os.environ["MASTER_ADDR"] = "160.40.53.85"
os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "3"
torch.set_float32_matmul_precision("medium")

version = "ablationcam"
model_name = "resnet50"
layers = [
    "encoder.layers.encoder_layer_9",
    "encoder.layers.encoder_layer_10",
    "encoder.layers.encoder_layer_11",
]
epochs = 8

mdl = model_prep(model_name)
model = COMPAREVIT(name=version, raw_model=mdl, mdl_name=model_name)
# model: pl.LightningModule = torch.compile(model)  # type: ignore

dataset = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model=model_name,
    batch_size=32,
)

# torch._dynamo.config.verbose=True
trainer = pl.Trainer()
trainer.logger = CSVLogger("logs", name=(version + "_" + model_name))
trainer.test(model, dataset)
send_email(
    version + "_" + model_name,
    os.environ["PASS"],
    "vit_b_16 gradcam run complete, check results and run vgg16 gradcam++ next",
)
