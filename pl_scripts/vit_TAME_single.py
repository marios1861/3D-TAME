import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from tame.utilities import send_email
from tame.utilities.pl_module import TAMELIT, LightnightDataset

load_dotenv()

os.environ["MASTER_ADDR"] = "160.40.53.85"
os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "3"
torch.set_float32_matmul_precision("medium")

version = "TAME"
model_name = "vit_b_16"
layers = [
    "encoder.layers.encoder_layer_9",
    "encoder.layers.encoder_layer_10",
    "encoder.layers.encoder_layer_11",
]
epochs = 8

model = TAMELIT(
    model_name=model_name,
    layers=layers,
    attention_version=version,
    train_method="new",
    eval_protocol="old",
    schedule="NEW",
    lr=0.001,
    epochs=epochs,
)
# model: pl.LightningModule = torch.compile(model)  # type: ignore

dataset = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model=model_name,
    batch_size=32,
)

checkpointer = ModelCheckpoint(every_n_epochs=1, save_on_train_epoch_end=False)

# torch._dynamo.config.verbose=True
trainer = pl.Trainer(
    # precision="16-mixed",
    # gradient_clip_algorithm="norm",
    max_epochs=epochs,
    callbacks=[checkpointer],
)
trainer.logger = TensorBoardLogger(
    "logs", name=(version + "_" + model_name), sub_dir="tb_logs"
)
trainer.fit(model, dataset)


trainer.logger = CSVLogger("logs", name=(version + "_" + model_name))
trainer.test(model, dataset)
send_email(
    version + "_" + model_name,
    os.environ["PASS"],
    "vit_b_16 TAME run complete, check results and run GradCam next",
)
