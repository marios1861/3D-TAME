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
version = "v3"
postfix = "_vit"
epochs = 8
model = TAMELIT.load_from_checkpoint(
    "/home/marios/Documents/T-TAME/pl_scripts/old_lightning_logs/tame_new/checkpoints/epoch=7-step=20024.ckpt",
    eval_protocol="old",
)
# model: pl.LightningModule = torch.compile(model)  # type: ignore

dataset = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model="resnet50",
    batch_size=32,
)
checkpointer = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)

# torch._dynamo.config.verbose=True
# trainer = pl.Trainer(
#     precision="16-mixed",
#     gradient_clip_algorithm="norm",
#     max_epochs=epochs,
#     callbacks=[checkpointer],
# )

# trainer.fit(model, dataset)


tester = pl.Trainer(
    accelerator="gpu",
    # logger=CSVLogger("logs", name=(version + postfix)),
)
tester.test(model, dataset)
# send_email(version + postfix, os.environ["PASS"])
