from functools import partial
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from tame.utilities.attention.factory import AMBuilder
from tame.utilities.attention.tattentionv3 import TAttentionV3
from tame.utilities.pl_module import TAMELIT, LightnightDataset

load_dotenv()

os.environ["MASTER_ADDR"] = "160.40.53.85"
os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "3"
torch.set_float32_matmul_precision("medium")
version = "evals"
postfix = "_test"
epochs = 8

pl.seed_everything(42, workers=True)


def forward_hook(module, input, output):
    a, c = output
    return a, c


model = TAMELIT.load_from_checkpoint(
    "/home/ntrougkas/Documents/T-TAME/pl_scripts/lightning_logs/version_13/checkpoints/epoch=7-step=320296.ckpt",
    eval_protocol="new",
    eval_length="short",
)
model.generic.attn_mech.register_forward_hook(forward_hook)
# model: pl.LightningModule = torch.compile(model)  # type: ignore

dataset = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model="resnet50",
    batch_size=32,
)
# checkpointer = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)

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
    deterministic=True,
    logger=CSVLogger("logs", name=(version + postfix)),
)
tester.test(model, dataset)
# send_email(version + postfix, os.environ["PASS"])
