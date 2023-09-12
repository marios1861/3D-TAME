import os
from pathlib import Path
from turtle import st

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

version = "TAttentionV3"
model_name = "vit_b_16"
postfix = "_vit_ln"
epochs = 8


ckpt = torch.load("/home/marios/Documents/T-TAME/pl_scripts/lightning_logs/version_4/checkpoints/epoch=7-step=320296.ckpt")
ckpt['state_dict']["generic.arrangement.norm.weight"] = ckpt['state_dict']["generic.arrangement.ln.weight"]
ckpt['state_dict']["generic.arrangement.norm.bias"] = ckpt['state_dict']["generic.arrangement.ln.bias"]
ckpt['state_dict'].pop("generic.arrangement.ln.weight")
ckpt['state_dict'].pop("generic.arrangement.ln.bias")
model = TAMELIT(**ckpt["hyper_parameters"], eval_length="short")
model.load_state_dict(ckpt["state_dict"])

dataset = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model=model_name,
    batch_size=32,
)

# checkpointer = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)

# # torch._dynamo.config.verbose=True
# trainer = pl.Trainer(
#     accelerator="gpu",
#     precision="16-mixed",
#     gradient_clip_algorithm="norm",
#     max_epochs=epochs,
#     callbacks=[checkpointer],
# )

# trainer.fit(model, dataset)


tester = pl.Trainer(
    logger=CSVLogger("logs", name=(version + postfix)),
    deterministic=True,
)
tester.test(model, dataset)
send_email(version + postfix, os.environ["PASS"])
