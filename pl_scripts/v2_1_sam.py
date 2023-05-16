import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger

from tame.utilities.pl_module import TAMELIT, LightnightDataset

load_dotenv()

os.environ["MASTER_ADDR"] = "160.40.53.85"
os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "3"
torch.set_float32_matmul_precision("medium")
model = TAMELIT(
    model_name="vit_b_16",
    layers=[
        "encoder.layers.encoder_layer_9",
        "encoder.layers.encoder_layer_10",
        "encoder.layers.encoder_layer_11",
    ],
    attention_version="TAttentionV2_1",
    schedule="NEW",
    lr=0.001,
    epochs=8,
    use_sam=True,
)
# compiled_model: pl.LightningModule = torch.compile(model)  # type: ignore

dataset = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model="vit_b_16",
    batch_size=64,
)
# torch._dynamo.config.verbose=True
trainer = pl.Trainer(
    accelerator="gpu",
    num_nodes=3,
    strategy="ddp",
    # precision="16-mixed",
    max_epochs=8,
)

trainer.fit(model, dataset)

if os.environ["NODE_RANK"] == "0":
    tester = pl.Trainer(
        accelerator="gpu",
        logger=CSVLogger("logs", name="vit_b_16_run_2_ddp_v1"),
    )
    tester.test(model, dataset)
