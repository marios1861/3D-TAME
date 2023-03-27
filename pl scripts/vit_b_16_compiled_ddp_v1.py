from pathlib import Path
import lightning.pytorch as pl
import torch
from utilities.pl_module import TAMELIT, LightnightDataset
import os
os.environ["MASTER_ADDR"] = "160.40.53.85"
os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "3"

model = TAMELIT(
    model_name="vit_b_16",
    layers=[
        "encoder.layers.encoder_layer_9",
        "encoder.layers.encoder_layer_10",
        "encoder.layers.encoder_layer_11",
    ],
    attention_version="V1",
    schedule="NEW",
    lr=0.0001,
    epochs=8,
)
compiled_model: pl.LightningModule = torch.compile(model)  # type: ignore

dataset = LightnightDataset(
    dataset_path=Path("/home/marios/Documents/imagenet-1k"),
    datalist_path=Path("/home/marios/Documents/T-TAME/datalist/ILSVRC"),
    model="vit_b_16",
    batch_size=64,
)

trainer = pl.Trainer(
    accelerator="gpu",
    num_nodes=3,
    strategy="ddp",
    precision="16-mixed",
    accumulate_grad_batches=4,
    gradient_clip_algorithm="norm",
)

trainer.fit(compiled_model, dataset)

trainer.test(compiled_model, dataset)
