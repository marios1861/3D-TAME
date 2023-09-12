import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger
from tame.utilities.attention.factory import AMBuilder
from tame.utilities.attention.tame import AttentionTAME

from tame.utilities.pl_module import TAMELIT, LightnightDataset
from tame.utilities import send_email

load_dotenv()

os.environ["MASTER_ADDR"] = "160.40.53.85"
os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "3"
torch.set_float32_matmul_precision("medium")
version = "TAME_resnet"


class TestTAME(AttentionTAME):
    def __init__(self, ft_size):
        super(TestTAME, self).__init__(ft_size)

    def forward(self, features):
        _, c = super(TestTAME, self).forward(features)
        return c, c


AMBuilder.register_attention(version, TestTAME)
postfix = "_resnet"
epochs = 8
model = TAMELIT(
    model_name="resnet50",
    layers=[
        "layer2",
        "layer3",
        "layer4",
    ],
    attention_version=version,
    schedule="NEW",
    lr=0.001,
    epochs=epochs,
)
# compiled_model: pl.LightningModule = torch.compile(model)  # type: ignore

dataset = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model="resnet50",
    batch_size=32,
)
# torch._dynamo.config.verbose=True
trainer = pl.Trainer(
    precision="16-mixed",
    gradient_clip_algorithm="norm",
    max_epochs=epochs,
    num_nodes=3,
    strategy="ddp",
    accumulate_grad_batches=4,
)

trainer.fit(model, dataset)


tester = pl.Trainer(
    accelerator="gpu",
    logger=CSVLogger("logs", name=(version + postfix)),
)
tester.test(model, dataset)
send_email(version + postfix, os.environ["PASS"])
