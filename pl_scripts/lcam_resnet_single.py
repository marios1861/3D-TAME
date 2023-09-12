import os
from pathlib import Path

from torch import nn

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from tame.utilities import send_email
from tame.utilities.attention.factory import AMBuilder
from tame.utilities.pl_module import TAMELIT, LightnightDataset
from tame.utilities.attention.generic_atten import AttentionMech

load_dotenv()

os.environ["MASTER_ADDR"] = "160.40.53.85"
os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "3"
torch.set_float32_matmul_precision("medium")
version = "lcam"
model_name = "resnet50"
layers = [
    "layer4",
]
epochs = 8
postfix = "_resnet"


class LCAM(AttentionMech):
    def __init__(self, ft_size):
        super(AttentionMech, self).__init__()
        in_channel = ft_size[0][1]
        self.op = nn.Conv2d(
            in_channels=in_channel,
            out_channels=1000,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, features):
        features = list(features)
        c = self.op(features[0])
        a = torch.sigmoid(c)
        return a, c


AMBuilder.register_attention(version, LCAM)
model = TAMELIT(
    model_name=model_name,
    layers=layers,
    attention_version=version,
    train_method="legacy",
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

checkpointer = ModelCheckpoint(every_n_epochs=1)

# torch._dynamo.config.verbose=True
trainer = pl.Trainer(
    # precision="16-mixed",
    # gradient_clip_algorithm="norm",
    max_epochs=epochs,
    callbacks=[checkpointer],
)
trainer.logger = TensorBoardLogger("logs", name=(version + postfix), sub_dir="tb_logs")
print(trainer.logger, trainer.loggers)
trainer.fit(model, dataset)


trainer.logger = CSVLogger("logs", name=(version + postfix))
trainer.test(model, dataset)
send_email(
    version + postfix,
    os.environ["PASS"],
    "resnet lcam run complete, check results and run resnet V5 next",
)
