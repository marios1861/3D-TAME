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
model_name = "resnet50"
layers = [
    "layer2",
    "layer3",
    "layer4",
]
<<<<<<< HEAD
epochs = 4
=======
epochs = 8
>>>>>>> 84b7918 (Fix ADIC metrics for hila method)

model = TAMELIT(
    model_name=model_name,
    layers=layers,
    attention_version=version,
<<<<<<< HEAD
    train_method="new",
    schedule="NEW",
    lr=0.0005,
=======
    train_method="legacy",
    schedule="NEW",
    lr=0.001,
>>>>>>> 84b7918 (Fix ADIC metrics for hila method)
    epochs=epochs,
)
# model: pl.LightningModule = torch.compile(model)  # type: ignore

dataset = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model=model_name,
    batch_size=32,
<<<<<<< HEAD
=======
    legacy=True,
>>>>>>> 84b7918 (Fix ADIC metrics for hila method)
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
<<<<<<< HEAD
<<<<<<<< HEAD:pl_scripts/ran/resnet_TAME_single_oldnorm.py
    "resnet TAME oldnorm run complete, check results and run vit_b_16 1 layer next",
========
    "resnet TAME new norm run complete, check results and run vit_b_16 TAME with 2 layers next",
>>>>>>>> 84b7918 (Fix ADIC metrics for hila method):pl_scripts/ran/resnet_TAME_single.py
=======
    "resnet TAME run complete, check results and run vit_b_16 V5 next",
>>>>>>> 84b7918 (Fix ADIC metrics for hila method)
)
