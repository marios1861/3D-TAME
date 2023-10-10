import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.loggers import CSVLogger
from torchvision import models
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

from tame.utilities import send_email
from tame.utilities.pl_module import TAMELIT, LightnightDataset

load_dotenv()

os.environ["MASTER_ADDR"] = "160.40.53.85"
os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "3"
torch.set_float32_matmul_precision("medium")
version = "TAME_SC"
model_name = "resnet50"
layers = [
    "layer2",
    "layer3",
    "layer4",
]
epochs = 8

model = TAMELIT.load_from_checkpoint(
    "logs/TAME_resnet50_oldnorm/version_0/checkpoints/epoch=7-step=320296.ckpt"
)

# sanity check: use randomized feature extractor
mdl = models.__dict__[model_name]()
train_names, eval_names = get_graph_node_names(mdl)

output = (train_names[-1], eval_names[-1])
if output[0] != output[1]:
    print("WARNING! THIS MODEL HAS DIFFERENT OUTPUTS FOR TRAIN AND EVAL MODE")
body = create_feature_extractor(mdl, return_nodes=(layers + [output[0]]))
body.eval()
model.generic.body = body
dataset = LightnightDataset(
    dataset_path=Path(os.getenv("DATA", "./")),
    datalist_path=Path(os.getenv("LIST", "./")),
    model=model_name,
    batch_size=32,
    legacy=True,
)

# torch._dynamo.config.verbose=True
trainer = pl.Trainer(
    # precision="16-mixed",
    # gradient_clip_algorithm="norm",
    max_epochs=epochs,
)

trainer.logger = CSVLogger("logs", name=(version + "_" + model_name))
trainer.test(model, dataset)
send_email(
    version + "_" + model_name,
    os.environ["PASS"],
    "resnet TAME run complete, check results and run vit_b_16 V5 next",
)
