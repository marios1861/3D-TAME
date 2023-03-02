from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchmetrics
import torchvision.transforms as transforms
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst

import utilities as ut
from utilities import metrics
from utilities.load_data import MyDataset


# define the LightningModule
class TAMELIT(pl.LightningModule):
    def __init__(
        self,
        model: str,
        layers: List[str],
        attention_version: str,
        noisy_masks: bool,
        optimizer: str,
        momentum: float,
        decay: float,
        schedule: str,
        lr: float,
        epochs: int,
        img_size: int = 224,
        percent_list: List[float] = [0.0, 0.5, 0.85],
        num_classes: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = ut.pl_get_config(
            model,
            layers,
            attention_version,
            noisy_masks,
            optimizer,
            momentum,
            decay,
            schedule,
            lr,
            epochs,
        )
        self.generic = ut.get_model(self.cfg)
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.img_size = img_size
        self.metric_AD_IC = metrics.AD_IC(
            self.generic, img_size, percent_list=percent_list
        )
        self.metric_ROAD = metrics.ROAD(self.generic, ROADMostRelevantFirst)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        images, labels = batch
        logits = self.generic(images, labels)
        masks = self.generic.get_a(labels)
        (
            loss,
            ce,
            mean_mask,
            var,
        ) = self.generic.get_loss(logits, labels, masks)
        self.log_dict(
            {
                "loss": loss,
                "ce": ce,
                "mean": mean_mask,
                "var": var,
                "accuracy": self.accuracy,
            }
        )
        return {"loss": loss, "ce": ce, "mean": mean_mask, "var": var}

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        images, labels = batch
        logits = self.generic(images, labels)
        masks = self.generic.get_a(labels)
        (
            loss,
            ce,
            mean_mask,
            var,
        ) = self.generic.get_loss(logits, labels, masks)
        self.log_dict(
            {
                "val_loss": loss,
                "val_ce": ce,
                "val_mean": mean_mask,
                "val_var": var,
                "val_accuracy": self.accuracy,
            }
        )

    def test_step(self, batch, batch_idx):
        # this is the test loop
        images, labels = batch
        logits = self.generic(images)
        logits = logits.softmax(dim=1)
        chosen_logits, model_truth = logits.max(dim=1)
        masks = self.generic.get_c(model_truth)
        masks = metrics.normalizeMinMax(masks)
        self.metric_AD_IC(images, chosen_logits, model_truth, masks)
        masks = torch.nn.functional.interpolate(
            masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks.squeeze().cpu().detach().numpy()
        self.metric_ROAD(images, model_truth, masks)

        ADs, ICs = self.metric_AD_IC.get_results()
        ROADs = self.metric_ROAD.get_results()
        self.log("ADs", torch.tensor(ADs))
        self.log("ICs", torch.tensor(ICs))
        self.log("ROADS", torch.tensor(ROADs))

    def configure_optimizers(self):
        optimizer = ut.get_optim(self.cfg, self.generic)
        scheduler = ut.get_schedule(
            self.cfg, optimizer, self.trainer.max_steps, self.current_epoch
        )
        return (optimizer, scheduler)


class LightnightDataset(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Path,
        datalist_path: Path,
        batch_size: int = 32,
        input_size: int = 256,
        crop_size: int = 224,
        datalist_file: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.crop_size = crop_size
        self.dataset_path = dataset_path
        self.datalist_path = datalist_path

        self.batch_size = batch_size
        self.datalist_file = datalist_file
        self.model = model

    def setup(self, stage: str):
        tsfm_train = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        tsfm_val = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
            ]
        )

        self.dataset_train = MyDataset(
            self.dataset_path / "ILSVRC2012_img_train",
            self.datalist_path / self.datalist_file
            if self.datalist_file
            else self.datalist_path / f"{self.model}_train.txt",
            transform=tsfm_train,
        )
        self.dataset_val = MyDataset(
            Path(self.dataset_path) / "ILSVRC2012_img_val",
            self.datalist_path / "Validation_2000.txt",
            transform=tsfm_val,
        )
        self.dataset_test = MyDataset(
            Path(self.dataset_path) / "ILSVRC2012_img_val",
            self.datalist_path / "Evaluation_2000.txt",
            transform=tsfm_val,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return test_loader
