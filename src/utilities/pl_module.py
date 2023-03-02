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
        model_name: str,
        layers: List[str],
        attention_version: str = "TAME",
        noisy_masks: bool = True,
        optimizer: str = "SGD",
        momentum: float = 0.9,
        decay: float = 5.0e-4,
        schedule: str = "NEW",
        lr: float = 1.0e-3,
        epochs: int = 8,
        img_size: int = 224,
        percent_list: List[float] = [0.0, 0.5, 0.85],
        num_classes: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = ut.pl_get_config(
            model_name,
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

        # threshold is 0 because we use un-normalized logits to save on computation time
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, threshold=0
        )
        self.img_size = img_size
        self.metric_AD_IC = metrics.AD_IC(
            self.generic, img_size, percent_list=percent_list
        )
        self.metric_ROAD = metrics.ROAD(self.generic, ROADMostRelevantFirst)
        self.generic.requires_grad_(False)
        self.generic.attn_mech.requires_grad_()

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
        self.accuracy(logits, labels)
        self.log_dict(
            {
                "Loss": loss,
                "CE": ce,
                "Mean": mean_mask,
                "Var": var,
                "Accuracy": self.accuracy,
            },
            sync_dist=True,
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
        self.accuracy(logits, labels)
        self.log_dict(
            {
                "val_loss": loss,
                "val_ce": ce,
                "val_mean": mean_mask,
                "val_var": var,
                "val_accuracy": self.accuracy,
            },
            sync_dist=True,
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
        self.log("ADs", torch.tensor(ADs), sync_dist=True)
        self.log("ICs", torch.tensor(ICs), sync_dist=True)
        self.log("ROADS", torch.tensor(ROADs), sync_dist=True)

    def configure_optimizers(self):
        optimizer = ut.get_optim(self.cfg, self.generic)
        # the total steps are divided between num_nodes
        scheduler = ut.get_schedule(
            self.cfg,
            optimizer,
            self.current_epoch,
            total_steps=int(self.trainer.estimated_stepping_batches),
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "name": "LR", "frequency": 1}
        ]


class LightnightDataset(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Path,
        datalist_path: Path,
        batch_size: int = 32,
        num_workers: int = 12,
        num_classes: int = 1000,
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

        self.num_workers = num_workers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.datalist_file = datalist_file
        self.model = model

    def train_dataloader(self):
        tsfm_train = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        dataset_train = MyDataset(
            self.dataset_path / "ILSVRC2012_img_train",
            self.datalist_path / self.datalist_file
            if self.datalist_file
            else self.datalist_path / f"{self.model}_train.txt",
            transform=tsfm_train,
        )
        train_loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        tsfm_val = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
            ]
        )

        dataset_val = MyDataset(
            Path(self.dataset_path) / "ILSVRC2012_img_val",
            self.datalist_path / "Validation_2000.txt",
            transform=tsfm_val,
        )

        val_loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return val_loader

    def test_dataloader(self):
        tsfm_val = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
            ]
        )

        dataset_test = MyDataset(
            Path(self.dataset_path) / "ILSVRC2012_img_val",
            self.datalist_path / "Evaluation_2000.txt",
            transform=tsfm_val,
        )

        test_loader = DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader
