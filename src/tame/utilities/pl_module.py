from pathlib import Path
from typing import List, Optional

import lightning.pytorch as pl
import torch
import torchmetrics
import torchvision.transforms as transforms
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
from torch.utils.data import DataLoader

from tame import utilities as ut
from tame.utilities.sam import SAM

from . import metrics
from .load_data import MyDataset


# define the LightningModule
class TAMELIT(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        layers: List[str],
        attention_version: str = "TAME",
        noisy_masks: str = "random",
        optimizer: str = "SGD",
        use_sam: bool = False,
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
        self.attention_version = attention_version

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
        self.use_sam = use_sam
        if use_sam:
            self.automatic_optimization = False

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
                "training/Loss": loss,
                "training/CE": ce,
                "training/Mean": mean_mask,
                "training/Var": var,
                "training/Accuracy": self.accuracy,
            },
            sync_dist=True,
        )
        if self.use_sam:
            optimizer = self.optimizers()
            sch = self.lr_schedulers()
            assert isinstance(optimizer, SAM)
            assert isinstance(sch, torch.optim.lr_scheduler.OneCycleLR)
            # step 1
            self.manual_backward(loss)
            optimizer.first_step(zero_grad=True)
            # step 2
            logits = self.generic(images, labels)
            masks = self.generic.get_a(labels)
            loss_2 = self.generic.get_loss(logits, labels, masks)[0]
            self.manual_backward(loss_2)
            optimizer.second_step(zero_grad=True)
            sch.step()  # type: ignore

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
                "validation/loss": loss,
                "validation/ce": ce,
                "validation/mean": mean_mask,
                "validation/var": var,
                "validation/accuracy": self.accuracy,
            },
            sync_dist=True,
        )

    def on_test_epoch_start(self):
        self.noisy_masks_state = self.generic.noisy_masks
        self.generic.noisy_masks = "diagonal"

    def on_test_epoch_end(self):
        self.generic.noisy_masks = self.noisy_masks_state
        self.log_dict(
            {
                "AD 100%": torch.tensor(self.ADs[0]),
                "IC 100%": torch.tensor(self.ICs[0]),
                "AD 50%": torch.tensor(self.ADs[1]),
                "IC 50%": torch.tensor(self.ICs[1]),
                "AD 15%": torch.tensor(self.ADs[2]),
                "IC 15%": torch.tensor(self.ICs[2]),
                "ROAD 10%": torch.tensor(self.ROADs[0]),
                "ROAD 20%": torch.tensor(self.ROADs[1]),
                "ROAD 30%": torch.tensor(self.ROADs[2]),
                "ROAD 40%": torch.tensor(self.ROADs[3]),
                "ROAD 50%": torch.tensor(self.ROADs[4]),
                "ROAD 70%": torch.tensor(self.ROADs[5]),
                "ROAD 90%": torch.tensor(self.ROADs[6]),
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

        self.ADs, self.ICs = self.metric_AD_IC.get_results()
        self.ROADs = self.metric_ROAD.get_results()

    def configure_optimizers(self):
        optimizer = ut.get_optim(self.cfg, self.generic, self.use_sam)
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
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
