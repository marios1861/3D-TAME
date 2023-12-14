from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
import torchmetrics
import torchshow as ts
import torchvision.transforms as transforms
from lightning.pytorch.loggers import WandbLogger
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
        epochs: int,
        model: Optional[torch.nn.Module] = None,
        input_dim: Optional[torch.Size] = None,
        stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        attention_version: str = "TAME",
        masking: Literal["random", "diagonal", "max"] = "random",
        normalized_data=True,
        train_method: Literal[
            "new", "renormalize", "raw_normalize", "layernorm", "batchnorm"
        ] = "new",
        optimizer_type: Literal["Adam", "AdamW", "RMSProp", "SGD", "OLDSGD"] = "SGD",
        use_sam: bool = False,
        momentum: float = 0.9,
        weight_decay: float = 5.0e-4,
        schedule_type: Literal["equal", "new_classic", "old_classic"] = "equal",
        lr: float = 1.0e-3,
        img_size: Union[int, List[int]] = 224,
        percent_list: List[float] = [0.0, 0.5, 0.85],
        num_classes: int = 1000,
        eval_length: Literal["long", "short"] = "long",
    ):
        super().__init__()
        if optimizer_type == "OLDSGD":
            schedule_type = "old_classic"
            assert normalized_data and train_method != "raw_normalize"
        self.save_hyperparameters(ignore=["model", "stats"])
        if model is None:
            self.generic = ut.get_model(
                model_name=model_name,
                layers=layers,
                version=attention_version,
                masking=masking,
                train_method=train_method,
            )
        else:
            self.generic = ut.get_new_model(
                model,
                input_dim,
                model_name=model_name,
                layers=layers,
                version=attention_version,
                masking=masking,
                train_method=train_method,
                num_classes=num_classes,
            )
            img_size = list(input_dim[-3:]) if input_dim else img_size
        self.attention_version = attention_version
        # threshold is 0 because we use un-normalized logits to save on computation time
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, threshold=0
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, threshold=0
        )
        self.img_size = img_size
        self.eval_length: Literal["long", "short"] = eval_length
        self.metric_AD_IC = metrics.AD_IC(
            self.generic,
            img_size,
            normalized_data=normalized_data,
            percent_list=percent_list,
            stats=stats,
        )
        self.metric_ROAD = metrics.ROAD(self.generic, ROADMostRelevantFirst)
        self.generic.requires_grad_(False)
        self.generic.attn_mech.requires_grad_()
        self.use_sam = use_sam
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer_type: Literal[
            "Adam", "AdamW", "RMSProp", "SGD", "OLDSGD"
        ] = optimizer_type
        self.schedule_type: Literal[
            "equal", "new_classic", "old_classic"
        ] = schedule_type
        self.lr = lr
        self.epochs = epochs
        if use_sam:
            self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        images, _ = batch
        logits, labels = self.generic(images)
        
        masks = self.generic.get_a(labels)
        (
            loss,
            ce,
            mean_mask,
            var,
        ) = self.generic.get_loss(logits, labels, masks)
        self.train_accuracy(logits, labels)
        self.log_dict(
            {
                "training/Loss": loss,
                "training/CE": ce,
                "training/Mean": mean_mask,
                "training/Var": var,
                "training/Accuracy": self.train_accuracy,
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
        images, _ = batch
        logits = self.generic(images)
        labels = logits.argmax(dim=1)
        masks = self.generic.get_a(labels)
        (
            loss,
            ce,
            mean_mask,
            var,
        ) = self.generic.get_loss(logits, labels, masks)
        self.val_accuracy(logits, labels)
        self.log_dict(
            {
                "validation/loss": loss,
                "validation/ce": ce,
                "validation/mean": mean_mask,
                "validation/var": var,
                "validation/accuracy": self.val_accuracy,
            },
            sync_dist=True,
        )

    def on_test_epoch_start(self):
        self.masking_state: Literal["random", "max", "diagonal"] = self.generic.masking
        self.generic.masking = "diagonal"

    def on_test_epoch_end(self):
        self.ADs, self.ICs = self.metric_AD_IC.get_results()
        self.generic.masking = self.masking_state
        if self.eval_length == "long":
            self.ROAD = self.metric_ROAD.get_results()
            if type(self.logger) is WandbLogger:
                columns_adic = [
                    "AD 100%",
                    "IC 100%",
                    "AD 50%",
                    "IC 50%",
                    "AD 15%",
                    "IC 15%",
                ]
                columns_road = ["area"]
                data = [
                    [
                        self.ADs[0],
                        self.ICs[0],
                        self.ADs[1],
                        self.ICs[1],
                        self.ADs[2],
                        self.ICs[2],
                    ]
                ]
                data_road = [[self.ROAD]]
                self.logger.log_table(key="ADIC", columns=columns_adic, data=data)
                self.logger.log_table(key="ROAD", columns=columns_road, data=data_road)
            else:
                self.log_dict(
                    {
                        "AD 100%": torch.tensor(self.ADs[0]),
                        "IC 100%": torch.tensor(self.ICs[0]),
                        "AD 50%": torch.tensor(self.ADs[1]),
                        "IC 50%": torch.tensor(self.ICs[1]),
                        "AD 15%": torch.tensor(self.ADs[2]),
                        "IC 15%": torch.tensor(self.ICs[2]),
                        "ROAD": torch.tensor(self.ROAD),
                    }
                )
        else:
            if type(self.logger) is WandbLogger:
                columns_adic = [
                    "AD 100%",
                    "IC 100%",
                    "AD 50%",
                    "IC 50%",
                    "AD 15%",
                    "IC 15%",
                ]
                data = [
                    [
                        self.ADs[0],
                        self.ICs[0],
                        self.ADs[1],
                        self.ICs[1],
                        self.ADs[2],
                        self.ICs[2],
                    ]
                ]
                self.logger.log_table(key="ADIC", columns=columns_adic, data=data)
            self.log_dict(
                {
                    "AD 100%": torch.tensor(self.ADs[0]),
                    "IC 100%": torch.tensor(self.ICs[0]),
                    "AD 50%": torch.tensor(self.ADs[1]),
                    "IC 50%": torch.tensor(self.ICs[1]),
                    "AD 15%": torch.tensor(self.ADs[2]),
                    "IC 15%": torch.tensor(self.ICs[2]),
                }
            )

    @torch.no_grad()
    def save_masked_image(self, image, id, model_name):
        self.generic.masking = "diagonal"
        self.generic.eval()
        image = image.unsqueeze(0).to(self.device)
        ts.save(image, f"_torchshow/{model_name}/image{id}.png")
        logits = self.generic(image)
        logits = logits.softmax(dim=1)
        chosen_logits, model_truth = logits.max(dim=1)
        mask = self.generic.get_c(model_truth)
        mask = metrics.normalizeMinMax(mask)
        ts.save(mask, f"_torchshow/{model_name}/small_mask{id}.png")
        mask = torch.nn.functional.interpolate(
            mask,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        ts.save(mask, f"_torchshow/{model_name}/big_mask{id}.png")
        ts.save(mask * image, f"_torchshow/{model_name}/masked_image{id}.png")
        opencvImage = cv2.cvtColor(
            image.squeeze().permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR
        )
        opencvImage = (np.asarray(opencvImage, np.float32) * 255).astype(np.uint8)
        np_mask = np.array(mask.squeeze().cpu().numpy() * 255, dtype=np.uint8)
        np_mask = cv2.applyColorMap(np_mask, cv2.COLORMAP_JET)
        mask_image = cv2.addWeighted(np_mask, 0.5, opencvImage, 0.5, 0)
        cv2.imwrite(f"_torchshow/{model_name}/cv_masked_image{id}.png", mask_image)
        
    @torch.no_grad()
    def get_3dmask(self, image):
        self.generic.masking = "diagonal"
        self.generic.eval()
        image = image.unsqueeze(0).to(self.device)
        logits = self.generic(image)
        logits = logits.softmax(dim=1)
        chosen_logits, model_truth = logits.max(dim=1)
        mask = self.generic.get_c(model_truth)
        mask = metrics.normalizeMinMax(mask)
        mask = torch.nn.functional.interpolate(
            mask,
            size=image.shape[-3:],
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        return mask

    def test_step(self, batch, batch_idx):
        # this is the test loop
        images, _ = batch
        logits = self.generic(images)
        logits = logits.softmax(dim=1)
        chosen_logits, model_truth = logits.max(dim=1)
        masks = self.generic.get_c(model_truth)
        masks = metrics.normalizeMinMax(masks)
        self.metric_AD_IC(images, chosen_logits, model_truth, masks)
        if self.eval_length == "long":
            if masks.ndim == 4:
                masks = torch.nn.functional.interpolate(
                    masks,
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                masks = torch.nn.functional.interpolate(
                    masks,
                    size=self.img_size,
                    mode="trilinear",
                    align_corners=False,
                )
            masks = masks.squeeze().cpu().detach().numpy()
            self.metric_ROAD(images, model_truth, masks)

    def configure_optimizers(self):
        optimizer = ut.get_optim(
            self.generic,
            use_sam=self.use_sam,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            optimizer_type=self.optimizer_type,
        )
        # the total steps are divided between num_nodes
        scheduler = ut.get_schedule(
            optimizer,
            self.current_epoch,
            total_steps=int(self.trainer.estimated_stepping_batches),
            schedule_type=self.schedule_type,
            lr=self.lr,
            epochs=self.epochs,
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
        normalize: bool = True,
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
        self.normalize = normalize

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
        if not self.normalize:
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
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        if not self.normalize:
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
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        if not self.normalize:
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
        self.test_dataset = dataset_test

        test_loader = DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader
