import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst

import utilities as ut
from utilities import metrics


# define the LightningModule
class TAMELIT(pl.LightningModule):
    def __init__(self, cfg, num_classes, img_size, percent_list):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.generic = ut.get_model(cfg)
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.img_size = img_size
        self.metric_AD_IC = metrics.AD_IC(self.generic, img_size, percent_list=percent_list)
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
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str):
        train_loader, val_loader, test_loader = ut.data_loader(self.cfg)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
