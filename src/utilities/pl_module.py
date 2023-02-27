import pytorch_lightning as pl
import torchmetrics

import utilities as ut


# define the LightningModule
class TAMELIT(pl.LightningModule):
    def __init__(self, cfg, max_iters, num_classes):
        super().__init__()
        self.cfg = cfg
        self.max_iters = max_iters
        self.generic = ut.get_model(cfg)
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

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

    def configure_optimizers(self):
        optimizer = ut.get_optim(self.cfg, self.generic)
        scheduler = ut.get_schedule(
            self.cfg, optimizer, self.max_iters, self.current_epoch
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
