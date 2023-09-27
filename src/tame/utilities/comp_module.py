from pathlib import Path
from typing import List

import lightning.pytorch as pl
import numpy
import torch
from pytorch_grad_cam import (
    AblationCAM,
    EigenCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    ScoreCAM,
    XGradCAM,
)
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst

from tame.masked_print import save_heatmap

from . import metrics


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# define the LightningModule
class COMPAREVIT(pl.LightningModule):
    def __init__(
        self,
        name: str = "gradcam",
        raw_model=None,
        img_size: int = 224,
        percent_list: List[float] = [0.0, 0.5, 0.85],
        num_classes: int = 1000,
        eval_length: str = "long",
        example_gen: bool = False,
    ):
        super().__init__()
        self.method = name
        cam_method = {
            "gradcam": GradCAM,
            "scorecam": ScoreCAM,
            "gradcam++": GradCAMPlusPlus,
            "ablationcam": AblationCAM,
            "xgradcam": XGradCAM,
            "eigencam": EigenCAM,
            "eigengradcam": EigenGradCAM,
            "layercam": LayerCAM,
            # "fullgrad": FullGrad,
        }
        if raw_model is None:
            self.raw_model = torch.hub.load(
                "facebookresearch/deit:main", "deit_tiny_patch16_224", pretrained=True
            )
        else:
            self.raw_model = raw_model
        target_layers = [self.raw_model.blocks[-1].norm1]  # type: ignore
        if name == "ablationcam":
            self.cam_model = cam_method[name](
                model=self.raw_model,
                target_layers=target_layers,
                use_cuda=self.on_gpu,
                reshape_transform=reshape_transform,
                ablation_layer=AblationLayerVit(),  # type: ignore
            )
        else:
            self.cam_model = cam_method[name](
                model=self.raw_model,
                target_layers=target_layers,  # type: ignore
                reshape_transform=reshape_transform if raw_model is None else None,  # type: ignore
                use_cuda=self.on_gpu,
            )
        if name == "scorecam" or name == "ablationcam":
            self.cam_model.batch_size = 8  # type: ignore
        self.img_size = img_size
        self.eval_length = eval_length
        self.metric_AD_IC = metrics.AD_IC(
            self.raw_model, img_size, percent_list=percent_list, legacy_mode=True
        )
        self.metric_ROAD = metrics.ROAD(self.raw_model, ROADMostRelevantFirst)

    def gen_explanation(self, dataloader, id):
        image, _ = dataloader.dataset[id]
        image = image.unsqueeze(0)
        mask = torch.tensor(self.cam_model(input_tensor=image))
        image = image.squeeze()
        save_heatmap(
            mask,
            image,
            Path(".") / "examples" / f"vit_{self.method}_{id}.jpg",
        )

    def on_test_epoch_end(self):
        self.ADs, self.ICs = self.metric_AD_IC.get_results()
        if self.eval_length == "long":
            self.ROADs = self.metric_ROAD.get_results()
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
        else:
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

    def on_test_model_eval(self, *args, **kwargs):
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def test_step(self, batch, batch_idx):
        with torch.inference_mode(False):
            images, labels = batch
            images = images.clone()
            labels = labels.clone()

            # this is the test loop
            logits = self.raw_model(images)
            logits = logits.softmax(dim=1)
            chosen_logits, model_truth = logits.max(dim=1)
            masks = self.cam_model(input_tensor=images)
            if numpy.isnan(masks).any():
                print("NaNs in masks")
                quit()
            masks = torch.tensor(masks).unsqueeze(dim=1).to(self.device)
            masks = metrics.normalizeMinMax(masks)

            self.metric_AD_IC(images, chosen_logits, model_truth, masks)
            masks = torch.nn.functional.interpolate(
                masks,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.eval_length == "long":
                masks = masks.squeeze().cpu().detach().numpy()
                self.metric_ROAD(images, model_truth, masks)
