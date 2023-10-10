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
from pytorch_grad_cam.ablation_layer import AblationLayer, AblationLayerVit
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst

from tame.masked_print import save_heatmap
from tame.transformer_explainability.baselines.ViT.ViT_explanation_generator import LRP
from tame.transformer_explainability.baselines.ViT.ViT_LRP import (
    vit_base_patch16_224 as vit_LRP,
)

from . import metrics


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def count_FP(self, input, output):
    self.fp_count += 1


# define the LightningModule
class COMPAREVIT(pl.LightningModule):
    def __init__(
        self,
        name: str = "gradcam",
        raw_model=None,
        mdl_name: str = "vit_b_16",
        img_size: int = 224,
        percent_list: List[float] = [0.0, 0.5, 0.85],
        eval_length: str = "long",
        example_gen: bool = False,
        once=True,
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
        self.raw_model.fp_count = 0
        self.raw_model.register_forward_hook(count_FP)
        if "vit" in mdl_name:
            target_layers = [self.raw_model.blocks[-1].norm1]  # type: ignore
        elif "vgg" in mdl_name:
            target_layers = [self.raw_model.features[29]]  # type: ignore
        elif "resnet" in mdl_name:
            target_layers = [self.raw_model.layer4[-1]]  # type: ignore
        else:
            raise ValueError("Model not supported")
        if name == "ablationcam":
            if "vit" in mdl_name:
                self.cam_model = cam_method[name](
                    model=self.raw_model,
                    target_layers=target_layers,
                    use_cuda=True,
                    reshape_transform=reshape_transform,
                    ablation_layer=AblationLayerVit(),  # type: ignore
                )
            else:
                self.cam_model = cam_method[name](
                    model=self.raw_model,
                    target_layers=target_layers,
                    use_cuda=True,
                    ablation_layer=AblationLayer(),  # type: ignore
                )
        else:
            self.cam_model = cam_method[name](
                model=self.raw_model,
                target_layers=target_layers,  # type: ignore
                reshape_transform=reshape_transform if raw_model is None else None,  # type: ignore
                use_cuda=True,
            )
        if name == "scorecam" or name == "ablationcam":
            self.cam_model.batch_size = 8  # type: ignore
        self.img_size = img_size
        self.eval_length = eval_length
        self.metric_AD_IC = metrics.AD_IC(
            self.raw_model, img_size, percent_list=percent_list, train_method="new"
        )
        self.metric_ROAD = metrics.ROAD(self.raw_model, ROADMostRelevantFirst)
        if once:
            self.once = True
        else:
            self.once = False

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
            if self.once:
                print(self.raw_model.fp_count)
            logits = logits.softmax(dim=1)
            chosen_logits, model_truth = logits.max(dim=1)
            masks = self.cam_model(input_tensor=images)
            if self.once:
                print(self.raw_model.fp_count)
                self.once = False
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


class HILAVIT(pl.LightningModule):
    def __init__(
        self,
        img_size: int = 224,
        percent_list: List[float] = [0.0, 0.5, 0.85],
        eval_length: str = "long",
        example_gen: bool = False,
    ):
        super().__init__()
        self.model = vit_LRP(pretrained=True)
        self.attribution_generator = LRP(self.model)
        self.model.eval()

        def gen_mask(image):
            transformer_attribution = self.attribution_generator.generate_LRP(
                image.cuda(),
                method="transformer_attribution",
            ).detach()
            transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
            transformer_attribution = torch.nn.functional.interpolate(
                transformer_attribution, scale_factor=16, mode="bilinear"
            )
            return transformer_attribution

        self.gen_mask = gen_mask
        self.model.fp_count = 0  # type: ignore
        self.model.register_forward_hook(count_FP)

        self.img_size = img_size
        self.eval_length = eval_length
        self.metric_AD_IC = metrics.AD_IC(
            self.model, img_size, percent_list=percent_list, train_method="new"
        )
        self.metric_ROAD = metrics.ROAD(self.model, ROADMostRelevantFirst)
        self.once = True

    def gen_explanation(self, dataloader, id):
        image, _ = dataloader.dataset[id]
        image = image.unsqueeze(0)
        mask = torch.tensor(self.gen_mask(image))
        image = image.squeeze()
        save_heatmap(
            mask,
            image,
            Path(".") / "examples" / f"vit_hila_{id}.jpg",
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
            logits = self.model(images)
            if self.once:
                print(self.model.fp_count)
            logits = logits.softmax(dim=1)
            chosen_logits, model_truth = logits.max(dim=1)
            masks = self.gen_mask(images)
            if self.once:
                print(self.model.fp_count)
                self.once = False
            masks = metrics.normalizeMinMax(masks)
            self.metric_AD_IC(images, chosen_logits, model_truth, masks)
            masks = torch.nn.functional.interpolate(
                masks,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.eval_length == "long":
                masks = masks.squeeze(0).cpu().detach().numpy()
                self.metric_ROAD(images, model_truth, masks)
