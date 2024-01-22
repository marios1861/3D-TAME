from pathlib import Path
from typing import List, Optional, Union

import lightning.pytorch as pl
import numpy
import torch
from lightning.pytorch.loggers import WandbLogger
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
from pytorch_grad_cam.metrics.road import ROADLeastRelevantFirst, ROADMostRelevantFirst

from tame.masked_print import save_heatmap
from tame.transformer_explainability.baselines.ViT.ViT_explanation_generator import LRP
from tame.transformer_explainability.baselines.ViT.ViT_LRP import (
    vit_base_patch16_224 as vit_LRP,
)

from . import metrics


def on_test_epoch_end(ADs, ICs, logger, ROADs=None, ROADs2=None):
    if type(logger) is WandbLogger:
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
                ADs[0],
                ICs[0],
                ADs[1],
                ICs[1],
                ADs[2],
                ICs[2],
            ]
        ]
        logger.log_table(key="ADIC", columns=columns_adic, data=data)
        if ROADs is not None:
            columns_road = [
                "ROAD 10%",
                "ROAD 20%",
                "ROAD 30%",
                "ROAD 40%",
                "ROAD 50%",
                "ROAD 70%",
                "ROAD 90%",
            ]
            data_road = [ROADs]
            logger.log_table(key="ROAD", columns=columns_road, data=data_road)
            if ROADs2 is not None:
                logger.log_table(key="ROAD2", columns=columns_road, data=[ROADs2])

    else:
        logger.log_dict(
            {
                "AD 100%": torch.tensor(ADs[0]),
                "IC 100%": torch.tensor(ICs[0]),
                "AD 50%": torch.tensor(ADs[1]),
                "IC 50%": torch.tensor(ICs[1]),
                "AD 15%": torch.tensor(ADs[2]),
                "IC 15%": torch.tensor(ICs[2]),
            }
        )
        if ROADs is not None:
            logger.log_dict(
                {
                    "ROAD 10%": torch.tensor(ROADs[0]),
                    "ROAD 20%": torch.tensor(ROADs[1]),
                    "ROAD 30%": torch.tensor(ROADs[2]),
                    "ROAD 40%": torch.tensor(ROADs[3]),
                    "ROAD 50%": torch.tensor(ROADs[4]),
                    "ROAD 70%": torch.tensor(ROADs[5]),
                    "ROAD 90%": torch.tensor(ROADs[6]),
                }
            )
            if ROADs2 is not None:
                logger.log_dict(
                    {
                        "ROAD2 10%": torch.tensor(ROADs2[0]),
                        "ROAD2 20%": torch.tensor(ROADs2[1]),
                        "ROAD2 30%": torch.tensor(ROADs2[2]),
                        "ROAD2 40%": torch.tensor(ROADs2[3]),
                        "ROAD2 50%": torch.tensor(ROADs2[4]),
                        "ROAD2 70%": torch.tensor(ROADs2[5]),
                        "ROAD2 90%": torch.tensor(ROADs2[6]),
                    }
                )




def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def count_FP(self, input, output):
    self.fp_count += 1


# define the LightningModule
class CompareModel(pl.LightningModule):
    def __init__(
        self,
        name: str = "gradcam",
        raw_model=None,
        target_layers=None,
        stats=None,
        normalized_data=True,
        mdl_name: str = "vit_b_16",
        input_dim: Optional[torch.Size] = None,
        img_size: Union[int, List[int]] = 224,
        percent_list: List[float] = [0.0, 0.5, 0.85],
        eval_length: str = "long",
        count_fp=True,
        LeRF=False,
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
            import timm

            self.raw_model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
        else:
            self.raw_model = raw_model
        self.raw_model.fp_count = 0
        self.raw_model.register_forward_hook(count_FP)
        if target_layers is not None:
            pass
        elif "vit" in mdl_name:
            target_layers = [self.raw_model.blocks[-1].norm1]  # type: ignore
        elif "vgg" in mdl_name:
            target_layers = [self.raw_model.features[29]]  # type: ignore
        elif "resnet" in mdl_name:
            target_layers = [self.raw_model.layer4[-1]]  # type: ignore
        else:
            raise ValueError(
                "Model not supported by default and target_layers not specified"
            )
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
                reshape_transform=reshape_transform if raw_model is None or "vit" in mdl_name else None,  # type: ignore
                use_cuda=True,
            )
        if name == "scorecam" or name == "ablationcam":
            self.cam_model.batch_size = 8  # type: ignore
        if input_dim:
            self.img_size = list(input_dim[-3:])
        else:
            self.img_size = img_size
        self.eval_length = eval_length
        self.metric_AD_IC = metrics.AD_IC(
            self.raw_model,
            self.img_size,
            percent_list=percent_list,
            normalized_data=normalized_data,
            stats=stats,
        )
        self.metric_ROAD = metrics.ROAD(self.raw_model, ROADMostRelevantFirst)
        if LeRF:
            self.LeRF = True
            self.metric_ROAD2 = metrics.ROAD(self.raw_model, ROADLeastRelevantFirst)
        self.count_fp = count_fp

    def get_3dmask(self, image):
        with torch.set_grad_enabled(True):
            image = image.unsqueeze(0)
            image.requires_grad = True
            print(image.requires_grad)
            mask = self.cam_model(input_tensor=image)
            return mask

    def on_test_epoch_end(self):
        ADs, ICs = self.metric_AD_IC.get_results()
        ROADs = None
        ROADs2 = None
        if self.eval_length == "long":
            ROADs = self.metric_ROAD.get_results()
            if self.LeRF is True:
                ROADs2 = self.metric_ROAD2.get_results()
        on_test_epoch_end(ADs, ICs, self.logger, ROADs, ROADs2)

    def on_test_model_eval(self, *args, **kwargs):
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def test_step(self, batch, batch_idx):
        with torch.inference_mode(False):
            images, _ = batch
            images = images.clone()

            # this is the test loop
            logits = self.raw_model(images)
            if self.count_fp:
                print(self.raw_model.fp_count)
            logits = logits.softmax(dim=1)
            chosen_logits, model_truth = logits.max(dim=1)
            masks = self.cam_model(input_tensor=images)
            if self.count_fp:
                print(self.raw_model.fp_count)
                self.count_fp = False
            if numpy.isnan(masks).any():
                print("NaNs in masks")
                quit()
            masks = torch.tensor(masks).unsqueeze(dim=1).to(self.device)
            masks = metrics.normalizeMinMax(masks)

            self.metric_AD_IC(images, chosen_logits, model_truth, masks)
            if self.eval_length == "long":
                masks = torch.nn.functional.interpolate(
                    masks,
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                    align_corners=False,
                )
                masks = masks.squeeze().cpu().detach().numpy()
                self.metric_ROAD(images, model_truth, masks)
                if self.LeRF:
                    self.metric_ROAD2(images, model_truth, masks)


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
            self.model, img_size, percent_list=percent_list, normalized_data=True
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
                    "ROAD": torch.tensor(self.ROAD),
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
