import math
from dataclasses import dataclass, field
from typing import List, Tuple, Type, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pytorch_grad_cam.metrics.road import ROADLeastRelevantFirst, ROADMostRelevantFirst
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
from sklearn import metrics
from torch.nn import functional as F

from .avg_meter import AverageMeter


@dataclass
class AD_IC:
    model: torch.nn.Module
    img_size: int
    percent_list: List[float] = field(default_factory=lambda: [0.0, 0.5, 0.85])
    noisy_masks: bool = False

    def __post_init__(self):
        self.chosen_logits_list = []
        self.new_logits_list = []

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        chosen_logits: torch.Tensor,
        model_truth: torch.Tensor,
        masks: torch.Tensor,
    ):
        masked_images_list = get_masked_inputs(
            images,
            masks,
            self.img_size,
            self.percent_list,
            self.noisy_masks,
        )
        new_logits_list = [
            new_logits.softmax(dim=1).gather(1, model_truth.unsqueeze(-1)).squeeze()
            for new_logits in [
                self.model(masked_images) for masked_images in masked_images_list
            ]
        ]
        self.chosen_logits_list.append(chosen_logits.cpu())
        if self.new_logits_list == []:
            self.new_logits_list = [[new_logits.cpu()] for new_logits in new_logits_list]
        else:
            for new_logits, old_new_logits in zip(new_logits_list, self.new_logits_list):
                old_new_logits.append(new_logits.cpu())

    def get_results(self) -> Tuple[List[float], List[float]]:
        metrics = []
        chosen_logits = torch.cat(self.chosen_logits_list)
        for new_logits in self.new_logits_list:
            metrics.append(
                (
                    get_AD(chosen_logits, torch.cat(new_logits)).item(),
                    get_IC(chosen_logits, torch.cat(new_logits)).item(),
                )
            )
        return [metric[0] for metric in metrics], [metric[1] for metric in metrics]


@dataclass
class ROAD:
    model: torch.nn.Module
    road: Union[
        Type[ROADMostRelevantFirst],
        Type[ROADLeastRelevantFirst],
    ]
    percent_list: List[int] = field(
        default_factory=lambda: [100 - pct for pct in [10, 20, 30, 40, 50, 70, 90]]
    )

    def __post_init__(self):
        self.target = None
        self.metric: List[AverageMeter] = []
        self.roads: List[Union[ROADMostRelevantFirst, ROADLeastRelevantFirst]] = []
        for percentile in self.percent_list:
            self.metric.append(AverageMeter(type="avg"))
            self.roads.append(self.road(percentile))

    @torch.no_grad()
    def __call__(
        self,
        input: torch.Tensor,
        model_truth: torch.Tensor,
        masks: np.ndarray,
    ):
        if self.target is None:
            self.target = [RawScoresOutputTarget() for _ in range(input.size(0))]
        for i in range(len(self.percent_list)):
            scores = self.roads[i](
                input, masks, self.target, self.model, return_diff=False  # type: ignore
            )
            self._updateAcc(model_truth.cpu(), torch.tensor(scores), i)

    def _updateAcc(self, preds: torch.Tensor, scores: torch.Tensor, i: int):
        _, new_preds = scores.max(dim=1)
        self.metric[i].update(((preds == new_preds).sum() / preds.shape[0]).item())

    def get_results(self) -> List[float]:
        return [metric() for metric in self.metric]


def drop_Npercent(cam_map, percent):
    # Select N percent of pixels
    if percent == 0:
        return cam_map

    N, C, H, W = cam_map.size()
    cam_map_tmp = cam_map
    f = torch.flatten(cam_map)
    value = int(H * W * percent)
    # print(value)
    m = torch.kthvalue(f, value)
    cam_map_tmp[cam_map_tmp < m.values] = 0
    num_pixels = math.ceil((1 - percent) * (H * W))
    k = torch.count_nonzero(cam_map_tmp > 0) - num_pixels
    k = math.floor(k)
    if k >= 1:
        indices = torch.nonzero(cam_map == m.values)
        for pi in range(0, int(k)):
            cam_map_tmp[
                indices[pi][0], indices[pi][1], indices[pi][2], indices[pi][3]
            ] = 0
    cam_map = cam_map_tmp
    #   cam_map[cam_map!=0] = 1
    # k = torch.count_nonzero(cam_map_tmp>0) - num_pixels
    #    print(k)
    return cam_map


def show_cam_on_image(
    img,
    mask,
    use_rgb: bool = True,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    mask = mask.cpu().data
    mask = mask.numpy()
    mask = mask[0, 0, :, :]

    img = img.cpu().data.numpy()
    img = img[0, :, :, :]
    img = np.transpose(img, (1, 2, 0))

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)  # type: ignore
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255  # type: ignore

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)  # type: ignore


def normalizeWithMax(Att_map):
    x1_max = torch.max(Att_map, dim=3, keepdim=True)[0].max(2, keepdim=True)[0].detach()
    Att_map = (Att_map) / (x1_max)  # values now in [0,1]
    return Att_map


def normalizeMinMax4Dtensor(Att_map):
    x1_min = torch.min(Att_map, dim=3, keepdim=True)[0].min(2, keepdim=True)[0].detach()
    x1_max = torch.max(Att_map, dim=3, keepdim=True)[0].max(2, keepdim=True)[0].detach()
    Att_map = (Att_map - x1_min) / (x1_max - x1_min)  # values now in [0,1]
    return Att_map


def normalizeMinMax(cam_maps: torch.Tensor) -> torch.Tensor:
    # min reduces the dimension, we want to reduce B, C, H, W to B so we need
    # to apply min 3 times to the last dimension
    cam_map_mins = cam_maps
    cam_map_maxs = cam_maps
    for _ in range(0, 3):
        cam_map_mins, _ = cam_map_mins.min(dim=-1)
        cam_map_maxs, _ = cam_map_maxs.max(dim=-1)
    # now both of these tensors are 1-dimensional holding the min and the max
    # of each batch
    cam_maps -= cam_map_mins.view(-1, 1, 1, 1)
    cam_maps /= (cam_map_maxs - cam_map_mins).view(-1, 1, 1, 1)
    return cam_maps


def get_masked_inputs(
    inp: torch.Tensor,
    masks: torch.Tensor,
    img_size: int,
    percent: List[float],
    noisy_masks: bool = True,
) -> List[torch.Tensor]:
    B, C, _, _ = masks.size()
    if noisy_masks:
        assert C == B
        masks = masks.diagonal().permute(2, 0, 1).unsqueeze(1)
    _, C, _, _ = masks.size()
    assert C == 1

    masks = F.interpolate(
        masks, size=(img_size, img_size), mode="bilinear", align_corners=False
    )
    _, _, H, W = masks.size()
    masks = masks.float()

    def percent_gen(pc: float) -> torch.Tensor:
        return (
            masks.flatten(start_dim=1, end_dim=3)
            .quantile(pc, dim=1)
            .expand(H, W, C, B)
            .permute(*range(masks.ndim - 1, -1, -1))
        )
    masks_ls = [masks.masked_fill(masks < percent_gen(pct), 0) for pct in percent]
    x_masked_ls = [mask * inp for mask in masks_ls]
    return x_masked_ls


def normalize(tensor):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
        tensor
    )
    return normalize


@torch.no_grad()
def accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Compute the top k accuracy of classification results.
    :param target: the ground truth label
    :param topk: tuple or list of the expected k values.
    :return: A list of the accuracy values. The list has the same lenght with para: topk
    """
    maxk = max(topk)
    batch_size = target.size(0)
    scores = logits
    _, pred = scores.topk(maxk, 1, True, True)
    pred: torch.Tensor = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_AUC(gt_labels, pred_scores):
    res = metrics.roc_auc_score(gt_labels, pred_scores)
    return res


def get_AD(original_logits: torch.Tensor, new_logits: torch.Tensor) -> torch.Tensor:
    AD = ((original_logits - new_logits).clip(min=0) / original_logits).sum() * (
        100 / original_logits.size()[0]
    )
    return AD


def get_IC(original_logits: torch.Tensor, new_logits: torch.Tensor) -> torch.Tensor:
    IC = ((new_logits - original_logits) > 0).sum() * (100 / original_logits.size()[0])
    return IC
