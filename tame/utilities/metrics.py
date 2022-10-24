from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn import metrics
from torch.nn import functional as F


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


def normalizeMinMax(cam_map):
    cam_map_min, cam_map_max = torch.max(cam_map) - torch.min(cam_map)
    cam_map -= cam_map_min
    cam_map /= cam_map_max - cam_map_min
    return cam_map


def get_masked_inputs(
    inp: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.LongTensor,
    img_size: int,
    percent: List[float],
):
    B, _, H, W = masks.size()
    indexes = labels.expand(H, W, 1, B).permute(*range(masks.ndim - 1, -1, -1))
    masks = torch.gather(masks, 1, indexes)  # select masks
    masks = F.interpolate(
        masks, size=(img_size, img_size), mode="bilinear", align_corners=False
    )
    B, _, H, W = masks.size()

    def percent_gen(pc):
        masks.flatten(start_dim=1, end_dim=3).quantile(pc, dim=1).expand(
            H, W, 1, B
        ).permute(*range(masks.ndim - 1, -1, -1))

    masks_ls: List[torch.Tensor] = [
        masks.masked_fill(masks < percent_gen(pct), 0) for pct in percent
    ]
    x_masked_ls = [mask * inp for mask in masks_ls]
    return x_masked_ls


def normalize(tensor):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
        tensor
    )
    return normalize


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


def _to_numpy(v):
    v = torch.squeeze(v)
    if torch.is_tensor(v):
        v = v.cpu()
        v = v.numpy()
    elif isinstance(v, torch.autograd.Variable):
        v = v.cpu().data.numpy()
    return v


def get_AD(original_logits: torch.Tensor, new_logits: torch.Tensor) -> float:
    AD = ((original_logits - new_logits).clip(min=0) / original_logits).sum() * (
        100 / original_logits.size()[0]
    )
    return AD.item()


def get_IC(original_logits: torch.Tensor, new_logits: torch.Tensor) -> float:
    IC = (new_logits - original_logits).clip(min=0).sum() * (
        100 / original_logits.size()[0]
    )
    return IC.item()
