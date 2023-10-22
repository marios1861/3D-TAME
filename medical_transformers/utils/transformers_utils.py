import math
import torch

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def download_weights(backbone_type, patch_size, pretrained_type):
    
    checkpoints_dino = {
        "deit_small" : {8: "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
                        16: "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"}, 
        "vit_base" : {8: "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth", 
                      16: "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"}
    }
    
    checkpoints_sup = {
        "deit_tiny" : {16: "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"}, 
        "deit_small" : {16: "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"},         
    }    
    
    checkpoints = {"supervised":checkpoints_sup, "dino":checkpoints_dino}
    
    try:
        chpt = torch.hub.load_state_dict_from_url(checkpoints[pretrained_type][backbone_type][patch_size], progress=True)
    except:
        raise ValueError(f"Pretrained weights for {backbone_type} with patch size {patch_size} with pretrained method {pretrained_type} not found.")
        
    if pretrained_type == "supervised":
        chpt = chpt["model"]
        if "head.weight" in chpt:            
            del chpt['head.weight']        
        if "head.bias" in chpt:
            del chpt['head.bias']
    return chpt        
    