from typing import Any, Dict
from torchvision import models
from torch import nn
from torch import optim
from .composite_models import Generic

def get_model(cfg: Dict[str, Any]) -> Generic:
    mdl = model_prep(cfg['model'])
    mdl = Generic(mdl, cfg['layers'].split(), cfg['version'].version)
    mdl.cuda()
    return mdl

def model_prep(model_name: str) -> nn.Module:
    model = models.__dict__[model_name](weights='IMAGENET1K_V1')
    return model

def get_optim(cfg: Dict[str, Any], model: Generic) -> optim.Optimizer:
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.attn_mech.modules():
        for p_name, p in v.named_parameters(recurse=False):
            if p_name == 'bias':  # bias (no decay)
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:  # weight (with decay)
                g[0].append(p)
    if cfg['optimizer'] == 'Adam':
        optimizer = optim.Adam(g[2], lr=0.1, betas=(cfg['momentum'], 0.999))  # adjust beta1 to momentum
    elif cfg['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(g[2], lr=0.1, betas=(cfg['momentum'], 0.999), weight_decay=0.0)
    elif cfg['optimizer'] == 'RMSProp':
        optimizer = optim.RMSprop(g[2], lr=0.1, momentum=cfg['momentum'])
    elif cfg['optimizer'] == 'SGD':
        optimizer = optim.SGD(g[2], lr=0.1, momentum=cfg['momentum'], nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {cfg["optimizer"]} not implemented.')
    
    optimizer.add_param_group({'params': g[0], 'weight_decay': cfg['decay']})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer