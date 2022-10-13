import torchvision.models as models


def model_prep(model_name):
    model = models.__dict__[model_name](weights='IMAGENET1K_V1')
    return model
