from torchvision import models
from torchvision.models.feature_extraction import get_graph_node_names

mdl = models.resnet50(pretrained=True)


if __name__ == '__main__':
    train_names, eval_names = get_graph_node_names(mdl)
    print(train_names)
    # layers = "features.16 features.23 features.30"
    # model(layers)
