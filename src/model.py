from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_model(pretrained=True):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
