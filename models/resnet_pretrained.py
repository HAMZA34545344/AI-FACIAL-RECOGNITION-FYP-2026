import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class ResNetPretrained(nn.Module):
    def __init__(self, num_classes):
        super(ResNetPretrained, self).__init__()

        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)