import torch
import torch.nn as nn
import torchvision

class DogCatClassifier(nn.Module):
    def __init__(self):
        super(DogCatClassifier, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)  # 2 output classes: dog and cat

    def forward(self, x):
        return self.resnet(x)
