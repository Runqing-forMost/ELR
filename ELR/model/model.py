import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .ResNet_Zoo import ResNet, BasicBlock
from .CNN import CNN


# def resnet34(num_classes=10):
#     return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
#


def cnn(num_classes=10):
    return CNN(n_outputs=num_classes)

