import torch
from torch import optim, nn
import lightning as L
from torchvision.models.resnet import conv1x1
from torchvision.models.resnet import conv3x3
from utils import compute_loss, compute_metrics
import numpy as np
from .model_base import BaseModel

class ResidualBlock(L.LightningModule):

    expansion: int = 1

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None, dropout_rate=0.05):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out

class BottleneckBlock(L.LightningModule):

    expansion: int = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.05):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out

class ResNet1D(BaseModel, L.LightningModule):
    def __init__(self, block, layers, num_classes=2, in_channels=12, dropout_rate=0.05, learning_rate=0.1):
        super().__init__(in_channels=in_channels, num_classes=num_classes,
                                        dropout_rate=dropout_rate, learning_rate=learning_rate)
        self.in_channels = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_rate=dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Add intermediate classifiers for early exits
        self.exit1 = nn.Linear(64 * block.expansion, num_classes)
        self.exit2 = nn.Linear(128 * block.expansion, num_classes)
        self.exit3 = nn.Linear(256 * block.expansion, num_classes)
        self.exit4 = nn.Linear(512 * block.expansion, num_classes)
        self.entropy_threshold = 0.5

    def _make_layer(self, block, out_channels, blocks, stride=1, dropout_rate=0.05):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample, dropout_rate=dropout_rate))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            return self.forward_training(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        exit1 = self.exit1(x.mean(dim=2))
        if self.should_exit(exit1, self.entropy_threshold):
            return torch.sigmoid(exit1)

        x = self.layer2(x)
        exit2 = self.exit2(x.mean(dim=2))
        if self.should_exit(exit2, self.entropy_threshold):
            return torch.sigmoid(exit2)

        x = self.layer3(x)
        exit3 = self.exit3(x.mean(dim=2))
        if self.should_exit(exit3, self.entropy_threshold):
            return torch.sigmoid(exit3)

        x = self.layer4(x)
        exit4 = self.exit4(x.mean(dim=2))
        if self.should_exit(exit4, self.entropy_threshold):
            return torch.sigmoid(exit4)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)
    
    def forward_training(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        exit1 = self.exit1(x.mean(dim=2))  # Intermediate classifier after layer1

        x = self.layer2(x)
        exit2 = self.exit2(x.mean(dim=2))  # Intermediate classifier after layer2

        x = self.layer3(x)
        exit3 = self.exit3(x.mean(dim=2))  # Intermediate classifier after layer3

        x = self.layer4(x)
        exit4 = self.exit4(x.mean(dim=2))  # Intermediate classifier after layer4

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        final_output = self.fc(x)

        return torch.sigmoid(exit1), torch.sigmoid(exit2), torch.sigmoid(exit3), torch.sigmoid(exit4), torch.sigmoid(final_output)


def resnet1d18(num_classes=2, in_channels=12, dropout_rate=0.05, learning_rate = 0.1):
    return ResNet1D(block=ResidualBlock, layers=[2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels, dropout_rate=dropout_rate, learning_rate=learning_rate)

def resnet1d34(num_classes=2, in_channels=12, dropout_rate=0.05, learning_rate = 0.1):
    return ResNet1D(block=ResidualBlock, layers=[3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels, dropout_rate=dropout_rate, learning_rate=learning_rate)

def resnet1d50(num_classes=2, in_channels=12, dropout_rate=0.05, learning_rate = 0.1):
    return ResNet1D(block=BottleneckBlock, layers=[3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels, dropout_rate=dropout_rate, learning_rate=learning_rate)
