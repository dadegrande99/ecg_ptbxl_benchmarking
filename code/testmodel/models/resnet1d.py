import torch
from torch import nn
import lightning as L
import numpy as np
from .model_base import BaseModelEE

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
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
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

class ResNet1D(BaseModelEE, L.LightningModule):
    def __init__(self, block, layers, num_classes=2, in_channels=12, dropout_rate=0.05, learning_rate=0.1):
        super().__init__(in_channels=in_channels, num_classes=num_classes, dropout_rate=dropout_rate, learning_rate=learning_rate)
        self.in_channels = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.modules_EE.append(self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate))
        for i in range(1, len(layers)):
            self.modules_EE.append(self._make_layer(block, 64 * 2**i, layers[i], stride=2, dropout_rate=dropout_rate))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(64 * 2**(len(layers)-1) * block.expansion, num_classes)

        # Add intermediate classifiers for early exits
        self.exits = nn.ModuleList([nn.Linear(64 * 2**i * block.expansion, num_classes) for i in range(len(self.modules_EE))])
        self.entropy_threshold = 0.5
        self.weights_ee = [1.0] * (len(self.exits)+1)
        self.exits_used = [0] * (len(self.exits)+1)

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

    def forward_intro(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    
    def forward_final(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))

def resnet1d18(num_classes=2, in_channels=12, dropout_rate=0.05, learning_rate = 0.1):
    return ResNet1D(block=ResidualBlock, layers=[2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels, dropout_rate=dropout_rate, learning_rate=learning_rate) # type: ignore

def resnet1d34(num_classes=2, in_channels=12, dropout_rate=0.05, learning_rate = 0.1):
    return ResNet1D(block=ResidualBlock, layers=[3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels, dropout_rate=dropout_rate, learning_rate=learning_rate) # type: ignore

def resnet1d50(num_classes=2, in_channels=12, dropout_rate=0.05, learning_rate = 0.1):
    return ResNet1D(block=BottleneckBlock, layers=[3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels, dropout_rate=dropout_rate, learning_rate=learning_rate) # type: ignore
