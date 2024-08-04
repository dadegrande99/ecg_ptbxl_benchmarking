import torch
from torch import optim, nn
import lightning as L
from utils import compute_loss, compute_metrics
import numpy as np
from .model_base import BaseModelEE

class InceptionModule(L.LightningModule):
    def __init__(self, in_channels, out_channels, dropout_rate=0.05):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(self.pool(x))
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class Inception1D(BaseModelEE, L.LightningModule):
    def __init__(self, in_channels, num_classes, out_channels=32, dropout_rate=0.5, learning_rate=0.1):
        super(Inception1D, self).__init__(in_channels=in_channels, num_classes=num_classes,
                                          dropout_rate=dropout_rate, learning_rate=learning_rate)
        self.modules_EE.append(InceptionModule(in_channels, out_channels, dropout_rate))
        self.modules_EE.append(InceptionModule(out_channels*4, out_channels*2, dropout_rate))
        out_channels *= 2
        self.modules_EE.append(InceptionModule(out_channels*4, out_channels*2, dropout_rate))
        out_channels *= 2
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels*4, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

        # Add intermediate classifiers for early exits
        base = 7
        self.exits = nn.ModuleList([nn.Linear(2**i, num_classes) for i in range(base, base+len(self.modules_EE))])  
        # after inception1, output shape: [128, 128]
        # after inception2, output shape: [128, 256]
        # after inception3, output shape: [128, 512]

        self.weights_ee = [1.0] * (len(self.exits)+1)

    
    def forward_intro(self, x):
        return x

    
    def forward_final(self, x):
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))
    