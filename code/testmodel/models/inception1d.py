import torch
from torch import optim, nn
import lightning as L
from utils import compute_loss, compute_metrics
import numpy as np
from .model_base import BaseModel

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

class Inception1D(BaseModel, L.LightningModule):
    def __init__(self, in_channels, num_classes, out_channels=32, dropout_rate=0.5, learning_rate=0.1):
        super(Inception1D, self).__init__(in_channels=in_channels, num_classes=num_classes,
                                          dropout_rate=dropout_rate, learning_rate=learning_rate)
        self.inception1 = InceptionModule(in_channels, out_channels, dropout_rate)
        self.inception2 = InceptionModule(out_channels*4, out_channels*2, dropout_rate)
        out_channels *= 2
        self.inception3 = InceptionModule(out_channels*4, out_channels*2, dropout_rate)
        out_channels *= 2
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels*4, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

        # Add intermediate classifiers for early exits
        self.exit1 = nn.Linear(128, num_classes)  # after inception1, output shape: [128, 128]
        self.exit2 = nn.Linear(256, num_classes)  # after inception2, output shape: [128, 256]
        self.exit3 = nn.Linear(512, num_classes)  # after inception3, output shape: [128, 512]

    def forward(self, x):
        if self.training:
            return self.forward_training(x)
        
        x = self.inception1(x)
        # print(f"Inception1 output shape: {x.shape}")  # Debug shape
        exit1 = self.exit1(x.mean(dim=2))  # Intermediate classifier after inception1
        if self.should_exit(exit1, self.entropy_threshold):
            return torch.sigmoid(exit1)

        x = self.inception2(x)
        # print(f"Inception2 output shape: {x.shape}")  # Debug shape
        exit2 = self.exit2(x.mean(dim=2))  # Intermediate classifier after inception2
        if self.should_exit(exit2, self.entropy_threshold):
            return torch.sigmoid(exit2)

        x = self.inception3(x)
        # print(f"Inception3 output shape: {x.shape}")  # Debug shape
        exit3 = self.exit3(x.mean(dim=2))  # Intermediate classifier after inception3
        if self.should_exit(exit3, self.entropy_threshold):
            return torch.sigmoid(exit3)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)
    
    def forward_training(self, x):
        x = self.inception1(x)
        # print(f"Inception1 output shape: {x.shape}")  # Debug shape
        exit1 = self.exit1(x.mean(dim=2))  # Intermediate classifier after inception1

        x = self.inception2(x)
        # print(f"Inception2 output shape: {x.shape}")  # Debug shape
        exit2 = self.exit2(x.mean(dim=2))  # Intermediate classifier after inception2

        x = self.inception3(x)
        # print(f"Inception3 output shape: {x.shape}")  # Debug shape
        exit3 = self.exit3(x.mean(dim=2))  # Intermediate classifier after inception3

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        final_output = self.fc(x)
        
        return torch.sigmoid(exit1), torch.sigmoid(exit2), torch.sigmoid(exit3), torch.sigmoid(final_output)

