import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.05):
        super(InceptionModule, self).__init__()
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

class Inception1D(nn.Module):
    def __init__(self, in_channels, num_classes, out_channels=32, dropout_rate=0.5):
        super(Inception1D, self).__init__()
        self.inception1 = InceptionModule(in_channels, out_channels)
        self.inception2 = InceptionModule(out_channels*4, out_channels*2)
        out_channels *= 2
        self.inception3 = InceptionModule(out_channels*4, out_channels*2)
        out_channels *= 2
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels*4, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)
