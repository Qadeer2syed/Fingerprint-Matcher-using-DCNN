import torch
import torch.nn as nn

class SiameseBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*16*16, 512), nn.ReLU(), nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = SiameseBranch()

    def forward(self, x1, x2):
        e1 = self.branch(x1)
        e2 = self.branch(x2)
        return e1, e2