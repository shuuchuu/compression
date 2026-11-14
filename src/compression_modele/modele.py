import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self, nb_classes: int = 6) -> None:
        super().__init__()
        self.caracteristiques = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.classifieur = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, nb_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifieur(self.caracteristiques(x))
