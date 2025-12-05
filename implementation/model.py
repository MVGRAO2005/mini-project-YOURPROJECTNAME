import torch
import torch.nn as nn
from torchvision import models

class ChestXrayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights="DEFAULT")
        self.model.classifier[1] = nn.Linear(1280, 2)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    print("Model initialized")
