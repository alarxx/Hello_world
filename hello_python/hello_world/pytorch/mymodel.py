import torch.nn as nn
import torch


class MyModel(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.lin = nn.Linear(nIn, 50)
        self.lin2 = nn.Linear(50, nOut)

    def forward(self, x):
        x = torch.relu(self.lin(x))
        x = torch.sigmoid(self.lin2(x))
        return x