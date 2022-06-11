import torch
import torch.nn as nn


class Sequence(nn.Module):

    def __init__(self, nIn, hidden, nOut, device="cpu"):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.hidden = hidden
        self.device = device

        self.lstm = nn.LSTMCell(nIn, hidden)
        self.linear = nn.Linear(hidden, nOut)

        self.h_t = torch.zeros([1, hidden], device=device)
        self.c_t = torch.zeros([1, hidden], device=device)

    def forward(self, x):
        outputs = []

        for i, input_t in enumerate(x):
            self.h_t, self.c_t = self.lstm(input_t, (self.h_t, self.c_t))
            output = torch.sigmoid(self.linear(self.h_t))
            outputs.append(output)

        outputs = torch.stack(outputs, 1)
        outputs = outputs.reshape(-1, 1, 3)

        return outputs

    def update(self):
        self.h_t = torch.zeros([1, self.hidden], device=self.device)
        self.c_t = torch.zeros([1, self.hidden], device=self.device)