import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F


class Sine(nn.Module):
    def forward(self, input):
        return torch.sin(input)


class BatchLinear(nn.Linear):
    def forward(self, input):
        # Handles input of shape [B, N, in_features] or [B, in_features]
        output = super().forward(input)
        return output


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, dropout_prob=0.2):
        super().__init__()
        layers = []

        # First layer
        layers.append(nn.Sequential(
            BatchLinear(in_features, hidden_features),
            Sine(),
            nn.Dropout(dropout_prob)

        ))

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Sequential(
                BatchLinear(hidden_features, hidden_features),
                Sine()
            ))

        # Final layer (no activation)
        layers.append(nn.Sequential(
            BatchLinear(hidden_features, out_features)
        ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SingleBVPNet(nn.Module):
    def __init__(self,
                 in_features=4,      # typically (x, t) or (x1, x2, ..., t)
                 out_features=1,     # V(x, t)
                 hidden_features=512,
                 num_hidden_layers=3,
                 dropout_prob=0.2,
                 mode='mlp',         # future support
                 type='sine'):       # future support
        super().__init__()
        self.net = FCBlock(in_features, out_features, num_hidden_layers, hidden_features)

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    model = SingleBVPNet()
    print(model)