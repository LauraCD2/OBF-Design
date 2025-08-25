import torch
import numpy as np
import torch.nn as nn


class SpectralNet(nn.Module):
    def __init__(self, input_dim=100, num_classes=10, architecture=[10, 10]):
        super(SpectralNet, self).__init__()

        architecture.append(num_classes)
        self.architecture = architecture
        self.layers = nn.ModuleList()
        self.input_dim = input_dim

        current_dim = self.input_dim
        self.batch0 = nn.BatchNorm1d(self.input_dim, momentum=0.9, affine=False)

        for i, layer in enumerate(self.architecture):
            next_dim = layer
            if i == len(self.architecture) - 1:
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim))
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        # nn.BatchNorm1d(current_dim, momentum=0.1, affine=False),
                        nn.Linear(current_dim, next_dim, bias=True),
                        nn.ReLU(),
                        nn.Dropout(0.05)  # Increased dropout for better regularization
                    )
                )
                current_dim = next_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch0(x)
        for layer in self.layers:
            x = layer(x)
        return x
