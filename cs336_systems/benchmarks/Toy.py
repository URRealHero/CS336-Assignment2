import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10,)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        print(f"Input dtype: {x.dtype}")
        x = self.fc1(x)
        print(f" After fc1: {x.dtype}")
        x = self.relu(x)
        print(f" After relu: {x.dtype}")
        x = self.ln(x)
        print(f" After ln: {x.dtype}")
        x = self.fc2(x)
        print(f" After fc2: {x.dtype}")
        return x
