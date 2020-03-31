import torch
import torch.nn as nn

class NeuralTensor(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, act='tanh'):
        super().__init__()
        self.bilinear = nn.Bilinear(in1_features, in2_features, out_features, bias=True)
        self.linear = nn.Linear(in1_features + in2_features, out_features, bias=False)

    def forward(self, x1, x2):
        return self.bilinear(x1, x2) + self.linear(torch.cat((x1, x2), -1))
