import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, Dropout, ModuleList
import torch.optim as optim
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dims = [64, 16, 16], n_classes=9, input_size=1000, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.dims = [input_size, *dims]
        self.dropout = dropout_rate
        self.layers = ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(Sequential(
                Linear(self.dims[i], self.dims[i + 1]),
                ReLU(),
                Dropout(self.dropout)))
        self.last = Linear(dims[-1], n_classes)


    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return self.last(x)

def test_mlp():
    model = MLP(n_classes=5)
    model.eval()
    x = torch.rand(64)
    y = model(x)
    print(y)
    return

if __name__ == '__main__':
    test_mlp()