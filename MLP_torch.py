import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dims=0, n_classes=10, input_size=64, hidden=16):
        super(MLP, self).__init__()
        self.first = torch.nn.Linear(input_size,hidden)
        self.middle = torch.nn.ModuleList()
        for i in range(dims):
            self.middle.append(torch.nn.Linear(hidden,hidden))
        self.last = torch.nn.Linear(hidden,n_classes)

    def forward(self, x):
        x = F.relu(self.first(x))
        for m in self.middle:
            x = F.relu(m(x))
        x = self.last(x)
        return x

def test_mlp():
    model = MLP(n_classes=5)
    model.eval()
    x = torch.rand(64)
    y = model(x)
    print(y)
    return

if __name__ == '__main__':
    test_mlp()
 