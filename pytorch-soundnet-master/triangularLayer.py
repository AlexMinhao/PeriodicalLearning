from torch import nn
import torch
import math
from torch import optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F     # 激励函数都在这
class TriangularLayer(nn.Module):
    def __init__(self, in_features, bias=True):
        super(TriangularLayer, self).__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, 3)
        self.weight = nn.Parameter(torch.tril(torch.ones(in_features, in_features)))
        self.residual = nn.Sequential(
            nn.Conv2d(
                in_features,
                in_features,
                kernel_size=1,
                stride=(1, 1)),
            nn.BatchNorm2d(in_features),
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_features,1))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        res = x

        x = torch.einsum('jj,ijk->ijk', [self.weight, x])
        if self.bias is not None:
            x = x + self.bias
        # x = x + res
        x = F.relu(x)
        # x = x.view(-1, 5)
        # x = self.linear(x)
        # x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        # x = self.predict(x)  # 输出值
        return x


if __name__ == '__main__':
    model = TriangularLayer(5)
    loss_function = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=10e-3, momentum=0.9, weight_decay=0.9)

    x = torch.randn(10, 5, 1)*10
    y = torch.randn(10, 3)

    # x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    # y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

    for i in range(1000):
        out = model(x)
        loss = loss_function(out, y)
        # loss = Variable(loss, requires_grad=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(model.weight)

        print(loss)