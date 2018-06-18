import torch.nn as nn
import torch.nn.functional as F


class F_extractor(nn.Module):
    def __init__(self):
        super(F_extractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.linear = nn.Linear(6272, 3072)
        self.bn = nn.BatchNorm1d(3072)

    def forward(self, x):
        net = self.conv1(x)
        net = F.relu(net)
        net = F.max_pool2d(net, 3, 2)

        net = self.conv2(net)
        net = F.relu(net)
        net = F.max_pool2d(net, 3, 2)
        net = self.conv3(net)
        net = net.view(net.size(0), -1)
        net = self.linear(net)
        net = F.relu(net)
        net = self.bn(net)
        # net = F.batch_norm(net)

        return net


class F_label(nn.Module):
    def __init__(self):
        super(F_label, self).__init__()
        # self.extractor = extractor
        self.F = nn.Sequential(nn.Linear(3072, 2048),
                               nn.ReLU(),
                               nn.BatchNorm1d(2048),
                               nn.Linear(2048, 10),
                               nn.BatchNorm1d(10),
                               nn.Softmax(dim=-1))

    def forward(self, x):
        return self.F(x)