import torch.nn as nn
import torch
from lib.util import print_model_parameters


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, 2),
                norm_layer(planes),
            )
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class simpleCNN(nn.Module):
    def __init__(self, n_lstm_units: int = 64):
        super(simpleCNN, self).__init__()
        self.planes = 48
        self.conv1 = self.make_cnn_block()
        self.conv2 = self.make_cnn_block()
        self.conv3 = self.make_cnn_block()
        self.conv4 = self.make_cnn_block()
        self.layer = BasicBlock(self.planes * 4, self.planes * 4, stride=2)
        # self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.pool = nn.MaxPool2d(3)
        self.batch_norm = nn.BatchNorm1d(576, affine=False)
        self.linear = nn.Linear(576, 24)

    def make_cnn_block(self):
        return nn.ModuleList(
            [nn.Conv2d(12, self.planes, kernel_size=4, stride=2, padding=1),
             nn.BatchNorm2d(self.planes),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(2)
             ])

    def forward(self, sst, t300, ua, va):
        for conv1 in self.conv1:
            sst = conv1(sst)  # batch * 12 * (24 - 2) * (72 -2)
        for conv2 in self.conv2:
            t300 = conv2(t300)
        for conv3 in self.conv3:
            ua = conv3(ua)
        for conv4 in self.conv4:
            va = conv4(va)

        x = torch.cat([sst, t300, ua, va], dim=1)
        x = self.layer(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(self.batch_norm(x))
        return x


class simpleSpatailTimeNN(nn.Module):
    def __init__(self, n_cnn_layer: int = 1, kernals: list = [3], n_lstm_units: int = 64):
        super(simpleSpatailTimeNN, self).__init__()
        self.planes = 12
        self.conv1 = self.make_cnn_block()
        self.conv2 = self.make_cnn_block()
        self.conv3 = self.make_cnn_block()
        self.conv4 = self.make_cnn_block()
        self.pool3 = nn.AdaptiveAvgPool2d((1, 128))
        self.batch_norm = nn.BatchNorm1d(12, affine=False)
        self.lstm = nn.LSTM(432 * 4, n_lstm_units, 2, bidirectional=True)
        self.linear = nn.Linear(128, 24)

    def make_cnn_block(self):
        return nn.ModuleList(
            [nn.Conv2d(12, self.planes, kernel_size=4, stride=2, padding=1),
             nn.BatchNorm2d(self.planes),
             nn.ReLU(inplace=True),
             ])

    def forward(self, sst, t300, ua, va):
        for conv1 in self.conv1:
            sst = conv1(sst)  # batch * 12 * (24 - 2) * (72 -2)
        for conv2 in self.conv2:
            t300 = conv2(t300)
        for conv3 in self.conv3:
            ua = conv3(ua)
        for conv4 in self.conv4:
            va = conv4(va)
        sst = torch.flatten(sst, start_dim=2)  # batch * 12 * 1540
        t300 = torch.flatten(t300, start_dim=2)
        ua = torch.flatten(ua, start_dim=2)
        va = torch.flatten(va, start_dim=2)  # if flat, lstm input_dims = 1540 * 4

        x = torch.cat([sst, t300, ua, va], dim=-1)
        x = self.batch_norm(x)
        x, _ = self.lstm(x)
        x = self.pool3(x).squeeze(dim=-2)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    devcie = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = simpleSpatailTimeNN().to(devcie)
    input1 = torch.rand(64, 12, 24, 72).to(devcie)
    input2 = torch.rand(64, 12, 24, 72).to(devcie)
    input3 = torch.rand(64, 12, 24, 72).to(devcie)
    input4 = torch.rand(64, 12, 24, 72).to(devcie)
    print_model_parameters(model)
    x = model(input1, input2, input3, input4)
    print(x.shape)
