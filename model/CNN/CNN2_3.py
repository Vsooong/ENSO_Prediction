import torch.nn as nn
import torch
from lib.util import print_model_parameters
from lib.land_sea import land_mask
from lib.non_local_em_gaussian import NONLocalBlock3D


# 通道分离卷积 来自Xception
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class simpleSpatailTimeNN(nn.Module):
    def __init__(self, hidden_planes=64):
        super(simpleSpatailTimeNN, self).__init__()
        self.planes = hidden_planes
        self.conv1 = self.make_cnn_block()
        self.conv2 = nn.Conv3d(self.planes, self.planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, dilation=1)
        self.attention1 = NONLocalBlock3D(self.planes)
        self.conv3 = nn.Conv3d(self.planes, self.planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, dilation=1)
        self.attention2 = NONLocalBlock3D(self.planes)
        self.conv4 = nn.Conv3d(self.planes, self.planes, kernel_size=(3, 3, 3), stride=1)
        self.linear = nn.Linear(self.planes * 64, 24)

        # self.batch_norm = nn.BatchNorm1d(12, affine=False)
        # self.lstm = nn.LSTM(3456, n_lstm_units, 1, bidirectional=True, batch_first=True)
        # self.pool_rnn = nn.AdaptiveAvgPool2d((1, self.planes * 2))
        self.drop = nn.Dropout(0.3)

    def make_cnn_block(self):
        return nn.Sequential(
            SeparableConv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            SeparableConv2d(32, self.planes, 3, 1, 1),
            nn.BatchNorm2d(self.planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, sst, t300, ua, va, month=None):
        input = torch.stack([sst, t300, ua, va], dim=2)
        batch, month, var, h, w = input.size()
        input = input.view(batch * month, var, h, w)
        input = self.conv1(input)
        _, c, h, w = input.size()
        input = input.view(batch, month, c, h, w)

        x = input.transpose(1, 2)
        x = self.pool2(self.conv2(x))
        x = self.attention1(x)
        x = self.pool3(self.conv3(x))
        x = self.attention2(x)
        # print(x.shape)
        x = self.conv4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.drop(x)
        x = self.linear(x)

        # x, _ = self.lstm(x)
        # x = self.pool_rnn(x).squeeze(dim=-2)
        # x = self.linear(x)
        return x


#
class simpleSpatailTimeNN2(nn.Module):
    def __init__(self, n_cnn_layer: int = 1, kernals: list = [3, 3], n_lstm_units: int = 64):
        super(simpleSpatailTimeNN2, self).__init__()
        self.conv1 = nn.ModuleList(
            [nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i, padding=1) for i in kernals])
        self.conv2 = nn.ModuleList(
            [nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i, padding=1) for i in kernals])
        self.conv3 = nn.ModuleList(
            [nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i, padding=1) for i in kernals])
        self.conv4 = nn.ModuleList(
            [nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i, padding=1) for i in kernals])
        self.pool1 = nn.AdaptiveAvgPool2d((22, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 70))
        self.pool3 = nn.AdaptiveAvgPool2d((1, 128))
        self.batch_norm = nn.BatchNorm1d(12, affine=False)
        self.lstm = nn.LSTM(1728 * 4, n_lstm_units, 2, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(128 + 12, 24)
        self.drop = nn.Dropout(0.3)

    def forward(self, sst, t300, ua, va, month=None, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        batch, monthes, h, w = sst.size()
        sea_mask = torch.as_tensor(~land_mask(), dtype=torch.float).repeat(batch, monthes, 1, 1).to(device)
        for conv1 in self.conv1:
            sst = conv1(sst)  # batch * 12 * (24 - 2) * (72 -2)
        for conv2 in self.conv2:
            t300 = conv2(t300)
        for conv3 in self.conv3:
            ua = conv3(ua)
        for conv4 in self.conv4:
            va = conv4(va)
        sst = torch.mul(sst, sea_mask)
        t300 = torch.mul(t300, sea_mask)
        # sst = sst + mask_sst
        # t300 = t300 + mask_t300

        sst = torch.flatten(sst, start_dim=2)  # batch * 12 * 1540
        t300 = torch.flatten(t300, start_dim=2)
        ua = torch.flatten(ua, start_dim=2)
        va = torch.flatten(va, start_dim=2)  # if flat, lstm input_dims = 1540 * 4

        x = torch.cat([sst, t300, ua, va], dim=-1)

        x = self.batch_norm(x)
        x, _ = self.lstm(x)
        x = self.pool3(x).squeeze(dim=-2)
        if month is not None:
            x = torch.cat([x, month / 12], dim=-1)

        x = self.drop(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = simpleSpatailTimeNN().to(device)
    batch = 64
    input1 = torch.rand(batch, 12, 24, 72).to(device)
    input2 = torch.rand(batch, 12, 24, 72).to(device)
    input3 = torch.rand(batch, 12, 24, 72).to(device)
    input4 = torch.rand(batch, 12, 24, 72).to(device)
    input5 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]).repeat(batch, 1).to(device)
    print_model_parameters(model)
    x = model(input1, input2, input3, input4, input5)
    print(x.shape)
