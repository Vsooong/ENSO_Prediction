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


class CNN2_5(nn.Module):
    def __init__(self, planes=64, hidden_dim=48, embedding_dim=48):
        super(CNN2_5, self).__init__()
        self.planes = planes
        self.embedding = nn.Embedding(13, embedding_dim)
        self.conv1 = self.make_cnn_block()
        self.conv2 = nn.Conv3d(self.planes, self.planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.pool2 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
        self.attention1 = NONLocalBlock3D(self.planes)
        self.conv3 = nn.Conv3d(self.planes, self.planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.pool3 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
        self.attention2 = NONLocalBlock3D(self.planes)
        self.conv4 = nn.Conv3d(self.planes, self.planes, kernel_size=(3, 3, 3), stride=1)
        # self.linear = nn.Linear(self.planes * 64, 24)

        # self.batch_norm = nn.BatchNorm1d(12, affine=False)
        # self.lstm = nn.LSTM(3456, n_lstm_units, 1, bidirectional=True, batch_first=True)
        # self.pool_rnn = nn.AdaptiveAvgPool2d((1, self.planes * 2))
        self.drop = nn.Dropout(0.3)

        self.reset_gate_x = nn.Linear(4096, hidden_dim, bias=True)
        self.reset_gate_n = nn.Linear(embedding_dim, hidden_dim, bias=True)

        self.update_gate_x = nn.Linear(4096, hidden_dim, bias=True)
        self.update_gate_n = nn.Linear(embedding_dim, hidden_dim, bias=True)

        self.select_gate_x = nn.Linear(4096, hidden_dim, bias=True)
        self.select_gate_n = nn.Linear(embedding_dim, hidden_dim, bias=True)

        self.output_gate = nn.Linear(hidden_dim, 24, bias=True)

    def make_cnn_block(self):
        return nn.Sequential(
            SeparableConv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            SeparableConv2d(32, self.planes, 3, 1, 1),
            nn.BatchNorm2d(self.planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, sst, t300, ua, va, n=None):
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
        x = self.conv4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.drop(x)

        n = self.embedding(n)
        Z = nn.Sigmoid()(self.reset_gate_x(x) + self.reset_gate_n(n))
        R = nn.Sigmoid()(self.update_gate_x(x) + self.update_gate_n(n))
        n_tilda = torch.tanh(self.select_gate_x(x) + self.select_gate_n(R * n))
        H = Z * n + (1 - Z) * n_tilda
        x = self.output_gate(H)

        return x


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = CNN2_5().to(device)
    batch = 64
    input1 = torch.rand(batch, 12, 24, 72).to(device)
    input2 = torch.rand(batch, 12, 24, 72).to(device)
    input3 = torch.rand(batch, 12, 24, 72).to(device)
    input4 = torch.rand(batch, 12, 24, 72).to(device)
    input5 = torch.rand(batch, 1).long().to(device)
    # print_model_parameters(model)
    x = model(input1, input2, input3, input4, input5)
    print(x.shape)
