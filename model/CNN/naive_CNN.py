import torch.nn as nn
import torch
from lib.util import print_model_parameters


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
