import torch.nn as nn
import torch
from lib.util import print_model_parameters
from lib.land_sea import land_mask


class simpleSpatailTimeNN(nn.Module):
    def __init__(self, n_cnn_layer: int = 1, kernals: list = [3], n_lstm_units: int = 64):
        super(simpleSpatailTimeNN, self).__init__()
        self.planes = 12
        self.conv1 = self.make_cnn_block()
        self.conv2 = self.make_cnn_block()
        self.conv3 = self.make_cnn_block()
        self.conv4 = self.make_cnn_block()
        self.pool = nn.AdaptiveAvgPool2d((1, 128))
        self.batch_norm = nn.BatchNorm1d(12, affine=False)
        # self.linear1 = nn.Linear(1728 * 4, n_lstm_units)
        self.lstm = nn.LSTM(1728 * 4, n_lstm_units, 1, bidirectional=True)
        self.drop = nn.Dropout(0.5)
        # self.linear2 = nn.Linear(12 * n_lstm_units, 24)
        self.linear3 = nn.Linear(128, 24)

    def make_cnn_block(self):
        return nn.ModuleList(
            [nn.Conv2d(12, self.planes, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(self.planes),
             nn.ReLU(inplace=True),

             nn.Conv2d(12, self.planes, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(self.planes),
             nn.ReLU(inplace=True),

             nn.Conv2d(12, self.planes, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(self.planes),
             nn.ReLU(inplace=True),
             ])

    def forward(self, sst, t300, ua, va, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        batch, month, h, w = sst.size()
        sea_mask = torch.as_tensor(~land_mask(), dtype=torch.float).repeat(batch, month, 1, 1).to(device)
        for conv1 in self.conv1:
            sst = conv1(sst)  # batch * 12 * (24 - 2) * (72 -2)
        for conv2 in self.conv2:
            t300 = conv2(t300)
        for conv3 in self.conv3:
            ua = conv3(ua)
        for conv4 in self.conv4:
            va = conv4(va)
        mask_sst = torch.mul(sst, sea_mask)
        mask_t300 = torch.mul(t300, sea_mask)
        sst = sst + mask_sst
        t300 = t300 + mask_t300

        sst = torch.flatten(sst, start_dim=2)  # batch * 12 * 1540
        t300 = torch.flatten(t300, start_dim=2)
        ua = torch.flatten(ua, start_dim=2)
        va = torch.flatten(va, start_dim=2)  # if flat, lstm input_dims = 1540 * 4

        x = torch.cat([sst, t300, ua, va], dim=-1)
        x = self.batch_norm(x)
        # x = self.linear1(x)
        x, _ = self.lstm(x)
        x = self.pool(x).squeeze(dim=-2)
        x = self.drop(x)
        x = self.linear3(x)
        return x


if __name__ == '__main__':
    devcie = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = simpleSpatailTimeNN().to(devcie)
    input1 = torch.rand(12, 12, 24, 72).to(devcie)
    input2 = torch.rand(12, 12, 24, 72).to(devcie)
    input3 = torch.rand(12, 12, 24, 72).to(devcie)
    input4 = torch.rand(12, 12, 24, 72).to(devcie)
    print_model_parameters(model)
    x = model(input1, input2, input3, input4)
    print(x.shape)
