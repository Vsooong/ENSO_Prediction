import torch.nn as nn
import numpy as np
import torch
from lib.util import print_model_parameters
from lib.land_sea import land_mask


class simpleSpatailTimeNN_month(nn.Module):
    def __init__(self, n_cnn_layer: int = 1, kernals: list = [3], n_lstm_units: int = 64, hidden_dim: int = 48,
                 embedding_dim: int = 48):
        super(simpleSpatailTimeNN_month, self).__init__()
        self.planes = 12
        self.conv1 = self.make_cnn_block()
        self.conv2 = self.make_cnn_block()
        self.conv3 = self.make_cnn_block()
        self.conv4 = self.make_cnn_block()
        self.embedding = nn.Embedding(13, embedding_dim)
        self.batch_norm = nn.BatchNorm1d(48, affine=False)
        self.conv3d = nn.Conv3d(48, 48, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0))
        self.lstm = nn.LSTM(1728, n_lstm_units, 1, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 2 * n_lstm_units))
        self.drop = nn.Dropout(0.5)
        self.linear = nn.Linear(2 * n_lstm_units, 24)

        self.reset_gate_x = nn.Linear(2 * n_lstm_units, hidden_dim, bias=True)
        self.reset_gate_n = nn.Linear(embedding_dim, hidden_dim, bias=True)

        self.update_gate_x = nn.Linear(2 * n_lstm_units, hidden_dim, bias=True)
        self.update_gate_n = nn.Linear(embedding_dim, hidden_dim, bias=True)

        self.select_gate_x = nn.Linear(2 * n_lstm_units, hidden_dim, bias=True)
        self.select_gate_n = nn.Linear(embedding_dim, hidden_dim, bias=True)

        self.output_gate = nn.Linear(hidden_dim, 24, bias=True)

    def make_cnn_block(self):
        return nn.ModuleList(
            [nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(12),
             nn.ReLU(inplace=True),

             nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(12),
             nn.ReLU(inplace=True),

             nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(12),
             nn.ReLU(inplace=True),
             ])

    def forward(self, sst, t300, ua, va, n, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        for conv1 in self.conv1:
            sst = conv1(sst)  # batch * 12 * (24 - 2) * (72 -2)
        for conv2 in self.conv2:
            t300 = conv2(t300)
        for conv3 in self.conv3:
            ua = conv3(ua)
        for conv4 in self.conv4:
            va = conv4(va)
        batch, month, h, w = sst.size()

        sea_mask = torch.as_tensor(~land_mask(), dtype=torch.float).repeat(batch, month, 1, 1).to(device)
        mask_sst = torch.mul(sst, sea_mask)
        mask_t300 = torch.mul(t300, sea_mask)
        sst = sst + mask_sst
        t300 = t300 + mask_t300

        x = torch.cat([sst, t300, ua, va], dim=-3)
        N, C, H, W = x.size()
        groups = 12
        x = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W, 1)
        x = self.conv3d(x)
        x = torch.flatten(x, start_dim=2)
        x = self.batch_norm(x)
        x, _ = self.lstm(x)
        x = self.pool(x).squeeze(dim=-2)
        x = self.drop(x)

        n = self.embedding(n)
        Z = nn.Sigmoid()(self.reset_gate_x(x) + self.reset_gate_n(n))
        R = nn.Sigmoid()(self.update_gate_x(x) + self.update_gate_n(n))
        n_tilda = torch.tanh(self.select_gate_x(x) + self.select_gate_n(R * n))
        H = Z * n + (1 - Z) * n_tilda
        x = self.output_gate(H)

        return x


if __name__ == '__main__':
    devcie = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = simpleSpatailTimeNN_month().to(devcie)
    input1 = torch.rand(12, 12, 24, 72).to(devcie)
    input2 = torch.rand(12, 12, 24, 72).to(devcie)
    input3 = torch.rand(12, 12, 24, 72).to(devcie)
    input4 = torch.rand(12, 12, 24, 72).to(devcie)
    input5 = torch.rand(12, 1).long().to(devcie)
    # print_model_parameters(model)
    x = model(input1, input2, input3, input4, input5)
    print(x.shape)
