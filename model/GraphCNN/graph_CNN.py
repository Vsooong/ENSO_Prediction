import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, adj, features):
        out = None
        for i in range(features.shape[0]):
            if out is None:
                out = torch.reshape(self.linear(torch.mm(adj, features[i])), (1, 4, -1))
            else:
                out = torch.cat([out, torch.reshape(self.linear(torch.mm(adj, features[i])), (1, 4, -1))], dim=0)
        return out


class graphCNN(nn.Module):
    def __init__(self, n_cnn_layer: int = 1, kernals: list = [3], n_lstm_units: int = 64):
        super(graphCNN, self).__init__()
        self.planes = 12
        self.conv1 = self.make_cnn_block()
        self.conv2 = self.make_cnn_block()
        self.conv3 = self.make_cnn_block()
        self.conv4 = self.make_cnn_block()
        self.graph1 = GraphConvolution(5184, 1024)
        self.graph2 = GraphConvolution(1024, 256)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 128))
        self.batch_norm = nn.BatchNorm1d(12, affine=False)
        self.lstm = nn.LSTM(256, n_lstm_units, 2, bidirectional=True)
        self.linear = nn.Linear(128, 24)

    def make_cnn_block(self):
        return nn.ModuleList(
            [nn.Conv2d(12, self.planes, kernel_size=4, stride=2, padding=1),
             nn.BatchNorm2d(self.planes),
             nn.ReLU(inplace=True),
             ])

    def forward(self, sst, t300, ua, va, adj):
        for conv1 in self.conv1:
            sst = conv1(sst)  # batch * 12 * (24 - 2) * (72 -2)
        for conv2 in self.conv2:
            t300 = conv2(t300)
        for conv3 in self.conv3:
            ua = conv3(ua)
        for conv4 in self.conv4:
            va = conv4(va)
        sst = torch.reshape(sst, (sst.shape[0], 1, -1))  # batch * 12 * 1540
        t300 = torch.reshape(sst, (t300.shape[0], 1, -1))
        ua = torch.reshape(sst, (ua.shape[0], 1, -1))
        va = torch.reshape(sst, (va.shape[0], 1, -1))  # if flat, lstm input_dims = 1540 * 4
        x = torch.cat([sst, t300, ua, va], dim=-2)
        x = self.graph1(adj, x)
        x = self.graph2(adj, x)
        x, _ = self.lstm(x)
        x = self.pool3(x).squeeze(dim=-2)
        x = self.linear(x)
        return x
