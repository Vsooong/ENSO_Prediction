import torch
import numpy as np
import torch.nn as nn
from lib.util import print_model_parameters

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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
    def __init__(self):
        super(graphCNN, self).__init__()
        self.sst_conv = [self.make_conv_block()] * 12
        self.t300_conv = [self.make_conv_block()] * 12
        self.ua_conv = [self.make_conv_block()] * 12
        self.va_conv = [self.make_conv_block()] * 12
        self.gcn1 = [GraphConvolution(1280, 256).to(device)] * 12
        self.gcn2 = [GraphConvolution(256, 64).to(device)] * 12
        self.lstm1 = nn.LSTM(256, 64, 2, bidirectional=True)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 128))
        self.linear = nn.Linear(128, 24)

    def make_conv_block(self):
        return nn.ModuleList([nn.Conv2d(1, 8, kernel_size=1),
                              nn.BatchNorm2d(8),
                              nn.Conv2d(8, 32, kernel_size=3),
                              nn.BatchNorm2d(32),
                              nn.MaxPool2d(kernel_size=3, stride=3),
                              nn.Conv2d(32, 64, kernel_size=3),
                              nn.BatchNorm2d(64),
                              nn.MaxPool2d(kernel_size=2, stride=2),
                              nn.ReLU(inplace=True)]).to(device)

    def forward(self, sst, t300, ua, va, adj):
        out = None
        for i in range(12):
            x_sst = sst[:, i, :, :]
            x_sst = torch.reshape(x_sst, (x_sst.shape[0], 1, x_sst.shape[1], x_sst.shape[2]))
            x_t300 = t300[:, i, :, :]
            x_t300 = torch.reshape(x_t300, (x_t300.shape[0], 1, x_t300.shape[1], x_t300.shape[2]))
            x_ua = ua[:, i, :, :]
            x_ua = torch.reshape(x_ua, (x_ua.shape[0], 1, x_ua.shape[1], x_ua.shape[2]))
            x_va = va[:, i, :, :]
            x_va = torch.reshape(x_va, (x_va.shape[0], 1, x_va.shape[1], x_va.shape[2]))
            for conv in self.sst_conv[i]:
                x_sst = conv(x_sst)
            for conv in self.t300_conv[i]:
                x_t300 = conv(x_t300)
            for conv in self.ua_conv[i]:
                x_ua = conv(x_ua)
            for conv in self.va_conv[i]:
                x_va = conv(x_va)
            x_sst = torch.reshape(x_sst, (sst.shape[0], 1, -1))
            x_t300 = torch.reshape(x_t300, (sst.shape[0], 1, -1))
            x_ua = torch.reshape(x_ua, (sst.shape[0], 1, -1))
            x_va = torch.reshape(x_va, (sst.shape[0], 1, -1))
            x = torch.cat([x_sst, x_t300, x_ua, x_va], dim=-2)
            x = self.gcn1[i](adj, x)
            x = self.gcn2[i](adj, x)
            if out is None:
                out = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
            else:
                out = torch.cat([out, torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))], dim=1)
        out = torch.reshape(out, (out.shape[0], out.shape[1], -1))
        out, _ = self.lstm1(out)
        out = self.pool3(out).squeeze(dim=-2)
        out = self.linear(out)
        return out
