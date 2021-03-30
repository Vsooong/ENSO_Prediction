import torch.nn as nn
import torch
from lib.util import print_model_parameters, nino_index
from lib.land_sea import land_mask


class lstmNN(nn.Module):
    def __init__(self, hidden_unit=64):
        super(lstmNN, self).__init__()
        self.lstm1 = nn.GRU(24 * 72, hidden_unit, batch_first=True)
        self.lstm2 = nn.GRU(24 * 72, hidden_unit, batch_first=True)
        self.lstm3 = nn.GRU(24 * 72, hidden_unit, batch_first=True)
        self.lstm4 = nn.GRU(24 * 72, hidden_unit, batch_first=True)
        self.nino_lstm = nn.GRU(2, hidden_unit, batch_first=True)
        self.fc1 = nn.Linear(hidden_unit, 24)
        self.Wg = nn.Linear(hidden_unit * 5, hidden_unit)
        self.Wi = nn.Linear(hidden_unit * 2, hidden_unit)
        self.Wf = nn.Linear(hidden_unit * 5, hidden_unit)
        self.drop = nn.Dropout(0.5)

    def forward(self, sst, t300, ua, va, month=None):
        ninos = nino_index(sst, shrink_month=False)
        months = month / 12
        base_feature = torch.stack([ninos, months], dim=2)
        _, base_feature = self.nino_lstm(base_feature)
        base_feature = base_feature[0]

        sst = torch.flatten(sst, start_dim=2)
        t300 = torch.flatten(t300, start_dim=2)
        ua = torch.flatten(ua, start_dim=2)
        va = torch.flatten(va, start_dim=2)
        _, sst = self.lstm1(sst)
        _, t300 = self.lstm2(t300)
        _, ua = self.lstm3(ua)
        _, va = self.lstm4(va)
        fuse_x = torch.cat([sst[0], t300[0], ua[0], va[0], base_feature], dim=1)
        image_x = torch.cat([sst[0], t300[0]], dim=1)
        z = torch.sigmoid(self.Wg(fuse_x))
        i = torch.tanh(self.Wi(image_x))
        f = torch.tanh(self.Wf(fuse_x))
        x = (1 - z) * i + z * f
        # print(x.shape)
        x = self.drop(x)
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = lstmNN().to(device)
    batch = 32
    input1 = torch.rand(batch, 12, 24, 72).to(device)
    input2 = torch.rand(batch, 12, 24, 72).to(device)
    input3 = torch.rand(batch, 12, 24, 72).to(device)
    input4 = torch.rand(batch, 12, 24, 72).to(device)
    input5 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]).repeat(batch, 1).to(device)
    print_model_parameters(model)
    x = model(input1, input2, input3, input4, input5)
    print(x.shape)
