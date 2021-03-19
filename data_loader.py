import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from configs import args


def load_data():
    train = xr.open_dataset(args['sota_data'])
    label = xr.open_dataset(args['sota_label'])
    train_sst = train['sst'][:80, :12].values
    train_t300 = train['t300'][:80, :12].values
    train_ua = train['ua'][:80, :12].values
    train_va = train['va'][:80, :12].values
    train_label = label['nino'][:80, 12:36].values

    train_ua = np.nan_to_num(train_ua)
    train_va = np.nan_to_num(train_va)
    train_t300 = np.nan_to_num(train_t300)
    train_sst = np.nan_to_num(train_sst)

    train2 = xr.open_dataset(args['sota_data'])
    label2 = xr.open_dataset(args['sota_label'])
    train_sst2 = train2['sst'][80:, :12].values
    train_t3002 = train2['t300'][80:, :12].values
    train_ua2 = train2['ua'][80:, :12].values
    train_va2 = train2['va'][80:, :12].values
    train_label2 = label2['nino'][80:, 12:36].values
    print('Train samples: {}, Valid samples: {}'.format(len(train_label), len(train_label2)))
    dict_train = {
        'sst': train_sst,
        't300': train_t300,
        'ua': train_ua,
        'va': train_va,
        'label': train_label}
    dict_valid = {
        'sst': train_sst2,
        't300': train_t3002,
        'ua': train_ua2,
        'va': train_va2,
        'label': train_label2}
    train_dataset = EarthDataSet(dict_train)
    valid_dataset = EarthDataSet(dict_valid)
    return train_dataset, valid_dataset


class EarthDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['sst'])

    def __getitem__(self, idx):
        return (self.data['sst'][idx], self.data['t300'][idx], self.data['ua'][idx], self.data['va'][idx]), \
               self.data['label'][idx]


if __name__ == '__main__':
    train = xr.open_dataset(args['cmip_data'])
    sst = train['va'][-50:].values
