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


class GridEarthDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['sst']) - 36

    def __getitem__(self, idx):
        return self.data['sst'][idx:idx + 12], self.data['t300'][idx:idx + 12], self.data['ua'][idx:idx + 12], \
               self.data['va'][idx:idx + 12], self.data['label'][idx + 12:idx + 36], self.data['sst'][idx + 12:idx + 36]


def load_temporal_data(which_data='soda', train_num=960):
    if 'soda' in which_data.lower():
        train_sst = np.nan_to_num(np.load(args['soda_sst'])['sst'][:train_num + 24])
        train_t300 = np.nan_to_num(np.load(args['soda_t300'])['t300'][:train_num + 24])
        train_ua = np.nan_to_num(np.load(args['soda_ua'])['ua'][:train_num + 24])
        train_va = np.nan_to_num(np.load(args['soda_va'])['va'][:train_num + 24])
        train_label = np.nan_to_num(np.load(args['soda_label'])['nino'][0, :train_num + 24])
    else:
        train_sst = np.nan_to_num(np.load(args['cmip_sst'])['sst'])
        train_t300 = np.nan_to_num(np.load(args['cmip_t300'])['t300'])
        train_ua = np.nan_to_num(np.load(args['cmip_ua'])['ua'])
        train_va = np.nan_to_num(np.load(args['cmip_va'])['va'])
        train_label = np.nan_to_num(np.load(args['cmip_labels'])['nino'][0])

    val_sst = np.nan_to_num(np.load(args['soda_sst'])['sst'][train_num:])
    val_t300 = np.nan_to_num(np.load(args['soda_t300'])['t300'][train_num:])
    val_ua = np.nan_to_num(np.load(args['soda_ua'])['ua'][train_num:])
    val_va = np.nan_to_num(np.load(args['soda_va'])['va'][train_num:])
    val_label = np.nan_to_num(np.load(args['soda_label'])['nino'][0, train_num:])
    print('Train samples: {}, Valid samples: {}'.format(len(train_label) - 24, len(val_label) - 24))
    dict_train = {
        'sst': train_sst,
        't300': train_t300,
        'ua': train_ua,
        'va': train_va,
        'label': train_label}
    dict_valid = {
        'sst': val_sst,
        't300': val_t300,
        'ua': val_ua,
        'va': val_va,
        'label': val_label}
    train_dataset = GridEarthDataSet(dict_train)
    valid_dataset = GridEarthDataSet(dict_valid)
    return train_dataset, valid_dataset


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_dataset, valid_dataset = load_temporal_data('soda')
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'])

    for step, (sst, t300, ua, va, label, sst_label) in enumerate(train_loader):
        print(sst.shape, va.shape, label.shape, sst_label.shape)
