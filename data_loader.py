import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from configs import args
import torch
from lib.land_sea import get_flat_lon_lat, land_mask


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
        return len(self.data['sst']) - 38

    def __getitem__(self, idx):
        return self.data['sst'][idx:idx + 12], self.data['t300'][idx:idx + 12], self.data['ua'][idx:idx + 12], \
               self.data['va'][idx:idx + 12], self.data['label'][idx + 12:idx + 36], \
               self.data['month'][idx:idx + 12], self.data['sst'][idx + 12:idx + 38]


class GraphDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['sst']) - 36

    def __getitem__(self, idx):
        return self.data['sst'][idx:idx + 12], self.data['t300'][idx:idx + 12], self.data['ua'][idx:idx + 12], \
               self.data['va'][idx:idx + 12], self.data['lon'][idx:idx + 12], self.data['lat'][idx:idx + 12], \
               self.data['label'][idx + 12:idx + 36]


def load_data():
    train = xr.open_dataset(args['cmip_data'])
    label = xr.open_dataset(args['cmip_label'])
    train_sst = train['sst'][:, :12].values
    train_t300 = train['t300'][:, :12].values
    train_ua = train['ua'][:, :12].values
    train_va = train['va'][:, :12].values
    train_label = label['nino'][:, 12:36].values

    train_ua = np.nan_to_num(train_ua)
    train_va = np.nan_to_num(train_va)
    train_t300 = np.nan_to_num(train_t300)
    train_sst = np.nan_to_num(train_sst)

    train2 = xr.open_dataset(args['sota_data'])
    label2 = xr.open_dataset(args['sota_label'])
    train_sst2 = train2['sst'][:, :12].values
    train_t3002 = train2['t300'][:, :12].values
    train_ua2 = train2['ua'][:, :12].values
    train_va2 = train2['va'][:, :12].values
    train_label2 = label2['nino'][:, 12:36].values
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


def load_train_data(which_data='soda', split_num=600, which_num=0):
    if 'soda' in which_data.lower():
        train_sst = np.nan_to_num(np.load(args['soda_sst'])['sst'][:split_num + 36])
        train_t300 = np.nan_to_num(np.load(args['soda_t300'])['t300'][:split_num + 36])
        train_ua = np.nan_to_num(np.load(args['soda_ua'])['ua'][:split_num + 36])
        train_va = np.nan_to_num(np.load(args['soda_va'])['va'][:split_num + 36])
        train_label = np.nan_to_num(np.load(args['soda_label'])['nino'][0, :split_num + 36])
        train_month = np.arange(0, split_num + 36) % 12 + 1
    else:
        train_sst = np.nan_to_num(np.load(args['cmip_sst'])['sst'][which_num])
        train_t300 = np.nan_to_num(np.load(args['cmip_t300'])['t300'][which_num])
        train_ua = np.nan_to_num(np.load(args['cmip_ua'])['ua'][which_num])
        train_va = np.nan_to_num(np.load(args['cmip_va'])['va'][which_num])
        train_label = np.nan_to_num(np.load(args['cmip_labels'])['nino'][which_num])
        train_month = np.arange(0, len(train_sst)) % 12 + 1

    # print('Train samples: {}'.format(len(train_label) - 36))
    dict_train = {
        'sst': train_sst,
        't300': train_t300,
        'ua': train_ua,
        'va': train_va,
        'month': train_month,
        'label': train_label}

    train_dataset = GridEarthDataSet(dict_train)
    return train_dataset


def load_val_data(which_data='soda', split_num=600, which_num=0):
    if 'soda' in which_data.lower():
        val_sst = np.nan_to_num(np.load(args['soda_sst'])['sst'][split_num:])
        val_t300 = np.nan_to_num(np.load(args['soda_t300'])['t300'][split_num:])
        val_ua = np.nan_to_num(np.load(args['soda_ua'])['ua'][split_num:])
        val_va = np.nan_to_num(np.load(args['soda_va'])['va'][split_num:])
        val_label = np.nan_to_num(np.load(args['soda_label'])['nino'][0, split_num:])
        val_month = np.arange(split_num, split_num + len(val_sst)) % 12 + 1
    else:
        val_sst = np.nan_to_num(np.load(args['cmip_sst'])['sst'][which_num])
        val_t300 = np.nan_to_num(np.load(args['cmip_t300'])['t300'][which_num])
        val_ua = np.nan_to_num(np.load(args['cmip_ua'])['ua'][which_num])
        val_va = np.nan_to_num(np.load(args['cmip_va'])['va'][which_num])
        val_label = np.nan_to_num(np.load(args['cmip_labels'])['nino'][which_num])
        val_month = np.arange(0, len(val_sst)) % 12 + 1

    # print('Valid samples: {}'.format(len(val_label) - 36))
    dict_valid = {
        'sst': val_sst,
        't300': val_t300,
        'ua': val_ua,
        'va': val_va,
        'month': val_month,
        'label': val_label}
    valid_dataset = GridEarthDataSet(dict_valid)
    return valid_dataset


def load_graph_data(which_data='soda', split_num=960, which_num=0, mode='train'):
    mask = land_mask()
    if 'soda' in which_data.lower():
        soda = True
        sst = args['soda_sst']
        t300 = args['soda_t300']
        ua = args['soda_ua']
        va = args['soda_va']
        label = args['soda_label']
        train_sst = np.nan_to_num(np.load(sst)['sst'])
        months, h, w = train_sst.shape
    else:
        soda = False
        sst = args['cmip_sst']
        t300 = args['cmip_t300']
        ua = args['cmip_ua']
        va = args['cmip_va']
        label = args['cmip_labels']
        train_sst = np.nan_to_num(np.load(sst)['sst'])
        model_nums, months, h, w = train_sst.shape

    if soda:
        month_range = np.arange(0, months)
        if mode == 'train':
            month_range = month_range[:split_num + 36]
            months = split_num + 36
        else:
            month_range = month_range[split_num:]
            months = months - split_num
    else:
        month_range = which_num

    def mask_tensor(input, month):
        input[:, mask] = np.nan
        input = torch.flatten(input)
        input = input[~torch.isnan(input)]
        input = torch.reshape(input, shape=(month, -1))
        return input

    train_sst = torch.as_tensor(train_sst[month_range], dtype=torch.float)
    train_sst = mask_tensor(train_sst, months)

    train_t300 = np.nan_to_num(np.load(t300)['t300'][month_range])
    train_t300 = torch.as_tensor(train_t300, dtype=torch.float)
    train_t300 = mask_tensor(train_t300, months)

    train_ua = np.nan_to_num(np.load(ua)['ua'][month_range])
    train_ua = torch.as_tensor(train_ua, dtype=torch.float)
    train_ua = mask_tensor(train_ua, months)

    train_va = np.nan_to_num(np.load(va)['va'][month_range])
    train_va = torch.as_tensor(train_va, dtype=torch.float)
    train_va = mask_tensor(train_va, months)

    train_label = torch.as_tensor(np.nan_to_num(np.load(label)['nino']), dtype=torch.float)
    train_label = train_label.squeeze(0)[month_range]

    # torch.set_printoptions(threshold=10_000)
    # nino_area = (10 <= lon_grid[0]) & (60 >= lon_grid[0]) & (-5 <= lat_grid[0]) & (5 >= lat_grid[0])
    # assert torch.sum(nino_area) == 33
    # print(nino_area)
    lon_grid, lat_grid = get_flat_lon_lat(months)

    dict_train = {
        'sst': train_sst / 1,
        't300': train_t300 / 1,
        'ua': train_ua / 5,
        'va': train_va / 5,
        'lon': lon_grid / 180,
        'lat': lat_grid / 60,
        'label': train_label}
    train_dataset = GraphDataSet(dict_train)
    return train_dataset


if __name__ == '__main__':
    train_dataset = load_graph_data('soda')
    # from torch.utils.data import DataLoader
    # start_time = time.time()
    # train_dataset = load_train_data('cmip')
    # valid_dataset = load_val_data('soda')
    # train_loader = DataLoader(train_dataset, batch_size=args['batch_size'])
    # print(time.time() - start_time)
    # for step, (sst, t300, ua, va, label, sst_label) in enumerate(train_loader):
    #     print(sst.shape, va.shape, label.shape, sst_label.shape)
