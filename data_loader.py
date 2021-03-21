import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from configs import args
import torch
from global_land_mask import globe
import time


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


class GraphDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['sst']) - 36

    def __getitem__(self, idx):
        return self.data['sst'][idx:idx + 12], self.data['t300'][idx:idx + 12], self.data['ua'][idx:idx + 12], \
               self.data['va'][idx:idx + 12], self.data['lon'][idx:idx + 12], self.data['lat'][idx:idx + 12], \
               self.data['label'][idx + 12:idx + 36], self.data['sst'][idx + 12:idx + 36]


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


def load_train_data(which_data='soda', split_num=960, which_num=0):
    if 'soda' in which_data.lower():
        train_sst = np.nan_to_num(np.load(args['soda_sst'])['sst'][:split_num + 36])
        train_t300 = np.nan_to_num(np.load(args['soda_t300'])['t300'][:split_num + 36])
        train_ua = np.nan_to_num(np.load(args['soda_ua'])['ua'][:split_num + 36])
        train_va = np.nan_to_num(np.load(args['soda_va'])['va'][:split_num + 36])
        train_label = np.nan_to_num(np.load(args['soda_label'])['nino'][0, :split_num + 36])

    else:
        train_sst = np.nan_to_num(np.load(args['cmip_sst'])['sst'][which_num])
        train_t300 = np.nan_to_num(np.load(args['cmip_t300'])['t300'][which_num])
        train_ua = np.nan_to_num(np.load(args['cmip_ua'])['ua'][which_num])
        train_va = np.nan_to_num(np.load(args['cmip_va'])['va'][which_num])
        train_label = np.nan_to_num(np.load(args['cmip_labels'])['nino'][which_num])

    print('Train samples: {}'.format(len(train_label) - 36))
    dict_train = {
        'sst': train_sst,
        't300': train_t300,
        'ua': train_ua,
        'va': train_va,
        'label': train_label}

    train_dataset = GridEarthDataSet(dict_train)
    return train_dataset


def load_val_data(which_data='soda', split_num=960, which_num=0):
    if 'soda' in which_data.lower():
        val_sst = np.nan_to_num(np.load(args['soda_sst'])['sst'][split_num:])
        val_t300 = np.nan_to_num(np.load(args['soda_t300'])['t300'][split_num:])
        val_ua = np.nan_to_num(np.load(args['soda_ua'])['ua'][split_num:])
        val_va = np.nan_to_num(np.load(args['soda_va'])['va'][split_num:])
        val_label = np.nan_to_num(np.load(args['soda_label'])['nino'][0, split_num:])
    else:
        val_sst = np.nan_to_num(np.load(args['cmip_sst'])['sst'][which_num])
        val_t300 = np.nan_to_num(np.load(args['cmip_t300'])['t300'][which_num])
        val_ua = np.nan_to_num(np.load(args['cmip_ua'])['ua'][which_num])
        val_va = np.nan_to_num(np.load(args['cmip_va'])['va'][which_num])
        val_label = np.nan_to_num(np.load(args['cmip_labels'])['nino'][which_num])
    print('Valid samples: {}'.format(len(val_label) - 36))
    dict_valid = {
        'sst': val_sst,
        't300': val_t300,
        'ua': val_ua,
        'va': val_va,
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

    # start = time.time()
    train_sst = train_sst[month_range]
    train_sst[:, mask] = np.nan
    train_sst = torch.flatten(torch.as_tensor(train_sst, dtype=torch.float))
    train_sst = train_sst[~torch.isnan(train_sst)]
    train_sst = torch.reshape(train_sst, shape=(months, -1))

    train_t300 = np.nan_to_num(np.load(t300)['t300'][month_range])
    train_t300[:, mask] = np.nan
    train_t300 = torch.flatten(torch.as_tensor(train_t300, dtype=torch.float))
    train_t300 = train_t300[~torch.isnan(train_t300)]
    train_t300 = torch.reshape(train_t300, shape=(months, -1))

    train_ua = np.nan_to_num(np.load(ua)['ua'][month_range])
    train_ua[:, mask] = np.nan
    train_ua = torch.flatten(torch.as_tensor(train_ua, dtype=torch.float))
    train_ua = train_ua[~torch.isnan(train_ua)]
    train_ua = torch.reshape(train_ua, shape=(months, -1))

    train_va = np.nan_to_num(np.load(va)['va'][month_range])
    train_va[:, mask] = np.nan
    train_va = torch.flatten(torch.as_tensor(train_va, dtype=torch.float))
    train_va = train_va[~torch.isnan(train_va)]
    train_va = torch.reshape(train_va, shape=(months, -1))

    train_label = torch.as_tensor(np.nan_to_num(np.load(label)['nino']), dtype=torch.float)
    train_label = train_label.squeeze(0)[month_range]
    # print(time.time() - start)

    lon_grid, lat_grid = get_lon_lat()
    lon_grid = torch.tensor(lon_grid, dtype=torch.float)
    lon_grid = lon_grid.repeat(months, 1, 1)
    lat_grid = torch.tensor(lat_grid, dtype=torch.float)
    lat_grid = lat_grid.repeat(months, 1, 1)
    lon_grid[:, mask] = np.nan
    lat_grid[:, mask] = np.nan

    lon_grid = torch.flatten(lon_grid)
    lon_grid = lon_grid[~torch.isnan(lon_grid)]
    lon_grid = torch.reshape(lon_grid, shape=(months, -1))

    lat_grid = torch.flatten(lat_grid)
    lat_grid = lat_grid[~torch.isnan(lat_grid)]
    lat_grid = torch.reshape(lat_grid, shape=(months, -1))
    torch.set_printoptions(threshold=10_000)
    nino_area = (10 <= lon_grid[0]) & (60 >= lon_grid[0]) & (-5 <= lat_grid[0]) & (5 >= lat_grid[0])
    assert torch.sum(nino_area) == 33
    print(nino_area)

    # print(lat_grid.shape)
    # print(time.time() - start)

    print('Samples: {}'.format(len(train_label) - 36))
    dict_train = {
        'sst': train_sst,
        't300': train_t300,
        'ua': train_ua,
        'va': train_va,
        'lon': lon_grid / 300,
        'lat': lat_grid / 100,
        'label': train_label}

    train_dataset = GraphDataSet(dict_train)
    return train_dataset


def get_lon_lat():
    x = np.arange(-180, 180, 5)
    y = np.arange(-55, 65, 5)
    lon_grid, lat_grid = np.meshgrid(x, y)
    return lon_grid, lat_grid


def land_mask():
    x = np.arange(-180, 180, 5)
    lon_grid, lat_grid = get_lon_lat()
    is_on_land = globe.is_land(lat_grid, lon_grid)
    is_on_land = np.concatenate([is_on_land[:, x >= 0], is_on_land[:, x < 0]], axis=1)
    # plt.imshow(is_on_land[::-1, :])
    # plt.show()
    return is_on_land


if __name__ == '__main__':
    load_graph_data('soda')
    # from torch.utils.data import DataLoader
    # start_time = time.time()
    # train_dataset = load_train_data('cmip')
    # valid_dataset = load_val_data('soda')
    # train_loader = DataLoader(train_dataset, batch_size=args['batch_size'])
    # print(time.time() - start_time)
    # for step, (sst, t300, ua, va, label, sst_label) in enumerate(train_loader):
    #     print(sst.shape, va.shape, label.shape, sst_label.shape)
