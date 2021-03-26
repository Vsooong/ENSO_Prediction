import torch
import random
import numpy as np
import os
from torch import nn
from collections import OrderedDict


def init_seed(seed=1):
    '''
    Disable cudnn to maximize reproducibility
    '''
    # torch.cuda.cudnn_enabled = False
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def print_model_parameters(model, only_num=False):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')


def get_memory_usage(device):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 * 1024.)
    cached_memory = torch.cuda.memory_cached(device) / (1024 * 1024.)
    return allocated_memory, cached_memory
    # print('Allocated Memory: {:.2f} MB, Cached Memory: {:.2f} MB'.format(allocated_memory, cached_memory))


# NINO3.4区（170°W-120°W，5°S-5°N）
def nino_index(ssta_year, keep_dim=False):
    if not keep_dim:
        batch, months, h, w = ssta_year.size()
    else:
        batch, months, h, w = ssta_year.size()
        x1 = ssta_year[:, months - 2:months - 1, ...]
        x2 = ssta_year[:, months - 1:months, ...]
        y1 = 0.5 * x1 + 0.5 * x2
        y2 = 1.5 * x2 - 0.5 * x1
        ssta_year = torch.cat([ssta_year, y1, y2], dim=1)
        months += 2
    ssta1 = ssta_year[:, 0:months - 2, 10:13, 38:49]
    ssta2 = ssta_year[:, 1:months - 1, 10:13, 38:49]
    ssta3 = ssta_year[:, 2:months, 10:13, 38:49]
    ssta1 = torch.mean(ssta1, dim=[2, 3])
    ssta2 = torch.mean(ssta2, dim=[2, 3])
    ssta3 = torch.mean(ssta3, dim=[2, 3])
    nino = (ssta1 + ssta2 + ssta3) / 3
    return nino


sea_grid_nino_area = torch.Tensor([False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, True, True, True, True, True,
                                   True, True, True, True, True, True, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, True, True, True,
                                   True, True, True, True, True, True, True, True, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, True, True, True, True, True, True, True, True,
                                   True, True, True, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False, False, False, False, False, False, False,
                                   False, False, False, False]).bool()


def nino_index_flat(ssta_year):
    mask = sea_grid_nino_area
    btch, months = ssta_year.size()[:2]
    ssta1 = ssta_year[:, 0:months - 2, mask]
    ssta2 = ssta_year[:, 1:months - 1, mask]
    ssta3 = ssta_year[:, 2:months, mask]
    ssta1 = torch.mean(ssta1, dim=2)
    ssta2 = torch.mean(ssta2, dim=2)
    ssta3 = torch.mean(ssta3, dim=2)
    nino = (ssta1 + ssta2 + ssta3) / 3
    return nino


def norm(adj):
    # adj += np.eye(adj.shape[0]) # 为每个结点增加自环
    degree = np.array(adj.sum(1))  # 为每个结点计算度
    degree = np.diag(np.power(degree, -0.5))
    return degree.dot(adj).dot(degree)


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))


if __name__ == '__main__':
    from configs import args
    import xarray as xr

    # from global_land_mask import globe
    #
    #
    # def get_lon_lat():
    #     x = np.arange(-180, 180, 5)
    #     y = np.arange(-55, 65, 5)
    #     lon_grid, lat_grid = np.meshgrid(x, y)
    #     return lon_grid, lat_grid
    #
    #
    # def land_mask():
    #     x = np.arange(-180, 180, 5)
    #     lon_grid, lat_grid = get_lon_lat()
    #     is_on_land = globe.is_land(lat_grid, lon_grid)
    #     is_on_land = np.concatenate([is_on_land[:, x >= 0], is_on_land[:, x < 0]], axis=1)
    #     # plt.imshow(is_on_land[::-1, :])
    #     # plt.show()
    #     return is_on_land
    #
    #
    # mask = land_mask()
    import torch

    data = torch.as_tensor(xr.open_dataset(args['cmip_data'])['sst'].values[1, ...]).float().unsqueeze(0)
    label = xr.open_dataset(args['cmip_label'])['nino']
    print(torch.as_tensor(label[1].values).float())
    print(nino_index(data, keep_dim=True))
    # print(data.shape)
    # data[:, mask] = np.nan
    # data = torch.flatten(torch.as_tensor(data, dtype=torch.float))
    # data = data[~torch.isnan(data)]
    # data = torch.reshape(data, shape=(36, -1)).unsqueeze(0)
    # print(data.shape)
    # print(nino_index_flat(data))
