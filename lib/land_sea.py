import torch
import numpy as np
from global_land_mask import globe


def get_flat_lon_lat(months):
    mask = land_mask()
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
    return lon_grid, lat_grid


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
