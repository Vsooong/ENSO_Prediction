import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from global_land_mask import globe
import numpy as np
from scipy import interpolate
from configs import args
import xarray as xr


def visualize():
    # data = netCDF4.Dataset(args['sota_data'])
    # label = netCDF4.Dataset(args['cmip_label'])
    # label = np.array(label.variables['nino'][4544:4645])
    #
    # # 指数趋势
    # label_all = [label[0]] + [label[i, 24:] for i in range(1, label.shape[0])]
    # label_all = np.concatenate(label_all, axis=0)
    # plt.plot(label_all, 'k', linewidth=1)
    # plt.xlabel('Time / month')
    # plt.ylabel('Nino')
    # plt.show()

    # 海陆掩膜
    # lon = np.array(data.variables['lon'])
    # lat = np.array(data.variables['lat'])
    # y = lat
    # x = lon
    # z = data.variables['sst'][0, 0]
    # f = interpolate.interp2d(x, y, z, kind='cubic')
    # xnew = np.arange(0, 356, 1)
    # ynew = np.arange(-65, 66, 1)
    # znew = f(xnew, ynew)
    # lon_grid, lat_grid = np.meshgrid(xnew - 180, ynew)
    # # Check if a point is on land:
    # is_on_land = globe.is_land(lat_grid, lon_grid)
    # is_on_land = np.concatenate([is_on_land[:, xnew >= 180], is_on_land[:, xnew < 180]], axis=1)
    # znew[is_on_land] = np.nan
    # plt.imshow(znew[::-1, :], cmap=plt.cm.RdBu_r)  # is_on_land[:,:])
    # plt.show()
    #
    # # 温度场和风场可视化
    # lon_grid, lat_grid = np.meshgrid(x - 180, y)
    # is_on_land = globe.is_land(lat_grid, lon_grid)
    # is_on_land = np.concatenate([is_on_land[:, x >= 180], is_on_land[:, x < 180]], axis=1)
    # mask = np.zeros(data.variables['t300'].shape, dtype=int)
    # mask[:, :, :, :] = is_on_land[np.newaxis, np.newaxis, :, :]
    # lon_grid, lat_grid = np.meshgrid(x - 180, y)
    # # Check if a point is on land:
    # is_on_land = globe.is_land(lat_grid, lon_grid)
    # is_on_land = np.concatenate([is_on_land[:, x >= 180], is_on_land[:, x < 180]], axis=1)
    #
    # ua = data.variables['ua'][0, 0]
    # ua[is_on_land] = np.nan
    # va = data.variables['va'][0, 0]
    # va[is_on_land] = np.nan
    #
    # lon_grid, lat_grid = np.meshgrid(xnew - 180, ynew)
    # # Check if a point is on land:
    # is_on_land = globe.is_land(lat_grid, lon_grid)
    # is_on_land = np.concatenate([is_on_land[:, xnew >= 180], is_on_land[:, xnew < 180]], axis=1)
    # znew[is_on_land] = np.nan
    # plt.figure(figsize=(15, 10))
    # plt.imshow(znew[::-1, :], cmap=plt.cm.RdBu_r)
    # plt.colorbar(orientation='horizontal')
    # plt.quiver(lon, lat + 65, ua[::-1, :], va[::-1, :], alpha=0.8)
    # plt.show()

    pass


# 缺失值分析
def nan_analysis():
    pass


if __name__ == '__main__':
    # visualize()
    # data = xr.open_dataset(args['cmip_data'])
    # sst = data['sst'][145, 6]
    # print(sst)
    # plt.imshow(sst)
    pass
