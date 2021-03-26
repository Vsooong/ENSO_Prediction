import shutil
from configs import args
import torch
import os
import numpy as np
import zipfile
from data_loader import land_mask, get_flat_lon_lat
from lib.util import norm
import re

args['model_name'] = 'CNN2_4'


def mask_flat_tensor(sst, t300, ua, va):
    batch, month = sst.size()[:2]
    mask = land_mask()

    def mask_tensor(input):
        input[:, :, mask] = np.nan
        input = torch.flatten(input)
        input = input[~torch.isnan(input)]
        input = torch.reshape(input, shape=(1, month, -1))
        return input

    sst = mask_tensor(sst)
    t300 = mask_tensor(t300)
    ua = mask_tensor(ua)
    va = mask_tensor(va)
    lon, lat = get_flat_lon_lat(month)
    return sst, t300, ua / 5, va / 5, lon.unsqueeze(0) / 180, lat.unsqueeze(0) / 60


def test(in_path='./tcdata/enso_final_test_data_B/',
         out_path='result'):
    # def test(in_path='D:\\github\\data\\data\\meta_data\\test样例',
    #          out_path='result'):
    if not os.path.exists(in_path):
        for path in args['path_list']:
            if os.path.exists(path):
                in_path = path

    if os.path.exists(out_path):
        shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path)

    test_sample_file = [os.path.join(in_path, i) for i in os.listdir(in_path) if i.endswith('.npy')]
    device = args['device']
    model = args['model_list'][args['model_name']]()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(current_dir, 'experiments', args['model_name'] + '.pth')
    model.load_state_dict(torch.load(save_dir, map_location=device))
    model.to(device)
    model.eval()

    for i in test_sample_file:
        data = np.load(i)
        data = np.nan_to_num(data)
        sst = torch.as_tensor(np.nan_to_num(data[..., 0]), dtype=torch.float).unsqueeze(0)
        t300 = torch.as_tensor(np.nan_to_num(data[..., 1]), dtype=torch.float).unsqueeze(0)
        ua = torch.as_tensor(np.nan_to_num(data[..., 2]), dtype=torch.float).unsqueeze(0)
        va = torch.as_tensor(np.nan_to_num(data[..., 3]), dtype=torch.float).unsqueeze(0)
        start_month = torch.as_tensor(int(re.split("-|_", os.path.basename(i))[2]))

        if args['model_name'] == 'AGCRN':
            sst, t300, ua, va, lon, lat = mask_flat_tensor(sst, t300, ua, va)
            preds = model(sst.to(device), t300.to(device), ua.to(device), va.to(device), lon.to(device), lat.to(device))
        elif args['model_name'] == 'graphCNN':
            adj = torch.tensor(norm(np.ones((4, 4))), dtype=torch.float).to(device)
            preds = model(sst.to(device), t300.to(device), ua.to(device), va.to(device), adj)
        elif args['model_name'] == 'CNN2_5' or args['model_name'] == 'CNN2_4':
            preds = model(sst.to(device), t300.to(device), ua.to(device), va.to(device), start_month.long().to(device))
        else:
            month = torch.arange(start_month, start_month + 12).unsqueeze(0)
            preds = model(sst.to(device), t300.to(device), ua.to(device), va.to(device), month.float().to(device))

        if len(preds) == 2:
            preds = preds[1]
        preds = preds.squeeze(0).cpu().detach().numpy()

        save_path = os.path.join(out_path, os.path.basename(i))
        np.save(file=save_path, arr=preds)
        del preds
        make_zip(out_path, 'result.zip')


def make_zip(source_dir, output_filename):
    f = zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            f.write(os.path.join(dirpath, filename))
    f.close()


if __name__ == '__main__':
    test()
