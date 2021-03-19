import shutil
from configs import args
import torch
import os
import numpy as np
import zipfile
from lib.util import norm


def test(in_path='./tcdata/enso_round1_test_20210201/',
         out_path='result'):
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
    # print(current_dir)
    save_dir = os.path.join(current_dir, 'experiments', args['model_name'] + '.pth')
    model.load_state_dict(torch.load(save_dir, map_location=device))
    model.to(device)
    model.eval()

    for i in test_sample_file:
        data = np.load(i)
        adj = torch.tensor(norm(np.ones((4, 4))), dtype=torch.float).to(device)
        sst = torch.as_tensor(data[..., 0], dtype=torch.float).to(device).unsqueeze(0)
        t300 = torch.as_tensor(data[..., 1], dtype=torch.float).to(device).unsqueeze(0)
        ua = torch.as_tensor(data[..., 2], dtype=torch.float).to(device).unsqueeze(0)
        va = torch.as_tensor(data[..., 3], dtype=torch.float).to(device).unsqueeze(0)
        preds = model(sst, t300, ua, va)
        preds = preds.squeeze(0).cpu().detach().numpy()

        save_path = os.path.join(out_path, os.path.basename(i))
        np.save(file=save_path, arr=preds)
    make_zip(out_path, 'result.zip')


def make_zip(source_dir, output_filename):
    f = zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            f.write(os.path.join(dirpath, filename))
    f.close()


if __name__ == '__main__':
    test()
