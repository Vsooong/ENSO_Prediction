import numpy as np
import torch

torch.set_printoptions(profile="full")


def weighted_mse_loss(input, target, weight=None, sum=False):
    if weight is None:
        weight = 10 - torch.relu(10.0 - torch.exp(target * target / 2))
    if not sum:
        loss = (weight * (input - target) ** 2).mean()
    else:
        loss = torch.sum(weight * (input - target) ** 2)
    return loss


def score_loss(pred, label, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    batch, time = pred.size()
    assert batch != 1
    normal_rmse = torch.sqrt(torch.mean((pred - label) ** 2))

    rmse = torch.sqrt(torch.sum((pred - label) ** 2, dim=0) / batch)
    x_mean = torch.mean(pred, dim=0, keepdim=True)
    y_mean = torch.mean(label, dim=0, keepdim=True)
    c1 = torch.sum((pred - x_mean) * (label - y_mean), dim=0)
    c2 = torch.sum((pred - x_mean) ** 2, dim=0) * torch.sum((label - y_mean) ** 2, dim=0)
    corr = c1 / torch.sqrt(c2)
    a = 2 / 3 * torch.tensor([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6).to(device)
    b = torch.log(torch.arange(1, 25)).to(device)
    accskill = torch.sum(a * b * corr)
    rmseskill = torch.sum(rmse)
    score = accskill - rmseskill
    # print('acc', accskill)
    # print('rmse', rmseskill)
    # print('N-rmse', normal_rmse)

    return normal_rmse - score / 24


def coreff(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    c1 = sum((x - x_mean) * (y - y_mean))
    c2 = sum((x - x_mean) ** 2) * sum((y - y_mean) ** 2)
    return c1 / np.sqrt(c2)


def rmse(preds, y):
    return np.sqrt(sum((preds - y) ** 2) / preds.shape[0])


def eval_score(preds, label):
    # preds = preds.cpu().detach().numpy().squeeze()
    # label = label.cpu().detach().numpy().squeeze()
    acskill = 0
    RMSE = 0
    a = [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6
    for i in range(24):
        RMSE += rmse(label[:, i], preds[:, i])
        cor = coreff(label[:, i], preds[:, i])

        acskill += a[i] * np.log(i + 1) * cor
    return 2 / 3 * acskill - RMSE


if __name__ == '__main__':
    import torch

    # :param
    # x: (b, c, t, h, w)

    # pred = torch.randn(64, 24).to('cuda') * 4 - 2
    pred = torch.tensor([-0.83591384, 0.7997291, -0.6917133, -0.4282707, -0.28578496, 0.06585672
                            , 0.06583616, 0.17199379, 0.0757147, -0.03063103, 0.17297582, 0.2939594
                            , 0.3090786, -0.2352544, -0.32079378, 0.24410161, 0.27430546, 0.11380985
                            , 0.11448687, 0.21461603, 0.40430343, 0.5040084, .60113555, 0.5126784]).unsqueeze(0)
    label = torch.tensor([-0.69425577, 0.66442853, 0.55160004, 0.35508764, 0.13081335, 0.0290962
                             , 0.15562464, 0.24810739, 0.42519936, 0.62377197, 0.78617758, 0.72442847
                             , 0.65604728, 0.660824, 0.71607655, 0.70218498, 0.57049966, 0.44788399
                             , 0.29764488, 0.1289155, .17055297, 0.29730734, 0.47233918, 0.49614564]).unsqueeze(0)
    print(score_loss(pred, label))
    print(eval_score(pred.cpu().numpy(), label.cpu().numpy()))
