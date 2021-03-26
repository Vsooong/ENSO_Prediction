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
    a = 0
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

    pred = torch.randn(12, 24)
    label = torch.randn(12, 24)
    weighted_mse_loss(pred, label)
