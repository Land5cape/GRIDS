# coding:utf8

import csv
import random

import numpy as np
import torch
import torch.nn.init as init
import torch.nn as nn
from scipy import stats

# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_pred_label(csv_path, *list):
    infos = zip(*list)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for info in infos:
            writer.writerow(info)


def adjust_learning_rate(optimizer, epoch, step, rate, min_lr):
    lr = optimizer.param_groups[0]['lr']

    if lr <= min_lr:
        return

    if epoch % step == step - 1:
        cur_lr = lr * rate
        new_lr = cur_lr if cur_lr > min_lr else min_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def weigth_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def MOS_label(MOS, MOS_range):
    MOS_min, MOS_max = MOS_range
    label = (MOS - MOS_min) / (MOS_max - MOS_min)
    return label


def label_MOS(label, MOS_range):
    MOS_min, MOS_max = MOS_range
    MOS = label * (MOS_max - MOS_min) + MOS_min
    return MOS


def regression(x, y, lower_better=False):
    from scipy.optimize import leastsq

    if lower_better:
        if isinstance(x, list):
            x = [-k for k in x]
        else:
            x = -x

    # 回归模型
    def f(p, _x):
        a1, a2, a3, a4 = p
        return (a1 - a2) / (1 + np.e ** (-(_x - a3) / np.abs(a4)) + a2)

    # 误差公式
    def error(p, _x, _y):
        return f(p, _x) - _y

    p = np.array([1, 1, 1, 1])  # 初始参数
    # -----------------------------------------
    para = leastsq(error, p, args=(x, y))[0]  #
    # -----------------------------------------
    print(para, 'lower_better:', lower_better)

    new_x = f(para, x)

    # new_x = pd.Series([new_x[k] if new_x[k] <= 1 else y[k] for k in range(len(new_x))])

    return new_x


def get_PLCC(y_pred, y_val, using_regression=True, lower_better=False):
    if using_regression:
        y_pred = regression(y_pred, y_val, lower_better=lower_better)
    return stats.pearsonr(y_pred, y_val)[0]


def get_SROCC(y_pred, y_val, lower_better=False):
    if lower_better:
        if isinstance(y_pred, list):
            y_pred = [-k for k in y_pred]
        else:
            y_pred = -y_pred
    return stats.spearmanr(y_pred, y_val)[0]


def get_KROCC(y_pred, y_val):
    return stats.stats.kendalltau(y_pred, y_val)[0]


def get_RMSE(y_pred, y_val, MOS_range):
    y_p = label_MOS(y_pred, MOS_range)
    y_v = label_MOS(y_val, MOS_range)
    return np.sqrt(np.mean((y_p - y_v) ** 2))


def get_MSE(y_pred, y_val, MOS_range):
    y_p = label_MOS(y_pred, MOS_range)
    y_v = label_MOS(y_val, MOS_range)
    return np.mean((y_p - y_v) ** 2)


def mos_scatter(pred, mos, show_fig=False):
    fig = plt.figure()
    plt.scatter(mos, pred, s=5, c='g', alpha=0.5)
    plt.xlabel('MOS')
    plt.ylabel('PRED')
    plt.plot([0, 1], [0, 1], linewidth=0.5)
    if show_fig:
        plt.show()
    return fig


def fig2data(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def get_encoding(file, filelen):
    import chardet
    with open(file, 'rb') as f:
        tmp = chardet.detect(f.read(filelen))
    return tmp['encoding']


def read_label(csv_path, id1, id2, tail):
    dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        head_row = next(reader)
        for row in reader:
            name = row[id1]
            if not name.endswith(tail):
                name = name + tail
            dict[name] = float(row[id2])
    return dict


def read_split(csv_path):
    info_dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        head_row = next(reader)
        for row in reader:  # name, class
            info_dict[row[0]] = row[1]
    return info_dict


def write_split(csv_path, infos, wtype):
    with open(csv_path, wtype, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if wtype == 'w':
            writer.writerow(['file_name', 'class'])
        for info in infos:
            writer.writerow(info)


def save_param(dir_path, param):
    with open('{}/param.txt'.format(dir_path), 'w+', encoding='utf-8') as f:
        f.writelines(['{}\n'.format(k) for k in param])


def cal(csv_path, d):
    KoNViD_1k_MOS = [1.22, 4.64]
    CVD2014_MOS = [-6.50, 93.38]
    LiveQ_MOS = [16.5621, 73.6428]
    LiveV_MOS = [6.2237, 94.2865]
    UGC_MOS = [1.242, 4.698]

    if d == 1:
        mos = KoNViD_1k_MOS
    elif d == 2:
        mos = CVD2014_MOS
    elif d == 4:
        mos = LiveV_MOS
    elif d == 5:
        mos = UGC_MOS

    l1 = []
    l2 = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        head_row = next(reader)
        for row in reader:
            l1.append(float(row[0]))
            l2.append(float(row[1]))

    l1 = np.array(l1)
    l2 = np.array(l2)

    plcc = get_PLCC(l1, l2)
    srocc = get_SROCC(l1, l2)
    krocc = get_KROCC(l1, l2)
    rmse = get_RMSE(l1, l2, mos)

    print(srocc, krocc, plcc, rmse)


# if __name__ == '__main__':
#     save_pred_label( './11111.csv', [1,2], [1,2], [1,2])


# if __name__ == '__main__':
#     # save_crop_video('E:\workplace\VQAdataset\Live-VQC\Video\R001.mp4', 'E:\workplace\VQAdataset\Live-VQC\\train', '.mp4')
#
#     cal('../model-save/1_2022_03_26_12_41_27/checkpoint/train_4.csv', 1)
#     cal('../model-save/1_2022_04_01_22_30_38/checkpoint/val_25.csv', 1)  #
#     # cal('../model-save/2_2022_04_05_12_50_26/checkpoint/val_18.csv', 2)
#     # cal('../model-save/2_2022_04_05_21_35_44/checkpoint/val_14.csv', 2)
#     cal('../model-save/2_2022_04_07_16_19_36/checkpoint/val_24.csv', 2)
#     cal('../model-save/2_2022_04_07_16_19_36/checkpoint/val_25.csv', 2)
#     cal('../model-save/2_2022_04_07_16_19_36/checkpoint/val_26.csv', 2)  #
#     cal('../model-save/4_2022_04_07_16_21_08/checkpoint/val_13.csv', 4)
#     cal('../model-save/4_2022_04_07_16_21_08/checkpoint/val_17.csv', 4)
#     cal('../model-save/4_2022_04_07_16_21_08/checkpoint/val_24.csv', 4)  #
#     cal('../model-save/4_2022_04_07_16_21_08/checkpoint/val_29.csv', 4)
#     cal('../model-save/4_2022_04_07_16_21_08/checkpoint/val_24.csv', 4)  #
