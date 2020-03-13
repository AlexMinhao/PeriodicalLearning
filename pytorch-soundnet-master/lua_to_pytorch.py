import argparse
import pathlib

import torch
# import torchfile
from soundnet import SoundNet
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
import json
from run import load_dataset
# %matplotlib inline

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument(dest='in_path', type=pathlib.Path)
    # parser.add_argument(dest='out_path', type=pathlib.Path)
    # args = parser.parse_args()
    #
    # lua_model = torchfile.load(args.in_path)
    #
    # print(" * Loaded lua model")
    # for i, module in enumerate(lua_model['modules']):
    #     print(f"    {i}, {module._typename}")

    model = SoundNet()
    model_map = {
        0: model.conv1,
        4: model.conv2,
        8: model.conv3,
        11: model.conv4,
        14: model.conv5,
        18: model.conv6,
        21: model.conv7,
    }

    for lua_idx, module in model_map.items():
        print(f" * Importing {module}")
        # weight = torch.from_numpy(lua_model['modules'][lua_idx]['weight'])
        # bias = torch.from_numpy(lua_model['modules'][lua_idx]['bias'])
        # module.weight.data.copy_(weight)
        # module.bias.data.copy_(bias)

    # lua_conv8_objs = lua_model['modules'][24]['modules'][0]
    # lua_conv8_scns = lua_model['modules'][24]['modules'][1]

    # print(f" * Importing {model.conv8_objs}")
    # weight = torch.from_numpy(lua_conv8_objs['weight'])
    # bias = torch.from_numpy(lua_conv8_objs['bias'])
    # model.conv8_objs.weight.data.copy_(weight)
    # model.conv8_objs.bias.data.copy_(bias)

    # print(f" * Importing {model.conv8_scns}")
    # weight = torch.from_numpy(lua_conv8_scns['weight'])
    # bias = torch.from_numpy(lua_conv8_scns['bias'])
    # model.conv8_scns.weight.data.copy_(weight)
    # model.conv8_scns.bias.data.copy_(bias)
    #
    # print(f" * Saving pytorch model to {args.out_path!s}")
    # torch.save(model.state_dict(), args.out_path)


if __name__ == '__main__':
    # main()
    X = []
    seg = np.array([465, 652, 857, 1048, 1260, 1480, 1698, 1897, 2075, 2251, 2452, 2679, 2855])
    s = np.array([[4364, 4556],[4850, 4981],[5216, 5387],[5592, 5753],[5913, 6075],[6317, 6497],[6709, 6900],[7205, 7383]])
    labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2];
    with open('C:\\Users\\Alex\\Desktop\\test\\kn.txt', 'r') as load_f:
        for line in load_f.readlines():
            line = line.split('\t')  # 去掉列表中每一个元素的换行符
            if len(line)<3:
                continue
            acc = np.zeros(3,)
            acc[0] = float(line[0])
            acc[1] = float(line[1])
            acc[2] = float(line[2])
            X.append(acc)
            print(line)
    X = np.array(X)

    data_seg = np.zeros((len(seg)+7, 685))

    for i in range(len(seg)-1):
        seg_data = X[seg[i]:seg[i+1], :]
        seg_data = seg_data.reshape(seg_data.shape[0]*seg_data.shape[1])
        for j in range(len(seg_data)):
            data_seg[i][j] = seg_data[j]

    for i in range(8):
        seg_data = X[s[i][0]:s[i][1], :]
        seg_data = seg_data.reshape(seg_data.shape[0] * seg_data.shape[1])
        for j in range(len(seg_data)):
            data_seg[len(seg)-1+ i][j] = seg_data[j]

    X = data_seg
    #0.1 红
    # color1 = np.ones(28,)*0.1
    # color2 = np.ones(24, )*0.2
    # color3 = np.ones(22, )*0.3
    # color4 = np.ones(20,)*0.4
    # color5 = np.ones(22, )*0.5
    # color6 = np.ones(30, )*0.6
    # color7 = np.ones(23, )*0.7
    # color8 = np.ones(28, )*0.8
    # color9 = np.ones(28, )*0.9
    # color10 = np.ones(155, )*0.1
    # color11 = np.ones(225, )*0.5
    # # color = np.hstack((color1,color2,color3,color4,color5,color6,color7,color8,color9,color10))
    # color = np.hstack((color10, color11))
    train_x, train_y = load_dataset('C:\\ALEX\\Doc\\Reference\\SoundNet\\PeriodicalLearning\\Dataset\\imu_train_data.json', FixLength=False, DataAugmentation=True)
    val_x, val_y = load_dataset('C:\\ALEX\\Doc\\Reference\\SoundNet\\PeriodicalLearning\\Dataset\\imu_validate_data.json', FixLength=False)
    X1 = train_x.reshape(train_x.shape[0],-1)
    X2 = val_x.reshape(-1,216) #2673
    X = np.vstack((X1, X2)) #155
    n_neighbors = 5
    n_components = 3

    print(X.shape)
    print(X[:10])
    # print(color.shape)
    # print(color[:10])

    fig = plt.figure(figsize=(8, 8))
    # 创建了一个figure，标题为"Manifold Learning with 1000 points, 10 neighbors"
    # plt.suptitle("Manifold Learning with %i points, %i neighbors"
    #              % (len(color), n_neighbors), fontsize=14)
    '''绘制S曲线的3D图像'''
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='y', cmap=plt.cm.Spectral)
    ax.view_init(4, -72)  # 初始化视角

    # t=SME
    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    Y = tsne.fit_transform(X)  # 转换后的输出
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))  # 打印算法用时


    ax = fig.add_subplot(212, projection='3d')
    ax.scatter(Y[0:X1.shape[0], 0], Y[0:X1.shape[0], 1], Y[0:X1.shape[0], 2], c='r', cmap=plt.cm.Spectral)
    ax.scatter(Y[X1.shape[0]:-1, 0], Y[X1.shape[0]:-1, 1], Y[X1.shape[0]:-1, 2], c='b', cmap=plt.cm.Spectral)
    ax.view_init(4, -72)  # 初始化视角
    # plt.scatter(Y[0:X1.shape[0], 0], Y[0:X1.shape[0], 1], Y[0:X1.shape[0], 2], c='r', cmap=plt.cm.Spectral)
    # plt.scatter(Y[X1.shape[0]:-1, 0], Y[X1.shape[0]:-1, 1], Y[X1.shape[0]:-1, 2], c='b', cmap=plt.cm.Spectral)

    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    ax.yaxis.set_major_formatter(NullFormatter())

    plt.show()
