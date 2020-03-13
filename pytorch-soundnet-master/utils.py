
from itertools import zip_longest
import numpy as np
import math
import torch
from torch.autograd import Variable
import csv
from definitions import *
from torch import nn
import os
from helper import ToDirection

class SeqGenerator:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.

    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.

    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """

    def __init__(self,  batch_size, data_x , data_y, pad=0,
                 augment=False, shuffle=True, random_seed=1234):
        assert data_x is None or len(data_x) == len(data_y)
        self.augment = augment
        self.random = np.random.RandomState(random_seed)
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.num_batches = (len(data_y) + batch_size - 1) // batch_size  #

        if data_x is not None:  # 1024             1                  17                     3
            self.batch_3d = np.empty((batch_size,  1,     data_x.shape[-2],        1))
            # 1024           81                  17                        2
        self.batch_2d = np.empty((batch_size,  1  , data_y.shape[-2], 1))
        self.data_y = []

        data_x = data_x.reshape((-1, channel, 3, Data_length))
        pairs = []  # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(data_y)):
            xy = (data_x[i], data_y[i])
            pairs.append(xy)
        self.pairs = pairs

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def set_augment(self, augment):
        self.augment = augment

    def getDataSet(self):
        return self.pairs
    # def next_pairs(self):
    #
    #     if self.shuffle:
    #         pairs = self.random.permutation(self.pairs)
    #     else:
    #         pairs = self.pairs
    #     return 0, pairs
    #
    # def next_epoch(self):
    #     enabled = True
    #     while enabled:
    #         for seq_imu, label in zip_longest(self.data_x, self.data_y):
    #
    #             batch_seq_imu = None if seq_imu is None else np.expand_dims(seq_imu, axis=0)
    #             batch_label = label
    #
    #             yield batch_seq_imu, batch_label


from datetime import datetime




def checkpoint(epoch, model,optimizer):
    model_out_path = os.path.join(os.getcwd(), r'results', "model_best_adjust.pth")
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    a = num_correct / total
    return  num_correct / total

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if pretrain:
        lr = BASE_lr*0.005 #* (0.5 ** (epoch // 2))
        if epoch > 50:
            lr = BASE_lr * 0.001
    else:
        # lr = BASE_lr * (1.5 ** (epoch //10))

        if epoch <= 10:
            lr = 0.05*BASE_lr
        elif epoch <= 200 and epoch > 10:
            lr = BASE_lr * (0.8 ** (epoch // 20))
        # elif epoch <= 200 and epoch > 100:
        #     lr = BASE_lr * (0.9 ** (epoch // 20))
        else:
            lr = BASE_lr*0.0005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

def distance(predicted, target):
    assert predicted.shape == target.shape
    loss = nn.MSELoss()
    return loss(predicted,target)


def ReshapeLabel(label):
    b, w = label.size()
    label_atten = label.view(-1, Short_Length)
    category, location = torch.max(label_atten.data, 1)
    label_location = location.view(b, -1) # 20 31
    label_atten = category.view(b, -1)  # 20 31
    b, w = label_atten.size()
    Y_category = []
    Y_location = []
    for i in range(b):
        y = torch.zeros(4)
        y_loc = torch.zeros(4)
        pos = 0
        neg = 0
        NUM = 0
        for j in range(w):
            if NUM>3:
                continue
            if label_atten[i][j] == 1:
                y[NUM] = 2*j-1
                y_loc[NUM] = label_location[i][j]
                pos = pos + 1
                NUM = NUM + 1
            elif label_atten[i][j] == 2:
                y[NUM] = 2*j
                y_loc[NUM] = label_location[i][j]
                neg = neg + 1
                NUM = NUM+1
        Y_category.append(y)
        Y_location.append(y_loc)
    Y_category = get_variable(torch.stack(Y_category).float())
    Y_location = get_variable(torch.stack(Y_location).float())

    return Y_category, Y_location

def ReshapeLabel_1(label):
    b, w = label.size()
    new_label = []
    new_atten = []
    for i in range(b):
        seq_l = (label[i] == 1).view(1, -1)
        val, loc_1 = torch.max(seq_l, 1)

        seq_2 = (label[i] == 2).view(1, -1)
        val, loc_2 = torch.max(seq_2, 1)
        seq_label = np.array([loc_1.item(), 1, loc_2.item(), 2])
        seq_atten = np.array([loc_1.item(), loc_2.item()])
        new_label.append(torch.from_numpy(seq_label))
        new_atten.append(torch.from_numpy(seq_atten))

    new_label = get_variable(torch.stack(new_label).float())
    new_atten = get_variable(torch.stack(new_atten).float())
    return new_label, new_atten

def ReshapeLabel_2(label):
    b, w = label.size()
    t = np.linspace(0, 71, 72)
    new_label = []
    new_atten = []
    X = []
    for i in range(b):
        seq_l = (label[i] == 1).view(1, -1)
        val, loc_1 = torch.max(seq_l, 1)

        seq_2 = (label[i] == 2).view(1, -1)
        val, loc_2 = torch.max(seq_2, 1)
        seq_atten = np.array([loc_1.item(), loc_2.item()])
        x1 = np.exp(-1 * ((t - seq_atten[0]) ** 2) / (2 * (1 ** 2))) / (math.sqrt(2 * np.pi) * 1)
        x2 = np.exp(-1 * ((t - seq_atten[1]) ** 2) / (2 * (2 ** 2))) / (math.sqrt(2 * np.pi) * 2)
        x = x1 + x2
        x = torch.from_numpy(x)

        X.append(x)
    X = torch.stack(X).float()
    return  X.view(b, w, 1)

                    #20 2    20 248  -> 20 31 8
def LongShortdistance(output, atten, label):
    loss = nn.L1Loss()
    loss_classification = nn.CrossEntropyLoss()
    new_label, new_atten = ReshapeLabel_1(label)

    Loss_output = loss(output, new_label)
    if attention:
        # Loss_atten = loss(atten.float(), new_atten)
        #
        # Loss_total = Loss_output + Loss_atten
        # return Loss_output, Loss_atten, Loss_total
        Loss_total = Loss_output
        return Loss_output, torch.Tensor([0.]), Loss_total
    else:
        Loss_total = Loss_output
        return Loss_output, torch.Tensor([0.]), Loss_total



def get_acc(output, label):
    total = output.shape[0]
    pred_con_loc = output[:, 0]
    pred_con_catory =  torch.ceil(output[:, 1])
    pred_teo_loc =  output[:, 2]
    pred_teo_catory = torch.ceil(output[:, 3])

    new_label, new_atten = ReshapeLabel_1(label)
    # pred_category = output[:, 1]
    # label_category, label_location = torch.max(label.data, 1)
    #
    num_correct_con_loc = 0
    num_correct_teo_loc = 0
    for i in range(len(pred_con_loc)):
        if pred_con_loc[i] >= new_label[i][0]-1 and pred_con_loc[i] <= new_label[i][0]+1:
            num_correct_con_loc = num_correct_con_loc +1
        if pred_teo_loc[i] >= new_label[i][2]-1 and pred_teo_loc[i] <= new_label[i][2]+1:
            num_correct_teo_loc = num_correct_teo_loc +1

    correct_con_loc = num_correct_con_loc / total
    correct_teo_loc = num_correct_teo_loc / total
    return correct_con_loc, correct_teo_loc


def data_augmentation(quat, vec, times):
    V = []
    for i in range(times):
        theta_x = np.pi * (np.random.rand(1)-1)
        theta_y = np.pi * (np.random.rand(1)-1)
        theta_z = np.pi * (np.random.rand(1)-1)
        qx = np.array([np.cos(theta_x/2), np.sin(theta_x/2), 0, 0])
        qy = np.array([np.cos(theta_y / 2),  0, np.sin(theta_y / 2),  0])
        qz = np.array([np.cos(theta_z / 2), 0, 0, np.sin(theta_z / 2)])
        v = []
        bais = np.random.rand(1,3)
        for i in range(len(quat)):
            q_init = quat[i,:]
            q_init[0] = -q_init[0]
            v_init = ToDirection(q_init, vec[i, :])
            v_rot = ToDirection(qx, v_init)
            v_rot = ToDirection(qy, v_rot)
            v_rot = ToDirection(qz, v_rot)
            vnew = ToDirection(quat[i,:], v_rot)
            v.append(vnew)
        v = np.array(v)
        V.append(v)
    return V

