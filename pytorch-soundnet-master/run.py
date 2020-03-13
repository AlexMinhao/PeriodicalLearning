from helper import sliding_window, AverageMeter
from helper import load_imu_data, segMent,ToDirection
from soundnet import SoundNet
import numpy as np
from utils import SeqGenerator, distance, LongShortdistance
from torch.utils.data import DataLoader
from validate import val_epoch
from trainer import train_epoch
from definitions import *
from torch import nn
from torch import optim
import torch
from utils import Logger, data_augmentation
import os
from tensorboardX import SummaryWriter
from model import Model, Seq2Seq, DeepConvLSTM

def load_dataset(path, FixLength = False, DataAugmentation = False, ws = Data_length, ss = Data_length/2):
    imu_data = load_imu_data(path)
    input_accy = []
    ax = []
    ay = []
    az = []
    Label = []
    input_ax = []
    input_ay = []
    input_az = []

    for i in range(len(imu_data)):
        ax = [imu_data[i][j]['accx'] for j in range(len(imu_data[i]))]
        ay = [imu_data[i][j]['accy'] for j in range(len(imu_data[i]))]
        az = [imu_data[i][j]['accz'] for j in range(len(imu_data[i]))]

        q = [[imu_data[i][j]['qw'], imu_data[i][j]['qx'] , imu_data[i][j]['qy'],imu_data[i][j]['qz']] for j in range(len(imu_data[i]))]
        a = [[imu_data[i][j]['accx'], imu_data[i][j]['accy'] , imu_data[i][j]['accz']] for j in range(len(imu_data[i]))]
        q = np.array(q).astype(np.float32)
        a = np.array(a).astype(np.float32)
        a_rot = data_augmentation(q, a, times=10)

        label = [imu_data[i][j]['label'] for j in range(len(imu_data[i]))]
        print(len(ax))
        length = len(ax)
        if length> 72:
            continue

        for k in range(72-length):

            ax.append(0)
            ay.append(0)
            az.append(0)
            label.append(0)

        input_ax.append(ax)
        input_ay.append(ay)
        input_az.append(az)
        Label.append(label)

        if DataAugmentation:
            for i in range(len(a_rot)):
                ax = np.hstack((a_rot[i][:, 0], np.zeros((72-length),)))
                ay = np.hstack((a_rot[i][:, 1], np.zeros((72 - length),)))
                az = np.hstack((a_rot[i][:, 2], np.zeros((72 - length),)))
                input_ax.append(ax)
                input_ay.append(ay)
                input_az.append(az)
                Label.append(label)

    input_ax =  np.array(input_ax).astype(np.float32)  #167 72
    input_ay =  np.array(input_ay).astype(np.float32)
    input_az =  np.array(input_az).astype(np.float32)

    input = np.asarray([input_ax, input_ay, input_az])
    input = np.stack(input, axis=1)
    Label = np.array(Label).astype(np.float32)


    if FixLength:
        #ws_shape = input_accy.shape()
        data_x = sliding_window(input_ay, ws, ss)
        # np.savetxt('data_x.txt', data_x, fmt='%0.8f')
        data_y = sliding_window(label, ws, ss)

        # data_period = sliding_window(period, ws, ss)

        data_x, data_y = segMent(data_x, data_y)
        return data_x.astype(np.float32), data_y.astype(np.float32), data_period.astype(np.float32)
    else:

        return input, Label


if __name__ == '__main__':
    writer = SummaryWriter(comment = 'ModelSoundPeaks-0228-seq2seq-3Axis-Baseline-Aug-More-10e6')


    train_x, train_y = load_dataset('C:\\ALEX\\Doc\\Reference\\SoundNet\\PeriodicalLearning\\Dataset\\imu_train_data.json',FixLength = False, DataAugmentation=True)
    training_set = SeqGenerator(20, train_x, train_y).getDataSet()
    train_loader = DataLoader(dataset = training_set, batch_size=10, shuffle=True)

    validate_x, validate_y = load_dataset('C:\\ALEX\\Doc\\Reference\\SoundNet\\PeriodicalLearning\\Dataset\\imu_validate_data.json', FixLength=False)
    validate_set = SeqGenerator(20, validate_x, validate_y).getDataSet()
    validation_loader = DataLoader(dataset=validate_set, batch_size=50, shuffle=True)

    # model = Model(
    #         d_model=64, d_inner=2048,
    #         n_head=8, d_k=32, d_v=32,
    #          dropout=0.1, NUM_FILTERS = 256*3)

    model = DeepConvLSTM()


    # If use CrossEntropyLoss，softmax wont be used in the final layer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=BASE_lr, momentum=0.9, weight_decay=0.9)



    if torch.cuda.is_available():
        model.cuda()
        loss_function.cuda()
        print("Model on gpu")

    if pretrain:
        pre_train_path = os.path.join(os.getcwd(),
                                      r'results\model_best.pth')
        pretrain_model = torch.load(pre_train_path)
        model.load_state_dict(pretrain_model['state_dict'])
        optimizer.load_state_dict(pretrain_model['optimizer'])
        print(model)

    result_path = 'log'
    train_logger = Logger(
        os.path.join(result_path, 'train.log'),
        ['epoch', 'loss', 'acc', 'lr', 'f1_score.avg'])
    train_batch_logger = Logger(
        os.path.join(result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr', 'Failure_case_True', 'Failure_case_Pred'])

    # training and testing
    train_correct = [0, ]
    train_total = [0, ]
    f1_train_total = AverageMeter()
    val_loss = []
    for epoch in range(EPOCH):
        train_epoch(epoch, train_loader, model, LongShortdistance, optimizer,
                    train_logger, train_batch_logger, writer)
        val_epoch(epoch, validation_loader, model, LongShortdistance, optimizer, val_loss, writer)

    # print('Accuracy of the Train model  {0} %, F1-score: {1}'.format(100 * train_correct[0] / train_total[0], f1_train_total.avg))