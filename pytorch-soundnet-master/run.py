from helper import sliding_window, AverageMeter
from helper import load_imu_data, segMent,ToDirection
from soundnet import SoundNet
import numpy as np
from utils import SeqGenerator, distance, LongShortdistance
from torch.utils.data import DataLoader
from validate import *
from trainer import *
from definitions import *
from torch import nn
from torch import optim
import torch
from utils import Logger, data_augmentation
import os
from tensorboardX import SummaryWriter
from model import  Seq2Seq, DeepConvLSTM, ae_spatial_LSTM_CNN
from Opportunity import *
from sklearn.metrics import f1_score
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
        a_rot = data_augmentation(q, a, times=1)

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
    writer = SummaryWriter(comment = '316ActionRcog_Deep_LSTM_CNN_113_10e-6')

##################Runner#############################################
    # train_x, train_y = load_dataset('C:\\ALEX\\Doc\\Reference\\SoundNet\\PeriodicalLearning\\Dataset\\Runner\\imu_train_data.json',FixLength = False, DataAugmentation=True)
    # training_set = SeqGenerator(20, train_x, train_y).getDataSet()
    # train_loader = DataLoader(dataset = training_set, batch_size=10, shuffle=True)
    #
    # validate_x, validate_y = load_dataset('C:\\ALEX\\Doc\\Reference\\SoundNet\\PeriodicalLearning\\Dataset\\Runner\\imu_validate_data.json', FixLength=False)
    # validate_set = SeqGenerator(20, validate_x, validate_y).getDataSet()
    # validation_loader = DataLoader(dataset=validate_set, batch_size=50, shuffle=True)

##################Action Recog#############################################

    path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                        r'PeriodicalLearning\Dataset\OppSegBySubjectGesturesFull_113Validation.data')
    print("Loading data...")
    Opp = OPPORTUNITY(path)
    X_train, y_train, X_validation, y_validation, X_test, y_test = Opp.load()  # load_dataset(dp)
    assert CHANNELS_OBJECT == X_train.shape[2]

    print(" ..after sliding window (training): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
    print(
        " ..after sliding window (validation): inputs {0}, targets {1}".format(X_validation.shape, y_validation.shape))
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

    # Data is reshaped since the input of the network is a 4 dimension tensor
    X_test = X_test.reshape((-1, 1, 24, CHANNELS_OBJECT))
    X_test.astype(np.float32), y_test.reshape(len(y_test)).astype(np.uint8)

    X_train = X_train.reshape(
        (-1, 1, 24, CHANNELS_OBJECT))  # inputs (46495, 1, 24, 51), targets (46495,)
    X_train.astype(np.float32), y_train.reshape(len(y_train)).astype(np.uint8)

    X_validation = X_validation.reshape((-1, 1, 24, CHANNELS_OBJECT))
    X_validation.astype(np.float32), y_validation.reshape(len(y_validation)).astype(np.uint8)

    print(" after reshape: inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
    training_set = []
    validation_set = []
    testing_set = []

    # X_train = list(X_train)
    # y_train = list(y_train)

    nullclass_index = np.argwhere(y_train == 0)
    X_train = list(np.delete(X_train, nullclass_index, axis=0))
    y_train = list(np.delete(y_train, nullclass_index))

    for i in range(len(y_train)):
        x = X_train[i]
        y = y_train[i]
        xy = (x, y)
        training_set.append(xy)


    X_validation = list(X_validation)
    y_validation = list(y_validation)
    for i in range(len(y_validation)):
        x = X_validation[i]
        y = y_validation[i]
        xy = (x, y)
        validation_set.append(xy)
    X_test = list(X_test)
    y_test = list(y_test)
    for j in range(len(y_test)):
        x = X_test[j]
        y = y_test[j]
        xy = (x, y)
        testing_set.append(xy)

    train_loader = DataLoader(dataset=training_set, batch_size= BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=testing_set, batch_size=BATCH_SIZE, shuffle=True)

    result_path = os.path.join(os.getcwd(), r'log')
    train_logger = Logger(
        os.path.join(result_path, 'train.log'),
        ['epoch', 'loss', 'acc', 'lr', 'f1_score.avg'])
    train_batch_logger = Logger(
        os.path.join(result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    val_logger = Logger(
        os.path.join(result_path, 'val.log'), ['epoch', 'loss', 'acc', 'f1_score.avg'])



    model = DeepConvLSTM() #ae_spatial_LSTM_CNN() #


    # If use CrossEntropyLoss，softmax wont be used in the final layer
    loss_function = [nn.CrossEntropyLoss(), nn.MSELoss()]
    optimizer = optim.RMSprop(model.parameters(), lr=BASE_lr, momentum=0.9, weight_decay=0.9)



    if torch.cuda.is_available():
        model.cuda()
        loss_function[0].cuda()
        loss_function[1].cuda()
        print("Model on gpu")

    if pretrain:
        pre_train_path = os.path.join(os.getcwd(),
                                      r'results\model_best.pth')
        pretrain_model = torch.load(pre_train_path)
        model.load_state_dict(pretrain_model['state_dict'])
        optimizer.load_state_dict(pretrain_model['optimizer'])
        print(model)


    # training and testing
    train_correct = [0, ]
    train_total = [0, ]
    f1_train_total = AverageMeter()
    val_correct = [0, ]
    val_total = [0, ]
    f1_val_total = AverageMeter()
    val_loss = []
    for epoch in range(EPOCH):
        # train_epoch(epoch, train_loader, model, LongShortdistance, optimizer,
        #             train_logger, train_batch_logger, writer)
        # val_epoch(epoch, validation_loader, model, LongShortdistance, optimizer, val_loss, writer)

        train_epoch_action(epoch, train_loader, model, loss_function, optimizer,
                    train_logger, train_batch_logger, train_total, train_correct, f1_train_total, writer)
        val_epoch_action(epoch, validation_loader, model, loss_function, optimizer,val_logger, val_total, val_correct,val_loss, f1_val_total, writer)

    # print('Accuracy of the Train model  {0} %, F1-score: {1}'.format(100 * train_correct[0] / train_total[0], f1_train_total.avg))

    # Test the model ####################################
    f1_test = AverageMeter()
    accuracies = AverageMeter()
    data_time = AverageMeter()
    end_time = time()
    model.eval()
    correct = 0
    total = 0
    test_pred = np.empty((0))
    test_true = np.empty((0))
    for i, (seqs, labels) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time() - end_time)

        seqs = get_variable(seqs.float())
        labels = get_variable(labels.long())
        ######################################################################
        # seqs = torch.squeeze(seqs)
        # seqs = seqs.permute(0, 2, 1).contiguous()
        #
        # outputs, out_decoder = model(seqs)
        #####################################################################

        outputs = model(seqs)

        labels = labels.squeeze()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        labels_reshape = labels
        correct += (predicted == labels_reshape.data).sum()

        ifCorrect = np.array((predicted == labels_reshape.data).cpu().numpy())
        failure_case_ind = np.where(ifCorrect == 0)
        label_for_failure_case = np.array(labels_reshape.cpu().numpy())
        label_for_pred_case = np.array(predicted.cpu().numpy())
        failure_case_True_label = label_for_failure_case[failure_case_ind]
        failure_case_Pred_label = label_for_pred_case[failure_case_ind]
        print('Failure_case_True  {0} % '.format(failure_case_True_label))
        print('Failure_case_Pred  {0} % '.format(failure_case_Pred_label))

        f1_test.update(f1_score(labels_reshape.cpu().numpy(), predicted.cpu().numpy(), average='weighted'))
        test_pred = np.append(test_pred, predicted.cpu().numpy(), axis=0)
        test_true = np.append(test_true, labels_reshape.cpu().numpy(), axis=0)

    f1 = f1_score(test_true, test_pred, average='weighted')
    print('Test Accuracy of the model  {0}%, F1-score {1}%'.format(100 * correct / total, f1))