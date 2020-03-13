from utils import adjust_learning_rate, checkpoint, get_variable, get_acc
from helper import AverageMeter, time, diffseq, seqLength
from definitions import *
import numpy as np
import torch
from sklearn.metrics import f1_score
from utils import Logger, ReshapeLabel_2
import os


def val_epoch(epoch, valition_loader, model, loss_function, optimizer, val_loss, Wtiter):
    print('validation at epoch {}'.format(epoch+1))

    model.eval()
    f1_val = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    f1score = AverageMeter()

    end_time = time()
    test_pred = np.empty((0))
    test_true = np.empty((0))
    with torch.no_grad():
        for i, (seqs, labels) in enumerate(valition_loader):
            # measure data loading time
            data_time.update(time() - end_time)

            # seqs = diffseq(seqs)
            # labels = BlockSliding(labels, True)
            seqs, seq_len = seqLength(seqs)
            seq_len = get_variable(seq_len.long())
            seq_y = ReshapeLabel_2(labels)
            seq_y = get_variable(seq_y)

            seqs = get_variable(seqs.float())  # 20 1 248 1
            labels = get_variable(labels.float())  # 20 248
            # labels = labels.view(-1, Short_Length)
            # Y_category, Y_location = ReshapeLabel(labels)


            # period = torch.mean(period, dim=1)
            # output, atten = model(seqs)
            output, atten = model(seqs, seq_len, seq_y)
            # print(outputs[1:3, :])

            loss_output, loss_atten, loss_total = loss_function(output, atten, labels)

            if len(val_loss) == 0 or loss_output < min(val_loss):
                checkpoint(epoch, model, optimizer)
            # f1 = f1_score(Y_location.detach().numpy(), x_location.detach().numpy(), average='weighted')
            acc_location_con, acc_location_teo = get_acc(output, labels)
            # acc_location = get_acc(x_location, Y_location)
            # accuracies.update(acc_categroy)
            # f1score.update(f1)

            print('-------label----------')
            print(labels[-1])
            print('-------pred----------')
            print(output[-1])
            print('-----------------')



            Wtiter.add_scalar('ValidateAccCon', acc_location_con, epoch)
            Wtiter.add_scalar('ValidateLossTotal', loss_total, epoch)
            Wtiter.add_scalar('ValidateAccTeo', acc_location_teo, epoch)
            # Wtiter.add_scalar('ValidateLossLong', loss_long, epoch)
            Wtiter.add_scalar('ValidateLossLocation', loss_output, epoch)
            Wtiter.add_scalar('ValidateLossAtten', loss_atten, epoch)

            batch_time.update(time() - end_time)
            end_time = time()


            # if (i + 1) % 10 == 0:
            print(
                    'Epoch [%d/%d], Validation_Iter [%d/%d] loss_output: %.6f, loss_atten: %.6f,  loss_total: %.6f, AccCon: %.6f,  AccTeo: %.6f,Time: %.3f '
                    % (epoch + 1, EPOCH, i + 1, len(valition_loader), loss_output.item(), loss_atten.item(), loss_total.item(), acc_location_con, acc_location_teo,
                       batch_time.val))
