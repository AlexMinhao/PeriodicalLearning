from utils import adjust_learning_rate, checkpoint, get_variable, get_acc
from helper import AverageMeter, time, diffseq, seqLength
from definitions import *
import numpy as np
import torch
from sklearn.metrics import f1_score
from utils import Logger, ReshapeLabel_2, mmd_custorm
import os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def train_epoch(epoch, train_loader, model, loss_function, optimizer,
                train_logger, train_batch_logger, Wtiter):
    print('train at epoch {}'.format(epoch+1))
    label_pred_log = Logger(
        os.path.join(os.path.join(os.getcwd(), r'log'), 'train_label_pred.log'), ['label', 'pred'])
    model.train()
    f1_train = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time()
    adjust_learning_rate(optimizer, epoch)
    test_pred = np.empty((0))
    test_true = np.empty((0))
    for i, (seqs, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time() - end_time)

        # seqs = diffseq(seqs)
        # labels = BlockSliding(labels, True)
        seqs, seq_len = seqLength(seqs)
        seq_len = get_variable(seq_len.long())
        seq_y = ReshapeLabel_2(labels)
        seq_y = get_variable(seq_y)

        seqs = get_variable(seqs.float())   # 10 1 3 72
        labels = get_variable(labels.float())  #10  72
        # labels = labels.view(-1, Short_Length)


        # period = torch.mean(period, dim = 1)
        # output, atten = model(seqs)
        output, atten = model(seqs, seq_len,seq_y)
        # print(outputs[1:3, :])

        loss_output, loss_atten, loss_total = loss_function(output, atten, labels)
        # _, preds = torch.max(short.data, 1)
        # _, labelindex = torch.max(labels.data, 1)
        # acc_categroy = get_acc(x_atten_selection, Y_category)
        acc_location_con, acc_location_teo = get_acc(output, labels)

        optimizer.zero_grad()
        # if attention:
        #     loss_long.backward(retain_graph=True)
        # loss_location.backward(retain_graph=True)
        # loss_category.backward(retain_graph=True)
        loss_output.backward()
        optimizer.step()



        Wtiter.add_scalar('TrainAccCon', acc_location_con, epoch)
        Wtiter.add_scalar('TrainLossTotal', loss_total, epoch)
        Wtiter.add_scalar('TrainAccTeo', acc_location_teo, epoch)
        Wtiter.add_scalar('TrainLossLocation', loss_output, epoch)
        Wtiter.add_scalar('TrainLossAtten', loss_atten, epoch)
        # Store loss data and epoch
        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']


        Wtiter.add_scalar('LR', lr, epoch)


        batch_time.update(time() - end_time)
        end_time = time()

        train_batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(train_loader) + (i + 1),
            'loss': loss_output.item(),
            'acc': acc_location_teo,
            'lr': optimizer.param_groups[0]['lr'],
            'Failure_case_True': 0,
            'Failure_case_Pred': 0
        })


        if (i + 1) % 10 == 0:
            print('-------label Train----------')
            print(labels[-1])
            print('-------pred Train----------')
            print(output[-1])
            print('-----------------')

            print(
                'Epoch [%d/%d], Train_Iter [%d/%d] loss_output: %.6f, loss_atten: %.6f,  loss_total: %.6f,  AccCon: %.6f,  AccTeo: %.6f, Time: %.3f, lr: %.7f '
                % (epoch + 1, EPOCH, i + 1, len(train_loader), loss_output.item(), loss_atten.item(), loss_total.item(), acc_location_con, acc_location_teo,
                   batch_time.val, optimizer.param_groups[0]['lr']))

    # Wtiter.add_graph(model, (seqs,))

    # f1 = f1_score(test_true, test_pred, average='weighted')
    # print('Accuracy of the Train model F1-score: {0}'.format(f1))
    # train_logger.log({
    #     'epoch': epoch,
    #     'loss': losses.avg,
    #     'acc': 0,
    #     'lr': optimizer.param_groups[0]['lr'],
    #     'f1_score.avg': 0
    # })

    # if epoch % CHECK_POINTS == 0:
    #     checkpoint(epoch, model, optimizer)


def train_epoch_action(epoch, train_loader, model, loss_function, optimizer,
                train_logger, train_batch_logger, total, correct, f1_train_total, Writer):
    print('train at epoch {}'.format(epoch+1))
    label_pred_log = Logger(
        os.path.join(os.path.join(os.getcwd(), r'results'), 'train_label_pred.log'), ['label', 'pred'])
    model.train()
    f1_train = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_class = AverageMeter()
    losses_ae = AverageMeter()
    accuracies = AverageMeter()

    end_time = time()
    adjust_learning_rate(optimizer, epoch)
    test_pred = np.empty((0))
    test_true = np.empty((0))


    for i, (seqs, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time() - end_time)

        seqs = get_variable(seqs.float())
        labels = get_variable(labels.long())
        ######################################################################
        # seqs = torch.squeeze(seqs)
        # seqs = seqs.permute(0, 2, 1).contiguous()

        outputs, out_decoder = model(seqs)
        #####################################################################
        # outputs  = model(seqs)
        # print(outputs[1:3, :])
        labels = labels.squeeze()
        loss_classify = loss_function[0](outputs, labels)
        ############################################################
        loss_ae = loss_function[1](seqs.view(seqs.size(0), -1), out_decoder)
        # loss_mmd = mmd_custorm(seqs.view(seqs.size(0), -1), out_decoder)
        # loss_mmd = loss_mmd.cuda().float()
        ###################################################################
        loss = loss_classify  + 1e-5 * loss_ae #+ 1.0 * loss_mmd
        losses.update(loss.data, seqs.size(0))
        losses_class.update(loss_classify.data, seqs.size(0))
        losses_ae.update(loss_ae.data, seqs.size(0))
        _, preds = torch.max(outputs.data, 1)


        accuracies.update(0, seqs.size(0))

        label_pred_log.log({'label': labels, 'pred': preds})
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total[0] += labels.size(0)
        labels_reshape = labels
        correct[0] += (preds == labels_reshape.data).sum().item()

        #
        ifCorrect = np.array((preds == labels_reshape.data).cpu().numpy())
        failure_case_ind = np.where(ifCorrect == 0)
        label_for_failure_case = np.array(labels_reshape.cpu().numpy())
        label_for_pred_case = np.array(preds.cpu().numpy())
        failure_case_True_label = label_for_failure_case[failure_case_ind]
        failure_case_Pred_label = label_for_pred_case[failure_case_ind]
        if (i + 1) % 100 == 0:
            print('Failure_case_True  {0} % '.format(failure_case_True_label))
            print('Failure_case_Pred  {0} % '.format(failure_case_Pred_label))

        f1_train.update(f1_score(labels_reshape.cpu().numpy(), preds.cpu().numpy(), average='weighted'))
        f1_train_total.update(f1_score(labels_reshape.cpu().numpy(), preds.cpu().numpy(), average='weighted'))
        batch_time.update(time() - end_time)
        end_time = time()




        train_batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(train_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr'],

        })
        test_pred = np.append(test_pred, preds.cpu().numpy(), axis=0)
        test_true = np.append(test_true, labels_reshape.cpu().numpy(), axis=0)

        if (i + 1) % 10 == 0:
            print(
                'Epoch [%d/%d], Train_Iter [%d/%d] Loss: %.6f,  Time: %.3f, F1-score: %.3f, F1-score.avg: %.3f, lr: %.7f '
                % (epoch + 1, EPOCH, i + 1, len(train_loader), loss.item(),
                   batch_time.val, f1_train.val, f1_train.avg, optimizer.param_groups[0]['lr']))

    f1 = f1_score(test_true, test_pred, average='weighted')
    print('Accuracy of the Train model F1-score: {0}'.format(f1))
    train_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr'],
        'f1_score.avg': f1
    })
    Writer.add_scalar('Trainf1', f1, epoch)
    Writer.add_scalar('TrainLoss', losses.avg, epoch)
    ###############################################################################
    Writer.add_scalar('TrainLoss_class', losses_class.avg, epoch)
    Writer.add_scalar('TrainLoss_ae', losses_ae.avg, epoch)
    #############################################################################
    # if epoch % CHECK_POINTS == 0:
    #     checkpoint(epoch, model, optimizer)