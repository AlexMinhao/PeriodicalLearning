from utils import adjust_learning_rate, checkpoint, get_variable, get_acc
from helper import AverageMeter, time, diffseq, seqLength
from definitions import *
import numpy as np
import torch
from sklearn.metrics import f1_score
from utils import Logger, ReshapeLabel_2, mmd_custorm
import os
from sklearn.metrics import f1_score

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


def val_epoch_action(epoch, valition_loader, model, loss_function, optimizer, logger, total, correct, val_loss, f1_val_total, Writer):
    print('validation at epoch {}'.format(epoch+1))
    label_pred_log = Logger(
        os.path.join(os.path.join(os.getcwd(), r'results'), 'val_label_pred.log'), ['label', 'pred'])
    model.eval()
    f1_val = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_class = AverageMeter()
    losses_ae = AverageMeter()
    accuracies = AverageMeter()

    end_time = time()
    test_pred = np.empty((0))
    test_true = np.empty((0))
    with torch.no_grad():
        for i, (seqs, labels) in enumerate(valition_loader):
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
            loss = loss_classify + 1e-5 * loss_ae  # + 1.0 * loss_mmd
            losses.update(loss.data, seqs.size(0))
            losses_class.update(loss_classify.data, seqs.size(0))
            losses_ae.update(loss_ae.data, seqs.size(0))
            _, preds = torch.max(outputs.data, 1)





            # acc = get_acc(outputs, labels)
            accuracies.update(0, seqs.size(0))
            label_pred_log.log({'label': labels, 'pred': preds})
            total[0] += labels.size(0)
            labels_reshape = labels
            correct[0] += (preds == labels_reshape.data).sum()

            ifCorrect = np.array((preds == labels_reshape.data).cpu().numpy())
            failure_case_ind = np.where(ifCorrect == 0)
            label_for_failure_case = np.array(labels_reshape.cpu().numpy())
            label_for_pred_case = np.array(preds.cpu().numpy())
            failure_case_True_label = label_for_failure_case[failure_case_ind]
            failure_case_Pred_label = label_for_pred_case[failure_case_ind]
            print('Failure_case_True  {0} % '.format(failure_case_True_label))
            print('Failure_case_Pred  {0} % '.format(failure_case_Pred_label))
            f1_val.update(f1_score(labels_reshape.cpu().numpy(), preds.cpu().numpy(), average='weighted'))
            f1_val_total.update(f1_score(labels_reshape.cpu().numpy(), preds.cpu().numpy(), average='weighted'))

            batch_time.update(time() - end_time)
            end_time = time()
            test_pred = np.append(test_pred, preds.cpu().numpy(), axis=0)
            test_true = np.append(test_true, labels_reshape.cpu().numpy(), axis=0)

            if len(val_loss) == 0 or loss < min(val_loss):
                val_loss.append(loss)
                checkpoint(epoch, model, optimizer)

            if (i + 1) % 1 == 0:
                print(
                    'Epoch [%d/%d], Validation_Iter [%d/%d] Loss: %.6f,  Time: %.3f, F1-score: %.3f, F1-score.avg: %.3f '
                    % (epoch + 1, EPOCH, i + 1, len(valition_loader), loss.item(),
                       batch_time.val, f1_val.val, f1_val.avg))

        f1 = f1_score(test_true, test_pred, average='weighted')
        Writer.add_scalar('ValidateF1', f1, epoch)
        Writer.add_scalar('ValidateLoss', losses.avg, epoch)
        ############################################################
        Writer.add_scalar('ValidateLoss_class', losses_class.avg, epoch)
        Writer.add_scalar('ValidateLoss_ae', losses_ae.avg, epoch)
        ############################################################
        print('Accuracy of the Validation model F1-score: {0}'.format(f1))
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg, 'f1_score.avg': f1})


