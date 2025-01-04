import numpy as np
import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


def sampleSelection(epoch_loss_data, vis, epoch, eid, exclude_step=0):
    sample_num = len(epoch_loss_data)

    logger.info('GT: sum: {}, clean: {}, noisy: {}'.format(
        sample_num,
        sample_num - sum([i[1] for i in epoch_loss_data]),
        sum([i[1] for i in epoch_loss_data])))

    clean_noisy_num = np.zeros((10, 2))
    steps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(10):
        data = epoch_loss_data[int(steps[i] * sample_num): int(steps[i + 1] * sample_num)]
        clean_noisy_num[i][0] = sum([1 - i[1] for i in data])
        clean_noisy_num[i][1] = sum([i[1] for i in data])

    if vis:
        opts_bar = {
            "title": 'ID: {} Epoch: {} Exclude step:{}'.format(eid,
                                                               epoch,
                                                               exclude_step),
            "xlabel": 'Percentage',
            "ylabel": 'Acc',
            "stacked": True,
            "legend": ['Clean', 'Noisy'],
            "rownames": ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'],
        }

        vis.bar(
            X=clean_noisy_num,
            win='clean_noisy_num_{}'.format(exclude_step),
            opts=opts_bar
        )

    exclude_rate_sets = [0, 0.05, 0.1]
    line_num = len(exclude_rate_sets)
    remain_data = []
    true_remain_num = []
    for j in range(line_num):
        exclude_rate = exclude_rate_sets[j]
        remain_data.append(epoch_loss_data[int(exclude_rate * sample_num):])
        remain_num = len(remain_data[j])
        true_remain_num.append(remain_num - sum([i[1] for i in remain_data[j]]))

        logger.info('ER: {}%, select Acc.: {}% All: {}, excluded: {}, remain: {}.'.format(
            exclude_rate * 100,
            true_remain_num[j] / remain_num * 100,
            sample_num,
            sample_num - remain_num,
            remain_num))

    if vis:
        opts_selectAcc = {
            "title": 'ID: {} Clean label select Acc.'.format(eid),
            "xlabel": '#Epochs',
            "ylabel": 'Acc',
            "legend": ['{}'.format(i) for i in exclude_rate_sets],
        }
        vis.line(X=[epoch],
                 Y=[[(true_remain_num[i] / len(remain_data[i])) for i in range(line_num)]],
                 update='append', win='select_acc_window',
                 opts=opts_selectAcc)

    # not available for exclude_noise
    if exclude_step == 0 and vis:
        opts_selectPreAcc = {
            "title": 'ID: {} Clean label select Pre Acc.'.format(eid),
            "xlabel": '#Epochs',
            "ylabel": 'Acc',
            "legend": ['{}'.format(i) for i in exclude_rate_sets],
        }
        acc_true = [0 for _ in range(line_num)]
        for j in range(line_num):
            for i in remain_data[j]:
                if i[2] == i[3]:
                    acc_true[j] += 1

        vis.line(X=[epoch],
                 Y=[[acc_true[i] / len(remain_data[i]) for i in range(line_num)]],
                 update='append', win='select_pre_acc_window',
                 opts=opts_selectPreAcc)


def loss_analyzer(epoch, vis, clean_loss, noisy_loss, eid):
    # create_pic(epoch_loss_data, epoch)
    logger.info('Epoch: {}. Clean Ave Loss: {:.4f}. Noisy Ave Loss: {:.4f}.'.format(epoch + 1, torch.mean(clean_loss),
                                                                              torch.mean(noisy_loss)))
    opts_loss = {
        "title": 'ID: {} Clean and noisy loss.'.format(eid),
        "xlabel": '#Epochs',
        "ylabel": 'Loss',
        "legend": ['clean_loss', 'noisy_loss']
    }
    vis.line(X=[epoch],
             Y=[[torch.mean(clean_loss), torch.mean(noisy_loss)]],
             update='append', win='clean_loss_window',
             opts=opts_loss)


def excluded_loss_analyzer(exclude_times, vis, excluded_loss, eid):
    excluded_loss = torch.Tensor([excluded_loss[i][0] for i in range(len(excluded_loss))])
    logger.info('Exclude times: {}. Excluded Ave Loss: {:.4f}. '.format(exclude_times + 1, torch.mean(excluded_loss)))
    opts_loss = {
        "title": 'ID: {} Excluded loss.'.format(eid),
        "xlabel": '#Exclude times',
        "ylabel": 'Loss'
    }
    vis.line(X=[exclude_times],
             Y=[torch.mean(excluded_loss)],
             update='append', win='excluded_loss_window',
             opts=opts_loss)


def entropy_analyzer(epoch, vis, clean_entropy, noisy_entropy, args):
    # Ave entropy for noisy and clean samples
    logger.info('Epoch: {}. Clean Ave Entropy: {:.4f}. Noisy Ave Entropy: {:.4f}.'.format(epoch + 1, torch.mean(clean_entropy),
                                                                                    torch.mean(noisy_entropy)))
    opts_entropy = {
        "title": 'ID: {} Clean and noisy entropy.'.format(args.eid),
        "xlabel": '#Epochs',
        "ylabel": 'Loss',
        "legend": ['clean_entropy', 'noisy_entropy']
    }
    vis.line(X=[epoch],
             Y=[[torch.mean(clean_entropy), torch.mean(noisy_entropy)]],
             update='append', win='entropy_window',
             opts=opts_entropy)


def Acc_analyzer(epoch, vis, train_accuracy, valid_accuracy, eid):
    """update visdom for Acc."""
    opts_acc = {
        "title": 'ID: {} train and valid Acc.'.format(eid),
        "xlabel": '#Epochs',
        "ylabel": 'Acc',
        "legend": ['Train Acc', 'Valid Acc']
    }
    vis.line(X=[epoch],
             Y=[[train_accuracy, valid_accuracy]],
             update='append', win='acc_window',
             opts=opts_acc)


def pre_Acc_Analyzer(all_images_dataloader, model, device):
    all_test_noisy_res = np.array([])
    all_test_clean_res = np.array([])
    softmax = nn.Softmax(dim=1)
    for i, (images, labels, true_labels, label_flags) in enumerate(all_images_dataloader):
        tmp_res = model(images.to(device)).cpu()
        tmp_res = softmax(tmp_res)
        tmp_res = torch.argmax(tmp_res, dim=1)
        all_test_clean_res = np.append(all_test_clean_res, tmp_res.detach()[~label_flags])
        all_test_noisy_res = np.append(all_test_noisy_res, tmp_res.detach()[label_flags])
    return all_test_clean_res, all_test_noisy_res
