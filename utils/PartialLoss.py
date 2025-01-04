import math
import numpy as np
import torch.nn.functional as func
import torch
from torch import nn
from torch.nn import functional as F


class PRODENPartialLoss(nn.Module):
    def __init__(self):
        super(PRODENPartialLoss, self).__init__()

    def forward(self, predict, target):
        output = func.softmax(predict, dim=1)
        l = target * torch.log(output)
        loss = (-torch.sum(l)) / l.size(0)

        revisedY = target.clone()
        revisedY[revisedY > 0] = 1
        revisedY = revisedY * output
        revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1)
        new_target = revisedY

        return loss, new_target


class partial_loss(nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99):
        super().__init__()
        self.confidence = confidence
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start

    def forward(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss

    def confidence_update(self, temp_un_conf, batch_index, batchY):
        with torch.no_grad():
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] \
                                              + (1 - self.conf_ema_m) * pseudo_label
        return None


class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning:
        https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss


def jin_lossb(outputs, partialY):
    Y = partialY / partialY.sum(dim=1, keepdim=True)
    q = 0.7
    sm_outputs = F.softmax(outputs, dim=1)
    pow_outputs = torch.pow(sm_outputs, q)
    sample_loss = (1 - (pow_outputs * Y).sum(dim=1)) / q
    return sample_loss


def jin_lossu(outputs, partialY):
    Y = partialY / partialY.sum(dim=1, keepdim=True)
    logsm = nn.LogSoftmax(dim=1)
    logsm_outputs = logsm(outputs)
    final_outputs = logsm_outputs * Y
    sample_loss = - final_outputs.sum(dim=1)
    return sample_loss


def cour_lossb(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    candidate_outputs = ((sm_outputs * partialY).sum(dim=1)) / (partialY.sum(dim=1))
    sig = nn.Sigmoid()
    candidate_loss = sig(candidate_outputs)
    noncandidate_loss = (sig(-sm_outputs) * (1 - partialY)).sum(dim=1)
    sample_loss = (candidate_loss + noncandidate_loss).mean()
    return sample_loss


def squared_hinge_loss(z):
    hinge = torch.clamp(1 - z, min=0)
    return hinge * hinge


def cour_lossu(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    candidate_outputs = ((sm_outputs * partialY).sum(dim=1)) / (partialY.sum(dim=1))
    candidate_loss = squared_hinge_loss(candidate_outputs)
    noncandidate_loss = (squared_hinge_loss(-sm_outputs) * (1 - partialY)).sum(dim=1)
    sample_loss = (candidate_loss + noncandidate_loss).mean()
    return sample_loss


def mae_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = nn.L1Loss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, partialY.float())
    sample_loss = loss_matrix.sum(dim=-1)
    sample_loss = torch.sum(sample_loss) / sample_loss.size(0)
    return sample_loss


def mse_loss(outputs, Y):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = nn.MSELoss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, Y.float())
    sample_loss = loss_matrix.sum(dim=-1)
    sample_loss = torch.sum(sample_loss) / sample_loss.size(0)
    return sample_loss


def gce_loss(outputs, Y):
    q = 0.7
    sm_outputs = F.softmax(outputs, dim=1)
    pow_outputs = torch.pow(sm_outputs, q)
    sample_loss = (1 - (pow_outputs * Y).sum(dim=1)) / q  # n
    sample_loss = torch.sum(sample_loss) / sample_loss.size(0)
    return sample_loss


def phuber_ce_loss(outputs, Y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trunc_point = 0.1
    n = Y.shape[0]
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * Y
    final_confidence = final_outputs.sum(dim=1)

    ce_index = (final_confidence > trunc_point)
    sample_loss = torch.zeros(n).to(device)

    if ce_index.sum() > 0:
        ce_outputs = outputs[ce_index, :]
        logsm = nn.LogSoftmax(dim=-1)  # because ce_outputs might have only one example
        logsm_outputs = logsm(ce_outputs)
        final_ce_outputs = logsm_outputs * Y[ce_index, :]
        sample_loss[ce_index] = - final_ce_outputs.sum(dim=-1)

    linear_index = (final_confidence <= trunc_point)

    if linear_index.sum() > 0:
        sample_loss[linear_index] = -math.log(trunc_point) + (-1 / trunc_point) * final_confidence[linear_index] + 1

    return sample_loss


def cce_loss(outputs, Y):
    logsm = nn.LogSoftmax(dim=1)
    logsm_outputs = logsm(outputs)
    final_outputs = logsm_outputs * Y
    sample_loss = - final_outputs.sum(dim=1)
    return sample_loss


def focal_loss(outputs, Y):
    logsm = nn.LogSoftmax(dim=1)
    logsm_outputs = logsm(outputs)
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = logsm_outputs * Y * (1 - sm_outputs) ** 0.5
    sample_loss = - final_outputs.sum(dim=1)
    return sample_loss


def pll_estimator(loss_fn, outputs, partialY, device):
    n, k = partialY.shape[0], partialY.shape[1]
    comp_num = partialY.sum(dim=1)
    temp_loss = torch.zeros(n, k).to(device)

    for i in range(k):
        tempY = torch.zeros(n, k).to(device)
        tempY[:, i] = 1.0
        temp_loss[:, i] = loss_fn(outputs, tempY)

    coef = 1.0 / comp_num
    total_loss = coef * (temp_loss * partialY).sum(dim=1)
    return total_loss


def entropy(outputs):
    outputs = outputs.cpu().detach()
    n = outputs.shape[0]
    # n_classes = outputs.shape[1]
    n_entropy = []
    softmax = nn.Softmax(dim=1)
    prob = softmax(outputs)
    for i in range(n):
        now_entropy = (-prob[i]) * torch.log(prob[i])
        n_entropy.append(torch.sum(now_entropy).item())
    return np.array(n_entropy)



def cc_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = -torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss


def rc_loss(outputs, confidence, index):
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * confidence[index, :]
    average_loss = -((final_outputs).sum(dim=1)).mean()
    return average_loss


def lws_loss(outputs, partialY, confidence, index, lw_weight, lw_weight0, epoch_ratio):
    device = outputs.device

    onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
    onezero[partialY > 0] = 1
    counter_onezero = 1 - onezero
    onezero = onezero.to(device)
    counter_onezero = counter_onezero.to(device)

    sig_loss1 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
    sig_loss1 = sig_loss1.to(device)
    sig_loss1[outputs < 0] = 1 / (1 + torch.exp(outputs[outputs < 0]))
    sig_loss1[outputs > 0] = torch.exp(-outputs[outputs > 0]) / (
        1 + torch.exp(-outputs[outputs > 0]))
    l1 = confidence[index, :] * onezero * sig_loss1
    average_loss1 = torch.sum(l1) / l1.size(0)

    sig_loss2 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
    sig_loss2 = sig_loss2.to(device)
    sig_loss2[outputs > 0] = 1 / (1 + torch.exp(-outputs[outputs > 0]))
    sig_loss2[outputs < 0] = torch.exp(
        outputs[outputs < 0]) / (1 + torch.exp(outputs[outputs < 0]))
    l2 = confidence[index, :] * counter_onezero * sig_loss2
    average_loss2 = torch.sum(l2) / l2.size(0)

    average_loss = lw_weight0 * average_loss1 + lw_weight * average_loss2
    return average_loss, lw_weight0 * average_loss1, lw_weight * average_loss2