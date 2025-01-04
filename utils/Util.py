import numpy as np
import torch
from exclude_models import densenet, resnet
from utils.PartialLoss import jin_lossb, jin_lossu, cour_lossb, cour_lossu, mae_loss, mse_loss, gce_loss, \
    phuber_ce_loss, cce_loss, focal_loss
from utils.models import mlp_model, linear_model, LeNet
from torch.nn import functional as F


class Util:
    @staticmethod
    def model_select(model_name, num_classes, feature_dim=None):
        if model_name == 'mlp':
            model = mlp_model(input_dim=feature_dim, output_dim=num_classes)
        elif model_name == 'linear':
            model = linear_model(input_dim=feature_dim, output_dim=num_classes)
        elif model_name == 'lenet':
            model = LeNet(output_dim=num_classes)
        elif model_name == 'densenet':
            model = densenet(num_classes=num_classes)
        elif model_name == 'resnet':
            model = resnet(depth=32, num_classes=num_classes)
        else:
            raise ValueError('Wrong model type.')
        return model

    @staticmethod
    def loss_func_select(loss):
        if loss == 'mae':
            loss_fn = mae_loss
        elif loss == 'mse':
            loss_fn = mse_loss
        elif loss == 'cce':
            loss_fn = cce_loss
        elif loss == 'gce':
            loss_fn = gce_loss
        elif loss == 'phuber_ce':
            loss_fn = phuber_ce_loss
        elif loss == 'fl':
            loss_fn = focal_loss
        elif loss == 'jinb':
            loss_fn = jin_lossb
        elif loss == 'courb':
            loss_fn = cour_lossb
        elif loss == 'jinu':
            loss_fn = jin_lossu
        elif loss == 'couru':
            loss_fn = cour_lossu
        else:
            raise ValueError('Wrong loss function type.')
        return loss_fn

    @staticmethod
    def create_plot_window(vis, xlabel, ylabel, title):
        return vis.line(X=np.array([1]), Y=np.array([np.nan]),
                        opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

    @staticmethod
    def getnewList(newlist):
        d = []
        for element in newlist:
            if not isinstance(element, list):
                d.append(element)
            else:
                d.extend(Util.getnewList(element))

        return d

    @staticmethod
    def accuracy_check(loader, model, device):
        with torch.no_grad():
            total, num_samples = 0, 0
            truew, totloss = 0.0, 0.0
            for images, labels in loader:
                labels, images = labels.to(device), images.to(device)
                outputs = model(images)
                w, predicted = torch.max(outputs.data, 1)
                _, y = torch.max(labels.data, 1)
                total += (predicted == y).sum().item()
                num_samples += labels.size(0)
                truew += w[predicted == y].sum().item()

        return 100 * (total / num_samples), (truew / total)

    @staticmethod
    def partial_accuracy_check(loader, model, device):
        with torch.no_grad():
            total, num_samples = 0, 0
            truew, totloss = 0.0, 0.0
            for images, partial_labels, _, _ in loader:
                partial_labels, images = partial_labels.to(device), images.to(device)
                outputs = model(images)
                w, predicted = torch.max(outputs.data, 1)
                # _, y = torch.max(labels.data, 1)
                # TODO Optimization
                total += (torch.diagonal(partial_labels[:, predicted])).sum().item()
                num_samples += partial_labels.size(0)
                truew += w[torch.diagonal(partial_labels[:, predicted]).long()].sum().item()

        return 100 * (total / num_samples), (truew / total)

    @staticmethod
    def confidence_update(model, confidence, batchX, batchY, batch_index):
        with torch.no_grad():
            batch_outputs = model(batchX)
            temp_un_conf = F.softmax(batch_outputs, dim=1)
            # un_confidence stores the weight of each example
            confidence[batch_index, :] = temp_un_conf * batchY
            # weight[batch_index] = 1.0/confidence[batch_index, :].sum(dim=1)
            base_value = confidence.sum(dim=1).unsqueeze(1).repeat(
                1, confidence.shape[1])
            confidence = confidence / base_value
        return confidence

    @staticmethod
    def confidence_update_lw(model, confidence, batchX, batchY, batch_index):
        with torch.no_grad():
            device = batchX.device
            batch_outputs = model(batchX)
            sm_outputs = F.softmax(batch_outputs, dim=1)

            onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1])
            onezero[batchY > 0] = 1
            counter_onezero = 1 - onezero
            onezero = onezero.to(device)
            counter_onezero = counter_onezero.to(device)

            new_weight1 = sm_outputs * onezero
            new_weight1 = new_weight1 / (new_weight1 + 1e-8).sum(dim=1).repeat(
                confidence.shape[1], 1).transpose(0, 1)
            new_weight2 = sm_outputs * counter_onezero
            new_weight2 = new_weight2 / (new_weight2 + 1e-8).sum(dim=1).repeat(
                confidence.shape[1], 1).transpose(0, 1)
            new_weight = new_weight1 + new_weight2

            confidence[batch_index, :] = new_weight
            return confidence