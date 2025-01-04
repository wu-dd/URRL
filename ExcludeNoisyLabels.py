import math
from operator import itemgetter
import logging
import torch
import visdom
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import TensorDataset
from utils.EarlyStopping import EarlyStopping
from utils.PartialLoss import pll_estimator
from utils.Util import Util
from utils.DataUtil import DataUtil
from data_analyzer import sampleSelection, loss_analyzer, Acc_analyzer, excluded_loss_analyzer

logger = logging.getLogger(__name__)


class ExcludeNoisyLabels:
    def __init__(self,
                 eid,
                 select_learning_rate,
                 select_weight_decay,
                 select_batch_size,
                 select_epoch,
                 dataset,
                 select_model,
                 select_scheduler,
                 select_loss,
                 partial_type,
                 partial_rate,
                 noisy_rate,
                 device,
                 exclude_rate=0.03,
                 visdom_port=49999,
                 select_ratio=2,
                 exclude=True,
                 exclusion_mode='adaptive',
                 fix_exclusion_rate=0.5,
                 patience=5,
                 reinit='every_step',
                 args=None
                 ):
        self.eid = eid
        self.select_learning_rate = select_learning_rate
        self.select_weight_decay = select_weight_decay
        self.select_batch_size = select_batch_size
        self.select_epoch = select_epoch
        self.dataset = dataset
        self.select_model = select_model
        self.select_scheduler = select_scheduler
        self.select_loss = select_loss
        self.partial_type = partial_type
        self.partial_rate = partial_rate
        self.noisy_rate = noisy_rate
        self.device = device
        self.exclude_rate = exclude_rate
        self.exclusion_mode = exclusion_mode
        if exclusion_mode == 'fix':
            self.exclude_times = round(select_ratio * math.log(1 - fix_exclusion_rate, 1 - self.exclude_rate))
        elif exclusion_mode == 'known':
            self.exclude_times = round(select_ratio * math.log(1 - self.noisy_rate, 1 - self.exclude_rate))
        else:
            self.exclude_times = round(2 * math.log(1 - 0.4, 1 - self.exclude_rate))
        self.visdom_port = visdom_port
        self.exclude = exclude
        # general small loss
        if self.exclude == 'small_loss':
            self.exclude_times = 1
            self.exclude_rate = self.noisy_rate + 0.05

        self.dataUtil = DataUtil(dataset_name=self.dataset,
                                 batch_size=self.select_batch_size,
                                 partial_type=self.partial_type,
                                 partial_rate=self.partial_rate,
                                 noisy_rate=self.noisy_rate,
                                 args=args,
                                 noise_type=args.noise_type)
        self.ordinary_train_loader, \
        self.partial_train_loader, \
        self.test_loader, \
        self.partial_train_dataset, \
        self.test_dataset, \
        self.valid_dataset, \
        self.valid_loader = self.dataUtil.getDataLoaders()
        self.num_classes = self.dataUtil.getNumClasses()
        self.feature_dim = self.dataUtil.getFeatureDim()
        self.origin_partial_train_loader = self.partial_train_loader
        self.patience = patience
        self.loss_fn = Util.loss_func_select(self.select_loss)
        self.reinit = reinit

        """link to visdom"""
        if args.visdom:
            self.vis = visdom.Visdom(port=self.visdom_port, env='{}'.format(self.eid))
        else:
            self.vis = None

    def getTestAndValidDataset(self):
        return self.test_dataset, self.valid_dataset

    def getPartialTrainDataset(self):
        return self.partial_train_dataset.images, self.partial_train_dataset.partial_labels

    def getNumClasses(self):
        return self.num_classes

    def getFeatureDim(self):
        return self.feature_dim

    def getNewDataset(self):
        excluded_data = []
        earlyStopping = EarlyStopping(patience=self.patience,
                                      verbose=True,
                                      delta=0,
                                      trace_func=logger.info)
        if self.reinit == 'once':
            # init model
            model = Util.model_select(model_name=self.select_model,
                                      num_classes=self.num_classes,
                                      feature_dim=self.feature_dim)
            model.to(self.device)
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=self.select_learning_rate,
                                        weight_decay=self.select_weight_decay,
                                        momentum=0.9)
            scheduler = MultiStepLR(optimizer,
                                    milestones=self.select_scheduler,
                                    gamma=0.1)

        for j in range(self.exclude_times):
            if self.reinit == 'every_step':
                # init model
                model = Util.model_select(model_name=self.select_model,
                                          num_classes=self.num_classes,
                                          feature_dim=self.feature_dim)
                model.to(self.device)
                optimizer = torch.optim.SGD(model.parameters(),
                                            lr=self.select_learning_rate,
                                            weight_decay=self.select_weight_decay,
                                            momentum=0.9)
                scheduler = MultiStepLR(optimizer,
                                        milestones=self.select_scheduler,
                                        gamma=0.1)
            final_valid_acc = 0

            for epoch in range(self.select_epoch):
                model.train()
                clean_loss = torch.Tensor()
                noisy_loss = torch.Tensor()
                clean_true_labels = torch.Tensor()
                noisy_true_labels = torch.Tensor()
                clean_images = torch.Tensor()
                noisy_images = torch.Tensor()
                clean_partial_labels = torch.Tensor()
                noisy_partial_labels = torch.Tensor()

                for i, (images, partial_labels, label_flags, true_labels) in enumerate(self.partial_train_loader):
                    X, Y = images.to(self.device), partial_labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(X)
                    if self.select_loss not in ['jinb', 'courb', 'jinu', 'couru']:
                        total_loss = pll_estimator(self.loss_fn, outputs, Y.float(), self.device)
                    else:
                        total_loss = self.loss_fn(outputs, Y.float())

                    clean_loss = torch.cat((clean_loss, total_loss.cpu().detach()[~label_flags]), dim=0)
                    noisy_loss = torch.cat((noisy_loss, total_loss.cpu().detach()[label_flags]), dim=0)
                    clean_images = torch.cat((clean_images, images[~label_flags]), dim=0)
                    noisy_images = torch.cat((noisy_images, images[~label_flags]), dim=0)
                    clean_true_labels = torch.cat((clean_true_labels, true_labels[~label_flags]), dim=0)
                    noisy_true_labels = torch.cat((noisy_true_labels, true_labels[label_flags]), dim=0)
                    clean_partial_labels = torch.cat((clean_partial_labels, partial_labels[~label_flags]), dim=0)
                    noisy_partial_labels = torch.cat((noisy_partial_labels, partial_labels[label_flags]), dim=0)

                    average_loss = total_loss.mean()
                    average_loss.backward()
                    optimizer.step()
                    scheduler.step(epoch)

                model.eval()
                # train_accuracy, train_true_weight = Util.accuracy_check(loader=self.ordinary_train_loader,
                #                                                         model=model,
                #                                                         device=self.device)
                train_accuracy, train_true_weight = Util.partial_accuracy_check(loader=self.origin_partial_train_loader,
                                                                                model=model,
                                                                                device=self.device)
                valid_accuracy, valid_true_weight = Util.accuracy_check(loader=self.valid_loader,
                                                                        model=model,
                                                                        device=self.device)
                logger.info(
                    '''Exclude step: {}. Epoch: {}. Train Acc: {:.4f}. Test Acc: {:.4f}. Train tW: {:.4f}. Test tW: {:.4f}.'''.format(
                        j + 1,
                        epoch + 1,
                        train_accuracy,
                        valid_accuracy,
                        train_true_weight,
                        valid_true_weight))

                epoch_loss_data = []
                for i in range(clean_loss.shape[0]):
                    epoch_loss_data.append([clean_loss[i],
                                            False,
                                            clean_true_labels[i],
                                            clean_images[i],
                                            clean_partial_labels[i]])
                for i in range(noisy_loss.shape[0]):
                    epoch_loss_data.append([noisy_loss[i],
                                            True,
                                            noisy_true_labels[i],
                                            noisy_images[i],
                                            noisy_partial_labels[i]])

                epoch_loss_data = sorted(epoch_loss_data, key=itemgetter(0), reverse=True)

                sampleSelection(epoch_loss_data,
                                self.vis,
                                j * self.select_epoch + epoch + 1,
                                self.eid,
                                exclude_step=j + 1)
                if self.vis:
                    Acc_analyzer(j * self.select_epoch + epoch + 1,
                                 self.vis,
                                 train_accuracy,
                                 valid_accuracy,
                                 self.eid)
                    loss_analyzer(j * self.select_epoch + epoch + 1,
                                  self.vis,
                                  clean_loss,
                                  noisy_loss,
                                  self.eid)

                final_valid_acc = valid_accuracy

            split_value = int(self.exclude_rate * len(epoch_loss_data))
            if self.vis:
                excluded_loss_analyzer(j, self.vis, epoch_loss_data[: split_value], self.eid)
            excluded_data += epoch_loss_data[: split_value]
            epoch_loss_data = epoch_loss_data[split_value:]
            sample_num = len(epoch_loss_data)
            all_images = torch.zeros((sample_num,
                                      epoch_loss_data[0][3].shape[0],
                                      epoch_loss_data[0][3].shape[1],
                                      epoch_loss_data[0][3].shape[2]))
            all_partial_labels = torch.zeros((sample_num, self.num_classes))
            all_label_flags = torch.zeros(sample_num, dtype=torch.bool)
            all_true_labels = torch.zeros((sample_num, self.num_classes))
            for i in range(sample_num):
                all_images[i, :, :, :] = epoch_loss_data[i][3]
                all_partial_labels[i] = epoch_loss_data[i][4]
                all_label_flags[i] = epoch_loss_data[i][1]
                all_true_labels[i] = epoch_loss_data[i][2]
            # create new partial train loader.
            new_partial_dataset = TensorDataset(all_images, all_partial_labels, all_label_flags, all_true_labels)
            self.partial_train_loader = torch.utils.data.DataLoader(dataset=new_partial_dataset,
                                                                    batch_size=self.select_batch_size,
                                                                    shuffle=True,
                                                                    drop_last=False)
            if j > 5:
                earlyStopping(final_valid_acc)
                if earlyStopping.early_stop:
                    logger.info("Exclusion stopped.")
                    break

        all_excluded_images = torch.zeros((len(excluded_data),
                                           excluded_data[0][3].shape[0],
                                           excluded_data[0][3].shape[1],
                                           excluded_data[0][3].shape[2]))
        all_excluded_partial_labels = torch.zeros((len(excluded_data), self.num_classes))
        all_excluded_label_flags = torch.zeros(len(excluded_data), dtype=torch.bool)
        all_excluded_true_labels = torch.zeros((len(excluded_data), self.num_classes))
        for i in range(len(excluded_data)):
            all_excluded_images[i, :, :, :] = excluded_data[i][3]
            all_excluded_partial_labels[i] = excluded_data[i][4]
            all_excluded_label_flags[i] = excluded_data[i][1]
            all_excluded_true_labels[i] = excluded_data[i][2]

        return all_images, all_partial_labels, all_excluded_images, all_excluded_partial_labels
