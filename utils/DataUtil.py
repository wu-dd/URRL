import itertools
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from scipy.special import comb
from torch.distributions import Binomial, Multinomial
from dataset.uci import UCI
from utils.PartialDataset import GeneralDataset, PartialDataset
from utils.Util import Util
from utils.dataset_kmnist import KuzushijiMnist
from torch.nn.functional import one_hot


class DataUtil:
    def __init__(self, dataset_name, batch_size, partial_type, partial_rate, noisy_rate, proportion=(4, 1, 1), args=None, noise_type='symmetric'):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.partial_type = partial_type
        self.partial_rate = partial_rate
        self.noisy_rate = noisy_rate
        self.noise_type = noise_type

        self.ordinary_train_dataset = None
        self.test_dataset = None

        if self.dataset_name == 'mnist':
            self.ordinary_train_dataset = dsets.MNIST(root='./data',
                                                      train=True,
                                                      transform=transforms.ToTensor(),
                                                      download=True)
            self.test_dataset = dsets.MNIST(root='./data',
                                            train=False,
                                            transform=transforms.ToTensor())
        elif self.dataset_name == 'kmnist':
            self.ordinary_train_dataset = KuzushijiMnist(root='./data',
                                                         train=True,
                                                         transform=transforms.ToTensor(),
                                                         download=True)
            self.test_dataset = KuzushijiMnist(root='./data',
                                               train=False,
                                               transform=transforms.ToTensor())
        elif self.dataset_name == 'fashion':
            self.ordinary_train_dataset = dsets.FashionMNIST(root='./data',
                                                             train=True,
                                                             transform=transforms.ToTensor(),
                                                             download=True)
            self.test_dataset = dsets.FashionMNIST(root='./data',
                                                   train=False,
                                                   transform=transforms.ToTensor())
        elif self.dataset_name == 'cifar10':
            train_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            self.ordinary_train_dataset = dsets.CIFAR10(root='./data',
                                                        train=True,
                                                        transform=train_transform,
                                                        download=True)
            self.test_dataset = dsets.CIFAR10(root='./data',
                                              train=False,
                                              transform=test_transform)
        elif self.dataset_name == 'cifar100':
            cifar100_mean = (0.5071, 0.4867, 0.4408)
            cifar100_std = (0.2675, 0.2565, 0.2761)
            train_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(cifar100_mean, cifar100_std)])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(cifar100_mean, cifar100_std)])
            self.ordinary_train_dataset = dsets.CIFAR100(root='./data',
                                                         train=True,
                                                         transform=train_transform,
                                                         download=True)
            self.test_dataset = dsets.CIFAR100(root='./data',
                                               train=False,
                                               transform=test_transform)
        elif self.dataset_name == 'uci':
            self.ordinary_dataset = UCI(root='./data/UCIDATA',
                                        name=args.uci_name)
        else:
            raise ValueError('Wrong dataset name!')

        self.feature_dim = 1
        if self.dataset_name not in ['uci']:
            self.ordinary_train_sample_num = self.ordinary_train_dataset.data.shape[0]
            self.num_classes = len(self.ordinary_train_dataset.classes)
            self.test_sample_num = self.test_dataset.data.shape[0]
            self.all_sample_num = self.ordinary_train_sample_num + self.test_sample_num
            self.all_dataset = torch.utils.data.ConcatDataset([self.ordinary_train_dataset, self.test_dataset])
            for i in self.ordinary_train_dataset.data.shape[1:]:
                self.feature_dim *= i
        else:
            self.num_classes = int(torch.max(self.ordinary_dataset.labels) + 1)
            self.all_sample_num = self.ordinary_dataset.labels.shape[0]
            self.all_dataset = self.ordinary_dataset
            for i in self.all_dataset.datas.shape[1:]:
                self.feature_dim *= i

        self.split_test_num = int(self.all_sample_num * proportion[2] / sum(proportion))
        self.split_valid_num = int(self.all_sample_num * proportion[1] / sum(proportion))
        self.split_train_sample_num = self.all_sample_num - self.split_valid_num - self.split_test_num
        self.split_train_dataset, self.split_valid_dataset, self.split_test_dataset = \
            torch.utils.data.random_split(self.all_dataset,
                                          [self.split_train_sample_num, self.split_valid_num, self.split_test_num],
                                          torch.manual_seed(args.seed))

    def getNumClasses(self):
        return self.num_classes

    def getFeatureDim(self):
        return self.feature_dim

    def getDataLoaders(self):
        full_train_loader = torch.utils.data.DataLoader(dataset=self.split_train_dataset,
                                                        batch_size=self.split_train_sample_num,
                                                        shuffle=False)
        for i, (train_data, train_labels) in enumerate(full_train_loader):
            pass
        train_onehot_labels = one_hot(train_labels, num_classes=self.num_classes).float()

        full_test_loader = torch.utils.data.DataLoader(dataset=self.split_test_dataset,
                                                       batch_size=self.split_test_num,
                                                       shuffle=False)
        for i, (test_data, test_labels) in enumerate(full_test_loader):
            pass
        test_onehot_labels = one_hot(test_labels, num_classes=self.num_classes).float()

        full_valid_loader = torch.utils.data.DataLoader(dataset=self.split_valid_dataset,
                                                        batch_size=self.split_valid_num,
                                                        shuffle=False)
        for i, (valid_data, valid_labels) in enumerate(full_valid_loader):
            pass
        valid_onehot_labels = one_hot(valid_labels, num_classes=self.num_classes).float()

        label_flags = None

        if self.partial_type == 'uset':
            partial_labels = self.generate_uniformset(train_labels)
        elif self.partial_type == 'ulabel':
            partial_labels = self.generate_uniformlabel(train_labels)
        elif self.partial_type == 'nset':
            partial_labels = self.generate_nonuniformset(train_labels)
        elif self.partial_type == 'ccnlabel1':
            partial_labels = self.generate_ccnlabel_1(train_labels)
        elif self.partial_type == 'ccnlabel5':
            partial_labels = self.generate_ccnlabel_5(train_labels)
        elif self.partial_type == 'noise+partial':
            partial_labels, label_flags = self.noisyPartial(train_labels,
                                                            partial_rate=self.partial_rate,
                                                            noisy_rate=self.noisy_rate)
        elif self.partial_type == 'partial+noise':
            partial_labels, label_flags = self.generate_partialnoise(train_labels,
                                                                     pr=self.partial_rate,
                                                                     nr=self.noisy_rate)
        else:
            raise ValueError('Partial type is wrong.')

        ordinary_train_dataset = GeneralDataset(train_data, train_onehot_labels)
        partial_train_dataset = PartialDataset(train_data,
                                               partial_labels,
                                               label_flags,
                                               train_onehot_labels)
        test_onehot_dataset = GeneralDataset(test_data, test_onehot_labels)
        valid_onehot_dataset = GeneralDataset(valid_data, valid_onehot_labels)
        test_dataset = GeneralDataset(test_data, test_labels)
        valid_dataset = GeneralDataset(valid_data, valid_labels)

        partial_train_loader = torch.utils.data.DataLoader(dataset=partial_train_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=True)
        ordinary_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_onehot_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_onehot_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False)

        print('The average number of candidate labels is: {}'.format(
            (partial_labels.sum() / self.split_train_sample_num)))
        return ordinary_train_loader, partial_train_loader, test_loader, partial_train_dataset, test_dataset, valid_dataset, valid_loader

    def generate_uniformset(self, train_labels):
        if torch.min(train_labels) > 1:
            raise RuntimeError('testError')
        elif torch.min(train_labels) == 1:
            train_labels = train_labels - 1

        K = torch.max(train_labels) - torch.min(train_labels) + 1
        n = train_labels.shape[0]
        cardinality = (2 ** K - 2).float()
        number = torch.tensor([comb(K, i + 1) for i in range(K - 1)]).float()
        frequency_dis = number / cardinality
        prob_dis = torch.zeros(K - 1)
        for i in range(K - 1):
            if i == 0:
                prob_dis[i] = frequency_dis[i]
            else:
                prob_dis[i] = frequency_dis[i] + prob_dis[i - 1]

        random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float()
        mask_n = torch.ones(n)
        partialY = torch.zeros(n, K)
        partialY[torch.arange(n), train_labels] = 1.0

        temp_num_partial_train_labels = 0

        for j in range(n):  # for each instance
            for jj in range(K - 1):
                if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                    temp_num_partial_train_labels = jj + 1  # number of candidate labels
                    mask_n[j] = 0

            temp_num_fp_train_labels = temp_num_partial_train_labels - 1  # number of negative labels
            candidates = torch.from_numpy(np.random.permutation(K.item())).long()
            candidates = candidates[candidates != train_labels[j]]
            temp_fp_train_labels = candidates[:temp_num_fp_train_labels]
            # temp_comp_train_labels = candidates[temp_num_fp_train_labels:]

            partialY[j, temp_fp_train_labels] = 1.0
        return partialY

    def generate_nonuniformset(self, train_labels):
        if torch.min(train_labels) > 1:
            raise RuntimeError('testError')
        elif torch.min(train_labels) == 1:
            train_labels = train_labels - 1

        K = torch.max(train_labels) - torch.min(train_labels) + 1
        n = train_labels.shape[0]

        # np.random.seed(0)
        frequency_dis = np.sort(np.random.uniform(1e-4, 1, 2 ** (K - 1) - 2))
        while len(set(frequency_dis)) < 2 ** (K - 1) - 2:
            frequency_dis = np.sort(np.random.uniform(1e-4, 1, 2 ** (K - 1) - 2))
        prob_dis = torch.ones(2 ** (K - 1) - 1)
        for i in range(2 ** (K - 1) - 2):
            prob_dis[i] = frequency_dis[i]

        # np.random.seed(0)
        random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float()
        mask_n = torch.ones(n)
        partialY = torch.zeros(n, K)
        partialY[torch.arange(n), train_labels] = 1.0

        d = {}
        for i in range(K):
            value = []
            for ii in range(1, K - 1):
                candidates = torch.arange(K).long()
                candidates = candidates[candidates != i].numpy().tolist()
                value.append(list(itertools.combinations(candidates, ii)))
            d[i] = Util.getnewList(value)

        temp_fp_train_labels = []
        for j in range(n):  # for each instance
            if random_n[j] <= prob_dis[0] and mask_n[j] == 1:
                mask_n[j] = 0
                continue
            for jj in range(1, 2 ** (K - 1) - 1):
                if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                    temp_fp_train_labels = d[train_labels[j].item()][jj - 1]
                    break
            partialY[j, temp_fp_train_labels] = 1.0

        return partialY

    def generate_uniformlabel(self, train_labels):
        K = torch.max(train_labels) - torch.min(train_labels) + 1
        n = train_labels.shape[0]
        partialY = torch.zeros(n, K)
        partialY[torch.arange(n), train_labels] = 1.0

        p = 0

        for i in range(n):
            # partialY[i, np.where(np.random.binomial(1, flip_prob, K)==1)] = 1.0
            partialY[i, np.where(np.random.binomial(1, p, K) == 1)] = 1.0
        return partialY

    def generate_nonuniformlabel(self, dataname, train_labels):
        K = torch.max(train_labels) - torch.min(train_labels) + 1
        n = train_labels.shape[0]
        partialY = torch.zeros(n, K)
        partialY[torch.arange(n), train_labels] = 1.0

        # np.random.seed(0)
        P = np.random.rand(K, K)
        for i in range(n):
            partialY[i, np.where(np.random.binomial(1, P[train_labels[i], :]) == 1)] = 1.0
        return partialY

    def generate_ccnlabel_1(self, train_labels):
        K = (torch.max(train_labels) - torch.min(train_labels) + 1).item()
        n = train_labels.shape[0]
        partialY = torch.zeros(n, K)
        partialY[torch.arange(n), train_labels] = 1.0

        p = 0.5

        # np.random.seed(0)
        P = np.eye(K)
        for idx in range(0, K):
            P[idx, (idx + 1) % K] = p
        for i in range(n):
            partialY[i, np.where(np.random.binomial(1, P[train_labels[i], :]) == 1)] = 1.0
        return partialY

    def generate_ccnlabel_5(self, train_labels):
        K = (torch.max(train_labels) - torch.min(train_labels) + 1).item()
        n = train_labels.shape[0]
        partialY = torch.zeros(n, K)
        partialY[torch.arange(n), train_labels] = 1.0

        p = 0.5

        # np.random.seed(0)
        P = np.eye(K)
        for idx in range(0, K):
            if (idx + 1) % K + 5 < K:
                P[idx, (idx + 1) % K:(idx + 1) % K + 5] = p
            else:
                P[idx, (idx + 1) % K:(idx + 1) % K + 5] = p
                P[idx, 0:(idx + 1) % K + 5 - K] = p
        for i in range(n):
            partialY[i, np.where(np.random.binomial(1, P[train_labels[i], :]) == 1)] = 1.0
        return partialY

    def generate_partialnoise(self, train_labels, pr=0.1, nr=0.25):
        K = torch.max(train_labels) - torch.min(train_labels) + 1
        n = train_labels.shape[0]
        partialY = torch.zeros(n, K)
        partialY[torch.arange(n), train_labels] = 1.0

        for i in range(n):
            # partialY[i, np.where(np.random.binomial(1, flip_prob, K)==1)] = 1.0
            partialY[i, np.where(np.random.binomial(1, pr, K) == 1)] = 1.0

        complementary = torch.ones(1, K)
        """label flag 0 is ordinary and 1 is noisy."""
        label_flags = torch.zeros(n, dtype=torch.bool)
        for i in range(n):
            flag = np.random.binomial(1, nr)
            if flag == 1:
                partialY[i, :] = complementary - partialY[i, :]
                label_flags[i] = True
        return partialY, label_flags

    def noisyPartial(self, true_labels, partial_rate, noisy_rate):

        noisy_partial_labels = None
        if self.noise_type == 'pairflip':
            noisy_partial_labels = self.noisify_pairflip(true_labels,
                                                         noisy_rate,
                                                         self.num_classes,
                                                         self.split_train_sample_num)
        if self.noise_type == 'symmetric':
            noisy_partial_labels = self.noisify_multiclass_symmetric(true_labels,
                                                                     self.split_train_sample_num)

        """label flag 0 is clean and 1 is noisy."""
        label_flags = torch.zeros(self.split_train_sample_num, dtype=torch.bool)

        binomial = Binomial(1, probs=torch.tensor([partial_rate for _ in range(self.num_classes)]))
        for i in range(self.split_train_sample_num):
            noisy_partial_labels[i, binomial.sample() == 1.] = 1.
            label_flags[i] = False if noisy_partial_labels[i, true_labels[i]] == 1 else True

        return noisy_partial_labels, label_flags

    def noisify_multiclass_symmetric(self, labels, labels_number):
        P = torch.ones(self.num_classes, self.num_classes)
        P = (self.noisy_rate / (self.num_classes - 1)) * P
        for i in range(self.num_classes):
            P[i, i] = 1. - self.noisy_rate

        noisy_labels = torch.zeros(labels_number, self.num_classes)
        for i in range(labels_number):
            m = Multinomial(1, P[labels[i], :])
            noisy_labels[i, :] = m.sample()

        return noisy_labels

    def noisify_pairflip(self, y_train, noise, nb_classes, number):
        P = np.eye(nb_classes)
        n = noise
        for i in range(0, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = y_train.clone()
        for idx in np.arange(number):
            i = y_train[idx]
            flipped = np.random.multinomial(1, P[i, :], 1)[0]
            y_train_noisy[idx] = torch.from_numpy(np.where(flipped == 1)[0]).long()

        y_train_noisy = one_hot(y_train_noisy)

        return y_train_noisy
