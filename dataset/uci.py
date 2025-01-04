import torch
import torch.utils.data as data
import scipy.io as sio
import os
from sklearn.preprocessing import scale
from torch import Tensor
from torch.utils.data import TensorDataset


class UCI(data.Dataset):
    def __init__(self, root, name):
        filename = os.path.join(root, (name + '.mat'))
        all_data = sio.loadmat(filename)

        self.datas = torch.from_numpy(scale(all_data['data'])).float()
        # consistency with images
        self.datas = self.datas.unsqueeze(1).unsqueeze(1)
        self.labels = torch.squeeze(torch.from_numpy(all_data['label']))

    def __getitem__(self, index):
        img, target = self.datas[index], self.labels[index]
        return img, target

    def __len__(self):
        return self.labels.shape[0]


class DatasetWithIndex(TensorDataset):
    def __init__(self, instances: Tensor, labels: Tensor):
        super().__init__(instances, labels)

        self.imgs, self.targets = instances, labels
        labels_sum = self.targets.sum(dim=1)
        for i in range(self.targets.shape[0]):
            self.targets[i] = self.targets[i] / labels_sum[i]

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        img = self.imgs[index]
        target = self.targets[index]
        return img, target, index
