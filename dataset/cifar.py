import logging
import math
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import TensorDataset
from torchvision import datasets
from torchvision import transforms
from ExcludeNoisyLabels import ExcludeNoisyLabels
from dataset.randaugment import RandAugmentMC
from dataset.uci import DatasetWithIndex
from external_models.RCR.augment.autoaugment_extra import CIFAR10Policy
from external_models.RCR.augment.cutout import Cutout

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):
    # transform_labeled = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(size=32,
    #                           padding=int(32 * 0.125),
    #                           padding_mode='reflect'),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    # ])
    transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.ToPILImage(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
        ])
    transform_general = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    if args.partial:
        excludeNoisyLabels = ExcludeNoisyLabels(eid=args.eid,
                                                select_learning_rate=args.select_learning_rate,
                                                select_weight_decay=args.select_weight_decay,
                                                select_batch_size=args.select_batch_size,
                                                select_epoch=args.select_epoch,
                                                dataset=args.dataset,
                                                select_model=args.select_model,
                                                select_scheduler=args.select_scheduler,
                                                select_loss=args.select_loss,
                                                partial_type=args.partial_type,
                                                partial_rate=args.partial_rate,
                                                noisy_rate=args.noisy_rate,
                                                exclude_rate=args.exclude_rate,
                                                device=args.device,
                                                select_ratio=args.select_ratio,
                                                exclude=args.exclusion,
                                                exclusion_mode=args.exclusion_mode,
                                                fix_exclusion_rate=args.fix_exclusion_rate,
                                                reinit=args.reinit,
                                                args=args
                                                )
        args.num_classes = excludeNoisyLabels.getNumClasses()
        args.feature_dim = excludeNoisyLabels.getFeatureDim()
        if args.exclusion in ['progressive_exclusion', 'small_loss']:
            all_images, all_partial_labels, all_excluded_images, all_excluded_partial_labels = excludeNoisyLabels.getNewDataset()

            train_labeled_dataset = PartialSelectDataset(cifar10_mean,
                                                         cifar10_std,
                                                         all_images,
                                                         all_partial_labels,
                                                         transform=transform_labeled if args.aug else transform_general
                                                         )

            train_unlabeled_dataset = PartialSelectDataset(cifar10_mean,
                                                           cifar10_std,
                                                           all_excluded_images,
                                                           all_excluded_partial_labels,
                                                           transform=TransformFixMatch(mean=cifar10_mean,
                                                                                       std=cifar10_std) if args.SSLAug else transform_general
                                                           )
        else:
            all_images, all_partial_labels = excludeNoisyLabels.getPartialTrainDataset()
            train_labeled_dataset = PartialSelectDataset(cifar10_mean,
                                                         cifar10_std,
                                                         all_images,
                                                         all_partial_labels,
                                                         transform=transform_general
                                                         )
            train_unlabeled_dataset = None
        test_dataset, valid_dataset = excludeNoisyLabels.getTestAndValidDataset()

    else:
        base_dataset = datasets.CIFAR10(root, train=True, download=True)

        train_labeled_idxs, train_unlabeled_idxs = x_u_split(
            args, base_dataset.targets)

        train_labeled_dataset = CIFAR10SSL(
            root, train_labeled_idxs, train=True,
            transform=transform_labeled)

        train_unlabeled_dataset = CIFAR10SSL(
            root, train_unlabeled_idxs, train=True,
            transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

        test_dataset = datasets.CIFAR10(
            root, train=False, transform=transform_val, download=True)

        valid_dataset = None

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, valid_dataset


def get_cifar100(args, root):
    # transform_labeled = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(size=32,
    #                           padding=int(32 * 0.125),
    #                           padding_mode='reflect'),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4, padding_mode='reflect'),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.ToPILImage(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
    ])
    transform_general = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    if args.partial:
        excludeNoisyLabels = ExcludeNoisyLabels(eid=args.eid,
                                                select_learning_rate=args.select_learning_rate,
                                                select_weight_decay=args.select_weight_decay,
                                                select_batch_size=args.select_batch_size,
                                                select_epoch=args.select_epoch,
                                                dataset=args.dataset,
                                                select_model=args.select_model,
                                                select_scheduler=args.select_scheduler,
                                                select_loss=args.select_loss,
                                                partial_type=args.partial_type,
                                                partial_rate=args.partial_rate,
                                                noisy_rate=args.noisy_rate,
                                                exclude_rate=args.exclude_rate,
                                                device=args.device,
                                                select_ratio=args.select_ratio,
                                                exclude=args.exclusion,
                                                exclusion_mode=args.exclusion_mode,
                                                fix_exclusion_rate=args.fix_exclusion_rate,
                                                reinit=args.reinit,
                                                args=args
                                                )
        args.num_classes = excludeNoisyLabels.getNumClasses()
        args.feature_dim = excludeNoisyLabels.getFeatureDim()
        if args.exclusion in ['progressive_exclusion', 'small_loss']:
            all_images, all_partial_labels, all_excluded_images, all_excluded_partial_labels = excludeNoisyLabels.getNewDataset()

            train_labeled_dataset = PartialSelectDataset(cifar100_mean,
                                                         cifar100_std,
                                                         all_images,
                                                         all_partial_labels,
                                                         transform=transform_labeled if args.aug else transform_general
                                                         )

            train_unlabeled_dataset = PartialSelectDataset(cifar100_mean,
                                                           cifar100_std,
                                                           all_excluded_images,
                                                           all_excluded_partial_labels,
                                                           transform=TransformFixMatch(mean=cifar100_mean,
                                                                                       std=cifar100_std) if args.SSLAug else transform_general
                                                           )
        else:
            all_images, all_partial_labels = excludeNoisyLabels.getPartialTrainDataset()
            train_labeled_dataset = PartialSelectDataset(cifar100_mean,
                                                         cifar100_std,
                                                         all_images,
                                                         all_partial_labels,
                                                         transform=transform_general
                                                         )
            train_unlabeled_dataset = None
        test_dataset, valid_dataset = excludeNoisyLabels.getTestAndValidDataset()

    else:
        base_dataset = datasets.CIFAR100(
            root, train=True, download=True)

        train_labeled_idxs, train_unlabeled_idxs = x_u_split(
            args, base_dataset.targets)

        train_labeled_dataset = CIFAR100SSL(
            root, train_labeled_idxs, train=True,
            transform=transform_labeled)

        train_unlabeled_dataset = CIFAR100SSL(
            root, train_unlabeled_idxs, train=True,
            transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

        test_dataset = datasets.CIFAR100(
            root, train=False, transform=transform_val, download=True)

        valid_dataset = None

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, valid_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for i in range(tensor.shape[1]):
            tensor[:, i, :, :] = tensor[:, i, :, :] * self.std[i] + self.mean[i]
        return tensor


class PartialSelectDataset(TensorDataset):
    def __init__(self, mean, std, *tensors: Tensor, transform=None):
        super().__init__(*tensors)

        self.transform = transform
        self.toPILImage = ToPILImage()
        self.imgs, self.targets = self.tensors[0], self.tensors[1]
        self.imgs = UnNormalize(mean, std)(self.imgs)
        labels_sum = self.targets.sum(dim=1)
        for i in range(self.targets.shape[0]):
            self.targets[i] = self.targets[i] / labels_sum[i]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, item):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.toPILImage(self.imgs[item])
        target = self.targets[item]
        if self.transform is not None:
            img = self.transform(img)
        # logger.info("Getting item {}/{} ...".format(item, self.__len__()))
        # logger.info(id(self))
        return img, target, item


class TransformFixMatch:
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.num_classes = 10
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            # convert to onehot target
            self.targets = np.eye(self.num_classes)[self.targets]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_uci(args, root):
    excludeNoisyLabels = ExcludeNoisyLabels(eid=args.eid,
                                            select_learning_rate=args.select_learning_rate,
                                            select_weight_decay=args.select_weight_decay,
                                            select_batch_size=args.select_batch_size,
                                            select_epoch=args.select_epoch,
                                            dataset=args.dataset,
                                            select_model=args.select_model,
                                            select_scheduler=args.select_scheduler,
                                            select_loss=args.select_loss,
                                            partial_type=args.partial_type,
                                            partial_rate=args.partial_rate,
                                            noisy_rate=args.noisy_rate,
                                            exclude_rate=args.exclude_rate,
                                            device=args.device,
                                            select_ratio=args.select_ratio,
                                            exclude=args.exclusion,
                                            exclusion_mode=args.exclusion_mode,
                                            fix_exclusion_rate=args.fix_exclusion_rate,
                                            reinit=args.reinit,
                                            args=args,
                                            patience=args.exclusion_patience
                                            )
    args.num_classes = excludeNoisyLabels.getNumClasses()
    args.feature_dim = excludeNoisyLabels.getFeatureDim()
    if args.exclusion in ['progressive_exclusion', 'small_loss']:
        all_instances, all_partial_labels, all_excluded_instances, all_excluded_partial_labels = excludeNoisyLabels.getNewDataset()

        train_labeled_dataset = DatasetWithIndex(all_instances,
                                                 all_partial_labels)

        train_unlabeled_dataset = DatasetWithIndex(all_excluded_instances,
                                                   all_excluded_partial_labels)
    else:
        all_instances, all_partial_labels = excludeNoisyLabels.getPartialTrainDataset()
        train_labeled_dataset = DatasetWithIndex(all_instances,
                                                 all_partial_labels)
        train_unlabeled_dataset = None
    test_dataset, valid_dataset = excludeNoisyLabels.getTestAndValidDataset()

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, valid_dataset


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'uci': get_uci}
