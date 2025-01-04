import logging
import torch
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from ExcludeNoisyLabels import ExcludeNoisyLabels
from dataset.cifar import UnNormalize
from external_models.PiCO.randaugment import RandomAugment

logger = logging.getLogger(__name__)
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args):
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
                                            device=args.device,
                                            select_ratio=args.select_ratio,
                                            exclude=args.exclusion,
                                            exclusion_mode=args.exclusion_mode,
                                            fix_exclusion_rate=args.fix_exclusion_rate,
                                            patience=args.exclusion_patience,
                                            args=args
                                            )

    if args.exclusion in ['progressive_exclusion', 'small_loss']:
        all_images, all_partial_labels, all_excluded_images, all_excluded_partial_labels = excludeNoisyLabels.getNewDataset()

        train_labeled_dataset = PiCOPartialSelectDataset(cifar10_mean,
                                                         cifar10_std,
                                                         all_images,
                                                         all_partial_labels,
                                                         args=args)

        train_unlabeled_dataset = PiCOPartialSelectDataset(cifar10_mean,
                                                           cifar10_std,
                                                           all_excluded_images,
                                                           all_excluded_partial_labels,
                                                           args=args)
    else:
        all_images, all_partial_labels = excludeNoisyLabels.getPartialTrainDataset()
        train_labeled_dataset = PiCOPartialSelectDataset(cifar10_mean,
                                                         cifar10_std,
                                                         all_images,
                                                         all_partial_labels,
                                                         args=args)
        train_unlabeled_dataset = None
    test_dataset, valid_dataset = excludeNoisyLabels.getTestAndValidDataset()

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, valid_dataset


def get_cifar100(args):
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
                                            device=args.device,
                                            select_ratio=args.select_ratio,
                                            exclude=args.exclusion,
                                            exclusion_mode=args.exclusion_mode,
                                            fix_exclusion_rate=args.fix_exclusion_rate,
                                            args=args
                                            )

    if args.exclusion in ['progressive_exclusion', 'small_loss']:
        all_images, all_partial_labels, all_excluded_images, all_excluded_partial_labels = excludeNoisyLabels.getNewDataset()

        train_labeled_dataset = PiCOPartialSelectDataset(cifar100_mean,
                                                         cifar100_std,
                                                         all_images,
                                                         all_partial_labels,
                                                         args=args)

        train_unlabeled_dataset = PiCOPartialSelectDataset(cifar100_mean,
                                                           cifar100_std,
                                                           all_excluded_images,
                                                           all_excluded_partial_labels,
                                                           args=args)
    else:
        all_images, all_partial_labels = excludeNoisyLabels.getPartialTrainDataset()
        train_labeled_dataset = PiCOPartialSelectDataset(cifar100_mean,
                                                         cifar100_std,
                                                         all_images,
                                                         all_partial_labels,
                                                         args=args)
        train_unlabeled_dataset = None
    test_dataset, valid_dataset = excludeNoisyLabels.getTestAndValidDataset()

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, valid_dataset


class PiCOPartialSelectDataset(TensorDataset):
    def __init__(self, mean, std, *tensors: Tensor, args):
        super().__init__(*tensors)

        if args.dataset == 'cifar10':
            self.weak_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            self.strong_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    RandomAugment(3, 5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        elif args.dataset == 'cifar100':
            self.weak_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
            self.strong_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    RandomAugment(3, 5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.toPILImage = ToPILImage()
        self.imgs, self.partial_labels = self.tensors[0], self.tensors[1]
        self.imgs = UnNormalize(mean, std)(self.imgs)
        # TODO Judge if need sum to one.
        # labels_sum = self.partial_labels.sum(dim=1)
        # for i in range(self.partial_labels.shape[0]):
        #     self.partial_labels[i] = self.partial_labels[i] / labels_sum[i]

    def __getitem__(self, index):
        image_w = self.weak_transform(self.imgs[index])
        image_s = self.strong_transform(self.imgs[index])
        partial_label = self.partial_labels[index]
        return image_w, image_s, partial_label, index


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}
