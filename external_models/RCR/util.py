import torch
from dataset.cifar import UnNormalize, PartialSelectDataset, TransformFixMatch
from external_models.RCR.augment.autoaugment_extra import CIFAR10Policy
from external_models.RCR.augment.cutout import Cutout
import logging
from torchvision.transforms import ToPILImage
from torch import Tensor
from torch.utils.data import TensorDataset
from torchvision import transforms
from ExcludeNoisyLabels import ExcludeNoisyLabels

logger = logging.getLogger(__name__)
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y, index,
                      num_classes):
    y_pred_aug0_probas = y_pred_aug0_probas.detach()
    y_pred_aug1_probas = y_pred_aug1_probas.detach()
    y_pred_aug2_probas = y_pred_aug2_probas.detach()

    revisedY0 = part_y.clone()

    revisedY0 = revisedY0 * torch.pow(y_pred_aug0_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug1_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug2_probas, 1 / (2 + 1))
    revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(num_classes, 1).transpose(0, 1)

    confidence[index, :] = revisedY0.cpu().numpy()


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
                                            args=args,
                                            exclude_rate=args.exclude_rate
                                            )

    if args.exclusion in ['progressive_exclusion', 'small_loss']:
        all_images, all_partial_labels, all_excluded_images, all_excluded_partial_labels = excludeNoisyLabels.getNewDataset()

        train_labeled_dataset = RCRPartialSelectDataset(cifar10_mean,
                                                        cifar10_std,
                                                        all_images,
                                                        all_partial_labels,
                                                        )

        train_unlabeled_dataset = PartialSelectDataset(cifar10_mean,
                                                       cifar10_std,
                                                       all_excluded_images,
                                                       all_excluded_partial_labels,
                                                       transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std)
                                                       )
    else:
        all_images, all_partial_labels = excludeNoisyLabels.getPartialTrainDataset()
        train_labeled_dataset = RCRPartialSelectDataset(cifar10_mean,
                                                        cifar10_std,
                                                        all_images,
                                                        all_partial_labels,
                                                        )
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
                                            patience=args.exclusion_patience,
                                            args=args,
                                            exclude_rate=args.exclude_rate
                                            )

    if args.exclusion in ['progressive_exclusion', 'small_loss']:
        all_images, all_partial_labels, all_excluded_images, all_excluded_partial_labels = excludeNoisyLabels.getNewDataset()

        train_labeled_dataset = RCRPartialSelectDataset(cifar100_mean,
                                                        cifar100_std,
                                                        all_images,
                                                        all_partial_labels,
                                                        )
        train_unlabeled_dataset = PartialSelectDataset(cifar100_mean,
                                                       cifar100_std,
                                                       all_excluded_images,
                                                       all_excluded_partial_labels,
                                                       transform=TransformFixMatch(mean=cifar100_mean,
                                                                                   std=cifar100_std)
                                                       )
    else:
        all_images, all_partial_labels = excludeNoisyLabels.getPartialTrainDataset()
        train_labeled_dataset = RCRPartialSelectDataset(cifar100_mean,
                                                        cifar100_std,
                                                        all_images,
                                                        all_partial_labels,
                                                        )
        train_unlabeled_dataset = None
    test_dataset, valid_dataset = excludeNoisyLabels.getTestAndValidDataset()

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, valid_dataset


class RCRPartialSelectDataset(TensorDataset):
    def __init__(self, mean, std, *tensors: Tensor):
        super().__init__(*tensors)

        self.transform_0 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.transform_1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.ToPILImage(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.toPILImage = ToPILImage()
        self.imgs, self.partial_labels = self.tensors[0], self.tensors[1]
        self.imgs = UnNormalize(mean, std)(self.imgs)
        # TODO Remove following sum to one.
        # labels_sum = self.partial_labels.sum(dim=1)
        # for i in range(self.partial_labels.shape[0]):
        #     self.partial_labels[i] = self.partial_labels[i] / labels_sum[i]

    def __getitem__(self, index):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.toPILImage(self.imgs[index])

        img_ori = self.transform_0(img)
        img1 = self.transform_1(img)
        img2 = self.transform_1(img)
        partial_label = self.partial_labels[index]
        return img_ori, img1, img2, partial_label, index


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}
