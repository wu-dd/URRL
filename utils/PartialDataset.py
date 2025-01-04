from torch.utils.data import Dataset


class GeneralDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label


class PartialDataset(Dataset):
    def __init__(self, images, partial_labels, label_flags, true_labels):
        self.images = images
        self.partial_labels = partial_labels
        self.label_flags = label_flags
        self.true_labels = true_labels

    def __len__(self):
        return len(self.partial_labels)

    def __getitem__(self, index):
        image = self.images[index]
        partial_label = self.partial_labels[index]
        label_flag = self.label_flags[index]
        true_label = self.true_labels[index]
        return image, partial_label, label_flag, true_label
