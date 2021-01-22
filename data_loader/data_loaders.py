from data_loader import custom_datasets

from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.training = training
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class BREAKDataLoader(BaseDataLoader):
    """
    Class for loading the Break dataset (logical-forms).
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        """
        Initiates the Break dataset loader and the superclass.
        :param data_dir (str): Path to the data in which to save/ from which to read the dataset.
        :param batch_size (int): how many samples per batch to load.
        :param shuffle (bool): set to ``True`` to have the data reshuffled at every epoch.
        :param validation_split (int or float): (NOT USED) If integer, treats as amount of validation examples.
                                                If float, treats as percent of validation examples.
        :param num_workers (int): how many subprocesses to use for data loading.
        :param training: True for loading the training split, False for loading the testing split.
        """
        self.data_dir = data_dir
        self.dataset = custom_datasets.BREAKLogical(self.data_dir, train=training, valid=False)
        self.dataset_type = self.dataset.get_dataset_type()
        self.validation_split = 0.0  # make sure to use all data
        self.drop_last = True
        super().__init__(self.dataset, batch_size, shuffle, self.validation_split, num_workers,
                         drop_last=self.drop_last)

    def split_validation(self):
        """
        Defines a dataloader for the validation split.
        :return: The generated dataloader.
        """
        return BREAKDataLoaderValid(self.data_dir, self.batch_size, self.shuffle, self.validation_split,
                                    self.num_workers)

    def get_dataset_type(self):
        return self.dataset_type


class BREAKDataLoaderValid(BaseDataLoader):
    """
    Class for loading the validation data for BREAK.
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = custom_datasets.BREAKLogical(self.data_dir, train=training, valid=True)
        self.dataset_type = self.dataset.get_dataset_type()
        self.drop_last = True
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, drop_last=self.drop_last)

    def get_dataset_type(self):
        return self.dataset_type
