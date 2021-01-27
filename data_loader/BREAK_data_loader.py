from data_loader import custom_datasets
from .base_data_loader import BaseDataLoader


class BREAKDataLoader(BaseDataLoader):
    """
    Class for loading the Break dataset (logical-forms).
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 debug=False):
        """
        Initiates the Break dataset loader and the superclass.
        :param data_dir (str): Path to the data in which to save/ from which to read the dataset.
        :param batch_size (int): how many samples per batch to load.
        :param shuffle (bool): set to ``True`` to have the data reshuffled at every epoch.
        :param validation_split (int or float): (NOT USED) If integer, treats as amount of validation examples.
                                                If float, treats as percent of validation examples.
        :param num_workers (int): how many subprocesses to use for data loading.
        :param training: True for loading the training split, False for loading the testing split.
        :param debug: True for running with a small subset of the dataset.
        """
        self.data_dir = data_dir
        self.dataset = custom_datasets.BREAKLogical(self.data_dir, train=training, valid=False, debug=debug)
        self.dataset_type = self.dataset.get_dataset_type()
        self.validation_split = 0.0  # make sure to use all data
        self.drop_last = True
        self.debug = debug
        super().__init__(self.dataset, batch_size, shuffle, self.validation_split, num_workers,
                         drop_last=self.drop_last)

    def split_validation(self):
        """
        Defines a dataloader for the validation split.
        :return: The generated dataloader.
        """
        return BREAKDataLoaderValid(self.data_dir, self.batch_size, self.shuffle, self.validation_split,
                                    self.num_workers, debug=self.debug)

    def get_dataset_type(self):
        """
        :return: The dataset type.
        """
        return self.dataset_type


class BREAKDataLoaderValid(BaseDataLoader):
    """
    Class for loading the validation data for BREAK.
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 debug=False):
        self.data_dir = data_dir
        self.dataset = custom_datasets.BREAKLogical(self.data_dir, train=training, valid=True, debug=debug)
        self.dataset_type = self.dataset.get_dataset_type()
        self.drop_last = True
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, drop_last=self.drop_last)
        # print('valid len', len(self.dataset))

    def get_dataset_type(self):
        return self.dataset_type