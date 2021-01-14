import numpy as np
import logging

from logger import setup_logging, LOGGER_SETUP
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from abc import abstractmethod


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        """
        Initiates the dataloader with the given parameters and initiates the super class.
        :param dataset (Dataset): dataset from which to load the data.
        :param batch_size (int): how many samples per batch to load.
        :param shuffle (bool): set to ``True`` to have the data reshuffled at every epoch.
        :param validation_split (int or float): If integer, treats as amount of validation examples.
                                                If float, treats as percent of validation examples.
        :param num_workers (int): how many subprocesses to use for data loading.
        :param collate_fn (Callable): merges a list of samples to form a mini-batch of Tensor(s).
        """
        # Define logger
        if not LOGGER_SETUP:
            setup_logging()
        self.logger = logging.getLogger('BaseDataLoader')
        self.logger.setLevel(logging.INFO)

        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        # Define samplers for the training split and the validation split.
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        # Define arguments to initiate the superclass.
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        """
        Splits the dataset into train and dev sets.
        :param split (int or float):    If integer, treats as amount of validation examples.
                                        If float, treats as percent of validation examples.
        :return: the samplers for the train and validation splits.
        """
        # If the split value is 0.0 use the whole dataset as training split.
        if split == 0.0:
            return None, None

        # Create full index space for the dataset.
        idx_full = np.arange(self.n_samples)

        # Shuffle the indexes
        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):  # If split is int, define the amount of validation examples to be split.
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        elif isinstance(split, float):  # if float, define the amount of validation examples to be split percent of
            # the whole dataset.
            assert split <= 1, "split should be lesser then 1 if float."
            len_valid = int(self.n_samples * split)
        else:  # split is invalid, use all the dataset for training.
            self.logger.warning("Split should be integer or float, ignoring split.")
            return None, None

        # Split the shuffled indices to validation and training indices.
        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        # Define samplers for train and validation.
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    @abstractmethod
    def split_validation(self):
        """
        Generate the validation Dataloader.
        :return: The generated dataloader.
        Note: this function is abstract so that child dataloaders could define their own validation dataloader.
        """
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
