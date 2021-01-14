import custom_datasets

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
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class BREAKDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = custom_datasets.BREAKLogical(self.data_dir, train=training, valid=False)
        self.validation_split = 0.0  # make sure to use all data
        super().__init__(self.dataset, batch_size, shuffle, self.validation_split, num_workers)

    def split_validation(self):
        return BREAKDataLoaderValid(self.data_dir, self.batch_size, self.shuffle, self.validation_split,
                                    self.num_workers)


class BREAKDataLoaderValid(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = custom_datasets.BREAKLogical(self.data_dir, train=training, valid=True)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

#dads
#dsadasd
#sd