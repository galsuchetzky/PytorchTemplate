from torchvision import datasets, transforms
from .base_data_loader import BaseDataLoader

DEBUG_EXAMPLES_AMOUNT = 4000


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 debug=False):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.training = training
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        if debug:
            self.dataset.data = self.dataset.data[:DEBUG_EXAMPLES_AMOUNT]
            self.dataset.targets = self.dataset.targets[:DEBUG_EXAMPLES_AMOUNT]

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
