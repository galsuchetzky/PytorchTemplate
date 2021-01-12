from torchvision import datasets, transforms
from base import BaseDataLoader
from nlp import load_dataset
from subprocess import run


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
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.dataset_qdmr = load_dataset('break_data', 'QDMR', cache_dir='.\\data\\')
        self.dataset_qdmr_lexicon = load_dataset('break_data', 'QDMR-lexicon', cache_dir='.\\data\\')
        self.create_lexicon()


        # Download spacy language model
        # print('Downloading spacy language model...')
        run(['python', '-m', 'spacy', 'download', 'en'])

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def create_lexicon():
