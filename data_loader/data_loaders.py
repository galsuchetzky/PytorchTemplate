from torchvision import datasets, transforms
from base import BaseDataLoader
from nlp import load_dataset
from subprocess import run
from pathlib import Path
from utils import save_obj, load_obj

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
        # self.dataset_qdmr = load_dataset('break_data', 'QDMR', cache_dir='.\\data\\')
        self.dataset_logical = load_dataset('break_data', 'logical-forms', cache_dir='.\\data\\')
        self.lexicon_dict = self.get_lexicon()


        # Download spacy language model
        # print('Downloading spacy language model...')
        run(['python', '-m', 'spacy', 'download', 'en'])

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def create_matching_lexicon(self, dir_path, file_name):
        dataset_qdmr_lexicon = load_dataset('break_data', 'QDMR-lexicon', cache_dir='.\\data\\')

        lexicon_dict = {'train': dict(), 'validation': dict(), 'test': dict()}
        for data_split in self.dataset_logical:
            lex_idx = 0
            lexicon_split = dataset_qdmr_lexicon[data_split]
            for i, example in enumerate(self.dataset_logical[data_split]):
                ques = example['question_text']
                for j in range(lex_idx, len(lexicon_split)):
                    lexicon_example = lexicon_split[j]
                    if lexicon_example['source'] == ques:
                        lexicon_dict[data_split][i] = lexicon_example['allowed_tokens']
                        lex_idx = j + 1
                        break
        save_obj(dir_path, lexicon_dict, file_name)

    def get_lexicon(self):
        current_dir = Path()
        dir_path = current_dir / "data" / "break_data" / "lexicon_by_logical"
        file_name = "lexicon.pkl"
        if not (dir_path / file_name).is_file():
            self.create_matching_lexicon(dir_path, file_name)
        data = load_obj(dir_path, file_name)
        return data
