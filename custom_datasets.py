import torch.utils.data as data
import numpy as np
import spacy
import logging
from logger import setup_logging, LOGGER_SETUP

from nlp import load_dataset
from pathlib import Path
from utils import save_obj, load_obj
from subprocess import run


class BREAKLogical(data.Dataset):
    """
    The Break dataset: https://github.com/allenai/Break
    """

    def __init__(self, data_dir, train=True, valid=False):
        # Define logger
        if not LOGGER_SETUP:
            setup_logging()
        self.logger = logging.getLogger('BREAKLogical')
        self.logger.setLevel(logging.INFO)

        super(BREAKLogical, self).__init__()
        # Load dataset and lexicon
        dataset_type = 'test'
        if train:
            dataset_type = 'train'
            if valid:
                dataset_type = 'validation'

        self.logger.info('Downloading and preparing datasets...')
        self.dataset_logical = load_dataset('break_data', 'logical-forms', cache_dir=data_dir)
        self.lexicon_dict = self.get_lexicon()[dataset_type]
        self.logger.info('datasets ready.')

        # Download spacy language model
        if not spacy.util.is_package("en_core_web_sm"):
            self.logger.info('Downloading spacy english core...')
            run(['python', '-m', 'spacy', 'download', 'en'])

        self.questions = self.dataset_logical[dataset_type]['question_text']
        self.golds = self.dataset_logical[dataset_type]['decomposition']

    def __getitem__(self, idx):
        example = (self.questions[idx], self.golds[idx], self.lexicon_dict[idx])
        return example

    def __len__(self):
        return len(self.questions)

    def get_random_example(self, idx=-1):
        if idx == -1:
            idx = np.random.randint(len(self))
        return self[idx]

    def create_matching_lexicon(self, dir_path, file_name):
        self.logger.info('Creating lexicon...')
        dataset_qdmr_lexicon = load_dataset('break_data', 'QDMR-lexicon', cache_dir='.\\data\\')

        lexicon_dict = {'train': dict(), 'validation': dict(), 'test': dict()}
        for data_split in self.dataset_logical:
            lex_idx = 0
            lexicon_split = dataset_qdmr_lexicon[data_split]
            for i, example in enumerate(self.dataset_logical[data_split]):
                question = example['question_text']
                for j in range(lex_idx, len(lexicon_split)):
                    lexicon_example = lexicon_split[j]
                    if lexicon_example['source'] == question:
                        lexicon_dict[data_split][i] = lexicon_example['allowed_tokens']
                        lex_idx = j + 1
                        break
        save_obj(dir_path, lexicon_dict, file_name)
        self.logger.info('Done creating lexicon.')

    def get_lexicon(self):
        current_dir = Path()
        dir_path = current_dir / "data" / "break_data" / "lexicon_by_logical"
        file_name = "lexicon.pkl"
        if not (dir_path / file_name).is_file():
            self.create_matching_lexicon(dir_path, file_name)
        data = load_obj(dir_path, file_name)
        return data
