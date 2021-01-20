import torch.utils.data as data
import numpy as np
import spacy
import logging
import ast
import re

from logger import setup_logging, LOGGER_SETUP
from nlp import load_dataset
from pathlib import Path
from utils import save_obj, load_obj
from subprocess import run


class BREAKLogical(data.Dataset):
    """
    The Break dataset: https://github.com/allenai/Break.
    """

    def __init__(self, data_dir, train=True, valid=False):
        """
        Initiates the Bread dataset.
        :param data_dir (str):  Path to the data in which to save/ from which to read the dataset.
        :param train:           True to load the train data, False to load the test data.
        :param valid:           If train is False, ignored. If train is True, then valid=True will load the validation
                                data and valid=False will load the train data.
        """
        # Define logger
        if not LOGGER_SETUP:
            setup_logging()
        self.logger = logging.getLogger('BREAKLogical')
        self.logger.setLevel(logging.INFO)
        self.logger.info("Preparing dataset")
        super(BREAKLogical, self).__init__()
        # Load dataset and lexicon
        self.dataset_type = 'test'
        if train:
            self.dataset_type = 'train'
            if valid:
                self.dataset_type = 'validation'

        self.dataset_logical = self.load_dataset(data_dir, 'logical-forms', self.logger)
        self.lexicon_dict = self.get_lexicon()[self.dataset_type]
        self.logger.info('dataset and lexicon ready.')

        # Download spacy language model
        if not spacy.util.is_package("en_core_web_sm"):
            self.logger.info('Downloading spacy english core...')
            run(['python', '-m', 'spacy', 'download', 'en'])

        # Prepare the questions and golds lists.
        # TODO remove the list slice, it is for debugging.
        self.questions = self.dataset_logical[self.dataset_type]['question_text']#[:200]
        self.golds = self.dataset_logical[self.dataset_type]['decomposition']#[:200]
        # Replace all the reference tokens of the form #<num> with the tokens @@<num>@@
        self.golds = [re.sub(r'#(\d+)', r'@@\1@@', qdmr) for qdmr in self.golds]

    @staticmethod
    def load_dataset(data_dir, dataset_type, logger=None):
        """
        loads the requested Break dataset from Hugging Face.
        :param data_dir: The path of the directory where the preprocessed dataset should be saved to or loaded from.
        :param dataset_type: The type of dataset to download from HF.
        :param logger: A logger for logging events.
        :return: The loaded dataset.
        """
        current_dir = Path()
        dir_path = current_dir / "data" / "break_data" / "preprocessed"
        file_name = "dataset_preprocessed_" + dataset_type + ".pkl"
        if not (dir_path / file_name).is_file():
            # Download and preprocess the BREAK dataset (logical form and lexicon), and save the preprocessed data.
            if logger:
                logger.info('Downloading and preparing datasets...')
            dataset_logical = load_dataset('break_data', dataset_type, cache_dir=data_dir)
            save_obj(dir_path, dataset_logical, file_name)

        # Load the saved preprocessed data.
        dataset = load_obj(dir_path, file_name)
        return dataset

    def __getitem__(self, idx):
        """
        Retrieves an example from the dataset.
        :param idx: The index of the example to retrieve.
        :return: The retrieved example.
        """
        example = (self.questions[idx], self.golds[idx])
        return example[0], self.clean_qdmr(example[1])

    def clean_qdmr(self, qdmr):
        """
        Removes the 'return' statement from the beginning of each qdmr entry.
        :param qdmr: The qdmr to clean.
        :return: The cleaned qdmr.
        """
        return qdmr.replace('return ', '')

    def __len__(self):
        """
        :return: The amount of examples in the dataset.
        """
        return len(self.questions)

    def get_example(self, idx=-1):
        """
        Retrieves an example from the dataset.
        :param idx: the index of the example to retrieve, if negative returns a random example.
        :return: the retrieved example.
        """
        if idx < 0:
            idx = np.random.randint(len(self))
        return self[idx]

    def create_matching_lexicon(self, dir_path, file_name):
        # TODO add documentation
        self.logger.info('Creating lexicon...')
        dataset_qdmr_lexicon = self.load_dataset(dir_path, 'QDMR-lexicon', self.logger)

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
        # TODO add documentation
        self.logger.info("Preparing lexicon")
        current_dir = Path()
        dir_path = current_dir / "data" / "break_data" / "lexicon_by_logical"
        file_name = "lexicon.pkl"
        if not (dir_path / file_name).is_file():
            self.create_matching_lexicon(dir_path, file_name)
        self.logger.info("loading lexicon")
        data = load_obj(dir_path, file_name)
        self.logger.info("lexicon loaded")
        # TODO what is this?? it's superrrrr slow optimize or save to the file on lex creation.
        # TODO uncomment this
        # for type in data:
        #     for ex in data[type]:
        #         data[type][ex] = ast.literal_eval(data[type][ex])
        self.logger.info("done literal eval")
        return data

    @staticmethod
    def visualize(question, gold):
        """
        Visualized a question and it's qdmr.
        :param question: The question to visualize.
        :param gold: The corresponding qdmr.
        """
        print("Question:\n", question)
        print("QDMR:\n", gold.replace(";", ";\n"))
