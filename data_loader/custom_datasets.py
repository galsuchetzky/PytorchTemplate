import torch.utils.data as data
import numpy as np
import spacy
import logging
import ast

from logger.logger import setup_logging, LOGGER_SETUP
from nlp import load_dataset
from pathlib import Path
from utils.util import save_obj, load_obj, read_json, write_json
from subprocess import run
from tester.BREAK_evaluate_predictions import format_qdmr
from utils.qdmr_identifier import *

DEBUG_EXAMPLES_AMOUNT = 3000


class BREAKLogical(data.Dataset):
    """
    The Break dataset: https://github.com/allenai/Break.
    """

    def __init__(self, data_dir, gold_type, train=True, valid=False, debug=False):
        """
        Initiates the Bread dataset.
        :param data_dir (str):  Path to the data in which to save/ from which to read the dataset.
        :param train:           True to load the train data, False to load the test data.
        :param valid:           If train is False, ignored. If train is True, then valid=True will load the validation
                                data and valid=False will load the train data.
        :param debug:           True for using a small subset of the dataset.
        """
        # Define logger
        if not LOGGER_SETUP:
            setup_logging()
        self.logger = logging.getLogger('BREAKLogical')
        self.logger.setLevel(logging.INFO)
        self.logger.info("Preparing dataset")
        super(BREAKLogical, self).__init__()

        self.gold_type = gold_type

        # Load dataset and lexicon
        self.dataset_split = 'test'
        if train:
            self.dataset_split = 'train'
            if valid:
                self.dataset_split = 'validation'

        self.logger.info('loading data split:' + self.dataset_split)

        self.dataset_logical = self.load_dataset(data_dir, 'logical-forms', self.logger)

        self.lexicon_dict = self.get_lexicon()[self.dataset_split]
        self.logger.info('dataset and lexicon ready.')

        # Download spacy language model
        if not spacy.util.is_package("en_core_web_sm"):
            self.logger.info('Downloading spacy english core...')
            run(['python', '-m', 'spacy', 'download', 'en'])

        # Prepare the data parts
        self.ids = self.dataset_logical[self.dataset_split]['question_id']
        self.questions = self.dataset_logical[self.dataset_split]['question_text']
        self.lexicon_dict = self.get_lexicon()[self.dataset_split]
        self.logger.info('dataset and lexicon ready.')

        # uses QDMR
        self.qdmrs = [format_qdmr(decomp) for decomp in self.dataset_logical[self.dataset_split]["decomposition"]]
        # TODO empty string for test
        self.programs = self.get_programs()
        if debug:
            self.ids = self.ids[:DEBUG_EXAMPLES_AMOUNT]
            self.questions = self.questions[:DEBUG_EXAMPLES_AMOUNT]
            self.qdmrs = self.qdmrs[:DEBUG_EXAMPLES_AMOUNT]
            self.lexicon_dict = self.lexicon_dict[:DEBUG_EXAMPLES_AMOUNT]
            self.programs = self.programs[:DEBUG_EXAMPLES_AMOUNT]

        # # Replace all the reference tokens of the form #<num> with the tokens @@<num>@@
        # self.qdmrs = [re.sub(r'#(\d+)', r'@@\1@@', qdmr) for qdmr in self.qdmrs]

    def get_dataset_split(self):
        return self.dataset_split

    @staticmethod
    def load_dataset(data_dir, dataset_split, logger=None):
        """
        loads the requested Break dataset from Hugging Face.
        :param data_dir: The path of the directory where the preprocessed dataset should be saved to or loaded from.
        :param dataset_split: The type of dataset to download from HF.
        :param logger: A logger for logging events.
        :return: The loaded dataset.
        """
        current_dir = Path()
        dir_path = current_dir / "data" / "break_data" / "preprocessed"
        file_name = "dataset_preprocessed_" + dataset_split + ".pkl"
        if not (dir_path / file_name).is_file():
            # Download and preprocess the BREAK dataset (logical form and lexicon), and save the preprocessed data.
            if logger:
                logger.info('Downloading and preparing datasets...')
            dataset_logical = load_dataset('break_data', dataset_split, cache_dir=data_dir)
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
        # example = (self.ids[idx], self.questions[idx], self.qdmrs[idx].to_string())
        golds = self.qdmrs[idx].to_string() if self.gold_type == 'qdmr' else self.programs[idx]
        example = (self.ids[idx], self.questions[idx], golds, ';'.join(self.lexicon_dict[idx]))
        #todo there is false for some reason in the lexicon
        return example

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
        self.logger.info("Preparing lexicon...")
        current_dir = Path()
        dir_path = current_dir / "data" / "break_data" / "lexicon_by_logical"
        file_name = "lexicon.pkl"
        if not (dir_path / file_name).is_file():
            self.create_matching_lexicon(dir_path, file_name)
        data = load_obj(dir_path, file_name)
        # TODO these lines turn the string repr of lists to real lists, but slow. save to file or something.
        # TODO uncomment this
        for type in data:
            for ex in data[type]:
                data[type][ex] = ast.literal_eval(data[type][ex])
        self.logger.info("Lexicon ready.")
        return data

    def create_matching_programs(self, dir_path, file_name):
        # TODO add documentation
        self.logger.info('Creating programs...')
        programs = []
        for gold in self.dataset_logical[self.dataset_split]["decomposition"]:
            builder = QDMRProgramBuilder(gold)
            builder.build()
            programs.append(str(builder))
        save_obj(dir_path, programs, file_name)
        self.logger.info('Done creating programs.')

    def get_programs(self):
        self.logger.info("Preparing programs...")
        current_dir = Path()
        dir_path = current_dir / "data" / "break_data" / "programs"

        file_name = "programs_" + self.dataset_split +".pkl"
        if not (dir_path / file_name).is_file():
            self.create_matching_programs(dir_path, file_name)
        data = load_obj(dir_path, file_name)

        self.logger.info("Programs ready.")
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
