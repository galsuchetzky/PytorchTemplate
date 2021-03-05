import torch.utils.data as data
import numpy as np
import spacy
import logging
import pandas as pd

from logger.logger import setup_logging, LOGGER_SETUP
from nlp import load_dataset
from datasets import Dataset
from pathlib import Path
from utils.util import save_obj, load_obj, read_json, write_json, minimize_program
from subprocess import run
from tester.BREAK_evaluate_predictions import format_qdmr
from utils.qdmr_identifier import *

DEBUG_EXAMPLES_AMOUNT = 150


class BREAKLogical(data.Dataset):
    """
    The Break dataset: https://github.com/allenai/Break.
    """

    def __init__(self, data_dir, gold_type, domain_split, length_split, train=True, valid=False, debug=False):
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
        self.domain_split = domain_split
        self.length_split = length_split
        # Load dataset and lexicon
        self.dataset_split = 'test'
        if train:
            self.dataset_split = 'train'
            if valid:
                self.dataset_split = 'validation'

        self.logger.info('loading data split:' + self.dataset_split)

        self.logger.info('loading vanilla dataset')
        self.dataset_logical = self.load_dataset(data_dir, 'logical-forms', self.logger)
        if self.domain_split:
            self.logger.info('loading domain split dataset')
            self.dataset_logical = self.load_domain_split_dataset(data_dir, self.logger)
        elif self.length_split:
            self.logger.info('loading length split dataset')
            self.dataset_logical = self.load_length_split_dataset(data_dir, self.logger)
        self.logger.info('dataset ready.')

        # Download spacy language model
        if not spacy.util.is_package("en_core_web_sm"):
            self.logger.info('Downloading spacy english core...')
            run(['python', '-m', 'spacy', 'download', 'en'])

        # Prepare the data parts
        self.ids = self.dataset_logical[self.dataset_split]['question_id']
        self.questions = self.dataset_logical[self.dataset_split]['question_text']
        # lexicon is based on vanilla/ domain_split type of dataset_logical
        self.lexicon_str = self.get_lexicon()[self.dataset_split]
        self.logger.info('dataset and lexicon ready.')

        # uses QDMR
        self.qdmrs = [format_qdmr(decomp) for decomp in self.dataset_logical[self.dataset_split]["decomposition"]]
        self.programs = self.get_programs()

        if debug:
            self.ids = self.ids[:DEBUG_EXAMPLES_AMOUNT]
            self.questions = self.questions[:DEBUG_EXAMPLES_AMOUNT]
            self.qdmrs = self.qdmrs[:DEBUG_EXAMPLES_AMOUNT]
            self.lexicon_str = self.lexicon_str[:DEBUG_EXAMPLES_AMOUNT]
            self.programs = self.programs[:DEBUG_EXAMPLES_AMOUNT]

        # # Replace all the reference tokens of the form #<num> with the tokens @@<num>@@
        # self.qdmrs = [re.sub(r'#(\d+)', r'@@\1@@', qdmr) for qdmr in self.qdmrs]

    def get_dataset_split(self):
        return self.dataset_split

    def load_domain_split_dataset(self, data_dir, logger=None):
        """
        Loads break dataset with domain split. Train - on text. val + test - on DB + images
        :param data_dir: The path of the directory where the preprocessed dataset should be saved to or loaded from.
        :param logger: A logger for logging events.
        :return: The loaded dataset.
        """
        current_dir = Path()
        dir_path = current_dir / "data" / "break_data" / "preprocessed"
        file_name = "dataset_preprocessed_domain_split.pkl"
        if not (dir_path / file_name).is_file():
            if logger:
                logger.info('Creating domain split dataset...')
            text_domain_dataset_prefixes = ('COMQA', 'CWQ', 'DROP', 'HOTP')
            image_domain_dataset_prefixes = ('CLEVR', 'NLVR2')
            DB_domain_dataset_prefixes = ('ACADEMIC', 'ATIS', 'GEO', 'SPIDER')
            image_plus_DB = image_domain_dataset_prefixes + DB_domain_dataset_prefixes
            train_filtererd = pd.DataFrame()
            validation_filtererd = pd.DataFrame()
            test_filtererd = pd.DataFrame()

            for i, example in enumerate(self.dataset_logical['train']):
                if example['question_id'].startswith(text_domain_dataset_prefixes):
                    train_filtererd = train_filtererd.append(example, ignore_index=True)
            for i, example in enumerate(self.dataset_logical['validation']):
                if example['question_id'].startswith(image_plus_DB):
                    validation_filtererd = validation_filtererd.append(example, ignore_index=True)
            for i, example in enumerate(self.dataset_logical['test']):
                if example['question_id'].startswith(image_plus_DB):
                    test_filtererd = test_filtererd.append(example, ignore_index=True)

            # train_dataset = self.dataset_logical['train'].filter(
            #     lambda example: example['question_id'].startswith(text_domain_dataset_prefixes))
            # validation_dataset = self.dataset_logical['validation'].filter(
            #     lambda example: example['question_id'].startswith(image_plus_DB))
            # test_dataset = self.dataset_logical['test'].filter(
            #     lambda example: example['question_id'].startswith(image_plus_DB))
            # train_filtererd_ds = Dataset.from_pandas(train_filtererd)
            to_save = {'train': Dataset.from_pandas(train_filtererd),
                       'validation': Dataset.from_pandas(validation_filtererd),
                       'test': Dataset.from_pandas(test_filtererd)}
            save_obj(dir_path, to_save, file_name)

        dataset = load_obj(dir_path, file_name)
        return dataset

    def load_length_split_dataset(self, data_dir, logger=None):
        """
        Loads break dataset with length split based on number of operators.
        Train - <= 4 steps.
        val + test - on DB + images
        :param data_dir: The path of the directory where the preprocessed dataset should be saved to or loaded from.
        :param logger: A logger for logging events.
        :return: The loaded dataset.
        """
        current_dir = Path()
        dir_path = current_dir / "data" / "break_data" / "preprocessed"
        file_name = "dataset_preprocessed_length_split.pkl"
        if not (dir_path / file_name).is_file():
            if logger:
                logger.info('Creating length split dataset...')
            threshold_amount_ops = 4

            train_filtererd = pd.DataFrame()
            validation_filtererd = pd.DataFrame()
            test_filtererd = pd.DataFrame()

            for i, example in enumerate(self.dataset_logical['train']):
                if example['operators'].count(',') < threshold_amount_ops:
                    train_filtererd = train_filtererd.append(example, ignore_index=True)
            for i, example in enumerate(self.dataset_logical['validation']):
                if example['operators'].count(',') >= threshold_amount_ops:
                    validation_filtererd = validation_filtererd.append(example, ignore_index=True)
            for i, example in enumerate(self.dataset_logical['test']):
                if example['operators'].count(',') >= threshold_amount_ops:
                    test_filtererd = test_filtererd.append(example, ignore_index=True)

            to_save = {'train': Dataset.from_pandas(train_filtererd),
                       'validation': Dataset.from_pandas(validation_filtererd),
                       'test': Dataset.from_pandas(test_filtererd)}
            save_obj(dir_path, to_save, file_name)

        dataset = load_obj(dir_path, file_name)
        return dataset

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
        if self.gold_type == 'qdmr':
            golds = self.qdmrs[idx].to_string()
        elif self.gold_type == 'program':
            golds = self.programs[idx]
        elif self.gold_type == 'minimized_program':
            golds = minimize_program(self.programs[idx])
        example = (self.ids[idx], self.questions[idx], golds, self.lexicon_str[idx])

        # without lexicon
        # example = (self.ids[idx], self.questions[idx], golds)
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
        # There are more examples in the lexicon dataset than in the logical examples
        # This function creates one-to-one mapping between them and stores lexicon dict in a file
        self.logger.info('Creating lexicon...')
        dataset_qdmr_lexicon = self.load_dataset(dir_path, 'QDMR-lexicon', self.logger)

        lexicon_lists = {'train': [], 'validation': [], 'test': []}
        for data_split in self.dataset_logical:
            lex_idx = 0
            lexicon_split = dataset_qdmr_lexicon[data_split]
            for i, example in enumerate(self.dataset_logical[data_split]):
                question = example['question_text']
                lexcion_found = False
                for j in range(lex_idx, len(lexicon_split)):
                    lexicon_example = lexicon_split[j]
                    if lexicon_example['source'] == question:
                        str_lex = lexicon_example['allowed_tokens']
                        lexicon_lists[data_split].append(str_lex)
                        # TODO remove, its testing code
                        # lexicon_dict[data_split][i]
                        # if str_lex[2] != 'h':
                        #     print("Holy")
                        lex_idx = j + 1
                        lexcion_found = True
                        break
                # if it got here, no matching lexicon found in lexicon file
                if not lexcion_found:
                    raise EOFError
        save_obj(dir_path, lexicon_lists, file_name)
        self.logger.info('Done creating lexicon.')

    def get_lexicon(self):
        # TODO add documentation
        self.logger.info("Preparing lexicon...")
        current_dir = Path()
        dir_path = current_dir / "data" / "break_data" / "lexicon_by_logical"
        file_name = "lexicon"
        if self.domain_split:
            file_name += "_domain_split"
        elif self.length_split:
            file_name += "_length_split"
        file_name += ".pkl"
        if not (dir_path / file_name).is_file():
            self.create_matching_lexicon(dir_path, file_name)
        data = load_obj(dir_path, file_name)

        # for type in data:
        #     for ex in data[type]:
        #         data[type][ex] = ast.literal_eval(data[type][ex])
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

        file_name = "programs_" + self.dataset_split + ".pkl"
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
