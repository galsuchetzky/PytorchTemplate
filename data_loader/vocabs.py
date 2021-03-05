import spacy
import torch
import ast

from pathlib import Path
from collections import Counter
from torchtext.vocab import Vocab
from utils.util import save_obj, load_obj
from data_loader.custom_datasets import BREAKLogical
from torchtext.data.utils import get_tokenizer
from utils.qdmr_identifier import *
from tester.BREAK_qdmr_to_program import prediction_to_qdmr

# English tokenizer for tokenizing natural language sentences when building a vocabulary.
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


def BREAK_vocab_qdmr():
    """
    This function builds a vocabulary for the simple seq2seq model of qdmr.
    :return: The vocab.
    """
    # Special characters to include in the vocabulary.
    # TODO move the special tokens (unk, pad ...) to constants out of here.
    specials = get_specials_qdmr()

    # Load the dataset and get the words.
    # TODO The dictionary as for now is only created from the training data. maybe change that.
    current_dir = Path()
    dir_path = current_dir / "data" / "break_data" / "vocabs"
    file_name = "vocab_counter.pkl"
    sents = BREAKLogical.load_dataset(dir_path, 'logical-forms')['train']['question_text']
    vocab = build_vocab(specials, dir_path, file_name, sents, en_tokenizer)
    return vocab


def get_specials_qdmr():
    """

    :return:
    """
    specials = ['<unk>', '<sos>', '<pad>', '<eos>', '@@SEP@@', '@@10@@', '@@11@@', '@@12@@', '@@13@@',
                '@@14@@', '@@15@@', '@@16@@', '@@17@@', '@@18@@', '@@19@@', '@@1@@',
                '@@2@@', '@@3@@', '@@4@@', '@@5@@', '@@6@@', '@@7@@', '@@8@@', '@@9@@']
    return specials


def get_specials_program():
    """

    :return:
    """  # operators = ["select", "filter", "project", "aggregate",
    #              "group", "superlative", "comparative",
    #              "union", "intersection", "discard",
    #              "sort", "boolean", "arithmetic",
    #              "comparison"]
    phrases_by_operators = {
        "select": [],
        "filter": [],
        "project": [],
        "aggregate": ["max", "min", "count", "sum", "avg"],
        "group": ["max", "min", "count", "sum", "avg"],
        "superlative": ["argmax", "argmin"],
        "comparative": ["smaller", "smaller or equal to", "bigger", "bigger or equal to", "equal", "not equal"],
        "union": [],
        "intersection": [],
        "discard": [],
        "sort": [],
        "boolean": ["if", "is"],
        "arithmetic": ["addition", "difference", "multiplication", "division"]
    }

    sep_specials = ['@@OP_SEP@@', '@@ARG_SEP@@', '@@REF@@']

    operators = list(phrases_by_operators.keys())
    # phrases = [phrase for phrase_list in phrases_by_operators.values() for phrase in phrase_list]
    simple_specials = get_specials_qdmr()
    specials = sep_specials + operators + simple_specials
    return specials


def BREAK_vocab_program():
    specials = get_specials_program()

    # model should learn that after BOS or '@@SEP@@' it should predict operators
    # otherwise, it should predict phrase or other words or '@@SEP@@'
    # improvement- different models for each operator

    # gold should be qdmr2program ["SELECT['H. V. Jagadish']", "PROJECT['papers of #REF', '#1']", "FILTER['#2', 'that are on PVLDB']", "PROJECT['citations of #REF', '#3']", "GROUP['count', '#4', '#3']", "COMPARATIVE['#3', '#5', 'is higher than 200']"]
    # SELECT @@ARGS@@ H. V. Jagadish @@SEP@@ PROJECT @@ARGS@@ papers of #REF @@ARGS_SEP@@ #1
    # seq2seq over programs

    # Load the dataset and get the words.
    # TODO The dictionary as for now is only created from the training data. maybe change that.
    current_dir = Path()
    dir_path = current_dir / "data" / "break_data" / "vocabs"
    file_name = "vocab_logical.pkl"
    sents = BREAKLogical.load_dataset(dir_path, 'logical-forms')['train']['question_text']
    vocab = build_vocab(specials, dir_path, file_name, sents, en_tokenizer)
    return vocab


def build_vocab(specials, dir_path, file_name, sents, tokenizer):
    """
    Builds a vocabulary from the questions of the split.
    The created vocabulary is saved to a file for future usage.
    :return: The Vocab object.
    """
    if not (dir_path / file_name).is_file():
        spacy.load('en_core_web_sm')

        # Build the vocabulary with the questions, note that the questions currently are of one split only.
        counter = Counter()
        for sent in sents:
            counter.update(tokenizer(sent))
        # Save the counter and the specials.
        to_save = {'counter': counter, 'specials': specials}
        save_obj(dir_path, to_save, file_name)

    return load_vocab(dir_path, file_name)


def load_vocab(dir_path, file_name):
    """
    Loads a vocabulary from a file.
    :param dir_path: The path of the directory.
    :param file_name: The name of the vocab file.
    :return: The loaded Vocab.
    """
    properties = load_obj(dir_path, file_name)
    vocab = Vocab(properties['counter'], specials=properties['specials'])

    return vocab


def batch_to_tensor(vocab, batch, pad_max_length, device):
    """
    Converts a batch of questions to a tensor of indices from the vocab.
    In the mask, 1 means original text and 0 means padding.
    :param vocab: The vocabulary from which to take the indices.
    :param batch: The batch to convert.
    :return: out_tensor, out_mask of dim: (batch_size, pad_max_length)
    """
    # TODO Make the computation parallel.
    out_tensor = []
    out_mask = []
    for data in batch:
        # Tokenize
        tokenized_data = en_tokenizer(data) + ['<eos>']

        # Pad and create a mask
        padded = ['<pad>'] * pad_max_length
        mask = [0] * pad_max_length
        data_len = min(len(tokenized_data), pad_max_length)

        padded[:data_len] = tokenized_data[:data_len]
        mask[:data_len] = [1] * data_len

        tensor = torch.tensor([vocab[token] for token in padded], dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)

        # Add to list
        out_tensor.append(tensor)
        out_mask.append(mask)

    # Stack
    out_tensor = torch.stack(out_tensor).to(device)
    out_mask = torch.stack(out_mask).to(device)

    return out_tensor, out_mask


def tokenize_lexicon_str(vocab, lexicon_str, pad_max_length, device):
    """

    :param lexicon_str:
    :return: a tensor of ids from vocabulary for each . shape=(batch_size,max_length?)
    """
    out_tensor = []
    out_mask = []
    for row in lexicon_str:
        lexicon_words = []
        row_lst = ast.literal_eval(row)
        for phrase in row_lst:
            # handle phrase = False
            if not phrase:
                continue
            tokenized_words = en_tokenizer(phrase)
            lexicon_words.extend(tokenized_words)
        # remove duplicates
        lexicon_words = list(dict.fromkeys(lexicon_words))

        lexicon_words.extend(get_specials_program())

        # Pad and create a mask
        padded = ['<pad>'] * pad_max_length
        mask = [0] * pad_max_length
        data_len = min(len(lexicon_words), pad_max_length)

        padded[:data_len] = lexicon_words[:data_len]
        mask[:data_len] = [1] * data_len

        tensor = torch.tensor([vocab[token] for token in padded], dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)

        # Add to list
        out_tensor.append(tensor)
        out_mask.append(mask)
    # Stack
    out_tensor = torch.stack(out_tensor).to(device)
    out_mask = torch.stack(out_mask).to(device)
    return out_tensor, out_mask

def minimize_program(program):
    """
    :param program:
    :return: program without anchor tokens
    """
    ARG_SEP = '@@ARG_SEP@@ '
    OP_SEP = '@@OP_SEP@@ '
    minimized = program.replace(ARG_SEP, "")
    minimized = minimized.replace(OP_SEP, "")
    return minimized

def tensor_to_str(vocab, tensor, convert_to_program):
    """
    Converts a tensor of indices to a string.
    :param vocab: The vocab to take the strings from.
    :param tensor: A tensor of indices.
    :return: The result string.
    """
    text = " ".join(vocab.itos[idx] for idx in tensor)
    if convert_to_program:
        # first convert to untokenized form
        untokenized = prediction_to_qdmr(text)
        builder = QDMRProgramBuilder(untokenized)
        builder.build()
        text = str(builder)
    text = minimize_program(text)
    return text


def batch_to_str(vocab, batch, mask, convert_to_program):
    """

    :param vocab:
    :param batch:
    :param mask:
    :return:
    """
    lst = []
    for data_row, mask_row in zip(batch, mask):
        mask_row = mask_row == 1
        lst.append(tensor_to_str(vocab, torch.masked_select(data_row, mask_row), convert_to_program))
    return lst


def pred_batch_to_str(vocab, pred, convert_to_program):
    """
    create mask according to the first appearance of <'EOS_STR'>
    then convert to list of str
    :param vocab:
    :param pred:
    :return:
    """
    eos_id = vocab['<eos>']
    eos_mask = pred == eos_id
    if eos_id in pred:
        pass
    # operations on the mask to find first eos values in the rows
    mask_max_values, mask_max_indices = torch.max(eos_mask, dim=1)
    # include EOS token
    mask_max_indices = torch.add(mask_max_indices, 1)
    # in case there are rows with no eos
    mask_max_indices[mask_max_values == 0] = pred.shape[1]
    mask = torch.ones(pred.shape)
    # once encountered eos, mask out the rest of the prediction
    for i in range(mask.shape[0]):
        mask[i][mask_max_indices[i]:] = 0
    return batch_to_str(vocab, pred, mask, convert_to_program)
