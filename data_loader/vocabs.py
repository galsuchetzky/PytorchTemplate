import spacy
import torch

from pathlib import Path
from collections import Counter
from torchtext.vocab import Vocab
from utils import save_obj, load_obj
from data_loader import BREAKLogical
from torchtext.data.utils import get_tokenizer

# English tokenizer for tokenizing natural language sentences when building a vocabulary.
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


def BREAK_vocab_simple():
    """
    This function builds a vocabulary for the simple seq2seq model of qdmr.
    :return: The vocab.
    """
    # Special characters to include in the vocabulary.
    specials = ['<unk>', '<sos>', '<eos>', '@@10@@', '@@11@@', '@@12@@', '@@13@@',
                '@@14@@', '@@15@@', '@@16@@', '@@17@@', '@@18@@', '@@19@@', '@@1@@',
                '@@2@@', '@@3@@', '@@4@@', '@@5@@', '@@6@@', '@@7@@', '@@8@@', '@@9@@']

    # Load the dataset and get the words.
    # TODO The dictionary as for now is only created from the training data. maybe change that.
    current_dir = Path()
    dir_path = current_dir / "data" / "break_data" / "vocabs"
    file_name = "vocab_counter.pkl"
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


def batch_to_tensor(vocab, batch):
    """
    Converts a batch of questions to a tensor of indices from the vocab.
    :param vocab: The vocabulary from which to take the indices.
    :param batch: The batch to convert.
    :return: The index tensor.
    """
    # TODO change it to pad the outputs to a max length and hstack them to a 2d tensor.
    #  Make the computation parallel.
    out_tensor = []
    for question in batch:
        out_tensor.append(torch.tensor([vocab[token] for token in en_tokenizer(question)],
                                       dtype=torch.long).unsqueeze(1))

    out_tensor = torch.vstack(out_tensor)
    return out_tensor


def tensor_to_str(vocab, tensor):
    """
    Converts a tensor of indices to a string.
    :param vocab: The vocab to take the strings from.
    :param tensor: A tensor of indices.
    :return: The result string.
    """
    return " ".join(vocab.itos[idx] for idx in tensor)
