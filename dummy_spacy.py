import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
import io
from data_loader import BREAKDataLoader
import spacy

spacy.load('en_core_web_sm')
# url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
# train_urls = ('train.de.gz', 'train.en.gz')
# val_urls = ('val.de.gz', 'val.en.gz')
# test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

# train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
# val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
# test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

# de_tokenizer = get_tokenizer('spacy', language='de')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


def build_vocab(sents, tokenizer):
    counter = Counter()
    for string_ in sents:
        counter.update(tokenizer(string_))
    return Vocab(counter, specials=['<unk>', '<sos>', '<eos>', '@@10@@', '@@11@@', '@@12@@', '@@13@@', '@@14@@',
                                    '@@15@@', '@@16@@', '@@17@@', '@@18@@', '@@19@@', '@@1@@', '@@2@@', '@@3@@',
                                    '@@4@@', '@@5@@', '@@6@@', '@@7@@', '@@8@@', '@@9@@'])


dataloader = BREAKDataLoader('data/', 128)

sents = dataloader.dataset.questions
# de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
en_vocab = build_vocab(sents, en_tokenizer)

# print(en_vocab[0])
# print(en_vocab['higher than'])
# print(en_vocab[';'])
# print(en_vocab[';'])


def data_process(sents):
    data = []
    for sent in sents:
        tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(sent)],
                               dtype=torch.long)
        data.append(tensor_)
    return data


data = data_process(sents)

# print(data[0])

# train_data = data_process(train_filepaths)
# val_data = data_process(val_filepaths)
# test_data = data_process(test_filepaths)
