# # from nlp import load_dataset
# # import time
# #
# # logical = load_dataset('break_data', 'logical-forms', cache_dir='.\\data\\')
# # qdmr_lexicon = load_dataset('break_data', 'QDMR-lexicon', cache_dir='.\\data\\')
# #
# # print(logical)
# # print(qdmr_lexicon)
# # # for a in qdmr_lexicon['train']:
# # #     print(a['source'])
# # lex_train = {}
# # s = time.time()
# # for i, ex in enumerate(logical['train']):
# #     lex_train[ex['question_text']] = [i]
# #
# # for i, ex in enumerate(qdmr_lexicon['train']):
# #     if ex['source'] in lex_train:
# #         lex_train[ex['source']].append(i)
# #
# # lex_train = {i: j for i, j in lex_train.values()}
# #
# # print(time.time() - s)
# #
# # print(len(lex_train))
# #
# # for a in logical['train'][777]:
# #     print(a, logical['train'][777][a])
# # print('question_text', qdmr_lexicon['train'][lex_train[777]]['source'])
#
# # from data_loader.data_loaders import BREAKDataLoader
# # import re
# # break_dataset = BREAKDataLoader('data/', 128, True, 0.1, 2)
# # validation_split = break_dataset.split_validation()
#
# # random_example = break_dataset.dataset.get_example()
#
# # break_dataset.dataset.visualize(*random_example)
# # print(re.sub(r'#(\d+)', r'@@\1@@', random_example[1]))
# # print(sorted(list(set([tok.strip() for tok in random_example[2]]))))
# #
# # random_example = validation_split.dataset.get_random_example()
# # for part in random_example:
# #     print(part)
#
# # from abc import abstractmethod
# #
# #
# # class A:
# #     @abstractmethod
# #     def a(self):
# #         print('A')
# #
# #
# # class B(A):
# #     def a(self):
# #         print('B')
# #
# #
# # class C(A):
# #     pass
# #
# #
# # a = A()
# # b = B()
# # c = C()
# #
# # a.a()
# # b.a()
# # c.a()
#
# # import logging
# #
# # # create logger
# # logger = logging.getLogger('simple_example')
# # logger.setLevel(logging.DEBUG)
# #
# # # create console handler and set level to debug
# # ch = logging.StreamHandler()
# # ch.setLevel(logging.DEBUG)
# #
# # # create formatter
# # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# #
# # # add formatter to ch
# # ch.setFormatter(formatter)
# #
# # # add ch to logger
# # logger.addHandler(ch)
# #
# # # 'application' code
# # logger.debug('debug message')
# # logger.info('info message')
# # logger.warning('warn message')
# # logger.error('error message')
# # logger.critical('critical message')
#
# from torchtext.data.utils import get_tokenizer
# from data_loader.data_loaders import BREAKDataLoader
# from data_loader.custom_datasets import BREAKLogical
#
# # print("starting")
# # en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
# # # break_dataloader = BREAKDataLoader('data/', 1, True, 0, 1)
# training = BREAKLogical('data/', train=True, valid=False)
# validation = BREAKLogical('data/', train=True, valid=True)
# testing = BREAKLogical('data/', train=False, valid=False)
# print(testing[50])

# count = 0
# for ex in testing:
#     if ex[1]:
#         count += 1
#
# print(count, len(testing))
# max_len = 0
# idx = 0
# # enumerate(break_dataloader)
# # for data, target in list(training) + list(validation) + list(testing):
# #     # print(data)
# #     max_len = max(max_len, len(en_tokenizer(target)))
# #     idx += 1
# #     if idx % 100 == 0:
# #         print("done", idx)
# #
# # print(max_len)
# # print("Done")
# from data_loader import batch_to_tensor, BREAK_vocab_simple, en_tokenizer
# vocab = BREAK_vocab_simple()
#
# data1, tar1 = training[3]
# data2, tar2 = training[5]
# data_pad_length = 64
# target_pad_length = 256
# print(data1)
# print(data2)
# batch_data = [data1, data2]
# batch_target = [tar1, tar2]
# padded_data, masks_data = batch_to_tensor(vocab, batch_data, data_pad_length, 'cpu')
# padded_target, masks_target = batch_to_tensor(vocab, batch_target, target_pad_length, 'cpu')
# print("padded data shape", padded_data.shape)
# print("data masks shape", masks_data.shape)
#
# print("tar1 len before pad", len(en_tokenizer(tar1)))
#
# print("padded target shape", padded_target.shape)
# print("target masks shape", masks_target.shape)
# print("tar1 interesting", masks_target[0].sum())

# import torch
# a=torch.tensor([[1,2],[3,4]])
# print(a)
# for i, it in enumerate(torch.transpose(a,0, 1)):
#     print(i, it)
# from tester.BREAK_evaluate_predictions import evaluate, get_exact_match
# ids = []
# questions = []
# decomps = []
# golds = []
# metadata = []
# output_path = ""
#
# id, question, gold = training[0]
# print(evaluate([id], [question], [gold], [gold], None, 'saved'))
# print(get_exact_match(training[0][1], training[0][1]))


import torch
import numpy as np
#
# eos_id = 3
# a = torch.tensor([[1, 3, 3, 4], [1, 1, 1, 1], [1, 2, 3, 3]])
#
# # non zero values mask
# eos_mask = a == eos_id
# # operations on the mask to find first non_eos values in the rows
# mask_max_values, mask_max_indices = torch.max(eos_mask, dim=1)
# mask_max_indices[mask_max_values == 0] = a.shape[1]
# mask = torch.ones(a.shape)
# for i in range(mask.shape[0]):
#     mask[i][mask_max_indices[i]:] = 0
# # a.gather(1, mask_max_indices.unsqueeze(1))
#
# # mask[0][4:] =3
# print(mask)

# a= [1,2,3,4]
# b =[1,2,3,3]
# c = sum([x==y for x,y in zip(a,b)])
# d = np.array(c).sum()
# print(c)


from model.metric import ged_score_decomp
from tester.BREAK_evaluate_predictions import format_qdmr
from utils.qdmr_identifier import *

# gold_str = [
#     'return H. V. Jagadish ;return papers of #1 ;return #2 that  are on PVLDB ;return citations of #3 ;return number '
#     'of  #4 for each #3 ;return #3 where #5 is  higher than 200']
#
# pred_str = [
#     'return H. V. Jagadish']

# gold = [format_qdmr(decomp) for decomp in gold_str]
# pred = [format_qdmr(decomp) for decomp in pred_str]


qdmr = "return H. V. Jagadish ;return papers of #1 ;return #2 that  are on PVLDB ;return citations of #3 ;return number of  #4 for each #3 ;return #3 where #5 is  higher than 200"
print("* Original QDMR: \n%s\n" % qdmr)
builder = QDMRProgramBuilder(qdmr)
builder.build()
program = str(builder)
print("* Identified steps: \n%a\n" % program)
# steps = [str(step) for step in builder.steps]
# print("* Identified steps: \n%a\n" % steps)
ops = [str(op) for op in builder.operators]
print("* Identified operators: \n%a\n" % ops)
# convert gold_qdmrs to logical_forms
# convert logical forms to prediction_forms (argsep, usage of phrases)
# build SARI metric (based of strings) over prediction_forms
