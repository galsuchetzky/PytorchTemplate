# from nlp import load_dataset
# import time
#
# logical = load_dataset('break_data', 'logical-forms', cache_dir='.\\data\\')
# qdmr_lexicon = load_dataset('break_data', 'QDMR-lexicon', cache_dir='.\\data\\')
#
# print(logical)
# print(qdmr_lexicon)
# # for a in qdmr_lexicon['train']:
# #     print(a['source'])
# lex_train = {}
# s = time.time()
# for i, ex in enumerate(logical['train']):
#     lex_train[ex['question_text']] = [i]
#
# for i, ex in enumerate(qdmr_lexicon['train']):
#     if ex['source'] in lex_train:
#         lex_train[ex['source']].append(i)
#
# lex_train = {i: j for i, j in lex_train.values()}
#
# print(time.time() - s)
#
# print(len(lex_train))
#
# for a in logical['train'][777]:
#     print(a, logical['train'][777][a])
# print('question_text', qdmr_lexicon['train'][lex_train[777]]['source'])

from data_loader.data_loaders import BREAKDataLoader
import re
break_dataset = BREAKDataLoader('data/', 128, True, 0.1, 2)
# validation_split = break_dataset.split_validation()

random_example = break_dataset.dataset.get_example()

break_dataset.dataset.visualize(*random_example)
# print(re.sub(r'#(\d+)', r'@@\1@@', random_example[1]))
# print(sorted(list(set([tok.strip() for tok in random_example[2]]))))
#
# random_example = validation_split.dataset.get_random_example()
# for part in random_example:
#     print(part)

# from abc import abstractmethod
#
#
# class A:
#     @abstractmethod
#     def a(self):
#         print('A')
#
#
# class B(A):
#     def a(self):
#         print('B')
#
#
# class C(A):
#     pass
#
#
# a = A()
# b = B()
# c = C()
#
# a.a()
# b.a()
# c.a()

# import logging
#
# # create logger
# logger = logging.getLogger('simple_example')
# logger.setLevel(logging.DEBUG)
#
# # create console handler and set level to debug
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
#
# # create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
# # add formatter to ch
# ch.setFormatter(formatter)
#
# # add ch to logger
# logger.addHandler(ch)
#
# # 'application' code
# logger.debug('debug message')
# logger.info('info message')
# logger.warning('warn message')
# logger.error('error message')
# logger.critical('critical message')