import time
import pickle

from nlp import load_dataset
from pathlib import Path
from utils.util import read_json, write_json
from collections import defaultdict


def save_obj(dir_path, obj, name):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    file_path = dir_path / (name + '.pkl')
    if not file_path.is_file():
        with open(str(file_path), 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        print("already there")


def load_obj(dir_path, name):
    file_path = dir_path / (name + '.pkl')
    with open(str(file_path), 'rb') as f:
        return pickle.load(f)


QDMR = load_dataset('break_data', 'QDMR', cache_dir='.\\data\\')
LEXICON = load_dataset('break_data', 'QDMR-lexicon', cache_dir='.\\data\\')
LOGICAL = load_dataset('break_data', 'logical-forms', cache_dir='.\\data\\')
# for t in s:
# 	print(t, s[t])
# print("QDMR len of train", len(QDMR['train']),"************")
# for p in QDMR['train'][70]:
# 	print(p, QDMR['train'][70][p])
# print("------------------------------------")
# print("LEXICON len of train", len(LEXICON['train']), "************")

# for p in LEXICON['train'][40096]:
# 	print(p, LEXICON['train'][40096][p])
# print("------------------------------------")
# print("LOGICAL len of train", len(LOGICAL['train']),"************")
# for p in LOGICAL['train'][40096]:
# 	print(p, LOGICAL['train'][40096][p])

lex_idx = 0
log_idx = 0
train_map = {}
train_check = {}
start = time.time()
# for i, example in enumerate(LOGICAL['train']):
# 	ques = example['question_text']
# 	for j in range(lex_idx, len(LEXICON['train'])):
# 		if LEXICON['train'][j]['source'] == ques:
# 			train_map[i] = LEXICON['train'][j]['allowed_tokens']
# 			train_check[i] = LEXICON['train'][j]['source']
# 			lex_idx = j + 1
# 			break


lexicon_dict = {'train': dict(), 'validation': dict(), 'test': dict()}
lexicon_check = {'train': dict(), 'validation': dict(), 'test': dict()}
for data_split in LOGICAL:
    lex_idx = 0
    lexicon_split = LEXICON[data_split]
    for i, logic_example in enumerate(LOGICAL[data_split]):
        ques = logic_example['question_text']
        for j in range(lex_idx, len(lexicon_split)):
            lexicon_example = lexicon_split[j]
            if lexicon_example['source'] == ques:
                lexicon_dict[data_split][i] = lexicon_example['allowed_tokens']
                lexicon_check[data_split][i] = lexicon_example['source']
                lex_idx = j + 1
                break

index_check = 6234
data_split = 'validation'
print("took ", time.time() - start, "sec")
for k in lexicon_check:
    print(k)
# print("check len", len(lexicon_check['train']), "************")
# print("train_map len", len(lexicon_dict['train']), "************")
#
# # for p in train_check[200]:
# print(train_check[index_check])
print("true logic", LOGICAL[data_split][index_check]['question_text'])
print("lexicon check", lexicon_check[data_split][index_check])
print("lexicon dict", lexicon_dict[data_split][index_check])
#
# print("------------------------------------")
# print("LOGICAL len of train", len(LOGICAL['train']),"************")
# for p in LOGICAL['train'][index_check]:
# 	print(p, LOGICAL['train'][index_check][p])

current_dir = Path()
dir_lexicon = current_dir / "data" / "break_data" / "lexicon_by_logical"

# write_json(train_map, file_lexicon)
# data = read_json(file_lexicon)
save_obj(dir_lexicon, lexicon_dict, 'test1')
data = load_obj(dir_lexicon, 'test1')
print("result after load---------------")
print(data[data_split][index_check])

# GAL's code
# lex_train = {}
# s = time.time()
# for i, ex in enumerate(LOGICAL['train']):
#     lex_train[ex['question_text']] = [i]
#
# for i, ex in enumerate(LEXICON['train']):
#     if ex['source'] in lex_train:
#         lex_train[ex['source']].append(i)
#
# lex_train = {i: j for i, j in lex_train.values()}
#
# print(time.time() - s)
#
# print(len(lex_train))
#
# for a in LOGICAL['train'][777]:
#     print(a, LOGICAL['train'][777][a])
# print('question_text', LEXICON['train'][lex_train[777]]['source'])
#
# np.save('data/train_lex_idx')
