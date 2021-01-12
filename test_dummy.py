from nlp import load_dataset
import time
import pickle

def save_obj(obj, name ):
    with open('/data/break_data/lexicon_by_logical' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('/data/break_data/lexicon_by_logical' + name + '.pkl', 'rb') as f:
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
print("LEXICON len of train", len(LEXICON['train']),"************")

# for p in LEXICON['train'][40096]:
# 	print(p, LEXICON['train'][40096][p])
# print("------------------------------------")
# print("LOGICAL len of train", len(LOGICAL['train']),"************")
# for p in LOGICAL['train'][40096]:
# 	print(p, LOGICAL['train'][40096][p])

# # lex_idx = {i:j for i in enumerate(LOGICAL['train'])
# lex_idx = 0
# log_idx = 0
# train_map = {}
# train_check = {}
# start = time.time()
# for i, example in enumerate(LOGICAL['train']):
# 	ques = example['question_text']
# 	for j in range(lex_idx, len(LEXICON['train'])):
# 		if LEXICON['train'][j]['source'] == ques:
# 			train_map[i] = LEXICON['train'][j]['allowed_tokens']
# 			train_check[i] = LEXICON['train'][j]['source']
# 			lex_idx = j + 1
# 			break
#
# index_check = 35001
# print("took ", time.time() - start, "sec")
# print("check len", len(train_check),"************")
# print("train_map len", len(train_map),"************")
#
# # for p in train_check[200]:
# print(train_check[index_check])
# print(train_map[index_check])
#
# print("------------------------------------")
# print("LOGICAL len of train", len(LOGICAL['train']),"************")
# for p in LOGICAL['train'][index_check]:
# 	print(p, LOGICAL['train'][index_check][p])

from pathlib import Path
current_dir = Path()
dir_lexicon = current_dir / "data/break_data/lexicon_by_logical"
Path(dir_lexicon).mkdir(parents=True, exist_ok=True)

print(a)
# save_obj(train_map, 'test1')
# data = load_obj('test1')


# j = json.dumps(train_map, indent=4)
# f = open('tmp_file.json', 'w')
# print >> f, j
# f.close()
#
# data={}
# with open('tmp_file.json') as json_file:
#     data = json.load(json_file)
#     print(data[0])
#     print(len(data))
# print(data[index_check])
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