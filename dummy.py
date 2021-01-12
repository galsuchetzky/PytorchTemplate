from nlp import load_dataset
import time

logical = load_dataset('break_data', 'logical-forms', cache_dir='.\\data\\')
qdmr_lexicon = load_dataset('break_data', 'QDMR-lexicon', cache_dir='.\\data\\')

print(logical)
print(qdmr_lexicon)
# for a in qdmr_lexicon['train']:
#     print(a['source'])
lex_train = {}
s = time.time()
for i, ex in enumerate(logical['train']):
    lex_train[ex['question_text']] = [i]

for i, ex in enumerate(qdmr_lexicon['train']):
    if ex['source'] in lex_train:
        lex_train[ex['source']].append(i)

lex_train = {i: j for i, j in lex_train.values()}

print(time.time() - s)

print(len(lex_train))

for a in logical['train'][777]:
    print(a, logical['train'][777][a])
print('question_text', qdmr_lexicon['train'][lex_train[777]]['source'])