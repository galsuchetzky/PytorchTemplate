import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

spacy.prefer_gpu()

# nlp = spacy.load('en_core_web_sm')

nlp = English()
# Create a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nlp.vocab)

print(type(tokenizer('this is a sentence')))
dragon = nlp.vocab.strings['dragon']
print(dragon in nlp.vocab)