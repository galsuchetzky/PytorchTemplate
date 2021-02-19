from transformers import BartForConditionalGeneration, BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", additional_special_tokens = ["@@OP_SEP@@", "@@ARG_SEP@@", "@@SEP@@", "@@REF@@"])
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
# Tokenize input (dummy example)
text = "'select @@OP_SEP@@ H. Jagadish @@SEP@@ project @@OP_SEP@@ papers"
tokenized_text = tokenizer.tokenize(text)
#outputs
print(tokenized_text)

# tokenizer("Hello world")['input_ids']