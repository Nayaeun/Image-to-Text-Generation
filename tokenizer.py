from transformers import BertTokenizer
import torch

# Load pre-trained BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Convert dictionary to clear list of descriptions
def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# Tokenize using BERT tokenizer
def tokenize_descriptions(descriptions):
    tokenized_descriptions = []
    for desc in descriptions:
        tokens = bert_tokenizer.encode(desc, add_special_tokens=True, max_length=64, truncation=True)
        tokenized_descriptions.append(tokens)
    return tokenized_descriptions

# Tokenize descriptions
desc_list = dict_to_list(train_descriptions)
tokenized_desc_list = tokenize_descriptions(desc_list)

# Calculate the maximum length of descriptions
max_length_value = max(len(tokens) for tokens in tokenized_desc_list)
print("Max length of description:", max_length_value)
