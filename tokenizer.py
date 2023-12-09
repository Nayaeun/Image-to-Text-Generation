# from transformers import BertTokenizer
#
# class BertTextTokenizer:
#     def __init__(self, model_name='bert-base-uncased', max_length=64):
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.max_length = max_length
#
#     def dict_to_list(self, descriptions):
#         all_desc = []
#         for key in descriptions.keys():
#             [all_desc.append(d) for d in descriptions[key]]
#         return all_desc
#
#     def tokenize_descriptions(self, descriptions):
#         tokenized_descriptions = []
#         for desc in descriptions:
#             tokens = self.tokenizer.encode(desc, add_special_tokens=True, max_length=self.max_length, truncation=True)
#             tokenized_descriptions.append(tokens)
#         return tokenized_descriptions
#
#     def tokenize_and_get_max_length(self, descriptions):
#         desc_list = self.dict_to_list(descriptions)
#         tokenized_desc_list = self.tokenize_descriptions(desc_list)
#         max_length_value = max(len(tokens) for tokens in tokenized_desc_list)
#         return max_length_value
#
# # Example Usage
# # Assuming train_descriptions is already defined
# bert_tokenizer = BertTextTokenizer()
# max_length_value = bert_tokenizer.tokenize_and_get_max_length(train_descriptions)
# print("Max length of description:", max_length_value)

from transformers import BertTokenizer
import torch

class BertTextTokenizer:
    def __init__(self, model_name='bert-base-uncased', max_length=64):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def dict_to_list(self, descriptions):
        all_desc = []
        for key in descriptions.keys():
            [all_desc.append(d) for d in descriptions[key]]
        return all_desc

    def tokenize_descriptions(self, descriptions):
        tokenized_descriptions = []
        for desc in descriptions:
            tokens = self.tokenizer.encode(desc, add_special_tokens=True, max_length=self.max_length, truncation=True)
            tokenized_descriptions.append(tokens)
        return tokenized_descriptions

    def tokenize_and_get_max_length(self, descriptions):
        desc_list = self.dict_to_list(descriptions)
        tokenized_desc_list = self.tokenize_descriptions(desc_list)
        max_length_value = max(len(tokens) for tokens in tokenized_desc_list)
        return max_length_value

    def get_vocab_size(self, descriptions):
        desc_list = self.dict_to_list(descriptions)
        self.tokenizer.fit(desc_list)
        vocab_size = len(self.tokenizer)
        return vocab_size

# Example Usage
# Assuming train_descriptions is already defined
bert_tokenizer = BertTextTokenizer()
max_length_value = bert_tokenizer.tokenize_and_get_max_length(train_descriptions)
vocab_size = bert_tokenizer.get_vocab_size(train_descriptions)
tokenizer = bert_tokenizer.tokenizer

print("Tokenizer:", tokenizer)
print("Vocabulary Size:", vocab_size)
print("Max length of description:", max_length_value)
