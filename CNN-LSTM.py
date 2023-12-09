from transformers import BertTokenizer, BertModel
import torch
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, add, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

class CaptioningModelWithBERT:
    def __init__(self, vocab_size, max_length):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_bert_length = 64  # Adjust as needed

    def define_model(self):
        # Features from the CNN model compressed from 2048 to 256 nodes
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)

        # BERT sequence model
        inputs_bert = Input(shape=(self.max_bert_length,))
        bert_embeddings = self.get_bert_embeddings(inputs_bert)

        # Merging both models
        decoder1 = add([fe2, bert_embeddings])
        decoder2 = Dense(256, activation='relu')(decoder1)

        # LSTM sequence model
        inputs2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)

        # Merge it [image, bert] [seq] [word]
        decoder3 = add([decoder2, se3])
        outputs = Dense(self.vocab_size, activation='softmax')(decoder3)

        # Merge it [image, bert, seq] [word]
        model = Model(inputs=[inputs1, inputs_bert, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        # Summarize model
        print(model.summary())
        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

        return model

    def get_bert_embeddings(self, inputs):
        input_ids = torch.Tensor(self.bert_tokenizer.encode("Example input", add_special_tokens=True)).unsqueeze(0)
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        outputs = bert_model(input_ids)
        bert_embeddings = outputs.last_hidden_state

        # Reduce BERT embeddings dimension to match LSTM input size
        bert_embeddings = Dense(256, activation='relu')(bert_embeddings)
        return bert_embeddings

# Example usage:
# Instantiate the class
captioning_model = CaptioningModelWithBERT(vocab_size, max_length_value)
# Define the model
model = captioning_model.define_model()

