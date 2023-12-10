#%%
import os   # handling the files
import pickle # storing numpy features
import numpy as np
from tqdm.notebook import tqdm # how much data is process till now

from tensorflow.keras.applications.vgg16 import VGG16 , preprocess_input # extract features from image data.
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input , Dense , LSTM , Embedding , Dropout , add

#%%
# dataset_text = "/home/ubuntu/NLP/home/ubuntu/DeepLearning/Flicker8k_text"
# dataset_images = "/home/ubuntu/NLP/home/ubuntu/DeepLearning/Flicker8k_Dataset"
BASE_DIR = "/home/ubuntu/NLP/home/ubuntu/DeepLearning/"
CODE_DIR = "/home/ubuntu/NLP/home/ubuntu/DeepLearning/Code"

# BASE_DIR = '/kaggle/input/flickr8k'
# WORKING_DIR = '/kaggle/working'

#%%
# Load vgg16 Model
base_model = VGG16()
# restructure model
model = Model(inputs = base_model.inputs , outputs = base_model.layers[-2].output)
# Summerize
print(model.summary())

#%%
# extract features from image
features = {}
directory = os.path.join(BASE_DIR, 'Flicker8k_Dataset')

for img_name in tqdm(os.listdir(directory)):
    # load the image from file
    img_path = directory + '/' + img_name
    image = load_img(img_path, target_size=(224, 224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)
    # extract features
    feature = model.predict(image, verbose=0)
    # get image ID
    image_id = img_name.split('.')[0]
    # store feature
    features[image_id] = feature


# Dictionary 'features' is created and will be loaded with the extracted features of image data
# load_img(img_path, target_size=(224, 224)) - custom dimension to resize the image when loaded to the array
# image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) - reshaping the image data to preprocess in a RGB type image.
# model.predict(image, verbose=0) - extraction of features from the image
# img_name.split('.')[0] - split of the image name from the extension to load only the image name.

#%%
# store features in pickle
pickle.dump(features, open(os.path.join(CODE_DIR, 'features.pkl'), 'wb'))

#Extracted features are not stored in the disk, so re-extraction of features can extend running time
#Dumps and store your dictionary in a pickle for reloading it to save time


#%%
# load features from pickle
with open(os.path.join(CODE_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)
# Load all your stored feature data to your project for quicker runtime

#%%
# Load the Captions Data
with open(os.path.join(CODE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()

#%%
# split and append the captions data with the image
# create mapping of image to captions
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)

# Dictionary 'mapping' is created with key as image_id and values as the corresponding caption text
# Same image may have multiple captions,
# if image_id not in mapping: mapping[image_id] = [] creates a list for appending captions to the corresponding image


#%%
# no. of images loaded
len(mapping)

#%%
# Preprocess Text Data
from tqdm import tqdm

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc.,
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption


#%%
# before preprocess of text
mapping['1000268201_693b08cb0e']

#%%
# preprocess the text
clean(mapping)

#%%
# after preprocess of text
mapping['1000268201_693b08cb0e']


#%%
# store the preprocessed captions into a list
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

#%%
#No. of unique captions stored
len(all_captions)


#%%
# Processing of Text Data
# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

#%%
# No. of unique words
vocab_size
# 8485

#%%
# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
print(max_length)
#35

#%%
#Train Test Split
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]

#%%
# define a batch and include the padding sequence
# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq],num_classes=vocab_size)[0]
                    # store the sequences
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0

#%%
# Model Creation
# encoder model
# image feature layers

inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

print("done for model")
# plot the model
# plot_model(model, show_shapes=True)

#%%
# train the model
epochs = 20
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    # create data generator
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

#%%
# save the model
model.save(CODE_DIR+'/best_model.h5')
# save the model in the working directory for reuse

#%%
# Generate Captions for the Image
# Convert the predicted index from the model into a word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

#%%
# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    return in_text
# Captiongenerator appending all the words for an image
# The caption starts with 'startseq' and the model continues to predict the caption until the 'endseq' appeared

#%%
# Model Validation
# Now we validate the data using BLEU Score
from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()

for key in tqdm(test):
    # get actual caption
    captions = mapping[key]
    # predict the caption for image
    y_pred = predict_caption(model, features[key], tokenizer, max_length)
    # split into words
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    # append to the list
    actual.append(actual_captions)
    predicted.append(y_pred)
# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

# BLEU Score is used to evaluate the predicted text against a reference text, in a list of tokens.
#
# The reference text contains all the words appended from the captions data (actual_captions)
#
# A BLEU Score more than 0.4 is considered a good result, for a better score increase the no. of epochs accordingly.

#%%
# Visualize the Results

from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Flicker8k_Dataset", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)
    plt.show()

# Image caption generator defined
#
# First prints the actual captions of the image then prints a predicted caption of the image

#%%
generate_caption("1001773457_577c3a7d70.jpg")
# generate_caption("1022454428_b6b660a67b.jpg")