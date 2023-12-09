import os
import string

class DataPreprocessor:
    def __init__(self, dataset_text, dataset_images, save_directory):
        self.dataset_text = dataset_text
        self.dataset_images = dataset_images
        self.save_directory = save_directory
        self.descriptions = None
        self.clean_descriptions = None
        self.vocabulary = None

    def load_fp(self, filename):
        file_path = os.path.join(self.dataset_text, filename)
        with open(file_path, 'r') as file:
            text = file.read()
        return text

    def img_capt(self, filename):
        file = self.load_fp(filename)
        captions = file.split('\n')
        descriptions = {}
        for caption in captions[:-1]:
            img, caption = caption.split('\t')
            if img[:-2] not in descriptions:
                descriptions[img[:-2]] = [caption]
            else:
                descriptions[img[:-2]].append(caption)
        return descriptions

    def txt_clean(self, captions):
        table = str.maketrans('', '', string.punctuation)
        for img, caps in captions.items():
            for i, img_caption in enumerate(caps):
                img_caption = img_caption.replace("-", " ")
                descp = img_caption.split()
                descp = [wrd.lower() for wrd in descp]
                descp = [wrd.translate(table) for wrd in descp]
                descp = [wrd for wrd in descp if (len(wrd) > 1)]
                descp = [wrd for wrd in descp if (wrd.isalpha())]
                img_caption = ' '.join(descp)
                captions[img][i] = img_caption
        return captions

    def txt_vocab(self, descriptions):
        vocab = set()
        for key in descriptions.keys():
            [vocab.update(d.split()) for d in descriptions[key]]
        return vocab

    def save_descriptions(self, descriptions, filename):
        lines = []
        for key, desc_list in descriptions.items():
            for desc in desc_list:
                lines.append(key + '\t' + desc)
        data = "\n".join(lines)
        file_path = os.path.join(self.save_directory, filename)
        with open(file_path, "w") as file:
            file.write(data)
    def preprocess_data(self, token_file="Flickr8k.token.txt", save_filename="descriptions.txt"):
        token_filepath = os.path.join(self.dataset_text, token_file)
        self.descriptions = self.img_capt(token_filepath)
        self.clean_descriptions = self.txt_clean(self.descriptions)
        self.vocabulary = self.txt_vocab(self.clean_descriptions)
        self.save_descriptions(self.clean_descriptions, save_filename)


# # Example usage:
# dataset_text = "/home/ubuntu/DL/dataset/Flicker8k_text"
# dataset_images = "/home/ubuntu/DL/dataset/Flicker8k_Dataset"
# save_directory = "/home/ubuntu/DL/code"
# preprocessor = DataPreprocessor(dataset_text, dataset_images, save_directory)
# preprocessor.preprocess_data()

# Example usage:
import os
new_directory = '/home/ubuntu/NLP/home/ubuntu/DeepLearning'
os.chdir(new_directory)

dataset_text = ""
dataset_images = ""
save_directory = "Code"
preprocessor = DataPreprocessor(dataset_text, dataset_images, save_directory)
preprocessor.preprocess_data()
