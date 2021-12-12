"""Dataset reader and process"""
import os
import numpy as np

from spacy.lang.de import German
from datasets import load_dataset
from src.lib.utils.words_ripper import ripper_level_one

class Dataset():

    def __init__(self):
        self.dataset_name = "gnad10"
        self.partitions = ['train', 'test']
        self.dataset = dict()
        self.size = {'total': 0}
        self.nlp = None
        self.directory_data_path = os.path.join("..", "data")
        self.directory_spelling_path = os.path.join(self.directory_data_path, "spelling")
        self.directory_spelling_words_path = os.path.join(self.directory_data_path, "spelling-words")
        self.directory_punctuation_path = os.path.join(self.directory_data_path, "punctuation")

        self.create_directory()

        for pt in self.partitions:
            self.size[pt] = 0

    def create_directory(self):
        if not os.path.exists(self.directory_data_path):
            os.makedirs(self.directory_data_path)

        if not os.path.exists(self.directory_spelling_path):
            os.makedirs(self.directory_spelling_path)

        if not os.path.exists(self.directory_spelling_words_path):
            os.makedirs(self.directory_spelling_words_path)

        if not os.path.exists(self.directory_punctuation_path):
            os.makedirs(self.directory_punctuation_path)

    def read_dataset(self):
        self.dataset = load_dataset(self.dataset_name)

    def init_nlp(self):
        self.nlp = German()
        self.nlp.add_pipe("sentencizer") # pipeline for sentances iteration

    def generate_spelling_data(self):
        self.init_nlp()
        train_dataset = self.dataset['train']
        test_dataset = self.dataset['test']

        print("Generating train spelling dataset...")
        self.generate_list_sentances(train_dataset, "spelling-train.txt")

        print("Generating test spelling dataset...")
        self.generate_list_sentances(test_dataset, "spelling-test.txt")

    def generate_spelling_words_data(self):
        self.init_nlp()
        train_dataset = self.dataset['train']
        test_dataset = self.dataset['test']

        print("Generating train spelling words dataset...")
        self.generate_list_words(train_dataset, "spelling-words-train.txt")

        print("Generating test spelling words dataset...")
        self.generate_list_words(test_dataset, "spelling-words-test.txt")

    def generate_punctuation_data(self):
        self.init_nlp()
        train_dataset = self.dataset['train']
        test_dataset = self.dataset['test']

        print("Generating train punctuation dataset...")
        self.generate_list_punctuation(train_dataset, "punctuation-train.txt")

        print("Generating test punctuation dataset...")
        self.generate_list_punctuation(test_dataset, "punctuation-test.txt")

    def get_words_dict(self, dataset):
        words_set = set()
        words_dict = dict()

        for row in dataset:
            doc = self.nlp(row['text'])

            for token in doc:
                if token.is_alpha:
                    words_set.add(token.text)

        for word in words_set:
            # Get the ripper set
            set_ripper = ripper_level_one(word)

            # Get the list of the set_ripper that are not in words_set
            words_not_in_set = [w for w in set_ripper if w not in words_set]

            # Add the list (word, words_not_in_set) in the words_dict
            words_dict[word] = words_not_in_set

        return words_dict

    def generate_list_sentances(self, dataset, file_name):
        words_dict = self.get_words_dict(dataset)

        f = open(os.path.join(self.directory_spelling_path, file_name), "w", encoding="UTF-8")

        sentences_count = 0

        for row in dataset:
            sentences = [j for j in self.nlp(row['text']).sents]

            for sentence in sentences:
                for token in sentence:
                    if token.is_alpha:
                        words_ripper = words_dict[token.text]
                        sentence_text = sentence.text
                        f.write("#" + str(sentences_count) + ": " + sentence_text + "\n")

                        for word in words_ripper:
                            ripper_sentence_text = sentence_text.replace(token.text, word)
                            f.write("#" + str(sentences_count) + ": " + ripper_sentence_text + "\n")

                sentences_count += 1

        f.close()

    def generate_list_words(self, dataset, file_name):
        words_dict = self.get_words_dict(dataset)

        f = open(os.path.join(self.directory_spelling_words_path, file_name), "w", encoding="UTF-8")

        words_count = 0

        for key in words_dict.keys():
            f.write("#" + str(words_count) + ": " + key + "\n")

            for word in words_dict[key]:
                f.write("#" + str(words_count) + ": " + word + "\n")

            words_count += 1

        f.close()

    def generate_list_punctuation(self, dataset, file_name):

        f = open(os.path.join(self.directory_punctuation_path, file_name), "w", encoding="UTF-8")

        sentence_count = 0

        for row in dataset:
            sentences = [j for j in self.nlp(row['text']).sents]

            for sentence in sentences:
                f.write("#" + str(sentence_count) + ": " + sentence.text + "\n")

                for token in sentence:
                    if token.text == "." or token.text == ",":
                        sentence_without_punct = ""

                        for token_j in sentence:

                            if token_j.i != token.i:
                                sentence_without_punct += token_j.text

                            if token_j.whitespace_:
                                sentence_without_punct += token_j.whitespace_

                            # if sentence_without_punct == "" and token_j.i != token.i:
                            #     sentence_without_punct += token_j.text
                            # elif token_j.i != token.i:
                            #     sentence_without_punct += " " + token_j.text

                        f.write("#" + str(sentence_count) + ": " + sentence_without_punct + "\n")

                sentence_count += 1

        f.close()

def read_from_txt(file_name, max_sentances=None, items_per_sentance=None):

    dt = {"dt": [], "gt": []}
    list_items = {"dt": [], "gt": []}

    if not os.path.exists(file_name):
        print("Path '" + file_name + "' does not exist.")
        exit(-1)

    with open(file_name, "r", encoding="UTF-8") as f:

        actual_number = -1
        actual_text = ""
        actual_item = -1

        for line in f:
            arr = line.split()

            if len(arr) == 0:
                continue

            x = " ".join(arr[1::])

            number = int(arr[0][1:-1])

            if number == actual_number:
                if items_per_sentance:
                    list_items['dt'].append(x)
                    list_items['gt'].append(actual_text)
                else:
                    dt['dt'].append(x)
                    dt['gt'].append(actual_text)
            else:

                if actual_number >= 0 and items_per_sentance:
                    arange = np.arange(len(list_items['gt']))
                    np.random.shuffle(arange)

                    for i in np.arange(items_per_sentance):
                        dt['dt'].append(list_items['dt'][arange[i]])
                        dt['gt'].append(list_items['gt'][arange[i]])

                    list_items['dt'].clear()
                    list_items['gt'].clear()

                if max_sentances and number >= max_sentances:
                    return dt

                actual_number = number
                actual_text = x

                if items_per_sentance:
                    list_items['dt'].append(actual_text)
                    list_items['gt'].append(actual_text)
                else:
                    dt['dt'].append(actual_text)
                    dt['gt'].append(actual_text)

    return dt