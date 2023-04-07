#!/usr/bin/env python3
# encoding: utf-8

"""
    Data pre-processing

    Author : TF4ces
"""


# Native imports
import re
import string
import pickle

# Third-party imports
import nltk
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm

# User imports
from config.conf import __WORKSPACE__
from src.TF4ces_search_engine.feature import config_preprocessing
from src.utils.file_handlers import delete_dir


class DataPreprocessing():

    def __init__(
            self,
            cache_dir=__WORKSPACE__ / "dataset" / "preprocessed" / "test"
    ):

        self.cache_dir = cache_dir
        self.cache_doc_file = self.cache_dir / "docs.preprocessed.pkl"
        self.cache_query_file = self.cache_dir / "queries.preprocessed.pkl"

        # download required nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)

    def text2int(self, textnum, numwords={}):
        if not numwords:
            units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
            ]

            tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

            scales = ["hundred", "thousand", "million", "billion", "trillion"]

            numwords["and"] = (1, 0)
            for idx, word in enumerate(units):  numwords[word] = (1, idx)
            for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
            for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

        ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5, 'eighth':8, 'ninth':9, 'twelfth':12}
        ordinal_endings = [('ieth', 'y'), ('th', '')]

        textnum = textnum.replace('-', ' ')

        current = result = 0
        curstring = ""
        onnumber = False
        for word in textnum.split():
            if word in ordinal_words:
                scale, increment = (1, ordinal_words[word])
                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True
            else:
                for ending, replacement in ordinal_endings:
                    if word.endswith(ending):
                        word = "%s%s" % (word[:-len(ending)], replacement)

                if word not in numwords:
                    if onnumber:
                        curstring += repr(result + current) + " "
                    curstring += word + " "
                    result = current = 0
                    onnumber = False
                else:
                    scale, increment = numwords[word]

                    current = current * scale + increment
                    if scale > 100:
                        result += current
                        current = 0
                    onnumber = True

        if onnumber:
            curstring += repr(result + current)

        return curstring

    def tokenize_text(self, text, preprocessing_switches):
        if preprocessing_switches["separate_out_punctuation"]:
            text = re.sub(r"(\w)([.,;:!?'\"”\)])", r"\1 \2", text) # separates punctuation at ends of strings
            text = re.sub(r"([.,;:!?'\"“\(\)])(\w)", r"\1 \2", text) # separates punctuation at beginning of strings
        if preprocessing_switches["convert_numbers"]:
            text = re.sub('\d+', 'NUMBER',text)
        # print("tokenising:", text) # uncomment for debugging
        tokens = text.split()
        return tokens

    def remove_characters_after_tokenization(self, tokens):
        # note preserving critical social media/twitter characters @ and #
        p = '[{}]'.format(re.escape(string.punctuation)+'\…').replace("@", "").replace("\#", "")
        #print(p)
        pattern = re.compile(p)
        filtered_tokens = [f for f in filter(None, [pattern.sub('', token) for token in tokens])]
        return filtered_tokens

    def convert_to_lowercase(self, tokens):
        return [token.lower() for token in tokens if token.isalpha()]

    def remove_stopwords(self, tokens):
        stopword_list = nltk.corpus.stopwords.words('english')
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        return filtered_tokens

    def apply_lemmatization(self, tokens, wnl=WordNetLemmatizer()):
        return [wnl.lemmatize(token) for token in tokens]

    def switches(self, text, preprocessing_switches, model_type):
        """ Technique which will apply the techniques if they are set to
        True in the global dict ::preprocessing_switches::
        """
        if preprocessing_switches["convert_usernames"]:
            text = re.sub("@[a-zA-Z0-9:.]+", "@username", text)
        if preprocessing_switches["convert_number_words_to_digits"]:
            text = self.text2int(text)
        tokens = self.tokenize_text(text, preprocessing_switches)
        if preprocessing_switches["remove_punctuation"]:
            tokens = self.remove_characters_after_tokenization(tokens)
        if preprocessing_switches["convert_to_lowercase"]:
            tokens = self.convert_to_lowercase(tokens)
        if preprocessing_switches["remove_stopwords"]:
            tokens = self.remove_stopwords(tokens)
        if preprocessing_switches["apply_lemmatization"]:
            tokens = self.apply_lemmatization(tokens)

        return " ".join(tokens)

    def load_data(self,):
        docs = pickle.load(open(self.cache_doc_file, 'rb'))
        queries = pickle.load(open(self.cache_query_file, 'rb'))
        print(f"Preprocessed data loaded from : {self.cache_dir}")
        return docs, queries

    def save_data(self, docs, queries):
        if not self.cache_dir.exists(): self.cache_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(docs, open(self.cache_doc_file, 'wb'))
        pickle.dump(queries, open(self.cache_query_file, 'wb'))
        print(f"Preprocessed data saved at : {self.cache_dir}")

    def pre_process(self, docs, queries, model_type, use_cache=False):

        if not use_cache and self.cache_dir.exists():
            delete_dir(dir_path=self.cache_dir)

        if use_cache and self.cache_dir.exists():
            return self.load_data()

        params = config_preprocessing.config_switches(model_type)

        print(f"Cache dir for Pre-processing : {self.cache_dir}")
        for doc_id, data in tqdm(docs.items(), desc=f"Pre-Processing Docs"):
            data["document"] = self.switches(text=data["document"], preprocessing_switches=params, model_type=model_type)

        for query_id, data in tqdm(queries.items(), desc=f"Pre-Processing Queries"):
            data["query"] = self.switches(text=data["query"], preprocessing_switches=params, model_type=model_type)

        self.save_data(docs=docs, queries=queries)

        return docs, queries

    def pre_process_raw(self, raw_texts, model_type):
        params = config_preprocessing.config_switches(model_type)
        return [
            self.switches(text=text, preprocessing_switches=params, model_type=model_type)
            for text in raw_texts
        ]

    def pre_process_query(self, queries, model_type):

        params = config_preprocessing.config_switches(model_type)

        for query_id, data in tqdm(queries.items(), desc=f"Pre-Processing Queries"):
            data["query"] = self.switches(text=data["query"], preprocessing_switches=params, model_type=model_type)

        return queries
