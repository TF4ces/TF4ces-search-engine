#!/usr/bin/env python3
# encoding: utf-8

"""
    File Handlers.

    Author : TF4ces
"""

from src.TF4ces_search_engine.feature import config_preprocessing
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
import re
import string
from nltk.stem import WordNetLemmatizer

class Preprocessing():
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

        if model_type == "tfidf":
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

        if model_type == "bm25":
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

            return tokens

    def pre_process(self, docs, queries, model_type):

        preprocessing_switches = config_preprocessing.config_switches(model_type)

        for key in docs:
            for inner_key in docs[key]:
                if (inner_key == "document"):
                    docs[key][inner_key] = self.switches(docs[key][inner_key], preprocessing_switches, model_type)

        for key in queries:
            for inner_key in queries[key]:
                if (inner_key == "query"):
                    queries[key][inner_key] = self.switches(queries[key][inner_key], preprocessing_switches, model_type)


        # for i in range (len(queries)):
        #     pre_preocessed_queries.append(switches(queries[i],preprocessing_switches))
        #
        # for i in range (len(docs)):
        #     pre_processed_docs.append(switches(docs[i],preprocessing_switches))

        return docs, queries
