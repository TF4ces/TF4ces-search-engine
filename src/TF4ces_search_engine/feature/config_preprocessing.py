#!/usr/bin/env python3
# encoding: utf-8

"""
    Main file for search engine

    Author : TF4ces
"""


from config.conf import __SENTENCE_TRANSFORMERS_MODELS__



def config_switches(model):

    preprocessing_switches = {}

    #Todo try different switches.

    if model=="tfidf":
        preprocessing_switches = {
                "convert_usernames" : False,                  #Redundant
                "separate_out_punctuation" : True,
                "convert_number_words_to_digits": False,      #Increasing the rank
                "convert_numbers" : False,                    #Redundant
                "remove_punctuation" : True,
                "convert_to_lowercase" : True,
                "remove_stopwords" : True,
                "apply_lemmatization" : True
        }

    if model=="bm25":
        preprocessing_switches = {
                "convert_usernames" : False,                  #Redundant
                "separate_out_punctuation" : True,
                "convert_number_words_to_digits": False,      #Increasing the rank
                "convert_numbers" : False,                    #Redundant
                "remove_punctuation" : True,
                "convert_to_lowercase" : True,
                "remove_stopwords" : True,
                "apply_lemmatization" : True
        }

    if model in __SENTENCE_TRANSFORMERS_MODELS__:
        preprocessing_switches = {
            "convert_usernames": False,  # Redundant
            "separate_out_punctuation": False,
            "convert_number_words_to_digits": False,
            "convert_numbers": False,  # Redundant
            "remove_punctuation": False,
            "convert_to_lowercase": False,
            "remove_stopwords": False,
            "apply_lemmatization": False
        }

    return preprocessing_switches
