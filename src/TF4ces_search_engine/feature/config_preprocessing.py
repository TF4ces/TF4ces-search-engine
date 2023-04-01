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

    return preprocessing_switches
