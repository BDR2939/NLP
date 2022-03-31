"""
*As a preparation for this task, download the data files from the course git repository.

The relevant files are under **lm-languages-data-new**:


*   en.csv (or the equivalent JSON file)
*   es.csv (or the equivalent JSON file)
*   fr.csv (or the equivalent JSON file)
*   in.csv (or the equivalent JSON file)
*   it.csv (or the equivalent JSON file)
*   nl.csv (or the equivalent JSON file)
*   pt.csv (or the equivalent JSON file)
*   tl.csv (or the equivalent JSON file)
*   test.csv (or the equivalent JSON file)

"""
import os 
import pandas as pd 
import os
import io
import sys
import pandas as pd
import numpy as np
import math
import joblib
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
from functools import partial
import threading # you can use easier threading packages
from collections import Counter
from IPython.display import display

def reorder_list(List, index_list):
    return [List[i] for i in index_list]



def string_split(word):
  return list(word)

def remove_duplicate_list(list1):
    return list(set(list1))

def list_of_list_2_single_list(list1):
    return [item for sublist in list1 for item in sublist]

global languages_list, columns_list, data_files, start_token, end_token, vocabulary
global test_files
data_files = {'en_df': 'nlp-course/lm-languages-data-new/en.csv',
              'es_df': 'nlp-course/lm-languages-data-new/es.csv',
              'fr_df': 'nlp-course/lm-languages-data-new/fr.csv',
              'in_df': 'nlp-course/lm-languages-data-new/in.csv',
              'it_df': 'nlp-course/lm-languages-data-new/it.csv',
              'nl_df': 'nlp-course/lm-languages-data-new/nl.csv',
              'pt_df': 'nlp-course/lm-languages-data-new/pt.csv',
              'tl_df': 'nlp-course/lm-languages-data-new/tl.csv'}



data_files = {'en_df': r'C:\MSC\NLP2\nlp-course\lm-languages-data\en.csv',
              'es_df': r'C:\MSC\NLP2\nlp-course\lm-languages-data\es.csv',
              'fr_df': r'C:\MSC\NLP2\nlp-course\lm-languages-data\fr.csv',
              'in_df': r'C:\MSC\NLP2\nlp-course\lm-languages-data\in.csv',
              'it_df': r'C:\MSC\NLP2\nlp-course\lm-languages-data\it.csv',
              'nl_df': r'C:\MSC\NLP2\nlp-course\lm-languages-data\nl.csv',
              'pt_df': r'C:\MSC\NLP2\nlp-course\lm-languages-data\pt.csv',
              'tl_df': r'C:\MSC\NLP2\nlp-course\lm-languages-data\tl.csv'}

# data_files = {'en_df': r'C:\MSC\NLP2\nlp-course\lm-languages-data\en.csv',
#               'es_df': r'C:\MSC\NLP2\nlp-course\lm-languages-data\es.csv'}

import glob
test_folder = r'C:\MSC\NLP2\nlp-course\lm-languages-data-new'
test_csv_files =  glob.glob(test_folder + '\\*.csv')
test_files =  {}
for i_file in test_csv_files:
    file_name_with_ending = os.path.basename(test_csv_files[0])
    file_name = os.path.splitext(file_name_with_ending)[0]
    test_files[file_name] = [i_file]
languages_list = list(data_files.keys())
# languages_list = languages_list[0:2]
start_token = '↠'
end_token = '↞'
"""
**Part 1**

Write a function *preprocess* that iterates over all the data files and creates a single vocabulary
 containing all the tokens in the data. **Our token definition is a single UTF-8 encoded character**. 
So, the vocabulary list is a simple Python list of all the characters that you see at least once in the data.
"""
                
                                              
def convert_data_frame_2_token_list(index, value):
    current_df = pd.read_csv(value, encoding='utf8')
    current_df = current_df[0:100]

    columns_list = current_df.columns.to_list()
    current_df[columns_list[-1]] = current_df[columns_list[-1]].apply(lambda x: remove_duplicate_list(string_split(x)))
    current_df_tokens = current_df[columns_list[-1]].to_numpy().tolist()
    current_df_tokens = list(set(list_of_list_2_single_list(current_df_tokens))) 
    del current_df
    return current_df_tokens                                          
def preprocess(data_files):
    """
    data frame is table from 2 columns:
        1. tweet id
        2. tweet text
    """  
    tokens = []
    for path in data_files.values():
        df = pd.read_csv(path)
        if tokens.__len__() == 0 :
            columns_list = df.columns.to_list()
        for text in df[columns_list[-1]].values:
            tokens.extend(list(text))
    return list(set(tokens))



import time
# s = time.time()
# vocabulary = preprocess(data_files)
# t = (time.time() - s)

# a=5


"""
**Part 2**

Write a function lm that generates a language model from a textual corpus. The function should return a dictionary (representing a model) where the keys are all the relevant n-1 sequences, and the values are dictionaries with the n_th tokens and their corresponding probabilities to occur. For example, for a trigram model (tokens are characters), it should look something like:

{
  "ab":{"c":0.5, "b":0.25, "d":0.25},
  "ca":{"a":0.2, "b":0.7, "d":0.1}
}

which means for example that after the sequence "ab", 
there is a 0.5 chance that "c" will appear, 0.25 for "b"
 to appear and 0.25 for "d" to appear.

Note - You should think how to add the add_one smoothing 
information to the dictionary and implement it.
"""

#helper function
#helper function
def tweets_to_text(data_file_path, n):
    """
    data frame is table from 2 columns:
        1. tweet id
        2. tweet text
    """
    df = pd.read_csv(r''+ data_file_path)
    debug = True
    if debug == True:
        df = df[0:100]
    columns_list = df.columns.to_list()
    tweets_list = df[columns_list[-1]].apply(lambda x: start_token + x + end_token).values
    text = ''.join(tweets_list)
    
    text = start_token * (n-1) + text + end_token * (n-1)

    return text
def lm(n, vocabulary, data_file_path, add_one):
    # n - the n-gram to use (e.g., 1 - unigram, 2 - bigram, etc.)
    # vocabulary - the vocabulary list (which you should use for calculating add_one smoothing)
    # data_file_path - the data_file from which we record probabilities for our model
    # add_one - True/False (use add_one smoothing or not)
  
    lm_dict = {}
    V = len(vocabulary)

    text = tweets_to_text(data_file_path, n)

    # Extract n - 1 length substrings
    n_1_gram = [text[i: i + n-1] for i in range(len(text) - (n-1))]
    counter_obj_n_1_gram = dict(Counter(n_1_gram))

    # Extract n length substrings
    n_gram = [text[i: i + n] for i in range(len(text) - n)]
    counter_obj_n_gram = dict(Counter(n_gram))

    for key in counter_obj_n_1_gram.keys():
        inner_dict = {}
        if add_one:
            gen = (key_1 for key_1 in counter_obj_n_gram.keys() if key_1[0:n-1] == key)
            for key_1 in gen:
                val = (int(counter_obj_n_gram[key_1]) + 1) / (int(counter_obj_n_1_gram[key]) + V)
                inner_dict[key_1[-1]] = val

            gen = (token for token in vocabulary if not(token in inner_dict))
            for key_1 in gen:
                val = 1 /  (int(counter_obj_n_1_gram[key]) + V)
                inner_dict[key_1[-1]] = val

        else:
            gen = (key_1 for key_1 in counter_obj_n_gram.keys() if key_1[0:n-1] == key)
            sum_vals = 0
            for key_1 in gen:
                val = int(counter_obj_n_gram[key_1]) / int(counter_obj_n_1_gram[key])
                inner_dict[key_1[-1]] = val
                sum_vals += val

        lm_dict[key] = inner_dict.copy()

    return lm_dict
# n = 2
# test_dict = lm(n, vocabulary, data_files['en_df'], False)
# a=5
"""
**Part 3**

Write a function *eval* that returns the perplexity of a model (dictionary) running over a given data file.
"""
def eval(n, model, data_file):
    # n - the n-gram that you used to build your model (must be the same number)
    # model - the dictionary (model) to use for calculating perplexity
    # data_file - the tweets file that you wish to claculate a perplexity score for

    # read file
    if os.path.exists(data_file):
        text = tweets_to_text(data_file, n)
    else:
        text = data_file
    # Extract n length substrings
    n_gram = [text[i: i + n] for i in range(len(text) - n)]

    model_keys = model.keys()
    entropy = 0 
    for i_letter in n_gram:
        if i_letter[0:n-1] in model_keys: 
            i_letter_model = model[i_letter[0:n-1]]
            if i_letter[n-1] in i_letter_model.keys():
                second_letter_prob = i_letter_model[i_letter[n-1]]
                entropy += -np.log2(second_letter_prob)
            else:
                entropy += 0
        else:
            entropy += 0
    entropy = entropy/len(n_gram)
    perplexity_score = 2**(entropy)
    return perplexity_score



"""
**Part 4**

Write a function *match* that creates a model for every relevant language, using a specific 
value of *n* and *add_one*. Then, calculate the perplexity of all possible pairs
(e.g., en model applied on the data files en ,es, fr, in, it, nl, pt, tl; es model applied on the data files en, es...).
This function should return a pandas DataFrame with columns [en ,es, fr, in, it, nl, pt, tl] 
and every row should be labeled with one of the languages. Then, the values are the relevant perplexity values.
"""  

def match(n, add_one, data_files):
    # n - the n-gram to use for creating n-gram models
    # add_one - use add_one smoothing or not
    result_dict = {}
    for i_language_model in languages_list:
        
        i_model = lm(n, vocabulary, data_files[i_language_model], add_one)
        result_dict[i_language_model] = {}

        for i_language_test in languages_list:
            i_language_model_i_score = eval(n, i_model, data_files[i_language_test])
            result_dict[i_language_model][i_language_test] = i_language_model_i_score
    perlexity_df = pd.DataFrame(result_dict)
    return perlexity_df      
 
def run_match(data_files):
    full_model_dict = {}
    # for n in range(2,3):

    for n in range(1,2):
        add_one = True
        perlexity_df = match(n, add_one, data_files)
        print(f'n = {n}, add_one = {add_one}')
        display(perlexity_df)

        add_one = False
        perlexity_df = match(n, add_one, data_files)
        print(f'n = {n}, add_one = {add_one}')
        display(perlexity_df)
    return 

def match_test(n, model_dict, data_file_path, add_one):
    # n - the n-gram to use for creating n-gram models
    # add_one - use add_one smoothing or not
    #data_file_path = r"C:\MSC\NLP2\nlp-course\lm-languages-data-new\test.csv"
    senstences_list = pd.read_csv(data_file_path)['tweet_text'].to_list()

    lines = [] 
    result_dict = {}

    for i_language_model in languages_list:
        # i_model = model_dict[n][add_one][i_language_model]
        result_dict[i_language_model] = {}
        i_model = lm(n, vocabulary, data_files[i_language_model], add_one)

        for i_test_senstence_idx in range(senstences_list.__len__()):
            i_test_senstence = senstences_list[i_test_senstence_idx]
            i_sentence_model_i_score = eval(n, i_model, i_test_senstence)
            result_dict[i_language_model][i_test_senstence_idx] = i_sentence_model_i_score
    # print('summary for '+ i_language_model +' model perlexity score for each language:\n')
    perlexity_df = pd.DataFrame(result_dict)
    print(perlexity_df)
    perlexity_array = perlexity_df.to_numpy()
    language_match_index = np.argmin(perlexity_array, axis=1)
    language_match_list = reorder_list(languages_list, language_match_index)
    perlexity_df['predict'] = language_match_index
    perlexity_df['predict_language'] = language_match_list
    print(perlexity_df)

    #TODO
    return perlexity_df


def classify(n, model_dict, data_file_path, add_one):
    # TODO
    match_dict  = match_test(n, model_dict, data_file_path, add_one)
    return match_dict


global vocabulary

vocabulary = preprocess(data_files)

model_dict = run_match(data_files)

data_file_path = r"C:\MSC\NLP2\nlp-course\lm-languages-data-new\test.csv"

clasification_result = classify(2, model_dict, data_file_path, False)
    
y_true = pd.read_csv(data_file_path).get('label').to_list()

y_true2 = list(map(lambda x: languages_list.index(x+'_df'),y_true))
y_pred = clasification_result['predict'].to_list()

def calc_f1(y_true,y_pred ):
    return f1_score(y_pred, y_pred,average="micro")

  # TODO

f_score_result = calc_f1(y_true,y_pred)


