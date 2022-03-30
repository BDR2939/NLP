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

def string_split(word):
  return list(word)

def remove_duplicate_list(list1):
    return list(set(list1))

def list_of_list_2_single_list(list1):
    return [item for sublist in list1 for item in sublist]


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

start_token = '↠'
end_token = '↞'
global columns_list
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
    return list(set(tokens))import time
s = time.time()
vocabulary = preprocess(data_files)
t = (time.time() - s)

a=5


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
def tweets_to_text(data_file_path, n):
    """
    data frame is table from 2 columns:
        1. tweet id
        2. tweet text
    """
    df = pd.read_csv(data_file_path)
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
    n_1_gram = [text[i: i + n-1] for i in range(len(text) - n-1)]
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
            #print(sum_vals)
            #print(sum(list(inner_dict.values())))
        lm_dict[key] = inner_dict.copy()

    return lm_dict

n = 2
test_dict = lm(n, vocabulary, data_files['en_df'], False)
a=5
"""
**Part 3**

Write a function *eval* that returns the perplexity of a model (dictionary) running over a given data file.
"""
def eval(n, model, data_file):
  # n - the n-gram that you used to build your model (must be the same number)
  # model - the dictionary (model) to use for calculating perplexity
  # data_file - the tweets file that you wish to claculate a perplexity score for
  
  # read file
  text = tweets_to_text(data_file, n)
  
  # Extract n - 1 length substrings
  # n_1_gram = [text[i: i + n-1] for i in range(len(text) - n-1)]

  # Extract n length substrings
  n_gram = [text[i: i + n] for i in range(len(text) - n)]
  
  model_keys = model.keys()
  entropy = 0 
  for i_letter in n_gram:
      if i_letter[0] in model_keys: 
          i_letter_model = model[i_letter[0]]
          if i_letter[1] in i_letter_model.keys():
              second_letter_prob = i_letter_model[i_letter[1]]
              entropy += -np.log2(second_letter_prob)
          else:
              entropy += 0
  
      else:
          entropy += 0
  entropy = entropy/n_gram.__len__()
  perplexity_score = 2**(entropy)

  # base model run on the senstence
  # 1. 
  
  
  # data_files {'file' : data_file}
  # vocabulary = preprocess(data_files)
  # # model = lm(n, vocabulary, data_file, False)
  
  # TODO
  return perplexity_score


eval(n,test_dict, data_files['en_df'])




"""
**Part 4**

Write a function *match* that creates a model for every relevant language, using a specific 
value of *n* and *add_one*. Then, calculate the perplexity of all possible pairs
(e.g., en model applied on the data files en ,es, fr, in, it, nl, pt, tl; es model applied on the data files en, es...).
This function should return a pandas DataFrame with columns [en ,es, fr, in, it, nl, pt, tl] 
and every row should be labeled with one of the languages. Then, the values are the relevant perplexity values.
"""  


def match(n, add_one):
  # n - the n-gram to use for creating n-gram models
  # add_one - use add_one smoothing or not

  #TODO
  return
  
def run_match():
  #TODO
  return 

"""
**Part 5**

Run match with *n* values 1-4, once with add_one and once without, 
and print the 8 tables to this notebook, one after another.
"""


run_match()



"""
**Part 6**

Each line in the file test.csv contains a sentence and the language it belongs to. 
Write a function that uses your language models to classify the correct language of each sentence.

Important note regarding the grading of this section: this is an open question,
 where a different solution will yield different accuracy scores. any solution that is not trivial 
 (e.g. returning 'en' in all cases) will be excepted. 
 We do reserve the right to give bonus points to exceptionally good/creative solutions.
"""
def classify():
    return 
  # TODO
clasification_result = classify()



"""
**Part 7**
Calculate the F1 score of your output from part 6. 
(hint: you can use https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). 
"""


def calc_f1(result):
    return 
  # TODO

calc_f1(clasification_result)



