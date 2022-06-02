# -*- coding: utf-8 -*-
"""Copy of Assignment 4 - RC

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JV8MzHXhKRZlIX_KWqlCRUWbiNW8fe6q

# Assignment 4
Training a simple neural net for relation classification.
"""

# -------------------------------------------------------------------------------
# Imports 



import torch
import torch.nn as nn
import tensorflow as tf



import numpy as np
import math
from tabulate import tabulate

from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AutoConfig, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


# -------------------------------------------------------------------------------


"""In this assignment you are required to build a full training and testing pipeline for a neural relation classification (RC), using BERT.

The dataset that you will be working on is called SemEval Task 8 dataset (https://arxiv.org/pdf/1911.10422v1.pdf). The dataset contain only train and test split, but you are allowed to split the train dataset into dev if needed.


The two files (train and test) are available from the course git repository (https://github.com/kfirbar/nlp-course)


In this work we will use the hugingface framework for transformers training and inference. We recomand reading the documentation in https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification *before* you start coding.

**Task 1:** Write a funtion *read_data* for reading the data from a single file (either train or test). This function recieves a filepath and returns a list of sentence. Every sentence is encoded as a touple, where the first element is the sentence string and the second the label (also represented as a sting).
"""

User = 'Or'

if User == 'Or':
    train_path = r'C:\MSC\NLP2\HW4\TRAIN_FILE.TXT'
    test_path = r'C:\MSC\NLP2\HW4\TEST_FILE_FULL.txt'
else:
    train_path = r'/content/nlp-course/TRAIN_FILE.TXT'
    test_path = r'/content/nlp-course/TEST_FILE_FULL.txt'

def read_data(filepath):
    """
    Parameters
    ----------
    filepath : string
    DESCRIPTION:
        1. for given path we read the txt file into list 
        2. for each sentence is shift of 4 index 0,4,8,....,4n, the label
           is the same only from the index 1,5,9,...4n+1
        3. first we will slice the sentences, and labels 
        4. for each setnence we will remodve the TAB & "\n" 
            ([1:-1] to remove the double quating ) 
        
          for each labels needed to remove  space and "\n"
        5. finnaly we need to concat setence to label to list of tupples 
    Returns
    -------
    data : list 
        list of tupples [ (sentence, label), (), ....].
    """
    # 1
    with open(filepath) as file:
        output = file.readlines()
    
    # 2
    STEP = 4
    
    # 3
    labels  = output[1::STEP]
    sentences  = output[0::STEP]

    #4 
    # sentence splited
    sentences = list(map(lambda x: x.split('\t')[1].replace('\n', '').replace('</e2>', '').replace('</e1>', '').replace('<e1>', '').replace('<e2>', '')[1:-1], sentences))
    # sentences = list(map(lambda x: x.replace('</e2>', ''), sentences))
    # sentences = list(map(lambda x: x.replace('</e1>', ''), sentences))
    # sentences = list(map(lambda x: x.replace('<e1>', ''), sentences))
    # sentences = list(map(lambda x: x.replace('<e2>', ''), sentences))

    labels = list(map(lambda x: x.replace('\n', '').replace('(e2,e1)', '').replace('(e1,e2)', ''), labels))
    # labels = list(map(lambda x: x.replace('(e1,e2)', ''), labels))
    # labels = list(map(lambda x: x.replace('(e2,e1)', ''), labels))

    # 5
    data = list(zip(sentences, labels))
    
    return data

train = read_data(train_path)
test = read_data(test_path)

"""Pytorch require the labels to be integers. Create a mapper (dictionary) from the string labels to integers (starting zero). """
def from_list_of_list_2_single_list(list_of_list):
    # convertign list of list to single list with all ellements
    flat_list = [item for sublist in list_of_list for item in sublist]
    return flat_list

def get_unique_tokens_and_token_indexs(data, is_sentence = True):
    """
    Parameters
    ----------
    data : list of tuples 
        contain list of tuples (sentences, labels)
    is_sentence : Bollan
        indicate whether to get setnece or label
    
    
    1. get desire list sentences\labels 
    2. split into tokens
    3. convert to single list with all ellements
    4. doing unique the list to get unique value and index 
    5. generate dict\mapper from token 2 index
    
    Returns
    -------
    mapper_dict : dict
        map from unique token to unique index

    """
    #1
    if is_sentence:
        list_of_list = list(map(lambda x: x[0], data))

    # label
    else: 
        list_of_list = list(map(lambda x: x[1], data))
    
    #2
    list_of_list_tokens = list(map(lambda x: x.split(' '), list_of_list))
    #3
    all_tokens = from_list_of_list_2_single_list(list_of_list_tokens)
    #4
    unique_tokens =list(set(all_tokens))
    #5
    mapper_dict = dict(zip(unique_tokens,np.arange(0, unique_tokens.__len__())))
    return mapper_dict

def create_label_mapper(data):
    """
    Parameters
    ----------
    data : list of tuples 
        contain list of tuples (sentences, labels)
    Returns
    -------
    sentences_mapper : dict
        map from unique token to unique index
    labels_mapper : dict
        map from unique token to unique index
    """ 
    #sentences_mapper = get_unique_tokens_and_token_indexs(data, is_sentence = True)
    labels_mapper = get_unique_tokens_and_token_indexs(data, is_sentence = False)
    return labels_mapper

labels_mapper = create_label_mapper(train)

"""**Task 2:** Write a function *prepare_data* that takes one of the [train, test] datasets and convert each pair of (words,labels) to a pair of indexes. The function also aggregate the samples into batches. BERT Uses pretrained tokanization and embedding. you can access the tokanization and indexing using the BertTokenizer class."""
from numpy.random import default_rng



def prepare_data(data, tokenizer, batch_size=8):
    """
    

    Parameters
    ----------
    data : list of tuples 
        contain list of tuples (sentences, labels)
    tokenizer : generator of transformers pakage
        for given sentence return dict of:
            1. input idx - tensor
            2. mask  - tensor 
            3. attention_mask - tensor
            
    batch_size : int, optional
        DESCRIPTION. The default is 8, the size of batch, after 8 input model will do backprop'.

    Returns
    -------
    data_sequences : list of tuples 
        the input to our model

    """
    data_sequences = []
    sentence_tuple_idx = 0
    label_tuple_idx = 1

    labels_mapper = create_label_mapper(data)
    for sentence_idx, i_data in enumerate(data):
        
        # initiate batch input list
        if sentence_idx%batch_size == 0 :
            batch_data = [] 
            
        # 
        single_input = get_single_input(i_data, label_tuple_idx, labels_mapper, sentence_tuple_idx, tokenizer)

        # append batch data
        batch_data.append(single_input)
        
        # if batch list is fill append batch data to all data 
        if sentence_idx%(batch_size-1):
            data_sequences.append(batch_data)
    
    padd_last_batch_size =  8 - batch_data.__len__() 
    if padd_last_batch_size>0:
        rng = default_rng()
        choosen_idx = \
            rng.choice(range(0, data.__len__()-padd_last_batch_size), size=padd_last_batch_size, replace=False)
        for i_idx in choosen_idx:
            
            i_data = data[i_idx]
            
            # get single input
            single_input = get_single_input(i_data, label_tuple_idx, labels_mapper, sentence_tuple_idx, tokenizer)
            
            # append batch data
            batch_data.append(single_input)
            
    return data_sequences


def get_single_input(i_data, label_tuple_idx, labels_mapper, sentence_tuple_idx, tokenizer):
    i_sentence = i_data[sentence_tuple_idx]
    i_label = i_data[label_tuple_idx]
    # do tokenization to the setnence
    i_tensor_input = tokenizer(i_sentence, padding=True, truncation=True, return_tensors="pt")
    # get input & mask tesnsors
    input_ids, attn_mask = i_tensor_input['input_ids'], i_tensor_input['attention_mask']
    # get label
    i_label = labels_mapper[i_label]
    # insert all to tupple
    single_input = (input_ids, attn_mask, i_label)
    return single_input


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_sequences = prepare_data(train, tokenizer)
test_sequences = prepare_data(test, tokenizer)

# """**Task 3:** In this part we classify the sentences using the BertForSequenceClassification model. To save resources, we initialize the optimizer with the final layer of the model. You are also allowed to change the learning rate."""

# def get_parameters(params):
#   # TODO - your code...
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# # Optimizer (ADAM is a fancy version of SGD)
# optimizer = torch.optim.Adam(get_parameters(model.named_parameters()), lr=0.0001)

# """**Task 4:** Write a training loop, which takes a BertForSequenceClassification model and number of epochs to train on. The loss is always CrossEntropyLoss and the optimizer is always Adam. You are allowed to split the train to train and dev sets."""

# from transformers import BertForSequenceClassification

# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# def train_loop(model, n_epochs, train_data, dev_data):
#   # Loss function
#   criterion = nn.CrossEntropyLoss()
 
#   for e in range(1, n_epochs + 1):
#     # TODO - your code goes here...

# """**Task 5:** write an evaluation loop on a trained model, using the dev and test datasets. This function print the true positive rate (TPR), also known as Recall and the opposite to false positive rate (FPR), also known as precision, of each label seperately (10 labels in total), and for all labels together. The caption argument for the function should be served for printing, so that when you print include it as a prefix."""

# def evaluate(model, test_data):
#   # TODO - your code goes here
#   print(...)

# """**Task 6:** In this part we'll improve the model accuracy by using a method called "entity markers - Entity start". The main idea of this approch is to add special markers ([e1], [\e1], ...) before and after each of the tagged entities. instead of using the CLS toekn for clasification, we will use the concatination of the embedding of [e1] and [e2] as shown in the image below. The complete method is described in details in the following paper - https://arxiv.org/pdf/1906.03158.pdf (specifically in Section 3.2). To use this method we'll need to create a new data load and a new model.

# ![Capture.PNG](data:image/png;base64,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)

# The new methods require the usage of special tokens. The following code will add the required tokens.
# """

# tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]')

# """Create a new dataloader that add entity markers to the dataset and return their indexes as part of the new sample (the expected sample should be (s, l, i) where s is the sentence embedding, l is the label, and i is a touple with the indexes of the start entities)"""

# def prepare_data_MTB(data, tokenizer, batch_size=8):
#     data_sequences = []
#     # TODO - your code...

#     return data_sequences

# train_sequences = prepare_data_MTB(train, tokenizer)
# test_sequences = prepare_data_MTB(test, tokenizer)

# """Create a new model that uses the "entity markers - Entity start" method."""

# class MTB(nn.Module):
#     def __init__(self, base_model_name):
#       config = AutoConfig.from_pretrained(name)
#       self.model = AutoModel.from_config(config)
#       # TODO - your code...
#     def forward(self, input, index):
#       # TODO - your code...
# model = MTB('bert-base-uncased')

# """Use the new dataloader and model to train and evaluate the new model as in task 4 and 5"""

# train_loop(model, n_epochs, train_data, dev_data)
# evaluate(model, test_data)

# """**Good luck!**"""