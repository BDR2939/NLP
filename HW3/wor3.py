

import torch
import torch.nn as nn
import os 
import numpy as np
from random import shuffle
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
############################################################################################################################
main_folder = r'C:\MSC\NLP2\HW3'
train_path = os.path.join(main_folder, 'connl03_train.txt')
test_path = os.path.join(main_folder, 'connl03_test.txt')
dev_path = os.path.join(main_folder, 'connl03_dev.txt')
############################################################################################################################
def read_data(filepath):
    data = []
    with open(filepath) as file:
        words = []
        labels = []

        for index, line in enumerate(file, start=1):
            if line != '\n':
                word, label = line.split()
                words.append(word)
                labels.append(label)
            else:
                data.append((words, labels))
                words = []
                labels = []
    
    return data

train = read_data(train_path)
dev = read_data(test_path)
test = read_data(dev_path)

############################################################################################################################

"""
The following Vocab class can be served as a dictionary that maps words and tags into Ids. 
The UNK_TOKEN should be used for words that are not part of the training data.
"""

UNK_TOKEN = 0


class Vocab:
    def __init__(self):
        """
        tag2id/id2tag  - tags to each other from label to integer number
        n_words - count the # of word in sentence
        """
        self.word2id = {"__unk__": UNK_TOKEN}
        self.id2word = {UNK_TOKEN: "__unk__"}
        self.n_words = 1
        
        self.tag2id = {"O":0, "B-PER":1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}
        self.id2tag = {0:"O", 1:"B-PER", 2:"I-PER", 3:"B-LOC", 4:"I-LOC", 5:"B-ORG", 6:"I-ORG"}
    
    
    def index_words(self, words):
        """
        for given token list get token index in sentence
        """
        word_indexes = [self.index_word(w) for w in words]
        return word_indexes


    def index_tags(self, tags):
        """
        for given label list get label index
        """
        tag_indexes = [self.tag2id[t] for t in tags]
        return tag_indexes
    

    def index_word(self, w):
        """
     
        """
        if w not in self.word2id:
            self.word2id[w] = self.n_words
            self.id2word[self.n_words] = w
            self.n_words += 1
        
        return self.word2id[w]
    
############################################################################################################################
"""
**Task 2:** Write a function prepare_data that takes one of the [train, dev, test] and the Vocab instance, 
for converting each pair of (words,tags) to a pair of indexes. Each pair should be added to data_sequences, 
which will be returned back from the function.
"""
vocab = Vocab()

def prepare_data(data, vocab):
    data_sequences = []
    # TODO - your code...
    """
    this loop run on the data, for each sequence we generating tesor to
    contain the token of sequence
    """
    for i_words, i_tags in data:
        
        words_indexes_tensor = torch.tensor(vocab.index_words(i_words), dtype=torch.long)
        tags_indexes_tensor = torch.tensor(vocab.index_tags(i_tags), dtype=torch.long)
        # append data and label tensors
        data_sequences.append((words_indexes_tensor, tags_indexes_tensor))

    return data_sequences, vocab

train_sequences, vocab = prepare_data(train, vocab)
dev_sequences, vocab = prepare_data(dev, vocab)
test_sequences, vocab = prepare_data(test, vocab)

############################################################################################################################
 
"""
**Task 3:
** Write NERNet, a PyTorch Module for labeling words with NER tags. 

*input_size:* the size of the vocabulary

*embedding_size:* the size of the embeddings

*hidden_size:* the LSTM hidden size

*output_size:* the number tags we are predicting for

*n_layers:* the number of layers we want to use in LSTM

*directions:* could 1 or 2, indicating unidirectional or bidirectional LSTM, respectively

The input for your forward function should be a single sentence tensor.

*note:* the embeddings in this section are learned embedding. That means that you don't need to use pretrained embedding like the one used in class. You will use them in part 5

"""

class NERNet(nn.Module):
    
    def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers, directions):
        super(NERNet, self).__init__()
        # TODO: your code...
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, bidirectional=(True if directions==2 else False))
        self.out = nn.Linear(hidden_size*directions, output_size)

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.directions = directions


    def forward(self, input_sentence):
        # TODO: your code...
        
        # get sentence token numbers to understand output & input size
        dimension = len(input_sentence)
        
        # initial the hidden to None because none sentence inter
        hidden = None

        # 1. foward input sentence into the embeding
        embedded = self.embedding(input_sentence)

        # 2. foward embedding to LSTM
        lstm_output, _ = self.lstm(embedded.view(dimension, 1, -1), hidden) # The view function is meant to reshape the tensor https://stackoverflow.com/a/48650355/7786691

        # 3. foward to get predictions  - linear transformation to the incoming data
        output = self.out(lstm_output.view(dimension, -1)) 

        return output


"""
**Task 4:** 
write a training loop, which takes a model (instance of NERNet) and number of epochs to train on.
The loss is always CrossEntropyLoss and the optimizer is always Adam.
"""
def get_model_results(model, test_sequences):
    """
    

    Parameters
    ----------
    model : Torch model  - 
        DESCRIPTION: LSTM model.
    test_sequences : list
        DESCRIPTION: input list of coupels [[word_tensor, lebel_tensor] , ...]
    
    the function get model results
    
    Returns
    -------
    all_test_words_pred : list
    all_test_words_true : list
    binary_test_words_pred : list
    binary_test_words_true : list
    """
    # generate test tokens prediction
    all_test_words_pred = []
    all_test_words_true = []

    # generate test binnary prediction
    binary_test_words_pred = []
    binary_test_words_true = []
    for sentence, labels in test_sequences:
        sentence_tensor = torch.LongTensor(sentence).cuda()
        labels_tensor = torch.LongTensor(labels).cuda()

        _, pred_labels = model(sentence_tensor).T.max(0)

        all_test_words_pred += pred_labels.tolist()
        all_test_words_true += labels.tolist()

        binary_test_words_pred += [1 if i >= 1 else i for i in all_test_words_pred]
        binary_test_words_true += [1 if i >= 1 else i for i in all_test_words_true]
    return all_test_words_pred, all_test_words_true, binary_test_words_pred, binary_test_words_true



def train_loop(model, n_epochs, train_sequences):
    #
    all_target_names = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
    binary_target_names = ["O", "OTHERS"]
    
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (ADAM is a fancy version of SGD)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  
    # shuffle data before training phase
    shuffle(train_sequences)
    STEP = 400 
    
    for e in range(1, n_epochs + 1):
        print('start ephoc #' + str(e))
        # TODO - your code goes here...
        """
        tqdm - add progress bar
        """
        for sequence_idx, sequence in tqdm(enumerate(train_sequences)):
            # get sentence tokens, and labels 
            sentence, labels = sequence
            
            # check if there is empty sentence
            if labels.__len__() == 0:
                continue
            
            # insert sentence tokens into tensor
            sentence_tensor = torch.LongTensor(sentence).cuda()
            
            # insert sentence labels into tensor
            labels_tensor = torch.LongTensor(labels).cuda()
            
            # Sets the gradients of all optimized to zero.
            model.zero_grad()
            
            # foward sentence to model
            scores = model(sentence_tensor)
            
            # Computes the gradient of current tensor
            criterion(scores, labels_tensor).backward()
            
            # once the gradients are computed use them to optimize model
            optimizer.step()
            print('start ephoc #' + str(e))
        
        print('finshed ephoc #' + str(e) + ', ephoch results:' )
        all_train_words_pred, all_train_words_true, \
        binary_train_words_pred, binary_train_words_true = get_model_results(model, train_sequences)
        train_Results_dict = pd.DataFrame(classification_report(all_train_words_true, all_train_words_pred, target_names=all_target_names))
        print(train_Results_dict)

            
            
"""
**Task 5:** 
write an evaluation loop on a trained model, using the dev and test datasets. 
This function print the true positive rate (TPR), also known as Recall and the opposite to false positive rate (FPR), 
also known as precision, of each label seperately (7 labels in total), and for all the 6 labels (except O) together.
The caption argument for the function should be served for printing, so that when you print include it as a prefix.
"""


def evaluate(model, caption, test_sequences, dev_sequences):
    # TODO - your code goes here
    # from Piazza: https://piazza.com/class/klxc3m1tzqz2o8?cid=59

    all_target_names = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
    binary_target_names = ["O", "OTHERS"]
    
    # self.tag2id = {"O":0, "B-PER":1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}
    # self.id2tag = {0:"O", 1:"B-PER", 2:"I-PER", 3:"B-LOC", 4:"I-LOC", 5:"B-ORG", 6:"I-ORG"}
    
    print(f"****************    Results for {caption}    ****************")



    # generate dev tokens prediction 
    all_dev_words_pred = []
    all_dev_words_true = []
    
    # generate dev binnary prediction 
    binary_dev_words_pred = []
    binary_dev_words_true = []

    # get test results
    all_test_words_pred, all_test_words_true, \
        binary_test_words_pred, binary_test_words_true = get_model_results(model, test_sequences)

    # get dev results
    all_dev_words_pred, all_dev_words_true, \
        binary_dev_words_pred, binary_dev_words_true = get_model_results(model, dev_sequences)

    print("Test Results:")
    Test_Results_dict = pd.DataFrame(classification_report(all_test_words_true, all_test_words_pred, target_names=all_target_names))
    print(Test_Results_dict)
    print("Dev Results:")
    Dev_Results_dict = pd.DataFrame(classification_report(all_dev_words_true, all_dev_words_pred, target_names=all_target_names))
    print(Dev_Results_dict)

    print("Binary Test Results:")
    Binary_Test_Results = pd.DataFrame(classification_report(binary_test_words_true, binary_test_words_pred, target_names=binary_target_names))
    print(Binary_Test_Results)

    print("Binary Dev Results:")
    Binary_Dev_Results  = pd.DataFrame(classification_report(binary_dev_words_true, binary_dev_words_pred, target_names=binary_target_names))
    print(Binary_Dev_Results)

    return 



"""
**Task 6:**

Train and evaluate a few models, all with embedding_size=300, 
and with the following hyper parameters (you may use that as captions for the models as well):

Model 1: (hidden_size: 500, n_layers: 1, directions: 1)

Model 2: (hidden_size: 500, n_layers: 2, directions: 1)

Model 3: (hidden_size: 500, n_layers: 3, directions: 1)

Model 4: (hidden_size: 500, n_layers: 1, directions: 2)

Model 5: (hidden_size: 500, n_layers: 2, directions: 2)

Model 6: (hidden_size: 500, n_layers: 3, directions: 2)

Model 7: (hidden_size: 800, n_layers: 1, directions: 2)

Model 8: (hidden_size: 800, n_layers: 2, directions: 2)

Model 9: (hidden_size: 800, n_layers: 3, directions: 2)

"""


# TODO - your code goes here...
EMBEDDING_SIZE = 300
EPOCHS = 10
HIDDEN_SIZE  = 500 
INPUT_SIZE = len(vocab.word2id) # 8955
OUTPUT_SIZE = len(vocab.tag2id) # 7


n_layers_array = np.arange(1,4)
directions_array = np.arange(1,3)
model_list  = []
for i_n_layers in n_layers_array:
    for i_directions in directions_array:
        print('Train model using:\n'+ \
              '1)hidden_size = ' + str(HIDDEN_SIZE)+'\n'+ \
              '2)n_layers = ' + str(i_n_layers) + '\n'+ \
              '3)directions = ' + str(i_directions))
        model = NERNet(INPUT_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, i_n_layers, i_directions).cuda()
        train_loop(model, EPOCHS)
        model_list.append(model)

DIRECTION = 2
HIDDEN_SIZE = 800
for i_n_layers in n_layers_array:
        print('Train model using:\n'+ \
              '1)hidden_size = ' + str(HIDDEN_SIZE)+'\n'+ \
              '2)n_layers = ' + str(i_n_layers) + '\n'+ \
              '3)directions = ' + str(i_directions))
        model = NERNet(INPUT_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, i_n_layers, DIRECTION).cuda()
        train_loop(model, EPOCHS)
        model_list.append(model)
        
  
    
for i, model in enumerate(model_list):
    model_name = "model_"+str(i)
    evaluate(model, model_name, test_sequences, dev_sequences)
    
    
        
"""
**Task 6:** 
Download the GloVe embeddings from https://nlp.stanford.edu/projects/glove/ (use the 300-dim vectors from glove.6B.zip). 
Then intialize the nn.Embedding module in your NERNet with these embeddings, 
so that you can start your training with pre-trained vectors.
Repeat Task 6 and print the results for each model.
Note: make sure that vectors are aligned with the IDs in your Vocab, in other words, make sure that for example the word with ID 0 is the first vector in the GloVe matrix of vectors that you initialize nn.Embedding with. For a dicussion on how to do that, check it this link:
https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222
"""


