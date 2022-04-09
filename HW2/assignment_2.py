# -*- coding: utf-8 -*-
"""Assignment 2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Y2s0rdmuLTUbIcx6argONkluCRfZoaei

# Assignment 2

This assignment is about training and evaluating a POS tagger with some real data. The dataset is available through the Universal Dependencies (https://universaldependencies.org/) (UD) project. To get to know the project, please visit https://universaldependencies.org/introduction.html)
"""

import numpy as np
import operator
import nltk

!pip install conllutils
import conllutils

"""**Part 1** (getting the data)

You can download the dataset files directly from the UD website, but it will let you only download all the languages in one compressed file. In this assignment you will be working with th GUM dataset, which you can download directly from:
https://github.com/UniversalDependencies/UD_English-GUM.
Please download it to your colab machine.


"""

!git clone https://github.com/UniversalDependencies/UD_English-GUM

"""We will use the (train/dev/test) files:

UD_English-GUM/en_gum-ud-train.conllu

UD_English-GUM/en_gum-ud-dev.conllu

UD_English-GUM/en_gum-ud-test.conllu

They are all formatted in the conllu format. You may read about it [here](https://universaldependencies.org/format.html). There is a utility library **conllutils**, which can help you read the data into the memory. It has already been installed and imported above.

You should write a code that reads the three datasets into memory. You may choose the data structure by yourself. As you can see, every word is represented by a line, with columns representing specific features. We are only interested in the first and fourth columns, corresponding to the word and its POS tag.
"""

# Your code goes here

"""**Part 2**

Write a class **simple_tagger**, with methods *train* and *evaluate*. The method *train* receives the data as a list of sentences, and use it for training the tagger. In this case, it should learn a simple dictionary that maps words to tags, defined as the most frequent tag for every word (in case there is more than one most frequent tag, you may select one of them randomly). The dictionary should be stored as a class member for evaluation.

The method *evaluate* receives the data as a list of sentences, and use it to evaluate the tagger performance. Specifically, you should calculate the word and sentence level accuracy.
The evaluation process is simply going word by word, querying the dictionary (created by the train method) for each word’s tag and compare it to the true tag of that word. The word-level accuracy is the number of successes divided by the number of words. For OOV (out of vocabulary, or unknown) words, the tagger should assign the most frequent tag in the entire training set (i.e., the mode). The function should return the two numbers: word level accuracy and sentence level accuracy.

"""

class simple_tagger:
  def train(self, data):
    # TODO
  
  def evaluate(self, data):
    # TODO

"""**Part 3**

Similar to part 2, write the class hmm_tagger, which implements HMM tagging. The method *train* should build the matrices A, B and Pi, from the data as discussed in class. The method *evaluate* should find the best tag sequence for every input sentence using he Viterbi decoding algorithm, and then calculate the word and sentence level accuracy using the gold-standard tags. You should implement the Viterbi algorithm in the next block and call it from your class.

Additional guidance:
1. The matrix B represents the emissions probabilities. Since B is a matrix, you should build a dictionary that maps every unique word in the corpus to a serial numeric id (starting with 0). This way columns in B represents word ids.
2. During the evaluation, you should first convert each word into it’s index and then create the observation array to be given to Viterbi, as a list of ids. OOV words should be assigned with a random tag. To make sure Viterbi works appropriately, you can simply break the sentence into multiple segments every time you see an OOV word, and decode every segment individually using Viterbi.

"""

class hmm_tagger:
  def train(self, data):
    # TODO

  def evaluate(self, data):
    # TODO

# Viterbi
def viterbi (observations, A, B, Pi):
  #...

  return best_sequence

# A simple example to run the Viterbi algorithm:
#( Same as in presentation "NLP 3 - Tagging" on slide 35)

# A = np.array([[0.3, 0.7], [0.2, 0.8]])
# B = np.array([[0.1, 0.1, 0.3, 0.5], [0.3, 0.3, 0.2, 0.2]])
# Pi = np.array([0.4, 0.6])
# print(viterbi([0, 3, 2, 0], A, B, Pi))
# Expected output: 1, 1, 1, 1

"""**Part 4**

Compare the results obtained from both taggers and a MEMM tagger, implemented by NLTK (a known NLP library), over both, the dev and test datasets. To train the NLTK MEMM tagger you should execute the following lines (it may take some time to train...):
"""

from nltk.tag import tnt 

tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(train_data)
print(tnt_pos_tagger.evaluate(test_data))

"""Print both, word level and sentence level accuracy for all the three taggers in a table."""

# Your code goes here