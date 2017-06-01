# Overview

The code in this repository implements ideas from 'Convolutional Neural Networks for Sentence Classification' by Yoon Kim from the New York University (https://arxiv.org/pdf/1404.2188.pdf) in context 
of detecting question duplicates on Quora. The implementation relies on pre-trained word2vec embeddings (https://code.google.com/archive/p/word2vec/) and adds a convolutional neural network to detect 
words in certain sequences. Specifically, there is a single embedding layer followed by one convolution - max-pooling - ReLu layer with dropout followed by a softmax classification layer.

The implementation is done in Tensorflow and therefore easily scalable.

# Requirements

The computer this is run on should have at least 16Gb of RAM.

0. Install python 3 and wget 
1. Install tensorflow for python 3 (https://www.tensorflow.org/install/)
2. Install gensim (pip install gensim)
3. Run ./prepare.sh

To train the model, use

python train.py

The resultant network is dumped every epoch.
To use the trained model to predict the outcome for the submission set, run

python evaluate.py



