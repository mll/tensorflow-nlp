# Created by Marek Lipert (2017). All rights reserved.
# Can be distributed under GPLv3
# See the LICENSE file for details

import random
import re
try:
   import cPickle as pickle
except:
   import pickle

class DataSet:
  def feature_representation_of_sentence(self, sentence):
      words = self._words_from_sentence(sentence)
      return self._word_indices(words)

  def _words_from_sentence(self, sentence): 
      word_list = re.sub('[^0-9a-zA-Z\']+', ' ', sentence).strip(" ").strip("'").lower().split(' ')
      return list(filter(lambda w: len(w) > 0, map(lambda word: word.strip(' ').strip("'"), word_list)))

  def _word_indices(self, words):
      assert self.vocabulary
      mapping = list(map(lambda word: self.vocabulary[word], words))
      self.pad_vector(mapping)
      return mapping
  
  def _read_vocabulary(self):
    with open('vocabulary.pickle', 'rb') as vocabulary_file:
       return pickle.load(vocabulary_file)

  def read_embeddings(self):
    with open('embeddings.pickle', 'rb') as embeddings_file:
       return pickle.load(embeddings_file)


  def _read_data(self):
    with open('training.pickle', 'rb') as vocabulary_file:
       data = pickle.load(vocabulary_file)
       
       for set in data:
        for point in set:
          self.pad_vector(point[0][0])
          self.pad_vector(point[0][1])
       return data

  def pad_vector(self, vector):
     for i in range(len(vector), self.sentence_length):
        vector.append(self.vocabulary_size - 1)
     assert len(vector) == self.sentence_length


  def __init__(self):
     print('Reading the vocabulary...')
     self.vocabulary = self._read_vocabulary()
    
     self.vocabulary_size = len(list(self.vocabulary.keys())) + 1
     self.embedding_size = 300
     self.sentence_length = 238
     self.batch_size = 100
     print('Reading the data...')
     self._data = self._read_data()
     random.shuffle(self._data[0])
     random.shuffle(self._data[1])
     random.shuffle(self._data[2])
     self.training_data = self._separate_data(self._data[0])
     self.validation_data = self._separate_data(self._data[1])
     self.test_data = self._separate_data(self._data[2])
     
     print('Data prepared.')
     self.batch_index = 0

  def get_next_batch(self):
    if self.batch_index + self.batch_size > len(self._data[0]):
       random.shuffle(self._data[0])
       self.batch_index = 0
       return self.get_next_batch()
    batch = self._data[0][self.batch_index:self.batch_index+self.batch_size]
    batch_left = list(map(lambda x: x[0][0], batch))
    batch_right = list(map(lambda x: x[0][1], batch))
    batch_labels = list(map(lambda x: x[1], batch))
    self.batch_index = self.batch_index + self.batch_size
    return batch_left, batch_right, batch_labels
  
  def _separate_data(self, data):
    batch_left = list(map(lambda x: x[0][0], data))
    batch_right = list(map(lambda x: x[0][1], data))
    batch_labels = list(map(lambda x: x[1], data))
    return batch_left, batch_right, batch_labels

