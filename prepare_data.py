import csv
import re
try:
   import cPickle as pickle
except:
   import pickle

def words_from_sentence(sentence): 
    word_list = re.sub('[^0-9a-zA-Z\']+', ' ', sentence).strip(" ").strip("'").lower().split(' ')
    return list(filter(lambda w: len(w) > 0, map(lambda word: word.strip(' ').strip("'"), word_list)))

def sentence_to_embedding(vocabulary, sentence, sentence_length, padding_index):
    sentence = list(map(lambda word: vocabulary[word], sentence))
    return sentence

def read_data(vocabulary, sentence_length):
  with open('train.csv', 'r') as csvfile:
    max_length = 0
    max_sentence = []
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    all_data = []    
    voc_len = len(vocabulary)
    reader.next()    
    
    for row in reader:
       first_sentence = sentence_to_embedding(vocabulary, words_from_sentence(row[3]), sentence_length, voc_len)
       second_sentence = sentence_to_embedding(vocabulary, words_from_sentence(row[4]), sentence_length, voc_len)
       all_data.append([[first_sentence, second_sentence], [int(row[5]), 1 - int(row[5])]])
    
    data_length = len(all_data)
    training_no = int(data_length * 0.8)
    validation_no = int(data_length * 0.9)
    
    return [all_data[0:training_no], all_data[training_no:validation_no], all_data[validation_no:data_length]]

def update_vocabulary(vocabulary, filename, first_sentence_index, second_sentence_index):
  with open(filename, 'r') as csvfile:
    max_length = 0
    max_sentence = []
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    reader.next()
    for row in reader:
         words_one = words_from_sentence(row[first_sentence_index])
         words_two = words_from_sentence(row[second_sentence_index])
         words = words_one + words_two
         max_length = max(len(words_one),len(words_two), max_length)

         if max_length == len(words_one):
            max_sentence = words_one
         if max_length == len(words_two):
            max_sentence = words_two         
         
         for word in words:
           if word in vocabulary:
              vocabulary[word] = vocabulary[word] + 1
           else:
              vocabulary[word] = 1
    return max_length, max_sentence

vocabulary = {}

m1,s1 = update_vocabulary(vocabulary, 'train.csv', 3, 4)
m2,s2 = update_vocabulary(vocabulary, 'test.csv', 1, 2)

sentence_length = max(m1,m2)
keys = list(vocabulary.keys())
keys.sort()

vocabulary = {value: idx for (idx, value) in enumerate(keys)}

print('Full vocabulary contains %d words' % len(vocabulary))

with open("vocabulary.pickle", "wb") as pickle_file:
  pickle.dump(vocabulary, pickle_file)

train = read_data(vocabulary, sentence_length)
print('Read %d training cases, %d validation cases and %d test cases.' % (len(train[0]),len(train[1]), len(train[2])))

with open("training.pickle", "wb") as pickle_file:
  pickle.dump(train, pickle_file)






