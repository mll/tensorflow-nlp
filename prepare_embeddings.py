from gensim.models.keyedvectors import KeyedVectors
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle
from dataset import DataSet

print('Loading the model...')
model = KeyedVectors.load_word2vec_format('pretrained.bin', binary=True)
dataset = DataSet()

print('Computing the embeddings')
embeddings = {}

inside_count = 0
outside_count = 0

for word, ind in dataset.vocabulary.items():
   embedding = []
   try:
     embedding = model[word]
     inside_count = inside_count + 1
   except:
     embedding = np.random.uniform(low=-1.0, high=1.0, size=dataset.embedding_size)
     outside_count = outside_count + 1
   embeddings[ind] = embedding

del model

final_embeddings = []

for i in range(0, len(embeddings)):
  final_embeddings.append(embeddings[i])
final_embeddings.append(np.zeros(dataset.embedding_size))

print('Fetched %d internal embeddings and %d random embeddings plus one zero embedding for the padding.' % (inside_count, outside_count))
print('Total number of embeddings: %d ' % len(final_embeddings))

with open("embeddings.pickle", "wb") as pickle_file:
  pickle.dump(final_embeddings, pickle_file)


