import tensorflow as tf
from dataset import DataSet
import numpy as np

dataset = DataSet()
sess = tf.InteractiveSession()
initial_embeddings = np.asarray(dataset.read_embeddings())
print("Initializing the embeddings...")
const_embeddings = tf.constant(initial_embeddings)

embeddings = tf.Variable(initial_value=initial_embeddings, dtype=tf.float32, name="embeddings")
print("Embeddings initialized with shape: ", embeddings.get_shape())
saver = tf.train.Saver({"embeddings":embeddings})    
init = tf.global_variables_initializer()
sess.run(init)
save_path = saver.save(sess, "./embeddings.ckpt")
print("Embeddings saved in file: %s" % save_path)
  
  

