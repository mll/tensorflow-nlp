import tensorflow as tf
import sys
from dataset import DataSet
import time

class NeuralNetwork:

    def __init__(self, conv_filters, filters_per_layer):
        dataset = DataSet()
        self.sess = tf.InteractiveSession()
        sess = self.sess
        
        self.conv_filters = conv_filters
        self.filters_per_layer = filters_per_layer
        
        embeddings = tf.Variable(tf.zeros([dataset.vocabulary_size, dataset.embedding_size]), name="embeddings")  
  
        variables = {'embeddings':embeddings}
      
        train_left = tf.placeholder(tf.int32, shape=[None, dataset.sentence_length])
        train_right = tf.placeholder(tf.int32, shape=[None, dataset.sentence_length])
        train_labels = tf.placeholder(tf.int32, shape=[None, 2])
        keep_prob = tf.placeholder(tf.float32)
  
        inputs = {'train_left': train_left,
                  'train_right': train_right,
                  'train_labels': train_labels,
                  'keep_prob': keep_prob}
  
        embed_left = tf.nn.embedding_lookup(embeddings, train_left)
        embed_right = tf.nn.embedding_lookup(embeddings, train_right)
  
        filter_variables = []
        filter_biases = []
        left_filter_layers = []
        right_filter_layers = []
        other_layers = []
  
        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        for width in conv_filters:
            W  = weight_variable([width, dataset.embedding_size, filters_per_conv], name=str(width)+"filter")
            variables[W.name] = W
            b = bias_variable([filters_per_conv], name=str(width)+"bias")
            variables[b.name] = b
            conv_left = tf.nn.conv1d(value=embed_left, filters=W, stride=1, padding='SAME', data_format='NHWC')
            conv_right = tf.nn.conv1d(value=embed_right, filters=W, stride=1, padding='SAME', data_format='NHWC')
            relu_left = tf.nn.relu(conv_left + b)
            relu_right = tf.nn.relu(conv_right + b)
            left = tf.reduce_max(relu_left, axis=[1])
            right = tf.reduce_max(relu_right, axis=[1])

            other_layers.append([conv_left, conv_right, relu_left, relu_right])
            filter_variables.append(W)
            filter_biases.append(b)
            left_filter_layers.append(left)
            right_filter_layers.append(right)
        
        full_left_layer = tf.concat(left_filter_layers, 1)
        full_right_layer =  tf.concat(right_filter_layers, 1)

        composed_layer = tf.concat([full_left_layer, full_right_layer], 1)
  
        dropout = tf.nn.dropout(composed_layer, keep_prob=keep_prob)
        W_softmax = weight_variable([len(conv_filters)*filters_per_conv*2, 2], name="Wsoftmax")
        b_softmax = bias_variable([2], name="Bsoftmax")
  
        variables["W_softmax"] = W_softmax
        variables["b_softmax"] = b_softmax
  
        y = tf.nn.softmax(tf.matmul(dropout, W_softmax) + b_softmax)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=y))
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
        zero_last_embedding_row = tf.scatter_update(embeddings,dataset.vocabulary_size - 1, tf.zeros([dataset.embedding_size]))
  
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(train_labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
  
        self.prediction = y
        self.dataset = dataset
        self.training_left, self.training_right, self.training_labels = dataset.training_data  
        self.validation_left, self.validation_right, self.validation_labels = dataset.validation_data
        self.test_left, self.test_right, self.test_labels = dataset.test_data

        self.train_left = inputs['train_left']
        self.train_right = inputs['train_right']
        self.train_labels = inputs['train_labels']
        self.keep_prob = inputs['keep_prob']

        self.embeddings_loader = tf.train.Saver({'embeddings': variables["embeddings"]})
        self.full_saver = tf.train.Saver(variables)
        self.sess = sess
        self.variables = variables
        self.train_step = train_step
        self.cross_entropy = cross_entropy
        self.accuracy = accuracy
        self.zero_last_embedding_row = zero_last_embedding_row
        
        init = tf.global_variables_initializer()
        sess.run(init)
        self.embeddings_loader.restore(sess, './embeddings.ckpt')
        tf.summary.FileWriter('./graph',sess.graph)
        self.full_saver.restore(sess, "./full_training.ckpt")

        

    def print_accuracy(self, left_words, right_words, labels, name, batch_size = 10000):
        batch_left_words = left_words[0:batch_size]
        batch_right_words = right_words[0:batch_size]
        batch_labels = labels[0:batch_size] 
    
        feed_dict = {self.train_left: batch_left_words, 
                     self.train_right: batch_right_words, 
                     self.train_labels: batch_labels, 
                     self.keep_prob: 1.0}
        entropy = self.cross_entropy.eval(feed_dict=feed_dict)
        print(name, 'cross_entropy:', '%.16f' % (entropy,))
        print(name, 'accuracy:', '%.16f' % (self.accuracy.eval(feed_dict=feed_dict),))    
  
    def train(self, epochs=60): 
        sess = self.sess
        self.print_accuracy(self.training_left, self.training_right, self.training_labels, 'Training')
        self.print_accuracy(self.validation_left, self.validation_right, self.validation_labels, 'Validation')
        sys.stdout.flush()
  
        for epoch in range(1, epochs):
            start = time.time()
            print("Epoch ", epoch, "starts...")
            steps = int(len(self.training_left) / self.dataset.batch_size)
            for i in range(0, steps): 
                sys.stdout.write(str(i)+"/"+str(steps)+"\r")
                sys.stdout.flush()
                batch = self.dataset.get_next_batch()
                sess.run(self.train_step, feed_dict={self.train_left: batch[0], self.train_right: batch[1], self.train_labels: batch[2], self.keep_prob: 0.5}) 
                sess.run(self.zero_last_embedding_row)
            print("Epoch ", epoch, "complete.")
            self.print_accuracy(self.training_left, self.training_right, self.training_labels, 'Training')
            self.print_accuracy(self.validation_left, self.validation_right, self.validation_labels, 'Validation')
            save_path = self.full_saver.save(sess, "./full_training.ckpt")
            print("Model saved in file: %s" % save_path)
            end = time.time()
            print('Elapsed time: ',end - start, 'seconds')

    def predict_for_sentence_pairs(self, sentence_pairs):
        left_sentences = list(map(self.dataset.feature_representation_of_sentence, map(lambda x: x[0], sentence_pairs)))
        right_sentences = list(map(self.dataset.feature_representation_of_sentence, map(lambda x: x[1], sentence_pairs)))
        return self.prediction.eval(session=self.sess, feed_dict={self.train_left: left_sentences, self.train_right: right_sentences, self.keep_prob: 1.0}) 


conv_filters = [3, 4, 5]
filters_per_conv = 100
  
if __name__ == "__main__": 
    network = NeuralNetwork(conv_filters, filters_per_conv)
    network.train()

