import numpy as np
import tensorflow as tf
import math
labels = np.ndarray(shape=(6, 1), dtype=np.int32)
labels[0,0] = 20
labels[1,0] = 20
labels[2,0] = 20
labels[3,0] = 8
labels[4,0] = 8
labels[5,0] = 8

train_inputs = np.ndarray(shape=(6), dtype=np.int32)
train_inputs[0] = 10
train_inputs[1] = 8
train_inputs[2] = 7
train_inputs[3] = 20
train_inputs[4] = 8
train_inputs[5] = 11


embeddings = tf.Variable(
    tf.random_uniform([1000, 6], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

sm_weights = tf.Variable(
    tf.truncated_normal([1000, 6],
                        stddev=1.0 / math.sqrt(6)))

# Get context embeddings from labels
true_w_1 = tf.nn.embedding_lookup(sm_weights, labels)
true_w = tf.reshape(true_w_1, [-1, 6])




#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    #print the random values that we sample
    print("u_o")
    print(sess.run(true_w_1))
    print(sess.run(true_w))
    print("v_c")
    print(sess.run(embed))
