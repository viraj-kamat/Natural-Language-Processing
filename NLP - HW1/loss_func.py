import tensorflow as tf
#virajk
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ============================================================= =============
    """

    shape = inputs.get_shape()
    row_lenth,col_length = shape.as_list()


    mult_tensor = tf.multiply(inputs, true_w) #Multiply the tensor element-wise
    mult_tensor =  tf.reduce_sum(mult_tensor, axis=1) #Add the elements along the column-axis
    mult_tensor = tf.where(tf.is_nan(mult_tensor), tf.zeros_like(mult_tensor), mult_tensor) #Remove NaN values
    mult_tensor = tf.reshape(mult_tensor, [row_lenth, 1]) #Reshape - yields single column of rowlength batchsize
    A = tf.log(tf.exp(mult_tensor)) #The Numerator

    matmul_tensor = tf.matmul(inputs, true_w, transpose_b=True) #Do a matrix multiplication of the tensor,transpose the context vector
    matmul_tensor = tf.reduce_sum(tf.exp(matmul_tensor),axis=1) #Take an exponent of each term then reduce (Summation)
    matmul_tensor = tf.log(matmul_tensor) #Take a log of each element
    matmul_tensor = tf.reshape(matmul_tensor,[row_lenth, 1]) #Reshape - yields single column of rowlength batchsize
    B = matmul_tensor



    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    inputs_length,embedding_size = inputs.get_shape().as_list()
    num_negs = len(sample)
    center_words = inputs

    unigram_prob = tf.convert_to_tensor(unigram_prob)
    sample = tf.convert_to_tensor(sample)
    weight_labels = tf.reshape(tf.gather(weights, labels),inputs.get_shape()) #Gather weights for labels
    weight_samples = tf.reshape(tf.gather(weights, sample),[num_negs,embedding_size]) #Gather weights for negative samples

    pr_labels = tf.gather(unigram_prob, labels) #Gather probabilities for labels
    pr_samples = tf.reshape(tf.gather(unigram_prob, sample),[num_negs,1]) #Gather probabilities for negative samples

    bias_lables = tf.gather(biases, labels) #Gather bias for labels
    bias_samples = tf.gather(biases, sample) #Gather bias for samples


    #For the true context words only
    s_wo_wc_a = tf.multiply(inputs,weight_labels)
    s_wo_wc_a =  tf.reduce_sum(s_wo_wc_a,axis=1,keepdims=True)
    s_wo_wc = tf.add(s_wo_wc_a, bias_lables)
    log_k_pr_wo = tf.log(tf.scalar_mul(num_negs, pr_labels)+ 1e-10)
    part_1 = tf.log(tf.sigmoid(tf.subtract(s_wo_wc, log_k_pr_wo))+ 1e-10)


    #For the noise
    s_wx_wc_a = tf.matmul(inputs,weight_samples,transpose_b=True)
    s_wx_wc_a = tf.reshape(s_wx_wc_a, [num_negs, inputs_length])
    s_wx_wc_a = tf.add(s_wx_wc_a, tf.reshape(bias_samples,[num_negs,1]))
    log_k_pr_wx = tf.log(tf.scalar_mul(num_negs, pr_samples)+ 1e-10)
    part_2_a = tf.sigmoid(tf.subtract(s_wx_wc_a,log_k_pr_wx))
    one_minus = [[1.0 for j in range(inputs_length)] for i in range(num_negs)]
    one_minus = tf.convert_to_tensor(one_minus)
    part_2 = tf.reshape(tf.reduce_sum(tf.reshape(tf.log(tf.subtract(one_minus, part_2_a)+ 1e-10),[inputs_length,num_negs]),axis=1),[inputs_length,1])


    part_1_part_2 = tf.add(part_1,part_2)
    part_1_part_2 = tf.negative(part_1_part_2)
    return part_1_part_2


