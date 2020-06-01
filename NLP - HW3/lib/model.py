# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

# project imports

import pdb

class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        # Comment the next line after implementing call.
        cubed_value = tf.pow(vector,3)

        return cubed_value

        #raise NotImplementedError
        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")
        
        # Trainable Variables
        # TODO(Students) Start
        #print('Init called...')
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.num_transitions = num_transitions
        self.regularization_lambda = regularization_lambda
        self.trainable_embeddings = trainable_embeddings


        #390
        self.embeddings = tf.Variable(tf.random.uniform([vocab_size, embedding_dim], -0.01, 0.01),
                                      trainable=self.trainable_embeddings)



        self.w1 = tf.Variable(tf.random.truncated_normal([hidden_dim, embedding_dim * (num_tokens)],mean=0, stddev=0.01),
                              trainable=True)
        self.w2 = tf.Variable(tf.random.truncated_normal([num_transitions, hidden_dim],mean=0, stddev=0.01), trainable=True)
        self.biases = tf.Variable(tf.zeros([hidden_dim, 1]), trainable=True)




        # TODO(Students) End

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        #print('Call called')

        #try :
        self.lookup_embeddings = tf.reshape(tf.nn.embedding_lookup(self.embeddings, inputs),[inputs.shape[0],-1])
        h = tf.add(tf.matmul(self.lookup_embeddings,tf.transpose(self.w1)), tf.transpose(self.biases))
        h = self._activation(h)
        p = tf.matmul(h,tf.transpose(self.w2))



        logits = p
        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)

        #pdb.set_trace()
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start


        mask = tf.cast(labels >= 0, tf.float32)
        mask2 = tf.cast(labels == 1, tf.float32)
        #pdb.set_trace()
        logits = tf.math.multiply(logits,mask)
        logits = self.custom_softmax(logits) #tf.nn.softmax(logits)
        loss =  tf.math.negative(tf.reduce_mean( tf.math.log(tf.reduce_sum(tf.math.multiply(logits,mask2),1)+1e-10) ))


        if self.trainable_embeddings :
            regularization = self.regularization_lambda * ( tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2) + tf.nn.l2_loss(self.biases) + tf.nn.l2_loss(self.lookup_embeddings) )
        else :
            regularization = self.regularization_lambda * ( tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2) + tf.nn.l2_loss(self.biases)  )

        # TODO(Students) End
        return loss + regularization

    def custom_softmax(self,logits: tf.Tensor) ->  tf.Tensor :
        num = tf.where(logits == 0,0,tf.exp(logits))
        num = tf.where(num == 1, 0,num)
        denom = tf.reshape(tf.reduce_sum(num,axis=1),[logits.shape[0],1])

        logits = num/denom

        return logits

