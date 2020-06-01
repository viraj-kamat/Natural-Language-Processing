# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models
import pdb
import numpy as np
import sys

class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:




        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start

        self.n_layers = num_layers
        self.dropout = dropout
        self.sequence_of_layers = []
        self.input_dim = input_dim





        for _ in range(0,self.n_layers) :
            self.sequence_of_layers.append(tf.keras.layers.Dense(units=input_dim,activation='relu',use_bias=True))


        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:


        # TODO(students): start

        random_uniform = tf.random.uniform(shape=(sequence_mask.shape))
        random_uniform = tf.math.greater(random_uniform,0.2)
        mask = tf.equal(sequence_mask, tf.ones_like(sequence_mask))
        if training :
            new_mask = tf.math.logical_and(random_uniform, mask)
        else :
            new_mask = mask


        vector_sequence = tf.ragged.boolean_mask(vector_sequence, new_mask)
        vector_avg = tf.math.reduce_mean(vector_sequence,axis=1) #Average of the word vectors
        layer_reprs = []


        # Dropout-Part
        #s = np.random.uniform(0, 1, input_dim)
        #s = s.tolist()
        #s = [int(np.round(x)) for x in s]
        dropout = self.dropout
        input_dim = self.input_dim


        #pdb.set_trace()

        for i,layer in enumerate(self.sequence_of_layers) : #Looping through the layers we defined
            vector_avg = self.sequence_of_layers[i](vector_avg)
            layer_reprs.append(vector_avg)



        combined_vector = vector_avg
        layer_representations = layer_reprs

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.gru_layers = []

        layer =  tf.keras.layers.GRU(units=input_dim,use_bias=True,return_sequences=True,return_state=True)
        for _ in range(0,num_layers) :
            self.gru_layers.append(layer)



        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start


        layer_rep = []
        for i,layer in enumerate(self.gru_layers) :
            vector_sequence,state = layer(inputs=vector_sequence,mask=sequence_mask,training=training)
            layer_rep.append(state)


        combined_vector = state
        layer_representations = layer_rep
        # TODO(students): end



        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
