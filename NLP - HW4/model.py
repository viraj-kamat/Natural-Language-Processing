import tensorflow as tf
from tensorflow.keras import layers, models
import pdb
from util import ID_TO_CLASS
import time 


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        
        self.gru_bidrect = layers.Bidirectional(layers.GRU(hidden_size,return_sequences=True))
        
        print("Now running default")
        time.sleep(5)
        
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START

        M = tf.tanh(rnn_outputs)   #---------- equation(9)
        
        alpha = tf.tensordot(M, self.omegas, axes=1, name='alphas')   #---------- equation(10)
        alpha = tf.nn.softmax(alpha, name='alphas')   #---------- equation(10)
        
        output = tf.reduce_sum(rnn_outputs*alpha, 1)   #---------- equation(11)
        output = tf.tanh(output)   #---------- equation(12)
        
        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)
        

        
        ### TODO(Students) START

        masking = tf.cast(inputs!=0, tf.float32)
        concatenated = tf.concat([word_embed,pos_embed],axis=2)
        #concatenated = word_embed
        output = self.gru_bidrect(concatenated,mask=masking) #Is this correct - confirm !
        attn_output = self.attn(output)

        logits = self.decoder(attn_output)

        
        
        

        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int,num_filters:int, filter_sizes: list,  training: bool = False):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        
        print("Now running advanced model.num_filters=64.")
        time.sleep(5)
        initializer = tf.keras.initializers.glorot_normal
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))
        
        self.conv1 = tf.keras.layers.Conv2D(64,3,padding="same",activation="relu",data_format="channels_last")
        self.pool1 = tf.keras.layers.GlobalMaxPool2D()
        #self.conv2 = tf.keras.layers.Conv2D(64,3,padding="same",activation="relu") #Why not two layers ?
        #self.pool2 = tf.keras.layers.GlobalMaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(0.6)
        self.d1 = tf.keras.layers.Dense(19,activation='softmax')
        
        #conf1 - dropout(0.1)
        #con2 - 10 epochs
        #conf3 - num_filters 5 

        
        
        ### TODO(Students END

    def call(self, inputs, pos_inputs, training):
        #raise NotImplementedError
        ### TODO(Students) START
        
        try :
            word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
            pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)
            self.concat = tf.concat([word_embed,pos_embed], axis=2)
            self.concat = tf.expand_dims(self.concat,3)
            
            processed = self.conv1(self.concat)
            processed = self.pool1(processed)
            #processed = self.conv2(processed)
            #processed = self.pool2(processed)
            processed = self.flatten(processed)
            processed= tf.reshape(processed,[self.concat.shape[0],-1])
            processed = self.dropout1(processed)
            processed = self.d1(processed)

        except Exception as e :
            print("There was an error \n")
            print("\n")
            print(e)
        
        
                                                  
        return {'logits': processed}
        
        ### TODO(Students END