import os
import pickle
import numpy as np
import sys
from scipy.spatial import distance

model_path = './models/'
loss_model = ['cross_entropy']
# loss_model = 'nce'

#model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

#dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))



"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
for loss in loss_model : #Generate model predictions in loop

    model_filepath = os.path.join(model_path, 'word2vec_%s.model' % (loss))
    dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))

    

    word_list = ["first","american","would"]
    word_list_ans = {}
    for word in word_list:
        word_list_ans[word] = []
        word_list_temp = []
        for comapare_word in dictionary :
            difference =   tuple ( ((1- distance.cosine(embeddings[dictionary[word]], embeddings[dictionary[comapare_word]])) , comapare_word) )
            word_list_temp.append(difference)
            
        word_list_temp = sorted(word_list_temp, key=lambda x: x[0])
        word_list_temp = word_list_temp[-21:-1]
        
        for match in word_list_temp :
            word_list_ans[word].append(match[1])
            
    print(word_list_ans)
        



'''
    examples = ['first','american','would']
    similar = {}
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    for i in range(3):
        similar[examples[i]] = []
        word = dictionary[examples[i]]
        top_k = 20  # number of nearest neighbors
        sim = cdist(embeddings,embeddings[word].reshape(1,-1)).reshape(-1,).argsort()
        nearest = sim[1:top_k + 1]
        log_str = "Nearest to %s:" % examples[i]
        for k in range(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log_str = "%s %s," % (log_str, close_word)
            similar[examples[i]].append(close_word)
        # print(log_str)
        print(examples[i],' : ',similar[examples[i]])
'''




