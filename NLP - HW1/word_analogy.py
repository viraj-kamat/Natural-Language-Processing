import os
import pickle
import numpy as np
import sys
from scipy.spatial import distance

model_path = './models/'
loss_model = ['cross_entropy','nce']
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

    analogy_file =  open("word_analogy_test.txt","r")
    ordered_file = open("word_analogy_"+loss+"_predictions_test.txt","w+")
    examples = []
    choices = []
    for line in analogy_file :
        concat_split = line.split('||')
        concat_split[0] = concat_split[0].replace('"','').replace('\n','').split(',')
        concat_split[1] = concat_split[1].replace('"', '').replace('\n','').split(',')
        examples = examples + concat_split[0]
        choices = choices + concat_split[1]

        all_diffs = []
        for ele in concat_split[0] :
            pair = ele.split(':')
            w1_vec = np.array(embeddings[dictionary[pair[0]]])
            w2_vec = np.array(embeddings[dictionary[pair[1]]])

            all_diffs.append(np.subtract(w1_vec,w2_vec))
        avg_diff = np.mean(all_diffs)
        sim_array = []
        for idx,ele in enumerate(concat_split[1]) :
            pair = ele.split(':')
            w1_vec = np.array(embeddings[dictionary[pair[0]]])
            w2_vec = np.array(embeddings[dictionary[pair[1]]])

            diff = np.subtract(w1_vec,w2_vec)
            sim_array.append(1-distance.cosine(diff,avg_diff))

            concat_split[1][idx] = '"'+concat_split[1][idx]+'"'

        sort_indices = np.argsort(sim_array)
        sorted_examples = [concat_split[1][x] for x in sort_indices]
        #ordered_file.write(' '.join(sorted_examples))
        write_string_1 = ' '.join(concat_split[1])
        write_string_2 = write_string_1+' '+sorted_examples[0]+' '+sorted_examples[len(sorted_examples)-1]
        #print(write_string_2)
        ordered_file.write(write_string_2)
        ordered_file.write("\n")

    ordered_file.close()




