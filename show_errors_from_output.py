import argparse
from vectorize_data import Word2vecWrapper
import gensim
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

#TODO: Everything is hardcoded here. Make it into command line parameters

O = "O"

# Only for one category, and token level and don't take B or I into account

def read_from_file(filename):
    corrects = []
    false_positives = []
    false_negatives = []
    
    f = open(filename)
    
    
    for line in f:
        line = line.strip()
    
        if line == "":
            continue
        sp = line.split()
        word = sp[0]
        expected = sp[1].replace("B-", "").replace("I-", "")
        classified = sp[2].replace("B-", "").replace("I-", "")
        if expected == O and classified == O:
            continue
        if expected == classified:
            corrects.append(word)
        if expected == O:
            false_positives.append(word)
        if classified == O:
            false_negatives.append(word)

    return list(set(corrects)), list(set(false_positives)), list(set(false_negatives))


def plot(word_lists, word2vec_model):
    all_vectors_list = []
    found_words = []
    for lst_nr, lst in enumerate(word_lists):
        for word in lst:
            try:
                vec  = word2vec_model[word]
                print(word)
                print(vec)
                all_vectors_list.append(vec)
                found_words.append((word, lst_nr))
            except KeyError:
                print(word + " not found")

    all_vectors_np = np.array(all_vectors_list)
    print(all_vectors_list)
    print(found_words)

    pca_model = PCA(n_components=2)
    tsne_model = TSNE(n_components=2, random_state=0)
    DX_pca = pca_model.fit_transform(all_vectors_np)
    DX = tsne_model.fit_transform(DX_pca)

    #colors = ['blue', 'green', 'cyan', 'red', 'orange', 'magenta'] #, 'yellow', "brown", "orange", "pink", "gray", "black"]
    colors = ['orange', 'orange', 'orange', 'gray', 'gray', 'gray'] #, 'yellow', "brown", "orange", "pink", "gray", "black"]
    markes = ['s', 's', 's', 'o', 'o', 'o']
    for point, i, (found_word, type) in zip(DX, range(0, len(DX)), found_words):

        w = "normal"
        if type == 2 or type == 5: # false negative
            w = "bold"
        s = 12
        if type == 0 or type == 3: # correct
            found_word = "(" + found_word + ")"
            s = 7

        plt.scatter(point[0], point[1], color = colors[type], marker = markes[type])
        plt.annotate(found_word, (point[0], point[1]), weight = w, size = s)

    current_color_index = 0

    plt.savefig(os.path.join('feature_vectors_pca', 'temp_feature_vector_word2vec_pca.pdf'))



if __name__ == "__main__":
    model_path = '/Users/maria/mariaskeppstedtdsv/post-doc/gavagai/googlespace/GoogleNews-vectors-negative300.bin'
    c_1, fp_1, fn_1=read_from_file("/Users/maria/mariaskeppstedtdsv/post-doc/gavagai/StaViCTA/Maria/pre_annotation/data/example_project/evaluation_simulate_active_learning/B-speculation/1/active/B-speculation_active_NonStructuredLogisticRegression_word2vec_True_conll.csv")
    c_2, fp_2, fn_2=read_from_file("/Users/maria/mariaskeppstedtdsv/post-doc/gavagai/StaViCTA/Maria/pre_annotation/data/example_project/evaluation_simulate_active_learning/B-speculation/1/active/B-speculation_active_NonStructuredLogisticRegression_word2vec_True_conll.csv")
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')
    plot([c_1, fp_1, fn_1, c_2, fp_2, fn_2], word2vec_model)
