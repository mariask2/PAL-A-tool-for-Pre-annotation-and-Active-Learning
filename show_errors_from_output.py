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
from sklearn import preprocessing


#TODO: Everything is hardcoded here. Make it into command line parameters

O = "O"

# Only for one category, and token level and don't take B or I into account

def read_from_file(filename):
    corrects = []
    false_positives = []
    false_negatives = []
    outs = []

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
            outs.append(word)
        elif expected == classified:
            corrects.append(word)
        elif expected == O:
            false_positives.append(word)
        elif classified == O:
            false_negatives.append(word)

    return list(set(corrects)), list(set(false_positives)), list(set(false_negatives)), list(set(outs))


def plot(word_lists, word2vec_model):
    all_vectors_list = []
    found_words = []
    for lst_nr, lst in enumerate(word_lists):
        for word in lst:
            try:
                vec_raw  = word2vec_model[word]
                norm_vector = list(preprocessing.normalize(np.reshape(vec_raw, newshape = (1, 400)), norm='l2')[0])
                print(word)
                #print(vec)
                all_vectors_list.append(norm_vector)
                found_words.append((word, lst_nr))
            except KeyError:
                print(word + " not found")

    all_vectors_np = np.array(all_vectors_list)
    #print(all_vectors_list)
    #print(found_words)

    pca_model = PCA(n_components=50)
    tsne_model = TSNE(n_components=2, random_state=0)
    DX_pca = pca_model.fit_transform(all_vectors_np)
    DX = tsne_model.fit_transform(DX_pca)

    colors = ["orange", "lightgray", "red", "orange", "lightgray", "red", "orange", "lightgray", "red"]
    #colors = ['blue', 'green', 'cyan', 'red', 'orange', 'magenta'] #, 'yellow', "brown", "orange", "pink", "gray", "black"]

    markes = ['x', '.', 's', 'x', '.', 's', 'x', '.', 's']
    index = 0

    fig = plt.figure()
    
    sub_plot_1 = fig.add_subplot(1,3,1)
    plt.title("Person")
    for point, i, (found_word, type) in zip(DX, range(0, len(DX)), found_words):
        if type > 2:
            break

        plt.scatter(point[0], point[1], color = colors[type], marker = markes[type], s=1)
        #if index % 50 == 0:
        #    plt.annotate(found_word, (point[0], point[1]))
        index = index + 1

    sub_plot_2 = fig.add_subplot(1,3,2)
    plt.title("Organisation")
    for point, i, (found_word, type) in zip(DX, range(0, len(DX)), found_words):
        if type > 5:
            break
        if type < 3:
            continue
        plt.scatter(point[0], point[1], color = colors[type], marker = markes[type], s=1)
        #if index % 50 == 0:
        #    plt.annotate(found_word, (point[0], point[1]))
        index = index + 1

    sub_plot_3 = fig.add_subplot(1,3,3)
    plt.title("Location")
    for point, i, (found_word, type) in zip(DX, range(0, len(DX)), found_words):
        if type > 8:
            break
        if type < 6:
            continue
        plt.scatter(point[0], point[1], color = colors[type], marker = markes[type], s=1)
        #if index % 50 == 0:
        #    plt.annotate(found_word, (point[0], point[1]))
        index = index + 1


    plt.savefig(os.path.join('feature_vectors_pca', 'temp_feature_vector_word2vec_pca.pdf'))



if __name__ == "__main__":
    model_path = "/mnt/data2/maria/twitter_space/word2vec_twitter_model.bin"
    c_1, fp_1, fn_1, outs_1 =read_from_file("/home/maria/pal_test_runs/2018-03_23/PAL-A-tool-for-Pre-annotation-and-Active-Learning/data/twitter_ner/evaluation_simulate_active_learning/B-per/1/active/B-per_active_NonStructuredLogisticRegression_word2vec_True_conll.csv")
    c_2, fp_2, fn_2, outs_2 =read_from_file("/home/maria/pal_test_runs/2018-03_23/PAL-A-tool-for-Pre-annotation-and-Active-Learning/data/twitter_ner/evaluation_simulate_active_learning/B-org/1/active/B-org_active_NonStructuredLogisticRegression_word2vec_True_conll.csv")
    c_3, fp_3, fn_3, outs_3 =read_from_file("/home/maria/pal_test_runs/2018-03_23/PAL-A-tool-for-Pre-annotation-and-Active-Learning/data/twitter_ner/evaluation_simulate_active_learning/B-loc/1/active/B-loc_active_NonStructuredLogisticRegression_word2vec_True_conll.csv")

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')
    print(len(c_1))
    print(len(fp_1))
    print(len(fn_1))
    print(len(c_2))
    print(len(fp_2))
    print(len(fn_2))

    plot([c_1, fp_1, fn_1, c_2, fp_2, fn_2, c_3, fp_3, fn_3, list(set(outs_1 + outs_2 + outs_3))], word2vec_model)

