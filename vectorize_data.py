from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
import gensim
import glob
import os
import gc
import argparse
import time
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics.pairwise import euclidean_distances

import active_learning_preannotation


#######################################
# To read data from conll-format files
######################################

def read_file_labelled_data(file_path, data_file_extension, minority_classes, outside_class):
    """
    read_file_labelled_data reads all files in the folder given by file_path with the file extensions data_file_extension.
    These files are to be in csv-format with one token per line and labelled in BIO-format (see the example project).
    params: minority_classes: List of the minority_classes to use
    params: outside_class: a string with the outside class (typically 'O')

    returns text_vector:
    Ex
    [['2_2', 'you', 'could', 'see', 'someone', 'moving,', 'regardless', 'of', 'the', 'darkness'], ['5_5', 'it', "'_'", 's_s', 'certainly', 'something', 'to', 'consider', '._.'],

    returns label_vector:
    Ex:
    [['O', 'O', 'B-speculation', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'B-speculation', 'O', 'O', 'O', 'O'], 

    return label_dict: A dictionary where the keys are the numerical representations of the classes, and the items are
    the classes in the form they appear in the annotated data-
    Ex:
    {'O': 2, 'B-speculation': 0, 'I-speculation': 1}

    """
    # First, read file, to get text and labels, grouped into sentences
    text_vector = []
    label_vector = []
    current_text = []
    current_label = []

    glob_for_files = os.path.join(file_path, "*" + data_file_extension)
    files = glob.glob(glob_for_files)

    if len(files) == 0:
        print("No labelled data with extension " + data_file_extension + " found in file_path " + str(file_path))
        exit(1)
    print("Reading labelled data from " + glob_for_files + ". Resulting in "+  str(len(files)) + " files.")


    for file_name in files:
        f = open(file_name)
        print("Opened the file " + file_name)
        previous_line = "first_in_" + file_name # only to use for writing out a god error message
        line_number = 0
        for line in f:
            line_number = line_number + 1 # for error printing
            stripped_line = line.strip()
            if stripped_line != "": # and '\t' in stripped_line:
                try:
                    if '\t' in stripped_line:
                        sp = stripped_line.split('\t')
                        word = sp[0]
                        if word.strip() != "": #omit when there is nothing associated with the label
                            if len(word) == 1:
                                word = word + "_" + word  # to cover for a bug in scikit learn's tokenization 
                            current_text.append(word.lower())
                            label = sp[1]
                            if label not in minority_classes:
                                label = outside_class
                            current_label.append(label)
                        else:
                            print("Will omit the incorrectly formated line of index " + str(line_number) + " Line: **" + stripped_line +  "**")
                    else:
                        print("Will omit the incorrectly formated line of index " + str(line_number) + " Line: **" + stripped_line +  "**")
                except IndexError:
                    print("Index error")
                    print("The following line is incorrect", line)
                    print("The last correct line is", previous_line)
                    print("The index of the incorrect line is " + str(line_number))
                    print("Stripped version **" + line.strip() +  "**")
                    exit(1)
            else: 
                if len(current_text) != 0: # end of sentence
                    text_vector.append(current_text)
                    label_vector.append(current_label)
                current_text = []
                current_label = []
            previous_line = line
        if len(current_text) != 0: # the last sentence
            text_vector.append(current_text)
            label_vector.append(current_label)

    class_dict = {}
    for i, c in enumerate(minority_classes[:] + [outside_class]):
        class_dict[c] = i

    f.close()

    return text_vector, label_vector, class_dict


def read_file_unlabelled_data(file_name):
    """
    read_file_unlabelled_data reades from file_name
    These files are to be in csv-format with one token per line (see the example project).

    returns text_vector:
    Ex:
    [['7_7', 'perhaps', 'there', 'is', 'a_a', 'better', 'way', '._.'], ['2_2', 'Why', 'are', 'you, 'doing','doing', 'it', '._.']]

    """
    # Read file, to get text, grouped into sentences
    text_vector = []
    current_text = []

    f = open(file_name)
    for line in f:
        word = line.strip()
        if word != "":
            if len(word) == 1:
                word = word + "_" + word # to cover for a bug in scikit learn's tokenization
            current_text.append(word.lower())
        else:
            if len(current_text) != 0: # end of sentence
                text_vector.append(current_text)
            current_text = []
    if len(current_text) != 0: # the last sentence
        text_vector.append(current_text)

    f.close()

    return text_vector


########################
# To vectorize the data
#########################

class Word2vecWrapper:
    """
    Word2vecWrapper 

    A class for storing the information regarding the distributional semantics space
    """

    def __init__(self, model_path, semantic_vector_length):
        self.word2vec_model = None
        self.model_path = model_path
        self.semantic_vector_length = semantic_vector_length
        self._vocabulary_list = None

        if semantic_vector_length is not None:
            self.default_vector = [0] * self.semantic_vector_length

        self.empty_vector = None
        self.nearest_centroid_clf = None
    
    def load(self):
        """
        load the semantic space in the memory
        """
        if self.word2vec_model == None:
            print("Loading word2vec model, this might take a while ....")
            self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True, unicode_errors='ignore')
            #self.word2vec_model = gensim.models.Word2Vec.load_word2vec_format(self.model_path, binary=True)
            #self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True)
            print("Loaded word2vec model")

    def get_semantic_vector_length(self):
        return self.semantic_vector_length

    def get_vector(self, word):
        if len(word) == 3 and word[1] == "_":
            word = word[0] # To cover for a bug in scikit learn, one char tokens have been transformed to longer. These are here transformed back
        
        #print("word, in word2vec wrapper", word)
        try:
            self.load()
            raw_vec = self.word2vec_model[word]
            if len(raw_vec) != self.semantic_vector_length:
                print("The true semantic vector has length " + str(len(raw_vec)))
                print("while the configuration file states that is should have length. Change the variable semantic_vector_length in the settings file to "\
                      + str(self.semantic_vector_length))
                exit(1)
            return raw_vec
        except KeyError:
            return self.default_vector

    def end(self):
        """
        remove the semantic space from the memory
        """
        self.word2vec_model = None
        gc.collect()

    def get_similar_word(self, word):
        try:
            similar = self.word2vec_model.most_similar(positive=[word], topn=1000)
            return similar
        except KeyError:
            return []

    def set_vocabulary(self, vocabulary_list):
        #if self._vocabulary_list is None:
        if True:
            self._vocabulary_list = []            
            for el in vocabulary_list:
                if len(el) == 3 and el[1] == "_":
                    self._vocabulary_list.append(el[0])
                else:
                    self._vocabulary_list.append(el)
                #print(self._vocabulary_list)
            
    def load_clustering(self):
        print("Clustering vectors, this might take a while ....")
        if self._vocabulary_list is None:
            raise Exception("set_vocabulary is not yet run")
        
        X_vectors = []
        cluster_words = []
        for word in self._vocabulary_list:
            vector = self.get_vector(word)
            if not all([el1 == el2 for el1, el2 in zip(vector, self.default_vector)]):
                norm_vector = preprocessing.normalize(np.reshape(vector, newshape = (1, self.semantic_vector_length)), norm='l2') # normalize the vector (l2 = eucledian)  
                list_vector = norm_vector[0]
                X_vectors.append(list_vector)
                cluster_words.append(word)

        # Compute DBSCAN
        X = np.matrix(X_vectors)
        self.cluster_word_dict = {}
        self.cluster_dict = {}
        self.cluster_vector_dict = {}
        db = DBSCAN(eps=0.8, min_samples=1).fit(X)
        #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        #core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        clusters_no_outliers_y  = []
        clusters_no_outliers_terms  = []
        clusters_no_outliers_X  = []
        for label, term, vector in zip(labels, cluster_words, X_vectors):
            self.cluster_word_dict[term] = label
            #print(label, term)
            if label != -1:
                clusters_no_outliers_y.append(label)
                clusters_no_outliers_terms.append(term)
                clusters_no_outliers_X.append(vector)
                
                if label not in self.cluster_dict:
                    self.cluster_dict[label] = []
                self.cluster_dict[label].append(term)
                
                if label not in self.cluster_vector_dict:
                    self.cluster_vector_dict[label] = []
                self.cluster_vector_dict[label].append(vector)
        #print("labels", set(labels))
        #print("self.cluster_word_dict", self.cluster_word_dict)

        self.nr_of_clusters = len(set(labels)) 
        self.empty_vector = [0] * self.nr_of_clusters

        self.nearest_centroid_clf = NearestCentroid()
        self.nearest_centroid_clf.fit(clusters_no_outliers_X, clusters_no_outliers_y)
        print("Clustered vectors")
        
    def get_cluster(self, word):
        if len(word) == 3 and word[1] == '_':
            word = word[0]

        if self.nearest_centroid_clf is None:
            raise Exception("load_clustering is not yet run")

        if word in self.cluster_word_dict:
            if self.cluster_word_dict[word] == -1:
                #print("Not found cluster for training data word", word)
                pass
            feature_vector = self.get_features(self.cluster_word_dict[word])
            #print("In training data", word, feature_vector)
            return feature_vector

        vector = self.get_vector(word)
        if not all([el1 == el2 for el1, el2 in zip(vector, self.default_vector)]):
            norm_vector = preprocessing.normalize(np.reshape(vector, newshape = (1, self.semantic_vector_length)), norm='l2') # normalize the vector (l2 = eucledian)
            cluster = self.nearest_centroid_clf.predict(norm_vector)
            cluster_vectors = self.cluster_vector_dict[cluster[0]]
            distances = euclidean_distances(cluster_vectors, norm_vector)    
            #print("distances", distances)
            min_distance = min(distances)
            if min_distance <= 1.2:
                #if word not in self.cluster_dict[cluster[0]]:
                #print(word + " " + str(self.cluster_dict[cluster[0]]))
                return self.get_features(cluster[0])
            else:
                #print("Not found cluster for new word", word)
                return self.empty_vector[:]
        else:
            return self.empty_vector[:]


    def get_features(self, label):
        if self.empty_vector is None:
            raise Exception("load_clustering is not yet run")
        
        if label == -1:
            return self.empty_vector

        vector = self.empty_vector[:]
        vector[label] = 1
        return vector
 

def get_resulting_x_vector(current_word_vectorizer, context_word_vectorizer, word, word_count, text_concatenated, vectorized_data, vectorized_data_context, index_in_sentence, sentence_length, use_word2vec, word2vecwrapper, number_of_previous_words, number_of_following_words, use_current_word_as_feature, len_context, use_clustering):
    """
    get_resulting_x_vector

    internal function for the model constructing the feature vector
    """
    start_time = time.time()

    #assert(text_concatenated[word_count] == word)

    #print("----------")
    #if word in current_word_vectorizer.vocabulary_:
    #    assert(current_word_vectorizer.vocabulary_[word] == list(vectorized_data[word_count].toarray()[0]).index(1)) 

    #if word in context_word_vectorizer.vocabulary_:
        #assert(context_word_vectorizer.vocabulary_[word] == list(vectorized_data_context[word_count].toarray()[0]).index(1)) 
        #print("context_word_vectorizer.vocabulary[word]", context_word_vectorizer.vocabulary_[word])
        #print("list(vectorized_data_context[word_count].toarray()[0]).index(1)", list(vectorized_data_context[word_count].toarray()[0]).index(1))

    vectorized_long_format = vectorized_data[word_count].toarray()[0]

    if use_current_word_as_feature:
        resulting_vector = vectorized_long_format
        #print("word", word)
        #print("Feature vector length without context", len(resulting_vector))

        #print("resulting_vector", resulting_vector)
        if use_word2vec:
            word2vecvector = word2vecwrapper.get_vector(word)
            resulting_vector = np.concatenate((resulting_vector, word2vecvector))

        if use_clustering:
            clustervector = word2vecwrapper.get_cluster(word)
            resulting_vector = np.concatenate((resulting_vector, clustervector))
        
        #print("resulting_vector, after", resulting_vector)    
        #print("")
    #print("Feature vector length without context with word2vec", len(resulting_vector))
    # Before current word

    not_start_end = [0, 0]
    start = [1, 0]
    end = [0, 1]
    for i in range(1, number_of_previous_words + 1):
        previous_word = None

        if index_in_sentence - i >= 0:
            previous_word = text_concatenated[word_count-i]
        #print("previous_word", previous_word)
    
        if previous_word:
            previous_long_vector = np.concatenate((vectorized_data_context[word_count-i].toarray()[0], not_start_end))
        else:
            previous_long_vector = [0] * len_context + start

        resulting_vector = np.concatenate((resulting_vector, previous_long_vector))
        #print("Feature vector length with previous context", len(resulting_vector))
    
        if use_word2vec:
            if previous_word:
                previous_vector = np.concatenate((word2vecwrapper.get_vector(previous_word), not_start_end))
            else:
                previous_vector = [0] * word2vecwrapper.get_semantic_vector_length() + start

            resulting_vector = np.concatenate((resulting_vector, previous_vector))
            #print("Feature vector length with word2vec info, previous context", len(resulting_vector))
        
        if use_clustering:    
            if previous_word:
                previous_vector_cluster = word2vecwrapper.get_cluster(previous_word)
            else:
                previous_vector_cluster = word2vecwrapper.empty_vector

            resulting_vector = np.concatenate((resulting_vector, previous_vector_cluster))

    # To ensure that the code is not written so these are reused
    previous_vector = "Empty"
    previous_long_vector = "Empty"
    previous_vector_cluster = "Empty"

    # After current word
    for i in range(1, number_of_following_words + 1):
        next_word = None
        if index_in_sentence + i < sentence_length:
            next_word = text_concatenated[word_count+i]
        #print("next_word", next_word)
    
        if next_word:
            next_long_vector = np.concatenate((vectorized_data_context[word_count+i].toarray()[0], not_start_end)) 
        else:
            next_long_vector = [0] * len_context + end

        resulting_vector = np.concatenate((resulting_vector, next_long_vector))
        #print("Feature vector length also with context after", len(resulting_vector))


        if use_word2vec:
            if next_word:
                next_vector = np.concatenate((word2vecwrapper.get_vector(next_word), not_start_end))
            else:
                next_vector = [0] * word2vecwrapper.get_semantic_vector_length() + end

            resulting_vector = np.concatenate((resulting_vector, next_vector))
            
        if use_clustering:    
            if next_word:
                next_vector_cluster = word2vecwrapper.get_cluster(next_word)
            else:
                next_vector_cluster = word2vecwrapper.empty_vector

            resulting_vector = np.concatenate((resulting_vector, next_vector_cluster))

    ### 
    #print(word)
    if word_count == 0 or word_count == len(text_concatenated) - 1 or word_count % 10000 == 0:
        print("Vectorizing word number: ", word_count, " Length feature vector", len(resulting_vector))

    return resulting_vector


def vectorize_unlabelled(text_vector_unlabelled, current_word_vectorizer, context_word_vectorizer, \
                             use_word2vec, number_of_previous_words, number_of_following_words, \
                             use_current_word_as_feature, word2vecwrapper, use_clustering):
    """
    vectorize_unlabelled
    internal function for the module for vectorizing unlabelled data
    """


    #original
    result_X_unlabelled_np = do_vectorize_unlabelled(text_vector_unlabelled,\
                                                               current_word_vectorizer, context_word_vectorizer, \
                                                                use_word2vec, number_of_previous_words,\
                                                                number_of_following_words, \
                                                                use_current_word_as_feature, word2vecwrapper, use_clustering)  

    """                                                            
    if use_word2vec:
        word2vecwrapper.load() # Do it before, otherwise it will be done many times in parallell

    
    #parallell, but not in a way that speeds up things
    result_X_unlabelled_append_np = Parallel(n_jobs=-1)(delayed(vectorize_unlabelled_sub_parts)([el], current_word_vectorizer, context_word_vectorizer, \
                                                                use_word2vec, number_of_previous_words,\
                                                                number_of_following_words, \
                                                                use_current_word_as_feature, word2vecwrapper) for el in text_vector_unlabelled)



    result_X_unlabelled_append_np = [res[0] for res in result_X_unlabelled_append_np]
    

    """
    text_vector_unlabelled_np = np.array([np.array(ti) for ti in text_vector_unlabelled])
    print("Vectorized finnished")
    
    return result_X_unlabelled_np, text_vector_unlabelled_np

def do_vectorize_unlabelled(text_vector_unlabelled, current_word_vectorizer, context_word_vectorizer, \
                             use_word2vec, number_of_previous_words, number_of_following_words, \
                             use_current_word_as_feature, word2vecwrapper, use_clustering):

    #Unlabelled data
    #print(text_vector_unlabelled)
    text_concatenated_unlabelled = np.concatenate(text_vector_unlabelled)
    vectorized_data_unlabelled = current_word_vectorizer.transform(text_concatenated_unlabelled)

    vectorized_data_unlabelled_context = context_word_vectorizer.transform(text_concatenated_unlabelled)

    context_example = vectorized_data_unlabelled_context[0].toarray()[0] # only to get out length of context features
    len_context = len(context_example)

    #vectorized_data_test_vocabulary = None

    result_X_unlabelled = []
    current_sentence_X = []


    word_count = 0
    for text in text_vector_unlabelled:
        for i, t in zip(range(0, len(text)), text):
            resulting_vector = get_resulting_x_vector(current_word_vectorizer, context_word_vectorizer, \
                                                          t, word_count, text_concatenated_unlabelled, \
                                                          vectorized_data_unlabelled, vectorized_data_unlabelled_context, \
                                                          i, len(text), use_word2vec, word2vecwrapper, \
                                                          number_of_previous_words, number_of_following_words, \
                                                          use_current_word_as_feature, len_context, use_clustering)
            current_sentence_X.append(resulting_vector)
            word_count = word_count + 1

        if len(current_sentence_X) != 0:
            result_X_unlabelled.append(current_sentence_X)
            current_sentence_X = []


    #print("Read unlabelled data, len: ", len(result_X_unlabelled), len(text_vector_unlabelled))
    result_X_unlabelled_np = np.array([np.array(xi) for xi in result_X_unlabelled])

    return result_X_unlabelled_np



def vectorize_data(text_vector_labelled, text_vector_unlabelled, label_vector_labelled, class_dict, use_word2vec,\
                        number_of_previous_words, number_of_following_words, use_current_word_as_feature,\
                        min_df_current, min_df_context, max_df_current, max_df_context, word2vecwrapper,\
                        current_word_vocabulary, context_word_vocabulary, use_clustering):

    """
    vectorize_data

    params: text_vector_labelled: list of samples containing the tokens in the labelled data
    Ex:
    [['2_2', 'you', 'could', 'see', 'someone', 'moving,', 'regardless', 'of', 'the', 'darkness'], ['5_5', 'it', "'_'", 's_s', 'certainly', 'something', 'to', 'consider', '._.'],

    params: text_vector_unlabelled: list of samples containing the tokens in the unlabelled data
    Ex:
    [['7_7', 'perhaps', 'there', 'is', 'a_a', 'better', 'way', '._.'], ['2_2', 'Why', 'are', 'you, 'doing','doing', 'it', '._.']]

    params: label_vector_labelled: label_vector
    Ex:
    [['O', 'O', 'B-speculation', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'B-speculation', 'O', 'O', 'O', 'O'], 

    params: class_dict: A dictionary where the keys are the numerical representations of the classes, and the items are
    the classes in the form they appear in the annotated data-
    Ex:
    {'O': 2, 'B-speculation': 0, 'I-speculation': 1}

    params: use_word2vec: Whether to use word2vec

    params: number_of_previous_words: The context in the form of the number of previous words before the current word to include when training the classifiers

    params: number_of_following_words: The context in the form of the number of following words after the current word to include when training the classifiers

    params: use_current_word_as_feature: Whether to include the current token as feature

    params: min_df_current:  A cut-off for the number of occurrences of a token in the data for it to be included as a feature for the current word

    params: min_df_context: A cut-off for the number of occurrences of a token in the data for it to be included as a feature for the context words
    
    params: max_df_current:  A maximum cut-off for the number (or %) of occurrences of a token in the data for it to be included as a feature for the current word
    
    params: max_df_context: A maximum cut-off for the number (or %) of occurrences of a token in the data for it to be included as a feature for the context words

    params: word2vecwrapper: An instance of the Word2vecWrapper, to be able to get semantic information

    params: current_word_vocabulary: If there is an external list to use to be decide whether a token should be included in the current
    vocabulary, this is a string with the search path to this vocabulary. Otherwise set to Fasle

    params: context_word_vocabulary: If there is an external list to use to be decide whether a token should be included in the context
    vocabulary, this is a string with the search path to this vocabulary. Otherwise set to Fasle

    """

    if len(text_vector_unlabelled) <= 0:
        print("There is no more unlabelled data available. System will exit")
        exit(1)

    # Vectorize
    text_concatenated_labelled = np.concatenate(text_vector_labelled)

    vocabulary_to_use = None

    print("Started vectorizing labelled data of length ", len(text_vector_labelled))

    # If an external vocabulary has been given, find out which of the words included in the corpus that is included in the vocabulary
    
    if current_word_vocabulary:
        temp_word_vectorizer = CountVectorizer(binary = True)
        temp_word_vectorizer.fit_transform(text_concatenated_labelled)
        vocabulary = set()
        f = open(current_word_vocabulary, encoding='utf-8', errors='ignore')
        for line in f:
            vocabulary.add(line.strip())
        vocabulary_to_use = []
        for word in temp_word_vectorizer.get_feature_names():
            if word in vocabulary or word.split("_")[0] in vocabulary:
                vocabulary_to_use.append(word)
        
    # Create a vectorizer for all words that are included (fit on training data)    
    # (min_df ignored when vocabulary is not none)
    current_word_vectorizer = CountVectorizer(binary = True, min_df=min_df_current, \
                                              max_df = max_df_current, vocabulary = vocabulary_to_use)

    # only include features that have occurred min_df_current in the labelled data
    vectorized_data_labelled = current_word_vectorizer.fit_transform(text_concatenated_labelled)

    # Clustering
    if use_clustering:
        word2vecwrapper.set_vocabulary(current_word_vectorizer.get_feature_names())
        word2vecwrapper.load_clustering()

    vocabulary_to_use_context = None
    # Create a vectorizer for all words that are included (fit on training data)
    if context_word_vocabulary:
        temp_word_vectorizer = CountVectorizer(binary = True)
        temp_word_vectorizer.fit_transform(text_concatenated_labelled)
        vocabulary = set()
        f = open(context_word_vocabulary, encoding='utf-8', errors='ignore')
        for line in f:
            vocabulary.add(line.strip())
        vocabulary_to_use_context = []
        for word in temp_word_vectorizer.get_feature_names():
            if word in vocabulary or word.split("_")[0] in vocabulary:
                vocabulary_to_use_context.append(word)

    context_word_vectorizer = CountVectorizer(binary = True, min_df = min_df_context,\
                                              max_df = max_df_context, vocabulary = vocabulary_to_use_context)

    # include features that have occurred at least min_df_context in the labelled data
    vectorized_data_labelled_context = context_word_vectorizer.fit_transform(text_concatenated_labelled)
    

    context_example = vectorized_data_labelled_context[0].toarray()[0] # only to get out length of context features
    len_context = len(context_example)

    # Then, use the vectorizer to create vectorized data
    # Labelled
    result_X_labelled = []
    current_sentence_X = []
    result_y_labelled= []

    word_count = 0
    for label, text in zip(label_vector_labelled, text_vector_labelled):
        for i, t, l in zip(range(0, len(text)), text, label):
            resulting_vector = get_resulting_x_vector(current_word_vectorizer, context_word_vectorizer, t, word_count,\
                                                          text_concatenated_labelled, vectorized_data_labelled, \
                                                          vectorized_data_labelled_context, i, len(text), \
                                                          use_word2vec, word2vecwrapper, number_of_previous_words, \
                                                          number_of_following_words, use_current_word_as_feature, len_context, use_clustering)
            #print(resulting_vector)
            current_sentence_X.append(resulting_vector)
            word_count = word_count + 1

        if len(current_sentence_X) != 0:
            transformed_y = [class_dict[l] for l in label]

            try:
                assert(len(current_sentence_X) == len(transformed_y))
            except AssertionError:
                print("len(current_sentence_X) != len(transformed_y)")
                print("current_sentence_X", current_sentence_X, len(current_sentence_X), "transformed_y", transformed_y, len(transformed_y), text)
                exit(1)

            result_X_labelled.append(current_sentence_X)
            current_sentence_X = []            

            result_y_labelled.append(transformed_y)

    print("Vectorized labelled data, len: ", len(result_X_labelled), len(result_y_labelled), len(text_vector_labelled))

    #Unlabelled

    print("Started vectorizing unlabelled data of length ", len(text_vector_unlabelled))
    result_X_unlabelled_np, text_vector_unlabelled_np = vectorize_unlabelled(text_vector_unlabelled, current_word_vectorizer, context_word_vectorizer, \
                             use_word2vec, number_of_previous_words, number_of_following_words, use_current_word_as_feature, word2vecwrapper, use_clustering)
    print("Finished vectorizing unlabelled data")

    result_X_labelled_np = np.array([np.array(xi) for xi in result_X_labelled])
    result_y_labelled_np = np.array([np.array(yi) for yi in result_y_labelled])
    text_vector_labelled_np = np.array([np.array(ti) for ti in text_vector_labelled])

    return result_X_labelled_np, result_X_unlabelled_np, result_y_labelled_np, text_vector_labelled_np, text_vector_unlabelled_np, \
        current_word_vectorizer, context_word_vectorizer



if __name__ == "__main__":
    # Run with, for instance: python vectorize_data.py --project=data.example_project
    parser = argparse.ArgumentParser()
    properties_main, path_slash_format, path_dot_format = active_learning_preannotation.load_properties(parser)
    word2vecwrapper = Word2vecWrapper(properties_main.model_path, properties_main.semantic_vector_length)
    #word2vecwrapper.load()


    word2vecwrapper.set_vocabulary(["good", "excellent", "superb", "outstanding", "exceptional", "marvellous", "wonderful", "would", "if", "could", "whether",\
                                        "unless", "provided", "as long as", "given that", "would", "but", "still", "while", "however", "yet",\
                                        "though", "although",  "despite", "anyway", "regardless", "on the other hand", "bad", "poor", "inferior",\
                                        "second-rate", "second-class", "unsatisfactory", "inadequate", "unacceptable", "january", "february", "march",\
                                        "june", "july", "september", "october", "november", "december", "milk", "juice", "coffee", "tea", "fanta", "water",\
                                        "or", "may", "should", "think", "might", "likely", "question", "probably", "possible", "thought", "believe", "either",\
                                        "seem", "unlikely", "possibility", "considered", "apparently", "suggest", "guess", "suggested", "possibly", "suspect"])

    word2vecwrapper.load_clustering()

    print("perhaps")
    print(word2vecwrapper.get_cluster("perhaps"))

    print("probably")
    print(word2vecwrapper.get_cluster("probably"))

    print("april")
    print(word2vecwrapper.get_cluster("april"))

    print("coffee")
    print(word2vecwrapper.get_cluster("coffee"))

    print("tea")
    print(word2vecwrapper.get_cluster("tea"))

    print("solkjer123")
    print(word2vecwrapper.get_cluster("solkjer123"))


    """
    print("green")
    print(word2vecwrapper.get_similar_word("green"))
    print("tea")
    print(word2vecwrapper.get_similar_word("tea"))
    print("car")
    print(word2vecwrapper.get_similar_word("car"))
    print("jump")
    print(word2vecwrapper.get_similar_word("jump"))
    print("say")
    print(word2vecwrapper.get_similar_word("say"))
    print("Obama")
    print(word2vecwrapper.get_similar_word("Obama"))
    """

    """
    print("model", word2vecwrapper.word2vec_model)
    word2vecwrapper.end()
    print("model", word2vecwrapper.word2vec_model)
    """

    
