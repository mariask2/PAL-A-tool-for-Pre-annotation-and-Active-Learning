from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
import gensim
import glob
import os
import gc
import argparse

import active_learning_preannotation


#######################################
# To read data from conll-format files
######################################

def read_file_labelled_data(file_path, data_file_extension, minority_classes, outside_class):
    # First, read file, to get text and labels, grouped into sentences
    text_vector = []
    label_vector = []
    current_text = []
    current_label = []

    glob_for_files = os.path.join(file_path, "*" + data_file_extension)
    files = glob.glob(glob_for_files)

    if len(files) == 0:
        print("New labelled data with extension " + data_file_extension + " found in file_path " + str(file_path))
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
    def __init__(self, model_path, semantic_vector_length):
        self.word2vec_model = None
        self.model_path = model_path
        self.semantic_vector_length = semantic_vector_length

    def load(self):
        if self.word2vec_model == None:
            print("Loading word2vec model, this might take a while ....")
            self.word2vec_model = gensim.models.Word2Vec.load_word2vec_format(self.model_path, binary=True)

    def get_semantic_vector_length(self):
        return self.semantic_vector_length

    def get_vector(self, word):

        if len(word) == 3 and word[1] == "_":
            word = word[0] # To cover for a bug in scikit learn, one char tokens have been transformed to longer. These are here transformed back
        default_vector = [0] * self.semantic_vector_length

        #print("word, in word2vec wrapper", word)
        try:
            self.load()
            raw_vec = self.word2vec_model[word]
            if len(raw_vec) != self.semantic_vector_length:
                print("The true semantic vector has length " + str(len(raw_vec)))
                print("while the configuration file states that is should have length " + str(self.semantic_vector_length))
                exit(1)
            return raw_vec
        except KeyError:
            return default_vector

    def end(self):
        self.word2vec_model = None
        gc.collect()


def get_resulting_x_vector(current_word_vectorizer, context_word_vectorizer, word, word_count, text_concatenated, vectorized_data, vectorized_data_context, index_in_sentence, sentence_length, use_word2vec, word2vecwrapper, number_of_previous_words, number_of_following_words, use_current_word_as_feature):

    assert(text_concatenated[word_count] == word)

    #print("----------")
    #if word in current_word_vectorizer.vocabulary_:
    #    assert(current_word_vectorizer.vocabulary_[word] == list(vectorized_data[word_count].toarray()[0]).index(1)) 

    #if word in context_word_vectorizer.vocabulary_:
        #assert(context_word_vectorizer.vocabulary_[word] == list(vectorized_data_context[word_count].toarray()[0]).index(1)) 
        #print("context_word_vectorizer.vocabulary[word]", context_word_vectorizer.vocabulary_[word])
        #print("list(vectorized_data_context[word_count].toarray()[0]).index(1)", list(vectorized_data_context[word_count].toarray()[0]).index(1))

    resulting_vector = []
    vectorized_long_format = vectorized_data[word_count].toarray()[0]

    context  = vectorized_data_context[word_count].toarray()[0] # only to get out length of context features
    len_context = len(context)
    

    if use_current_word_as_feature:
        resulting_vector = vectorized_long_format
        #print("word", word)
        #print("Feature vector length without context", len(resulting_vector))
    
        if use_word2vec:
            word2vecvector = word2vecwrapper.get_vector(word)
            resulting_vector = np.concatenate((resulting_vector, word2vecvector))

    #print("Feature vector length without context with word2vec", len(resulting_vector))
    # Before current word

    for i in range(1, number_of_previous_words + 1):
        previous_word = None

        if index_in_sentence - i >= 0:
            previous_word = text_concatenated[word_count-i]
        #print("previous_word", previous_word)
    
        if previous_word:
            previous_long_vector = vectorized_data_context[word_count-i].toarray()[0]
        else:
            previous_long_vector = [2] * len_context

        resulting_vector = np.concatenate((resulting_vector, previous_long_vector))
        #print("Feature vector length with previous context", len(resulting_vector))
    
        if use_word2vec:
            if previous_word:
                previous_vector = word2vecwrapper.get_vector(previous_word)
            else:
                previous_vector = [2] * word2vecwrapper.get_semantic_vector_length()

            resulting_vector = np.concatenate((resulting_vector, previous_vector))
            #print("Feature vector length with word2vec info, previous context", len(resulting_vector))

    # To ensure that these are not reused
    previous_vector = "Empty"
    previous_long_vector = "Empty"

    # After current word
    for i in range(1, number_of_following_words + 1):
        next_word = None
        if index_in_sentence + i < sentence_length:
            next_word = text_concatenated[word_count+i]
        #print("next_word", next_word)
    
        if next_word:
            next_long_vector = vectorized_data_context[word_count+i].toarray()[0] 
        else:
            next_long_vector = [2] * len_context

        resulting_vector = np.concatenate((resulting_vector, next_long_vector))
        #print("Feature vector length also with context after", len(resulting_vector))


        if use_word2vec:
            if next_word:
                next_vector = word2vecwrapper.get_vector(next_word)
            else:
                next_vector = [2] * word2vecwrapper.get_semantic_vector_length()

            #print("next_vector", next_vector[0], next_vector[1])
            resulting_vector = np.concatenate((resulting_vector, next_vector))
            #print("Feature vector length also with word2vec for context after", len(resulting_vector))


    ### 
    #print(word)
    if word_count == 0 or word_count == len(text_concatenated) - 1:
        print("Length final feature vector", len(resulting_vector))


    return resulting_vector


def vectorize_unlabelled(text_vector_unlabelled, current_word_vectorizer, context_word_vectorizer, \
                             use_word2vec, number_of_previous_words, number_of_following_words, use_current_word_as_feature, word2vecwrapper):

    #Unlabelled data
    text_concatenated_unlabelled = np.concatenate(text_vector_unlabelled)
    vectorized_data_unlabelled = current_word_vectorizer.transform(text_concatenated_unlabelled)

    vectorized_data_unlabelled_context = context_word_vectorizer.transform(text_concatenated_unlabelled)

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
                                                          use_current_word_as_feature)
            current_sentence_X.append(resulting_vector)
            word_count = word_count + 1

        if len(current_sentence_X) != 0:
            result_X_unlabelled.append(current_sentence_X)
            current_sentence_X = []


    print("Read unlabelled data, len: ", len(result_X_unlabelled), len(text_vector_unlabelled))

    result_X_unlabelled_np = np.array([np.array(xi) for xi in result_X_unlabelled])
    text_vector_unlabelled_np = np.array([np.array(ti) for ti in text_vector_unlabelled])

    return result_X_unlabelled_np, text_vector_unlabelled_np



# Check that the unlabelled data is not empty
def vectorize_data(text_vector_labelled, text_vector_unlabelled, label_vector_labelled, class_dict, classes, use_word2vec, number_of_previous_words, number_of_following_words, use_current_word_as_feature, min_df_current, min_df_context, word2vecwrapper, current_word_vocabulary, context_word_vocabulary):
    
    if len(text_vector_unlabelled) <= 0:
        print("There is no more unlabelled data available. System will exit")
        exit(1)

    # Vectorize
    text_concatenated_labelled = np.concatenate(text_vector_labelled)

    vocabulary_to_use = None

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
    current_word_vectorizer = CountVectorizer(binary = True, min_df=min_df_current, vocabulary = vocabulary_to_use)
    # only include features that have occurred min_df_current in the labelled data
    vectorized_data_labelled = current_word_vectorizer.fit_transform(text_concatenated_labelled)
 
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

    context_word_vectorizer = CountVectorizer(binary = True, min_df=min_df_context, vocabulary = vocabulary_to_use_context) 
    # include features that have occurred at least min_df_context in the labelled data
    vectorized_data_labelled_context = context_word_vectorizer.fit_transform(text_concatenated_labelled)
    

    #print("Created vectorizer for unlabelled data")
    
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
                                                          number_of_following_words, use_current_word_as_feature)
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

    print("Read labelled data, len: ", len(result_X_labelled), len(result_y_labelled), len(text_vector_labelled))

    #Unlabelled
    result_X_unlabelled_np, text_vector_unlabelled_np = vectorize_unlabelled(text_vector_unlabelled, current_word_vectorizer, context_word_vectorizer, \
                             use_word2vec, number_of_previous_words, number_of_following_words, use_current_word_as_feature, word2vecwrapper)

    result_X_labelled_np = np.array([np.array(xi) for xi in result_X_labelled])
    result_y_labelled_np = np.array([np.array(yi) for yi in result_y_labelled])
    text_vector_labelled_np = np.array([np.array(ti) for ti in text_vector_labelled])
    

    return result_X_labelled_np, result_X_unlabelled_np, result_y_labelled_np, text_vector_labelled_np, text_vector_unlabelled_np, \
        current_word_vectorizer, context_word_vectorizer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    properties, path_slash_format = active_learning_preannotation.load_properties(parser)
    word2vecwrapper = Word2vecWrapper(properties.model_path, properties.semantic_vector_length)
    word2vecwrapper.load()
    print("model", word2vecwrapper.word2vec_model)
    word2vecwrapper.end()
    print("model", word2vecwrapper.word2vec_model)
