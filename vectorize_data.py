from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
import gensim
import glob
import os
import gc

#from settings import model_path

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

    #print(len(text_vector))
    #print(len(label_vector))
    #print(class_dict)
    #for ts, ls in zip(text_vector, label_vector):
    #    print("*******")
    #    for t, l in zip(ts, ls):
    #        print(t,l)
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
    def __init__(self, model_path):
        self.word2vec_model = None
        self.model_path = model_path

    def get_vector(self, word):

        if len(word) == 3 and word[1] == "_":
            word = word[0] # To cover for a bug in scikit learn, one char tokens have been transformed to longer. These are here transformed back
        default_vector = [0] * 300

        #print("word", word)
        try:
            if self.word2vec_model == None:
                print("Loading word2vec model, this might take a while ....")
                self.word2vec_model = gensim.models.Word2Vec.load_word2vec_format(self.model_path, binary=True)
            raw_vec = self.word2vec_model[word]
            return raw_vec
        except KeyError:
            return default_vector

    def end(self):
        self.word2vec_model = None
        gc.collect()

def get_resulting_x_vector(word, word_count, text_concatenated, vectorized_data, index_in_sentence, sentence_length, use_word2vec, word2vecwrapper):

    #print("word", word)
    previous_word = None
    next_word = None
    if index_in_sentence > 0:
        previous_word = text_concatenated[word_count-1]
    if index_in_sentence < sentence_length - 1:
        next_word = text_concatenated[word_count+1]
    #print("previous_word", previous_word)
    #print("word", word)
    #print("next_word", next_word)

    assert(text_concatenated[word_count] == word)
    resulting_vector = None
    vectorized_long_format = vectorized_data[word_count].toarray()[0]
    #print("vectorized_long_format", vectorized_long_format)
    len_2_occ = len(vectorized_long_format)

    #print("vectorized_long_format", len(vectorized_long_format))
    #print("vectorized_vocabulary_long_format", len(vectorized_vocabulary_long_format))
    
    if previous_word:
            previous_long_vector = vectorized_data[word_count-1].toarray()[0]
    else:
        previous_long_vector = [2] * len_2_occ

    if next_word:
            next_long_vector = vectorized_data[word_count+1].toarray()[0] 
    else:
        next_long_vector = [2] * len_2_occ

    #print("Feature vector length without context", len(vectorized_long_format))
    resulting_vector = np.concatenate((vectorized_long_format, previous_long_vector, next_long_vector))
    #print("Feature vector length with context", len(resulting_vector))

    if use_word2vec:
        word2vecvector = word2vecwrapper.get_vector(word)

        #print(word2vecvector)
        if previous_word:
            previous_vector = word2vecwrapper.get_vector(previous_word)
        else:
            previous_vector = [2] * 300

        if next_word:
            next_vector = word2vecwrapper.get_vector(next_word)
        else:
            next_vector = [2] * 300

        resulting_vector = np.concatenate((resulting_vector, word2vecvector, previous_vector, next_vector))
        #print("Feature vector length with word2vec info", len(resulting_vector))

    #print(word)
    #print(list(resulting_vector))
    #print()

    return resulting_vector



def vectorize_data(text_vector_labelled, text_vector_unlabelled, label_vector_labelled, class_dict, classes, use_word2vec, model_path):

    word2vecwrapper = Word2vecWrapper(model_path)

    if len(text_vector_unlabelled) <= 0:
        print("There is no more unlabelled data available. System will exit")
        exit(1)
    text_concatenated_labelled = np.concatenate(text_vector_labelled)

    # Create a vectorizer for all words that are included (fit on training data)
    tf_vectorizer = CountVectorizer(binary = True, min_df=2) # only include features that have occurred at least twice in the labelled data
    vectorized_data_labelled = tf_vectorizer.fit_transform(text_concatenated_labelled)
    
    #print("Created vectorizer for unlabelled data")
    
    # Then, use the vectorizer to create vectorized data
    # Labelled
    result_X_labelled = []
    current_sentence_X = []
    result_y_labelled= []

    word_count = 0
    for label, text in zip(label_vector_labelled, text_vector_labelled):
        for i, t, l in zip(range(0, len(text)), text, label):
            resulting_vector = get_resulting_x_vector(t, word_count, text_concatenated_labelled, vectorized_data_labelled, i, len(text), use_word2vec, word2vecwrapper)
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


    #return_text_vector_train = [" ".join(el) for el in text_vector_train]
    print("Read labelled data, len: ", len(result_X_labelled), len(result_y_labelled), len(text_vector_labelled))

    #Unlabelled data
    text_concatenated_unlabelled = np.concatenate(text_vector_unlabelled)
    vectorized_data_unlabelled = tf_vectorizer.transform(text_concatenated_unlabelled)

    vectorized_data_test_vocabulary = None

    result_X_unlabelled = []
    current_sentence_X = []

    word_count = 0
    for text in text_vector_unlabelled:

        for i, t in zip(range(0, len(text)), text):
            resulting_vector = get_resulting_x_vector(t, word_count, text_concatenated_unlabelled, vectorized_data_unlabelled, i, len(text), use_word2vec, word2vecwrapper)
            current_sentence_X.append(resulting_vector)
            word_count = word_count + 1

        if len(current_sentence_X) != 0:
            result_X_unlabelled.append(current_sentence_X)
            current_sentence_X = []

    #return_text_vector_test = [" ".join(el) for el in text_vector_test]

    print("Read unlabelled data, len: ", len(result_X_unlabelled), len(text_vector_unlabelled))

    result_X_labelled_np = np.array([np.array(xi) for xi in result_X_labelled])
    result_X_unlabelled_np = np.array([np.array(xi) for xi in result_X_unlabelled])

    result_y_labelled_np = np.array([np.array(yi) for yi in result_y_labelled])

    text_vector_labelled_np = np.array([np.array(ti) for ti in text_vector_labelled])
    text_vector_unlabelled_np = np.array([np.array(ti) for ti in text_vector_unlabelled])

    word2vecwrapper.end()

    return result_X_labelled_np, result_X_unlabelled_np, result_y_labelled_np, text_vector_labelled_np, text_vector_unlabelled_np



    


