"""
Functions for simulation and evaluation
Not all functions are yet documented here.
But the ones required for loading a saved model and
classifying them are documented.
Those are, do_load_model, classify_from_loaded_model and
get_sentence_certainty_score
"""

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import sys
import time
import os
import shutil
import importlib
import argparse
import glob
from sklearn.externals import joblib
import numpy
import random
import math

import transform_to_brat_format
import vectorize_data
import classify_and_select
import active_learning_preannotation
import simple_tokenizer

#from classify_and_select import StructuredModelFrankWolfeSSVM

def get_binary_list(lst, category, outside_class, inside_prefix, beginning_prefix):
    print("category", category)
    category_inside = inside_prefix + category[len(beginning_prefix):]
    print("category_inside", category_inside)
    result_lst = []
    for res in lst:
        if category in res or category_inside in res:
            result_lst.append(category)
        else:
            result_lst.append(outside_class)
    return result_lst


def get_result_tag_format(results, model):
    results_tag_format = []
    for prediction in results:
        tag_format = [model.inv_label_dict[el] for el in prediction]
        results_tag_format.append(tag_format)
    return results_tag_format


def evaluate_category(category, test_sentences, test_results, expected_results, outside_class, project_path, \
                          output_dir, inside_prefix, beginning_prefix, c_value, model_type, whether_to_use_word2vec, \
                          whether_to_use_clustering, training_data_size):
    test_results_binary = get_binary_list(test_results, category, outside_class, inside_prefix, beginning_prefix)
    expected_results_binary = get_binary_list(expected_results, category, outside_class, inside_prefix, beginning_prefix)

    precision = precision_score(y_true=expected_results_binary, y_pred=test_results_binary, average='binary', pos_label=category)
    recall = recall_score(y_true=expected_results_binary, y_pred=test_results_binary, average='binary', pos_label=category)
    f1 = f1_score(y_true=expected_results_binary, y_pred=test_results_binary, average='binary', pos_label=category)

    output_path = os.path.join(project_path, output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    base_name = category + "_" + model_type.__name__ + "_c_" + str(c_value) + "_word2vec_" + str(whether_to_use_word2vec) + \
         "_clustering_" + str(whether_to_use_clustering)
    output_file_res_path = os.path.join(output_path, base_name + "_res.txt") 
    output_file_data_path = os.path.join(output_path, base_name + "_data.csv") 
    output_file_res = open(output_file_res_path, "w")
    output_file_data = open(output_file_data_path, "w")

    output_file_res.write("\t".join(["precision", str(precision)]) + "\n")
    output_file_res.write("\t".join(["recall", str(recall)]) + "\n")
    output_file_res.write("\t".join(["f1_score", str(f1)]) + "\n")
    output_file_res.write("\t".join(["training_data_size", str(training_data_size)]) + "\n")

    print("\t".join(["precision", str(precision)]))
    print("\t".join(["recall", str(recall)]))
    print("\t".join(["f1_score", str(f1)]))

    
    predicted_entities = 0
    relevant_entities = 0
    for sentence, result_full, expected_full, result, expected in \
            zip(test_sentences, test_results, expected_results, test_results_binary, expected_results_binary):
        result_type = "RES:OUTSIDE"
        if result == category and expected == category:
            result_type = "RES:TP"
            predicted_entities = predicted_entities + 1
            relevant_entities = relevant_entities + 1
        if result == category and expected != category:
            result_type = "RES:FP"
            predicted_entities = predicted_entities + 1
        if result != category and expected == category:
            result_type = "RES:FN"
            relevant_entities = relevant_entities + 1

        output_file_data.write("\t".join([" ".join(sentence), str(result_full), str(expected_full), result, expected, result_type]) + "\n")

        
    if predicted_entities > 0:
        precision_conf_interval = 1.96*math.sqrt(precision*(1-precision)/predicted_entities)
    else:
        precision_conf_interval = 1 # No predicted entities                                                                                                             

    if relevant_entities > 0:
        recall_conf_interval = 1.96*math.sqrt(recall*(1-recall)/relevant_entities)
    else:
        recall_conf_interval = 1 # No predicted entities
    output_file_res.write("\t".join(["precision_conf_interval", str(precision_conf_interval)]) + "\n")
    output_file_res.write("\t".join(["recall_conf_interval", str(recall_conf_interval)]) + "\n")
    
    output_file_res.close()
    output_file_data.close()

def do_save_model(properties_eval, project_path, category, model, properties_file_name):
    output_path = os.path.join(project_path, properties_eval.saved_model_dir, category)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = os.path.join(output_path, category + "_model")
    joblib.dump(model, filename, compress=9)
    print("Saved model with name " + filename)

    base = os.path.basename(properties_file_name)
    f = open(properties_file_name)
    settings_content = f.read()
    f.close()
    used_settings = open(os.path.join(output_path, base), "w")
    used_settings.write(settings_content)
    used_settings.close()

    return filename


def do_load_model(properties, model_path, category):
    """
    Loads a saved model into memory
    """
    #output_path = os.path.join(project_path, properties.saved_model_dir, category)
    filename = os.path.join(model_path, category + "_model")
    model = joblib.load(filename)
    print("Loaded model with name " + filename)
    if model.__class__ != properties.model_type:
        print("The saved model is not of the same type as the model in the properties file")
        raise ValueError("The saved model is not of the same type as the model in the properties file")
    return model


"""
def do_load_model(properties, project_path, category):
    output_path = os.path.join(project_path, properties.saved_model_dir, category)
    filename = os.path.join(output_path, category + "_model")
    model = joblib.load(filename)
    print("Loaded model with name " + filename)
    if model.__class__ != properties.model_type:
        print("The saved model is not of the same type as the model in the properties file")
        raise ValueError("The saved model is not of the same type as the model in the properties file")
    return model
"""

# When only one category is allowed. Check that there is only one category, and return it
def get_category(properties):
    categories = [el for el in properties.minority_classes if el.startswith(properties.beginning_prefix)]
    if len(categories) > 1:
        print("there is only support for one category")
        print("Here, the following categories were given: " + str(categories))
        raise ValueError("there is only support for one category. Here, the following categories were given: " + str(categories))
    category_with_prefix = categories[0]
    category = category_with_prefix.replace(properties.beginning_prefix, "") # Just want the category name, without the beginning prefix
    return category_with_prefix, category

def get_category_inside(properties):
    categories = [el for el in properties.minority_classes if el.startswith(properties.inside_prefix)]
    if len(categories) > 1:
        print("there is only support for one category")
        print("Here, the following categories were given: " + str(categories))
        raise ValueError("there is only support for one category. Here, the following categories were given: " + str(categories))
    category_with_inside_prefix = categories[0]
    return category_with_inside_prefix


def classify_from_loaded_model(properties, project_path, text_vector, word2vecwrapper, model=None, result_X_unlabelled_np=None, text_vector_unlabelled_np = None):
    """
    Classifies from saved model
    It is possible to submit pre-vectorized data to the function 
    (in result_X_unlabelled_np and text_vector_unlabelled_np,
    but if they are None, the data in text_vector is instead used
    to vectorize the data

    If the mode is None, a model is loaded from the data in properies
    
    """
    start_time = time.time()
    time_file = open("time_taken.txt", "w")
    time_file.write("len(text_vector) " + str(len(text_vector)))


    category_with_prefix, category = get_category(properties)
    category_with_inside_prefix = get_category_inside(properties)
    if not model:
        model = do_load_model(properties, project_path, category)

    text_tokenized = simple_tokenizer.simple_tokenize_list(text_vector)
    time_file.write(" ".join(["After tokenize", str(time.time() - start_time), '\n']))

    if result_X_unlabelled_np ==None or text_vector_unlabelled_np == None:
        result_X_unlabelled_np, text_vector_unlabelled_np = \
            vectorize_data.vectorize_unlabelled(text_tokenized, model.current_word_vectorizer, model.context_word_vectorizer, \
                                                    properties.whether_to_use_word2vec, properties.number_of_previous_words, \
                                                    properties.number_of_following_words, properties.use_current_word_as_feature, \
                                                    word2vecwrapper, properties.whether_to_use_clustering)

    time_file.write(" ".join(["After vectorize", str(time.time() - start_time), '\n']))


    results = model.predict(result_X_unlabelled_np)
    time_file.write(" ".join(["After predict", str(time.time() - start_time), '\n']))

    probabilities = model.predict_proba(result_X_unlabelled_np)
    time_file.write(" ".join(["After predict probabilities", str(time.time() - start_time), '\n']))

    #"probability" is the probability output from the classifier
    # in "word_probabilities_expanded", these have been mapped to
    # categories of the classifier
    # tag_format is the output of the classifier with the tags, e.g. ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    to_return = []
    for prediction, sentence, sentence_vectorized_np, probability, sentence_tokenized in \
            zip(results, text_vector, result_X_unlabelled_np, probabilities, text_tokenized):
        word_probabilities_expanded = []
        for word_probabilities in probability:
            class_prob_score_dict = {}
            for class_prob_score, i in zip(word_probabilities, range(0, len(model.inv_label_dict.keys()))):
                class_prob_score_dict[model.inv_label_dict[i]] = class_prob_score
            word_probabilities_expanded.append(class_prob_score_dict)
        tag_format = [model.inv_label_dict[el] for el in prediction]
        
        # If there is one occurrence of the category
        if category_with_prefix in tag_format or category_with_inside_prefix in tag_format: 
            binary_category = category
        else: 
            binary_category = properties.outside_class
        certainty_score = get_sentence_certainty_score(sentence_vectorized_np, prediction, model)
        to_return.append((tag_format, binary_category, certainty_score, sentence, word_probabilities_expanded))
        certainty_score = None
        binary_category = None
        tag_format = None
    time_file.write(" ".join(["After loop", str(time.time() - start_time), '\n']))    
    return to_return, result_X_unlabelled_np, text_vector_unlabelled_np



def get_sentence_certainty_score(xi, yi, model):
    """
    Returns the sentence-level certainty score, 
    which is based on the token-level certainty score.
    The algorithm for this is as follows:
    If a minority category has been predicted for at least one token, 
    return the most certain of these
    If not, return the score for the most UNcertain of the majority 
    category predictions
    """
    probabilities = model.predict_proba([xi])
    probabilities_for_sentence = probabilities[0] # Only called it for one sentence, that's why the index is needed

    only_majority = True
    for el in yi:
        if el != model.majority_class:
            only_majority = False
            
    if only_majority:
        min_score = float("inf")
        # everything is predicted to belong to the majority category
        # return the score, where this prediction is least certain
        for word_prob in probabilities_for_sentence:
            #print("word_prob only majority", word_prob)
            #print("word_prob[self.majority_class]", word_prob[self.majority_class])
            if word_prob[model.majority_class] < min_score: 
                    min_score = word_prob[model.majority_class]
        return min_score
    else:
        max_score = float("-inf")
            # there is a minority category prediction been made
            # return the score, where the classifier is most certain of this category
        for word_prob, prediction in zip(probabilities_for_sentence, yi):
        #print("word_prob minority", word_prob)
            if prediction != model.majority_class:
                # i.e. the same as probability for the beginning category + probability for the inside category
                prob_not_majority = 1 - word_prob[model.majority_class] 
                #print("prob_not_majority", prob_not_majority)
                if prob_not_majority > max_score:
                    max_score = prob_not_majority
        return max_score      


def train_and_evaluate_model_cross_validation(properties, project_path, word2vecwrapper, cross_validation_properties):
    print()
    print("**************************************************************")
    print("* Start classification and evaluation samples *")
    print("**************************************************************")

    # Classes
    classes = properties.minority_classes[:]
    classes.append(properties.outside_class)
    print("Classes to use:\t" + str(classes))
    
    labelled_data_dir_for_project = os.path.join(project_path, properties.labelled_data_dir)
    labelled_text_vector, labelled_label_vector, label_dict = \
        vectorize_data.read_file_labelled_data(labelled_data_dir_for_project, properties.data_file_extension, \
                                                   properties.minority_classes, properties.outside_class)
    
    active_learning_preannotation.check_frequency_of_labels(labelled_label_vector, classes)

    skf = StratifiedKFold(n_splits=cross_validation_properties.nr_of_cross_validation_splits_for_evaluation,\
                          shuffle = True, random_state = 3)

    # need to input a vector of the same length as labelled_label_vector, so just constuct on only with zeros
    empty_y = [0 for el in labelled_label_vector]

    test_sentences = []
    test_results = []
    expected_results = []

    foldnr = 0
    for train_index, test_index in skf.split(empty_y, empty_y):
        print("foldnr", foldnr)
     
        x_train_sentences = [ vec for (i, vec) in enumerate(labelled_text_vector) if i in train_index]
        y_train = [ vec for (i, vec) in enumerate(labelled_label_vector) if i in train_index]

        x_test_sentences = [ vec for (i, vec) in enumerate(labelled_text_vector) if i in test_index]
        y_test = [ vec for (i, vec) in enumerate(labelled_label_vector) if i in test_index]

        active_learning_preannotation.check_frequency_of_labels(y_train, classes)

        X_train_np, X_test_np, y_train_np, text_vector_train_np, text_vector_text_test_np, \
            current_word_vectorizer, context_word_vectorizer = \
            vectorize_data.vectorize_data(text_vector_labelled = x_train_sentences, \
                                          text_vector_unlabelled = x_test_sentences, \
                                            label_vector_labelled = y_train, \
                                            class_dict = label_dict, \
                                            use_word2vec = properties.whether_to_use_word2vec, \
                                            number_of_previous_words = properties.number_of_previous_words,\
                                            number_of_following_words = properties.number_of_following_words, \
                                            use_current_word_as_feature = properties.use_current_word_as_feature, \
                                            min_df_current = properties.min_df_current, \
                                            min_df_context = properties.min_df_context, \
                                            max_df_current = properties.max_df_current, \
                                            max_df_context = properties.max_df_context, \
                                            word2vecwrapper = word2vecwrapper, \
                                            current_word_vocabulary = properties.current_word_vocabulary, \
                                            context_word_vocabulary = properties.context_word_vocabulary,\
                                            use_clustering = properties.whether_to_use_clustering)

        model = properties.model_type(label_dict, properties.minority_classes, properties.outside_class, properties.beginning_prefix, \
                                          properties.inside_prefix, properties.max_iterations, properties.use_cross_validation, \
                                          properties.nr_of_cross_validation_splits, properties.c_value)

        test_sentences.extend(x_test_sentences)
        
        print("Starts to train")
        model.fit(X_train_np, y_train_np)
        #print("Score", model.score(X_train_np, y_train_np))

        print("Starts to predict")
        results = model.predict(X_test_np)
        foldnr = foldnr + 1

        results_tag_format = get_result_tag_format(results, model)
        test_results.extend(results_tag_format)
        expected_results.extend(y_test)
        
    for category in [el for el in label_dict.keys() if el.startswith(properties.beginning_prefix)]:
        evaluate_category(category, test_sentences, test_results, expected_results, properties.outside_class, project_path, \
                              cross_validation_properties.evaluation_output_dir, properties.inside_prefix, properties.beginning_prefix, \
                              properties.c_value, properties.model_type, properties.whether_to_use_word2vec, properties.whether_to_use_clustering, \
                              len(labelled_text_vector))




def train_and_evaluate_model_against_evaluation_data(properties, project_path, word2vecwrapper, properties_eval, properties_file_name):
    print()
    print("***********************************************************************")
    print("* Start classification and evaluation samples against evaluation data*")
    print("***********************************************************************")

    # Classes
    classes = properties.minority_classes[:]
    classes.append(properties.outside_class)
    print("Classes to use:\t" + str(classes))
    
    labelled_data_dir_for_project = os.path.join(project_path, properties.labelled_data_dir)
    labelled_text_vector, labelled_label_vector, label_dict = \
        vectorize_data.read_file_labelled_data(labelled_data_dir_for_project, properties.data_file_extension, \
                                                   properties.minority_classes, properties.outside_class)

    active_learning_preannotation.check_frequency_of_labels(labelled_label_vector, classes)

    category_with_prefix, category = get_category(properties)
    print("Performing evaluation for the category " + category)

    separate_evaluation_base_path = os.path.join(properties_eval.separate_evaluation_data_dir, category)
    positive_path = os.path.join(project_path, separate_evaluation_base_path, category)
    negative_path = os.path.join(project_path, separate_evaluation_base_path, properties_eval.negative_category)

    expected_results = []
    positive_sent = []
    for pos_file in glob.glob(os.path.join(positive_path, "*.txt")):
        positive_sent.extend(simple_tokenizer.simple_tokenize(pos_file))
        expected_results.append([category_with_prefix])

    negative_sent = []
    for neg_file in glob.glob(os.path.join(negative_path, "*.txt")):
        negative_sent.extend(simple_tokenizer.simple_tokenize(neg_file))
        expected_results.append([properties.outside_class])

    evaluation_data = positive_sent + negative_sent

    # The unlabelled data sent to the vectorizer is here the evaluation data 
    X_train_np, X_test_np, y_train_np, text_vector_train_np, text_vector_text_test_np, \
        current_word_vectorizer, context_word_vectorizer = \
        vectorize_data.vectorize_data(text_vector_labelled = labelled_text_vector, \
                                    text_vector_unlabelled = evaluation_data, \
                                        label_vector_labelled = labelled_label_vector, \
                                        class_dict = label_dict, \
                                        use_word2vec = properties.whether_to_use_word2vec, \
                                        number_of_previous_words = properties.number_of_previous_words, \
                                        number_of_following_words = properties.number_of_following_words, \
                                        use_current_word_as_feature = properties.use_current_word_as_feature, \
                                        min_df_current = properties.min_df_current, \
                                        min_df_context = properties.min_df_context, \
                                        max_df_current = properties.max_df_current, \
                                        max_df_context = properties.max_df_context, \
                                        word2vecwrapper = word2vecwrapper, \
                                        current_word_vocabulary = properties.current_word_vocabulary, \
                                        context_word_vocabulary = properties.context_word_vocabulary, \
                                        use_clustering = properties.whether_to_use_clustering)

    model = properties.model_type(label_dict, properties.minority_classes, properties.outside_class, properties.beginning_prefix, \
                                      properties.inside_prefix, properties.max_iterations, properties.use_cross_validation, \
                                      properties.nr_of_cross_validation_splits, properties.c_value)

    print("Starts to train")
    model.fit(X_train_np, y_train_np)

    if properties_eval.save_model:
        model.current_word_vectorizer = current_word_vectorizer
        model.context_word_vectorizer = context_word_vectorizer

        print("model.minority_classes_index", model.minority_classes_index)

        
        for c_nr in range(0, len(model.minority_classes_index)):
            category_name =  model.inv_label_dict[c_nr]
            if category_name.startswith(model.beginning_prefix):
                model.beginning_category = c_nr
            if category_name.startswith(model.inside_prefix):
                model.inside_category = c_nr
        print("model.beginning_category", model.beginning_category)

        savedmodel_filename = do_save_model(properties_eval, project_path, category, model, properties_file_name)

    print("Starts to predict")
    results = model.predict(X_test_np)

    if properties_eval.save_model:
        output_path = os.path.join(project_path, properties_eval.saved_model_dir, category) 
        loaded_model = do_load_model(properties, output_path, category)
        loaded_results = loaded_model.predict(X_test_np)
        for res1, res2 in zip(results, loaded_results):
            if not numpy.array_equal(res1, res2):
                print(res1, res2)
                print("ERROR: Loaded model different from saved one")
                exit(1)
        loaded_model = None

    test_results = get_result_tag_format(results, model)    

    evaluate_category(category_with_prefix, evaluation_data, test_results, expected_results, \
                          properties.outside_class, project_path, properties_eval.separate_evaluation_output_dir, \
                          properties.inside_prefix, properties.beginning_prefix, properties.c_value, properties.model_type, \
                          properties.whether_to_use_word2vec, properties.whether_to_use_clustering, len(labelled_text_vector))


##################################################
# For simulating different sizes of training data
##################################################

def evaluate_category_different_data_sizes(category, test_sentences, test_results, expected_results, outside_class, project_path, \
                                               output_dir, inside_prefix, beginning_prefix, parameters, cs, model_type,\
                                               whether_to_use_word2vec, data_size, selection_type, fold_nr):
    test_results_binary = get_binary_list(test_results, category, outside_class, inside_prefix, beginning_prefix)
    expected_results_binary = get_binary_list(expected_results, category, outside_class, inside_prefix, beginning_prefix)

    precision = precision_score(y_true=expected_results_binary, y_pred=test_results_binary, average='binary', pos_label=category)
    recall = recall_score(y_true=expected_results_binary, y_pred=test_results_binary, average='binary', pos_label=category)
    f1 = f1_score(y_true=expected_results_binary, y_pred=test_results_binary, average='binary', pos_label=category)

    print("\n---------")
    print("selection_type", selection_type)
    print("data_size", data_size)

    output_path_base = os.path.join(project_path, output_dir + "_simulate_active_learning", str(category))
    if not os.path.exists(output_path_base):
        os.makedirs(output_path_base)

    output_path_base_fold = os.path.join(output_path_base, str(fold_nr))
    if not os.path.exists(output_path_base_fold):
        os.makedirs(output_path_base_fold)

    output_path = os.path.join(output_path_base_fold, selection_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    base_name = category + "_" + selection_type + "_" + model_type.__name__ + "_word2vec_" + str(whether_to_use_word2vec) + "_" + str(data_size)
    output_file_res_path = os.path.join(output_path, base_name + "_res.txt") 
    #output_file_data_path = os.path.join(output_path, base_name + "_data.csv") 
    output_file_res = open(output_file_res_path, "w")
    #output_file_data = open(output_file_data_path, "w")

    output_file_res.write("\t".join(["precision", str(precision)]) + "\n")
    output_file_res.write("\t".join(["recall", str(recall)]) + "\n")
    output_file_res.write("\t".join(["f1_score", str(f1)]) + "\n")
    output_file_res.write("\t".join(["parameters", str(parameters).replace("\n", " ")]) + "\n")

    print("\t".join(["precision", str(precision)]))
    print("\t".join(["recall", str(recall)]))
    print("\t".join(["f1_score", str(f1)]))
    print("\t".join(["parameters", str(parameters).replace("\n", " ")]))
    print("\t".join(["cs", str(cs).replace("\n", " ")]))
    
    """
    for sentence, result_full, expected_full, result, expected in \
            zip(test_sentences, test_results, expected_results, test_results_binary, expected_results_binary):
        result_type = "RES:OUTSIDE"
        if result == category and expected == category:
            result_type = "RES:TP"
        if result == category and expected != category:
            result_type = "RES:FP"
        if result != category and expected == category:
            result_type = "RES:FN"

        output_file_data.write("\t".join([" ".join(sentence), str(result_full), str(expected_full), result, expected, result_type]) + "\n")
    """    
    output_file_res.close()
    #output_file_data.close()
    

def train_and_evaluate_simulation(x_train_sentences, y_train, x_test_sentences, y_test, label_dict, classes, properties, word2vecwrapper, \
                                      project_path, whether_to_use_word2vec, selection_type, fold_nr):
    print("start train and evaluate method")
    print("whether_to_use_word2vec", whether_to_use_word2vec)
    nr_of_samples = len(x_train_sentences)
    active_learning_preannotation.check_frequency_of_labels(y_train, classes)

    X_train_np, X_test_np, y_train_np, text_vector_train_np, text_vector_text_test_np, \
                      current_word_vectorizer, context_word_vectorizer = \
                      vectorize_data.vectorize_data(text_vector_labelled = x_train_sentences, \
                                            text_vector_unlabelled = x_test_sentences, \
                                            label_vector_labelled = y_train, \
                                            class_dict = label_dict, \
                                            use_word2vec = whether_to_use_word2vec, \
                                            number_of_previous_words = properties.number_of_previous_words,\
                                            number_of_following_words = properties.number_of_following_words, \
                                            use_current_word_as_feature = properties.use_current_word_as_feature, \
                                            min_df_current = properties.min_df_current, \
                                            min_df_context = properties.min_df_context, \
                                            max_df_current = properties.max_df_current, \
                                            max_df_context = properties.max_df_context,\
                                            word2vecwrapper = word2vecwrapper, \
                                            current_word_vocabulary = properties.current_word_vocabulary, \
                                            context_word_vocabulary = properties.context_word_vocabulary, \
                                            use_clustering = False)

    model = properties.model_type(label_dict, properties.minority_classes, properties.outside_class, properties.beginning_prefix, \
                                          properties.inside_prefix, properties.max_iterations, properties.use_cross_validation, \
                                          properties.nr_of_cross_validation_splits, properties.c_value)

    print(model)
    test_sentences = x_test_sentences  # not really need a new variable
        
    print("Starts to train")
    model.fit(X_train_np, y_train_np)

    print("Starts to predict")
    results = model.predict(X_test_np)

    results_tag_format = get_result_tag_format(results, model)
    test_results = results_tag_format # not really need a new variable
    expected_results = y_test  # not really need a new variable
        
    for category in [el for el in label_dict.keys() if el.startswith(properties.beginning_prefix)]:
        evaluate_category_different_data_sizes(category, test_sentences, test_results, expected_results, properties.outside_class, project_path, 
                                               properties.evaluation_output_dir, properties.inside_prefix, properties.beginning_prefix, \
                                                   model.get_params(), model.get_cs(), properties.model_type, whether_to_use_word2vec, nr_of_samples, \
                                                   selection_type, fold_nr)

def run_active_selection(labelled_text_vector, labelled_label_vector, train_index, x_test_sentences, y_test, label_dict, classes, \
                             properties, word2vecwrapper, project_path, seed_set_size, step_size, max_size, whether_to_use_word2vec, fold_nr):

    print("")
    print("Active selection")
    nr_of_samples = seed_set_size              
    x_train_sentences = [ vec for (i, vec) in enumerate(labelled_text_vector) if i in train_index[:seed_set_size]]
    y_train = [ vec for (i, vec) in enumerate(labelled_label_vector) if i in train_index[:seed_set_size]]
    used_indeces = [i for (i, vec) in enumerate(labelled_label_vector) if i in train_index[:seed_set_size]]
              #print("used_indeces", used_indeces)

    print("whether_to_use_word2vec", whether_to_use_word2vec)
    print("nr_of_samples + step_size", str(nr_of_samples + step_size))
    print("len(train_index)", str(len(train_index)))
    seed_set_run = True
    while nr_of_samples + step_size < len(train_index) and nr_of_samples + step_size < max_size:
        if not seed_set_run:
            x_pool_sentences = [ vec for (i, vec) in enumerate(labelled_text_vector) if i in train_index[seed_set_size:] and i not in used_indeces]
            print("len(used_indeces)", len(used_indeces))
            X_labelled_np, X_unlabelled_np, y_labelled_np, text_vector_labelled_np, text_vector_unlabelled_np, \
                          current_word_vectorizer, context_word_vectorizer = \
                          vectorize_data.vectorize_data(\
                text_vector_labelled = x_train_sentences, text_vector_unlabelled = x_pool_sentences,\
                    label_vector_labelled = y_train, \
                                    class_dict = label_dict, use_word2vec = whether_to_use_word2vec, \
                                        number_of_previous_words = properties.number_of_previous_words, \
                                        number_of_following_words = properties.number_of_following_words, \
                                        use_current_word_as_feature = properties.use_current_word_as_feature, \
                                        min_df_current = properties.min_df_current,  \
                                        min_df_context = properties.min_df_context, \
                                        max_df_current = properties.max_df_current, \
                                        max_df_context = properties.max_df_context, \
                                        word2vecwrapper = word2vecwrapper, \
                                        current_word_vocabulary = properties.current_word_vocabulary, \
                                        context_word_vocabulary = properties.context_word_vocabulary, \
                    use_clustering = False)

            to_select_X, new_unlabelled_x, to_select_text, new_sentences_unlabelled, predicted_for_selected = \
                          classify_and_select.get_new_data(X_labelled_np, X_unlabelled_np, y_labelled_np, text_vector_labelled_np, \
                                             text_vector_unlabelled_np, label_dict, properties.minority_classes, \
                                             step_size, properties.maximum_samples_to_search_among, \
                                             properties.outside_class, properties.beginning_prefix, \
                                             properties.inside_prefix, properties.inactive_learning, \
                                             properties.max_iterations, properties.prefer_predicted_chunks, \
                                             properties.model_type, properties.use_cross_validation, \
                                             properties.nr_of_cross_validation_splits, \
                                             properties.c_value)

                      #print("to_select_text", to_select_text)
            selected_set = set()
            for el in to_select_text:
                st = " ".join(el)
                selected_set.add(st)


            selected_indeces_text = [(i, " ".join(vec)) for (i, vec) in enumerate(labelled_text_vector) if " ".join(vec) in selected_set\
                                         and i in train_index[seed_set_size:] and i not in used_indeces]

            found_text_set = set() # if there are duplicates in training data, only find this sentence once
            selected_indeces = []

            # To remove duplicates
            how_much_too_long = len(selected_indeces_text) - step_size
            for selected_index, found_text in selected_indeces_text:
                if found_text not in found_text_set or how_much_too_long == 0:
                    selected_indeces.append(selected_index)
                    found_text_set.add(found_text)
                else:
                    how_much_too_long = how_much_too_long - 1
                    

            print("len(selected_indeces)", len(selected_indeces))
            if len(selected_indeces) != step_size:
                print("selected_indeces", selected_indeces)
                print("selected_set", selected_set)
                print("not enough selected")
                exit(1)
                
            used_indeces.extend(selected_indeces)
                  
        x_train_sentences = [ vec for (i, vec) in enumerate(labelled_text_vector) if i in used_indeces]
        y_train = [ vec for (i, vec) in enumerate(labelled_label_vector) if i in used_indeces]

        print("used_indeces", used_indeces)
        print("len(used_indeces)", len(used_indeces))
        print("len(x_train_sentences)", len(x_train_sentences))
        assert(len(used_indeces) == len(x_train_sentences))
        nr_of_samples = len(used_indeces)
        train_and_evaluate_simulation(x_train_sentences, y_train, x_test_sentences, y_test, label_dict, classes, \
                                          properties, word2vecwrapper, project_path, whether_to_use_word2vec, \
                                          selection_type = "active", fold_nr = fold_nr)
                  
        seed_set_run = False



def simulate_different_data_sizes(properties, simulation_properties, project_path, \
                                      word2vecwrapper, start_fold, end_fold):
    print()
    print("********************************************************************************")
    print("* Start classification and evaluation with the training data, where different *")
    print("* amounts of training data is available*")
    print("********************************************************************************")


    # Classes
    classes = properties.minority_classes[:]
    classes.append(properties.outside_class)
    print("Classes to use:\t" + str(classes))

    
    labelled_data_dir_for_project = os.path.join(project_path, properties.labelled_data_dir)
    labelled_text_vector, labelled_label_vector, label_dict = \
        vectorize_data.read_file_labelled_data(labelled_data_dir_for_project, properties.data_file_extension, \
                                                   properties.minority_classes, properties.outside_class)
    
    active_learning_preannotation.check_frequency_of_labels(labelled_label_vector, classes)

    category_with_prefix, category = get_category(properties)
    print("Performing evaluation for the category " + category)

    seed_set_size = simulation_properties.seed_set_size
    step_size = simulation_properties.step_size
    max_size = simulation_properties.max_size

    for fold_nr in range(start_fold, end_fold):
        print("\n\n")
        print("Running fold number " + str(fold_nr))
        print("--------")

        # one test_fold and one train_fold
        skf = StratifiedKFold(n_splits=simulation_properties.train_test_splits, shuffle = True, random_state = fold_nr)
        
        # need to input a vector of the same length as labelled_label_vector, so just constuct one only with zeros
        empty_y = [0 for el in labelled_label_vector]

        foldnr_skf = 0
        for train_index, test_index_all in skf.split(empty_y, empty_y):
            if foldnr_skf > 0:
                  break # only use the first fold, that is created by the StratifiedKFold class

            random.Random(fold_nr).shuffle(train_index)
            random.Random(fold_nr).shuffle(test_index_all)
            
            if len(test_index_all) <= simulation_properties.max_test_data_size:
                test_index = test_index_all
            else:
                test_index = test_index_all[:simulation_properties.max_test_data_size]

            # To save time when vectorizing (not to vectorize data that will not be searched among):
            if properties.maximum_samples_to_search_among != "all":
                if len(train_index) > properties.maximum_samples_to_search_among + simulation_properties.seed_set_size:
                    train_index = train_index[:properties.maximum_samples_to_search_among + simulation_properties.seed_set_size]
        
            print("test_index, len: ", len(test_index), " data: ", test_index)
            print("pool index, len: ", len(train_index)," data: ", train_index)

                       
            x_test_sentences = [ vec for (i, vec) in enumerate(labelled_text_vector) if i in test_index]
            y_test = [ vec for (i, vec) in enumerate(labelled_label_vector) if i in test_index]
     
            # Random selection of data
              
            print("\nRandom selection")
            print("------")
            nr_of_samples = seed_set_size
            while nr_of_samples < len(train_index) and nr_of_samples < max_size:
                print("Training with " + str(nr_of_samples) + " samples.")
                x_train_sentences = [ vec for (i, vec) in enumerate(labelled_text_vector) if i in train_index[:nr_of_samples]]
                y_train = [ vec for (i, vec) in enumerate(labelled_label_vector) if i in train_index[:nr_of_samples]]

                train_and_evaluate_simulation(x_train_sentences, y_train, x_test_sentences, y_test, label_dict, classes, \
                                                    properties, word2vecwrapper, project_path, whether_to_use_word2vec = False, \
                                                    selection_type = "random", fold_nr = fold_nr)
                train_and_evaluate_simulation(x_train_sentences, y_train, x_test_sentences, y_test, label_dict, classes, \
                                                    properties, word2vecwrapper, project_path, whether_to_use_word2vec = True, \
                                                    selection_type = "random", fold_nr = fold_nr)

                nr_of_samples = nr_of_samples + step_size
              
            print("\nActive selection")
            print("------")
            # Active selection of data
            run_active_selection(labelled_text_vector, labelled_label_vector, train_index, x_test_sentences, y_test, label_dict, classes, \
                                       properties, word2vecwrapper, project_path, seed_set_size, step_size, max_size, \
                                       whether_to_use_word2vec = False, fold_nr = fold_nr)
            run_active_selection(labelled_text_vector, labelled_label_vector, train_index, x_test_sentences, y_test, label_dict, classes, \
                                       properties, word2vecwrapper, project_path, seed_set_size, step_size, max_size, \
                                       whether_to_use_word2vec = True, fold_nr = fold_nr)

            ###
    

            # only use the first fold (could as well do a break here)
            foldnr_skf = foldnr_skf + 1
