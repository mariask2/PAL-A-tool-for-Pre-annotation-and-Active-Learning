import sys
import time
import os
import shutil
import importlib
import argparse

import transform_to_brat_format
import vectorize_data
import classify_and_select
import default_settings


def check_frequency_of_labels(labelled_label_vector, classes):
    """
    If the minority classes in the properties are not in the labelled data, the
    pre-annotation and active learning doesn't work. This function checks that
    they exist in the annotated data, and otherwise the program terminates.
    """
    freq_dict = {}
    for cl in classes:
        freq_dict[cl] = 0

    for labels in labelled_label_vector:
        for label in labels:
            freq_dict[label] = freq_dict[label] + 1

    for label, freq in freq_dict.items():
        if freq == 0:
            print("There are no occurrences of the label " + label + \
                      " in the labelled data. Remove this category from the settings file.")
            exit(1)

def select_new_data(properties, project_path, word2vecwrapper):
    """
    select_new_data performs the active learning and pre-annotation.

    Uses the vectorize_data module to read and vectorize data
    and the classify_and_select to select and pre-annotate data.
    Thereafter, writes the pre-annotated and selected data in csv and brat format
    and updates the pool of unlabelled data.

    :param properties: an instance of PropertiesContainer which contains the settings for running the active learning and pre-annotation
    :param path_slash_format: a string containing the path to the folder with the data
    :param word2vecwrapper: an instance of the vectorize_data.Word2vecWrapper class (to use for incorporating additional features)
    """

    print()
    print("**************************************************************")
    print("* Start selection and pre-annotation of new training samples *")
    print("**************************************************************")

    unlabelled_data_dir_for_project = os.path.join(project_path, properties.unlabelled_data_dir)
    unlabelled_data_path = os.path.join(unlabelled_data_dir_for_project, properties.unlabelled_data_file)
    
    # Classes
    classes = properties.minority_classes[:]
    classes.append(properties.outside_class)
    print("Classes to use:\t" + str(classes))
    
    labelled_data_dir_for_project = os.path.join(project_path, properties.labelled_data_dir)
    labelled_text_vector, labelled_label_vector, label_dict =  \
        vectorize_data.read_file_labelled_data(labelled_data_dir_for_project, properties.data_file_extension, \
                                                   properties.minority_classes, properties.outside_class)

    check_frequency_of_labels(labelled_label_vector, classes)

    unlabelled_text_vector = vectorize_data.read_file_unlabelled_data(unlabelled_data_path)
    print("Read unlabelled data from:\t" + unlabelled_data_path)
    
    X_labelled_np, X_unlabelled_np, y_labelled_np, text_vector_labelled_np, text_vector_unlabelled_np, \
        current_word_vectorizer, context_word_vectorizer = \
        vectorize_data.vectorize_data(labelled_text_vector, unlabelled_text_vector, labelled_label_vector, \
                                          label_dict, use_word2vec = properties.whether_to_use_word2vec, \
                                          number_of_previous_words = properties.number_of_previous_words, \
                                          number_of_following_words = properties.number_of_following_words, \
                                          use_current_word_as_feature = properties.use_current_word_as_feature, \
                                          min_df_current = properties.min_df_current,  \
                                          min_df_context = properties.min_df_context, \
                                          word2vecwrapper = word2vecwrapper, \
                                          current_word_vocabulary = properties.current_word_vocabulary, \
                                          context_word_vocabulary = properties.context_word_vocabulary)    

    to_select_X, new_unlabelled_x, to_select_text, new_sentences_unlabelled, predicted_for_selected = \
        classify_and_select.get_new_data(X_labelled_np, X_unlabelled_np, y_labelled_np, text_vector_labelled_np, \
                                             text_vector_unlabelled_np, label_dict, properties.minority_classes, \
                                             properties.nr_of_samples, properties.maximum_samples_to_search_among, \
                                             properties.outside_class, properties.beginning_prefix, \
                                             properties.inside_prefix, properties.inactive_learning, \
                                             properties.max_iterations, properties.prefer_predicted_chunks, \
                                             properties.model_type, properties.use_cross_validation, \
                                             properties.nr_of_cross_validation_splits, \
                                             properties.c_value)

    tolabel_data_dir_for_project = os.path.join(project_path, properties.tolabel_data_dir)
    if not os.path.exists(tolabel_data_dir_for_project):
        os.mkdir(tolabel_data_dir_for_project)

    to_annotate_file_path = os.path.join(tolabel_data_dir_for_project, time.strftime("tolabel_%Y%m%d_%H%M%S.csv"))
    to_annotate_file = open(to_annotate_file_path, "w")

    label_dict_inv = {v: k for k, v in label_dict.items()}

    # Create the file for annotation data
    for texts, labels in zip(to_select_text, predicted_for_selected):
        try:
            assert(len(texts) == len(labels))
        except AssertionError:
            print("different length on labels and text")
            exit(1)
        for text, label in zip(texts, labels):
            to_annotate_file.write("\t".join([text, label_dict_inv[label]]) + "\n")
        to_annotate_file.write("\n")
    to_annotate_file.close()

    # Rewrite the file with unlabelled data
    old_unlabelled_data_path = os.path.join(unlabelled_data_dir_for_project, \
                                                time.strftime("unlabelled_%Y%m%d_%H%M%S.csv"))
    shutil.move(unlabelled_data_path,  old_unlabelled_data_path)
    unlabelled_data_path_file = open(unlabelled_data_path, "w")

    # Create the file for annotation data
    for texts in new_sentences_unlabelled:
        for text in texts:
            unlabelled_data_path_file.write(text + "\n")
        unlabelled_data_path_file.write("\n")
    unlabelled_data_path_file.close()

    interesting_tags = []
    for tag in properties.minority_classes:
        if tag.startswith(properties.beginning_prefix):
            interesting_tags.append(tag[len(properties.beginning_prefix):])
    print("interesting_tags", interesting_tags)

    transform_to_brat_format.transform(to_annotate_file_path, tolabel_data_dir_for_project, \
                                           interesting_tags, properties.outside_class, \
                                           properties.beginning_prefix)


def load_properties(parser):
    """
    load_properties reads the command line arguments

    param: parser: an instance of argparse.ArgumentParser
    :returns properties: an instance of PropertiesContainer which contains the settings for running the active learning and pre-annotation                                                                                 
    :returns path_slash_format: a string containing the path to the folder with the data          
    """

    parser.add_argument('--project', action='store', dest='project_path', \
                            help='The path, separated by dots, to where the project i located. For instance: data.example_project')
    args = parser.parse_args()
    if not args.project_path:
        print("The argument '--project' with the path to the data needs to be given")
        exit(1)

    print(args.project_path)

    SETTINGS  = "settings"
    path_slash_format = ""
    for path_part in args.project_path.split("."):
        path_slash_format = os.path.join(path_slash_format, path_part)
    
    if not os.path.exists(path_slash_format):
        print("The directory '" + str(path_slash_format) + "' i.e., the directory matching the project path given '" + \
                  str(args.project_path) + "', does not exist")
        exit(1)

    if not os.path.exists(os.path.join(path_slash_format, SETTINGS + ".py")):
        print("The directory '" + str(path_slash_format) + "' does not have a " + SETTINGS + ".py file.")
        exit(1)

    properties = importlib.import_module(args.project_path + "." + SETTINGS)

    properties_container = PropertiesContainer(properties)
    return properties_container, path_slash_format



class PropertiesContainer:
    """
    PropertiesContainer

    A class containing the properties for running the pre-annotation and active learning, 
    and for checking the validity of the properties.
    """

    def check_properties(self): 
        """
        check_properties 
        Checks that the names of the minority classes matches with the properties of the 
        prefixes to use and that the outside class is not a minority class 
        """
        if self.outside_class in self.minority_classes:
            print("Error in properties file (setup.py), outside_class " + self.outside_class + " should not be in the list of minority_classes")
            exit(1)
        if len(self.beginning_prefix) != len(self.inside_prefix):
            print("Error in properties file (setup.py). The beginning_prefix and the inside_prefix must have the same length (" \
                  + self.beginning_prefix  + " and " + self.inside_prefix + ").")
            exit(1)
    
        for minority_class in self.minority_classes:
            if not (minority_class.startswith(self.beginning_prefix) or minority_class.startswith(self.inside_prefix)):
                print("The class " + minority_class + " does not start with the beginning_prefix or the inside_prefix.")
                exit(1)

    def __init__(self, properties):
        """
        :params properties: a python model including properties (retrieved by importlib.import_module(<PATH>))
        """

        try:
            self.minority_classes = properties.minority_classes
        except AttributeError:
            print("Settings file lacks the property 'minority_classes'.")
            exit(1)
        try:    
            self.nr_of_samples = properties.nr_of_samples
        except AttributeError:    
            self.nr_of_samples = default_settings.nr_of_samples

        try:    
            self.maximum_samples_to_search_among = properties.maximum_samples_to_search_among
        except AttributeError:
            self.maximum_samples_to_search_among = default_settings.maximum_samples_to_search_among

        try:    
            self.number_of_previous_words = properties.number_of_previous_words
        except AttributeError:
            self.number_of_previous_words = default_settings.number_of_previous_words 
        try:
            self.number_of_following_words = properties.number_of_following_words
        except AttributeError:
            self.number_of_following_words = default_settings.number_of_following_words 

        try:    
            self.use_current_word_as_feature = properties.use_current_word_as_feature
        except AttributeError:
            self.use_current_word_as_feature = default_settings.use_current_word_as_feature

        try:    
            self.min_df_current = properties.min_df_current
        except AttributeError:
            self.min_df_current = default_settings.min_df_current 
            
        try:    
            self.min_df_context = properties.min_df_context
        except AttributeError:
            self.min_df_context = default_settings.min_df_context 

        try:  
            self.whether_to_use_word2vec = properties.whether_to_use_word2vec
        except AttributeError: 
            self.whether_to_use_word2vec = default_settings.whether_to_use_word2vec 

        if self.whether_to_use_word2vec:    
            try:    
                self.model_path = properties.model_path
            except AttributeError:    
                print("The settings 'whether_to_use_word2vec' is set to True, but there is no 'model_path' showing where the model is")
                exit(1)
            try:    
                self.semantic_vector_length = properties.semantic_vector_length
            except AttributeError:
                print("The settings 'whether_to_use_word2vec' is set to True, but there is no 'semantic_vector_length' showing how large the vector is")
                exit(1)
        else:
            if hasattr(properties, 'model_path') or hasattr(properties, 'semantic_vector_length'):
                print("WARNING: You have given a 'model_path' or a 'semantic_vector_length', but 'whether_to_use_word2vec' is set to 'False'")
            self.model_path = None
            self.semantic_vector_length = None

        try:    
            self.inactive_learning = properties.inactive_learning
        except AttributeError:
            self.inactive_learning = default_settings.inactive_learning 

        try:    
            self.prefer_predicted_chunks = properties.prefer_predicted_chunks
        except AttributeError:
            self.prefer_predicted_chunks = default_settings.prefer_predicted_chunks 

        try:    
            self.max_iterations = properties.max_iterations
        except AttributeError:
            self.max_iterations = default_settings.max_iterations 

        try:    
            self.model_type = properties.model_type
        except AttributeError:
            self.model_type = default_settings.model_type 

        try:    
            self.save_model = properties.save_model
        except AttributeError:
            self.save_model = default_settings.save_model 

        try:    
            self.current_word_vocabulary = properties.current_word_vocabulary
        except AttributeError:
            self.current_word_vocabulary = default_settings.current_word_vocabulary 

        try:    
            self.context_word_vocabulary = properties.context_word_vocabulary
        except AttributeError:    
            self.context_word_vocabulary = default_settings.context_word_vocabulary 

        if (self.context_word_vocabulary != False and  not isinstance(self.context_word_vocabulary, str)) \
                or (self.current_word_vocabulary != False and  not isinstance(self.current_word_vocabulary, str)):
            raise ValueError("'context_word_vocabulary' and 'context_word_vocabulary' " + \
                                 "should either be False or a string indicating search path")
        try:    
            self.use_cross_validation = properties.use_cross_validation
        except AttributeError:
            self.use_cross_validation = default_settings.use_cross_validation 
        # Only implemented for NonStructuredLogisticRegression so far
        if self.use_cross_validation and self.model_type != classify_and_select.NonStructuredLogisticRegression:
            raise NotImplementedError("cross_validation is only implemented for NonStructuredLogisticRegression")

        try:    
            self.c_value = properties.c_value
        except AttributeError:
            self.c_value = default_settings.c_value 
        # If cross validation is chosen, this c_value is not used
        if self.use_cross_validation and hasattr(properties, 'c_value'):
            raise ValueError("If cross_validation is chosen, the c-value is not decided by the user 'no c_value attribute should be given'")
        try:
            self.labelled_data_dir = properties.labelled_data_dir
        except AttributeError:
            self.labelled_data_dir = default_settings.labelled_data_dir

        try:    
            self.data_file_extension = properties.data_file_extension
        except AttributeError: 
            self.data_file_extension = default_settings.data_file_extension 

        try:    
            self.unlabelled_data_dir = properties.unlabelled_data_dir
        except AttributeError: 
            self.unlabelled_data_dir = default_settings.unlabelled_data_dir 

        try:    
            self.unlabelled_data_file = properties.unlabelled_data_file
        except AttributeError: 
            self.unlabelled_data_file = default_settings.unlabelled_data_file 

        try:    
            self.tolabel_data_dir = properties.tolabel_data_dir
        except AttributeError: 
            self.tolabel_data_dir = default_settings.tolabel_data_dir 

        try:    
            self.evaluation_output_dir = properties.evaluation_output_dir
        except AttributeError: 
            self.evaluation_output_dir = default_settings.evaluation_output_dir 

        try:    
            self.separate_evaluation_output_dir = properties.separate_evaluation_output_dir
        except AttributeError: 
            self.separate_evaluation_output_dir = default_settings.separate_evaluation_output_dir 

        try:  
            self.separate_evaluation_data_dir = properties.separate_evaluation_data_dir
        except AttributeError: 
            self.separate_evaluation_data_dir = default_settings.separate_evaluation_data_dir 

        try:  
            self.saved_model_dir = properties.saved_model_dir
        except AttributeError: 
            self.saved_model_dir = default_settings.saved_model_dir 

        try:  
            self.beginning_prefix = properties.beginning_prefix
        except AttributeError: 
            self.beginning_prefix = default_settings.beginning_prefix 

        try:  
            self.inside_prefix = properties.inside_prefix
        except AttributeError: 
            self.inside_prefix = default_settings.inside_prefix 

        try:  
            self.outside_class = properties.outside_class
        except AttributeError: 
            self.outside_class = default_settings.outside_class 

        try:  
            self.negative_category = properties.negative_category
        except AttributeError: 
            self.negative_category = default_settings.negative_category

        try:  
            self.nr_of_cross_validation_splits = properties.nr_of_cross_validation_splits
        except AttributeError: 
            self.nr_of_cross_validation_splits = default_settings.nr_of_cross_validation_splits 

        if not self.use_cross_validation and hasattr(properties, 'nr_of_cross_validation_splits'):
            raise ValueError("If 'use_cross_validation' is False, there is no point of giving a 'nr_of_cross_validation_splits'")

        # Only implemented for NonStructuredLogisticRegression so far
        if self.model_type == classify_and_select.NonStructuredLogisticRegression and hasattr(properties, 'max_iterations'):
            raise NotImplementedError("The variable 'max_iterations' is not used for NonStructuredLogisticRegression")

        self.check_properties()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    properties_main, path_slash_format_main = load_properties(parser)
    word2vecwrapper = vectorize_data.Word2vecWrapper(properties_main.model_path, properties_main.semantic_vector_length)

    select_new_data(properties_main, path_slash_format_main, word2vecwrapper)


    



