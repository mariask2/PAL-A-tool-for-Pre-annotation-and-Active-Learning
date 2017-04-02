"""
"""
import argparse
import train_and_evaluate_model
import active_learning_preannotation
import vectorize_data
import os
import importlib

def do_cross_validation(parser):
    properties_main, path_slash_format, path_dot_format = active_learning_preannotation.load_properties(parser)
    word2vecwrapper = vectorize_data.Word2vecWrapper(properties_main.model_path, properties_main.semantic_vector_length)

    CROSS_VALIDATION_SETTINGS = "cross_validation_settings"
    if not os.path.exists(os.path.join(path_slash_format, CROSS_VALIDATION_SETTINGS + ".py")):
        print("The directory '" + str(path_slash_format) + "' does not have a " + CROSS_VALIDATION_SETTINGS + ".py file.")
        exit(1)
    properties_cross_validation = importlib.import_module(path_dot_format + "." + CROSS_VALIDATION_SETTINGS)
    properties_container_cross_validation = CrossValidationPropertiesContainer(properties_cross_validation)

    for c_value in properties_container_cross_validation.c_values:
        for whether_to_use_word2vec in properties_container_cross_validation.whether_to_use_word2vec:
            for whether_to_use_clustering in properties_container_cross_validation.whether_to_use_clustering:
                properties_main.c_value = c_value
                properties_main.whether_to_use_word2vec = whether_to_use_word2vec
                properties_main.whether_to_use_clustering = whether_to_use_clustering
                train_and_evaluate_model.train_and_evaluate_model_cross_validation(properties_main, path_slash_format, word2vecwrapper, properties_container_cross_validation)



class CrossValidationPropertiesContainer:
    """
    CrossValidationPropertiesContainer

    A container for properties specific to cross validation
    """

    def __init__(self, properties):
        """
        :params properties: a python model including properties (retrieved by importlib.import_module(<PATH>))
        """
        try:
            self.nr_of_cross_validation_splits_for_evaluation = properties.nr_of_cross_validation_splits_for_evaluation
        except AttributeError:
            print("Cross-validation settings file lacks the property 'nr_of_cross_validation_splits_for_evaluation'.")
            exit(1)

        try:
            self.c_values = properties.c_values
        except AttributeError:
            print("Cross-validation settings file lacks the property 'c_values'.")
            exit(1)

        try:
            self.whether_to_use_word2vec = properties.whether_to_use_word2vec
        except AttributeError:
            print("Cross-validation settings file lacks the property 'whether_to_use_word2vec'.")
            exit(1)

        try:
            self.whether_to_use_clustering = properties.whether_to_use_clustering
        except AttributeError:
            print("Cross-validation settings file lacks the property 'whether_to_use_clustering'.")
            exit(1)

        try:
            self.evaluation_output_dir = properties.evaluation_output_dir
        except AttributeError:
            print("Cross-validation settings file lacks the property 'evaluation_output_dir'.")
            exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    do_cross_validation(parser)
    



