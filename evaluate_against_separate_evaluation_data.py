"""
"""
import argparse
import train_and_evaluate_model
import active_learning_preannotation
import vectorize_data
import os
import importlib

def do_evaluate_against_separate_evaluation_data(parser):
    SETTINGS = "settings"
    properties_main, path_slash_format, path_dot_format = active_learning_preannotation.load_properties(parser)
    word2vecwrapper = vectorize_data.Word2vecWrapper(properties_main.model_path, properties_main.semantic_vector_length)
    main_properties_file_name = os.path.join(path_slash_format, SETTINGS + ".py")

    EVAL_SETTINGS = "evaluate_against_separate_evaluation_data_settings"
    properties_file_name = os.path.join(path_slash_format, EVAL_SETTINGS + ".py")
    if not os.path.exists(properties_file_name):
        print("The directory '" + str(path_slash_format) + "' does not have a " + EVAL_SETTINGS + ".py file.")
        exit(1)
    properties_eval = importlib.import_module(path_dot_format + "." + EVAL_SETTINGS)
    properties_container_eval = EvalPropertiesContainer(properties_eval)

    train_and_evaluate_model.train_and_evaluate_model_against_evaluation_data(properties_main, path_slash_format, word2vecwrapper, properties_container_eval, main_properties_file_name)


class EvalPropertiesContainer:
    """
    CrossValidationPropertiesContainer

    A container for properties specific to cross validation
    """

    def __init__(self, properties):
        """
        :params properties: a python model including properties (retrieved by importlib.import_module(<PATH>))
        """
        try:
            self.separate_evaluation_data_dir = properties.separate_evaluation_data_dir
        except AttributeError:
            print("Settings file lacks the property 'separate_evaluation_data_dir'.")
            exit(1)
        try:
            self.separate_evaluation_output_dir = properties.separate_evaluation_output_dir
        except AttributeError:
            print("Settings file lacks the property 'separate_evaluation_output_dir'.")
            exit(1)

        try:
            self.negative_category = properties.negative_category
        except AttributeError:
            print("Settings file lacks the property 'separate_evaluation_output_dir'.")
            exit(1)

        try:
            self.save_model = properties.save_model
        except AttributeError:
            print("Settings file lacks the property 'separate_evaluation_output_dir'.")
            exit(1)

        try:
            self.saved_model_dir = properties.saved_model_dir
        except AttributeError:
            print("Settings file lacks the property 'separate_evaluation_output_dir'.")
            exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    do_evaluate_against_separate_evaluation_data(parser)


