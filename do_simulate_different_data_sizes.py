"""
Functionality for initializing and running a simulation of different sizes and different data
selection methods

Run, for instance, as follows:
python do_simulate_different_data_sizes.py --project=data.unshared_task_argument_mining --start_fold=1 --end_fold=2
"""

import argparse
import train_and_evaluate_model
import active_learning_preannotation
import vectorize_data
import os
import importlib

def load_properties(parser, settings_name):
    
    parser.add_argument('--project', action='store', dest='project_path', \
                            help='The path, separated by dots, to where the project i located. For instance: data.example_project')

    parser.add_argument('--start_fold', action='store', dest='start_fold', \
                            help='The number of the fold which to start with')

    parser.add_argument('--end_fold', action='store', dest='end_fold', \
                            help='The number of the fold which to end with (i.e., the number just before this one will be the last)')

    args = parser.parse_args()
    if not args.project_path:
        print("The argument '--project' with the path to the data needs to be given")
        exit(1)

    if not args.start_fold:
        print("The argument '--start_fold' needs to be given")
        exit(1)

    if not args.end_fold:
        print("The argument '--end_fold' needs to be given")
        exit(1)

    print(args.project_path)

    start_fold_int = int(args.start_fold)
    end_fold_int = int(args.end_fold)
    
        
    path_slash_format = ""
    for path_part in args.project_path.split("."):
        path_slash_format = os.path.join(path_slash_format, path_part)
    
    if not os.path.exists(path_slash_format):
        print("The directory '" + str(path_slash_format) + "' i.e., the directory matching the project path given '" + \
                  str(args.project_path) + "', does not exist")
        exit(1)

    if not os.path.exists(os.path.join(path_slash_format, settings_name + ".py")):
        print("The directory '" + str(path_slash_format) + "' does not have a " + settings_name + ".py file.")
        exit(1)

    properties = importlib.import_module(args.project_path + "." + settings_name)

    #properties = importlib.import_module(project_path + "." + SETTINGS)

    properties_container = active_learning_preannotation.PropertiesContainer(properties)
    simulation_properties_container = SimulationProperties(properties)

    #active_learning_preannotation.check_properties(properties.minority_classes, properties.outside_class, properties.beginning_prefix, \
     #                    properties.inside_prefix)

    return properties_container, simulation_properties_container, path_slash_format, start_fold_int, end_fold_int


class SimulationProperties:

    def __init__(self, properties):
        """                                                                                                                
        :params properties: a python model including properties (retrieved by importlib.import_module(<PATH>))             
        """

        try:
            self.seed_set_size = properties.seed_set_size
        except AttributeError:
            print("Settings file lacks the property 'seed_set_size'.")
            exit(1)

        try:
            self.max_size = properties.max_size
        except AttributeError:
            print("Settings file lacks the property 'max_size'.")
            exit(1)

        try:
            self.step_size = properties.step_size
        except AttributeError:
            print("Settings file lacks the property 'step_size'.")
            exit(1)

        try:
            self.max_test_data_size = properties.max_test_data_size
        except AttributeError:
            print("Settings file lacks the property 'max_test_data_size'.")
            exit(1)

        try:
            self.train_test_splits = properties.train_test_splits
        except AttributeError:
            print("Settings file lacks the property 'train_test_splits'.")
            exit(1)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    SETTINGS = "different_sizes_simulation_settings"
    properties, simulation_properties, path_slash_format, start_fold, end_fold = load_properties(parser, SETTINGS)
    word2vecwrapper = vectorize_data.Word2vecWrapper(properties.model_path, properties.semantic_vector_length)
    train_and_evaluate_model.simulate_different_data_sizes(properties, simulation_properties,\
                                                               path_slash_format, word2vecwrapper, start_fold, end_fold)


