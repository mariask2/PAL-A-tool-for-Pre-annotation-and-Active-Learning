"""
"""

import sys
import time
import os
import shutil
import importlib
import argparse

import transform_to_brat_format
import vectorize_data
import classify_and_select

def select_new_data(properties, project_path):
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
    labelled_text_vector, labelled_label_vector, label_dict = vectorize_data.read_file_labelled_data(labelled_data_dir_for_project, properties.data_file_extension, properties.minority_classes, properties.outside_class)

    unlabelled_text_vector = vectorize_data.read_file_unlabelled_data(unlabelled_data_path)
    print("Read unlabelled data from:\t" + unlabelled_data_path)
    
    
    X_labelled_np, X_unlabelled_np, y_labelled_np, text_vector_labelled_np, text_vector_unlabelled_np = vectorize_data.vectorize_data(labelled_text_vector, unlabelled_text_vector, labelled_label_vector, label_dict, classes, use_word2vec = properties.whether_to_use_word2vec, model_path = properties.model_path)    

    to_select_X, new_unlabelled_x, to_select_text, new_sentences_unlabelled, predicted_for_selected = classify_and_select.get_new_data(X_labelled_np, X_unlabelled_np, y_labelled_np, text_vector_labelled_np, text_vector_unlabelled_np, label_dict, properties.minority_classes, properties.nr_of_samples, properties.maximum_samples_to_search_among)

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
    old_unlabelled_data_path = os.path.join(unlabelled_data_dir_for_project, time.strftime("unlabelled_%Y%m%d_%H%M%S.csv"))
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



    transform_to_brat_format.transform(to_annotate_file_path, tolabel_data_dir_for_project, interesting_tags, properties.outside_class, properties.beginning_prefix)


def load_properties():
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', action='store', dest='project_path', help='The path, separated by dots, to where the project i located. For instance: data.example_project')
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
        print("The directory '" + str(path_slash_format) + "' i.e., the directory matching the project path given '" + str(args.project_path) + "', does not exist")
        exit(1)

    if not os.path.exists(os.path.join(path_slash_format, SETTINGS + ".py")):
        print("The directory '" + str(path_slash_format) + "' does not have a " + SETTINGS + ".py file.")
        exit(1)

    
    properties = importlib.import_module(args.project_path + "." + SETTINGS)

    return properties, path_slash_format


if __name__ == "__main__":

    #project_path = "data.example_project"
    properties, path_slash_format = load_properties()

    select_new_data(properties, path_slash_format)


