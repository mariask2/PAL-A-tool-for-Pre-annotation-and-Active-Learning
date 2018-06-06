from sklearn.feature_extraction.text import CountVectorizer
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

VECTORIZER_NAME = "vectorizer"


def get_full_process_monitoring_dir_path(path_slash_format, properties):
    full_process_monitoring_dir = os.path.join(path_slash_format, properties.process_monitoring_dir)
    return full_process_monitoring_dir

def init_process_monitoring(path_slash_format, properties, unlabelled_text_vector):
    text_concatenated = np.concatenate(unlabelled_text_vector)
    word_vectorizer = CountVectorizer(binary = True, min_df=properties.min_df_current, \
                                      max_df = properties.max_df_current)
    word_vectorizer.fit_transform(text_concatenated)
    
        #for word in word_vectorizer.get_feature_names():
        #print(word)
    
    if properties.write_process_monitoring:
        full_process_monitoring_dir_path = get_full_process_monitoring_dir_path(path_slash_format, properties)
        if not os.path.exists(full_process_monitoring_dir_path):
            os.mkdir(full_process_monitoring_dir_path)
            vectorizer_full_name = os.path.join(full_process_monitoring_dir_path, VECTORIZER_NAME)
            joblib.dump(vectorizer_full_name, vectorizer_full_name, compress=9)
            return True
        else:
            return False

def write_process_monitoring_info(sentences_unlabelled, all_diffs, selected_indeces, ys, majority_class, inv_labelled_dict):
    for index, diffs, y in zip(selected_indeces, all_diffs, ys):
        #print(index)
        print(sentences_unlabelled[index])
        print(diffs)
        print(y)
        print(majority_class)
        print(inv_labelled_dict)
        print("****")


