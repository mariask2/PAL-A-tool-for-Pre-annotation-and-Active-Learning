from sklearn.feature_extraction.text import CountVectorizer
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

VECTORIZER_NAME = "vectorizer"
PREDICTION = "PREDICTION"
SCORE = "SCORE"
MINORITY_CLASSES = "MINORITY_CLASSES"


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


def remove_underscore(word):
    if len(word) == 3 and word[1] == "_":
        return word[0]
    else:
        return word


def write_process_monitoring_info(sentences_unlabelled, all_diffs, selected_indeces, ys, majority_class, inv_labelled_dict):
    word_hash = {}
    for index, diffs, y in zip(selected_indeces, all_diffs, ys):
        current_sentence = sentences_unlabelled[index]
        for word, conf, prediction in zip(current_sentence, diffs, y):
            word = remove_underscore(word)
            if word not in word_hash:
                word_hash[word] = {PREDICTION:[], SCORE:[]}
            word_hash[word][PREDICTION].append(prediction)
            word_hash[word][SCORE].append(conf)
    final_hash = {}
    for key, item in word_hash.items():
        mean_conf = sum(item[SCORE])/len(item[SCORE])
        nr_of_majority = item[PREDICTION].count(majority_class)
        half_length = len(item[PREDICTION])/2
        if nr_of_majority > half_length:
            most_common_predicted = inv_labelled_dict[majority_class]
        else:
            most_common_predicted = MINORITY_CLASSES
        final_hash[key] = {PREDICTION: most_common_predicted, SCORE: mean_conf}

    for key, item in final_hash.items():
        print(key, item)




