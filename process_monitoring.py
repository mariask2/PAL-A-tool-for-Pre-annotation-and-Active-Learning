from sklearn.feature_extraction.text import CountVectorizer
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class ProcessMonitor():
    def __init__(self, path_slash_format, properties, unlabelled_text_vector):
        self.VECTORIZER_NAME = "vectorizer"
        self.PREDICTION = "PREDICTION"
        self.SCORE = "SCORE"
        self.MOST_COMMON_PREDICTION = "MOST_COMMON_PREDICTION"
        self.MEAN_SCORE = "MEAN_SCORE"
        self.MINORITY_CLASSES = "MINORITY_CLASSES"
        self.path_slash_format = path_slash_format
        self.process_monitoring_dir = properties.process_monitoring_dir
        self.write_process_monitoring = properties.write_process_monitoring
        self.init_process_monitoring(path_slash_format, properties, unlabelled_text_vector)
    

    def get_full_process_monitoring_dir_path(self):
        full_process_monitoring_dir = os.path.join(self.path_slash_format, self.process_monitoring_dir)
        return full_process_monitoring_dir

    def init_process_monitoring(self, path_slash_format, properties, unlabelled_text_vector):
        if self.write_process_monitoring:
            full_process_monitoring_dir_path = self.get_full_process_monitoring_dir_path()

            if not os.path.exists(full_process_monitoring_dir_path):
                text_concatenated = np.concatenate(unlabelled_text_vector)
                word_vectorizer = CountVectorizer(binary = True, min_df=properties.min_df_current, \
                                                                max_df = properties.max_df_current)
                word_vectorizer.fit_transform(text_concatenated)
                
                os.mkdir(full_process_monitoring_dir_path)
                vectorizer_full_name = os.path.join(full_process_monitoring_dir_path, self.VECTORIZER_NAME)
                joblib.dump(vectorizer_full_name, vectorizer_full_name, compress=9)
                return True
            else:
                return False


    def remove_underscore(self, word):
        if len(word) == 3 and word[1] == "_":
            return word[0]
        else:
            return word

    def get_mean_conf_from_lst(self, conf_lst):
         return sum(conf_lst)/len(conf_lst)
    
    def get_most_common_predicted(self, predicted_lst, majority_class, inv_labelled_dict):
        nr_of_majority = predicted_lst.count(majority_class)
        half_length = len(predicted_lst)/2
        if nr_of_majority > half_length:
            most_common_predicted = inv_labelled_dict[majority_class]
        else:
            most_common_predicted = self.MINORITY_CLASSES
        return most_common_predicted
    
    
    def write_process_monitoring_info(self, sentences_unlabelled, all_diffs, selected_indeces, ys, majority_class, inv_labelled_dict):
        if not self.write_process_monitoring:
            return
        
        word_hash = {}
        for index, diffs, y in zip(selected_indeces, all_diffs, ys):
            current_sentence = sentences_unlabelled[index]
            for word, conf, prediction in zip(current_sentence, diffs, y):
                word = self.remove_underscore(word)
                if word not in word_hash:
                    word_hash[word] = {self.PREDICTION:[], self.SCORE:[]}
                word_hash[word][self.PREDICTION].append(prediction)
                word_hash[word][self.SCORE].append(conf)
        final_hash = {}
        for key, item in word_hash.items():
            mean_conf = self.get_mean_conf_from_lst(item[self.SCORE])
            most_common_predicted = self.get_most_common_predicted(item[self.PREDICTION], majority_class, inv_labelled_dict)
            final_hash[key] = {self.MOST_COMMON_PREDICTION: most_common_predicted, self.MEAN_SCORE: mean_conf,\
                self.PREDICTION:item[self.PREDICTION], self.SCORE:item[self.SCORE] }

        for key, item in final_hash.items():
            print(key, item)




