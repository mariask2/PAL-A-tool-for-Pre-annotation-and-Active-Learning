from sklearn.feature_extraction.text import CountVectorizer
import os
import joblib
import glob
import numpy as np
import argparse
from matplotlib.pyplot import plot, show, bar, grid, axis, savefig, clf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
import gensim

import active_learning_preannotation


class ProcessMonitor():
    def __init__(self, path_slash_format, properties, unlabelled_text_vector = None):
        self.VECTORIZER_NAME = "vectorizer"
        self.PREDICTION = "PREDICTION"
        self.SCORE = "SCORE"
        self.MOST_COMMON_PREDICTION = "MOST_COMMON_PREDICTION"
        self.MEAN_SCORE = "MEAN_SCORE"
        self.MINORITY_CLASSES = "MINORITY_CLASSES"
        self.SAVED_DICTIONARY_PREFIX = "saved_dict_"
        self.NUMBER_OF_LABELLED_KEY = "NUMBER_OF_LABELLED"
        self.PLOT_PREFIX = "plot_"
        self.PLOT_FILE_ENDING = ".png"

        self.path_slash_format = path_slash_format
        self.process_monitoring_dir = properties.process_monitoring_dir
        self.write_process_monitoring = properties.write_process_monitoring
        self.vector_length = properties.semantic_vector_length
        self.model_path = properties.model_path
        if unlabelled_text_vector: # If used during data selection
            self.init_process_monitoring(path_slash_format, properties, unlabelled_text_vector)
 
    def get_full_process_monitoring_dir_path(self):
        full_process_monitoring_dir = os.path.join(self.path_slash_format, self.process_monitoring_dir)
        return full_process_monitoring_dir

    def init_process_monitoring(self, path_slash_format, properties, unlabelled_text_vector):
        if self.write_process_monitoring:
            full_process_monitoring_dir_path = self.get_full_process_monitoring_dir_path()

            if not os.path.exists(full_process_monitoring_dir_path):
                text_concatenated = np.concatenate(unlabelled_text_vector)
                word_vectorizer = CountVectorizer(binary = True, min_df=max([properties.min_df_current, 2]), \
                                                                max_df = properties.max_df_current)
                word_vectorizer.fit_transform(text_concatenated)
                
                os.mkdir(full_process_monitoring_dir_path)
                vectorizer_full_name = os.path.join(full_process_monitoring_dir_path, self.VECTORIZER_NAME)
                joblib.dump(word_vectorizer, vectorizer_full_name, compress=9)
                return True
            else:
                return False

    def set_number_of_labelled(self, number_of_labelled):
        self.number_of_labelled = number_of_labelled
    
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

        # Note, not thread safe at all. Not intended to be run by more than one thread or process
        path_and_prefix = os.path.join(self.get_full_process_monitoring_dir_path(),\
                                       self.SAVED_DICTIONARY_PREFIX + str(self.number_of_labelled) + "_")
        previously_saved_files = glob.glob(path_and_prefix + "*")
        
        if len(previously_saved_files) == 0:
            suffix_to_use = 1
        else:
            suffixes = sorted([int(el[-1]) for el in previously_saved_files])
            last_used_suffix = suffixes[-1]
            suffix_to_use = last_used_suffix + 1

        file_to_save_in = path_and_prefix + str(suffix_to_use)
        print("Saving in " + file_to_save_in)
        joblib.dump(final_hash, file_to_save_in, compress=9)

    def analyse_saved_files(self):
        count_vectorizer = joblib.load(os.path.join(self.get_full_process_monitoring_dir_path(), self.VECTORIZER_NAME))
        print(count_vectorizer)
        types_at_process_start = count_vectorizer.get_feature_names()
        self.plot(types_at_process_start)


    def plot(self, word_list):
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True, unicode_errors='ignore')
        all_vectors_list = []
        found_words = []
        for word in word_list:
            try:
                vec_raw  = word2vec_model[self.remove_underscore(word)]
                norm_vector = list(preprocessing.normalize(np.reshape(vec_raw, newshape = (1, self.vector_length)), norm='l2')[0])
                print(word)
                #print(vec)
                all_vectors_list.append(norm_vector)
                found_words.append(word)
            except KeyError:
                print(word + " not found")

        all_vectors_np = np.array(all_vectors_list)
        pca_model = PCA(n_components=50)
        tsne_model = TSNE(n_components=2, random_state=0)
        DX_pca = pca_model.fit_transform(all_vectors_np)
        DX = tsne_model.fit_transform(DX_pca)

        fig = plt.figure()
        
        for point, found_word in zip(DX, found_words):
            plt.scatter(point[0], point[1], color = "gray", marker = ".", s=1)

        save_figure_file_name = os.path.join(self.get_full_process_monitoring_dir_path(), self.PLOT_PREFIX + self.PLOT_FILE_ENDING)
        plt.savefig(save_figure_file_name, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    properties, path_slash_format, path_dot_format = active_learning_preannotation.load_properties(parser)
    process_monitor_instance =  ProcessMonitor(path_slash_format, properties)
    process_monitor_instance.analyse_saved_files()





