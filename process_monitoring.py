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
import pickle

import active_learning_preannotation

############
## The ProcessMonitor class has quite specialised functionality, and is still under development
## so its functionality is not yet described in the general readme file.
## The aim of the functionality is to monitor the state of the data pool during the
## active learning process.
## For the process monitoring  to work, the directory "process_monitoring" most be deleted (or renamed to a new name)
## before the active learning process starts
## The first thing that will happen is then that a directory with the name 'process_monitoring' is created
## in the folder where you have your project. In this directory, the 'vectorizer' file will be saved
## Which contains information of all types included in the pool when the process starts
## The actual state of the pool during the active learning process is save in pickled python dictionaries
## in the two folders 'word2vec_false' and 'word2vec_true', depending on the settings.
## For saving the data, the 'write_process_monitoring' variable in 'settings.py' must be set to True.
## The settings.py file is also used for loading other kinds of information, e.g. where the word2vec-space needed
## for the visualisation is stored.
## Note that is the process is run as a simulation, then the data in the 'different_sizes_simulation_settings.py'
## is what governs what will happen regarding the saving of the data. However, for then using the
## saved data for visualisation, 'settings.py' is used. So make sure that the info in these two are consistent.
## (Some kind of automatic check should be added in the future.)
## Also, note that the visualisation treats tokens as either 'an entity' or 'not an entity', disregarding
## which kind of entities there are, if there are many. (This should be made configurable in the future.)
## Therefore, if you use the monitoring in the simulation process, where you use one entity at a time. The practical way to do it is to,
## when the process is simulation process finished for one entity, rename the 'process_monitoring' folder to
## 'process_monitoring_my_entity_type_1', do the simulation of the next entity type and rename the folder to
## 'process_monitoring' to 'process_monitorin_my_entity_type_2' and so on.
##
## To visualise the states the pool in the active learning process, write
## python process_monitoring.py --project=data.example_project


class ProcessMonitor():
    def __init__(self, path_slash_format, properties, whether_to_use_word2vec, unlabelled_text_vector = None):
        self.VECTORIZER_NAME = "vectorizer"
        self.PREDICTION = "PREDICTION"
        self.SCORE = "SCORE"
        self.MOST_COMMON_PREDICTION = "P"
        self.MEAN_SCORE = "M"
        self.PREDICTION_STATISTICS = "S"
        self.VARIANCE_SCORE = "V"
        self.MINORITY_CLASSES = "MINORITY_CLASSES"
        self.FOLDER_FOR_WORD2VEC_TRUE = "word2vec_true"
        self.FOLDER_FOR_WORD2VEC_FALSE = "word2vec_false"
        self.SAVED_DICTIONARY_PREFIX = "saved_dict_"
        self.NUMBER_OF_LABELLED_KEY = "NUMBER_OF_LABELLED"
        self.PLOT_PREFIX = "plot_"
        self.PLOT_FILE_ENDING = ".png"
        self.WORD_PREFIX = "most_uncertain_words_"
        self.PLOT_MARGIN = 50
        
        self.path_slash_format = path_slash_format
        self.whether_to_use_word2vec = whether_to_use_word2vec # Don't use the value in the properies file, as
        # this does not always correspond to the truth when the acitve learning process is simulated
        self.process_monitoring_dir = properties.process_monitoring_dir
        self.write_process_monitoring = properties.write_process_monitoring
        self.vector_length = properties.semantic_vector_length
        self.model_path = properties.model_path
        self.majority_class = properties.outside_class
        if unlabelled_text_vector: # If used during data selection
            self.init_process_monitoring(path_slash_format, properties, unlabelled_text_vector)
    
 
    def get_full_process_monitoring_dir_path_no_word2vec_info(self):
        full_process_monitoring_dir = os.path.join(self.path_slash_format, self.process_monitoring_dir)
        return full_process_monitoring_dir
    
    def get_full_process_monitoring_dir_path(self):
        if self.whether_to_use_word2vec == True:
            sub_dir_name = self.FOLDER_FOR_WORD2VEC_TRUE
        else:
            sub_dir_name = self.FOLDER_FOR_WORD2VEC_FALSE
        return os.path.join(self.get_full_process_monitoring_dir_path_no_word2vec_info(), sub_dir_name)


    def init_process_monitoring(self, path_slash_format, properties, unlabelled_text_vector):
        if self.write_process_monitoring:
            full_process_monitoring_dir_path = self.get_full_process_monitoring_dir_path_no_word2vec_info()

            if not os.path.exists(full_process_monitoring_dir_path):
                text_concatenated = np.concatenate(unlabelled_text_vector)
                # To save storage space, only include types occurring at least three time in the statistics
                # or more than three times, if
                #word_vectorizer = CountVectorizer(binary = True, min_df=max([properties.min_df_current, 3]), \
                        #max_df = properties.max_df_current)
                word_vectorizer = CountVectorizer(binary = True, min_df=1, max_df = properties.max_df_current)
                word_vectorizer.fit_transform(text_concatenated)
                
                os.mkdir(full_process_monitoring_dir_path)
                vectorizer_full_name = os.path.join(full_process_monitoring_dir_path, self.VECTORIZER_NAME)
                joblib.dump(word_vectorizer, vectorizer_full_name, compress=9)
            
            full_process_monitoring_dir_path_word2vec_info = self.get_full_process_monitoring_dir_path()
            print("Will write process monitoring info in ", full_process_monitoring_dir_path_word2vec_info)
 
            if not os.path.exists(full_process_monitoring_dir_path_word2vec_info):
                os.mkdir(full_process_monitoring_dir_path_word2vec_info)
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

    def get_variance_from_lst(self, conf_lst):
        return np.var(conf_lst)
    
    def get_most_common_predicted(self, predicted_lst, majority_class, inv_labelled_dict):
        nr_of_majority = predicted_lst.count(majority_class)
        half_length = len(predicted_lst)/2
        if nr_of_majority > half_length:
            most_common_predicted = inv_labelled_dict[majority_class]
        else:
            most_common_predicted = self.MINORITY_CLASSES
        return most_common_predicted
    
    def get_stat_dictionary(self, predicted_lst, inv_labelled_dict):
        stat_dict = {}
        for el in predicted_lst:
            if inv_labelled_dict[el] not in stat_dict:
                stat_dict[inv_labelled_dict[el]] = 0
            stat_dict[inv_labelled_dict[el]] = stat_dict[inv_labelled_dict[el]] + 1
        return stat_dict
    
    
    #print("min_probability_differences", min_probability_differences)

    def write_process_monitoring_selected_words(self, sentence_index_selected_in_active_selection):
        
        if not self.write_process_monitoring:
            return
    
        saved_in = os.path.split(self.current_file_name)
        file_to_save_in = os.path.join(saved_in[0], self.WORD_PREFIX + saved_in[1])

        min_words_in_selected_sentences = []
        for el in sentence_index_selected_in_active_selection:
            min_words_in_selected_sentences.append(self.current_selected_indeces_min_prob_word_hash[el])

        min_words_in_selected_sentences.sort()

        open_file = open(file_to_save_in, "w")
        for order, word in enumerate(min_words_in_selected_sentences):
            open_file.write(str(word[1]) + "\t" + str(word[0]) + "\t" + str(order) + "\n")
        open_file.close()
        

    def write_process_monitoring_info(self, sentences_unlabelled, all_diffs, selected_indeces, ys, majority_class, \
                                      inv_labelled_dict, all_index_for_min_probabilities, min_probability_differences):
        
        if not self.write_process_monitoring:
            return
        
        self.current_selected_indeces_min_prob_word_hash = {}

        for sentence_nr, index_in_sentence_with_min_prop, min_prop_value in zip(selected_indeces, all_index_for_min_probabilities, min_probability_differences):
            sentence = list(sentences_unlabelled[sentence_nr])
            word_with_lowest_prob = sentence[index_in_sentence_with_min_prop]
            #print(sentence, word_with_lowest_prob)
            self.current_selected_indeces_min_prob_word_hash[sentence_nr] = (min_prop_value, word_with_lowest_prob)
            #self.current_selected_indeces_min_prob_hash[nr] =
            #print(sentences_unlabelled[index], all_index_for_min_probabilities[index])
        
        
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
            variance_conf = self.get_variance_from_lst(item[self.SCORE])
            most_common_predicted = self.get_most_common_predicted(item[self.PREDICTION], majority_class, inv_labelled_dict)
            stat_dict = self.get_stat_dictionary(item[self.PREDICTION], inv_labelled_dict)
            final_hash[key] = {self.MOST_COMMON_PREDICTION: most_common_predicted, self.MEAN_SCORE: mean_conf,\
                self.PREDICTION_STATISTICS:stat_dict, self.VARIANCE_SCORE: variance_conf}

        self.current_file_name = self.get_file_to_save_in()
        #print("Saving in " + file_to_save_in)
        pickle.dump(final_hash, open(self.current_file_name, "wb"))

    def get_file_to_save_in(self):
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
        return file_to_save_in

    
    
    def analyse_saved_files(self):
        count_vectorizer = joblib.load(os.path.join(self.get_full_process_monitoring_dir_path_no_word2vec_info(), self.VECTORIZER_NAME))
        print(count_vectorizer)
        types_at_process_start = count_vectorizer.get_feature_names()
        self.plot(types_at_process_start)


    def plot(self, word_list):
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True, unicode_errors='ignore')
        all_vectors_list = []
        found_words = []
        for word in word_list:
            word = self.remove_underscore(word)
            print(word)
            try:
                vec_raw  = word2vec_model[word]
                norm_vector = list(preprocessing.normalize(np.reshape(vec_raw, newshape = (1, self.vector_length)), norm='l2')[0])
                all_vectors_list.append(norm_vector)
                found_words.append(word)
            except KeyError:
                print(word + " not found")

        all_vectors_np = np.array(all_vectors_list)
        pca_model = PCA(n_components=50)
        tsne_model = TSNE(n_components=2, random_state=0)
        DX_pca = pca_model.fit_transform(all_vectors_np)
        DX = tsne_model.fit_transform(DX_pca)
        self.plot_each_state(DX, found_words, whether_to_use_word2vec = True)
        self.plot_each_state(DX, found_words, whether_to_use_word2vec = False)

    def plot_each_state(self, DX, found_words, whether_to_use_word2vec):
        self.whether_to_use_word2vec = whether_to_use_word2vec
        path_and_prefix_states = os.path.join(self.get_full_process_monitoring_dir_path(),\
                               self.SAVED_DICTIONARY_PREFIX)
        previously_saved_files = glob.glob(path_and_prefix_states + "*")
        

        
        if len(previously_saved_files) == 0:
            print("No saved files were found in. " + self.get_full_process_monitoring_dir_path() +\
                  " Probably, no active learning process have been run with the setting 'whether_to_use_word2vec' = "\
                  + str(whether_to_use_word2vec))
            return
        
        suffixes_names = sorted([(int(el[-1]), el) for el in previously_saved_files])
        print(suffixes_names)

        smallest_x = float("inf")
        smallest_y = float("inf")
        largest_x = -1*float("inf")
        largest_y = -1*float("inf")

        fig = plt.figure()
        for (nr, filename) in suffixes_names:
            
            annotated_points = set()
            sp = filename.split("_")
            nr_ending = sp[-2] + "_" + sp[-1]
            result_dict = pickle.load(open(filename, "rb"))
        
            plt.clf()
            plt.axis('off')
            plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off',\
                            labelleft='off', labeltop='off', labelright='off', labelbottom='off')


            # outside class plot
            for point, found_word in zip(DX, found_words):
                if found_word in result_dict:
                    if point[0] < smallest_x:
                        smallest_x = point[0]
                    if point[1] < smallest_y:
                        smallest_y = point[1]
                    if point[0] > largest_x:
                        largest_x = point[0]
                    if point[1] > largest_y:
                        largest_y = point[1]

                    if result_dict[found_word][self.MOST_COMMON_PREDICTION] == self.majority_class:
                        # Make sure its visible even if it certain
                        if result_dict[found_word][self.MEAN_SCORE] < 0.9:
                            alfa = max((1 - result_dict[found_word][self.MEAN_SCORE]),0.1)
                            print(str(alfa) + " " + found_word)
                            color_to_use = (0,0,0,alfa)
                            plt.scatter(point[0], point[1], color = color_to_use, marker = "o", s=3)

                if smallest_x != float("inf"): # Not first time in loop
                    #"Plot to make sure that the image has the same size"
                    plt.scatter(smallest_x-self.PLOT_MARGIN, 0, color = "white", marker = "o", s=1)
                    plt.scatter(0, smallest_y, color = "white", marker = "o", s=1)
                    plt.scatter(largest_x+self.PLOT_MARGIN, 0, color = "white", marker = "o", s=1)
                    plt.scatter(0, largest_y, color = "white", marker = "o", s=1)

            print(smallest_x, smallest_y, largest_x, largest_y)
            
            saved_in = os.path.split(filename)
            most_uncertain_words_file_name = os.path.join(saved_in[0], self.WORD_PREFIX + saved_in[1])
            most_uncertain_words_file = open(most_uncertain_words_file_name)
            most_uncertain_words = {}
            for row in most_uncertain_words_file:
                sp = row.strip().split("\t")
                most_uncertain_words[self.remove_underscore(sp[0])] = (sp[1], sp[2])

            print(most_uncertain_words)
            most_uncertain_words_file.close()
            """
                # minority class annotation
                for point, found_word in zip(DX, found_words):
                    if found_word in result_dict:
                    if not result_dict[found_word][self.MOST_COMMON_PREDICTION] == self.majority_class:
                    rounded_tuple = (round(point[0]), round(point[0]))
                    if rounded_tuple not in annotated_points: # not to many annotations close in the plot
                        annotated_points.add(rounded_tuple)
            
                    # TODO. Make this code smarter. The point is that the labels are not supposed to overlap
            # but they do anyway
            annotated_points.add((rounded_tuple[0] + 1, rounded_tuple[1]))
            annotated_points.add((rounded_tuple[0] - 1, rounded_tuple[1]))
            annotated_points.add((rounded_tuple[0], rounded_tuple[1] + 1))
            annotated_points.add((rounded_tuple[0], rounded_tuple[1] - 1))
            
            annotated_points.add((rounded_tuple[0] + 2, rounded_tuple[1]))
            annotated_points.add((rounded_tuple[0] + 3, rounded_tuple[1]))
            annotated_points.add((rounded_tuple[0] + 4, rounded_tuple[1]))
            annotated_points.add((rounded_tuple[0] + 5, rounded_tuple[1]))
            
                    annotated_points.add(rounded_tuple)
                    #plt.annotate(found_word, (point[0], point[1]), xytext=(point[0], point[1]), color = "black", fontsize=9)
                    #arrowprops=dict(facecolor="gray", shrink=0.05, frac=0.05)
                """

            # minority class plot
            for point, found_word in zip(DX, found_words):
                if found_word in result_dict:
                    if not result_dict[found_word][self.MOST_COMMON_PREDICTION] == self.majority_class:
                        # Make sure its visible even if it certain
                        alfa = max((1 - result_dict[found_word][self.MEAN_SCORE]),0.1)
                        print(str(alfa) + " " + found_word + " " + "minority" )
                        color_to_use = (1,0,0,alfa)
                        plt.scatter(point[0], point[1], color = color_to_use, marker = "o", s=3)

            # chosen word annotation
            for point, found_word in zip(DX, found_words):
                if found_word in result_dict and found_word in most_uncertain_words:
                    word_nr = str(int(most_uncertain_words[found_word][1]) + 1)
                    plt.annotate(word_nr, (point[0], point[1]), xytext=(point[0] + 1, point[1] + 1), color = "white",\
                                 fontsize=7, weight = "bold")
                    plt.annotate(word_nr, (point[0], point[1]), xytext=(point[0] + 1, point[1] + 1), color = "black",\
                                     fontsize=7, weight = "medium")
                    # Give the full annotation information in the margin
                    # Sort them verticaly by their uncertainty order, i.e., as given by most_uncertain_words[found_word][1]
                    uncertainty_to_print = str(int(100*(round(float(most_uncertain_words[found_word][0]),2))))
                    y_cord = largest_y - int(most_uncertain_words[found_word][1])*9
                    plt.annotate(word_nr + ": " + found_word + " " + uncertainty_to_print + "%",\
                         (smallest_x-self.PLOT_MARGIN-10, y_cord),\
                         xytext=(smallest_x-self.PLOT_MARGIN-10, y_cord), color = "black", fontsize=9)



            save_figure_file_name = os.path.join(self.get_full_process_monitoring_dir_path(), self.PLOT_PREFIX +\
                                                 nr_ending + self.PLOT_FILE_ENDING)
            plt.savefig(save_figure_file_name, dpi = 700) #, bbox_inches='tight')
            print("Saved plot in " + save_figure_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    properties, path_slash_format, path_dot_format = active_learning_preannotation.load_properties(parser)
    process_monitor_instance =  ProcessMonitor(path_slash_format, properties, True)
    process_monitor_instance.analyse_saved_files()






