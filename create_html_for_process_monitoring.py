# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
import os
import joblib
import glob
import numpy as np
import argparse
from matplotlib.pyplot import plot, show, bar, grid, axis, savefig, clf
import matplotlib.markers
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
import gensim
import pickle

import active_learning_preannotation

############

##
## To create a html-file which shows changes in states in the process,  write
## python create_html_for_process_monitoring.py --project=data.example_project


class HtmlCreator():
    def __init__(self, path_slash_format, properties):

        self.PLOT_PREFIX = "plot_"
        self.PLOT_FILE_ENDING = ".png"
        self.HTML_NAME = "annotation_status.html"
        self.FOLDER_FOR_WORD2VEC_TRUE = "word2vec_true"
        self.FOLDER_FOR_WORD2VEC_FALSE = "word2vec_false"
        
        self.path_slash_format = path_slash_format
        self.process_monitoring_dir = properties.process_monitoring_dir
        self.majority_class = properties.outside_class
        self.entities = []
        for c in properties.minority_classes:
            if "-" in c:
                self.entities.append(c.split("-")[1])
            else:
                self.entities.append(c)
        self.entities = list(set(self.entities))
        #self.entity_class = properties.minority_classes[0].split("-")[1]
   
        self.first_part_html = \
        """
            <!doctype html>
                <html lang="en">
                    <head>
                        <meta charset="utf-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1">
                                <title>Annotation status</title>
                                    <link rel="stylesheet" href="https://code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">
                                        <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
                                        <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
                                        <style type="text/css">
                                            .currentImage {
                                            margin-top: 20px;
                                            margin-left: 20px;
                                            margin-bottom: 5px;
                                            border-style: solid;
                                            border-color: lightgrey;
                                            border-width: 1px;
           """
        self.start_image_prefix = 'background-image: url("'
        self.start_image_suffix = '");'
        
        self.first_part_html_after_image = \
        """
                                            background-repeat: no-repeat;
                                            background-size: 1500px;
                                            width: 1500px;
                                            height: 650px;
                                            border-radius: 25px;
                                            }
                                            .sliderStyle{
                                            margin-left: 20px;
                                            margin-top; 20px;
                                            color: lightgrey;
                                            background-color: red;
                                            border-radius: 25px;
                                            }
            
                                    </style>
                                    <script>
            """
        
        self.second_part_html = \
        """
                                    $( function() {
                                    $( "#slider" ).slider({
                                    disabled: false,
                                    range: "min",
                                    animate: "fast",
                                    value: 0,
                                    min: 0,
                                    max: suffixes.length,
                                    step: 1,
                                    slide: function(event, ui){
                                    $(".currentImage").css("background-image", "url('" + suffixes[ui.value] + "')");
                                    }});
                                    });
            
                                    $( document ).ready(function() {
                                    $(".sliderStyle").css("width", 1150 + "px");
                                    });
                                    </script>
                                </head>
                                <body>
                                    <div id = "results" class = "currentImage"></div>
                                    <div id="slider" class = "sliderStyle"></div>
                                </body>
                            </html>

        """
 
        self.suffixes_text = "\t\t\t\t\tvar suffixes = ["
            
    def create_and_save_html(self, number_list):
        dir_with_word2vec = self.get_full_process_monitoring_dir_path_word2vec_or_not(True)
        if os.path.isdir(dir_with_word2vec):
            for c in self.entities:
                save_in = os.path.join(dir_with_word2vec, c + "_with_word2vec_" + self.HTML_NAME)
                print("Creating html for process using word2vec in " + save_in)
                self.create_html(save_in, c, dir_with_word2vec)
        else:
            print("No folder with results using word2vec found")

        dir_without_word2vec = self.get_full_process_monitoring_dir_path_word2vec_or_not(False)
        
        if os.path.isdir(dir_without_word2vec):
            for c in self.entities:
                save_in = os.path.join(dir_without_word2vec, c + "_without_word2vec_" + self.HTML_NAME)
                print("Creating html for process without using word2vec in " + save_in)
                self.create_html(save_in, c, dir_without_word2vec)
        else:
            print("No folder with results without using word2vec found")

    
    def create_html(self, file_to_save_in, entity_category, glob_dir):
        glob_exp = os.path.join(glob_dir, self.PLOT_PREFIX + "*" + entity_category + self.PLOT_FILE_ENDING)
        category_plots = glob.glob(glob_exp)
        category_plots_base_tuple = [(int(os.path.basename(el).split("_")[1]), os.path.basename(el))for el in category_plots]
        category_plots_base = [name for (nr, name) in sorted(category_plots_base_tuple)]
        
        plot_str = self.suffixes_text
        for plot in category_plots_base:
            plot_str = plot_str + '"' + plot + '", '
        plot_str = plot_str + "];"
        
        html = self.first_part_html + self.start_image_prefix + category_plots_base[0] + self.start_image_suffix + self.first_part_html_after_image + "\n" + plot_str + "\n" + self.second_part_html
        f = open(file_to_save_in, "w")
        f.write(html)
        f.close()
    
    def get_full_process_monitoring_dir_path_no_word2vec_info(self):
        full_process_monitoring_dir = os.path.join(self.path_slash_format, self.process_monitoring_dir)
        return full_process_monitoring_dir
    
    def get_full_process_monitoring_dir_path_word2vec_or_not(self, whether_to_use_word2vec):
        if whether_to_use_word2vec == True:
            sub_dir_name = self.FOLDER_FOR_WORD2VEC_TRUE
        else:
            sub_dir_name = self.FOLDER_FOR_WORD2VEC_FALSE
        return os.path.join(self.get_full_process_monitoring_dir_path_no_word2vec_info(), sub_dir_name)


    def init_process_monitoring(self, path_slash_format, properties, unlabelled_text_vector):
        if self.write_process_monitoring:
            full_process_monitoring_dir_path = self.get_full_process_monitoring_dir_path_no_word2vec_info()

            if not os.path.exists(full_process_monitoring_dir_path):
                text_concatenated = np.concatenate(unlabelled_text_vector)
                # To save storage space, only include the 10000 most frequent types
                word_vectorizer = CountVectorizer(binary = True, min_df=properties.min_df_current, \
                        max_df = properties.max_df_current, max_features = 10000)
                #word_vectorizer = CountVectorizer(binary = True, min_df=1, max_df = properties.max_df_current)
                word_vectorizer.fit_transform(text_concatenated)
                
                os.mkdir(full_process_monitoring_dir_path)
                vectorizer_full_name = os.path.join(full_process_monitoring_dir_path, self.VECTORIZER_NAME)
                joblib.dump(word_vectorizer, vectorizer_full_name, compress=9)
            
            full_process_monitoring_dir_path_word2vec_info = self.get_full_process_monitoring_dir_path()
            print("Will write process monitoring info in ", full_process_monitoring_dir_path_word2vec_info)
            #score_file_name = os.path.join(self.get_full_process_monitoring_dir_path(), "scores.csv")
 
            if not os.path.exists(full_process_monitoring_dir_path_word2vec_info):
                os.mkdir(full_process_monitoring_dir_path_word2vec_info)
                #self.score_file = open(score_file_name, "w")
                return True
            else:
                #self.score_file = open(score_file_name, "a")
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

    def get_lowest_conf_from_lst(self, conf_lst):
        return min(conf_lst)

    def get_variance_from_lst(self, conf_lst):
        return np.var(conf_lst)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    properties, path_slash_format, path_dot_format = active_learning_preannotation.load_properties(parser)
    html_creator_instance =  HtmlCreator(path_slash_format, properties)
    html_creator_instance.create_and_save_html([12, 14, 16])






