"""
Documentation of settings:
##############

minority_classes:
# Minority classes with their prefix. If the classes in the labelled data
# are not present in this list, they will be ignored
# They should be prefixed with "B" or "I" (matching beginning_prefix below)
minority_classes = ["B-speculation", "I-speculation", "B-contrast", "I-contrast"]

model_type:
# Type of model to use 
#model_type = StructuredModelFrankWolfeSSVM
#model_type = NonStructuredLogisticRegression

nr_of_samples:
# Number of sentences to be actively seleected and pre-annotated in each round
# nr_of_samples = 5

number_of_previous_words and number_of_following_words:
# The context around the current word to include when training the classifiers
#number_of_previous_words = 1
#number_of_following_words = 1

# A cut-off for the number of occurrences of a token in the data for it to be included as a feature
#min_df_current = 2
#min_df_context = 3

use_current_word_as_feature:
# Whether to include the current token as feature
#use_current_word_as_feature = True

model_path:
# The path to the semantic space that is read by Gensim
#model_path = '/mnt/data/maria/word2vec/GoogleNews-vectors-negative300.bin'

semantic_vector_length:
# Vector length for the semantic vectors (300 for GoogleNews-vectors-negative300.bin, for instance.)
#semantic_vector_length = 300

inactive_learning:
# This is to be set to False when active learning is to be applied 
# If it is true, it instead does the reverse, return the samples for which the learner is most certain
# This is to be used in early stages, when the classifier is too bad for doing a good pre-annotation. 
# Ensure to use as at least 50 samples with this option, so there will be a spread in what kinds of samples that are 
# extracted.                                                                                                               
# inactive_learning = True

maximum_samples_to_search_among:
# The maxium number of sentences to search among when actively selecting useful 
# training samples
# The larger value that is given, the more data will be scanned for useful examples
# however, it will also take more time
# "all" let's you search among all samples
#maximum_samples_to_search_among = "all"
#maximum_samples_to_search_among = 100

prefer_predicted_chunks:
# With this option set to True, the active learning prefers unlabelled samples in which chunks are predicted
# regardless of if they are more of less uncertain than the ones in which no chunks are predicted
# prefer_predicted_chunks = True

max_iterations:
# The number of iterations to use by the learning. See the pystruct documentation for 
# a suitable number (e.g. 1000). The larger, the longer things will take, so if the pre-annnotation take too much time
# reduce this nuber (which will also reduce the quality).
# Not used by the NonStructuredLogisticRegression class
#max_iterations = 10

current_word_vocabulary and context_word_vocabulary
# Decide what vocabulary to include based on an external list of words
# False, if no external list is to be used 
#current_word_vocabulary = False
#current_word_vocabulary = "/mnt/data/maria/stavicta_models/vocabulary/occurred_twice_or_more_in_pool.txt"
#context_word_vocabulary = "/mnt/data/maria/stavicta_models/vocabulary/occurred_twice_or_more_in_pool.txt"  
#context_word_vocabulary = False

use_cross_validation:
# If cross validation is to be used
# Only implemented for NonStructuredLogisticRegression
#use_cross_validation = True

c_value
# If 10-fold cross validation is chosen, this c_value is not used
#c_value = 1

# Settings, typically not changed, but that can be changed
#################################

#labelled_data_dir = "labelled"
#data_file_extension = ".csv"
#unlabelled_data_dir = "unlabelled"
#unlabelled_data_file = "unlabelled.csv"
#tolabel_data_dir = "tolabel"
#beginning_prefix = "B-"
#inside_prefix = "I-"
#outside_class = "O"
#negative_category = "no"
#whether_to_use_word2vec = True
#nr_of_cross_validation_splits = 2

"""

####################################################################
# Settings that typicallly need to be adapted to different projects
####################################################################

# Import of the classifiers that are possible to use

from classify_and_select import StructuredModelFrankWolfeSSVM
from classify_and_select import NonStructuredLogisticRegression



# Minority classes with their prefix. If the classes in the labelled data
# are not present in this list, they will be ignored
# They should be prefixed with "B" or "I" (matching beginning_prefix below)

minority_classes = ["B-speculation", "I-speculation"]
#minority_classes = ["B-speculation", "I-speculation", "B-contrast", "I-contrast"]

#######
# Type of model to use 
#model_type = StructuredModelFrankWolfeSSVM
model_type = NonStructuredLogisticRegression

# Number of sentences to be actively seleected and pre-annotated in each round
# nr_of_samples = 5

# The context around the current word to include when training the classifiers
#number_of_previous_words = 1
#number_of_following_words = 1

# A cut-off for the number of occurrences of a token in the data for it to be included as a feature
#min_df_current = 2
#min_df_context = 3

# A maximum cut-off for the number of occurrences of a token in the data for a token, or a neighbouring token to be included as a feature
# float in range [0.0, 1.0] or int. See the scikit learn documentation for the max_df for the class CountVectorizer
max_df_current = 0.95
max_df_context = 0.80

#######
#######

# Whether to include the current token as feature
#use_current_word_as_feature = True

# The path to the semantic space that is read by Gensim
model_path = '/Users/maria/mariaskeppstedtdsv/post-doc/gavagai/googlespace/GoogleNews-vectors-negative300.bin'
#model_path = '/Users/maria/mariaskeppstedtdsv/post-doc/gavagai/googlespace/word2vec_twitter_model/word2vec_twitter_model.bin'

# Vector length for the semantic vectors (300 for GoogleNews-vectors-negative300.bin, for instance.)
semantic_vector_length = 300

#######
# This is to be set to False when active learning is to be applied 
# If it is true, it instead does the reverse, return the samples for which the learner is most certain
# This is to be used in early stages, when the classifier is too bad for doing a good pre-annotation. 
# Ensure to use as at least 50 samples with this option, so there will be a spread in what kinds of samples that are 
# extracted.                                                                                                               
# inactive_learning = True

#########
# The maxium number of sentences to search among when actively selecting useful 
# training samples
# The larger value that is given, the more data will be scanned for useful examples
# however, it will also take more time
# "all" let's you search among all samples
#maximum_samples_to_search_among = "all"
#maximum_samples_to_search_among = 100

############
# With this option set to True, the active learning prefers unlabelled samples in which chunks are predicted
# regardless of if they are more of less uncertain than the ones in which no chunks are predicted
# prefer_predicted_chunks = True

# The number of iterations to use by the learning. See the pystruct documentation for 
# a suitable number (e.g. 1000). The larger, the longer things will take, so if the pre-annnotation take too much time
# reduce this nuber (which will also reduce the quality).
# Not used by the NonStructuredLogisticRegression class
#max_iterations = 10



######
# Decide what vocabulary to include based on an external list of words
# False, if no external list is to be used 
#current_word_vocabulary = False
#current_word_vocabulary = "/mnt/data/maria/stavicta_models/vocabulary/occurred_twice_or_more_in_pool.txt"
#context_word_vocabulary = "/mnt/data/maria/stavicta_models/vocabulary/occurred_twice_or_more_in_pool.txt"  
#context_word_vocabulary = False

# If cross validation is to be used
# Only implemented for NonStructuredLogisticRegression
#use_cross_validation = True

# If 10-fold cross validation is chosen, this c_value is not used
#c_value = 1

# Settings, typically not changed
#################################

#labelled_data_dir = "labelled"
#data_file_extension = ".csv"
#unlabelled_data_dir = "unlabelled"
#unlabelled_data_file = "unlabelled.csv"
#tolabel_data_dir = "tolabel"
#beginning_prefix = "B-"
#inside_prefix = "I-"
#outside_class = "O"
#negative_category = "no"
#whether_to_use_word2vec = True
whether_to_use_clustering = True
#nr_of_cross_validation_splits = 2
