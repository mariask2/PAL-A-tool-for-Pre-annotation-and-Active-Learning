##############################
# Settings to use for evaluating the functionality of active learning on a
# a data set that has already been labelled
# This settings-file is not needed to run the standard functionality of PAL
############################

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


# The context around the current word to include when training the classifiers
number_of_previous_words = 2
number_of_following_words = 1


# The maxium number of sentences to search among when actively selecting useful 
# training samples
# The larger value that is given, the more data will be scanned for useful examples
# however, it will also take more time

maximum_samples_to_search_among = 1000

# Whether to include the current token as feature
use_current_word_as_feature = True

# A cut-off for the number of occurrences of a token in the data for it to be included as a feature 
min_df_current = 1
min_df_context = 3

# Vector length for the semantic vectors (300 for GoogleNews-vectors-negative300.bin, for instance.) 
semantic_vector_length = 300

# The path to the semantic space that is read by Gensim

model_path = '/Users/maria/mariaskeppstedtdsv/post-doc/gavagai/googlespace/GoogleNews-vectors-negative300.bin'

#######
# Type of model to use
#model_type = StructuredModelFrankWolfeSSVM
model_type = NonStructuredLogisticRegression

# If cross validation is to be used
# Only implemented for NonStructuredLogisticRegression
# Should be set to False if this category is not used
use_cross_validation = True


# Settings, typically not changed
#################################

labelled_data_dir = "labelled"
data_file_extension = ".csv"
unlabelled_data_dir = "unlabelled"
unlabelled_data_file = "unlabelled.csv"
tolabel_data_dir = "tolabel"
beginning_prefix = "B-"
inside_prefix = "I-"
outside_class = "O"
negative_category = "no"
nr_of_cross_validation_splits = 3


#########
# For simulation
#########

seed_set_size = 10
max_size = 100
step_size = 1
max_test_data_size = 400

# Split into train and test. 2 means equal, 3 means use 1/3 as test data, 4 means use 1/4 as test data and so on
train_test_splits = 3
