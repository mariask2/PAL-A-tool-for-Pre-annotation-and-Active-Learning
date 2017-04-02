####################################################################
# Settings that typicallly need to be adapted to different projects
####################################################################

# Import of the classifiers that are possible to use
from classify_and_select import StructuredModelFrankWolfeSSVM
from classify_and_select import NonStructuredLogisticRegression

# Minority classes with their prefix. If the classes in the labelled data
# are not present in this list, they will be ignored
# They should be prefixed with "B" or "I" (matching beginning_prefix below)


# Number of sentences to be actively seleected and pre-annotated in each round

nr_of_samples = 5


# The maxium number of sentences to search among when actively selecting useful 
# training samples
# The larger value that is given, the more data will be scanned for useful examples
# however, it will also take more time

maximum_samples_to_search_among = "all"

# The context around the current word to include when training the classifiers
number_of_previous_words = 1
number_of_following_words = 1

# Whether to include the current token as feature
use_current_word_as_feature = True

# A cut-off for the number of occurrences of a token in the data for it to be included as a feature
min_df_current = 1
min_df_context = 2


# This is to be set to False when active learning is to be applied 
# If it is true, it instead does the reverse, return the samples for which the learner is most certain
# This is to be used in early stages, when the classifier is too bad for doing a good pre-annotation. 
# Ensure to use as at least 50 samples with this option, so there will be a spread in what kinds of samples that are 
# extracted.                                                                                                               

inactive_learning = False


############
# With this option set to True, the active learning prefers unlabelled samples in which chunks are predicted
# regardless of if they are more of less uncertain than the ones in which no chunks are predicted
prefer_predicted_chunks = False

# The number of iterations to use by the learning. See the pystruct documentation for 
# a suitable number (e.g. 1000). The larger, the longer things will take, so if the pre-annnotation take too much time
# reduce this nuber (which will also reduce the quality).
# Only used by the structured prediction
max_iterations = 1000

#######
# Type of model to use (There is only one type available, but to prefer for future ones.)

#model_type = StructuredModelFrankWolfeSSVM
model_type = NonStructuredLogisticRegression

#####
# If the model is to be saved, when a evaluation against an external reference standard is carried out
save_model = True

######
# Decide what vocabulary to include based on an external list of words
# False, if no external list is to be used 
current_word_vocabulary = False
context_word_vocabulary = False

# If cross validation is to be used
# Only implemented for NonStructuredLogisticRegression
# Should be set to False if this category is not used
use_cross_validation = False

# If 10-fold cross validation is chosen, this c_value is not used
c_value = 1

# Settings, typically not changed
#################################

labelled_data_dir = "labelled"
data_file_extension = ".csv"
unlabelled_data_dir = "unlabelled"
unlabelled_data_file = "unlabelled.csv"
tolabel_data_dir = "tolabel"
evaluation_output_dir = "evaluation"
separate_evaluation_output_dir = "separate_evaluation"
separate_evaluation_data_dir = "separate_evaluation_data"
saved_model_dir = "saved_model"
beginning_prefix = "B-"
inside_prefix = "I-"
outside_class = "O"
negative_category = "no"
whether_to_use_word2vec = False
whether_to_use_clustering = False
nr_of_cross_validation_splits = 10
