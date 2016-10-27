####################################################################
# Settings that typicallly need to be adapted to different projects
####################################################################

# Minority classes with their prefix. If the classes in the labelled data
# are not present in this list, they will be ignored
# They should be prefixed with "B" or "I" (matching beginning_prefix below)

minority_classes = ["B-speculation", "I-speculation", "B-contrast", "I-contrast"]

# Number of sentences to be actively seleected and pre-annotated in each round

nr_of_samples = 5


# The maxium number of sentences to search among when actively selecting useful 
# training samples
# The larger value that is given, the more data will be scanned for useful examples
# however, it will also take more time

maximum_samples_to_search_among = "all"


# The path to the semantic space that is read by Gensim

model_path = '/myword2vecpath/GoogleNews-vectors-negative300.bin'

# This is to be set to False when active learning is to be applied                                                                                                           
# If it is true, it instead does the reverse, return the samples for which the learner is most certain                                                                       
# This is to be used in early stages, when the classifier is too bad for doing a good pre-annotation.                                                                        
# Ensure to use as at least 50 samples with this option, so there will be a spread in what kinds of samples that are                                                         
# extracted.                                                                                                                                                                  
inactive_learning = False


# The number of iterations to use by the learning. See the pystruct documentation for 
# a suitable number (e.g. 1000). The larger, the longer things will take, so if the pre-annnotation take too much time
# reduce this nuber (which will also reduce the quality).
max_iterations = 10

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
whether_to_use_word2vec = True
