####################################################################
# Settings that typicallly need to be adapted to different projects
####################################################################

# Minority classes with their prefix. If the classes in the labelled data
# are not present in this list, they will be ignored
# They should be prefixed with "B" or "I" (matching beginning_prefix below)

minority_classes = ["B-speculation", "I-speculation", "B-contrast", "I-contrast"]

# Number of sentences to be actively seleected and pre-annotated in each round

nr_of_samples = 20


# The maxium number of sentences to search among when actively selecting useful 
# training samples
# The larger value that is given, the more data will be scanned for useful examples
# however, it will also take more time

maximum_samples_to_search_among = 1000


# The path to the semantic space that is read by Gensim

model_path = '/word2vec/GoogleNews-vectors-negative300.bin'



# Settings, typically not changed
#################################

labelled_data_dir = "labelled"
data_file_extension = ".csv"
unlabelled_data_dir = "unlabelled"
unlabelled_data_file = "unlabelled.csv"
tolabel_data_dir = "tolabel"
beginning_prefix = "B-"
outside_class = "O"
whether_to_use_word2vec = True
