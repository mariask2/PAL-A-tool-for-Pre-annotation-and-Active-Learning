import numpy as np
#from pystruct.models import ChainCRF
#from pystruct.learners import FrankWolfeSSVM
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from process_monitoring import ProcessMonitor


#from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import StratifiedKFold



def get_new_data(X_labelled_np, X_unlabelled_np, y_labelled_np, text_vector_labelled_np, text_vector_unlabelled_np, \
                     label_dict, minority_categories, nr_of_samples,  maximum_samples_to_search_among, outside_class, \
                     beginning_prefix, inside_prefix, inactive_learning, max_iterations, prefer_predicted_chunks, \
                     model_type, use_cross_validation, nr_of_cross_validation_splits, c_value, process_monitoring_instance):

    """

    get_new_data is the main function of this module. It is the function to call to get actively selected and pre-annotated data

    :param X_labelled_np: A numpy.ndarray containing the features representing the labelled data.
    Ex:
    [ array([[0, 0, 0, ..., 0, 0, 1],
    [0, 0, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0],
    ..., 
    [0, 1, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0]])
    array([[0, 0, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0],
    ...,

    :param X_unlabelled_np: A numpy.ndarray containing the features representing the unlabelled data.
    Ex:
    [ array([[0, 0, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0],
    ..., 
    [0, 0, 0, ..., 0, 0, 0],
    [0, 1, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0]])
    array([[0, 0, 0, ..., 0, 0, 1],
    [0, 0, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0],
    ..., 
   
    :param y_labelled_np: A numpy.ndarray containing the classes of the labelled data.
    Ex:
    [array([2, 2, 0, 2, 2, 2, 2, 2, 2, 2]) 
    array([2, 2, 2, 2, 0, 2, 2, 2, 2])
    array([2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    array([2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2])
    
    :param text_vector_labelled_np: A numpy.ndarray containing the tokens in the samples in the labelled data.
    Ex:
    [ array(['2_2', 'you', 'could', 'see', 'someone', 'moving,', 'regardless',
       'of', 'the', 'darkness'], 
      dtype='<U10')
      array(['5_5', 'it', "'_'", 's_s', 'certainly', 'something', 'to',
       'consider', '._.'], 
      dtype='<U9')
      array(['7_7', 'perhaps', 'there', 'is', 'a_a', 'better', 'way', 'of',
       'doing', 'it', '._.'], 
    ...
    
    :param text_vector_unlabelled_np: A numpy.ndarray containing the tokens in the samples in the unlabelled data.
    Ex:
    [ array(['1_1', 'it', 'rained', 'today', 'despite', 'what', 'was', 'said',
       'on', 'the', 'news', '._.'], 
      dtype='<U7')
      array(['3_3', 'they', 'decided', 'to', 'follow', 'it', 'in', 'spite', 'of',
       'the', 'warnings', '._.'],  
    ...   
    
    :param label_dict: A dictionary where the keys are the numerical representations of the classes, and the items are
    the classes in the form they appear in the annotated data
    Ex:
    {'O': 2, 'I-speculation': 1, 'B-speculation': 0}
    
    :param minority_categories: List of minority classes with their prefix. 
    
    :param nr_of_samples: Number of sentences to be actively seleected and pre-annotated in each round.
    
    :param maximum_samples_to_search_among: The maxium number of sentences to search among when actively selecting useful (integer or the string "all")
    
    :param outside_class: The name of the outside class, i.e., the one not among the mindory classes (typically "O")
    
    :param beginning_prefix: The prefix to use on a minority class to indicate beginning of class (typically "B-)
    
    :param inside_prefix: The prefix to use on a minority class to indicate inside of class (typically "I-)
    
    :param inactive_learning: If this is set to True, the reverse of active learning is to be used (typically this is set to False, therefore)
    
    :param max_iterations: The number of iterations to use by the learning (not applicable to all classes)
    
    :param prefer_predicted_chunks:  With this option set to True, the active learning prefers unlabelled samples in which chunks are predicted (typically this is set to False, therefore) 
    
    :param model_type: Type of model to use, e.g. StructuredModelFrankWolfeSSVM or NonStructuredLogisticRegression
    
    :param use_cross_validation: If cross validation is to be used to decide to decide c-value (True or False)
    
    :param nr_of_cross_validation_splits: Nr of splits in cross validation (only relevant if use_cross_validation is True)
    
    :param c_value: c_value to use (only relevant if use_cross_validation is False)
 
    :return: to_select_X: A numpy.ndarray containing the features representing the selected data
    Ex:
    [ array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ..., 
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 1, 0, 0]])
       array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ..., 
    
    :return: unlabelled_x: A numpy.ndarray containing the features representing the still unlabelled data
    Ex:
    [ array([[0, 0, 0, ..., 0, 1, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ..., 
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 1, 0]])
       array([[0, 0, 0, ..., 0, 0, 1],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ..., 
    
    :return: to_select_text: A list containing numpy.ndarray with the tokens in the text that is selected for annotation and pre-labelling
    Ex:
    [array(['14', 'try', 'to', 'see', 'it', 'my', 'way', '!_!'], 
      dtype='<U3'), array(['9_9', 'the', 'certainty', 'is', 'quite', 'low', '._.'], 
      dtype='<U9'), array(['1_1', 'it', 'rained', 'today', 'despite', 'what', 'was', 'said',
       'on', 'the', 'news', '._.'], 
      dtype='<U7'), array(['3_3', 'they', 'decided', 'to', 'follow', 'it', 'in', 'spite', 'of',
       'the', 'warnings', '._.'], 
      dtype='<U8')]
     :return: sentences_unlabelled: A numpy.ndarray with samples containing the tokens in the text that is not selected, and which is 
      therefore to remain in the pool of unlabelled data
     [array(['3_3', 'they', 'decided', 'to', 'follow', 'it', 'in', 'spite', 'of',
       'the', 'warnings', '._.'], 
      dtype='<U8')
      array(['4_4', 'it', 'is', 'clearly', 'wrong', '._.'], 
      dtype='<U7')
      ...
    
     :return: predicted_for_selected: A list of predictions made by the currently trained model on the selected data
     Ex:
     [array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), array([2, 2, 2, 2, 2, 2]), 
     array([2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2]), array([2, 2, 2, 2, 2, 2, 2]), array([2, 2, 2, 2, 0, 2, 2, 2, 2])]
    """

    maximum_samples_to_search_among = get_maximum_samples_to_search_among(maximum_samples_to_search_among, X_unlabelled_np, nr_of_samples)
    
    model = model_type(label_dict, minority_categories, outside_class, beginning_prefix, inside_prefix, max_iterations, \
                           use_cross_validation, nr_of_cross_validation_splits, c_value)
    print("Started to train the model on the labelled data")
    model.fit(X_labelled_np, y_labelled_np)
    print("Training on labelled data finished")

    to_select_X, unlabelled_x, to_select_text, sentences_unlabelled, predicted_for_selected = \
        model.get_selected_unlabelled(X_labelled_np, y_labelled_np, X_unlabelled_np, nr_of_samples, text_vector_labelled_np, \
                               text_vector_unlabelled_np,  maximum_samples_to_search_among, inactive_learning, prefer_predicted_chunks,\
                                      process_monitoring_instance)

    #print(predicted_for_selected)
    #print(predicted_for_selected.__class__.__name__)

    return(to_select_X, unlabelled_x, to_select_text, sentences_unlabelled, predicted_for_selected)


def get_maximum_samples_to_search_among(maximum_samples_to_search_among, X_unlabelled_np, nr_of_samples):
    """
    get_maximum_samples_to_search_among internal function used by the module

    Returns the number of samples that should be searched among. 
    If maximum_samples_to_search_among.lower() == "all", it returns the number of unlabelled samples that are available.
    If maximum_samples_to_search_among is a number it returns the the mininum of this number and the the number of unlabelled samples that are available.
    """
    try:
        nr_of_samples_int = int(nr_of_samples)

        if isinstance(maximum_samples_to_search_among, str) and maximum_samples_to_search_among.lower() == "all":
            print("Will search for samples to select among all " + str(len(X_unlabelled_np)) + " unlabelled samples.") 
            return len(X_unlabelled_np)    
        try:
            maximum_samples_to_search_among_int = int(maximum_samples_to_search_among)


            if maximum_samples_to_search_among_int > len(X_unlabelled_np):
                print("There are " + str(len(X_unlabelled_np)) + " unlabelled samples left. Will search for samples to select among those.") 
                return len(X_unlabelled_np)
            else:
                print("Will search for samples to select among " + str(maximum_samples_to_search_among_int) + " randomly selected unlabelled samples.")
                return maximum_samples_to_search_among_int
        except ValueError:
            print("The property maximum_samples_to_search_among can only take a numerical value or the string 'all', " + \
                      maximum_samples_to_search_among + " is not valid.")
            exit(1)
    except ValueError:
        print("The property nr_of_samples can only take a numerical value, " + str(nr_of_samples) + " is not valid.")
        exit(1)




#####
# Classes
#####

#Abstract class
class ModelWrapperBase:
    """
    ModelWrapperBase is an abstract class that all models to be used in the active learning and pre-annotation framework
    are to be subclasses of (it's not enforced, but it's recommended).

    """
    # Abstract methods, to show what needs to be implemented
    def fit(self, X, Y):
        """
        Fits the model on the training data

        params: X: The feature representation of the training data. ndarray, with each element representing a sample, which in turn
        has an array representing each token.
        Ex:
        [ array([[0, 0, 0, ..., 0, 0, 1],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ..., 
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 1, 0]])
        array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,

        params: Y: The numerical representation of the classifications in the training data. ndarray, with each element representing a sample, which in turn
        has a numerical representing for each token.
        Ex:

        [array([2, 2, 0, 2, 2, 2, 2, 2, 2, 2]) array([2, 2, 2, 2, 0, 2, 2, 2, 2])
        array([2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 2, 0, 1, 1, 2])
        array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 0, 2, 2, 2, 2, 2, 0, 2]) array([2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]) array([2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2])
        ...

        raises a NotImplementedError in ModelWrapperBase, and is to be implemented in the subclasses
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Predics the classes given the observations in X

        params: X: The feature representation of the training data. ndarray, with each element representing a sample, which in turn
        has an array representing each token.
        Ex:
        [ array([[0, 0, 0, ..., 0, 0, 1],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ..., 
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 1, 0]])
        array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,

        is to returns the numerical representation of the classifications resulting from the predictions in form of a list
        Ex:
        [array([2, 2, 2, 2, 2, 2]), array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), array([2, 2, 2, 2, 2, 2]), 
        array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), 
        array([2, 2, 2, 2, 2, 2, 2, 2, 2]), array([2, 2, 2, 1, 1, 1, 2, 2]), array([2, 2, 0, 2, 2, 2, 2, 2, 2, 2]),
        ...

        raises a NotImplementedError in ModelWrapperBase, and is to be implemented in the subclasses
        """
        raise NotImplementedError

    def score(self, X, Y):
        """
        Is to return a goodness-score for the classification Y, given the feature representation X

        params: X: The feature representation of the training data. ndarray, with each element representing a sample, which in turn
        has an array representing each token.
        Ex:
        [ array([[0, 0, 0, ..., 0, 0, 1],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ..., 
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 1, 0]])
        array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,

        params: Y: The numerical representation of the classifications in the training data. ndarray, with each element representing a sample, which in turn
        has a numerical representing for each token.
        Ex:

        [array([2, 2, 0, 2, 2, 2, 2, 2, 2, 2]) array([2, 2, 2, 2, 0, 2, 2, 2, 2])
        array([2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 2, 0, 1, 1, 2])
        array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 0, 2, 2, 2, 2, 2, 0, 2]) array([2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]) array([2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        array([2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2])
        ...

        raises a NotImplementedError in ModelWrapperBase, and is to be implemented in the subclasses                                                                               
        """
        raise NotImplementedError
        

    def init_params(self, label_dict, minority_classes, outside_class, beginning_prefix, inside_prefix, max_iterations, \
                        use_cross_validation, nr_of_cross_validation_splits, c_value):
        """
        Method for setting all the parameters of the modelwrapper (as specified by the properties).
        """
        self.outside_class = outside_class
        self.beginning_prefix = beginning_prefix
        self.inside_prefix = inside_prefix
        self.use_cross_validation = use_cross_validation
        self.nr_of_cross_validation_splits = nr_of_cross_validation_splits
        self.c_value = c_value

        self.max_iterations = max_iterations

        self.label_dict = label_dict
        self.minority_classes = minority_classes
        self.minority_classes_index = []
        for el in self.minority_classes:
            print("class", el)
            self.minority_classes_index.append(self.label_dict[el])
    
        self.inv_label_dict = {v: k for k, v in label_dict.items()}
        for el in self.inv_label_dict.keys():
            if el not in self.minority_classes_index:
                self.majority_class = el
        print("self.majority_class", self.majority_class)
        
        # These two variables are only set if the model is to be saved
        self.current_word_vectorizer = None
        self.context_word_vectorizer = None
        self.beginning_category = None
        self.inside_category = None


    def predict_nonstructured(self, X):
        """
        Help method for non structured models. Turns the observations in X from a structured format to a nonstructured format,
        predicts with the internal model, and transforms back to a structured format.

        params: X: The feature representation of the training data. ndarray, with each element representing a sample, which in turn
        has an array representing each token.
        Ex:
        [ array([[0, 0, 0, ..., 0, 0, 1],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ..., 
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 1, 0]])
        array([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,

        is to returns the numerical representation of the classifications resulting from the predictions in form of a list
        Ex:
        [array([2, 2, 2, 2, 2, 2]), array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), array([2, 2, 2, 2, 2, 2]), 
        array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), 
        array([2, 2, 2, 2, 2, 2, 2, 2, 2]), array([2, 2, 2, 1, 1, 1, 2, 2]), array([2, 2, 0, 2, 2, 2, 2, 2, 2, 2]),
        ...

        """
        X_flat = np.concatenate(X)
        predicted =  self.model.predict(X_flat)
        X_flat_counter = 0
        predicted_in_sentences = []
        for sentence in X:
            predicted_for_sentence = []
            for word in sentence:
                predicted_for_sentence.append(predicted[X_flat_counter])
                X_flat_counter  = X_flat_counter + 1
            predicted_in_sentences.append(predicted_for_sentence)
        return predicted_in_sentences

    def get_scores_unlabelled_with_predicted_chunks(self, to_search_among_x, ys, selected_indeces, sentences_unlabelled):
        """
        get_scores_unlabelled_with_predicted_chunks is to return the certainty scores for the classifications for the unlabelled data with
        selected_indeces. (Where selected_indeces are randomly selected indeces from the pool of unlabelled data to be used in the data selection. 
        The len of is decided by the settings parameter maximum_samples_to_search_among, and if this paramter is set to "all", is is all the
        samples in the pool of unlabelled data.
     
        params: to_search_among_x: An ndarray, with each element representing a sample corresponding to the index in the randomly selected
        indeces in the pool of unlabelled data, which in turn has an array representing each token.
        Ex:
        [array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 1, 0, 0]]), array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ..., 
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 1, 0, 0]]), array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...

        params: ys: The classification made by the current model for these randomly selected unlabelled samples
        Ex:
        [array([2, 2, 2, 2, 2, 2]), array([2, 2, 2, 2, 2, 2, 2, 2, 2]), array([2, 2, 2, 2, 0, 2, 2, 2, 2]),
        ...

        params: selected_indeces: The indeces of the randomly selected samples from the pool of unlabelled data.
        Ex:
        [11, 15, 4, 0, 3, 8, 12, 14, 13, 16, 10, 9, 7, 6, 19, 2, 5, 1, 17, 18, ...
        params: sentences_unlabelled: The tokens in the 
        Ex:
        [ array(['1_1', 'it', 'rained', 'today', 'despite', 'what', 'was', 'said',
       'on', 'the', 'news', '._.'], 
      dtype='<U7')
 array(['2_2', 'you', 'could', 'see', 'someone', 'moving,', 'regardless',
       'of', 'the', 'darkness'],
       ..

       returns: scores_with_index: a list of tuples of (certainty-score, index, classification, tokens) for the samples
       that contain predicted chunks
        Ex:
        [(1.3228231538976338, 19, array([2, 2, 2, 1, 1, 1, 2, 2]), array(['19', 'i_i', 'couln', "'_'", 't_t', 'believe', 'you', '._.'], 
      dtype='<U7')), (1.2375469333946558, 1, array([2, 2, 0, 2, 2, 2, 2, 2, 2, 2]), array(['2_2', 'you', 'could', 'see', 'someone', 'moving,', 'regardless',
       'of', 'the', 'darkness'], 
      dtype='<U10')), (0.63244446618097783, 6, array([2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2]), array(['7_7', 'perhaps', 'there', 'is', 'a_a', 'better', 'way', 'of',
       'doing', 'it', '._.'], 

        returns: index_in_which_no_minority_categories_are_predicted: a list of tuples for the samples that do not contain predicted chunks, with
        (xi, yi, index)
        Ex:
        [(array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ..., 
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 1, 0]]), array([2, 2, 2, 2, 2, 2, 2, 2]), 13)]

        raises a NotImplementedError in ModelWrapperBase, and is to be implemented in the subclasses                                                                              
        """
        raise NotImplementedError

    def get_scores_unlabelled_sorted_no_predicted_chunks(self, number_of_unlabelled_to_select, index_in_which_no_minority_categories_are_predicted,\
                                                             sentences_unlabelled, inactive_learning):
        """
        Is to return the score for those with no predicted chunks in sorted order, and only to return the number_of_unlabelled_to_select best of them
        (where number_of_unlabelled_to_select is equal to all of them if the setting to NOT prioritize samples with predicted chunks is chosen)
        
        returns: a sorted list of tuples of (certainty-score, index, classification, tokens)
        Ex:
        [(0.96155715070017145, 8, array([2, 2, 2, 2, 2, 2]), array(['13', 'quite', 'hard', 'to', 'believe', '._.'], 
      dtype='<U7')), (1.0611824917072603, 0, array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), array(['1_1', 'it', 'rained', 'today', 'despite', 'what', 'was', 'said',
       'on', 'the', 'news', '._.'], 
      dtype='<U7')), (1.0611824917072612, 3, array([2, 2, 2, 2, 2, 2]), array(['4_4', 'it', 'is', 'clearly', 'wrong', '._.'], 
      dtype='<U7')),
      ...
        raises a NotImplementedError in ModelWrapperBase, and is to be implemented in the subclasses                                                                               
        """
        raise NotImplementedError

    def get_selected_unlabelled(self, labelled_x, labelled_y, unlabelled_x, step_size, sentences_labelled, sentences_unlabelled,    maximum_samples_to_search_among,\
        inactive_learning, prefer_predicted_chunks,\
                                process_monitoring_instance):
        """
        get_new_data is the main function of this module. It is the function to call to get actively selected and pre-annotated data
        This method should only be called after the fit method has been called. Otherwise, and sklearn.utils.validation.NotFittedError will be raised
        The method is called to retrieve the actively selected and pre-annotated data.
        
        The main code for sorting and selecting is carried out in this method, but the code for determining a score for the certainty of the unlabelled data
        is left for the subclasses (since this score is computed differently for different models and active learning methods). For determining these scores,
        the method invokes:
        get_scores_unlabelled_with_predicted_chunks and get_scores_unlabelled_sorted_no_predicted_chunks that is implemented in the non-abstract subclasses.
        For insuring there is a lexical spread within the selected samples, the help function get_selected_sentences_with_different_vocabulary is called.

    :param labelled_x: A numpy.ndarray containing the features representing the labelled data.
    Ex:
    [ array([[0, 0, 0, ..., 0, 0, 1],
    [0, 0, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0],
    ..., 
    [0, 1, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0]])
    array([[0, 0, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0],
    ...,

    :param y_labelled_np: A numpy.ndarray containing the classes of the labelled data.
    Ex:
    [array([2, 2, 0, 2, 2, 2, 2, 2, 2, 2]) 
    array([2, 2, 2, 2, 0, 2, 2, 2, 2])
    array([2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    array([2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2])

    :param X_unlabelled_np: A numpy.ndarray containing the features representing the unlabelled data.
    Ex:
    [ array([[0, 0, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0],
    ..., 
    [0, 0, 0, ..., 0, 0, 0],
    [0, 1, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0]])
    array([[0, 0, 0, ..., 0, 0, 1],
    [0, 0, 0, ..., 0, 0, 0],
    [0, 0, 0, ..., 0, 0, 0],
    ..., 
   
    :param nr_of_samples: Nr of samples to return

    :param text_vector_labelled_np: A numpy.ndarray containing the tokens in the samples in the labelled data.
    Ex:
    [ array(['2_2', 'you', 'could', 'see', 'someone', 'moving,', 'regardless',
       'of', 'the', 'darkness'], 
      dtype='<U10')
      array(['5_5', 'it', "'_'", 's_s', 'certainly', 'something', 'to',
       'consider', '._.'], 
      dtype='<U9')
      array(['7_7', 'perhaps', 'there', 'is', 'a_a', 'better', 'way', 'of',
       'doing', 'it', '._.'], 
    ...
    
    :param text_vector_unlabelled_np: A numpy.ndarray containing the tokens in the samples in the unlabelled data.
    Ex:
    [ array(['1_1', 'it', 'rained', 'today', 'despite', 'what', 'was', 'said',
       'on', 'the', 'news', '._.'], 
      dtype='<U7')
      array(['3_3', 'they', 'decided', 'to', 'follow', 'it', 'in', 'spite', 'of',
       'the', 'warnings', '._.'],  
    ...   
    
    :param maximum_samples_to_search_among: The maxium number of sentences to search among when actively selecting useful (integer or the string "all")
        
    :param inactive_learning: If this is set to True, the reverse of active learning is to be used (typically this is set to False, therefore)
    
    :param prefer_predicted_chunks:  With this option set to True, the active learning prefers unlabelled samples in which chunks are predicted (typically this is set to False, therefore) 
     
    :return: to_select_X: A numpy.ndarray containing the features representing the selected data
    Ex:
    [ array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ..., 
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 1, 0, 0]])
       array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ..., 
    
    :return: unlabelled_x: A numpy.ndarray containing the features representing the still unlabelled data
    Ex:
    [ array([[0, 0, 0, ..., 0, 1, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ..., 
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 1, 0]])
       array([[0, 0, 0, ..., 0, 0, 1],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ..., 
    
    :return: to_select_text: A list containing numpy.ndarray with the tokens in the text that is selected for annotation and pre-labelling
    Ex:
    [array(['14', 'try', 'to', 'see', 'it', 'my', 'way', '!_!'], 
      dtype='<U3'), array(['9_9', 'the', 'certainty', 'is', 'quite', 'low', '._.'], 
      dtype='<U9'), array(['1_1', 'it', 'rained', 'today', 'despite', 'what', 'was', 'said',
       'on', 'the', 'news', '._.'], 
      dtype='<U7'), array(['3_3', 'they', 'decided', 'to', 'follow', 'it', 'in', 'spite', 'of',
       'the', 'warnings', '._.'], 
      dtype='<U8')]
     :return: sentences_unlabelled: A numpy.ndarray with samples containing the tokens in the text that is not selected, and which is 
      therefore to remain in the pool of unlabelled data
     [array(['3_3', 'they', 'decided', 'to', 'follow', 'it', 'in', 'spite', 'of',
       'the', 'warnings', '._.'], 
      dtype='<U8')
      array(['4_4', 'it', 'is', 'clearly', 'wrong', '._.'], 
      dtype='<U7')
      ...
    
     :return: predicted_for_selected: A list of predictions made by the currently trained model on the selected data
     Ex:
     [array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), array([2, 2, 2, 2, 2, 2]), 
     array([2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2]), array([2, 2, 2, 2, 2, 2, 2]), array([2, 2, 2, 2, 0, 2, 2, 2, 2])]
     """



        # Check the number of samples to select
        if step_size == 0:
            print("You have chosen to select 0 new samples to pre-annotated. The variable 'nr_of_samples' in 'settings.py' should be at least 1")
            exit(1)
        if step_size > len(unlabelled_x):
            print("More samples have been asked for than exist among unlabelled. A maximum of " + str(len(unlabelled_x)) + " nr of samples can be returned")
            step_size = len(unlabelled_x)


        # Randomly select samples among which to search for to search for the most informative training instance
        selected_indeces = shuffle(range(0, len(unlabelled_x)))[:maximum_samples_to_search_among]
        to_search_among_x = []
        for selected_index in selected_indeces:
            to_search_among_x.append(unlabelled_x[selected_index])
        ys = self.predict(to_search_among_x)
        print("Requested a search among a maximum of " + str(maximum_samples_to_search_among) + " samples")

        # Get scores for the unlabelled samples for which a minority category has been predicted
        scores_with_index, index_in_which_no_minority_categories_are_predicted = \
            self.get_scores_unlabelled_with_predicted_chunks(to_search_among_x, ys, selected_indeces,\
                                                             sentences_unlabelled, process_monitoring_instance)

        # if there are too few samples among the unlabelled in which minority categoies are predict, also return unlabelled samples without minority categories
        # or if the setting is chosen to don't prefer samples in which minority categores are predicted, compute certainty score for all those unlabelled
        if len(scores_with_index) < step_size or not prefer_predicted_chunks and len(index_in_which_no_minority_categories_are_predicted) > 0:
            if len(scores_with_index) < step_size:
                number_of_unlabelled_to_select = step_size - len(scores_with_index)
            else: # i.e. not prefer_predicted_chunks 
                number_of_unlabelled_to_select = len(index_in_which_no_minority_categories_are_predicted) # include all of them, and filter out later
            print("Will search among " + str(number_of_unlabelled_to_select) + " without labelled chunks.")
            sorted_indeces_no_predicted_chunks = \
                self.get_scores_unlabelled_sorted_no_predicted_chunks(number_of_unlabelled_to_select,\
                                                                          index_in_which_no_minority_categories_are_predicted, sentences_unlabelled, inactive_learning)
        else:
            print("Will search for the best ones among the " + str(len(scores_with_index)) + " samples that contained a minority category prediction.")
            sorted_indeces_no_predicted_chunks = [] # nothing without predicted chunks included


        index_to_select_among_checked = \
            get_selected_sentences_with_different_vocabulary(scores_with_index, sorted_indeces_no_predicted_chunks,\
                                                                 step_size, self.majority_class, inactive_learning, prefer_predicted_chunks)

        to_select_X = []
        to_select_text = []
        predicted_for_selected = []
        for its in index_to_select_among_checked:
            to_select_X.append(unlabelled_x[its])
            to_select_text.append(sentences_unlabelled[its])
            predicted_for_selected.append(self.predict(unlabelled_x[its:its+1])[0]) # must submit an numpy array to predict
        print("__________________________")

        unlabelled_x = np.delete(unlabelled_x, index_to_select_among_checked, 0)
        sentences_unlabelled = np.delete(sentences_unlabelled, index_to_select_among_checked, 0)
        to_select_X = np.array(to_select_X)

        process_monitoring_instance.write_process_monitoring_selected_words(index_to_select_among_checked, self.inv_label_dict)
        
        return to_select_X, unlabelled_x, to_select_text, sentences_unlabelled, predicted_for_selected

    def get_params(self):
        return self.model.get_params()

    def get_cs(self):
        return "No cs for this model"


class StructuredModelFrankWolfeSSVM(ModelWrapperBase):
    def __init__(self, label_dict, minority_classes, outside_class, beginning_prefix, inside_prefix, max_iterations, \
                     use_cross_validation, nr_of_cross_validation_splits, c_value):
        if use_cross_validation:
            raise NotImplementedError("Cross validatio not implemented for StructuredModelFrankWolfeSSVM")
        self.model = ChainCRF()
        self.__name__ = "StructuredModelFrankWolfeSSVM"
        self.init_params(label_dict, minority_classes, outside_class, beginning_prefix, inside_prefix, max_iterations, \
                             use_cross_validation, nr_of_cross_validation_splits, c_value)
    
    def get_params(self):
        return "Params undefined for StructuredModelFrankWolfeSSVM"


    def fit(self, X, Y):
        self.model = ChainCRF() # make a new model each time
        self.ssvm = FrankWolfeSSVM(model=self.model, max_iter=self.max_iterations, C=self.c_value)
        ret = self.ssvm.fit(X, Y)
        return ret

    def predict(self, X):
        try:
            return self.ssvm.predict(X)
        except ValueError as e:
            print("Predict failed, perhaps one feature set was used for training the model, and another feature set is used when predicting")
            print(str(e))
            exit(1)

    def score(self, X, Y):
        return self.ssvm.score(X,Y)

    def get_smallest_diff_alternative(self, xi, yi, permutation_method):
        joint = self.model.joint_feature(xi, yi)
        score = np.dot(self.ssvm.w, joint)
        #print("score", score)
        
        # min_difference is the score difference between the predicted classification and the second best classification
        min_difference = float("inf")

        alternative_score_to_compare_with = None

        # for alternatives to annotated chunks 'get_permutations_with_predicted_chunks' is the permutation_method
        # for alternatives when there are no annotated chunks 'get_permutations_no_predicted_chunks' is the permutation_method
        yi_alternatives = permutation_method(yi, self)
    

        best_alternative = None
        for yi_alternative in yi_alternatives:
            joint_alternative = self.model.joint_feature(xi, yi_alternative)
            score_alternative = np.dot(self.ssvm.w, joint_alternative)

            difference_between_predicted_and_alternative = score - score_alternative

            if difference_between_predicted_and_alternative < min_difference:
                min_difference = difference_between_predicted_and_alternative
                alternative_score_to_compare_with = score_alternative
                best_alternative = yi_alternative
        #print("best_alternative", best_alternative)
        #print("alternative_score_to_compare_with", alternative_score_to_compare_with)
        return min_difference

    def get_scores_unlabelled_with_predicted_chunks(self, to_search_among_x, ys, selected_indeces,\
                                                    sentences_unlabelled, process_monitoring_instance):
        scores_with_index = []
        index_in_which_no_minority_categories_are_predicted = []
        searched_among = 0 # Only to print information 
        for xi, yi, index in zip(to_search_among_x, ys, selected_indeces):
            if is_minority_classes_in_vector(yi, self.minority_classes_index): # search among those in which minority category has been predicted
                difference_between_predicted_and_second_best = self.get_smallest_diff_alternative(xi, yi, get_permutations_with_predicted_chunks)
                scores_with_index.append((difference_between_predicted_and_second_best, index, yi, sentences_unlabelled[index])) 
            else:
                index_in_which_no_minority_categories_are_predicted.append((xi, yi, index))
            searched_among = searched_among + 1
            if searched_among % 100 == 0: # only to print information
                print("Searched among " + str(searched_among) + " so far.")
        return scores_with_index, index_in_which_no_minority_categories_are_predicted

    
    # Returns the score for those with no predicted chunks in sorted order, and the number_of_unlabelled_to_select best
    # (where number_of_unlabelled_to_select is equal to len(index_in_which_no_minority_categories_are_predicted) if the setting to not prioritize chunks is chosen.)
    def get_scores_unlabelled_sorted_no_predicted_chunks(self, number_of_unlabelled_to_select, index_in_which_no_minority_categories_are_predicted, sentences_unlabelled, inactive_learning):

        scores_with_index_no_predicted_chunks = []
        for xi, yi, index in index_in_which_no_minority_categories_are_predicted:
            difference_between_predicted_and_second_best_no_predicted_chunks = self.get_smallest_diff_alternative(xi, yi, get_permutations_no_predicted_chunks)
            scores_with_index_no_predicted_chunks.append((difference_between_predicted_and_second_best_no_predicted_chunks, index, yi, sentences_unlabelled[index]))
        if inactive_learning:
            print("Running in reversed mode, selecting the samples for which the learning is most certain.")
            sorted_score_index_no_predicted_chunks = sorted(scores_with_index_no_predicted_chunks, reverse=True)
        else:
            # This is the the option that is typically used. The one in which active learning is achieved.
            sorted_score_index_no_predicted_chunks = sorted(scores_with_index_no_predicted_chunks)
        return sorted_score_index_no_predicted_chunks[:number_of_unlabelled_to_select]





class NonStructuredLogisticRegression(ModelWrapperBase):
    def __init__(self, label_dict, minority_classes, outside_class, beginning_prefix, inside_prefix, max_iterations,  \
                     use_cross_validation, nr_of_cross_validation_splits, c_value, process_monitor_instance = None):
        # max_iter not used by the liblinear solver
        self.model = LogisticRegression(verbose=0, penalty='l1', solver='liblinear', C=c_value, random_state = 1)
        self.__name__ = "NonStructuredLogisticRegression"
        self.init_params(label_dict, minority_classes, outside_class, beginning_prefix, inside_prefix, max_iterations, \
                             use_cross_validation, nr_of_cross_validation_splits, c_value)
        self.process_monitor_instance = process_monitor_instance
     
    def fit(self, X, Y):
        X_flat = np.concatenate(X)
        Y_flat = np.concatenate(Y)

        if self.use_cross_validation:
            
            beginning_classes = []
            for m in self.minority_classes:
                if m.startswith(self.beginning_prefix):
                    beginning_classes.append(self.label_dict[m])
        
            temp_min = "temp_min"
            for m in self.minority_classes:
                if m.startswith(self.beginning_prefix):
                    temp_min = m.replace(self.beginning_prefix, "")

            if len(beginning_classes) > 1:
                print("Cross validation not applicable when having many classes")
                exit(1)
            Y_flat_remove_bi_dist  = [] # only take beginning class into account for determining c-value
            for el in Y_flat:
                if el == self.label_dict[self.outside_class]:
                    Y_flat_remove_bi_dist.append(el)
                else:
                    Y_flat_remove_bi_dist.append(beginning_classes[0])
            Y_flat_remove_bi_dist = np.array(Y_flat_remove_bi_dist)

            parameters={'C': [1, 5, 10]}
            
            print("Starting cross-validation with parameters: " + str(parameters))
            skf = StratifiedKFold(self.nr_of_cross_validation_splits)

            f1_scorer = make_scorer(f1_score, average='binary', pos_label=beginning_classes[0])

            grid_search_clf = GridSearchCV(self.model, parameters, cv=skf, scoring = f1_scorer)
            grid_search_clf.fit(X_flat, Y_flat_remove_bi_dist)
            self.model = grid_search_clf.best_estimator_
            self.C = self.model.C
            
            if self.process_monitor_instance:
                self.process_monitor_instance.write_score_process_monitoring(len(X), grid_search_clf.best_score_, temp_min)
    
        else:
            print("No cross-validation")
        ret = self.model.fit(X_flat, Y_flat)
        print("Model params ", self.model.get_params())
        return ret

    def predict(self, X):
        return self.predict_nonstructured(X)

    def predict_proba(self, X):
        X_flat = np.concatenate(X)
        predicted =  self.model.predict_proba(X_flat)
        
        X_flat_counter = 0
        predicted_in_sentences = []
        for sentence in X:
            predicted_for_sentence = []
            for word in sentence:
                predicted_for_sentence.append(predicted[X_flat_counter])
                X_flat_counter  = X_flat_counter + 1
            predicted_in_sentences.append(predicted_for_sentence)
        return predicted_in_sentences

    def get_probabilities(self, to_search_among_x):
        probabilities = self.predict_proba(to_search_among_x)
        
        all_diffs = [] # To use in process monitoring only
        min_probabilities = []
        all_index_for_min_probabilities = [] # To use in process monitoring only
        for sentence in probabilities:
            all_diff_sentence = [] # To use in process monitoring only
            min_difference_in_sentence = float("inf")
            index_for_min_difference_in_sentence = 0
            for word_index, word in enumerate(sentence):
                word = list(word)
                #print("probabilities for word not sorted", word)
                word_sorted = sorted(word, reverse=True)
                #print("probabilities for word", word_sorted)
                diff_best_second_best = word_sorted[0] - word_sorted[1]
                all_diff_sentence.append(diff_best_second_best)
                #print("diff_best_second_best", diff_best_second_best)
                if diff_best_second_best < min_difference_in_sentence:
                    min_difference_in_sentence = diff_best_second_best
                    index_for_min_difference_in_sentence = word_index
            all_diffs.append(all_diff_sentence)
            min_probabilities.append(min_difference_in_sentence)
            all_index_for_min_probabilities.append(index_for_min_difference_in_sentence)
        return min_probabilities, all_diffs, all_index_for_min_probabilities
        # all_index_for_min_probabilities contains, for each sentence among the sentences in "to_search_among_x", the index of the token that
        # has the lowest probability

    def get_scores_unlabelled_with_predicted_chunks(self, to_search_among_x, ys, selected_indeces,\
                                                    sentences_unlabelled, process_monitoring_instance):
        
        min_probability_differences, all_diffs, all_index_for_min_probabilities = self.get_probabilities(to_search_among_x)
        process_monitoring_instance.write_process_monitoring_info(sentences_unlabelled, all_diffs,\
                                                                  selected_indeces, ys, self.majority_class, self.inv_label_dict,\
                                                                  all_index_for_min_probabilities, min_probability_differences)
        
        scores_with_index = []
        index_in_which_no_minority_categories_are_predicted = []
        searched_among = 0 # Only to print information 
        for xi, yi, index, min_probability_difference in zip(to_search_among_x, ys, selected_indeces, min_probability_differences):
        #print("search for index " + str(index))
        #print(sentences_unlabelled[index])
            if is_minority_classes_in_vector(yi, self.minority_classes_index): # search among those in which minority category has been predicted
                scores_with_index.append((min_probability_difference, index, yi, sentences_unlabelled[index])) 
            else:
                index_in_which_no_minority_categories_are_predicted.append((min_probability_difference, xi, yi, index))
            searched_among = searched_among + 1
            if searched_among % 100 == 0: # only to print information
                print("Searched among " + str(searched_among) + " so far.")
        #if len(scores_with_index) > 0:
        #    print("scores_with_index[0]", scores_with_index[0])
        #print("index_in_which_no_minority_categories_are_predicted[0]", index_in_which_no_minority_categories_are_predicted[0])
        return scores_with_index, index_in_which_no_minority_categories_are_predicted


    # Returns the score for those with no predicted chunks in sorted order, and the number_of_unlabelled_to_select best
    # (where number_of_unlabelled_to_select is equal to len(index_in_which_no_minority_categories_are_predicted) if the setting to not prioritize chunks is chosen.)
    def get_scores_unlabelled_sorted_no_predicted_chunks(self, number_of_unlabelled_to_select, \
                                                             index_in_which_no_minority_categories_are_predicted, sentences_unlabelled, inactive_learning):

        scores_with_index_no_predicted_chunks = []
        for min_probability_difference, xi, yi, index in index_in_which_no_minority_categories_are_predicted:
            scores_with_index_no_predicted_chunks.append((min_probability_difference, index, yi, sentences_unlabelled[index]))
        if inactive_learning:
            print("Running in reversed mode, selecting the samples for which the learning is most certain.")
            sorted_score_index_no_predicted_chunks = sorted(scores_with_index_no_predicted_chunks, reverse=True)
        else:
            # This is the the option that is typically used. The one in which active learning is achieved.
            sorted_score_index_no_predicted_chunks = sorted(scores_with_index_no_predicted_chunks)
        #print("sorted_score_index_no_predicted_chunks[0]", sorted_score_index_no_predicted_chunks[0])
        return sorted_score_index_no_predicted_chunks[:number_of_unlabelled_to_select]

    def get_params(self):
        self.model.get_params()

    def get_cs(self):
        try:
            c = self.C
            return str(c)
        except:
            return str(self.c_value)



def is_minority_classes_in_vector(predicted, minority_classes):
    """

    Small help function that checks if a vector contains minority classes (only there to make the code more self-explaining).
    """

    for m in minority_classes:
        if m in predicted:
            return True
    return False

def get_selected_sentences_with_different_vocabulary(sorted_score_index, sorted_indeces_no_predicted_chunks, step_size, majority_category, inactive_learning, prefer_predicted_chunks):
    """
    Help function to do the final selection of samples. 
    """
    #print("sorted_score_index", sorted_score_index)
    #print("sorted_indeces_no_predicted_chunks", sorted_indeces_no_predicted_chunks)

    if not prefer_predicted_chunks:
        sorted_score_index = sorted_score_index + sorted_indeces_no_predicted_chunks
    if inactive_learning:
        print("Running in reversed mode, selecting the samples for which the learning is most certain.")
        sorted_score_index = sorted(sorted_score_index, reverse=True)
    else:
        # This is the the option that is typically used. The one in which active learning is achieved.
        sorted_score_index = sorted(sorted_score_index)
    #print("sorted_score_index", sorted_score_index)

    """    
    print("The best candidates before word spread is taken into account")        
    for el in sorted_score_index[:10]:
        print(el)
    """

    
    indeces_to_use = []
    indeces_not_to_use = []
    predicted_words = set()
    already_used_sentence = set()
    for (score, index, predicted, sentence) in sorted_score_index:
        current_sentence =  " ".join(sentence)
        sentence_has_already_used_word = False
        if current_sentence in already_used_sentence:
            similar_sentence_has_alread_been_picked = True
            print("Sentence already used")
            print("Current sentence")
        else:
            similar_sentence_has_alread_been_picked = False
            already_used_sentence.add(current_sentence)
        for i, el in enumerate(predicted):
            if el != majority_category:
                predicted_word = sentence[i]
                if predicted_word in predicted_words:
                    sentence_has_already_used_word = True
                predicted_words.add(predicted_word)
        if not sentence_has_already_used_word:
            indeces_to_use.append(index)
        else:
            indeces_not_to_use.append(index)
        if len(indeces_to_use) >= step_size:
            break

    print("predicted_words", predicted_words)
    
    if len(indeces_to_use) < step_size: #if there weren't enough uncertain with large word spread, take those that have been filtered out
        print("Can't return samples with the requested word spread")
        print("Filtered out indeces, that will be used anyway, therefore:")
        print(step_size - len(indeces_to_use))
        print(indeces_not_to_use[:step_size - len(indeces_to_use)])
        indeces_to_use = indeces_to_use + indeces_not_to_use[:step_size - len(indeces_to_use)]
    #first_indeces = [index for (score, index, predicted, sentence) in sorted_score_index[:step_size]]

    print("indeces_to_use", indeces_to_use)

    # Only for printing information
    if prefer_predicted_chunks:
        indeces_to_use = indeces_to_use + [el[1] for el in sorted_indeces_no_predicted_chunks]
        print("The best candidates without a predicted chunk")        
        print("indeces_to_use with chunks all", indeces_to_use)

    return indeces_to_use


###############
#Help functions for creating permutations for uncertainty sampling
##############

def get_permutations_with_predicted_chunks(yi, previous_model_wrapper):
    """
    Help function for creating permutations for uncertainty sampling
    """

    #print("\n******")
    #print("\noriginal: ", [previous_model_wrapper.inv_label_dict[el] for el in yi])
    yi_alternatives = [[]]
        
    index_to_permute_beginning = []
    index_to_permute_inside = []
    for index, el in enumerate(yi): 
        if el != previous_model_wrapper.majority_class:
            if previous_model_wrapper.inv_label_dict[el].startswith(previous_model_wrapper.beginning_prefix):
                index_to_permute_beginning.append(index)
            elif previous_model_wrapper.inv_label_dict[el].startswith(previous_model_wrapper.inside_prefix):
                index_to_permute_inside.append(index)
            else:
                print("Unknown prefix:")
                print(previous_model_wrapper.inv_label_dict[el])
                exit(1)
            if False: # TODO: Add an option to also add this permutation
                if index + 1 < len(yi):
                    index_to_permute.append(index + 1) # also add the index right after, since that might be interesting due to span difficulties
                if index - 1 >= 0:
                    index_to_permute.append(index - 1) # also add the index right before, since that might be interesting due to span difficulties

    index_to_permute_beginning = set(index_to_permute_beginning)
    index_to_permute_inside = set(index_to_permute_inside)

    if len(index_to_permute_beginning) + len(index_to_permute_inside) > 6: # If there are too many alternatives, just go for the option of no annotations, otherwise it will take too much time
        #print("Too many to permute, only use unlabelled")
        yi_alternatives = [[previous_model_wrapper.majority_class]*len(yi)]
    else:
        #print("index_to_permute", index_to_permute)    
        for position in range(0, len(yi)):
            if position in index_to_permute_beginning:
                new_yi_alternatives = []
                for category in range(0, len(previous_model_wrapper.minority_classes_index) + 1): #add the majority category
                    #print(previous_model_wrapper.inv_label_dict[category])
                    if previous_model_wrapper.inv_label_dict[category].startswith(previous_model_wrapper.beginning_prefix) or previous_model_wrapper.inv_label_dict[category] == previous_model_wrapper.outside_class:
                        for el in yi_alternatives:
                            new_el = el[:]
                            new_el.append(category)
                            new_yi_alternatives.append(new_el)
                yi_alternatives = new_yi_alternatives
            elif position in index_to_permute_inside:
                new_yi_alternatives = []
                for category in range(0, len(previous_model_wrapper.minority_classes_index) + 1): #add the majority category
                    #print(previous_model_wrapper.inv_label_dict[category])
                    if previous_model_wrapper.inv_label_dict[category].startswith(previous_model_wrapper.inside_prefix):
                        for el in yi_alternatives:
                            if len(el) > 0 and (previous_model_wrapper.inv_label_dict[el[-1]].startswith(previous_model_wrapper.inside_prefix) or previous_model_wrapper.inv_label_dict[el[-1]].startswith(previous_model_wrapper.beginning_prefix)):
                                if previous_model_wrapper.inv_label_dict[el[-1]][len(previous_model_wrapper.inside_prefix):] == previous_model_wrapper.inv_label_dict[category][len(previous_model_wrapper.inside_prefix):]:
                                    # only allow an inside permutation for an inside class following a similar begin or inside class
                                    # an inside permutation can also not be the start of a sentence
                                    new_el = el[:]
                                    new_el.append(category)
                                    new_yi_alternatives.append(new_el)
                    elif previous_model_wrapper.inv_label_dict[category] == previous_model_wrapper.outside_class:
                        for el in yi_alternatives:
                            new_el = el[:]
                            new_el.append(category)
                            new_yi_alternatives.append(new_el)
                                    
                yi_alternatives = new_yi_alternatives
            else:
                for j in range(0, len(yi_alternatives)):                                                                                                          
                    yi_alternatives[j].append(previous_model_wrapper.majority_class)
    yi_alternatives = [el for el in yi_alternatives if not(np.array_equal(el,yi))]             

    # Print created permutations
    #for alt in yi_alternatives:
        #print([previous_model_wrapper.inv_label_dict[el] for el in alt])
            
    yi_alternatives_np = np.array(yi_alternatives)
    
    return yi_alternatives_np


def get_permutations_no_predicted_chunks(yi, previous_model_wrapper):
    """
    Help function for creating permutations for uncertainty sampling
    """

    #print("\n******")
    #print("\noriginal: ", [previous_model_wrapper.inv_label_dict[el] for el in yi])
    #print("\noriginal: ", yi)
    yi_alternatives = []
    
    for category in range(0, len(previous_model_wrapper.minority_classes_index)):
        if previous_model_wrapper.inv_label_dict[category].startswith(previous_model_wrapper.beginning_prefix): # only include the permutations with beginning-prefix
            for index, el in enumerate(yi):
                yi_copy = [yi[0]] * len(yi)
                yi_copy[index] = category
                yi_alternatives.append(yi_copy)

    # Print created permutations
    #for alt in yi_alternatives:
    #    print(alt)
            
    yi_alternatives_np = np.array(yi_alternatives)
    
    return yi_alternatives_np

