import numpy as np
import os
import sys
from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM
from pystruct.learners import NSlackSSVM
from pystruct.learners import FrankWolfeSSVM
from sklearn.utils import shuffle
from joblib import Parallel, delayed
from sklearn.svm import LinearSVC
from sklearn.metrics import  make_scorer
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split


# For uncertainty selection
def is_minority_classes_in_vector(predicted, minority_classes):
    for m in minority_classes:
        if m in predicted:
            return True
    return False

def get_uncertainty_selected_sentences_with_different_vocabulary(sorted_score_index, sorted_indeces_no_predicted_chunks, step_size, majority_category, inactive_learning, prefer_predicted_chunks):
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

    print("The best candidates before word spread is taken into account")        
    for el in sorted_score_index[:10]:
        print(el)

    indeces_to_use = []
    indeces_not_to_use = []
    predicted_words = set()
    for (score, index, predicted, sentence) in sorted_score_index:
        print("len(indeces_to_use)", len(indeces_to_use))
        print("step_size", step_size)
        sentence_has_already_used_word = False
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

    if prefer_predicted_chunks:
        indeces_to_use = indeces_to_use + [el[1] for el in sorted_indeces_no_predicted_chunks]
        print("The best candidates without a predicted chunk")        
        for el in sorted_indeces_no_predicted_chunks:
            print(el)
        print("indeces_to_use with chunks all", indeces_to_use)

    return indeces_to_use


def get_permutations(yi, previous_model_wrapper):

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

# for alternatives to annotated chunks 'get_permutations' is the permutation_method
# for alternatives when there are no annotated chunks 'get_permutations_no_predicted_chunks' is the permutation_method
def get_smallest_diff_alternative(previous_model, previous_learner, previous_model_wrapper, xi, yi, permutation_method):
    joint = previous_model.joint_feature(xi, yi)
    score = np.dot(previous_learner.w, joint)

    # min_difference is the score difference between the predicted classification and the second best classification

    min_difference = 10000
    yi_alternatives = permutation_method(yi, previous_model_wrapper)
    
    for yi_alternative in yi_alternatives:
        joint_alternative = previous_model.joint_feature(xi, yi_alternative)
        score_alternative = np.dot(previous_learner.w, joint_alternative)
        #if True:
            #print("yi_alternative", yi_alternative)
            #print("score_predicted", score)
            #print("score_alternative", score_alternative)
        difference_between_predicted_and_alternative = score - score_alternative
        if difference_between_predicted_and_alternative < min_difference:
            min_difference = difference_between_predicted_and_alternative
    #print("min_difference ", min_difference)
    return min_difference

def get_unlabelled_no_predicted_chunks(number_of_unlabelled_to_select, index_in_which_no_minority_categories_are_predicted, sentences_unlabelled, previous_model, previous_learner, previous_model_wrapper, inactive_learning):
    scores_with_index_no_predicted_chunks = []
    for xi, yi, index in index_in_which_no_minority_categories_are_predicted:
        difference_between_predicted_and_second_best_no_predicted_chunks = get_smallest_diff_alternative(previous_model, previous_learner, previous_model_wrapper, xi, yi, get_permutations_no_predicted_chunks)
        scores_with_index_no_predicted_chunks.append((difference_between_predicted_and_second_best_no_predicted_chunks, index, yi, sentences_unlabelled[index]))
    if inactive_learning:
        print("Running in reversed mode, selecting the samples for which the learning is most certain.")
        sorted_score_index_no_predicted_chunks = sorted(scores_with_index_no_predicted_chunks, reverse=True)
    else:
        # This is the the option that is typically used. The one in which active learning is achieved.
        sorted_score_index_no_predicted_chunks = sorted(scores_with_index_no_predicted_chunks)
    return sorted_score_index_no_predicted_chunks[:number_of_unlabelled_to_select]

    # Return indeces in the 'number_of_unlabelled_to_select' best, among the ones without predicted chunks    
    #return [el[1] for el in sorted_score_index_no_predicted_chunks[:number_of_unlabelled_to_select]]

def get_uncertainty_unlabelled(labelled_x, labelled_y, unlabelled_x, step_size, previous_model_wrapper, sentences_labelled, sentences_unlabelled, maximum_samples_to_search_among, inactive_learning, prefer_predicted_chunks):
    if step_size == 0:
        print("You have chosen to select 0 new samples to pre-annotated. The variable 'nr_of_samples' in 'settings.py' should be at least 1")
        exit(1)
    if step_size > len(unlabelled_x):
        print("More samples have been asked for than exist among unlabelled. A maximum of " + str(len(unlabelled_x)) + " nr of samples can be returned")
        step_size = len(unlabelled_x)
    previous_model = previous_model_wrapper.model
    previous_learner = previous_model_wrapper.ssvm

    minority_classes = previous_model_wrapper.minority_classes_index

    # Randomly select samples among which to search for to search for the most informative training instance
    selected_indeces = shuffle(range(0, len(unlabelled_x)))[:maximum_samples_to_search_among]

    to_search_among_x = []
    for selected_index in selected_indeces:
        to_search_among_x.append(unlabelled_x[selected_index])

    ys = previous_learner.predict(to_search_among_x)

    print("Requested a search among a maximum of " + str(maximum_samples_to_search_among) + " samples")
    scores_with_index = []
    index_in_which_no_minority_categories_are_predicted = []
    searched_among = 0 # Only to print information 
    for xi, yi, index in zip(to_search_among_x, ys, selected_indeces):
        #print("search for index " + str(index))
        #print(sentences_unlabelled[index])
        if is_minority_classes_in_vector(yi, minority_classes): # search among those in which minority category has been predicted
            difference_between_predicted_and_second_best = get_smallest_diff_alternative(previous_model, previous_learner, previous_model_wrapper, xi, yi, get_permutations)
            scores_with_index.append((difference_between_predicted_and_second_best, index, yi, sentences_unlabelled[index])) 
        else:
            index_in_which_no_minority_categories_are_predicted.append((xi, yi, index))
        searched_among = searched_among + 1
        if searched_among % 100 == 0: # only to print information
            print("Searched among " + str(searched_among) + " so far.")

    # if there are too few samples among the unlabelled in which minority categoies are predict, also return unlabelled samples without minority categories
    # or if the setting is chosen to don't prefer samples in which minority categores are predicted, compute certainty score for all those unlabelled
    if len(scores_with_index) < step_size or not prefer_predicted_chunks:
        if len(scores_with_index) < step_size:
            number_of_unlabelled_to_select = step_size - len(scores_with_index)
        else: # i.e. not prefer_predicted_chunks 
            number_of_unlabelled_to_select = len(index_in_which_no_minority_categories_are_predicted) # include all of them, and filter out later
        #step_size = len(scores_with_index)
        #print("Will select " + str(step_size) + " samples with labelled chunks.")
        print("Will search among " + str(number_of_unlabelled_to_select) + " without labelled chunks.")
        sorted_indeces_no_predicted_chunks = get_unlabelled_no_predicted_chunks(number_of_unlabelled_to_select, index_in_which_no_minority_categories_are_predicted, sentences_unlabelled, previous_model, previous_learner, previous_model_wrapper, inactive_learning)
        #print("Selected indeces where no chunks are predicted", sorted_indeces_no_predicted_chunks)
    else:
        print("Will search for the best ones among the " + str(len(scores_with_index)) + " samples that contained a minority category prediction.")
        sorted_indeces_no_predicted_chunks = [] # nothing without predicted chunks included
    #if step_size == 0:
    #    print("No named entities were found in the unlabelled data. This could either be caused by that you have a too small seed set, or by that the set of unlabelled data is too small")
        
    if inactive_learning:
        print("Running in reversed mode, selecting the samples for which the learning is most certain. This mode is only sensible in the case when pre-annotation is used only, and all data in the pool available is to be used")
        # in this mode, still prefer samples in which chunks have been predicted
        sorted_score_index = sorted(scores_with_index, reverse=True)
    else:
        # This is the the option that is typically used. The one in which active learning is achieved.
        # Get the step_size samples with lowest score    
        sorted_score_index = sorted(scores_with_index)

    index_to_select_among_checked = get_uncertainty_selected_sentences_with_different_vocabulary(sorted_score_index, sorted_indeces_no_predicted_chunks, step_size, previous_model_wrapper.majority_class, inactive_learning, prefer_predicted_chunks)

    # Only for printing out information. This info is not used in the selection process

    """
    most_certain_index = [index for (score, index, predicted, sentence) in sorted_score_index[len(sorted_score_index) - 2:]]
    #print("sorted_score_index", sorted_score_index[:step_size])
    print("__________________________")
    print("Selected with predicted labelled chunks")
    for i in index_to_select_among_checked:
        print("i", i, sentences_unlabelled[i], "predicted:", previous_learner.predict([unlabelled_x[i]]))
    print("Least likely to be selected among the ones with predicted labelled chunks")
    for i in most_certain_index:
        print("i", i, sentences_unlabelled[i])
    print("Selected without predicted labelled chunks")
    for i in [el[1] for el in sorted_indeces_no_predicted_chunks]:
        print("i", i, sentences_unlabelled[i], "predicted:", previous_learner.predict([unlabelled_x[i]]))
    """    

    to_select_X = []
    to_select_text = []
    predicted_for_selected = []
    #final_selected_indeces = index_to_select_among_checked + sorted_indeces_no_predicted_chunks
    for its in index_to_select_among_checked:
        to_select_X.append(unlabelled_x[its])
        to_select_text.append(sentences_unlabelled[its])
        predicted_for_selected.append(previous_learner.predict(unlabelled_x[its:its+1])[0]) # must submit an numpy array to predict
    print("__________________________")

    unlabelled_x = np.delete(unlabelled_x, index_to_select_among_checked, 0)
    sentences_unlabelled = np.delete(sentences_unlabelled, index_to_select_among_checked, 0)

    return to_select_X, unlabelled_x, to_select_text, sentences_unlabelled, predicted_for_selected

#####
# Public method
######
def get_maximum_samples_to_search_among(maximum_samples_to_search_among, X_unlabelled_np, nr_of_samples):
    try:
        nr_of_samples_int = int(nr_of_samples)

        if isinstance(maximum_samples_to_search_among, str) and maximum_samples_to_search_among.lower() == "all":
            print("Will search for samples to select among all " + str(len(X_unlabelled_np)) + " unlabelled samples.") 
            return len(X_unlabelled_np)


        #if nr_of_samples_int > maximum_samples_to_search_among:
            #print("The property nr_of_samples is larger than the samples for which a request to search among was made, i.e., maximum_samples_to_search_among ")
            #exit(1)
    
        try:
            maximum_samples_to_search_among_int = int(maximum_samples_to_search_among)


            if maximum_samples_to_search_among_int > len(X_unlabelled_np):
                print("There are " + str(len(X_unlabelled_np)) + " unlabelled samples left. Will search for samples to select among those.") 
                return len(X_unlabelled_np)
            else:
                print("Will search for samples to select among " + str(maximum_samples_to_search_among_int) + " randomly selected unlabelled samples.")
                return maximum_samples_to_search_among_int
        except ValueError:
            print("The property maximum_samples_to_search_among can only take a numerical value or the string 'all', " + maximum_samples_to_search_among + " is not valid.")
            exit(1)
    except ValueError:
        print("The property nr_of_samples can only take a numerical value, " + str(nr_of_samples) + " is not valid.")
        exit(1)


def get_new_data(X_labelled_np, X_unlabelled_np, y_labelled_np, text_vector_labelled_np, text_vector_unlabelled_np, label_dict, minority_categories, nr_of_samples,  maximum_samples_to_search_among, outside_class, beginning_prefix, inside_prefix, inactive_learning, max_iterations, prefer_predicted_chunks):

    maximum_samples_to_search_among = get_maximum_samples_to_search_among(maximum_samples_to_search_among, X_unlabelled_np, nr_of_samples)
    
    model = StructuredModel(label_dict, minority_categories, outside_class, beginning_prefix, inside_prefix, max_iterations)
    print("Started to train the model on the labelled data")
    model.fit(X_labelled_np, y_labelled_np)
    print("Training on labelled data finished")

    predicted_y = model.predict(X_unlabelled_np)
    print("Predicted results on the unlabelled data")

    to_select_X, unlabelled_x, to_select_text, sentences_unlabelled, predicted_for_selected = get_uncertainty_unlabelled(X_labelled_np, y_labelled_np, X_unlabelled_np, nr_of_samples, model, text_vector_labelled_np, text_vector_unlabelled_np,  maximum_samples_to_search_among, inactive_learning, prefer_predicted_chunks)
    return(to_select_X, unlabelled_x, to_select_text, sentences_unlabelled, predicted_for_selected)


class StructuredModel:
    def __init__(self, label_dict, minority_classes, outside_class, beginning_prefix, inside_prefix, max_iterations):
        self.model = ChainCRF()
        self.__name__ = "StructuredModel"

        self.outside_class = outside_class
        self.beginning_prefix = beginning_prefix
        self.inside_prefix = inside_prefix

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
        
    def fit(self, X, Y):
        self.model = ChainCRF() # make a new model each time
        #self.ssvm = OneSlackSSVM(model=self.model, max_iter=1000, n_jobs = -1, check_constraints=False)
        #self.ssvm = NSlackSSVM(model=self.model, max_iter=100, C=best_c, check_constraints=False)
        self.ssvm = FrankWolfeSSVM(model=self.model, max_iter=self.max_iterations)
        ret = self.ssvm.fit(X, Y)
        return ret

    def predict(self, X):
        return self.ssvm.predict(X)





