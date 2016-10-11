import sys
import active_learning_preannotation
import argparse
import os


def transform_from_brat_format(annotation_file_name, text_file_name, properties, project_path):
    labelled_data_dir_for_project = os.path.join(project_path, properties.labelled_data_dir)
    new_annotated_file_name = os.path.basename(annotation_file_name).replace(".ann", "") + properties.data_file_extension
    new_annotated_file_path = os.path.join(labelled_data_dir_for_project, new_annotated_file_name)

    if os.path.exists(new_annotated_file_path):
        print("The file that is meant to be generated already exists. Did you run this script before, using this file? Or do you have several annotated files with the same name")
        exit(1)

    new_annotated_file = open(new_annotated_file_path, "w")
    
    outside_tag = properties.outside_class
    beginning_prefix = properties.beginning_prefix
    inside_prefix = properties.inside_prefix 

    text_file = open(text_file_name)
    text = text_file.read()
    text_file.close()
    #print(text)
    print("Text of length " + str(len(text)) + " read")
    annotated_char_list = [None for el in text]
    
    annotated_words = set()
    annotation_file = open(annotation_file_name)
    for line in annotation_file:
        sp = line.strip().split()
        #print(sp)
        for word in sp[4:]:
            annotated_words.add(word)
        for i in range(int(sp[2]), int(sp[3])+1):
            annotated_char_list[i] = sp[1]
    annotation_file.close()
    print("Annotated words: ")
    print(annotated_words)

    #print(annotated_char_list)

    start_index = 0
    annotation_last_word = None
    for index, ch in enumerate(text):
        if ch == " " or ch == "\n": # a word has ended
            annotation_current_word = None
            annotation_to_write = outside_tag
            annotated_tag_set = set() # should be a set of one
            end_index = index - 1
            #sys.stdout.write(" " + str(start_index) + " " + str(index - 1))
            for i in range(start_index, end_index):
                if annotated_char_list[i]:
                     annotated_tag_set.add(annotated_char_list[i])
            # if a subset of the word is annotated with a category, the entire word gets this category
            # if a word is annotated with several categories, it is incorrectly annotated and the program will terminate
            if len(annotated_tag_set) > 1:
                print("The word ending at index " + str(end_index) + " is annotated with several categories. Change this annotation before running this script")
                exit(1)
            if len(annotated_tag_set) == 1:
                annotation_current_word = annotated_tag_set.pop()
                if annotation_current_word != annotation_last_word: # First word in a chunk
                    annotation_to_write = beginning_prefix + annotation_current_word
                else: # inside a chunk
                    annotation_to_write = inside_prefix + annotation_current_word
            new_annotated_file.write("\t" + annotation_to_write)
            
            # Information for starting a new word
            start_index = index + 1
            annotation_last_word = annotation_current_word 
        if ch == " ":
            new_annotated_file.write('\n')
        elif ch == "\n":
            new_annotated_file.write('\n\n')
        else:
            new_annotated_file.write(ch)

def load_annotated_data_path(parser):
    args = parser.parse_args()
    if not args.annotated_path:
        print("The argument '--annotated' (including the file name without suffix)with the path to the manually data needs to be given")
        exit(1)

    annotated_file = args.annotated_path + ".ann"
    if not os.path.exists(annotated_file):
        print("The file "  + annotated_file + " does not exist")
        exit(1)

    text_file = args.annotated_path + ".txt"
    if not os.path.exists(text_file):
        print("The file "  + text_file + " does not exist")
        exit(1)

    return annotated_file, text_file

if __name__ == "__main__":

    # Use the same functionality as for running the active learning and pre-annnotation, for getting properties,
    # such as path of where to put the output and what tags for begin, end and outside to use.

    parser = argparse.ArgumentParser()


    parser.add_argument('--annotated', action='store', dest='annotated_path', help='The path, in slash format, (including the file name without suffix) to where the manually annotated files')

    properties, path_slash_format = active_learning_preannotation.load_properties(parser)

    print(properties, path_slash_format)

    annotated_file, text_file = load_annotated_data_path(parser)

    print("Will read data from: ")
    print(text_file)
    print(annotated_file)
    transform_from_brat_format(annotated_file, text_file, properties, path_slash_format)
