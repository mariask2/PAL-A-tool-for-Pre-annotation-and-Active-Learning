# Code to transform simulation results into a format that can be read by gnuplot
# As well as to do calculations for average f-score
# Run for instance as:
# python read_simulation_results.py --project=data.example_project --category=B-speculation --xspace=0.05 --xaxisstep=200 --maxy=0.8

import argparse
import os
import glob
import math
from matplotlib.pyplot import plot, show, bar, grid, axis, savefig, clf
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

import active_learning_preannotation


OUTPUT_DIR = "evaluation_simulate_active_learning"
ACTIVE = "active"
RANDOM = "random"

def extract_data(result_dict, files):
    for f in files:
        number = int(f.split("conll_res")[0].split("_")[-1])
        if number not in result_dict:
            result_dict[number] = []
        open_f = open(f)
        for line in open_f:
            if "f1_score" in line:
                f_score = line.strip().split("\t")[1]
                result_dict[number].append(float(f_score))
        open_f.close()

def write_dict(name, result_dict, output_file, color, marker, markersize, x_value_extra):
    x_values = []
    y_values = []
    error_max = []
    error_min = []
    output_file.write(name + "\n")
    output_file.write("# sample size\tf_score\n")
    for key in sorted(result_dict.keys()):
        sorted_res = sorted(result_dict[key][:])
        min = sorted_res[0]
        max = sorted_res[-1]
        mean = sum(result_dict[key])/len(result_dict[key])
        output_str = str(key) + "\t" + str(mean) + "\t"  +  str(min)  + "\t" + str(max) + "\n"
        output_file.write(output_str)
        x_values.append(key + x_value_extra)
        y_values.append(mean)
        error_max.append(max - mean)
        error_min.append(mean - min)
    #print(x_values)
    #print(y_values)
    #print("min", error_min)
    #print("max", error_max)


    plt.errorbar(x_values, y_values, yerr=[error_min, error_max], color=color, marker=marker, linewidth=1, markersize=markersize)
    plot_handle, = plt.plot(x_values, y_values, color=color, marker=marker, linewidth=1, markersize=markersize)

    output_file.write("\n\n")
    return plot_handle

def read_results(result_path, category, extra_xspace, category_index, sub_plot, x_axis_step, max_y):
    random_word2vecfalse = {}
    random_word2vectrue = {}
    active_word2vecfalse = {}
    active_word2vectrue = {}
    
    handles_labels = []
    
    for subdir in [el for el in os.listdir(result_path) if not el.startswith(".")]:
        full_subdir = os.path.join(result_path, subdir)
        random_dir = os.path.join(full_subdir, RANDOM)
        active_dir = os.path.join(full_subdir, ACTIVE)

        extract_data(random_word2vecfalse, glob.glob(os.path.join(random_dir, "*False*conll_res.txt")))
        extract_data(random_word2vectrue, glob.glob(os.path.join(random_dir, "*True*conll_res.txt")))
        extract_data(active_word2vecfalse, glob.glob(os.path.join(active_dir, "*False*conll_res.txt")))
        extract_data(active_word2vectrue, glob.glob(os.path.join(active_dir, "*True*conll_res.txt")))
    
    
    title = category.replace("B-", "")

    print(title)
    # If the three standard ner categories are used
    if title == "org":
        title = "Organisation"
    if title == "per":
        title = "Person"
    if title == "loc":
        title = "Location"

    title = title[0].upper() + title[1:]
    plt.title(title)
    if category_index % 2 != 0: # Only write label at every other subplot
        plt.xlabel('Training data size')
    if category_index == 0: # Only need to write this once
        plt.ylabel('F-score')
    


    output_file = open(os.path.join(result_path, "conll_media_fscore.dat"), "w")

    handles_labels.append((write_dict("#random_word2vecfalse", random_word2vecfalse, output_file, "red", 's', 4, 0),\
                           "Random"))
    handles_labels.append((write_dict("#active_word2vecfalse", active_word2vecfalse, output_file, "green", 'd', 4, 1*extra_xspace), \
                           "Active"))
    handles_labels.append((write_dict("#random_word2vectrue", random_word2vectrue, output_file, "blue", '*', 5, 2*extra_xspace), \
                           "Random, \nWord2Vec"))
    handles_labels.append((write_dict("#active_word2vectrue", active_word2vectrue, output_file, "black", 'o', 4, 3*extra_xspace), \
                           "Active, \nWord2Vec"))


    min_x = sorted(list(random_word2vecfalse.keys()) + list(active_word2vecfalse.keys()) +\
                   list(random_word2vectrue.keys()) + list(active_word2vectrue.keys()))[0]
    max_x = sorted(list(random_word2vecfalse.keys()) + list(active_word2vecfalse.keys()) +\
                                  list(random_word2vectrue.keys()) + list(active_word2vectrue.keys()))[-1]


    print(max_y, "max_y")
    plt.xlim(200, 1000)
    plt.ylim(0, max_y)
    plt.xticks(np.arange(min_x, max_x, step=int(x_axis_step)))

    # Only need to show axes once
    if category_index != 0:
        plt.gca().axes.get_yaxis().set_visible(False)
               

    output_file.close()
    return handles_labels


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--category', action='store', dest='category', help='The category that was evaluated, with its B-prefix, e.g., B-speculation')
    
    parser.add_argument('--xspace', action='store', dest='xspace', help='An extra space on the x-axes to improve visability of the results')

    parser.add_argument('--xaxisstep', action='store', dest='xaxisstep', help='The step with which the vales on the x-axis are to be printed')

    parser.add_argument('--maxy', action='store', dest='maxy', help='Max y-value')


    
    properties, path_slash_format, path_dot_format = active_learning_preannotation.load_properties(parser)
    
    args = parser.parse_args()
    
    categories = []
    
    # If no categories are given on the comand line, plot results for all categories in the current project (the one that is given with the option --project
    # and that starts with a B
    if not args.category:
        default_path = os.path.join(path_slash_format, OUTPUT_DIR)
        print("No category given, will use all in the folder '" + OUTPUT_DIR + "' that starts with 'B'")
        for dir in os.listdir(default_path):
            if os.path.isdir(os.path.join(path_slash_format, OUTPUT_DIR, dir)):
                if not dir.startswith(".") and dir.startswith("B"):
                    categories.append(dir)
    else:
        categories.append(args.category)

    if not args.xspace:
        xspace = 0
    else:
        xspace = float(args.xspace)

    if not args.xaxisstep:
        x_axis_step = 20
    else:
        x_axis_step = int(args.xaxisstep)

    if not args.maxy:
        max_y = 1.0
    else:
        max_y = float(args.maxy)

    categories.sort()

    fig = plt.figure()
    for index, category in enumerate(categories):
        sub_plot = fig.add_subplot(1, len(categories), index+1)
        print("Plots results for ", category)
        result_path = os.path.join(path_slash_format, OUTPUT_DIR, category)
        print("Reads results from ", result_path)
        handles_labels = read_results(result_path, category, xspace, index, sub_plot, x_axis_step, max_y)



    fig.legend(handles = [handle for (handle, label) in handles_labels][::-1],\
               labels = [label for (handle, label) in handles_labels][::-1])
    plt.subplots_adjust(right = 0.6, wspace = 0.05)
    #plt.show()


    figure_output_path = os.path.join(path_slash_format, OUTPUT_DIR, "_".join(categories) + ".pdf")
    print("Will save pdf of figure at '" + figure_output_path + "'.")
    fig.savefig(figure_output_path)

    


