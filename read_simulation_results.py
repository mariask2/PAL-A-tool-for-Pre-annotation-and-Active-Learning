# Code to transform simulation results into a format that can be read by gnuplot
# As well as to do calculations for average f-score
# Run for instance as:
# python read_simulation_results.py --project=data.example_project --category=B-speculation

import argparse
import os
import glob

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

def write_dict(name, result_dict, output_file):
    output_file.write(name + "\n")
    output_file.write("# sample size\tf_score\n")
    for key in sorted(result_dict.keys()):
        mean = sum(result_dict[key])/len(result_dict[key])
        output_file.write(str(key) + "\t" + str(mean) + "\n")
    output_file.write("\n\n")

def read_results(result_path):
    random_word2vecfalse = {}
    random_word2vectrue = {}
    active_word2vecfalse = {}
    active_word2vectrue = {}
    
    for subdir in [el for el in os.listdir(result_path) if not el.startswith(".")]:
        full_subdir = os.path.join(result_path, subdir)
        random_dir = os.path.join(full_subdir, RANDOM)
        active_dir = os.path.join(full_subdir, ACTIVE)

        extract_data(random_word2vecfalse, glob.glob(os.path.join(random_dir, "*False*conll_res.txt")))
        extract_data(random_word2vectrue, glob.glob(os.path.join(random_dir, "*True*conll_res.txt")))
        extract_data(active_word2vecfalse, glob.glob(os.path.join(active_dir, "*False*conll_res.txt")))
        extract_data(active_word2vectrue, glob.glob(os.path.join(active_dir, "*True*conll_res.txt")))

    output_file = open(os.path.join(result_path, "conll_media_fscore.dat"), "w")
    write_dict("#random_word2vecfalse", random_word2vecfalse, output_file)
    write_dict("#active_word2vecfalse", active_word2vecfalse, output_file)
    write_dict("#random_word2vectrue", random_word2vectrue, output_file)
    write_dict("#active_word2vectrue", active_word2vectrue, output_file)

    output_file.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--category', action='store', dest='category', help='The category that was evaluated, with its B-prefix, e.g., B-speculation')
    
    properties, path_slash_format, path_dot_format = active_learning_preannotation.load_properties(parser)
    
    args = parser.parse_args()
    
    if not args.category:
        print("No category given, e.g. B-speculation")
    
    result_path = os.path.join(path_slash_format, OUTPUT_DIR, args.category)
    print("Reads results from ", result_path)
    read_results(result_path)

    


