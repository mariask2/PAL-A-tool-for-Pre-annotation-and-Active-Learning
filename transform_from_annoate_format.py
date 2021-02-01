import os

def transform(taggedFileName, outputdir, interestingTags, outside_class, beginning_prefix, inside_prefix, inside_tag):
    taggedFile = open(taggedFileName, encoding="utf-8")
    outputFileName = os.path.join(outputdir, "pal_" + os.path.basename(taggedFileName).split(".")[0] + ".csv")
    outputFile = open(outputFileName, "w", encoding="utf-8")
    tokenCounter = 0

    current_entity = outside_class
    for line in taggedFile:
        line = line.strip()
        if line != "":
            sp = line.split('\t')
            word = sp[0]
            label = sp[-1]
            
            if label == outside_class:
                current_entity == outside_class
            elif label.startswith(beginning_prefix):
                current_entity = label[2:]
            elif label == inside_tag:
                if current_entity == outside_class:
                    print("Inside class should not follow outside tag")
                    print(line)
                    exit(1)
                else:
                    label = inside_prefix + current_entity
            else:
                print("Incorrect label", line)
            tokenCounter = tokenCounter + 1
        
            outputFile.write(word + "\t" + label + "\n")
        else:
            outputFile.write('\n')
    
    outputFile.flush()
      
    outputFile.close()
    taggedFile.close()
    
    print(tokenCounter)


if __name__ == "__main__":   transform("/Users/marsk757/pal/PAL-A-tool-for-Pre-annotation-and-Active-Learning/data/covid-19/tolabel/20210131_181502.csv", "annotate_test", ["RISK", "NO"], "O", "B-", "I-", "I")

