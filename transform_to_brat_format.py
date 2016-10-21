import os



def transform(taggedFileName, outputdir, interestingTags, outside_class, beginning_prefix):
    taggedFile = open(taggedFileName)
    outputTextFileName = os.path.join(outputdir, "brat_" + os.path.basename(taggedFileName).split(".")[0] + ".txt")
    outputAnnFileName = os.path.join(outputdir, "brat_" + os.path.basename(taggedFileName).split(".")[0] + ".ann")
    outputTextFile = open(outputTextFileName, "w")
    outputAnnFile = open(outputAnnFileName, "w")
    entityCounter = 0
    tokenCounter = 0
    first = True
    insideATag = False
    currentEntity = ""

    for line in taggedFile:
        if line.strip() != "":
            #sp = line.decode('utf8').split(u'\t')
            sp = line.split('\t')
            word = sp[0]
            pos = sp[1]
            label = sp[-1].strip()

            if len(word) == 3 and word[1] == "_":
                word = word[0] # To cover for a bug in scikit learn, one char tokens have been transformed to longer. These are here transformed back
 
            if insideATag and (label == outside_class or label.startswith(beginning_prefix)): # determine if the last token was the last for its span
                    #outputAnnFile.write((str(tokenCounter) + u'\t' + currentEntity + u'\n').encode('utf8'))
                outputAnnFile.write(str(tokenCounter) + u'\t' + currentEntity + u'\n')
                insideATag = False
                currentEntity = ""
        
            if not first:
                outputTextFile.write(" ")
                tokenCounter = tokenCounter + 1
        
            # determine if a new span should start here
            if label.startswith(beginning_prefix)  and label.split(u"-")[1] in interestingTags:
                insideATag = True
                entityCounter = entityCounter + 1
                #outputAnnFile.write(("T" + str(entityCounter) + u'\t' + label.split("-")[1]  + ' ' + str(tokenCounter) + ' ').encode('utf8'))
                outputAnnFile.write(("T" + str(entityCounter) + u'\t' + label.split("-")[1]  + ' ' + str(tokenCounter) + ' '))
            # if inside a span, always add to currentEntity
            if insideATag:   
                currentEntity = currentEntity + word + u' '        

            for l in word:
                #outputTextFile.write(l.encode('utf8'))
                outputTextFile.write(l)
                tokenCounter = tokenCounter + 1
            first = False
        else:
            outputTextFile.write('\n')
            tokenCounter = tokenCounter + 1
            first = True
    outputTextFile.flush()
    outputAnnFile.flush()        
    outputTextFile.close()
    taggedFile.close()
    outputAnnFile.close()
    print(tokenCounter)


if __name__ == "__main__":
    transform("data/example_project/tolabel/tolabel_20161018_125530.csv", "temp", ["speculation", "contrast"], "O", "B-")

