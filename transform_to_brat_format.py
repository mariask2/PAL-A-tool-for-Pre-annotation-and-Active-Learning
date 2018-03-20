import os



def transform(taggedFileName, outputdir, interestingTags, outside_class, beginning_prefix):
    taggedFile = open(taggedFileName, encoding="utf-8")
    outputTextFileName = os.path.join(outputdir, "brat_" + os.path.basename(taggedFileName).split(".")[0] + ".txt")
    outputAnnFileName = os.path.join(outputdir, "brat_" + os.path.basename(taggedFileName).split(".")[0] + ".ann")
    outputTextFile = open(outputTextFileName, "w", encoding="utf-8")
    outputAnnFile = open(outputAnnFileName, "w", encoding="utf-8")
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
                # brat wants an offset in utf16 length
                utf16_length = int(len(l.encode("utf-16-be"))/2)
                outputTextFile.write(l)
                tokenCounter = tokenCounter + utf16_length
            first = False
        else:
            if insideATag:
                outputAnnFile.write(str(tokenCounter) + u'\t' + currentEntity + u'\n')
                insideATag = False
                currentEntity = ""
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
    #transform("data/example_project/tolabel/tolabel_20180319_135958.csv", "temp", ["speculation", "contrast"], "O", "B-")
    #transform("data/twitter_test/tolabel/tolabel_20180320_105209.csv", "temp", ["per", "org", "loc"], "O", "B-")
    transform("data/twitter_test/tolabel/tolabel_20180320_142014.csv", "temp", ["per", "org", "loc"], "O", "B-")

