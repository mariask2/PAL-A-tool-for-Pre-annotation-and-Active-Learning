
"""
 a VERY simple tokenizer. Just tokenizes on space and a few common punctiation marks.
 Works for simple applications and some languagues though.
"""

def remove_spaces(line):
    """
    Removes double spaces
    """
    line = line.lstrip()
    result = ""
    for i, ch in enumerate(line):
        if i+1 == len(line):
            result = result + ch
        elif (ch == " " and (line[i+1] == " " or line[i+1] == "\n")):
            pass
        else:
            result = result + ch

    return result


def simple_tokenize(file_name):
    """
    Tokenize the content of a file with only one sentence in it.
    """
    f = open(file_name)
    text = f.read()
    f.close()
    return simple_tokenize_list([text])

def simple_tokenize_list(lines):
    """
    Tokenizes each string in a list of strings, and replaces
    tab with space, and double space with space.
    param: A list of strings
    :returns A list that contains a list of tokens in the input string
    """
    output = []
    for line in lines:
        if line.strip() != "":
            line = line.strip()
            line = line.replace("\t", " ")
            for ch in ['.', ',', '!', '?', '%', ":", ";", '"', "'", "-", "/", "(", ")"]:
                line = line.replace(ch, " " + ch + " ")
            line = line.strip()
            removed_space = remove_spaces(line)
            output.append(removed_space.split(" "))
    return output


#simple_tokenize("2000_more_other.txt", "2000_more_other_tokenized.txt")
