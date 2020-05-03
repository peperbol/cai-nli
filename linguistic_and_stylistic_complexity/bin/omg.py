from conllu import parse_tree
import os

data_directory = "nli"



for filename in os.listdir(data_directory):
    y, _ = filename.split("_")
    doc = open(os.path.join(data_directory, filename), "r").read()
    sentences = parse_tree(doc)
    print(sentences)
