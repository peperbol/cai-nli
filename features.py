import spacy
import pandas as pd
import os

'''
1. Open the csv file with the raw texts and their labels

2. Preprocess each text to calculate their syntactic and stylistic complexity
    The script needs a CoNLL text file for each raw text to calculate SSC
    2.1. Extract raw text file from csv
    2.2. Create empty list to store features per text
    2.2. Extract features with spacy:
        - Word index
        - Word
        - POS tag
        - Index of dependency head
        - Dependency relation
        - '_' -> Phrase structure tree is not necessary nor applicable here
    2.3. Add features to list
        ! Make sure to add empty line after each sentence.
    2.4. Store list as CoNLL formatted txt file, tab separated, in a new folder
        - File name needs to be the index of the file in the original csv

3. Calculate SSC
    3.1. Loop through files in the folder with the CoNLL files
    3.2. Run the SSC script on each file
    3.3. Add SSC score to training_dataset.csv to the according index
'''


model = spacy.load("en_core_web_sm")

#################################################
#	Generation of a CoNLL styled text file with
#	the following features:
#   word index, word, part-of-speech tag,
#   index of dependency head, dependency relation,
#   phrase structure tree
#   ->  Preprocess to calculate the syntactic
#       and stylistic complexity
#################################################

texts = pd.read_csv('training_dataset.csv', index_col = 0)

# Generate new folder if not yet available for the formatted texts
mypath = '/ssc_data'
if not os.path.isdir(mypath):
   os.makedirs(mypath)

# Initialise counter for feedback and index to name the files
cnt = 1
i = 0

# Loop through texts in the original training dataset file
for doc_ in texts['text']:
    print(f'Processing text {cnt} of {len(texts)}')
    conll = []
    doc = model(doc_)

    # Extract features and add them to a list to be converted into txt
    for sent in doc.sents:
        for token in sent:
            row = [str(token.i - sent.start), token.text, token.pos_, str(token.head.i - sent.start), token.dep_, '_']
            conll.append(row)
            if token.i + 1 == sent.end:
                conll.append(['\n'])


    # Write new file

    t = '\t'

    with open(f'{mypath}/{i}.txt', 'w+') as f:
        for line in conll:
            f.write(t.join(line))
            f.write('\n')

    cnt += 1
    i += 1


#################################################
#	             Calculate SSC                  #
#################################################
