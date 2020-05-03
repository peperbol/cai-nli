import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB as nb

#################################################
#	Generation of the dataframe containing the 	#
#	raw training text and their assigned labels	#
#################################################

df = pd.DataFrame(columns = ['label', 'text'])

cnt = 1

data_directory = 'nli-training'

for filename in os.listdir(data_directory):
    print(f'Now processing text: {filename}, text {cnt} of {len(os.listdir(data_directory))}')
    cnt += 1
    
    label_, _ = filename.split("_")
    doc_ = open(f'{data_directory}/{filename}', 'r').read()
    
    df = df.append({'label': label_, 'text': doc_}, ignore_index = True)
    

df.to_csv(r'training_dataset.csv')


print('Dataframe generation finished')

#################################################
#	Generation of the necessary vectorizers, 	#
#			pipeline and trainers 				#
#################################################

countvec = CountVectorizer()
dictvec = DictVectorizer()
svm = SVC()