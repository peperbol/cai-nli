##################
# Import modules #
##################

import os
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

##################
#   Load data    #
##################

data_directory = "data"
X, Y = [], []
for filename in os.listdir(data_directory):
    y, _ = filename.split("_")
    x = open(os.path.join(data_directory, filename), "r").read()
    Y.append(y)
    X.append(x)

##########################
#   Check what you get   #
##########################


print(X[0])
print(Y[:3])

##########################
#    Simple pipeline     #
##########################

countvec = CountVectorizer()

svm = SVC()

pipe = Pipeline([("vec", countvec), ("cls", svm)])
pipe.fit(X, Y)

#########################################
#    Gridsearch on the trainingdata     #
#########################################

dictvec = DictVectorizer()

nb = MultinomialNB()

params = {
    "cls__fit_prior": [True, False],
    "cls__alpha": [0.1, 0.5, 1.0]}

pipe = Pipeline([("vec", dictvec), ("cls", nb)])
search = GridSearchCV(pipe, params, cv=10, n_jobs=-1, verbose=5)
search.fit(X_train, y_train)

print(search.best_estimator_)
print(search.best_score_)
print(search.best_params_)
