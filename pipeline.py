from generation import countvec, svm, Pipeline, dictvec, nb

##########################
#    Simple pipeline     #
##########################

pipe = Pipeline([("vec", countvec), ("cls", svm)])
pipe.fit(X, Y)

#########################################
#    Gridsearch on the trainingdata     #
#########################################


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
