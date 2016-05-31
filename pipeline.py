import sklearn
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import matplotlib
import json
matplotlib.style.use('ggplot')
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/allegations_clean.csv") #csv name goes here
label = "" #predicted variable goes here
X = df.drop(label, axis=1)
y = df[label]

#estimators
clf = {
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        # 'AB': AdaBoostClassifier(
        #             DecisionTreeClassifier(max_depth=1),
        #             algorithm="SAMME",
        #             n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(
                    learning_rate=0.05,
                    subsample=0.5,
                    max_depth=6,
                    n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=3)
        }

#not yet used. eventually, will iterate through parameters for estimators
grid = {
        'RF': {
                'n_estimators': [1,10,100,1000,10000],
                'max_depth': [1,5,10,20,50,100],
                'max_features': ['sqrt','log2'],
                'min_samples_split': [2,5,10]
                },
        'LR': {
                'penalty': ['l1','l2'],
                'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]
                },
        'AB': {
                'algorithm': ['SAMME', 'SAMME.R'],
                'n_estimators': [1,10,100,1000,10000]
                },
        'GB': {
                'n_estimators': [1,10,100,1000,10000],
                'learning_rate' : [0.001,0.01,0.05,0.1,0.5],
                'subsample' : [0.1,0.5,1.0],
                'max_depth': [1,3,5,10,20,50,100]
                },
        'NB' : {},
        'DT': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [1,5,10,20,50,100],
                'max_features': ['sqrt','log2'],
                'min_samples_split': [2,5,10]
                },
        'SVM' :{
                'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],
                'kernel':['linear']
                },
        'KNN' :{
                'n_neighbors': [1,5,10,25,50,100],
                'weights': ['uniform','distance'],
                'algorithm': ['auto','ball_tree','kd_tree']
                }
       }


if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    #runs through each estiamtor
    for estimator in clfs.items():
        #will need to figure out what preprocessing we want
        #http://scikit-learn.org/stable/modules/preprocessing.html
        estimator_name = estimator[0]

        clf = pipeline.Pipeline([
         ("normalization", preprocessing.RobustScaler()),
         estimator)
        ])
        parameters = grid[estimator_name]
        grid_search = GridSearchCV(clf, parameters, n_jobs=-1, verbose=1)
        scores = cross_validation.cross_val_score(clf, X, y, scoring='roc_auc')

        print(estimator, scores.mean())

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
