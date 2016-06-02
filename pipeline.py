import sklearn
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import random
import matplotlib
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
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
import sklearn.pipeline
import evaluation

df = pd.read_csv("test_fulldata.csv") #csv name goes here
label = "Findings Sustained" #predicted variable goes here
X = df.drop(label, axis=1)
y = df[label]

#estimators
clfs = {
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        # 'AB': AdaBoostClassifier(
        #             DecisionTreeClassifier(max_depth=1),
        #             algorithm="SAMME",
        #             n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        # 'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        # 'GB': GradientBoostingClassifier(
        #             learning_rate=0.05,
        #             subsample=0.5,
        #             max_depth=6,
        #             n_estimators=10),
        # 'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        # 'KNN': KNeighborsClassifier(n_neighbors=3)
        }

#not yet used. eventually, will iterate through parameters for estimators
grid = {
        'RF': {
                'RF__n_estimators': [1,10,100,1000],
                'RF__max_depth': [1,5,10,20,50,100],
                'RF__max_features': ['sqrt','log2'],
                'RF__min_samples_split': [2,5,10]
                },
        'LR': {
                'LR__penalty': ['l1','l2'],
                'LR__C': [0.00001,0.0001,0.001,0.01,0.1,1,10]
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
                'DT__criterion': ['gini', 'entropy'],
                'DT__max_depth': [1,5,10,20,50,100],
                'DT__max_features': ['sqrt','log2'],
                'DT__min_samples_split': [2,5,10]
                },
        # 'DT': {
        #         'DT__criterion': ['gini'],
        #         'DT__max_depth': [1,5],
        #         'DT__max_features': ['sqrt'],
        #         'DT__min_samples_split': [2,5,10]
        #         },
        'SVM' :{
                'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],
                'kernel':['linear']
                },
        'KNN' :{
                'KNN__n_neighbors': [1,5,10,25,50,100],
                'KNN__weights': ['uniform','distance'],
                'KNN__algorithm': ['auto','ball_tree','kd_tree']
                }
       }

def get_feature_importances(model):
    try:
        return model.feature_importances_
    except:
        
        try:
            print('This model does not have feature_importances, '
                          'returning .coef_[0] instead.')
            return model.coef_[0]
        except:
            print('This model does not have feature_importances, '
                          'nor coef_ returning None')
            return None

def output_evaluation_statistics(y_test, predictions):
    print("Statistics with probability cutoff at 0.5")
    # binary predictions with some cutoff for these evaluations
    cutoff = 0.5
    predictions_binary = np.copy(predictions)
    predictions_binary[predictions_binary >= cutoff] = 1
    predictions_binary[predictions_binary < cutoff] = 0
    # df1 = pd.DataFrame(predictions_binary)
    # # print(df1.head())
    # print("Value counts for binary predictions")
    # print(df1[0].value_counts())
    evaluation.print_model_statistics(y_test, predictions_binary)
    evaluation.print_confusion_matrix(y_test, predictions_binary)

    precision1 = precision_at(y_test, predictions, 0.01)
    logger.debug("Precision at 1%: {} (probability cutoff {})".format(
                 round(precision1[0], 2), precision1[1]))
    precision10 = precision_at(y_test, predictions, 0.1)
    logger.debug("Precision at 10%: {} (probability cutoff {})".format(
                 round(precision10[0], 2), precision10[1]))

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # find the best parameters for both the feature extraction and the
    # classifier
    #runs through each estiamtor
    for estimator in clfs.items():
        #will need to figure out what preprocessing we want
        #http://scikit-learn.org/stable/modules/preprocessing.html
        estimator_name = estimator[0]
        estimator_obj = estimator[1]

        print("Trying {}".format(estimator_name))
        steps = [("normalization", preprocessing.RobustScaler()),
         (estimator_name, estimator_obj)]
        pipeline = sklearn.pipeline.Pipeline(steps)
        print("Created pipeline")
        parameters = grid[estimator_name]
        print("Starting parameter search")
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=0)
        scores = cross_validation.cross_val_score(pipeline, X_train, y_train, scoring='roc_auc')

        print(estimator, scores.mean())
        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        print(parameters)
        t0 = time.clock()
        grid_search.fit(X_train,y_train)
        print("done fitting in %0.3fs" % (time.clock() - t0))
        
        print("Predicting binary outcome on test X")
        predicted = grid_search.predict(X_test)
        
        df = pd.DataFrame(predicted)
        # print(df.head())
        print("Value counts for predictions")
        print(df[0].value_counts())
        print()
        print("Predicting probability of outcome on test X")
        predicted_prob = grid_search.predict_proba(X_test)

        predicted_prob = predicted_prob[:, 1]  # probability that label is 1
        df1 = pd.DataFrame(predicted_prob)
        # print(df1.head())
        print("Value counts for prob predictions")
        print(df1[0].value_counts())
        print()
        ydf = pd.DataFrame(y_test)
        # print(ydf.head())
        print("Value counts for y actual")
        print(ydf[label].value_counts())
        print()
        # statistics
        # output_evaluation_statistics(y_test, predicted_prob)
        print("Feature Importance")
        feature_importances = get_feature_importances(grid_search)
        print(feature_importances)
        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
