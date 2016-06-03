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
from sklearn_evaluation.metrics import precision_at
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import sklearn.pipeline
import evaluation
import argparse
import sys
# %pylab inline

df = pd.read_csv("test_fulldata.csv") #csv name goes here
label = "Findings Sustained" #predicted variable goes here
TEST_SPLITS =[[0,1],[1,2],[2,3]]
TRAIN_SPLITS = [2,3,4]
# X = df.drop(label, axis=1)
# y = df[label]


#estimators
clfs = {
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'AB': AdaBoostClassifier(
                    DecisionTreeClassifier(max_depth=1),
                    algorithm="SAMME",
                    n_estimators=200),
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
                'AB__algorithm': ['SAMME', 'SAMME.R'],
                'AB__n_estimators': [1,10,100,1000]
                },
        'GB': {
                'GB__n_estimators': [1,10,100,1000],
                'GB__learning_rate' : [0.001,0.01,0.05,0.1,0.5],
                'GB__subsample' : [0.1,0.5,1.0],
                'GB__max_depth': [1,3,5,10,20,50,100]
                },
        'NB' : {},
        # 'DT': {
        #         'DT__criterion': ['gini', 'entropy'],
        #         'DT__max_depth': [1,5,10,20,50,100],
        #         'DT__max_features': ['sqrt','log2'],
        #         'DT__min_samples_split': [2,5,10]
        #         },
        'DT': {
                'DT__criterion': ['gini'],
                'DT__max_depth': [1,5],
                'DT__max_features': ['sqrt'],
                'DT__min_samples_split': [5,10]
                },
        'SVM' :{
                'SVM__C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],
                'SVM__kernel':['linear']
                },
        'KNN' :{
                'KNN__n_neighbors': [1,5,10,25,50,100],
                'KNN__weights': ['uniform','distance'],
                'KNN__algorithm': ['auto','ball_tree','kd_tree']
                }
       }

def get_feature_importances(model):
    # print("Trying to get feature importance for:")
    # print(model)
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
def plot_feature_importances(importances, filename):
    plt.figure()
    plt.style.use('ggplot')
    importances.plot(kind="barh", legend=False)
    plt.tight_layout()
    plt.savefig(filename)

def output_evaluation_statistics(y_test, predictions):
    print("Statistics with probability cutoff at 0.5")
    # binary predictions with some cutoff for these evaluations
    cutoff = 0.5
    y_test = y_test.astype(int)
    predictions_binary = np.copy(predictions)
    predictions_binary[predictions_binary >= cutoff] = int(1)
    predictions_binary[predictions_binary < cutoff] = int(0)
    # predictions_binary = predictions_binary.astype(int)

    # print(type)
    # df1 = pd.DataFrame(predictions_binary)
    # # print(df1.head())
    # print("Value counts for binary predictions")
    # print(df1[0].value_counts())
    evaluation.print_model_statistics(y_test, predictions_binary)
    evaluation.print_confusion_matrix(y_test, predictions_binary)
    if len(list(set(predictions_binary))) > 1:
        precision1 = precision_at(y_test, predictions, 0.01)
        print("Precision at 1%: {} (probability cutoff {})".format(
                     round(precision1[0], 2), precision1[1]))
        precision10 = precision_at(y_test, predictions, 0.1)
        print("Precision at 10%: {} (probability cutoff {})".format(
                     round(precision10[0], 2), precision10[1]))

def temporal_split_data(df):
    df_sorted = df.sort_values(by='dateobj')
    chunks = np.array_split(df_sorted, 5)
    return chunks

if __name__ == "__main__":
    # can pass in a list of models to try
    if len(sys.argv) > 1:
        to_try = sys.argv[1:]
    else:
        to_try = ['DT','LR','RF']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    chunks = temporal_split_data(df)

    for chunk in chunks:
        #need to impute nulls and do other trnsformations
        pass 

    for model_name in to_try:
        model = clfs[model_name]
         





    # find the best parameters for both the feature extraction and the
    # classifier
    #runs through each estiamtor
    for estimator in to_try:
    # for estimator in clfs.items():
        #will need to figure out what preprocessing we want
        #http://scikit-learn.org/stable/modules/preprocessing.html
        print(estimator)
        estimator_name = estimator
        estimator_obj = clfs[estimator]

        print("Trying {}".format(estimator_name))
        steps = [("normalization", preprocessing.RobustScaler()),
         (estimator_name, estimator_obj)]
        pipeline = sklearn.pipeline.Pipeline(steps)
        print("Created pipeline")
        parameters = grid[estimator_name]
        print("Starting parameter search")
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=0)
        scores = cross_validation.cross_val_score(pipeline, X_train, y_train, scoring='roc_auc')
        print("mean score:")
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
        # print("Value counts for prob predictions")
        # print(df1[0].value_counts())
        # print()
        # ydf = pd.DataFrame(y_test)
        # # print(ydf.head())
        # print("Value counts for y actual")
        # print(ydf[label].value_counts())
        # print()
        # statistics
        print("Evaluation Statistics")
        output_evaluation_statistics(y_test, predicted_prob)
        print()
        print("Feature Importance")
        feature_importances = get_feature_importances(grid_search.best_estimator_.named_steps[estimator_name])
        if feature_importances != None:
            df_best_estimators = pd.DataFrame(feature_importances, columns = ["Imp"], index = X_train.columns).sort(['Imp'], ascending = False)
            # filename = "plots/ftrs_"+estimator_name+"_"+time.strftime("%d-%m-%Y-%H-%M-%S"+".png")
            # plot_feature_importances(df_best_estimators, filename)
            print(df_best_estimators.head(20))
        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))