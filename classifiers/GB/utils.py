#Libraries from scikit-learn
from sklearn.svm import SVC
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
import sklearn.feature_selection
import sklearn.pipeline

#All other libraries
import argparse, sys, pickle, random, time, json, datetime, itertools, csv
import matplotlib
import pandas as pd
import numpy as np
import pylab as pl
from scipy import optimize
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from pydoc import locate
from argparse import ArgumentParser

#Own modules
import evaluation
from utils import *

CLFS = {
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
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'test': DecisionTreeClassifier(),
        }

GRID = {
        'RF': {
                'n_estimators': [1,10,100,1000],
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
                'n_estimators': [1,10,100,1000]
                },
        'GB': {
                'n_estimators': [1,10,100,1000],
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
        'test': {
                'criterion': ['gini'],
                'max_depth': [1,5],
                'max_features': ['sqrt'],
                'min_samples_split': [5,10]
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
'''
Data Transformation
'''

def add_col(df, col, transform_type='log'):
    '''
    Adds new columns to a dataframe. Current options are log and squared.
    Returns dataframe with new column.
    '''
    if transform_type == 'log':
        #+1 to avoid taking log of 0
        df['log_'+col] = np.log(df[col] + 1)
    elif transform_type == 'squared':
        df[col+'^2'] = df[col]**2
    else:
        print('type not supported')
    return df

def transform_feature( df, column_name ):
    unique_values = set( df[column_name].tolist() )
    transformer_dict = {}
    for ii, value in enumerate(unique_values):
        transformer_dict[value] = ii

    def label_map(y):
        return transformer_dict[y]
    df[column_name] = df[column_name].apply( label_map )
    return df


'''
Data Manipulation (Test/Train Splits)
'''

def temporal_split_data(df):
    '''
    Splits data into 3 temporal folds, tacking on more data for each new
    time window at each fold.
    '''
    df_sorted = df.sort_values(by='dateobj')
    chunks = np.array_split(df_sorted, 5)
    for chunk in chunks:
        chunk.drop('dateobj',axis=1, inplace=True)
    return chunks


'''
Iterating Over Models/Parameters
'''

def _generate_grid(model_parameters):
    #Iterate over keys and values
    parameter_groups = []
    for key in model_parameters:
        #Generate tuple key,value
        t = list(itertools.product([key], model_parameters[key]))
        parameter_groups.append(t)
    #Cross product over each group
    parameters = list(itertools.product(*parameter_groups))
    #Convert each result to dict
    dicts = [_tuples2dict(params) for params in parameters]
    return dicts


def _tuples2dict(tuples):
    return dict((x, y) for x, y in tuples)


def grid_from_class(class_name):
    #Get grid values for the given class
    values = GRID[class_name]
    #Generate cross product for all given values and return them as dicts
    grids = _generate_grid(values)
    return grids


'''
Model Evaluation
'''
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
    evaluation.print_model_statistics(y_test, predictions_binary)
    evaluation.print_confusion_matrix(y_test, predictions_binary)


def convert_to_binary(predictions):
    print("Statistics with probability cutoff at 0.5")
    # binary predictions with some cutoff for these evaluations
    cutoff = 0.5
    predictions_binary = np.copy(predictions)
    predictions_binary[predictions_binary >= cutoff] = int(1)
    predictions_binary[predictions_binary < cutoff] = int(0)
    return predictions_binary
