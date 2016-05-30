from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
from sklearn.cross_validation import cross_val_score


# 1. Read Data: For this assignment, assume input is CSV

def read_data(filename, index_col):
    '''
    Reads data from csv
    '''
    df = pd.read_csv(filename, index_col = index_col)
    return df

# 2. Explore Data: You can use the code you wrote for assignment 1 here to generate distributions and data summaries
def explore_data(df):
    print('### Summary Statistics ###')
    for col in df.describe():
        print(col)
        print(df.describe()[col])
        print("Null values: {}".format(df[col].isnull().sum()))
        print()
   
    print('### Correlations ###')
    for col in df.corr():
        print("### "+col+" ###")
        print(df.corr()[col])
        print()

def make_hist(df, col, num_bins=10):
    '''
    Generate a simple histogram
    df: DataFrame
    col: column to visualize
    num_bins: how many bins to have for the visualization
    '''
    print('Plotting ' + col)
    plt.hist(df[col], bins = num_bins)
    plt.title(col)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    filename = 'imgs/'+str(col) + '_hist'
    plt.savefig(filename)
    plt.close()

# 3. Pre-Process Data: Fill in misssing values

def split_train_test(df, features, outcome, test_size):
    '''
    Splits dataframe into test/train sets
    Inputs:
    features: list of feature column names as strings
    outcome: outcome column name as a string
    test size: float between 0 and 1

    returns: four dataframes
    '''
    X = df[features]
    y = df[outcome]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def fill_train_na_mean(df, cols_to_fill=False):
    '''
    Fills missing values of training set with mean
    cols_to_fill: list of columns to impute values for. If none specified, fill all.

    Returns:
        vals: dictionary mapping transformed columns to the value used to fill them
    '''
    vals = {}
    #if no columns specified, fill all
    if not cols_to_fill:
        cols_to_fill = list(df.columns.values)
    for col in cols_to_fill:
        mean = df[col].mean()
        df[col].fillna(mean, inplace=True)
        vals[col] = mean

    return vals


def fill_test_na_mean(df, vals):
    '''
    Fills null values in test dataset with training set means
    '''
    for col in vals.keys():
        df[col].fillna(vals[col], inplace=True)




# 4. Generate Features: Write a sample function that can discretize a continuous variable and 
# one function that can take a categorical variable and create binary variables from it.

def transform_squared(df, col, features):
    '''
    Creates a new feature by squaring another column
    '''

    new_name = str(col) + '_sq'
    df[new_name] = df[col]**2
    if new_name not in features:
        features.append(new_name)



def transform_log(df, col, features, amt_to_add=0):
    '''
    Creates a new feature by squaring another column
    If some values are zero, specify a small amount to amt_to_add
    to avoid taking log of zero
    '''
    new_name = str(col) + '_log'
    if amt_to_add != 0:
        df['temp'] = df[col]+amt_to_add
    df[new_name] = np.log(df['temp'])
    df.drop('temp', axis=1, inplace=True)
    if new_name not in features:
        features.append(new_name)

def discretize(df, col, bins, labels):
    '''
    Discretizes a continuous variable 
    Inputs:
        df: dataframe
        col: col name of variable to discretize
        bins: integer or list of cutoff points for different bins
        labels: False or list of names for each bin. must be same length as number of bins
    Modifies existing dataframe in place
    '''
    new_name = str(col) + '_disc'
    df[new_name] = pd.cut(df[col], bins, labels=labels)

def binarize(df, col):
    '''
    Takes a categorical variable and creates binary dummy variables from it
    Inputs:
        df: dataframe
        col: col name of variable to discretize
    
    Returns: new dataframe
    '''
    dummies = pd.get_dummies(df[col])
    df = df.join(dummies)
    return df

# Fit and evaluate models
# Grid credit to https://github.com/rayidghani/magicloops

def define_clfs_params():
    '''
    Returns dictionaries of classifiers and respective parameters to try
    Grid courtesy of https://github.com/rayidghani/magicloops
    '''

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3)
        # 'SVML': svm.LinearSVC()
            }

    grid = { 
    # 'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'RF':{'n_estimators': [1,10,100], 'max_depth': [1,5,10], 'max_features': ['sqrt','log2'],'min_samples_split': [5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100]},
    # 'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    # 'SVML': {'C': [.0001, .001, 0.1,1, 10], 'dual': [False]},
    'SVM' :{'C' :[1],'kernel':['linear']},
    # 'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
    # 'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }

    return clfs, grid

def try_models(clfs_to_try, clfs, grid, X_train, y_train, X_test, y_true):
    '''
    Fits, tests, and scores models using GridSearchCV
    Tries every possible permutation of paramaters provided for each model specified
    and outputs the best variation found for each type of model
    Inputs:
        clfs_to_try: list of classifiers to evaluate
        clfs: dictionary of models
        grid: dictionary of parameters
        X_train, y_train, X_test, y_true: dataframes
    Outputs:
        results: Dictionary with an entry for each model tried:
                    - cv: GridSearchCV object containing best fitted model
                    - metrics including classification score, TPR, FPR etc
                    - Y_pred, Y_pred_prob: predicted outcomes for X_test set
                    - Time: seconds to train and predict model
    '''
    results = {}
    for c in clfs_to_try:
        start_time = time.time()
        clf = clfs[c]
        steps = [(str(c), clf)]
        params = {}
        for p in grid[c]:
            name = str(c) + '__' + str(p)
            params[name] = grid[c][p]
        pipe = Pipeline(steps)
        cv = GridSearchCV(pipe, param_grid=params, cv = 3)
        cv.fit(X_train, y_train)
        y_pred = cv.predict(X_test)
        y_pred_prob = cv.predict_proba(X_test)[:, 1]
        end_time = time.time()
        classification = metrics.classification_report(y_true, y_pred)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        roc_auc_score = metrics.roc_auc_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_prob)
        rv = dict([("Best Model", cv),("Classification Report", classification),("ROC/AUC Score", roc_auc_score),
            ("Confusion matrix", conf_matrix),("Accuracy", accuracy),("FPR", fpr),
            ('TPR',tpr),("Thresholds",thresholds),
            ("Y_pred", y_pred),("Y_pred_prob",y_pred_prob),("Time",end_time-start_time)])
        results[c] = rv
        print_report(results, c)
    return results
    
def print_report(results, clf):
    '''
    Prints evaluation report for fitted model
    inputs:
        results: dictionary returned by try_models()
        clf: key name of classifier to score
    '''
    print('###')
    print(clf)
    print('###')
    print()
    print("Time: {}s".format(round(results[clf]['Time'])))
    print("Accuracy: {}".format(round(results[clf]['Accuracy'],2)))
    print("ROC/AUC Score: {}".format(round(results[clf]['ROC/AUC Score'],2)))
    print("Classication Report:")
    print(results[clf]['Classification Report'])
    print()
    plt.plot(results[clf]['FPR'], results[clf]['TPR'])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC curve for {}'.format(clf))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show() 

def print_full_report(results):
    '''
    Prints evaluation report for all fitted models
    inputs:
        results: dictionary returned by try_models()
    '''
    for clf in results.keys():
        print_report(results, clf)

# 5. Generate predictions on held out data


def predict(final_df, features, clf):
    '''
    Predict outcomes for test data and write to csv
    inputs:
        test_df: test DataFrame
        features: list of features to use
        clf: fitted model
    '''
    X = final_df[features]
    y = clf.predict(X)
    index_list = list(final_df.index)
    y_df = pd.DataFrame(y, index=index_list, columns=['predicted'])
    final = pd.concat([final_df, y_df], axis=1)
    final.to_csv('predictions.csv', header=True)


