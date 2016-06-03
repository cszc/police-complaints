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
import sklearn.pipeline


#All other libraries
import argparse, sys, pickle, random, time, json, datetime
import pandas as pd
import numpy as np
import pylab as pl
import pickle
from scipy import optimize
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import itertools
from pydoc import locate

#Own modules
import evaluation

#create timestamp
TS = time.time()
TIMESTAMP = datetime.datetime.fromtimestamp(TS).strftime('%Y-%m-%d %H:%M:%S')

# df = pd.read_csv("test_fulldata.csv") #csv name goes here
label = "Findings Sustained" #predicted variable goes here
TRAIN_SPLITS =[[0,1],[1,2],[2,3]]
TEST_SPLITS = [2,3,4]
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

def temporal_split_data(df):
    df_sorted = df.sort_values(by='dateobj')
    chunks = np.array_split(df_sorted, 5)
    return chunks



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
    #if len(list(set(predictions_binary))) > 1:
       # precision1 = precision_at(y_test, predictions, 0.01)
       # print("Precision at 1%: {} (probability cutoff {})".format(
        #             round(precision1[0], 2), precision1[1]))
        #precision10 = precision_at(y_test, predictions, 0.1)
        #print("Precision at 10%: {} (probability cutoff {})".format(
         #            round(precision10[0], 2), precision10[1]))
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
    values = grid[class_name]
    #Generate cross product for all given values and return them as dicts
    grids = _generate_grid(values)

    return grids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='csv filename')
    # parser.add_argument('label', help='label of dependent variable')
    parser.add_argument('--to_try', help='list of model abbreviations', action='append')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # label = args.label #predicted variable goes here
    to_try = args.to_try
    print(to_try)

    chunks = temporal_split_data(df)

    for chunk in chunks:
        #need to impute nulls and do other trnsformations
        chunk.fillna(0, inplace=True)

    for model_name in to_try:
        print("### STARTING {} ###".format(model_name))
        clf = clfs[model_name]
        grid = grid_from_class(model_name)
        for params in grid:
            clf.set_params(**params)
            if hasattr(clf, 'n_jobs'):
                    clf.set_params(n_jobs=-1)
            print("Initialized model")
            # print(clf)
            steps = [("normalization", preprocessing.RobustScaler()),
         (model_name, clf)]
            pipeline = sklearn.pipeline.Pipeline(steps)
            print("pipeline:", [name for name, _ in pipeline.steps])
            auc_scores = []
            for i in range(len(TRAIN_SPLITS)):
                print("Training sets are: {}".format(TRAIN_SPLITS[i]))
                print("Testing sets are: {}".format(TEST_SPLITS[i]))
                train_df = pd.concat([chunks[TRAIN_SPLITS[i][0]],chunks[TRAIN_SPLITS[i][1]]])
                X_train = train_df.drop(label, axis=1)
                y_train = train_df[label]
                test_df = chunks[TEST_SPLITS[i]]
                X_test = train_df.drop(label, axis=1)
                y_test = test_df[label]
                
                t0 = time.clock()
                clf.fit(X_train,y_train)
                print("done fitting in %0.3fs" % (time.clock() - t0))


                print("Predicting binary outcome on test X")
                predicted = clf.predict(X_test)

                df = pd.DataFrame(predicted)
                # print(df.head())
                print("Value counts for predictions")
                print(df[0].value_counts())
                print()
                print("Predicting probability of outcome on test X")
                try:
                    predicted_prob = clf.predict_proba(X_test)
                except:
                    print("Model has no predict_proba method")
                print("Evaluation Statistics")
                output_evaluation_statistics(y_test, predicted_prob)
                print()
                auc_score = metrics.roc_auc_score(y_test, predicted)
                auc_scores.append(auc_score)
                print("AUC score: %0.3f" % auc_score)

                print("Feature Importance")
                feature_importances = get_feature_importances(grid_search.best_estimator_.named_steps[estimator_name])
                if feature_importances != None:
                    df_best_estimators = pd.DataFrame(feature_importances, columns = ["Imp"], index = X_train.columns).sort(['Imp'], ascending = False)
                    # filename = "plots/ftrs_"+estimator_name+"_"+time.strftime("%d-%m-%Y-%H-%M-%S"+".png")
                    # plot_feature_importances(df_best_estimators, filename)
                    print(df_best_estimators.head(20))
            average_auc = sum(auc_scores)/len(auc)
            print("Average AUC: %0.3f" % auc_score)

        





   
        # scores = cross_validation.cross_val_score(pipeline, X_train, y_train, scoring='roc_auc')
        # print("mean score:")
        # print(estimator, scores.mean())
        # print("Performing grid search...")
        # print("pipeline:", [name for name, _ in pipeline.steps])
        # print("parameters:")
        # print(parameters)
        # t0 = time.clock()
        # grid_search.fit(X_train,y_train)
        # print("done fitting in %0.3fs" % (time.clock() - t0))

        # print("Predicting binary outcome on test X")
        # predicted = grid_search.predict(X_test)

        # df = pd.DataFrame(predicted)
        # # print(df.head())
        # print("Value counts for predictions")
        # print(df[0].value_counts())
        # print()
        # print("Predicting probability of outcome on test X")
        # predicted_prob = grid_search.predict_proba(X_test)

        # predicted_prob = predicted_prob[:, 1]  # probability that label is 1
        # df1 = pd.DataFrame(predicted_prob)

        # # statistics
        # print("Evaluation Statistics")
        # output_evaluation_statistics(y_test, predicted_prob)
        # print()
        # print("Feature Importance")
        # feature_importances = get_feature_importances(grid_search.best_estimator_.named_steps[estimator_name])
        # if feature_importances != None:
        #     df_best_estimators = pd.DataFrame(feature_importances, columns = ["Imp"], index = X_train.columns).sort(['Imp'], ascending = False)
        #     # filename = "plots/ftrs_"+estimator_name+"_"+time.strftime("%d-%m-%Y-%H-%M-%S"+".png")
        #     # plot_feature_importances(df_best_estimators, filename)
        #     print(df_best_estimators.head(20))
        # print("Best score: %0.3f" % grid_search.best_score_)


        # file_name = "pickles/{0}_{1}.p}".format(TIMESTAMP, estimator)
        # data = [] #need to fill in with things that we want to pickle
        # pickle.dump( data, open(file_name, "wb" ) )
