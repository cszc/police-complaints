#Libraries from scikit-learn
from sklearn import preprocessing, cross_validation, metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
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

'''
GLOBAL VARIABLES
'''
#create timestamp
TS = time.time()
TIMESTAMP = datetime.datetime.fromtimestamp(TS).strftime('%Y-%m-%d_%H_%M_%S')


TRAIN_SPLITS_WINDOW =[[0,1],[1,2],[2,3]]
TEST_SPLITS_WINDOW = [2,3,4]
TRAIN_SPLITS_TACK =[[0,1],[0,1,2],[0,1,2,3]]
TEST_SPLITS_TACK = [2,3,4]
TO_DROP = ['crid', 'officer_id', 'agesqrd', 'index', 'Unnamed: 0', '']
FILL_WITH_MEAN = ['officers_age', 'transit_time', 'car_time']
if args.demographic:
    FILL_WITH_MEAN.append('complainant_age')

#estimators
CLFS = {
        'GB': GradientBoostingClassifier(
                    learning_rate=0.05,
                    subsample=0.5,
                    max_depth=6,
                    n_estimators=10)
        }

GRID = {
        'GB': {
                'n_estimators': [1,10,100,1000],
                'learning_rate' : [0.001,0.01,0.05,0.1,0.5],
                'subsample' : [0.1,0.5,1.0],
                'max_depth': [1,3,5,10,20,50,100]
                }
       }


if __name__ == "__main__":
    '''
    Argument Parsing
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='csv filename')
    parser.add_argument('label', help='label of dependent variable')
    parser.add_argument('to_try', nargs='+', default=['DT'], help='list of model abbreviations')
    parser.add_argument('--tack_one_on', action="store_true", default=False, help='use tack one on window validation')
    parser.add_argument('--demographic', action="store_true", default=False, help='using demographic info')
    args = parser.parse_args()

    '''
    Labels for Documenting
    '''
    label = args.label #y, predicted variable
    dem_label = 'with_demo' if args.demographic else 'non-demo'
    stage = 'stage-1'

    if args.tack_one_on:
        train_split, test_split = TRAIN_SPLITS_TACK, TEST_SPLITS_TACK
    else:
        train_split, test_split = TRAIN_SPLITS_WINDOW, TEST_SPLITS_WINDOW

    to_try = args.to_try
    df = pd.read_csv(args.csv)
    if args.demographic:
        FILL_WITH_MEAN.append('complainant_age')

    '''
    Trying Models
    '''
    for model_name in to_try:
        print(model_name)
        '''
        Processing and Imputation
        '''
        this_df = df.drop(TO_DROP)
        chunks = temporal_split_data(this_df, axis=1, inplace=True)

        #IMPUTE EACH TEMPORAL CHUNK
        for chunk in chunks:
            for col in chunk.columns:
                try:
                    if col in FILL_WITH_MEAN:
                        mean = round(df[col].mean())
                        df[col].fillna(value=mean, inplace=True)
                    else:
                        df[col].fillna(0, inplace=True)
                except:
                    print('Could not impute column:{}'.format(col))
                    continue

        '''
        Iterating Through Model
        '''
        print("### STARTING {} ###".format(model_name))
        clf = CLFS[model_name]
        grid = grid_from_class(model_name)

        for i, params in enumerate(grid):
            '''
            Iterating Through Parameters
            '''
            print("### TRYING PARAMS ###")
            clf.set_params(**params)
            if hasattr(clf, 'n_jobs'):
                    clf.set_params(n_jobs=-1)

            folds_completed = 0
            print(clf)

            steps = [(model_name, clf)]
            pipeline = sklearn.pipeline.Pipeline(steps)
            print("pipeline:", [name for name, _ in pipeline.steps])

            auc_scores = []

            for i in range(len(train_split)):
                '''
                Cross Validation
                '''
                print("Training sets are: {}".format(train_split[i]))
                print("Testing sets are: {}".format(test_split[i]))
                for x in range(len(train_split[i])-1):
                    if x == 0:
                        train_df=chunks[train_split[i][x]]

                    train_df = pd.concat(
                        [train_df, chunks[train_split[i][x+1]]])

                X_train = train_df.drop(label, axis=1)
                y_train = train_df[label]
                test_df = chunks[test_split[i]]
                X_test = test_df.drop(label, axis=1)
                y_test = test_df[label]

                t0 = time.clock()
                pipeline.fit(X_train, y_train)
                time_to_fit = (time.clock() - t0)
                print("done fitting in {}".format(time_to_fit))

                '''
                Predictions
                '''
                predicted = pipeline.predict(X_test)

                try:
                    predicted_prob = pipeline.predict_proba(X_test)
                    predicted_prob = predicted_prob[:, 1]  # probability that label is 1

                except:
                    print("Model has no predict_proba method")
                # output_evaluation_statistics(y_test, predicted_prob)

                '''
                Evaluation Statistics
                '''
                print("Evaluation Statistics")
                if model_name=='KNN':
                    print("Getting feature support")
                    features = pipeline.named_steps['feat']
                    print(X_train.columns[features.transform(np.arange(
                        len(X_train.columns)))])

                print()
                auc_score = metrics.roc_auc_score(y_test, predicted)
                precision, recall, thresholds = metrics.precision_recall_curve(
                    y_test, predicted_prob)
                auc_scores.append(auc_score)
                print("AUC score: %0.3f" % auc_score)

                # save predcited/actual to calculate precision/recall
                if folds_completed==0:
                    overall_predictions = pd.DataFrame(predicted_prob)
                    overall_actual = y_test
                else:
                    overall_predictions = pd.concat(
                        [overall_predictions, pd.DataFrame(predicted_prob)])
                    overall_actual = pd.concat([overall_actual, y_test])

                '''
                Important Features
                '''
                print("Feature Importance")
                feature_importances = get_feature_importances(
                    pipeline.named_steps[model_name])
                if feature_importances != None:
                    df_best_estimators = pd.DataFrame(feature_importances, columns = ["Imp/Coef"], index = X_train.columns).sort_values(by='Imp/Coef', ascending = False)
                    # filename = "plots/ftrs_"+estimator_name+"_"+time.strftime("%d-%m-%Y-%H-%M-%S"+".png")
                    # plot_feature_importances(df_best_estimators, filename)
                    print(df_best_estimators.head(5))

                folds_completed += 1

                with open('result/results.csv', 'a') as csvfile:
                    spamwriter = csv.writer(csvfile)
                    spamwriter.writerow(
                        [stage,
                        dem_label,
                        label,
                        model_name,
                        params,
                        folds_completed,
                        auc_score,
                        precision,
                        recall,
                        thresholds])

                file_name = (
                    "results/pickles/{0}_{1}_{2}_paramset:{3}_fold:{4}_{5}.p".format(
                    TIMESTAMP, stage, model_name, i, folds_completed, dem_label)
                    )

                data = [clf, pipeline, predicted, params]
                pickle.dump( data, open(file_name, "wb" ) )

            print("### Cross Validation Statistics ###")
            precision, recall, thresholds = metrics.precision_recall_curve(
                overall_actual, overall_predictions)
            average_auc = sum(auc_scores) / len(auc_scores)
            print("Average AUC: %0.3f" % auc_score)
            print("Confusion Matrix")
            overall_binary_predictions = convert_to_binary(overall_predictions)
            confusion = metrics.confusion_matrix(
                overall_actual, overall_binary_predictions)
            print(confusion)

            with open('results/final_results.csv', 'a') as csvfile:
                spamwriter = csv.writer(csvfile)
                spamwriter.writerow([
                    stage,
                    dem_label,
                    label,
                    model_name,
                    params,
                    folds_completed,
                    precision,
                    recall,
                    thresholds,
                    average_auc])
