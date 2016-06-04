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

'''
GLOBAL VARIABLES
'''
#create timestamp
TS = time.time()
TIMESTAMP = datetime.datetime.fromtimestamp(TS).strftime('%Y-%m-%d_%H_%M_%S')

#what is this?
FTRS = [6,7,8,9,16,17,18,21,24,27,29,30,35,37,40,42,44,45,50,51,52,57,62,63,66,70,71,76,77,80,81,85,88,91,94,96,98,103,104,106,116,117,119,120,121,122,128,135,136,138,139,140,142,143,144,147,151,152,155,161,163,165,166,167,168,170,171,172,173,174,181,182,183,184,185,193,194,195,196,197,198,200,202,205,207,208,214,215,218,222,225,226,228,230,231,236,243,244,252,253,255,256,257,258,259,262,264,266,268,269,273,279,285,293,296,297,298,303,304,306,308,312,319,322,328,331,334,337,338,343,344,348,352,359,360,363,364,365,368,369,371,372,377,381,382,385,386,387,388,389,395,397,398,401,409,410,411,414,415,420,429,430,431,437,442,444,446,447,448,454,455,458,463,464,466,467,468,469,471,479,480,481,483,485,489,491,493,494,497,499,500,502,507,509,512,515,516,518,521,524,530,531,532,535,537,538,539,544,547,548,550,554,555,560,561,564,570,571,574,576,577,578,579,580,586,587,590,594,596,598,599,600,606,607,611,617,624,632,637,642,644,645,658,660,669,673,676,677,679,685,686,689,694,695,696,698,702,707,710,711,713,719,722,725,733,734,735,738,748,749,752,755,756,758,765,766,768,771,772,782,788,792,802,803,806,810,812,814,815,822,824,829,832,833,835,840,843,849,850,853,854,856,861,864,866,872,874,879,888,890,894,895,899,902,903,907,910,911,913,915,916,918,921,922,925,927,931,933,937,940,948,951,954,955,956,959,963,964,965,966,968,969,970,971,974,975,976,979,983,991,992,993,995,999,1001,1002,1005,1008,1013,1015,1016,1021,1022,1026,1035,1042,1043,1045,1046,1047,1050,1051,1053,1060,1063,1064,1065,1066,1069,1070,1071,1072,1074,1075,1081,1083,1089,1091,1093,1094,1101,1103,1106,1112,1122,1124,1128,1129,1132,1133,1134,1136,1137,1143,1145,1149,1158,1159,1163,1164,1166,1168,1169,1175,1178,1180,1181,1185,1190,1193,1195,1196,1204,1206,1207,1219,1221,1225,1227,1230,1235,1239,1240,1242,1243,1246,1249,1250,1258,1260,1263,1264,1268,1272,1275,1276,1277,1279,1281,1289,1290,1296,1306,1309,1312,1319,1320,1329,1330,1331,1333,1334,1344,1345,1351,1354,1356,1358,1359,1665,1781]
TRAIN_SPLITS_WINDOW =[[0,1],[1,2],[2,3]]
TEST_SPLITS_WINDOW = [2,3,4]
TRAIN_SPLITS_TACK =[[0,1],[0,1,2],[0,1,2,3]]
TEST_SPLITS_TACK = [2,3,4]
CATEGORICAL = ['officer_id', 'investigator_id', 'beat', 'officer_gender','officer_race', 'rank', 'complainant_gender', 'complainant_race']
FILL_WITH_MEAN = ['agesqrd','officers_age', 'complainant_age']

#estimators
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

def temporal_split_data(df):
    df_sorted = df.sort_values(by='dateobj')
    chunks = np.array_split(df_sorted, 10)
    for chunk in chunks:
        chunk.drop('dateobj',axis=1, inplace=True)
    return chunks


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
    values = GRID[class_name]
    #Generate cross product for all given values and return them as dicts
    grids = _generate_grid(values)
    return grids


def convert_to_binary(predictions):
    print("Statistics with probability cutoff at 0.5")
    # binary predictions with some cutoff for these evaluations
    cutoff = 0.5

    predictions_binary = np.copy(predictions)
    predictions_binary[predictions_binary >= cutoff] = int(1)
    predictions_binary[predictions_binary < cutoff] = int(0)

    return predictions_binary


def transform_feature( df, column_name ):
    unique_values = set( df[column_name].tolist() )
    transformer_dict = {}
    for ii, value in enumerate(unique_values):
        transformer_dict[value] = ii

    def label_map(y):
        return transformer_dict[y]
    df[column_name] = df[column_name].apply( label_map )
    return df


if __name__ == "__main__":
    '''
    Argument Parsing
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='csv filename')
    parser.add_argument('label', help='label of dependent variable')
    parser.add_argument('to_try', nargs='+', default=['DT'], help='list of model abbreviations')
    parser.add_argument('--tack_one_on', action="store_true", default=False,help='use tack one on window validation')
    parser.add_argument('--demographic', action="store_true", default=False,help='using demographic info')
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
    print(to_try)

    '''
    Trying Models
    '''
    for model_name in to_try:
        df = pd.read_csv(args.csv)
        for col in df.columns:
            if col not in FILL_WITH_MEAN:
                df[col].fillna(0, inplace=True)
        # if model_name in ['RF','DT']:
        #     df = pd.read_csv(args.csv)
        #     for c in CATEGORICAL:
        #         df[c].fillna(-1, inplace=True)
        #         print(c)
        #         df = transform_feature(df, c)
        #         cats = list(df[c].unique())
        #         df[c]=df[c].astype('category', categories=cats)
        #     for col in df.columns:
        #         if col not in CATEGORICAL and col not in FILL_WITH_MEAN:
        #             print(col)
        #             df[col].fillna(0, inplace=True)
        # else:
        #     df = pd.read_csv(args.csv)
        #     for col in df.columns:
        #         if col not in CATEGORICAL and col not in FILL_WITH_MEAN:
        #             print(col)
        #             df[col].fillna(0, inplace=True)
        #         if col in CATEGORICAL and col not in FILL_WITH_MEAN:
        #             print(col)
        #             df[col].fillna(-1, inplace=True)
        chunks = temporal_split_data(df)

        for chunk in chunks:
            chunk.drop('Unnamed: 0',axis=1, inplace=True)
            for col in FILL_WITH_MEAN:
                chunk[col].fillna(chunk[col].mean(), inplace=True)
            # except:
            #     print("can't fill col {}".format(col))
        print("### STARTING {} ###".format(model_name))

        '''
        Iterating Through Model
        '''
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
            # print("Initialized model")
            folds_completed = 0
            print(clf)
            if model_name=='KNN':
                steps = [("normalization", preprocessing.RobustScaler()),('feat', sklearn.feature_selection.SelectKBest(k=10)),
                 (model_name, clf)]
            else:
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
                # if model_name == 'KNN':
                #     select = sklearn.feature_selection.SelectKBest(k=5)
                #     X_train = pd.DataFrame(select.fit_transform(X_train, y_train))
                #     X_test = pd.DataFrame(select.transform(X_test))
                    # print("Selected {} features".format(len(X_test.columns)))
                t0 = time.clock()
                # print(X_train.head())
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

                with open('results.csv', 'a') as csvfile:
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
                    "pickles/{0}_{1}_{2}_paramset:{3}_fold:{4}_{5}.p".format(
                    TIMESTAMP, stage, model_name, i, folds_completed, dem_label)
                    )

                data = [clf, pipeline, predicted, params] #need to fill in with things that we want to pickle
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

            with open('final_results.csv', 'a') as csvfile:
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
