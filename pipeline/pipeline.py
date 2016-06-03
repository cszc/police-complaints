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
label = "no_affidavit" #predicted variable goes here
TRAIN_SPLITS =[[0,1],[1,2],[2,3]]
TEST_SPLITS = [2,3,4]
CATEGORICAL = ['officer_id', 'investigator_id', 'beat', 'officer_gender','officer_race', 'rank', 'complainant_gender', 'complainant_race']
FILL_WITH_MEAN = ['agesqrd','officers_age', 'complainant_age']
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
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'test': DecisionTreeClassifier(),
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
    chunks = np.array_split(df_sorted, 5)
    for chunk in chunks:
        chunk.drop('dateobj',axis=1, inplace=True)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='csv filename')
    # parser.add_argument('label', help='label of dependent variable')
    parser.add_argument('--to_try', default=[], dest='to_try', help='list of model abbreviations', action='append')
    args = parser.parse_args()

    # label = args.label #predicted variable goes here
    to_try = args.to_try
    print(to_try)


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
        clf = clfs[model_name]
        grid = grid_from_class(model_name)
        for params in grid:
            print("### TRYING PARAMS ###")
            clf.set_params(**params)
            if hasattr(clf, 'n_jobs'):
                    clf.set_params(n_jobs=-1)
            # print("Initialized model")
            folds_completed = 0
            print(clf)
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
                X_test = test_df.drop(label, axis=1)
                y_test = test_df[label]

                t0 = time.clock()
                pipeline.fit(X_train,y_train)
                print("done fitting in %0.3fs" % (time.clock() - t0))

                predicted = pipeline.predict(X_test)

                try:
                    predicted_prob = pipeline.predict_proba(X_test)
                    predicted_prob = predicted_prob[:, 1]  # probability that label is 1

                except:
                    print("Model has no predict_proba method")
                # print("Evaluation Statistics")
                # output_evaluation_statistics(y_test, predicted_prob)
                print()
                auc_score = metrics.roc_auc_score(y_test, predicted)
                precision, recall, thresholds = metrics.precision_recall_curve(y_test, predicted_prob)
                auc_scores.append(auc_score)
                print("AUC score: %0.3f" % auc_score)

                # save predcited/actual to calculate precision/recall
                if folds_completed==0:
                    overall_predictions = pd.DataFrame(predicted_prob)
                else:
                    overall_predictions = pd.concat([overall_predictions, pd.DataFrame(predicted_prob)])
                if folds_completed==0:
                    overall_actual = y_test
                else:
                    overall_actual = pd.concat([overall_actual, y_test])

                print("Feature Importance")
                feature_importances = get_feature_importances(pipeline.named_steps[model_name])
                if feature_importances != None:
                    df_best_estimators = pd.DataFrame(feature_importances, columns = ["Imp/Coef"], index = X_train.columns).sort_values(by='Imp/Coef', ascending = False)
                    # filename = "plots/ftrs_"+estimator_name+"_"+time.strftime("%d-%m-%Y-%H-%M-%S"+".png")
                    # plot_feature_importances(df_best_estimators, filename)
                    print(df_best_estimators.head(5))
                folds_completed += 1
            print("### Cross Validation Statistics ###")
            precision, recall, thresholds = metrics.precision_recall_curve(overall_actual, overall_predictions)
            average_auc = sum(auc_scores)/len(auc_scores)
            print("Average AUC: %0.3f" % auc_score)
            print("Confusion Matrix")
            overall_binary_predictions = convert_to_binary(overall_predictions)
            confusion = metrics.confusion_matrix(overall_actual, overall_binary_predictions)
            print(confusion)

        # file_name = "pickles/{0}_{1}.p}".format(TIMESTAMP, estimator)
        # data = [] #need to fill in with things that we want to pickle
        # pickle.dump( data, open(file_name, "wb" ) )
