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
# from sklearn_evaluation.metrics import precision_at
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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from pydoc import locate
from argparse import ArgumentParser
from unbalanced_dataset.over_sampling import RandomOverSampler
from unbalanced_dataset.over_sampling import SMOTE
from unbalanced_dataset.combine import SMOTETomek
from unbalanced_dataset.combine import SMOTEENN


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
# CATEGORICAL = ['officer_id', 'investigator_id', 'beat', 'officer_gender','officer_race', 'rank', 'complainant_gender', 'complainant_race']
# FILL_WITH_MEAN = ['agesqrd','officers_age', 'complainant_age']
TO_DROP = ['crid', 'officer_id', 'agesqrd', 'index', 'Unnamed: 0', 'Category nan','physical:nan']
FILL_WITH_MEAN = ['officers_age', 'transit_time', 'car_time','agesqrd','complainant_age']

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
        'AB': {
                'algorithm': ['SAMME', 'SAMME.R'],
                'n_estimators': [10,100,500,1000],
                'base_estimator':[],
                'learning_rate' : [0.001,0.01,0.1]
                },
        'DT': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5,10,50],
                'max_features': ['sqrt','log2'],
                'min_samples_split': [10,50,100,200],
                'class_weight': ['balanced', None]
                }
       }

SVM_ARGS={'class_weight': 'auto'}
OVER_SAMPLERS = {'ROS': RandomOverSampler(ratio='auto', verbose=False),
                'SMOTE': SMOTE(ratio='auto', verbose=False, kind='regular'),
                # 'SMOTE borderline 1': SMOTE(ratio='auto', verbose=False, kind='borderline1'),
                # 'SMOTE borderline 2': SMOTE(ratio='auto', verbose=False, kind='borderline2'),
                # # 'SMOTE SVM': SMOTE(ratio='auto', verbose=False, kind='svm', **SVM_ARGS),
                'SMOTETomek': SMOTETomek(ratio='auto', verbose=False),
                # 'SMOTEEEN': SMOTEENN(ratio='auto', verbose=False),
                # 'None': None
                }

def temporal_split_data(df):
    df_sorted = df.sort_values(by='dateobj')
    chunks = np.array_split(df_sorted, 5)
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


if __name__ == "__main__":
    '''
    Argument Parsing
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='csv filename')
    parser.add_argument('label', help='label of dependent variable')
    parser.add_argument('to_try', nargs='+', default=['DT'], help='list of model abbreviations')
    # parser.add_argument('-t', action="store_true", default=False,help='use tack one on window validation')
    parser.add_argument('-d', action="store_true", default=False,help='using demographic info')
    args = parser.parse_args()
    print(args)
    '''
    Labels for Documenting
    '''
    label = args.label #y, predicted variable
    train_split, test_split = TRAIN_SPLITS_TACK, TEST_SPLITS_TACK
    # if args.t:
    #     train_split, test_split = TRAIN_SPLITS_TACK, TEST_SPLITS_TACK
    # else:
    #     train_split, test_split = TRAIN_SPLITS_WINDOW, TEST_SPLITS_WINDOW
    if args.d:
        dem_label = 'with_demo'
    else:
        dem_label = 'non-demo'
    stage = 'stage-1'
    to_try = args.to_try
    # if '-t' in to_try:
    #     to_try = to_try.remove('-t')
    print(to_try)
    '''
    Generate decision tree bases
    '''
    dt_grid = grid_from_class('DT')
    base_dts = []
    for i, params in enumerate(dt_grid):
        dt = CLFS['DT']
        dt.set_params(**params)
        base_dts.append(dt)
    GRID['AB']['base_estimator'] = base_dts

    '''
    Trying Models
    '''
    for model_name in to_try:
        df = pd.read_csv(args.csv)

        for col in TO_DROP:
            try:
                df.drop(col, axis=1, inplace = True)
            except:
                continue

        chunks = temporal_split_data(df)

        #IMPUTE EACH TEMPORAL CHUNK
        for chunk in chunks:
            for col in chunk.columns:
                try:
                    if col in FILL_WITH_MEAN:
                        mean = round(chunk[col].mean())
                        chunk[col].fillna(value=mean, inplace=True)
                    else:
                        chunk[col].fillna(0, inplace=True)
                except:
                    print('Could not impute column:{}'.format(col))
                    continue

        print("### STARTING {} ###".format(model_name))

        '''
        Iterating Through Model
        '''
        grid = grid_from_class(model_name)

        for i, params in enumerate(grid):
            '''
            Iterating Through Parameters
            '''

            for over_sampler in OVER_SAMPLERS.keys():
                print()
                print("#################")
                print("### STARTING ###")
                print("#################")
                
                clf = CLFS[model_name]
                clf.set_params(**params)
                if hasattr(clf, 'n_jobs'):
                        clf.set_params(n_jobs=-1)

                if model_name=='KNN':
                    steps = [("normalization", preprocessing.RobustScaler()),('feat', sklearn.feature_selection.SelectKBest(k=10)),
                     (model_name, clf)]
                else:
                    steps = [(model_name, clf)]

                pipeline = sklearn.pipeline.Pipeline(steps)
                print("pipeline:", [name for name, _ in pipeline.steps])

                auc_scores = []
                folds_completed = 0

                os_object = OVER_SAMPLERS[over_sampler]
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

                    if over_sampler != "None":
                        X_resampled = np.array(X_train)
                        y_resampled = np.array(y_train)
                        X_resampled, y_resampled = os_object.fit_transform(X_resampled,y_resampled)

                    else:
                        X_resampled = X_train
                        y_resampled = y_train


                    t0 = time.clock()
                    pipeline.fit(X_resampled, y_resampled)
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

                    '''
                    Evaluation Statistics
                    '''
                    print()
                    print("Evaluation Statistics")
                    if model_name=='KNN':
                        print("Getting feature support")
                        features = pipeline.named_steps['feat']
                        print(X_train.columns[features.transform(np.arange(
                            len(X_train.columns)))])


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

                        print(df_best_estimators.head(5))

                    folds_completed += 1
                    print()
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
                print(clf)
                print(over_sampler)
                precision, recall, thresholds = metrics.precision_recall_curve(
                    overall_actual, overall_predictions)
                print("Length of AUC SCORES: {}".format(len(auc_scores)))
                average_auc = sum(auc_scores) / len(auc_scores)
                print("Average AUC: %0.3f" % average_auc)
                print("Confusion Matrix")
                overall_binary_predictions = convert_to_binary(overall_predictions)
                confusion = metrics.confusion_matrix(
                    overall_actual, overall_binary_predictions)
                print(confusion)

                with open('Results/final_results.csv', 'a') as csvfile:
                    spamwriter = csv.writer(csvfile)
                    spamwriter.writerow([
                        stage,
                        dem_label,
                        label,
                        model_name, 
                        params,
                        over_sampler,
                        folds_completed,
                        precision,
                        recall,
                        thresholds,
                        average_auc])