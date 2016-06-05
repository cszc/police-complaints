#Libraries from scikit-learn
from sklearn import preprocessing, cross_validation, metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
# from sklearn_evaluation.metrics import precision_at
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
matplotlib.use('Agg')
import pylab as pl
from scipy import optimize
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
from pydoc import locate
from argparse import ArgumentParser
from unbalanced_dataset.over_sampling import RandomOverSampler
from unbalanced_dataset.over_sampling import SMOTE
from unbalanced_dataset.combine import SMOTETomek
from unbalanced_dataset.combine import SMOTEENN

#Own modules
# import evaluation
# from utils import *

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
TO_DROP = ['crid', 'officer_id', 'agesqrd', 'index', 'Unnamed: 0', 'physical:False', 'physical:True', 'physical:nan']
FILL_WITH_MEAN = ['officers_age', 'transit_time', 'car_time']

#estimators
CLFS = {
        'GB': GradientBoostingClassifier(
                    learning_rate=0.05,
                    subsample=0.5,
                    max_depth=6,
                    n_estimators=10),
        'test': DecisionTreeClassifier(),

        }

GRID={'GB': {'learning_rate': [0.1], 'subsample': [0.1], 'max_depth': [5], 'n_estimators': [100]},
    'test': {
                'criterion': ['gini'],
                'max_depth': [1,5],
                'max_features': ['sqrt'],
                'min_samples_split': [5,10]
                }}

SVM_ARGS={'class_weight': 'auto'}
OVER_SAMPLERS = {'ROS': RandomOverSampler(ratio='auto', verbose=False),
                # 'SMOTE': SMOTE(ratio='auto', verbose=False, kind='regular'),
                # 'None': None
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

if __name__ == "__main__":
    '''
    Argument Parsing
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='csv filename')
    parser.add_argument('label', help='label of dependent variable')
    parser.add_argument('to_try', nargs='+', default=['DT'], help='list of model abbreviations')
    parser.add_argument('--pared_down', action="store_true", default=False, help='use tack one on window validation')
    parser.add_argument('--tack_one_on', action="store_true", default=False, help='use tack one on window validation')
    parser.add_argument('--demographic', action="store_true", default=False, help='using demographic info')
    args = parser.parse_args()

    '''
    Labels for Documenting
    '''
    label = args.label #y, predicted variable
    dem_label = 'with_demo' if args.demographic else 'non-demo'
    stage = 'stage-1'
    if args.pared_down:
        data_label = "pared_"+dem_label
    else:
        data_label = dem_label



    train_split, test_split = TRAIN_SPLITS_TACK, TEST_SPLITS_TACK

    to_try = args.to_try
    df = pd.read_csv(args.csv)

    if args.demographic:
        FILL_WITH_MEAN.append('complainant_age')
    '''
    Generate decision tree bases
    '''
    # dt_grid = grid_from_class('DT')
    # base_dts = []
    # for i, params in enumerate(dt_grid):
    #     dt = CLFS['DT']
    #     dt.set_params(**params)
    #     base_dts.append(dt)
    # GRID['AB']['base_estimator'] = base_dts
    
    '''
    Trying Models
    '''
    for model_name in to_try:
        print(model_name)
        '''
        Processing and Imputation
        '''
        this_df = df
        for col in TO_DROP:
            try:
                this_df.drop(col, axis=1, inplace = True)
            except:
                continue

        categories = [col for col in this_df.columns if 'Category' in col]
        for cat in categories:
            this_df.drop(cat, axis=1, inplace = True)

        if args.pared_down:
            beats = [col for col in this_df.columns if 'beat' in col]
            for beat in beats:
                this_df.drop(beat, axis=1, inplace = True)

            counts_311 = [col for col in this_df.columns if '_count_' in col]
            for count in counts_311:
                this_df.drop(count, axis=1, inplace = True)

            crimes = [col for col in this_df.columns if 'crime' in col]
            for crime in crimes:
                this_df.drop(crime, axis=1, inplace = True)

        chunks = temporal_split_data(this_df)

        #IMPUTE EACH TEMPORAL CHUNK
        # for chunk in chunks:
        #     for col in chunk.columns:
        #         try:
        #             if col in FILL_WITH_MEAN:
        #                 mean = round(chunk[col].mean())
        #                 chunk[col].fillna(value=mean, inplace=True)
        #             else:
        #                 chunk[col].fillna(0, inplace=True)
        #         except:
        #             print('Could not impute column:{}'.format(col))
        #             continue

        '''
        Iterating Through Model
        '''
        print("### STARTING {} ###".format(model_name))
        # clf = CLFS[model_name]
        grid = grid_from_class(model_name)
        print(grid)

        for i, params in enumerate(grid):
            '''
            Iterating Through Parameters
            '''
            print("### TRYING PARAMS ###")
            # clf.set_params(**params)
            # if hasattr(clf, 'n_jobs'):
            #         clf.set_params(n_jobs=-1)
            #
            # folds_completed = 0
            # print(clf)
            #
            # steps = [(model_name, clf)]
            # pipeline = sklearn.pipeline.Pipeline(steps)
            # print("pipeline:", [name for name, _ in pipeline.steps])
            #
            overall_accuracy = []
            overall_auc_scores = []
            for over_sampler in OVER_SAMPLERS.keys():
                trial = 0
                while trial < 5:
                    trial += 1
                    print("### STARTING TRIAL {} ###".format(trial))

                    clf = CLFS[model_name]
                    clf.set_params(**params)
                    if hasattr(clf, 'n_jobs'):
                            clf.set_params(n_jobs=-1)

                    print(clf)
                    print(over_sampler)
                    print("######")
                    steps = [(model_name, clf)]

                    pipeline = sklearn.pipeline.Pipeline(steps)
                    print("pipeline:", [name for name, _ in pipeline.steps])

                    accuracy_scores =[]
                    auc_scores = []
                    print("Resetting AUC")
                    folds_completed = 0

                    os_object = OVER_SAMPLERS[over_sampler]
                    print("Oversampler: {}".format(over_sampler))

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
                        
                        #IMPUTE VALUES
                        if args.demographic:
                            X_train.loc[X_train['complainant_age']>100,'complainant_age'] = np.nan
                            X_test.loc[X_test['complainant_age']>100,'complainant_age'] = np.nan
                        for col in X_train.columns:
                            try:
                                if col in FILL_WITH_MEAN:
                                    mean = round(X_train[col].mean())
                                    X_train[col].fillna(value=mean, inplace=True)
                                    X_test[col].fillna(value=mean, inplace=True)
                                else:
                                    X_train[col].fillna(0, inplace=True)
                                    X_test[col].fillna(0, inplace=True)

                            except:
                                print('Could not impute column:{}'.format(col))
                                continue
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
                        # output_evaluation_statistics(y_test, predicted_prob)

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

                        print()
                        auc_score = metrics.roc_auc_score(y_test, predicted)
                        accuracy = metrics.accuracy_score(y_test, predicted)
                        precision, recall, thresholds = metrics.precision_recall_curve(
                            y_test, predicted_prob)
                        auc_scores.append(auc_score)
                        accuracy_scores.append(accuracy)
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
                        print("Completed {} fold".format(folds_completed))

                        with open('Evaluation/stage1/individual_results.csv', 'a') as csvfile:
                            spamwriter = csv.writer(csvfile)
                            spamwriter.writerow(
                                [dem_label,
                                data_label,
                                label,
                                model_name,
                                params,
                                over_sampler,
                                folds_completed,
                                auc_score,
                                precision,
                                recall,
                                thresholds,
                                accuracy])

                        file_name = (
                            "Evaluation/stage1/pickles/{0}_{1}_{2}_paramset:{3}_fold:{4}_{5}.p".format(
                            TIMESTAMP, data_label, model_name, i, folds_completed, dem_label)
                            )

                        data = [clf, pipeline, predicted_prob, params]
                        pickle.dump( data, open(file_name, "wb" ) )

                    print("### Cross Validation Statistics ###")
                    print(clf)
                    print(over_sampler)
                    precision, recall, thresholds = metrics.precision_recall_curve(
                        overall_actual, overall_predictions)
                    average_auc = sum(auc_scores) / len(auc_scores)
                    overall_auc_scores.append(average_auc)
                    average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
                    overall_accuracy.append(average_accuracy)
                    print("Average AUC: %0.3f" % average_auc)
                    print("Confusion Matrix")
                    overall_binary_predictions = convert_to_binary(overall_predictions)
                    confusion = metrics.confusion_matrix(
                        overall_actual, overall_binary_predictions)
                    print(confusion)

                    #Plot precision recall
                    if trial == 1:
                        overall_precision, overall_recall, overall_thresholds = metrics.precision_recall_curve(
                                overall_actual, overall_predictions)
                        plot_file = "Evaluation/stage1/plots/aPR_{0}_{1}_{2}_{3}.png".format(
                                TIMESTAMP, data_label, model_name, label)
                        plt.plot(overall_recall, overall_precision)
                        plt.title('Precision-Recall curve for {}'.format(model_name+" "+data_label+" "+label))
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                        plt.grid(True)
                        plt.savefig(plot_file)
                        plt.close()

                        #plot AUC
                        plot_file = "Evaluation/stage1/plots/aROC_{0}_{1}_{2}_{3}.png".format(TIMESTAMP, data_label, model_name, label)
                        overall_fpr, overall_tpr, overall_roc_thresholds = metrics.roc_curve(
                                overall_actual, overall_predictions)
                        plt.plot(overall_fpr, overall_tpr)
                        plt.plot([0, 1], [0, 1], 'k--')
                        plt.title('ROC curve for {} (AUC = {})'.format(model_name+" "+data_label+" "+label, round(average_auc,3)))
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.grid(True)
                        plt.savefig(plot_file)
                        plt.close()

                    with open('Evaluation/stage1/trial_results.csv', 'a') as csvfile:
                        spamwriter = csv.writer(csvfile)
                        spamwriter.writerow([
                            data_label,
                            dem_label,
                            label,
                            model_name,
                            params,
                            over_sampler,
                            folds_completed,
                            precision,
                            recall,
                            thresholds,
                            average_auc,
                            average_accuracy,
                            confusion])
            
            print("####Completed trials###")
            avg_auc_overall = sum(overall_auc_scores)/len(overall_auc_scores)
            avg_acc_overall = sum(overall_accuracy)/len(overall_accuracy)
            print("Overall AUC score: %0.3f" % avg_auc_overall)

            with open('Evaluation/stage1/full_results.csv', 'a') as csvfile:
                    spamwriter = csv.writer(csvfile)
                    spamwriter.writerow([
                        data_label,
                        dem_label,
                        label,
                        model_name,
                        params,
                        over_sampler,
                        avg_auc_overall,
                        avg_acc_overall])