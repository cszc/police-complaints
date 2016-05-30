#!/usr/bin/env python
import pandas as pd
import datetime
import yaml
import os
import logging
import logging.config
import copy
import numpy as np
# from lib_cinci import dataset
import evaluation
# from features import feature_parser
import argparse
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn_evaluation.Logger import Logger
from sklearn_evaluation.metrics import precision_at
from grid_generator import grid_from_class

import pipeline as pl


logging.config.dictConfig(load('logger_config.yaml'))
logger = logging.getLogger()

# pths to files?

def make_datasets(config):
    pass
    # return test, train

def output_evaluation_statistics(test, predictions):
    logger.info("Statistics with probability cutoff at 0.5")
    # binary predictions with some cutoff for these evaluations
    cutoff = 0.5
    predictions_binary = np.copy(predictions)
    predictions_binary[predictions_binary >= cutoff] = 1
    predictions_binary[predictions_binary < cutoff] = 0

    evaluation.print_model_statistics(y_test, predictions_binary)
    evaluation.print_confusion_matrix(y_test, predictions_binary)

    precision1 = precision_at(y_test, predictions, 0.01)
    logger.debug("Precision at 1%: {} (probability cutoff {})".format(
                 round(precision1[0], 2), precision1[1]))
    precision10 = precision_at(y_test, predictions, 0.1)
    logger.debug("Precision at 10%: {} (probability cutoff {})".format(
                 round(precision10[0], 2), precision10[1]))


def get_feature_importances(model):
    try:
        return model.feature_importances_
    except:
        pass
    try:
        logging.info(('This model does not have feature_importances, '
                      'returning .coef_[0] instead.'))
        return model.coef_[0]
    except:
        logging.info(('This model does not have feature_importances, '
                      'nor coef_ returning None'))
    return None

# # def log_results(model, config, test, predictions, feature_importances,
#                 imputer, scaler):    
def binarize(x):
    if x == 'sustained':
        return 1
    else:
        return 0

def main(f):
    outcome_var='final_outcome'
    models_selected=["sklearn.ensemble.RandomForestClassifier"]
    df['final_outcome'] = df['final_outcome_class'].apply(binarize)
    # cols = list(df.columns)
    # features = [x for x in cols if x != OUTCOME_VAR]
    features=['recc_outcome', 'investigator_id', 'officer_id', 'beat']
    df = pd.read_csv(f)
    logger.info('Loading datasets...')
    X_train, X_test, y_train, y_test = pl.split_train_test(df, features, OUTCOME_VAR, .2)
    logger.debug('Train x shape: {} Test x shape: {}'.format(X_train.shape,
        X_test.shape))

    #fill/impute/scale hhere


    grids = [grid_from_class(m, size=grid_size) for m in models_selected]
    models = [a_grid for a_model_grid in grids for a_grid in a_model_grid]


    # fit each model for all of these
    for idx, model in enumerate(models):
        #Try to run in parallel if possible
        if hasattr(model, 'n_jobs'):
            model.set_params(n_jobs=args.n_jobs)

        #SVC does not predict probabilities by default
        if hasattr(model, 'probability'):
            model.probability = True

        timestamp = datetime.datetime.now().isoformat()

        # train
        logger.info("{} out of {} - Training {}".format(idx+1,
                                                        len(models),
                                                        model))
        model.fit(X_train, y_train)

        # predict
        logger.info("Predicting on validation samples...")
        predicted = model.predict_proba(X_test)
        predicted = predicted[:, 1]  # probability that label is 1

        # statistics
        output_evaluation_statistics(test, predicted)
        feature_importances = get_feature_importances(model)


        # save results
        print model.get_params()

if __name__=="__main__":
    f = sys.argv[1]
    main(f)
