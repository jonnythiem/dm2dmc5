import argparse
import logging

import yaml
import pandas as pd
import numpy as np
import featuretools as ft
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score, roc_curve

models = {
    'LogisticRegression': {
        'model': LogisticRegression(),
        'params': None
    },
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': 300
        },
        'param_grid': {
            'n_estimators': range(100, 301, 100)
        }
    }
}


def manual_feature_generation(df: pd.DataFrame):
    """
    This function computes all the manually generated features from the baseline notebook.
    :param df: 
    :return: 
    """
    df['totalScanned'] = df['scannedLineItemsPerSecond'] * df['totalScanTimeInSeconds']
    df['avgTimePerScan'] = 1 / df['scannedLineItemsPerSecond']
    df['avgValuePerScan'] = df['avgTimePerScan'] * df['valuePerSecond']
    df['withoutRegisPerPosition'] = df['scansWithoutRegistration'] / df['totalScanned']
    df['quantiModPerPosition'] = df['quantityModifications'] / df['totalScanned']
    df['lineItemVoidsPerTotal'] = df['lineItemVoids'] / df['grandTotal']
    df['withoutRegisPerTotal'] = df['scansWithoutRegistration'] / df['grandTotal']
    df['quantiModPerTotal'] = df['quantityModifications'] / df['grandTotal']
    df['lineItemVoidsPerTime'] = df['lineItemVoids'] / df['totalScanTimeInSeconds']
    df['withoutRegisPerTime'] = df['scansWithoutRegistration'] / df['totalScanTimeInSeconds']
    df['quantiModPerTime'] = df['quantityModifications'] / df['totalScanTimeInSeconds']
    return df


def profit_scorer(y, y_pred):
    profit_matrix = {(0, 0): 0, (0, 1): -5, (1, 0): -25, (1, 1): 5}
    return sum(profit_matrix[(pred, actual)] for pred, actual in zip(y_pred, y))


profit_scoring = make_scorer(profit_scorer, greater_is_better=True)

logger = logging.getLogger('Pipeline')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':

    config = yaml.safe_load(open('pipeline_config.yml', 'r'))
    df = pd.read_csv('train.csv', sep='|')
    X, y = df.drop(columns='fraud'), df['fraud']

    switches = config['switches']
    options = config['dataset']
    sampling = switches['sampling']
    feature_eng = switches['feature_engineering']

    logging.info(
        f'Running with configuration: \n'
        f'{config}'
    )
    # has to be only computed once and does not leak train information
    if feature_eng['manual']:
        X = manual_feature_generation(df=X)

    if switches['split'] == 'bagging':
        partitioner = StratifiedShuffleSplit(n_splits=switches['partitions'], train_size=0.7)
    elif switches['split'] == 'cv':
        partitioner = StratifiedKFold(n_splits=switches['partitions'])
    else:
        logging.error('No valid partitioning scheme selected. Possible types are: [bagging, cv].')
        raise ValueError

    result_container = []

    for i, (train, test) in enumerate(partitioner.split(X=X, y=y)):
        logging.info(f'Running Iteration {i + 1} of {switches["partitions"]}')
        X_train, X_test, y_train, y_test = X.iloc[train, :], X.iloc[test, :], y.iloc[train], y.iloc[test]

        if switches['preprocessing']:
            scaler = StandardScaler()
            X_train.loc[:, :] = scaler.fit_transform(X=X_train)
            X_test.loc[:, :] = scaler.transform(X=X_test)

        if sampling:
            if sampling['method'] == 'under':
                sampler = RandomUnderSampler(sampling_strategy=sampling['pos_to_neg_ratio'])
            elif sampling['method'] == 'over':
                sampler = RandomOverSampler(sampling_strategy=sampling['pos_to_neg_ratio'])
            elif sampling['method'] == 'smote':
                sampler = SMOTE(sampling_strategy=sampling['pos_to_neg_ratio'])
            else:
                logging.error('No valid selection for sampling scheme. Possible values are: [under, over, smote].')
                raise ValueError

            X_train, y_train = sampler.fit_resample(X=X_train, y=y_train)

        if feature_eng['featuretools']:
            pass

        if switches['hyperparameter_tuning']:
            pass

        for model in models:
            logging.info(f'Currently computing {model}')
            clf = models[model]['model']
            clf.fit(X=X_train, y=y_train)
            probs = clf.predict_proba(X=X_test)
            preds = np.int(probs > options['cutoff'])
            result_container.append(
                {
                    'iteration': i,
                    'model': model,
                    'params': ' '.join(clf.get_params()),
                    'probabilities': probs,
                    'accuracy': accuracy_score(y_pred=preds, y_true=y_test),
                    'precision': precision_score(y_pred=preds, y_true=y_test),
                    'recall': recall_score(y_pred=preds, y_true=y_test),
                    'f1': f1_score(y_pred=preds, y_true=y_test),
                    'profit': profit_scorer(y_pred=preds, y=y_test),
                    'auc': roc_auc_score(y_score=probs, y_true=y_test),
                    'roc': roc_curve(y_score=probs, y_true=y_test)
                }
            )
            logging.info(f'{model} completed. \n'
                         f'acc: {result_container[-1]["accuracy"]} \n'
                         f'precision: {result_container[-1]["precision"]} \n'
                         f'recall: {result_container[-1]["recall"]}\n'
                         f'f1: {result_container[-1]["f1"]}')

    result_df = pd.DataFrame(result_container)
    result_sum = (result_df
        .drop(['iteration', 'roc', 'params', 'probabilties'], axis=1)
        .groupby('model')
        .agg(['mean', 'std', 'quantile'], q=[0, 0.25, 0.5, 0.75, 1])
    )
    logging.info(f'Model results:\n'
                 f'{result_sum}')

    result_df.to_csv('result_df', index=False)
    result_sum.to_csv('result_sum', index=False)
