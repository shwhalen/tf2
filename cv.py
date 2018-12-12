#!/usr/bin/env python

import feather
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier


def get_training(cell_line):
    training = (
        feather.read_dataframe(f'{cell_line}-training.feather')
        .set_index(['chr1', 'x1', 'x2', 'chr2', 'y1', 'y2'])
    )

    # chromhmm states
    active_chromhmm_states = {
        '1_TssA',
        '2_TssAFlnk',
        '3_TxFlnk',
        '4_Tx',
        '5_TxWk',
        '6_EnhG',
        '7_Enh'
    }
    inactive_chromhmm_states = {
        '8_ZNF/Rpts',
        '9_Het',
        '11_BivFlnk',
        '12_EnhBiv',
        '13_ReprPC',
        '14_ReprPCWk',
        '15_Quies'
    }

    # filtering on chromatin states
    f1_active_cols = [f'{state} (bin1)' for state in active_chromhmm_states]
    f2_active_cols = [f'{state} (bin2)' for state in active_chromhmm_states]
    f1_inactive_cols = [f'{state} (bin1)' for state in inactive_chromhmm_states]
    f2_inactive_cols = [f'{state} (bin2)' for state in inactive_chromhmm_states]

    training['active chromatin (bin1)'] = training[f1_active_cols].sum(axis = 1) > 0
    training['active chromatin (bin2)'] = training[f2_active_cols].sum(axis = 1) > 0
    training['inactive chromatin (bin1)'] = training[f1_inactive_cols].sum(axis = 1) > 0
    training['inactive chromatin (bin2)'] = training[f2_inactive_cols].sum(axis = 1) > 0
    training = training[
        training['active chromatin (bin1)'].values &
        training['active chromatin (bin2)'].values
    ]

    # subsampling for desired class balance
    n_negatives = training.eval('label == 0').sum()
    n_positives = training.eval('label == 1').sum()
    if n_negatives > n_positives * negatives_per_positive:
        pos_training = training.query('label == 1')
        neg_training = (
            training
            .query('label == 0')
            .sample(
                n_positives * negatives_per_positive,
                random_state = 0
            )
        )
        training = pd.concat([pos_training, neg_training])
        training.sort_index(inplace = True)

    return (
        cell_line,
        training.drop('label', axis = 1).astype(float),
        training['label'],
        training.index.get_level_values('chr1')
    )


n_jobs = -1

# training data
negatives_per_positive = 10
datasets = [
    get_training('HeLa-S3'),
    get_training('K562'),
    get_training('IMR-90'),
    get_training('GM12878')
]

# classifiers
baseline = make_pipeline(
    DummyClassifier(random_state = 0)
)
elastic_net = make_pipeline(
    StandardScaler(),
    SGDClassifier(
        loss = 'log',
        penalty = 'elasticnet',
        alpha = 0.01,
        max_iter = 1000,
        tol = 1e-3,
        random_state = 0
    )
)
extra_trees = make_pipeline(
    ExtraTreesClassifier(
        n_estimators = 1000,
        max_features = 'log2',
        random_state = 0
    )
)
xgb = make_pipeline(
    XGBClassifier(
        booster = 'gbtree',
        n_estimators = 100,
        max_depth = 4,
        learning_rate = 0.1,
        subsample = 1,
        colsample_bytree = 0.5,
        gamma = 10,
        scoring = 'map',
        silent = True,
        random_state = 0
    )
)
estimators = [baseline, elastic_net, extra_trees, xgb]

# cross validation
cv = GroupKFold(n_splits = 10)
all_scores = []

for estimator in tqdm(estimators, 'estimator'):
    for (name, features, labels, chroms) in tqdm(datasets, 'dataset'):
        scores = cross_val_score(
            estimator,
            features,
            labels,
            groups = chroms,
            scoring = 'average_precision',
            cv = cv,
            n_jobs = n_jobs
        )
        all_scores.append(
            [estimator.steps[-1][0], name, np.mean(scores)]
        )

all_scores = pd.DataFrame(
    all_scores,
    columns = ['classifier', 'cell line', 'mean auPR']
)
print(f'\n{all_scores}')
