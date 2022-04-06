r"""
This file defines the method required to training the regression models for forecasting the number of chats in each day.
"""
import datetime
from typing import Type, Callable, Dict, Optional, List, AnyStr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from macorp.forecast.utils import extract_features, extract_features_from_series


def chat_train(df: Type[pd.DataFrame], pool_of_regressors: List[object],
               regressor_params: Optional[List[Dict]],
               scoring: AnyStr ='neg_mean_absolute_error',
               visualize: bool = False,
               verbosity: int = 0) -> tuple:
    r"""
    Method to train the regressor for predicting the number of chats per day.

    :param df: a dataframe indexed by date (str 'YYYY-MM-DD') and the following columns: "eligible_users, chats".
    :param pool_of_regressors: pool of regression model classes. Each class must have fit and predict methods.
    :param regressor_params: parameters corresponding to each of regressors in the pool_of_regressors.
    :param scoring: validation/model selection scoring function. For more regression scoring options look at https://scikit-learn.org/stable/modules/model_evaluation.html
    :param visualize: if true, a train-val is applied before deployment and the results for val is visualized.
    :param verbosity: the level of verbosity to help with monitoring the progress.
    :return: a callable that predict number of chats
    """
    if verbosity > 0:
        print('validating the training data ...')
    assert 'eligible_users' in df
    assert 'chats' in df
    assert (~df['eligible_users'].isnull()).all()
    assert (~df['chats'].isnull()).all()

    if verbosity > 0:
        print('extracting features ...')
    # dissolve time into features that can better leverage from the temporal localities in the data
    date_index = pd.to_datetime(df.index).date
    extracted_features = extract_features_from_series(date_index)
    for key, item in extracted_features.items():
        df[key] = item
    feature_cols = list(extracted_features.keys())
    feature_cols += ['eligible_users']

    if verbosity > 0:
        print('choosing the best model ...')

    if regressor_params is None:
        regressor_params = [{}] * len(pool_of_regressors)

    chosen_params = [None] * len(pool_of_regressors)

    # find the best model from the pool
    best_score = None
    best_score_idx = None
    for i, estimator_cls in enumerate(pool_of_regressors):
        estimator = estimator_cls()
        # search parameter space
        if regressor_params[i] is not None and len(regressor_params[i]) > 0:
            regr = GridSearchCV(estimator, regressor_params[i], cv=5, scoring=scoring)
            regr.fit(df[feature_cols].values, df['chats'].values)
            chosen_params[i] = regr.best_params_
            # TODO: the code needs to be optimized here
            estimator = estimator_cls(**regr.best_params_)
        else:
            chosen_params[i] = dict()

        scores = cross_val_score(estimator,
                                 df[feature_cols].values,
                                 df['chats'].values,
                                 cv=5,
                                 scoring=scoring)
        mean_score = sum(scores) / len(scores)
        if best_score is None or mean_score > best_score:
            best_score = mean_score
            best_score_idx = i

    if verbosity > 0:
        print('training the chosen model ...')

    if visualize and len(df) > 10:
        fig, ax = plt.subplots()
        if verbosity > 0:
            print('visualizing the results ...')
        estimator = pool_of_regressors[best_score_idx](**chosen_params[best_score_idx])
        df_train = df.iloc[:4 * len(df) // 5, :]
        df_val = df.iloc[4 * len(df) // 5:, :]
        estimator.fit(df_train.loc[:, feature_cols].values, df_train['chats'].values)
        pred = estimator.predict(df_val.loc[:, feature_cols].values)
        ax.plot(range(len(pred)), df_val['chats'].values, label='true values')
        ax.plot(range(len(pred)), pred, label='prediction')
        ax.set_xlabel('day offset')
        ax.set_ylabel('chats')
        mae = np.absolute(pred - df_val['chats'].values).mean()
        plt.plot([], [], ' ', label="MAE={:.2f}".format(mae))
        ax.legend()
    # train the best model with the whole data
    estimator = pool_of_regressors[best_score_idx](**chosen_params[best_score_idx])
    estimator.fit(df[feature_cols].values, df['chats'].values)
    if visualize:
        return estimator, best_score_idx, best_score, fig, ax
    return estimator, best_score_idx, best_score
