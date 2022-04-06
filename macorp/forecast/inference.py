r"""
This files contains inference methods used in post deployment stage.
"""

import datetime
from typing import Type, Callable, List
import pandas as pd
import numpy as np

from macorp.forecast.utils import extract_features, extract_features_from_series


def chat_inference(date: datetime.date, eligible_users: int, regressor: Callable) -> int:
    r"""
    forecasts and returns the number of chats for a specific date.

    :param date: an instance of datetime.date
    :param eligible_users: number of users connected to the platform.
    :param regressor: the pre-trained regression model used for forecasting.
    :return: number of forecasted chats.
    """
    features = extract_features(date)
    chats = regressor.predict([list(features.values()) + [eligible_users]])
    return round(chats[0])


# no transduction is assumed
def chat_inference_batch(eligible_series: Type[pd.Series], regressor: Callable) -> List[int]:
    """
    forecasts and returns the number of chats for a batch of dates.

    :param eligible_series: a pandas.Series representing the number of users connected to the platform at each date
    :param regressor: the pretrained model for forecasting the number of chats.
    :return: a list of chat forecasts each corresponding to a date entry in the provided series.
    """
    date_index = pd.to_datetime(eligible_series.index).date
    features = extract_features_from_series(date_index)
    features_np = np.concatenate(
        [np.array(list(features.values())), np.expand_dims(eligible_series.values, axis=0)]).transpose()
    chats = regressor.predict(features_np)
    return list(map(lambda x: round(x), chats))
