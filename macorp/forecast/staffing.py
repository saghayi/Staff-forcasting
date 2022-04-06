r"""
Since staffing is being determined for providing healthcare, it seems better for staff not to multiplex they task.
Therefore, we assume the full concentration of the nurse when connected with a patient whether the medium is text-based,
audio-based or visual.
Therefore, we use Erlang C formula to determine the number of staff (nurses) needed at each day at MaCorp
communication centre.
Erlang C makes a few assumptions in order to come up with an accurate recommendation
(based on https://www.techtarget.com/searchunifiedcommunications/definition/Erlang-C):
 - The order of customer requests follows a Poisson Distribution
 - Service times are exponentially distributed.
 - Customers never abandon any service request while waiting for a support agent.
 - All lost calls are not abandoned, but simply delayed.
 - A support agent handles only one customer exclusively for the specified period.
 - The total number of support resources is lower than the number of customers.

"""
from typing import Type, List
import math
import pandas as pd
from macorp.forecast.utils import ErlangC


def nurse_demand(chats, average_handling_time: float = 20., target_wait_time: float = 10,
                 agent_refresh_fraction: float = 0.1, satisfaction: float = 0.8) -> int:
    """
    Calculates and returns the recommended number of nurses required to be present for serving give number of chats (single entry)

    :param chats: number of chats requiring service
    :param average_handling_time: average time spent with a customer from when they are connected to a nurse
    :param target_wait_time: targeted wait time; increasing this number reduces operation costs of the company through recommending less number of nurses but a long time makes patients impatient!
    :param agent_refresh_fraction: fraction of time each nurse is expected to spend on refreshment.
    :param satisfaction: level of satisfaction of given criteria (0. to 1.)
    :return: None
    """
    er_obj = ErlangC(chats, average_handling_time, target_wait_time, 24 * 60, agent_refresh_fraction)
    return er_obj.required_positions(satisfaction, 1)['positions']


def nurse_demand_batch(chats_series: Type[pd.Series], average_handling_time: float = 20., target_wait_time: float = 10,
                       agent_refresh_fraction: float = 0.1, satisfaction: float = 0.8, verbosity: int = 0) -> List[int]:
    """
    Calculates and returns the recommended number of nurses required to be present for serving give a series of number of chats (batch entry)

    :param chats_series: pandas.Series containing number of chats indexed by dates.
    :param average_handling_time: average time spent with a customer from when they are connected to a nurse
    :param target_wait_time: targeted wait time; increasing this number reduces operation costs of the company through recommending less number of nurses but a long time makes patients impatient!
    :param agent_refresh_fraction: fraction of time each nurse is expected to spend on refreshment.
    :param satisfaction: level of satisfaction of given criteria (0. to 1.)
    :return: None
    """
    chats = chats_series.values
    results = []
    for i, chat in enumerate(chats):
        nurse = nurse_demand(chat, average_handling_time, target_wait_time, agent_refresh_fraction, satisfaction)
        if verbosity > 0:
            print('{} nurses is recommended for {}'.format(nurse, chats_series.index[i]))
        results.append(nurse)
    return results


