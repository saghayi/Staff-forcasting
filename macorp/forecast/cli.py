r"""
This files defines the commands of MaCorp's forecast module.
"""

import os
import click
import pathlib
from typing import AnyStr, Union, Optional
import pandas as pd
from pathlib import Path
import datetime
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import svm
import matplotlib.dates as mdates

from macorp.forecast.train import chat_train
from macorp.forecast.inference import chat_inference, chat_inference_batch
from macorp.forecast.staffing import nurse_demand, nurse_demand_batch


def main() -> None:
    """
    main method of forecast cli.
    :return: None
    """
    run()


@click.group()
def run() -> None:
    """
    group definition for click commands.
    :return: None
    """
    pass


@run.command()
@click.option('-p', '--path_to_chat_history',
              type=click.Path(exists=True, resolve_path=True), required=True,
              help='path to a csv file with the following column format (date, eligible_users, chats) ')
@click.option('-d', '--deployment-dir', type=click.Path(exists=True, resolve_path=True), default='deployed_models',
              help='directory to write out the trained model')
@click.option('-s', '--scoring', type=str, default='neg_mean_absolute_error',
              help='scoring function for model exploration. Find more option at '
                   'https://scikit-learn.org/stable/modules/model_evaluation.html')
@click.option('-i', '--visualize/--no-visualize', default=False)
@click.option('-v', '--verbosity', type=click.IntRange(0, 5, clamp=True), default=0,
              help='optional verbosity for controlling the output info.')
def train_chat_forecast(path_to_chat_history: Union[str, pathlib.Path],
                        deployment_dir: AnyStr,
                        scoring: AnyStr,
                        visualize: bool,
                        verbosity: int) -> None:
    """
    Trains a regressor for predicting number of chats in a day.
    Example:
    # macorp.forecast train-chat-forecast -p data/chat_demand.csv -i


    :param path_to_chat_history: path to chat data in csv format. The file must have the following columns: (date, eligible_users, chats)
    :param deployment_dir: the directory in which the deployment model should be store.
    :param scoring: loss functions for training, default is 'neg_mean_absolute_error'. Other options are listed at https://scikit-learn.org/stable/modules/model_evaluation.html
    :param visualize: boolean flag for visualizing the validation results.
    :param verbosity: level of verbosity.
    :return: None
    """
    if verbosity > 0:
        print('checking the deployment directory {} ...'.format(deployment_dir))

    Path(deployment_dir).mkdir(parents=True, exist_ok=True)

    if verbosity > 0:
        print('loading chat history file at {} ...'.format(path_to_chat_history))
    df = pd.read_csv(path_to_chat_history, delimiter=',', index_col=0)
    if verbosity > 0:
        print('filtering invalid entries ...')
    df_train = df[~df['chats'].isnull()]

    if verbosity > 0:
        print('collecting candidate models ...')

    # the length of pool_of_regressors and regressors_params should match!
    pool_of_regressors = [linear_model.LinearRegression,
                          svm.SVR,
                          # Todo: add more regressors here
                          ]
    regressors_params = [None,
                         dict(C=[0.5, 1, 5, 10]),
                         # Todo: add more params
                         ]

    if verbosity > 0:
        print('Start training procedure ...')
    train_outputs = chat_train(df_train,
                               pool_of_regressors=pool_of_regressors,
                               regressor_params=regressors_params,
                               scoring=scoring,
                               visualize=visualize,
                               verbosity=verbosity)
    if visualize:
        model, best_score_idx, best_score, fig, ax = train_outputs
    else:
        model, best_score_idx, best_score = train_outputs
        fig, ax = None, None

    model_name = pool_of_regressors[best_score_idx].__name__
    if verbosity > 0:
        print('found best score of {}={:.3f} by {}'.format(scoring, best_score, model_name))
        print('saving the trained model ...')

    date = datetime.date.today().strftime("%Y-%m-%d")
    file_name = '{}.{}___{}'.format(model_name, scoring, date)

    with open(os.path.join(deployment_dir, file_name+'.pkl'), 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if verbosity > 0:
        print('model was successfully saved with name {}'.format(file_name))

    if visualize:
        fig.savefig(os.path.join(deployment_dir, file_name+'.png'.format(model_name, date)), dpi=160)


@run.command()
@click.option('-d', '--deployment-path', type=str, required=True, help='path to deployment model')
@click.option('-t', '--date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True,
              help='date to forecast number of chats for (ex: 2017-07-01)')
@click.option('-e', '--eligible-users', required=True, type=click.IntRange(0, 100000000, clamp=True),
              help='number of eligible users in the provided date.')
@click.option('-v', '--verbosity', type=click.IntRange(0, 5, clamp=True), default=0,
              help='optional verbosity for controlling the output info.')
def chat_forecast(deployment_path: Union[str, pathlib.Path], date: AnyStr, eligible_users: int, verbosity: int) -> None:
    """
    predicts number of chats for a specific date.
    Example:
    # macorp.forecast chat-forecast -d deployed_models/LinearRegression___2022-03-31.pkl -t 2017-07-01 -e 45000


    :param deployment_path: path to a deployment model to load.
    :param date: date string ('example 2017-07-01')
    :param eligible_users: number of users connected to the platform
    :param verbosity: level of verbosity
    :return: None
    """
    if verbosity > 0:
        print('verifying the date')
    if not isinstance(date, datetime.datetime):
        date_obj = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    else:
        date_obj = date.date()

    if verbosity > 0:
        print('loading the deployment model')

    with open(deployment_path, 'rb') as handle:
        model = pickle.load(handle)
    result = chat_inference(date_obj, eligible_users, regressor=model)
    print(result)


@run.command()
@click.option('-p', '--deployment-path', type=click.Path(exists=True, resolve_path=True), required=True,
              help='path to deployment model')
@click.option('-d', '--demand-path', type=click.Path(exists=True, resolve_path=True), required=True,
              help='Must point to a csv file with the following column format (date, eligible_users) or (date, '
                   'eligible_users, chats)')
@click.option('-i', '--visualization-path', type=str, default=None,
              help='A visualization will be made in the given path. If not provided, nothing is plotted.')
@click.option('-v', '--verbosity', type=click.IntRange(0, 5, clamp=True)
    , default=0, help='optional verbosity for controlling the output info.')
def chat_forecast_batch(deployment_path: Union[str, pathlib.Path],
                        demand_path: Union[str, pathlib.Path],
                        visualization_path: AnyStr,
                        verbosity: int):
    """
    predicts number of chats for a batch of dates stored in a csv file.
    Example:
    macorp.forecast chat-forecast-batch -p ./deployed_models/LinearRegression___2022-03-31.pkl -d ./data/chat_demand.csv -i chat_forecast_visual.png


    :param deployment_path: path to a deployment model to load.
    :param demand_path: path to demand csv file with the following columns: (date, eligible_users) or (date, eligible_users, chats)
    :param visualization_path: if provided the result visualized is stored in the given path
    :param verbosity: level of verbosity
    :return: None
    """
    if verbosity > 0:
        print('loading demand file ...')

    df = pd.read_csv(demand_path, delimiter=',', index_col=0)

    if verbosity > 0:
        print('Filtering invalid entries ...')
    if 'chats' in df:
        eligible_series = df.loc[df['chats'].isnull(), 'eligible_users']
    else:
        eligible_series = df.loc[:, 'eligible_users']

    if verbosity > 0:
        print('loading the deployment model')

    with open(deployment_path, 'rb') as handle:
        model = pickle.load(handle)

    result = chat_inference_batch(eligible_series, regressor=model)

    if visualization_path is not None:
        if verbosity > 0:
            print('visualizing results ...')
        fig, ax = plt.subplots()
        plt.xticks(rotation=70)
        if 'chats' in df:
            ax.plot(df.loc[~df['chats'].isnull()].index.values, df.loc[~df['chats'].isnull(), 'chats'].values,
                    label='existing')
        ax.plot(eligible_series.index.values, result, label='forecast')

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.set_ylabel('chats')
        fig.tight_layout()
        ax.legend(loc='best')
        fig.savefig(visualization_path)

    print(*result)


@run.command()
@click.option('-c', '--chats', type=click.IntRange(0, 5000000, clamp=True), required=True,
              help='number of chats')
@click.option('-h', '--avg-handling-time', required=True, type=click.FloatRange(0., 200., clamp=False), default=10.,
              help='average handling time of a customer in minutes.')
@click.option('-w', '--wait-time', required=True, type=click.FloatRange(0., 200., clamp=False), default=10.,
              help='minutes each client would wait on average.')
@click.option('-r', '--agent-refresh-fraction', required=True, type=click.FloatRange(0., 1., clamp=True), default=10.,
              help='fraction of time a nurse needs to refresh.')
@click.option('-s', '--satisfaction', required=True, type=click.FloatRange(0., 1., clamp=False), default=0.9,
              help='satisfaction level between 0 and 1')
def staff_recom(chats: int, avg_handling_time: float, wait_time: float, agent_refresh_fraction: float,
                satisfaction: float):
    """
    Recommends number of nurses based on a single demand entry (number of chats) and given corporate standards
    Example:
    # macorp.forecast staff-recom -c 450 -h 20 -w 10 -r 0.1 -s 0.9

    :param chats: number of chats
    :param avg_handling_time: average time spent with a customer from when they are connected to a nurse
    :param wait_time: targeted wait time; increasing this number reduces operation costs of the company through recommending less number of nurses but a long time makes patients impatient!
    :param agent_refresh_fraction: fraction of time each nurse is expected to spend on refreshment.
    :param satisfaction: level of satisfaction of given criteria (0. to 1.)
    :return: None
    """
    staff = nurse_demand(chats, avg_handling_time, wait_time, agent_refresh_fraction, satisfaction)
    print(staff)


@run.command()
@click.option('-p', '--deployment-path', type=click.Path(exists=True, resolve_path=True), required=False,
              help='path to deployment model. Only used if chats is missing partially or totally from demand file')
@click.option('-d', '--demand-path', type=click.Path(exists=True, resolve_path=True), required=True,
              help='Must point to a csv file with the following column format (date, eligible_users) or '
                   '(date, eligible_users, chats)')
@click.option('-i', '--visualization-path', type=str, default=None,
              help='A visualization will be made in the given path. If not provided, nothing is plotted.')
@click.option('-h', '--avg-handling-time', required=True, type=click.FloatRange(0., 200., clamp=False), default=10.,
              help='average handling time of a customer in minutes.')
@click.option('-w', '--wait-time', required=True, type=click.FloatRange(0., 200., clamp=False), default=10.,
              help='minutes each client would wait on average.')
@click.option('-r', '--agent-refresh-fraction', required=True, type=click.FloatRange(0., 1., clamp=True), default=10.,
              help='fraction of time a nurse needs to refresh.')
@click.option('-s', '--satisfaction', required=True, type=click.FloatRange(0., 1., clamp=False), default=0.9,
              help='satisfaction level between 0 and 1')
@click.option('-v', '--verbosity', type=click.IntRange(0, 5, clamp=True), default=0,
              help='optional verbosity for controlling the output info.')
def staff_recom_batch(deployment_path: Optional[Union[str, pathlib.Path]],
                      demand_path: Union[str, pathlib.Path],
                      visualization_path: AnyStr,
                      avg_handling_time: float,
                      wait_time: float,
                      agent_refresh_fraction: float,
                      satisfaction: float,
                      verbosity: int):
    """
    Recommends number of nurses based on a batch of demand entry (number of chats) and given corporate standards
    Example:
    # macorp.forecast staff-recom-batch -p deployed_models/LinearRegression___2022-03-31.pkl -d data/chat_demand.csv -h 20 -w 10 -r 0.1 -s 0.9 -i docs/nurse_recom.png

    :param deployment_path: path to a deployment model to load.
    :param demand_path: path to demand csv file with the following columns: (date, eligible_users) or (date, eligible_users, chats)
    :param visualization_path: if provided the result visualized is stored in the given path
    :param avg_handling_time: average time spent with a customer from when they are connected to a nurse
    :param wait_time: targeted wait time. In creasing this number reduces operation costs of the company through recommending less number of nurses but a long time makes patients impatient!
    :param agent_refresh_fraction: fraction of time each nurse is expected to spend on refreshment.
    :param satisfaction: level of satisfaction of given criteria (0. to 1.)
    :param verbosity:
    :return: None
    """
    if verbosity > 0:
        print('loading demand file ...')

    df = pd.read_csv(demand_path, delimiter=',', index_col=0)

    if verbosity > 0:
        print('Filtering invalid entries ...')

    requires_chat_forecast = 'chats' not in df
    if not requires_chat_forecast:
        missing_chats = df.loc[df['chats'].isnull()]
    else:
        missing_chats = df
    #requires_chat_forecast = requires_chat_forecast or len(missing_chats) == len(df)
    if len(missing_chats) > 0:
        assert deployment_path is not None

        if verbosity > 0:
            print('found missing chats in demand file')
            print('loading the deployment model ...')

        with open(deployment_path, 'rb') as handle:
            model = pickle.load(handle)
        eligible_series = missing_chats['eligible_users']
        result = chat_inference_batch(eligible_series, regressor=model)
        df.loc[eligible_series.index, 'predicted_chats'] = result

    if visualization_path is not None:
        fig, ax = plt.subplots()
        plt.xticks(rotation=70)
    else:
        fig, ax = None, None

    results = []
    if 'chats' in df:
        chats = df.loc[~df['chats'].isnull(), 'chats']
        result = nurse_demand_batch(chats, avg_handling_time, wait_time, agent_refresh_fraction, satisfaction,
                                    verbosity)
        if visualization_path is not None:
            if verbosity > 0:
                print('visualizing results for days with existing chats count ...')
            ax.plot(df.loc[~df['chats'].isnull()].index, result, label='with existing chats')
        results.append(result)
    if 'predicted_chats' in df:
        chats = df.loc[~df['predicted_chats'].isnull(), 'predicted_chats']
        result = nurse_demand_batch(chats, avg_handling_time, wait_time, agent_refresh_fraction, satisfaction,
                                    verbosity)
        if visualization_path is not None:
            if verbosity > 0:
                print('visualizing results for days with predicting chats count ...')
            ax.plot(df.loc[~df['predicted_chats'].isnull()].index, result, label='with predicted chats')
        results.append(result)

    if visualization_path is not None:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.set_ylabel('nurse demand')
        fig.tight_layout()
        ax.legend(loc='best')
        fig.savefig(visualization_path)

    for result in results:
        print(*result)


if __name__ == '__main__':
    main()
