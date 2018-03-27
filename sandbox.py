"""
sandbox.py includes a simple experimentation with Surprise and the ML-100k dataset
"""
# pylint: disable=E0401
import argparse
import json
import time
from collections import OrderedDict

import pandas as pd
import numpy as np
from utils import movielens_to_df, movielens_1m_to_df
from joblib import Parallel, delayed

from surprise.model_selection import cross_validate, cross_validate_users
from surprise import SVD, KNNBasic, Dataset, KNNBaseline
from surprise.builtin_datasets import BUILTIN_DATASETS
from surprise.reader import Reader

# long task: abstract this code so it will work w/ WP algorithms


# Quick TODO: zero pad the identifiers for the spreadsheet output

# Notes on default algo params:
# KNN uses 40 max neighbors by default
# min neighbors is 1, and if there's no neighbors then use global average!
# default similarity is MSD
# default user or item is USER-BASED but we override that.

NUM_FOLDS = 5


def task(algo_name, algo, data, all_uids, out_uids, measures, cv, verbose, identifier, num_ratings, num_users, num_movies, name):
    return {
        'subset_results': cross_validate_users(algo, data, all_uids, out_uids, measures, cv, n_jobs=1),
        'num_ratings': num_ratings,
        'num_users': num_users,
        'num_movies': num_movies,
        'name': name,
        'algo_name': algo_name,
        'identifier': identifier
    }


def get_dfs(dataset):
    """i
    Takes a dataset string and return that data in a dataframe!
    """
    if dataset == 'ml-100k':
        ratings_path = BUILTIN_DATASETS[dataset].path
        users_path = ratings_path.replace('.data', '.user')
        movies_path = ratings_path.replace('.data', '.item')
        dfs = movielens_to_df(ratings_path, users_path, movies_path)
    elif dataset == 'ml-1m':
        ratings_path = BUILTIN_DATASETS[dataset].path
        users_path = ratings_path.replace('ratings.', 'users.')
        movies_path = ratings_path.replace('ratings.', 'movies.')
        dfs = movielens_1m_to_df(ratings_path, users_path, movies_path)
    return dfs

def main(args):
    """
    Run the sandbox experiments
    """
    # HEY LISTEN
    # uncomment to make sure the dataset is downloaded (e.g. first time on a new machine)
    # data = Dataset.load_builtin('ml-1m')
    # TODO: support FLOAT ratings for ml-20m... only supports int right now!
    times = OrderedDict()
    times['start'] = time.time()
    algos = {
        'SVD': SVD(),
        # 'KNNBasic_user_msd': KNNBasic(sim_options={'user_based': True}),
        # 'KNNBasic_user_cosine': KNNBasic(sim_options={'user_based': True, 'name': 'cosine'}),
        # 'KNNBasic_user_pearson': KNNBasic(sim_options={'user_based': True, 'name': 'pearson'}),
        # 'KNNBasic_item_msd': KNNBasic(sim_options={'user_based': False}),
        # 'KNNBasic_item_cosine': KNNBasic(sim_options={'user_based': False, 'name': 'cosine'}),
        # 'KNNBasic_item_pearson': KNNBasic(sim_options={'user_based': False, 'name': 'pearson'}),
        'KNNBaseline_item_msd': KNNBaseline(sim_options={'user_based': False}),
        # 'KNNBaseline_item_cosine': KNNBaseline(sim_options={'user_based': False, 'name': 'cosine'}),
        # 'KNNBaseline_item_pearson': KNNBaseline(sim_options={'user_based': False, 'name': 'pearson'}),
    }

    dfs = get_dfs(args.dataset)

    times['dfs_loaded'] = time.time() - times['start']
    ratings_df, users_df, _ = dfs['ratings'], dfs['users'], dfs['movies']
    all_uids = list(set(ratings_df.user_id))
    data = Dataset.load_from_df(
        ratings_df[['user_id', 'movie_id', 'rating']],
        reader=Reader()
    )
    times['data_constructed'] = time.time() - times['dfs_loaded']

    # why are precision, recall, and ndcg all stuffed together in one string
    # this ensures they will be computed all at once. Evaluation code will split them up for presentation
    measures = ['RMSE', 'MAE', 'precision10t4_recall10t4_ndcg10']
    baseline = {}

    for algo_name in algos:
        filename_usercv_standards = 'standard_results/{}_usercv_standards_for_{}.json'.format(
            args.dataset, algo_name)
        filename_ratingcv_standards = 'standard_results/{}_ratingcv_standards_for_{}.json'.format(
            args.dataset, algo_name)
        try:
            with open(filename_usercv_standards, 'r') as f:
                results = json.load(f)
            print('Loaded standard results for {} from {}'.format(
                algo_name, filename_ratingcv_standards))
        except:
            print('Computing standard results for {}'.format(algo_name))
            # todo fix keys here...
            results = cross_validate_users(algos[algo_name], data, all_uids, [], measures, NUM_FOLDS)
            results = {
                'mae': np.mean(results['mae_all']),
                'rmse': np.mean(results['rmse_all']),
                'precision10t4': np.mean(results['precision10t4_all']),
                'recall10t4': np.mean(results['recall10t4_all']),
                'ndcg10': np.mean(results['ndcg10_all']),
            }

            # results_itemsplit = cross_validate(algos[algo_name], data, measures, NUM_FOLDS)
            # print(results_itemsplit)
            # results_itemsplit = {
            #     'mae': np.mean(results_itemsplit['test_mae']),
            #     'rmse': np.mean(results_itemsplit['test_rmse']),
            #     'precision10t4': np.mean(results_itemsplit['test_precision10t4']),
            #     'recall10t4': np.mean(results_itemsplit['test_recall10t4']),
            #     'ndcg10': np.mean(results_itemsplit['test_ndcg10']),
            # }
            # with open(filename_ratingcv_standards, 'w') as f:
            #     json.dump(results_itemsplit, f)

            with open(filename_usercv_standards, 'w') as f:
                json.dump(results, f)
            
        baseline[algo_name] = results

    times['standards_loaded'] = time.time() - times['data_constructed']

    experiment_configs = []
    if args.grouping == 'individual_users':
        experiment_configs += [{'type': 'individual_users', 'size': None}]
    elif args.grouping == 'sample':
        if args.sample_sizes:
            experiment_configs += [
                {
                    'type': 'sample_users', 'size': sample_size
                } for sample_size in args.sample_sizes]
        else:
            raise ValueError(
                'When using grouping="sample", you must provide a set of sample sizes')
    elif args.grouping == 'gender':
        experiment_configs += [{'type': 'gender', 'size': None}]
        print(users_df.gender.unique())
    elif args.grouping == 'age':
        experiment_configs += [{'type': 'age', 'size': None}]
        print(users_df.age.unique())
    elif args.grouping == 'zip':
        experiment_configs += [{'type': 'zip', 'size': None}]
        print(users_df.zip_code.unique())
    elif args.grouping == 'genre':
        experiment_configs += [{'type': 'genre', 'size': None}]

    num_configs = len(experiment_configs)
    if args.sample_sizes:
        num_runs = num_configs * args.num_samples
        print('{} total 1train/3tests will be run because you chose {} sample_sizes and number of samples of {}'.format(
            num_runs, num_configs, args.num_samples
        ))
    else:
        print('{} total 1train/3tests will be run'.format(num_configs))
        num_runs = num_configs
    secs = 125 * num_runs
    hours = secs / 3600
    time_estimate = """
        Assuming that you're running KNN (65 sec) and SVD (60 sec), this will probably take
        an upper bound of {} seconds ({} hours)
    """.format(secs, hours)
    print(time_estimate)
    uid_to_error = {}
    for config in experiment_configs:
        if config['type'] == 'individual_users':
            experimental_iterations = list(users_df.iterrows())
        elif config['type'] == 'sample_users':
            experimental_iterations = [{
                'df': users_df.sample(config['size']),
                'name': '{} user sample'.format(config['size'])
            } for _ in range(args.num_samples)
            ]
        elif config['type'] == 'gender':
            experimental_iterations = [
                {'df': users_df[users_df.gender == 'M'],
                    'name': 'exclude male users'},
                {'df': users_df[users_df.gender == 'F'],
                    'name': 'exclude female users'},
            ]
        elif config['type'] == 'age':
            gte40 = [
                x >= 40 for x in list(users_df.age)
            ]
            users_df = users_df.assign(gte40=gte40)
            experimental_iterations = [
                {'df': users_df[users_df.gte40 == 1],
                    'name': 'exclude gte age 40 users'},
                {'df': users_df[users_df.gte40 == 0],
                    'name': 'exclude lt age 40 users'},
            ]
        elif config['type'] == 'zip':
            # implement
            pass
        elif config['type'] == 'genre':
            # implement
            pass

        for algo_name in algos:
            delayed_iteration_list = []
            for i, experimental_iteration in enumerate(experimental_iterations):
                print(i, algo_name)
                if config['type'] == 'individual_users':
                    if args.num_users_to_stop_at:
                        if i >= args.num_users_to_stop_at:
                            break
                    row = experimental_iteration[1]
                    print(row)
                    subset_ratings_df = ratings_df[ratings_df.user_id != row.user_id]
                    excluded_ratings_df = ratings_df[ratings_df.user_id == row.user_id]
                    identifier = row.user_id
                    name = 'individual'
                elif config['type'] in ['sample_users', 'gender', 'age', 'rural']:
                    ids = list(experimental_iteration['df'].user_id)
                    subset_ratings_df = ratings_df[~ratings_df.user_id.isin(ids)]
                    excluded_ratings_df = ratings_df[ratings_df.user_id.isin(ids)]

                    identifier = i
                    name = experimental_iteration['name']

                identifier = str(identifier).zfill(4)
                out_uids = list(set(excluded_ratings_df.user_id))
                num_users = len(subset_ratings_df.user_id.value_counts())
                num_movies = len(subset_ratings_df.movie_id.value_counts())
                delayed_iteration_list += [delayed(task)(
                    algo_name, algos[algo_name], data, all_uids, out_uids, measures, NUM_FOLDS,
                    False, identifier,
                    len(subset_ratings_df.index), #num ratings
                    num_users,
                    num_movies, name
                )]

            out_dicts = Parallel(n_jobs=-1, max_nbytes=1e7)(tuple(delayed_iteration_list))
            for d in out_dicts:
                res = d['subset_results']
                print(res)
                algo_name = d['algo_name']
                uid = str(d['identifier']) + '_' + d['algo_name']
                print(uid)
                uid_to_error[uid] = {
                    'num_ratings_in-group': d['num_ratings'],
                    'num_users_in-group': d['num_users'],
                    'num_movies_in-group': d['num_movies'],
                    'name': d['name'],
                    'algo_name': d['algo_name'],
                }
                for metric in ['rmse', 'ndcg10', 'fit_time', 'test_times', 'num_tested']:
                    for group in ['all', 'in-group', 'out-group']:
                        key = '{}_{}'.format(metric, group)

                        if group == 'out-group':
                            val = np.nanmean(res[key])
                        else:
                            val = np.mean(res[key])
                        uid_to_error[uid].update({
                            key: val,
                        })
                        try:
                            uid_to_error[uid].update({
                                'increase_from_baseline_{}'.format(key): val - baseline[algo_name][metric],
                            })
                        except KeyError:
                            pass
        err_df = pd.DataFrame.from_dict(uid_to_error, orient='index')
        outname = 'results/err_df-dataset_{}_type_{}-size_{}-sample_size_{}.csv'.format(
            args.dataset, config['type'], config['size'],
            args.num_samples if args.num_samples else None)
        err_df.to_csv(outname)
        # means = err_df.mean()
        # means.to_csv(outname.replace('err_df', 'means'))
        print('Full runtime was: {} for {} runs'.format(time.time() - times['start'], num_runs))


def parse():
    """
    Parse args and handles list splitting
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users_to_stop_at', type=int)
    parser.add_argument('--grouping', default='individual_users')
    parser.add_argument('--sample_sizes')
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--dataset', default='ml-1m')
    args = parser.parse_args()
    if args.sample_sizes:
        args.sample_sizes = [int(x) for x in args.sample_sizes.split(',')]
        if args.num_samples is None:
            args.num_samples = 1000
    print(args.num_users_to_stop_at)
    main(args)


if __name__ == '__main__':
    parse()
