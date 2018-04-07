"""
A flexible script for exploring performance of the
Surprise recommender system library
using various MovieLens datasets
"""
# pylint: disable=E0401
import argparse
import json
import time
from collections import OrderedDict

import pandas as pd
import numpy as np
from utils import get_dfs, concat_output_filename
from prep_organized_boycotts import (
    group_by_age, group_by_gender, group_by_genre,
    group_by_occupation, group_by_power, group_by_state
)
from joblib import Parallel, delayed

from surprise.model_selection import cross_validate, cross_validate_custom
from surprise import SVD, KNNBasic, Dataset, KNNBaseline
from surprise.reader import Reader

# long-term: massively abstract this code so it will work w/ non-recsys algorithsm

# Notes on default algo params:
# KNN uses 40 max neighbors by default
# min neighbors is 1, and if there's no neighbors then use global average!
# default similarity is MSD
# default user or item is USER-BASED but we override that.

NUM_FOLDS = 5

def task(algo_name, algo, nonboycott, boycott, boycott_uid_set, like_boycotters_uid_set, measures, cv, verbose, identifier, num_ratings, num_users, num_movies, name):
    return {
        'subset_results': cross_validate_custom(algo, nonboycott, boycott, boycott_uid_set, like_boycotters_uid_set, measures, cv, n_jobs=1),
        'num_ratings': num_ratings,
        'num_users': num_users,
        'num_movies': num_movies,
        'name': name,
        'algo_name': algo_name,
        'identifier': identifier
    }


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
        # 'SVD50': SVD(n_factors=50),
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
    print('Going to load data in pd dataframes')
    dfs = get_dfs(args.dataset)
    times['dfs_loaded'] = time.time() - times['start']
    print('Got dataframes, took {} seconds'.format(times['dfs_loaded']))

    ratings_df, users_df, movies_df = dfs['ratings'], dfs['users'], dfs['movies']
    if args.mode == 'info':
        print(ratings_df.info())
        print(users_df.info())
        return
    data = Dataset.load_from_df(
        ratings_df[['user_id', 'movie_id', 'rating']],
        reader=Reader()
    )
    times['data_constructed'] = time.time() - times['dfs_loaded']

    # note to reader: why are precision, recall, and ndcg all stuffed together in one string?
    # this ensures they will be computed all at once. Evaluation code will split them up for presentation
    measures = ['RMSE', 'MAE', 'prec10t4_prec5t4_rec10t4_rec5t4_ndcg10_ndcg5_ndcgfull']
    metric_names = []
    for measure in measures:
        if '_' in measure:
            metric_names += measure.lower().split('_')
        else:
            metric_names.append(measure.lower())
    
    standard_results = {}
    for algo_name in algos:
        filename_ratingcv_standards = 'standard_results/{}_ratingcv_standards_for_{}.json'.format(
            args.dataset, algo_name)
        try:
            if not args.use_precomputed:
                raise ValueError('This might be a bad pattern, but we need to go to the except block!')
            with open(filename_ratingcv_standards, 'r') as f:
                results = json.load(f)
        except:
            print('Computing standard results for {}'.format(algo_name))
            results = cross_validate_custom(algos[algo_name], data, Dataset.load_from_df(pd.DataFrame(), reader=Reader()), [], [], measures, NUM_FOLDS)
            print(results)
            saved_results = {}
            for metric in metric_names:
                saved_results[metric] = np.mean(results[metric + '_all'])
                frac_key = metric + '_frac_all'
                if frac_key in results:
                    saved_results[frac_key] = np.mean(results[frac_key])

            with open(filename_ratingcv_standards, 'w') as f:
                json.dump(saved_results, f)
            
        standard_results[algo_name] = saved_results

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
    elif args.grouping in [
        'gender', 'age', 'power', 'state', 'genre',
    ]:
        experiment_configs += [{'type': args.grouping, 'size': None}]

    num_configs = len(experiment_configs)
    if args.sample_sizes:
        num_runs = num_configs * args.num_samples
        print('{} total train/tests will be run because you chose {} sample_sizes and number of samples of {}'.format(
            num_runs, num_configs, args.num_samples
        ))
    else:
        print('{} total train/tests will be will be run'.format(num_configs))
        num_runs = num_configs
    secs = 47 * num_runs
    hours = secs / 3600
    time_estimate = """
        At a rate of 47 seconds per run, this should take: {} seconds ({} hours)
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
            experimental_iterations = group_by_gender(users_df)
        elif config['type'] == 'age':
            experimental_iterations = group_by_age(users_df)
        elif config['type'] == 'state':
            experimental_iterations = group_by_state(users_df, dataset=args.dataset)

        elif config['type'] == 'genre':
            experimental_iterations = group_by_genre(
                users_df=users_df, ratings_df=ratings_df, movies_df=movies_df,
                dataset=args.dataset)
        elif config['type'] == 'power':
            experimental_iterations = group_by_power(users_df=users_df, ratings_df=ratings_df, dataset=args.dataset)
        elif config['type'] == 'occupation':
            experimental_iterations = group_by_occupation(users_df)

        for algo_name in algos:
            delayed_iteration_list = []
            for i, experimental_iteration in enumerate(experimental_iterations):
                if config['type'] == 'individual_users':
                    row = experimental_iteration[1]
                    identifier = row.user_id
                    name = 'individual'
                    if args.indices != 'all':
                        if identifier < args.indices[0] or identifier > args.indices[1]:
                            continue
                    boycott_uid_set = set([row.user_id])
                    like_boycotters_uid_set = set([])
                    
                elif config['type'] in [
                    'sample_users',
                    'gender', 'age', 'power', 'state', 'genre',
                ]:
                    identifier = i
                    name = experimental_iteration['name']

                    possible_boycotters_df = experimental_iteration['df']
                    print(name)
                    print(possible_boycotters_df.head())
                    if args.userfrac != 1.0:
                        boycotters_df = possible_boycotters_df.sample(frac=args.userfrac)
                    else:
                        boycotters_df = possible_boycotters_df
                    boycott_uid_set = set(boycotters_df.user_id)
                    like_boycotters_df = possible_boycotters_df.drop(boycotters_df.index)
                    like_boycotters_uid_set = set(like_boycotters_df.user_id)

                non_boycott_user_ratings_df = ratings_df[~ratings_df.user_id.isin(boycott_uid_set)]
                boycott_ratings_df = None
                boycott_user_lingering_ratings_df = None
                for uid in boycott_uid_set:
                    ratings_belonging_to_user = ratings_df[ratings_df.user_id == uid]
                    if args.ratingfrac != 1.0:
                        boycott_ratings_for_user = ratings_belonging_to_user.sample(frac=args.ratingfrac)
                    else:
                        boycott_ratings_for_user = ratings_belonging_to_user
                    lingering_ratings_for_user = ratings_belonging_to_user.drop(boycott_ratings_for_user.index)
                    if boycott_ratings_df is None:
                        boycott_ratings_df = boycott_ratings_for_user
                    else:
                        boycott_ratings_df = pd.concat([boycott_ratings_df, boycott_ratings_for_user])
                    if boycott_user_lingering_ratings_df is None:
                        boycott_user_lingering_ratings_df = lingering_ratings_for_user
                    else:
                        boycott_user_lingering_ratings_df = pd.concat([boycott_user_lingering_ratings_df, lingering_ratings_for_user])
                # print(boycott_ratings_df.head())
                # print(boycott_user_lingering_ratings_df.head())
                print('Boycott ratings: {}, Lingering Ratings from Boycott Users: {}'.format(
                    len(boycott_ratings_df.index), len(boycott_user_lingering_ratings_df.index)
                ))
                all_non_boycott_ratings_df = pd.concat(
                    [non_boycott_user_ratings_df, boycott_user_lingering_ratings_df])

                nonboycott = Dataset.load_from_df(
                    all_non_boycott_ratings_df[['user_id', 'movie_id', 'rating']],
                    reader=Reader()
                )
                boycott = Dataset.load_from_df(
                    boycott_ratings_df[['user_id', 'movie_id', 'rating']],
                    reader=Reader()
                )
                identifier = str(identifier).zfill(4)
                num_users = len(all_non_boycott_ratings_df.user_id.value_counts())
                num_movies = len(all_non_boycott_ratings_df.movie_id.value_counts())
                num_ratings =  len(all_non_boycott_ratings_df.index)
                delayed_iteration_list += [delayed(task)(
                    algo_name, algos[algo_name], nonboycott, boycott, boycott_uid_set, like_boycotters_uid_set, measures, NUM_FOLDS,
                    False, identifier,
                    num_ratings,
                    num_users,
                    num_movies, name
                )]

            out_dicts = Parallel(n_jobs=-1, max_nbytes=1e7)(tuple(delayed_iteration_list))
            for d in out_dicts:
                res = d['subset_results']
                algo_name = d['algo_name']
                uid = str(d['identifier']) + '_' + d['algo_name']
                uid_to_error[uid] = {
                    'num_ratings': d['num_ratings'],
                    'num_users': d['num_users'],
                    'num_movies': d['num_movies'],
                    'name': d['name'],
                    'algo_name': d['algo_name'],
                }
                for metric in metric_names + ['fit_time', 'test_times', 'num_tested']:
                    for group in ['all', 'non-boycott', 'boycott', 'like-boycott', 'all-like-boycott']:
                        key = '{}_{}'.format(metric, group)
                        # if group in ['boycott', ]:
                        #     val = np.nanmean(res[key])
                        vals = res.get(key)
                        if vals:
                            val = np.mean(res[key])
                            uid_to_error[uid].update({
                                key: val,
                            })
                            try:
                                uid_to_error[uid].update({
                                    'increase_from_baseline_{}'.format(key): val - standard_results[algo_name][metric],
                                })
                            except KeyError:
                                pass
        err_df = pd.DataFrame.from_dict(uid_to_error, orient='index')

        outname = concat_output_filename(
            args.dataset, config['type'], args.userfrac,
            args.ratingfrac,
            config['size'], args.num_samples
        )
        err_df.to_csv(outname)
        # means = err_df.mean()
        # means.to_csv(outname.replace('err_df', 'means'))
        print('Full runtime was: {} for {} runs'.format(time.time() - times['start'], num_runs))


def parse():
    """
    Parse args and handles list splitting

    Example: 
    python sandbox.py --grouping state
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', default='all')
    parser.add_argument('--grouping', default='individual_users')
    parser.add_argument('--sample_sizes')
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument(
        '--use_precomputed', action='store_true',
        help='Defaults to false. Pass --use_precomputed if you WANT to use precomputed')
    parser.add_argument('--mode', default='compute')
    parser.add_argument('--userfrac', type=float, default=1.0)
    parser.add_argument('--ratingfrac', type=float, default=1.0)
    args = parser.parse_args()

    if args.sample_sizes:
        args.sample_sizes = [int(x) for x in args.sample_sizes.split(',')]
        if args.num_samples is None:
            args.num_samples = 1000

    if ',' in args.indices:
        args.indices = [int(x) for x in args.indices.split(',')]

    main(args)


if __name__ == '__main__':
    parse()
