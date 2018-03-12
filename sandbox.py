"""
sandbox.py includes a simple experimentation with Surprise and the ML-100k dataset
"""
import argparse
import json

import pandas as pd
import numpy as np
from utils import movielens_to_df, movielens_1m_to_df


from surprise.model_selection import cross_validate
from surprise import SVD, KNNBasic, Dataset
from surprise.builtin_datasets import BUILTIN_DATASETS
from surprise.reader import Reader

# TODO: re-run everything on 1mil
# TODO: how sparse is gender, ocupation, and age?


def main(args):
    """
    Run the sandbox experiments
    """
    # HEY LISTEN
    # uncomment to make sure the dataset is downloaded (e.g. first time on a new machine)
    # data = Dataset.load_builtin('ml-1m')
    algos = {
        'SVD': SVD(),
        # 'KNNBasic_user_msd': KNNBasic(),
        'KNNBasic_item_msd': KNNBasic(sim_options={'user_based': False}),
        # 'KNNBasic_item_cosine': KNNBasic(sim_options={'user_based': False, 'name': 'cosine', }),
        # 'KNNBasic_item_pearson': KNNBasic(sim_options={'user_based': False, 'name': 'pearson', }),
    }

    dataset_suffix = '1m'
    

    

    if dataset_suffix == '100k':
        ratings_path = BUILTIN_DATASETS['ml-' + dataset_suffix].path
        users_path = ratings_path.replace('.data', '.user')
        movies_path = ratings_path.replace('.data', '.item')
        dfs = movielens_to_df(ratings_path, users_path, movies_path)
    elif dataset_suffix == '1m':
        ratings_path = BUILTIN_DATASETS['ml-' + dataset_suffix].path
        users_path = ratings_path.replace('ratings.', 'users.')
        movies_path = ratings_path.replace('ratings.', 'movies.')
        dfs = movielens_1m_to_df(ratings_path, users_path, movies_path)
    
    ratings_df, users_df, items_df = dfs['ratings'], dfs['users'], dfs['movies']
    data = Dataset.load_from_df(
        ratings_df[['user_id', 'movie_id', 'rating']],
        reader=Reader()
    )
    measures = ['RMSE', 'MAE', 'precision10t4_recall10t4_ndcg10']
    standard_results = {}
    for algo_name in algos:
        standard_results_filename = 'standard_measures_{}.json'.format(algo_name)
        try:
            with open(standard_results_filename, 'r') as f:
                results = json.load(f)
            print('Loaded standard results for {} from {}'.format(algo_name, standard_results_filename))
            print(results)
        except:
            print('Computing standard results for {}'.format(algo_name))
            results = cross_validate(
                algos[algo_name], data, measures=measures,
                cv=5, verbose=True)
            results = {
                'mae': np.mean(results['test_mae']),
                'rmse': np.mean(results['test_rmse']),
                'precision10t4': np.mean(results['test_precision10t4']),
                'recall10t4': np.mean(results['test_recall10t4']),
                'ndcg10': np.mean(results['test_ndcg10']),
            }
            with open(standard_results_filename, 'w') as f:
                json.dump(results, f)
        standard_results[algo_name] = results


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
            raise ValueError('When using grouping="sample", you must provide a set of sample sizes')
    elif args.grouping == 'gender':
        experiment_configs += [{'type': 'gender', 'size': None}]
        print(users_df.gender.head())
    elif args.grouping == 'age':
        experiment_configs += [{'type': 'age', 'size': None}]
        print(users_df.age.unique())

    uid_to_error = {}
    for config in experiment_configs:
        if config['type'] == 'individual_users':
            experimental_iterations = users_df.iterrows()
        elif config['type'] == 'sample_users':
            experimental_iterations = [{
                    'df': users_df.sample(config['size']),
                    'name': '{} user sample'.format(config['size'])
                } for _ in range(args.num_samples)
            ]
        elif config['type'] == 'gender':
            experimental_iterations = [
                {'df': users_df[users_df.gender == 'M'], 'name': 'exclude male users'},
                {'df': users_df[users_df.gender == 'F'], 'name': 'exclude female users'},
            ]
        elif config['type'] == 'age':
            gte40 = [
                x >= 40 for x in list(users_df.age)
            ]
            users_df = users_df.assign(gte40=gte40)
            experimental_iterations = [
                {'df': users_df[users_df.gte40 == 1], 'name': 'exclude gte age 40 users'},
                {'df': users_df[users_df.gte40 == 0], 'name': 'exclude lt age 40 users'},
            ]
        elif config['type'] == 'rural':
            # TODO
            pass

        for i, experimental_iteration in enumerate(experimental_iterations):
            print(experimental_iteration)
            if config['type'] == 'individual_users':
                if args.num_users_to_stop_at:
                    if i >= args.num_users_to_stop_at - 1:
                        break
                row = experimental_iteration[1]
                subset_ratings_df = ratings_df[ratings_df.user_id != row.user_id]
                identifier = row.user_id
                name = 'individual'
            elif config['type'] in ['sample_users', 'gender', 'age', 'rural']:
                ids = list(experimental_iteration['df'].user_id)
                subset_ratings_df = ratings_df[~ratings_df.user_id.isin(ids)]
                identifier = i
                name = experimental_iteration['name']
            subset_data = Dataset.load_from_df(
                subset_ratings_df[['user_id', 'movie_id', 'rating']],
                reader=Reader()
            )
            for algo_name in algos:
                subset_results = cross_validate(
                    algos[algo_name], subset_data, measures=measures,
                    cv=5, verbose=False)
                num_users = len(subset_ratings_df.user_id.value_counts())
                num_movies = len(subset_ratings_df.movie_id.value_counts())
                uid_to_error[str(identifier) + '_' + algo_name] = {
                    'mae increase': np.mean(subset_results['test_mae']) - standard_results[algo_name]['mae'],
                    'rmse increase': np.mean(subset_results['test_rmse']) - standard_results[algo_name]['rmse'],
                    'precision10t4 decrease': standard_results[algo_name]['precision10t4'] - np.mean(subset_results['test_precision10t4']),
                    'recall10t4 decrease': standard_results[algo_name]['recall10t4'] - np.mean(subset_results['test_recall10t4']),
                    'ndcg10 decrease': standard_results[algo_name]['ndcg10'] - np.mean(subset_results['test_ndcg10']),
                    'num_ratings': len(subset_ratings_df.index),
                    'num_tested': np.mean(subset_results['num_tested']),
                    'num_users': num_users,
                    'num_movies': num_movies,
                    'fit_time': np.mean(subset_results['fit_time']),
                    'test_time': np.mean(subset_results['test_time']),
                    'name': name,
                    'algo_name': algo_name,
                }
            
        err_df = pd.DataFrame.from_dict(uid_to_error, orient='index')
        outname = 'results/err_df-type_{}-size_{}-sample_size_{}.csv'.format(
            config['type'], config['size'],
            args.num_samples if args.num_samples else None)
        err_df.to_csv(outname)
        means = err_df.mean()
        means.to_csv(outname.replace('err_df', 'err_df_means'))
        print(means)


def parse():
    """
    Parse args and handles list splitting
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users_to_stop_at', type=int)
    parser.add_argument('--grouping', default='individual_users')
    parser.add_argument('--sample_sizes')
    parser.add_argument('--num_samples', type=int)
    args = parser.parse_args()
    if args.sample_sizes:
        args.sample_sizes = [int(x) for x in args.sample_sizes.split(',')]
        if args.num_samples is None:
            args.num_samples = 1000
    main(args)

if __name__ == '__main__':
    parse()
