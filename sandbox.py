"""
sandbox.py includes a simple experimentation with Surprise and the ML-100k dataset
"""
import argparse
import json

import pandas as pd
import numpy as np
from utils import movielens_to_df, movielens_1m_to_df
from joblib import Parallel, delayed

from surprise.model_selection import cross_validate
from surprise import SVD, KNNBasic, Dataset
from surprise.builtin_datasets import BUILTIN_DATASETS
from surprise.reader import Reader

# TODO: re-run everything on 1mil
# TODO: abstract this code so it will work w/ WP algorithms
def task(algo_name, algo, subset_data, measures, cv=5, verbose, identifier, num_ratings, num_movies, name, out_ids):
    return {
        'subset_results': cross_validate(algo_name, subset_data, measures=measures, cv, verbose, out_ids),
        'num_ratings': num_ratings,
        'num_movies': num_movies,
        'name': name,
    }


def main(args):
    """
    Run the sandbox experiments
    """
    # HEY LISTEN
    # uncomment to make sure the dataset is downloaded (e.g. first time on a new machine)
    # data = Dataset.load_builtin('ml-1m')
    # TODO: support FLOAT ratings for ml-20m... only supports int right now!
    algos = {
        'SVD': SVD(),
        # 'KNNBasic_user_msd': KNNBasic(),
        'KNNBasic_item_msd': KNNBasic(sim_options={'user_based': False}),
        # 'KNNBasic_item_cosine': KNNBasic(sim_options={'user_based': False, 'name': 'cosine', }),
        # 'KNNBasic_item_pearson': KNNBasic(sim_options={'user_based': False, 'name': 'pearson', }),
    }


    if args.dataset == 'ml-100k':
        ratings_path = BUILTIN_DATASETS[args.dataset].path
        users_path = ratings_path.replace('.data', '.user')
        movies_path = ratings_path.replace('.data', '.item')
        dfs = movielens_to_df(ratings_path, users_path, movies_path)
    elif args.dataset == 'ml-1m':
        ratings_path = BUILTIN_DATASETS[args.dataset].path
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
        standard_results_filename = '{}_standard_measures_{}.json'.format(args.dataset, algo_name)
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
        print('{} total train/tests will be run because you chose {} sample_sizes and number of samples of {}'.format(
            num_runs, num_configs, args.num_samples
        ))
    else:
        print('{} total train/tests will be run'.format(num_configs))
        num_runs = num_configs
    # in experiments butter can run SVD in 60 seconds for 1M ratings, and KNN in 65 seconds for 1M ratings
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
        elif config['type'] == 'zip':
            # TODO
            pass
        elif config['type'] == 'genre':
            # TODO
            pass

        # do multiple iterations at once - there's no dependence between them
        delayed_iteration_list = []
        for i, experimental_iteration in enumerate(experimental_iterations):
            if config['type'] == 'individual_users':
                if args.num_users_to_stop_at:
                    if i >= args.num_users_to_stop_at - 1:
                        break
                row = experimental_iteration[1]
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
            subset_data = Dataset.load_from_df(
                subset_ratings_df[['user_id', 'movie_id', 'rating']],
                reader=Reader()
            )
            out_ids = list(excluded_ratings_df.user_id)
            # candidate
            # excluded_data = Dataset.load_from_df(
            #     excluded_ratings_df[['user_id', 'movie_id', 'rating']],
            #     reader=Reader()
            # )

            
            num_users = len(subset_ratings_df.user_id.value_counts())
            num_movies = len(subset_ratings_df.movie_id.value_counts())
            delayed_iteration_list += [delayed(task(
                algo_name, algos[algo_name], subset_data, measures, 5, False, identifier,  len(subset_ratings_df.index), num_movies, name, out_ids
            )) for algo_name in algo]
            
            # for algo_name in algos:
            #     subset_results = cross_validate(
            #         algos[algo_name], subset_data, measures=measures,
            #         cv=5, verbose=False)
        out_dicts = Parallel(n_jobs=-1)(delayed_iteration_list)
        for d in out_dicts:
            res = d['subset_results']
            uid_to_error[str(d['identifier']) + '_' + d['algo_name']] = {
                'mae increase': np.mean(res['test_mae']) - standard_results[algo_name]['mae'],
                'rmse increase': np.mean(res['test_rmse']) - standard_results[algo_name]['rmse'],
                'precision10t4 decrease': standard_results[algo_name]['precision10t4'] - np.mean(res['test_precision10t4']),
                'recall10t4 decrease': standard_results[algo_name]['recall10t4'] - np.mean(res['test_recall10t4']),
                'ndcg10 decrease': standard_results[algo_name]['ndcg10'] - np.mean(res['test_ndcg10']),
                'num_ratings':, d['num_ratings']
                'num_tested': np.mean(res['num_tested']),
                'num_users': d['num_users'],
                'num_movies': d['num_movies'],
                'fit_time': np.mean(res['fit_time']),
                'test_time': np.mean(res['test_time']),
                'name': d['name'],
                'algo_name': d['algo_name'],
            }
            
        err_df = pd.DataFrame.from_dict(uid_to_error, orient='index')
        outname = 'results/err_df-dataset_{}_type_{}-size_{}-sample_size_{}.csv'.format(
            args.dataset, config['type'], config['size'],
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
    parser.add_argument('--dataset', default='ml-1m')
    args = parser.parse_args()
    if args.sample_sizes:
        args.sample_sizes = [int(x) for x in args.sample_sizes.split(',')]
        if args.num_samples is None:
            args.num_samples = 1000
    main(args)

if __name__ == '__main__':
    parse()
