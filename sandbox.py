"""
sandbox.py includes a simple experimentation with Surprise and the ML-100k dataset
"""
import argparse

import pandas as pd
import numpy as np
from utils import movielens_to_df


# # platform specific globals
# if platform.system() == 'Windows':
#     sys.path.append("C:/Users/Nick/Documents/GitHub/recsys_experiments/Surprise")
# else:
#     raise ValueError('todo')

# import pyximport
# pyximport.install()

from surprise.model_selection import cross_validate
from surprise import SVD, KNNBasic, Dataset
from surprise.builtin_datasets import BUILTIN_DATASETS
from surprise.reader import Reader


def main(args):
    """
    Run the sandbox experiments
    """
    # Load the movielens-100k dataset (download it if needed).
    # HEY LISTEN
    # uncomment to make sure the dataset is downloaded (e.g. first time on a new machine)
    # data = Dataset.load_builtin('ml-100k')
    algo = SVD()

    ratings_path = BUILTIN_DATASETS['ml-100k'].path
    users_path = ratings_path.replace('.data', '.user')
    movies_path = ratings_path.replace('.data', '.item')
    dfs = movielens_to_df(ratings_path, users_path, movies_path)
    ratings_df, users_df, movies_df = dfs['ratings'], dfs['users'], dfs['movies']
    data = Dataset.load_from_df(
        ratings_df[['user_id', 'movie_id', 'rating']],
        reader=Reader()
    )
    # Run 5-fold cross-validation and print results.
    results = cross_validate(
        algo, data, measures=['RMSE', 'MAE', 'precision10t4_recall10t4_ndcg10'],
        cv=5, verbose=True)
    # TODO: cache this result. no reason to be recomputing this every time.
    best_err = {
        'mae': np.mean(results['test_mae']),
        'rmse': np.mean(results['test_rmse']),
        'precision10t4': np.mean(results['test_precision10t4']),
        'recall10t4': np.mean(results['test_recall10t4']),
        'ndcg10': np.mean(results['test_ndcg10']),
    }
    experiment_configs = []
    if args.sample_sizes:
        experiment_configs += [
            {
                'type': 'sample_users', 'size': sample_size
            } for sample_size in args.sample_sizes]
    else:
        experiment_configs += [{'type': 'individual_users', 'size': 1}]

    uid_to_error = {}
    for config in experiment_configs:
        if config['type'] == 'individual_users':
            iterable = users_df.iterrows()
        elif config['type'] == 'sample_users':
            iterable = [users_df.sample(config['size']) for _ in range(args.num_samples)]
        for i, experimental_iteration in enumerate(iterable):
            if config['type'] == 'individual_users':
                if args.num_users_to_stop:
                    if i >= args.num_users_to_stop - 1:
                        break
                subset_ratings_df = ratings_df[ratings_df.user_id != experimental_iteration.user_id]
                identifier = experimental_iteration.user_id
            elif config['type'] == 'sample_users':
                ids = list(experimental_iteration.user_id)
                subset_ratings_df = ratings_df[~ratings_df.user_id.isin(ids)]
                identifier = i
            subset_data = Dataset.load_from_df(
                subset_ratings_df[['user_id', 'movie_id', 'rating']],
                reader=Reader()
            )
            subset_results = cross_validate(
                algo, subset_data, measures=['RMSE', 'MAE', 'precision10t4_recall10t4_ndcg10'],
                cv=5, verbose=False)
            num_users = len(subset_ratings_df.user_id.value_counts())
            num_movies = len(subset_ratings_df.movie_id.value_counts())
            uid_to_error[identifier] = {
                'mae increase': np.mean(subset_results['test_mae']) - best_err['mae'],
                'rmse increase': np.mean(subset_results['test_rmse']) - best_err['rmse'],
                'precision10t4 decrease': best_err['precision10t4'] - np.mean(subset_results['test_precision10t4']),
                'recall10t4 decrease': best_err['recall10t4'] - np.mean(subset_results['test_recall10t4']),
                'ndcg10 decrease': best_err['ndcg10'] - np.mean(subset_results['test_ndcg10']),
                'num_ratings': len(subset_ratings_df.index),
                'num_tested': np.mean(subset_results['num_tested']),
                'num_users': num_users,
                'num_movies': num_movies,
                'fit_time': np.mean(subset_results['fit_time']),
                'test_time': np.mean(subset_results['test_time']),

            }
            
        err_df = pd.DataFrame.from_dict(uid_to_error, orient='index')
        outname = 'err_df-{}-{}.csv'.format(config['type'], config['size'])
        err_df.to_csv(outname)
        means = err_df.mean()
        means.to_csv(outname.replace('err_df', 'err_df_means'))
        print(means)


def parse():
    """
    Parse args and handles list splitting
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users_to_stop', type=int)
    parser.add_argument('--sample_sizes')
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()
    args.sample_sizes = [int(x) for x in args.sample_sizes.split(',')]
    main(args)

if __name__ == '__main__':
    parse()
