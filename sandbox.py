"""
sandbox.py includes a simple experimentation with Surprise and the ML-100k dataset
"""
from collections import defaultdict
import platform
import sys
import argparse

import pandas as pd
import numpy as np
from utils import movielens_to_df
import matplotlib.pyplot as plt


# # platform specific globals
# if platform.system() == 'Windows':
#     sys.path.append("C:/Users/Nick/Documents/GitHub/recsys_experiments/Surprise")
# else:
#     raise ValueError('todo')

# import pyximport
# pyximport.install()

from surprise.model_selection import KFold, cross_validate
from surprise import SVD, KNNBasic, Dataset
from surprise.builtin_datasets import BUILTIN_DATASETS
from surprise.reader import Reader


def main(args):
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
    print('Results for full dataset')
    print('len(ratings_df.index)', len(ratings_df.index))
    results = cross_validate(algo, data, measures=['RMSE', 'MAE', 'precision10t4_recall10t4_ndcg10'], cv=5, verbose=True)
    best_err = {
        'mae': np.mean(results['test_mae']),
        'rmse': np.mean(results['test_rmse']),
        'precision10t4': np.mean(results['test_precision10t4']),
        'recall10t4': np.mean(results['test_recall10t4']),
        'ndcg10': np.mean(results['test_ndcg10']),
    }

    uid_to_error = {}
    count = 0

    if args.group_size:
        group = None
    for i_u, user in users_df.iterrows():
        if args.num_users:
            if i_u >= args.num_users - 1:
                break
        subset_ratings_df = ratings_df[ratings_df.user_id != user.user_id]
        subset_data = Dataset.load_from_df(
            subset_ratings_df[['user_id', 'movie_id', 'rating']],
            reader=Reader()
        )

        # TODO: make sure you understand exactly how the cross-folds work when manually deleting rows from ratings_df
        subset_results = cross_validate(algo, subset_data, measures=['RMSE', 'MAE', 'precision10t4_recall10t4_ndcg10'], cv=5, verbose=False)
        uid_to_error[user.user_id] = {
            'mae increase': np.mean(subset_results['test_mae']) - best_err['mae'],
            'rmse increase': np.mean(subset_results['test_rmse']) - best_err['rmse'],
            'precision10t4 decrease': best_err['precision10t4'] - np.mean(subset_results['test_precision10t4']),
            'recall10t4 decrease': best_err['recall10t4'] - np.mean(subset_results['test_recall10t4']),
            'ndcg10 decrease': best_err['ndcg10'] - np.mean(subset_results['test_ndcg10']),
            'num_ratings': len(subset_ratings_df.index),
            'num_users': len(users_df.index) - 1,
            'fit_time': np.mean(subset_results['fit_time']),
            'test_time': np.mean(subset_results['test_time']),
        }
        
    err_df = pd.DataFrame.from_dict(uid_to_error, orient='index')
    print(err_df)
    err_df.to_csv('err_df.csv')
    print(err_df.mean())
    
        
    


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users', type=int, default=None)
    parser.add_argument('--group_size', default=None)
    args = parser.parse_args()
    main(args)