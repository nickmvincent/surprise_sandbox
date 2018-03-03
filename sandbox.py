from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.builtin_datasets import BUILTIN_DATASETS
import pandas as pd

from utils import movielens_to_df

def main():
    # Load the movielens-100k dataset (download it if needed).
    data = Dataset.load_builtin('ml-100k')

    # Use the famous SVD algorithm.
    algo = SVD()

    ratings_path = BUILTIN_DATASETS['ml-100k'].path
    users_path = ratings_path.replace('.data', '.user')
    movies_path = ratings_path.replace('.data', '.item')
    dfs = movielens_to_df(ratings_path, users_path, movies_path)
    ratings_df, users_df, movies_df = dfs['ratings'], dfs['users'], dfs['movies']
    print(ratings_df.columns.values)
    return

    # Run 5-fold cross-validation and print results.
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

if __name__ == '__main__':
    main()