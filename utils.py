"""
Utility functions that might be useful outside the context of this project.
"""
import os.path
import pandas as pd

from surprise.builtin_datasets import BUILTIN_DATASETS

from surprise.builtin_datasets import download_builtin_dataset

GENRES = ['Action', 'Adventure', 'Animation', 
              "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western', ]


def concat_output_filename(dataset, type_, userfrac, ratingfrac, size=None, num_samples=None):
    ret = 'results/dataset-{}_type-{}_userfrac-{}_ratingfrac-{}'.format(
        dataset, type_,
        userfrac, ratingfrac
    )
    if size and num_samples:
        ret += '_sample_size-{}_num_samples-{}'.format(
            size, num_samples
        )
    ret += '.csv'
    return ret

def get_dfs(dataset):
    """i
    Takes a dataset string and return that data in a dataframe!
    """
    ratings_path = BUILTIN_DATASETS[dataset].path
    print('Path to ratings file is: {}'.format(ratings_path))
    if not os.path.isfile(ratings_path):
        download_builtin_dataset(dataset)
    if dataset == 'ml-100k':
        users_path = ratings_path.replace('.data', '.user')
        movies_path = ratings_path.replace('.data', '.item')
        dfs = movielens_to_df(ratings_path, users_path, movies_path)
    elif dataset == 'ml-1m':
        users_path = ratings_path.replace('ratings.', 'users.')
        movies_path = ratings_path.replace('ratings.', 'movies.')
        dfs = movielens_1m_to_df(ratings_path, users_path, movies_path)
    else:
        raise Exception("Unknown dataset: " + dataset)
    return dfs

def movielens_to_df(ratings_file, users_file, movies_file):
    """
    This function takes a movielens dataset and returns three dataframes
    Specifically it was written with the 100k dataset.
    """
    ratings_names = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_delim = '\t'
    users_names = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    users_delim = '|'
    movies_names = [ 'movie_id', 'movie_title', 'release_date', 'video_release_date',
              'IMDb_URL', 'unknown',] + GENRES
    movies_delim = '|'
    encoding = 'latin-1'
    engine = 'python'

    ratings_df = pd.read_csv(ratings_file, sep=ratings_delim, names=ratings_names, encoding=encoding, engine=engine)
    users_df = pd.read_csv(users_file, sep=users_delim, names=users_names, encoding=encoding, engine=engine)
    movies_df = pd.read_csv(movies_file, sep=movies_delim, names=movies_names, encoding=encoding, engine=engine)

    return {
        'ratings': ratings_df,
        'users': users_df,
        'movies': movies_df,
    }


def movielens_1m_to_df(ratings_file, users_file, movies_file):
    """
    This function takes a movielens dataset and returns three dataframes
    Specifically it was written with the 100k dataset.
    """
    ratings_names = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_delim = '::'
    users_names = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    users_delim = '::'
    movies_names = [ 'movie_id', 'movie_title', 'genres']
    movies_delim = '::'
    encoding = 'latin-1'
    engine = 'python'

    ratings_df = pd.read_csv(ratings_file, sep=ratings_delim, names=ratings_names, encoding=encoding, engine=engine)
    users_df = pd.read_csv(users_file, sep=users_delim, names=users_names, encoding=encoding, engine=engine)
    movies_df = pd.read_csv(movies_file, sep=movies_delim, names=movies_names, encoding=encoding, engine=engine)

    return {
        'ratings': ratings_df,
        'users': users_df,
        'movies': movies_df,
    }


