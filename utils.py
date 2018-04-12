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


def concat_output_filename(dataset, type_, userfrac, ratingfrac, size=None, num_samples=None, indices='all'):
    ret = 'results/dataset-{}_type-{}_userfrac-{}_ratingfrac-{}'.format(
        dataset, type_,
        userfrac, ratingfrac
    )
    if size:
        ret += '_sample_size-{}'.format(size)
    if num_samples:
        ret += '_num_samples-{}'.format(num_samples)
    if indices != 'all':
        ret += '_indices-{}-to-{}'.format(indices[0], indices[1])
    ret += '.csv'
    return ret

def load_head_items(dataset):
    with open('boycott_files/{}_top_five_percent_movies.csv'.format(dataset), 'r') as f:
        head_items = [int(x) for x in f.read().split(',')]
    return head_items


def get_dfs(dataset):
    """
    Takes a dataset string and return that data in a dataframe!
    """
    print(dataset)
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


def extract_from_filename(text, comes_after, length=None, ends_at=None):
    """
    dataset-ml-1m_type-gender_userfrac-1.0_ratingfrac-1.0
    """
    i = text.find(comes_after)
    if i == -1:
        raise ValueError('could not find')
    start = i + len(comes_after)
    if length:
        end = start + length
    elif ends_at:
        end = text.find(ends_at)
    return text[start:end]


if __name__ == '__main__':
    print('Testing...')
    assert extract_from_filename("dataset-ml-1m_type-gender_userfrac-1.0_ratingfrac-1.0", "userfrac-", 3) == '1.0'
    assert extract_from_filename("dataset-ml-1m_type-gender_userfrac-1.0_ratingfrac-1.0", "ratingfrac-", 3) == '1.0'
    assert extract_from_filename(
        "dataset-ml-1m_type-sample_users_userfrac-1.0_ratingfrac-1.0_sample_size-1_num_samples-250_indices-251-to-500.csv",
        "indices-", 3) == '251'
    assert extract_from_filename(
        "dataset-ml-1m_type-sample_users_userfrac-1.0_ratingfrac-1.0_sample_size-1_num_samples-250_indices-251-to-500.csv",
        "indices-", None, '.csv') == '251-to-500'


    assert load_head_items('ml-100k') == [
        '50', '258', '100', '181', '294', '286', '288', '1', '300', '121', '174', '127', '56',
        '7', '98', '237', '117', '172', '222', '204', '313', '405', '79', '210', '151', '173', '69', '748', '168', '269', '257', '195', '423', '9', '318', '276', '302', '22', '328', '96', '118', '15', '25', '183', '216', '176', '64', '234', '202', '191', '28', '89', '111', '275', '12', '742', '357', '82', '135', '289', '97', '238', '268', '546', '196', '333', '186', '70', '475', '153', '132', '228', '125', '144', '483', '194', '245', '323', '197', '185', '11', '282',
'496', '568']
    print('Success')