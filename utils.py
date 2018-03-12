"""
Utility functions that might be useful outside the context of this project.
"""
import pandas as pd

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
              'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
              "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western', ]
    movies_delim = '|'
    encoding = 'latin-1'

    ratings_df = pd.read_csv(ratings_file, sep=ratings_delim, names=ratings_names, encoding=encoding)
    users_df = pd.read_csv(users_file, sep=users_delim, names=users_names, encoding=encoding)
    movies_df = pd.read_csv(movies_file, sep=movies_delim, names=movies_names, encoding=encoding)

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
    movies_names = [ 'movie_id', 'movie_title', 'Action', 'Adventure', 'Animation', 
              "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western', ]
    movies_delim = '::'
    encoding = 'latin-1'

    ratings_df = pd.read_csv(ratings_file, sep=ratings_delim, names=ratings_names, encoding=encoding)
    users_df = pd.read_csv(users_file, sep=users_delim, names=users_names, encoding=encoding)
    movies_df = pd.read_csv(movies_file, sep=movies_delim, names=movies_names, encoding=encoding)

    return {
        'ratings': ratings_df,
        'users': users_df,
        'movies': movies_df,
    }


