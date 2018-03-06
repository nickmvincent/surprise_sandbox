from collections import defaultdict
import csv


from surprise import SVD, KNNBasic
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.builtin_datasets import BUILTIN_DATASETS
from surprise.reader import Reader
import pandas as pd
import numpy as np
from utils import movielens_to_df


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def main():
    # Load the movielens-100k dataset (download it if needed).
    # HEY LISTEN
    # uncomment to make sure the dataset is downloaded (e.g. first time on a new machine)
    # data = Dataset.load_builtin('ml-100k')

    # Use the famous SVD algorithm.
    algo = KNNBasic()

    ratings_path = BUILTIN_DATASETS['ml-100k'].path
    users_path = ratings_path.replace('.data', '.user')
    movies_path = ratings_path.replace('.data', '.item')
    dfs = movielens_to_df(ratings_path, users_path, movies_path)
    ratings_df, users_df, movies_df = dfs['ratings'], dfs['users'], dfs['movies']
    data = Dataset.load_from_df(
        ratings_df[['user_id', 'movie_id', 'rating']],
        reader=Reader())
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    # Run 5-fold cross-validation and print results.
    # cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    dyad = users_df.sample(n=2)
    print(dyad)
    print(ratings_df.info())

    predictions = []

    # this is just two users
    iid_to_ratings = defaultdict(list)
    for i_u, user in dyad.iterrows():
        for i_m, movie in movies_df.iterrows():
            uid = user.user_id
            iid = movie.movie_id
            try:
                rating = ratings_df[(ratings_df.user_id == user.user_id) & (ratings_df.movie_id == movie.movie_id)].iloc[0].rating
            except IndexError:
                rating = None
            pred = algo.predict(uid, iid, r_ui=rating, verbose=False)
            predictions.append(pred)
            iid_to_ratings[iid].append(pred.est)
            
    top_n = get_top_n(predictions, n=5)
    for uid, user_ratings in top_n.items():
        print(uid, [iid for (iid, _) in user_ratings])

    out = []
    weight_sets = (
        (0.25, 0.75),
        (0.75, 0.25),
    )
    print(iid_to_ratings)
    for weight_set in weight_sets:
        out.append('weights: {}'.format(str(weight_set)))
        iid_to_avg = {}
        for iid, ratings in iid_to_ratings.items():
            rating_avg = 0
            for i, rating in enumerate(ratings):
                rating_avg += weight_set[i] * rating
            iid_to_avg[iid] = rating_avg
            print(rating_avg)

        top5 = sorted(iid_to_avg.items(), key=lambda x: x[1], reverse=True)[:20]
        for iid, rating in top5:
            line = '{}, {}'.format(movies_df[movies_df.movie_id == iid].iloc[0].movie_title, rating)
            print(line)
            out.append(line)
        out.append('\n')
    with open("output.txt", "w") as f:
        for line in out:
            f.write(line + '\n')


    

if __name__ == '__main__':
    main()