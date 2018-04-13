"""
Computes baselines
"""
import os
import argparse
import json
from pprint import pprint

import pandas as pd
from joblib import delayed

from utils import get_dfs, load_head_items
from specs import ALGOS, ALGOS_FOR_STANDARDS, NUM_FOLDS
from constants import MEASURES
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate_many



def main(args):
    """
    driver function
    """
    dfs = get_dfs(args.dataset)
    ratings_df = dfs['ratings']
    data = Dataset.load_from_df(
        ratings_df[['user_id', 'movie_id', 'rating']],
        reader=Reader()
    )

    pathto = "standard_results"
    files = os.listdir(pathto)

    boycott_uid_sets = {}
    like_boycotters_uid_sets = {}

    for file in files:
        if 'uid_sets' not in file or '.csv' not in file:
            continue
        print(file)
        uid_sets_df = pd.read_csv(pathto + '/' + file, dtype=str)
        for i, row in uid_sets_df.iterrows():
            identifier_num = row[0]
            try:
                boycott_uid_set = set(row['boycott_uid_set'].split(';'))
            except AttributeError:
                boycott_uid_set = set([])
            try:
                like_boycotters_uid_set = set(row['like_boycotters_uid_set'].split(';'))
            except AttributeError:
                like_boycotters_uid_set = set([])

            full_identifier = file.replace('uid_sets_', '') + '__' + identifier_num
            boycott_uid_sets[full_identifier] = boycott_uid_set
            like_boycotters_uid_sets[full_identifier] = like_boycotters_uid_set

    # now boycott_uid_sets and co. are filled up!
    if args.algo_name:
        algo_names = [args.algo_name]
    else:
        algo_names = list(ALGOS.keys())
    for algo_name in algo_names:
        res = cross_validate_many(
            ALGOS[algo_name], data,
            Dataset.load_from_df(pd.DataFrame(), reader=Reader()),
            boycott_uid_sets, like_boycotters_uid_sets, 
            MEASURES, NUM_FOLDS, verbose=False
        )
        with open(
            'standard_results/{}_{}.json'.format(
                args.dataset, algo_name
            ), 'w') as f:
            json.dump(res, f)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument('--algo_name')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    parse()