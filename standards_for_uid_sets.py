"""
Computes baselines
"""
import os
import argparse
import json
from pprint import pprint
import time

import pandas as pd
from joblib import delayed

from utils import get_dfs, load_head_items
from specs import ALGOS, ALGOS_FOR_STANDARDS, NUM_FOLDS
from constants import MEASURES
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate_many


def batch(iterable, batch_size=1):
    """make batches for an iterable"""
    num_items = len(iterable)
    for ndx in range(0, num_items, batch_size):
        yield iterable[ndx:min(ndx + batch_size, num_items)]


def main(args):
    """
    driver function
    """
    starttime = time.time()
    dfs = get_dfs(args.dataset)
    ratings_df = dfs['ratings']
    data = Dataset.load_from_df(
        ratings_df[['user_id', 'movie_id', 'rating']],
        reader=Reader()
    )

    files = os.listdir(args.pathto)

    boycott_uid_sets = {}
    like_boycotters_uid_sets = {}

    for file in files:
        if 'uid_sets' not in file or '.csv' not in file:
            continue
        if args.dataset not in file:
            print('skip {} b/c dataset'.format(file))
            continue
        print(file)
        uid_sets_df = pd.read_csv(args.pathto + '/' + file, dtype=str)
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
    out = {}
    for algo_name in algo_names:
        for batch_num, key_batch in enumerate(batch(list(boycott_uid_sets.keys()), 50)):
            print('On key batch {} of {} keys'.format(batch_num, len(boycott_uid_sets)))
            batch_b = {}
            batch_l = {}
            for key in key_batch:
                batch_b[key] = boycott_uid_sets[key]
                batch_l[key] = boycott_uid_sets[key]

            res = cross_validate_many(
                ALGOS[algo_name], data,
                Dataset.load_from_df(pd.DataFrame(), reader=Reader()),
                batch_b, batch_l, 
                MEASURES, NUM_FOLDS, verbose=False
            )
            out.update(res)
            with open(
                'standards_results/{}_{}.json'.format(
                    args.dataset, algo_name
                ), 'w'
            ) as f:
                json.dump(out, f)
    print(time.time() - starttime)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument('--algo_name')
    parser.add_argument('--pathto', default='standard_results')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    parse()