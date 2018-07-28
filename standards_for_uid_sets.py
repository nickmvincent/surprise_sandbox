"""
Computes standards for comparison (i.e. what are the results without any boycott going on)

Also has functionality to merge many standards into one json file for convenience (see the `merge` argument)
"""
import os
import argparse
import json
from pprint import pprint
import datetime as datetime
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
    Calculate standards

    Configuration requqired for this function:
      you must have uid_sets files in the directory specified by the pathto argument
      uid_sets files are CSV files with:
        an iteration number (index) in one column, a list of boycott uids in 2nd column, and a list of like-boycott uids in a 3rd column
        uid lists are stored as strings delimited by semi-colon (;)
    """
    starttime = time.time()
    dfs = get_dfs(args.dataset)
    head_items = load_head_items(args.dataset)
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
            #print('skip {} b/c dataset'.format(file))
            continue
        if args.name_match and args.name_match not in file:
            continue
        print(file)
        uid_sets_df = pd.read_csv(args.pathto + '/' + file, dtype=str)
        for i, row in uid_sets_df.iterrows():
            identifier_num = row[0]
            try:
                boycott_uid_set = set([
                    int(x) for x in row['boycott_uid_set'].split(';')
                ])
            except AttributeError:
                boycott_uid_set = set([])
            try:
                like_boycotters_uid_set = set([
                    int(x) for x in row['like_boycotters_uid_set'].split(';')
                ])
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
        # why do we batch this - otherwise we could run out of memory if doing many experiment with one script run
        for batch_num, key_batch in enumerate(batch(list(boycott_uid_sets.keys()), 100)):
            print('On key batch {} of {} keys'.format(batch_num, len(boycott_uid_sets)))
            batch_b = {}
            batch_l = {}
            for key in key_batch:
                print(key)
                batch_b[key] = boycott_uid_sets[key]
                batch_l[key] = like_boycotters_uid_sets[key]

            # ideally we don't need to re-train the algorithm... we have the actual predictions saved for each rating within each crossfold!
            # if for some reason this was lost (or wasn't saved, e.g. using the pre-July 2018 version of this code) we can re-train
            # will take much longer
            if args.load_path == 'False':
                load_path = None
            elif args.load_path is None:
                load_path = os.getcwd() + '/predictions/standards/{}_{}_'.format(args.dataset, algo_name)
            else:
                load_path = args.load_path + '/standards/{}_{}_'.format(args.dataset, algo_name)
            res = cross_validate_many(
                ALGOS[algo_name], data,
                Dataset.load_from_df(pd.DataFrame(), reader=Reader()),
                batch_b, batch_l, 
                MEASURES, NUM_FOLDS, verbose=False, head_items=head_items,
                load_path=load_path
            )
            out.update(res)
            dtstr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            with open(
                'standard_results/{}_{}_{}.json'.format(
                    args.dataset, algo_name, dtstr
                ), 'w'
            ) as f:
                json.dump(out, f)
    print(time.time() - starttime)


def join(args):
    """
    Join together a bunch of standards results that were calculated separately!

    They will go into a json file prepended with MERGE
    Don't commit this file to git, it will be very large...
    """
    successes = 0
    sources = []
    if args.algo_name:
        algo_names = [args.algo_name]
    else:
        algo_names = list(ALGOS.keys())
    for algo_name in algo_names:
        merged = {}
        for root, dirs, _ in os.walk('misc_standards/'):
            for d1 in dirs:
                # this is a bit hacky
                # we're using the log.txt file (generated when we run on AWS spot instances)
                # to figure out which algo name was run
                # if some experiments were not run on AWS, you will need to make a log.txt file manually!
                try:
                    logf = '{}/{}/log.txt'.format(root, d1)
                    with open(logf, 'r') as f:
                        log = f.read()
                        if algo_name not in log:
                            #print('Skipping this dir b/c wrong log')
                            continue
                except FileNotFoundError:
                    continue
                for root2, _, files in os.walk(root + '/' + d1):
                    for file in files:
                        if file.endswith('.json'):
                            if algo_name not in file:
                                # print('Skip {}'.format(file))
                                continue
                            with open(root2 + '/' + file, 'r') as f:
                                data = json.load(f)
                                merged.update(data)
                            print('Success for {}'.format(file))  
                            successes += 1                        
                            sources.append(file)  
                        
        with open('MERGED_{}_{}.json'.format(args.dataset, algo_name), 'w') as f:
            json.dump(merged, f)
        print(successes)


def parse():
    """
    Examples

    python standards_for_uid_sets.py --dataset test_ml-1m --algo_name SVD --name_match sample --pathto standard_results
    python standards_for_uid_sets.py --dataset ml-20m --algo_name SVD --name_match sample_size-14_num_samples-20
    python standards_for_uid_sets.py --dataset ml-100k --algo_name SVD

    python standards_for_uid_sets.py --name_match sample_size-1_num_samples-1 --dataset ml-100k --algo_name SVD

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m', help="which dataset to use")
    parser.add_argument('--algo_name', help="which rec algo to use")
    parser.add_argument('--name_match', help="provide a string here and the script will only look at filenames that match the given string.")
    parser.add_argument('--pathto', default='uid_sets', help="Where are the uid_sets files?")
    parser.add_argument('--join', action='store_true', help="If true, just merge together standard results that have already been computed into one convenient json file. If false actually compute standards.")
    parser.add_argument(
        '--load_path', help='where to load predictions from.'
    )

    args = parser.parse_args()

    if args.join:
        join(args)
    else:
        main(args)


if __name__ == '__main__':
    parse()